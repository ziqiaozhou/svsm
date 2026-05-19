// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2022-2023 SUSE LLC
//
// Author: Joerg Roedel <jroedel@suse.de>

//! Generic page table types and structures for 4 KiB granule paging.
//!
//! They are designed for different OS or architectures.
//!
//! The implementation assumes a 4 KiB base page granule with a 4-level
//! page table hierarchy (8-byte PTEntry and 512 entries per page), supporting three page
//! sizes: 4 KiB, 2 MiB, and 1 GiB. This covers x86_64 (PML4) and
//! ARM64 with 4 KiB granule. Other granule sizes (16 KiB, 64 KiB on
//! ARM64) are not supported.

use crate::address::{Address, PhysAddr, VirtAddr};
use crate::sizes::{PAGE_SHIFT, PAGE_SIZE, PAGE_SIZE_1G, PAGE_SIZE_2M, PageSize};
use bitflags::Flags;
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use zerocopy::{FromBytes, FromZeros};

/// Number of virtual-address bits indexed by a single page-table level.
pub const PTE_SHIFT: usize = 9;

/// Number of entries in a page table (4KB/8B).
pub const ENTRY_COUNT: usize = 1 << PTE_SHIFT;

/// Page table levels (0-based).
///
/// Level0 is the leaf (PTE), Level3 is the root of 4-level paging (PML4E).
/// At most 4 levels are supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum PageLevel {
    Level0 = 0,
    Level1 = 1,
    Level2 = 2,
    Level3 = 3,
}

impl PageLevel {
    /// Returns the next level down, or `None` at Level0.
    pub fn next_down(self) -> Option<Self> {
        match self {
            Self::Level3 => Some(Self::Level2),
            Self::Level2 => Some(Self::Level1),
            Self::Level1 => Some(Self::Level0),
            Self::Level0 => None,
        }
    }
}

/// Describes how many levels a page table hierarchy has.
///
/// `TOP_LEVEL` is the highest (root) level, up to [`PageLevel::Level3`]
/// (i.e. at most 4 levels: L0–L3).
///
/// Architecture-specific ZSTs that implement this trait live in their
/// respective modules (e.g. `x86_64::Pml4Level`).
pub trait PagingLevel {
    /// Highest page table level.
    const TOP_LEVEL: PageLevel;
}

#[derive(Debug)]
pub struct PagingLevel3;

impl PagingLevel for PagingLevel3 {
    const TOP_LEVEL: PageLevel = PageLevel::Level3;
}

#[derive(Debug)]
pub struct PagingLevel2;

impl PagingLevel for PagingLevel2 {
    const TOP_LEVEL: PageLevel = PageLevel::Level2;
}

/// Errors that can occur during page table operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PagingError {
    /// Frame allocation failed.
    AllocFrame,
    /// The requested virtual address is not mapped.
    NotMapped,
}

/// Architecture-specific page table metadata for confidential computing.
///
/// Defines how encryption/confidentiality bits are managed in page table
/// entries. On AMD SEV-SNP the private mask is the C-bit and the shared mask
/// is zero (or vice-versa depending on convention). On platforms without
/// memory encryption both masks are zero.
///
/// # Required methods
///
/// * [`private_pte_mask`](Self::private_pte_mask) — bitmask ORed into PTEs
///   for private (encrypted) mappings.
/// * [`shared_pte_mask`](Self::shared_pte_mask) — bitmask ORed into PTEs
///   for shared (plaintext) mappings.
/// * [`flush_tlb_global`](Self::flush_tlb_global) — flush the TLB on all
///   CPUs; called after splitting or changing the encryption state of a page.
///
/// # Default methods
///
/// The remaining methods (`strip_*`, `make_*`, `is_shared_address`,
/// `supported_flags`) have default implementations derived from the masks.
/// Override only if the platform requires non-standard behaviour.
///
/// # Implementer requirements
///
/// * All methods must be stateless (no `&self`) — the type is used as a
///   zero-sized marker and never instantiated.
/// * `private_pte_mask` and `shared_pte_mask` must not share any set bits —
///   a page is either private *or* shared.
/// * `make_private_address` and `make_shared_address` must be idempotent.
pub trait ArchPagingMeta: 'static + Copy {
    type PTFlags: GenericPageTableFlags;
    // --- Required methods ---

    /// Returns the bitmask ORed into physical addresses for private
    /// (encrypted) page table entries.
    fn private_pte_mask() -> usize;

    /// Returns the bitmask ORed into physical addresses for shared
    /// (plaintext) page table entries.
    fn shared_pte_mask() -> usize;

    /// Physical address mask.
    /// x64 supports 52-bit physical addresses, so the mask is usually 0x000f_ffff_ffff_f000.
    fn address_mask() -> usize;

    /// Flush the TLB globally and synchronize across all CPUs.
    ///
    /// Called after modifying live PTEs (e.g., splitting a 2M page or
    /// toggling the encryption state of a mapping).
    fn flush_tlb_global();

    /// Returns a bitmask of PTEntryFlags that the hardware supports.
    ///
    /// Override this method to filter unsupported bits (e.g., `GLOBAL` before CR4.PGE is enabled)
    /// so that they are silently cleared. The default allows all flags.
    fn supported_flags() -> Self::PTFlags {
        Self::PTFlags::all()
    }

    // --- Default methods (derived from the masks above) ---

    /// Clears the private encryption bit(s) from `paddr`.
    fn strip_confidentiality_bits(paddr: PhysAddr) -> PhysAddr {
        (paddr.bits() & !Self::private_pte_mask()).into()
    }

    /// Clears the shared bit(s) from `paddr`.
    fn strip_shared_address_bits(paddr: PhysAddr) -> PhysAddr {
        (paddr.bits() & !Self::shared_pte_mask()).into()
    }

    /// Returns `paddr` with the private encryption mask applied.
    ///
    /// Any shared bits are stripped first so the result is exclusively
    /// private.
    fn make_private_address(paddr: PhysAddr) -> PhysAddr {
        (Self::strip_shared_address_bits(paddr).bits() | Self::private_pte_mask()).into()
    }

    /// Returns `paddr` with the shared mask applied.
    ///
    /// Any confidentiality (private) bits are stripped first so the result
    /// is exclusively shared.
    fn make_shared_address(paddr: PhysAddr) -> PhysAddr {
        (Self::strip_confidentiality_bits(paddr).bits() | Self::shared_pte_mask()).into()
    }

    /// Returns `true` if `paddr` already has the shared mask applied.
    fn is_shared_address(paddr: PhysAddr) -> bool {
        paddr == Self::make_shared_address(paddr)
    }
}

/// OS-level page table services: address translation and frame management.
///
/// Bridges the generic page table code to the OS memory allocator and
/// virtual-address layout. Every method is an associated function (no
/// `&self`) — the implementing type is used as a zero-sized marker and
/// never instantiated.
///
/// # Safety
///
/// Implementers must guarantee:
///
/// * **`paddr_to_vaddr`** — the returned virtual address is a valid,
///   dereferenceable mapping of `paddr` for the lifetime of the page table.
///   `paddr` is always a *clean* physical address (no encryption bits).
///
/// * **`allocate_physical_page`** — every successful call returns a *unique*,
///   page-aligned, *zeroed* physical frame whose address is *clean* (no
///   encryption/confidentiality bits). The frame remains valid until a
///   matching `deallocate_physical_page` call.
///
/// * **`deallocate_physical_page`** — `paddr` is a value previously returned by
///   `allocate_physical_page` that has not yet been freed.
///
/// # Cross-method invariant
///
/// `paddr_to_vaddr` must return a valid, writable mapping for every
/// address returned by `allocate_physical_page`. The generic page table code
/// calls `allocate_physical_page` and immediately passes the result to
/// `paddr_to_vaddr` in order to zero-initialise and populate newly
/// allocated page table pages. This invariant can be satisfied either
/// by a linear map of all physical memory (so `paddr_to_vaddr` works
/// for any physical address) or by having `allocate_physical_page` return only
/// frames that are already mapped.
pub unsafe trait PagingHandler: 'static + FromBytes {
    /// Translate a clean physical address to a virtual address suitable for
    /// accessing page table pages.
    fn paddr_to_vaddr(paddr: PhysAddr) -> VirtAddr;

    /// Allocate a zeroed page-table frame.
    ///
    /// Returns the *clean* physical address of the frame — no encryption
    /// or confidentiality bits are set. Callers apply
    /// [`ArchPagingMeta::make_private_address`] when storing the address
    /// in a PTE.
    fn allocate_physical_page() -> Result<PhysAddr, PagingError>;

    /// Deallocate a page-table frame previously returned by
    /// [`allocate_physical_page`](Self::allocate_physical_page).
    ///
    /// # Safety
    ///
    /// `paddr` must be a clean physical address previously returned by
    /// `allocate_physical_page` and not yet freed.
    unsafe fn deallocate_physical_page(paddr: PhysAddr);
}

pub const fn virt_from_lvl_idx(idx: usize, level: PageLevel) -> VirtAddr {
    VirtAddr::new(idx << ((level as usize * PTE_SHIFT) + PAGE_SHIFT))
}

/// Self-map support for page tables.
///
/// When a page table contains a self-map entry (a PML4 entry that points
/// back to the page table root), the CPU's address translation creates a
/// virtual window through which every PTE of the *active* page table can
/// be read and written at a known virtual address.
///
/// # Implementer requirements
///
/// * `PTE_BASE_VADDR` must return the virtual base address of that self-map
///   window. For a self-map installed at PML4 index *N*, this is
///   `sign_extend(N << 39 | N << 30 | N << 21 | N << 12)`.
/// * The self-map entry must be installed before any method in the
///   `impl<..., P: PagingHandler + SelfMap>` block is called.
pub trait SelfMap {
    /// The PML4 index of the self-map entry.
    const SELFMAP_IDX: usize;
}

pub trait GenericPageTableFlags:
    bitflags::Flags<Bits = usize>
    + core::ops::BitAnd<Output = Self>
    + core::ops::BitOr<Output = Self>
    + Copy
    + Clone
{

    /// Default flags of new parent page table entries for kernel mappings.
    /// Includes `PRESENT` but not `USER`, so the mapping is supervisor-only.
    fn kern_parent_flags() -> Self;

    /// Default flags for new parent page table entries.
    /// Use this if the mapping could be user-accessible;
    fn parent_flags() -> Self;

    const PRESENT: Self;
    const WRITABLE: Self;
    const USER: Self;
    const ACCESSED: Self;
    const DIRTY: Self;
    const HUGE: Self;
    const GLOBAL: Self;
    const NX: Self;
}

const _: () = assert!(
    core::mem::size_of::<PhysAddr>() == 8,
    "Only supports 8 bytes PTE entry",
);

/// Represents a page table entry.
#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes)]
pub struct PTEntry<P: ArchPagingMeta> {
    entry: PhysAddr,
    _phantom: PhantomData<fn() -> P>,
}

impl<A: ArchPagingMeta> PTEntry<A> {
    /// Check if the page table entry is clear (null).
    pub fn is_clear(&self) -> bool {
        self.entry.is_null()
    }

    /// Clear the page table entry.
    pub fn clear(&mut self) {
        self.entry = PhysAddr::null();
    }

    /// Check if the page table entry is present.
    pub fn present(&self) -> bool {
        self.flags().contains(A::PTFlags::PRESENT)
    }

    /// Check if the page table entry is huge.
    pub fn huge(&self) -> bool {
        self.flags().contains(A::PTFlags::HUGE)
    }

    /// Check if the page table entry is writable.
    pub fn writable(&self) -> bool {
        self.flags().contains(A::PTFlags::WRITABLE)
    }

    /// Check if the page table entry is NX (no-execute).
    pub fn nx(&self) -> bool {
        self.flags().contains(A::PTFlags::NX)
    }

    /// Check if the page table entry is user-accessible.
    pub fn user(&self) -> bool {
        self.flags().contains(A::PTFlags::USER)
    }

    /// Check if the page table entry is global.
    pub fn global(&self) -> bool {
        self.flags().contains(A::PTFlags::GLOBAL)
    }

    /// Get the raw bits (`usize`) of the page table entry.
    pub fn raw(&self) -> usize {
        self.entry.bits()
    }

    /// Get the flags of the page table entry.
    pub fn flags(&self) -> A::PTFlags {
        A::PTFlags::from_bits_truncate(self.entry.bits())
    }

    /// Set the page table entry with the specified address and flags.
    pub fn set_unrestricted(&mut self, addr: PhysAddr, flags: A::PTFlags) {
        let addr = addr.bits();
        assert_eq!(addr & !A::address_mask(), 0);
        self.entry = PhysAddr::from(addr | flags.bits());
    }

    /// Set the page table entry with the specified address, with flags
    /// constrained to the supported feature flags.
    pub fn set(&mut self, addr: PhysAddr, flags: A::PTFlags) {
        self.set_unrestricted(addr, flags & A::supported_flags());
    }

    /// Get the paddr field from the entry.
    ///
    /// Returns bits `[51:12]` of the entry — the address *including* any
    /// encryption/confidentiality bits the hardware stores in the upper
    /// physical address bits.
    pub fn paddr_field(&self) -> PhysAddr {
        PhysAddr::from(self.raw() & A::address_mask())
    }

    /// Get the address from the page table entry, including the shared bit.
    pub fn page_frame(&self) -> PhysAddr {
        A::strip_confidentiality_bits(self.paddr_field())
    }

    /// Get the address from the page table entry, excluding the C/shared bit.
    pub fn address(&self) -> PhysAddr {
        A::strip_shared_address_bits(self.page_frame())
    }

    /// Inserts the private address mask if the page is present.
    pub fn make_private_if_present(&mut self) {
        if self.flags().contains(A::PTFlags::PRESENT) {
            self.entry = A::make_private_address(self.entry);
        }
    }

    // Returns true if the address is shared.
    pub fn is_shared(&self) -> bool {
        A::is_shared_address(self.paddr_field())
    }

    /// Read a page table entry from the specified virtual address.
    ///
    /// # Safety
    ///
    /// Reads from an arbitrary virtual address, making this essentially a
    /// raw pointer read.  The caller must be certain to calculate the correct
    /// address.
    pub unsafe fn read_pte(vaddr: VirtAddr) -> Self {
        // SAFETY: When the methods safety requirements are met, the raw
        // pointer read is safe.
        unsafe { *vaddr.as_ptr::<Self>() }
    }
}

/// A pagetable page with multiple entries.
#[repr(C)]
#[derive(Debug, FromBytes)]
pub struct PTPage<A: ArchPagingMeta, P: PagingHandler> {
    pub entries: [PTEntry<A>; ENTRY_COUNT],
    _phantom: PhantomData<fn() -> P>,
}

impl<A: ArchPagingMeta, P: PagingHandler> PTPage<A, P> {
    /// Allocates a zeroed pagetable page and returns a mutable reference to
    /// it, plus its physical address.
    ///
    /// # Errors
    ///
    /// Returns [`PagingError`] if the page cannot be allocated.
    fn alloc() -> Result<(&'static mut Self, PhysAddr), PagingError> {
        let paddr = P::allocate_physical_page()?;
        let vaddr = P::paddr_to_vaddr(paddr);
        // SAFETY: allocate_physical_page returns a unique, zeroed frame and
        // paddr_to_vaddr returns a valid virtual mapping for it.
        let page = unsafe { Self::from_vaddr(vaddr) };
        Ok((page, paddr))
    }

    /// Converts a pagetable entry to a mutable reference to a [`PTPage`],
    /// if the entry is present and not huge.
    pub fn from_entry(entry: PTEntry<A>) -> Option<&'static mut Self> {
        if !entry.present() || entry.huge() {
            return None;
        }

        let address = P::paddr_to_vaddr(entry.address());
        // SAFETY: Every PTEntry points to a previously allocated page-table
        // page, so this pointer dereference is safe.
        Some(unsafe { Self::from_vaddr(address) })
    }

    /// Generates a `PTPage` from a virtual address.
    /// # Safety
    /// The caller must ensure that the virtual address is a valid page table.
    pub unsafe fn from_vaddr(vaddr: VirtAddr) -> &'static mut Self {
        // SAFETY: the caller guarantees the correctness of the virtual
        // address.
        unsafe { &mut *vaddr.as_mut_ptr::<Self>() }
    }
}

/// Can be used to access page table entries by index.
impl<A: ArchPagingMeta, P: PagingHandler> Index<usize> for PTPage<A, P> {
    type Output = PTEntry<A>;

    fn index(&self, index: usize) -> &PTEntry<A> {
        &self.entries[index]
    }
}

/// Can be used to modify page table entries by index.
impl<A: ArchPagingMeta, P: PagingHandler> IndexMut<usize> for PTPage<A, P> {
    fn index_mut(&mut self, index: usize) -> &mut PTEntry<A> {
        &mut self.entries[index]
    }
}

/// Mapping levels of page table entries.
#[derive(Debug)]
pub struct Mapping<'a, P: ArchPagingMeta> {
    pub level: PageLevel,
    pub entry: &'a mut PTEntry<P>,
}

impl<'a, P: ArchPagingMeta> Mapping<'a, P> {
    /// Construct a `Mapping` at the given level.
    pub fn new(level: PageLevel, entry: &'a mut PTEntry<P>) -> Self {
        Self { level, entry }
    }
}

/// A physical address within a page frame
#[derive(Clone, Copy, Debug)]
pub enum PageFrame<A: ArchPagingMeta> {
    Size4K(PhysAddr),
    Size2M(PhysAddr),
    Size1G(PhysAddr),
    #[doc(hidden)]
    _Phantom(PhantomData<fn() -> A>),
}

impl<A: ArchPagingMeta> PageFrame<A> {
    /// Get the address from the page frame, including the shared bit.
    pub fn page_frame(&self) -> PhysAddr {
        let paddr = match *self {
            Self::Size4K(pa) => pa,
            Self::Size2M(pa) => pa,
            Self::Size1G(pa) => pa,
            Self::_Phantom(_) => PhysAddr::null(),
        };
        // Redundant but explicit.
        A::strip_confidentiality_bits(paddr)
    }

    /// Get the address from the page frame, excluding the C/shared bit.
    pub fn address(&self) -> PhysAddr {
        A::strip_shared_address_bits(self.page_frame())
    }

    pub fn size(&self) -> usize {
        match self {
            Self::Size4K(_) => PAGE_SIZE,
            Self::Size2M(_) => PAGE_SIZE_2M,
            Self::Size1G(_) => PAGE_SIZE_1G,
            Self::_Phantom(_) => 0,
        }
    }

    pub fn start(&self) -> PhysAddr {
        let end = self.address().bits() & !(self.size() - 1);
        end.into()
    }

    pub fn end(&self) -> PhysAddr {
        self.start() + self.size()
    }
}

/// A page table hierarchy rooted at `L::TOP_LEVEL`.
///
/// The root is a concrete page-table page (`PTPage`) whose level is described
/// by `L`. This can represent either a complete top-level page table, such as
/// a PML4-rooted table, or a lower-level subtree, such as a PDPT-rooted table
/// that is installed into a top-level page table.
///
/// Ownership and synchronization are left to the OS-specific code.
/// If a lower-level subtree is shared between multiple top-level page tables,
/// all users of that subtree must coordinate updates to avoid concurrent
/// modifications to the same page-table entries.
#[repr(C)]
#[derive(Debug, FromZeros)]
pub struct GenericPageTable<A: ArchPagingMeta, P: PagingHandler, L: PagingLevel> {
    root: PTPage<A, P>,
    _level: PhantomData<L>,
}

/// Methods that use the self-map to inspect the *active* top-level page table.
///
/// Self-mapped address calculations assume a PML4-rooted table.
impl<A: ArchPagingMeta, P: PagingHandler + SelfMap> GenericPageTable<A, P, PagingLevel3> {
    /// Install the self-map entry in a freshly zeroed page table.
    ///
    /// `paddr` is the physical address of *this* page table's root page.
    /// The self-map PML4 entry is written at [`SelfMap::SELFMAP_IDX`].
    pub fn init_self_map(&mut self, paddr: PhysAddr) {
        let entry = &mut self.root_mut()[P::SELFMAP_IDX];
        let flags = A::PTFlags::PRESENT
            | A::PTFlags::WRITABLE
            | A::PTFlags::ACCESSED
            | A::PTFlags::DIRTY
            | A::PTFlags::NX;
        entry.set(A::make_private_address(paddr), flags);
    }

    const fn pte_base_vaddr() -> VirtAddr {
        virt_from_lvl_idx(P::SELFMAP_IDX, PagingLevel3::TOP_LEVEL)
    }

    /// Calculate the virtual address of a PTE in the self-map, which maps a
    /// specified virtual address.
    ///
    /// # Parameters
    /// - `vaddr': The virtual address whose PTE should be located.
    ///
    /// # Returns
    /// The virtual address of the PTE.
    fn get_pte_address(vaddr: VirtAddr) -> VirtAddr {
        Self::pte_base_vaddr() + ((usize::from(vaddr) & 0x0000_FFFF_FFFF_F000) >> PTE_SHIFT)
    }

    /// Perform a virtual to physical translation using the self-map.
    ///
    /// # Parameters
    /// - `vaddr': The virtual address to translate.
    ///
    /// # Returns
    /// Some(PageFrame) if the virtual address is valid.
    /// None if the virtual address is not valid.
    pub fn virt_to_frame(vaddr: VirtAddr) -> Option<PageFrame<A>> {
        let pte_addr = Self::get_pte_address(vaddr);
        let pde_addr = Self::get_pte_address(pte_addr);
        let pdpe_addr = Self::get_pte_address(pde_addr);
        let pml4e_addr = Self::get_pte_address(pdpe_addr);

        // SAFETY: Check each entry in the paging hierarchy to ensure it is
        // safe to read the next entry.
        let pml4e: PTEntry<A> = unsafe { PTEntry::read_pte(pml4e_addr) };
        if !pml4e.present() {
            return None;
        }

        // There is no need to check for a large page in the PML4E because
        // the architecture does not support the large bit at the top-level
        // entry.  If a large page is detected at a lower level of the
        // hierarchy, the low bits from the virtual address must be combined
        // with the physical address from the PDE/PDPE.

        // SAFETY: The PML4E was checked to be present, so the PDPE exists
        // and can be read safely.
        let pdpe: PTEntry<A> = unsafe { PTEntry::read_pte(pdpe_addr) };
        if !pdpe.present() {
            return None;
        }
        if pdpe.huge() {
            let pa = pdpe.page_frame() + (usize::from(vaddr) & 0x3FFF_FFFF);
            return Some(PageFrame::Size1G(pa));
        }

        // SAFETY: The PDPE was checked to be present and not to be a huge
        // page. So the PDE exists and can be read safely.
        let pde: PTEntry<A> = unsafe { PTEntry::read_pte(pde_addr) };
        if !pde.present() {
            return None;
        }
        if pde.huge() {
            let pa = pde.page_frame() + (usize::from(vaddr) & 0x001F_FFFF);
            return Some(PageFrame::Size2M(pa));
        }

        // SAFETY: The PDE was checked to be present and not to be a huge
        // page. So the PTE exists and can be read safely.
        let pte: PTEntry<A> = unsafe { PTEntry::read_pte(pte_addr) };
        if pte.present() {
            let pa = pte.page_frame() + (usize::from(vaddr) & 0xFFF);
            Some(PageFrame::Size4K(pa))
        } else {
            None
        }
    }
}

/// Methods for page table hierarchies, independent of the self-map.
impl<A: ArchPagingMeta, P: PagingHandler, L: PagingLevel> GenericPageTable<A, P, L> {
    /// Recursively free all page table pages starting from the root page.
    pub fn free_lvl(page: &PTPage<A, P>, level: PageLevel) {
        if level <= PageLevel::Level0 {
            return;
        }
        for entry in page.entries.iter() {
            if let Some(child) = PTPage::from_entry(*entry) {
                if let Some(next) = level.next_down() {
                    Self::free_lvl(child, next);
                }
                let paddr = entry.address();
                // SAFETY: the page was allocated via PagingHandler::allocate_physical_page.
                unsafe { P::deallocate_physical_page(paddr) };
            }
        }
    }

    /// Get a reference to the root page.
    pub fn root(&self) -> &PTPage<A, P> {
        &self.root
    }

    /// Get a mutable reference to the root page.
    pub fn root_mut(&mut self) -> &mut PTPage<A, P> {
        &mut self.root
    }

    /// Computes the index within a page table at the given level for a
    /// virtual address `vaddr`.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address to compute the index for.
    ///
    /// # Returns
    /// The index within the page table.
    pub fn index<const LVL: usize>(vaddr: VirtAddr) -> usize {
        vaddr.to_pgtbl_idx::<LVL>()
    }

    /// Copy an entry `entry` from another `GenericPageTable`.
    pub fn copy_entry(&mut self, other: &Self, entry: usize) {
        self.root.entries[entry] = other.root.entries[entry];
    }

    /// Walks a page table at level 0 to find a mapping.
    ///
    /// # Parameters
    /// - `page`: A mutable reference to the root page table.
    /// - `vaddr`: The virtual address to find a mapping for.
    ///
    /// # Returns
    /// A `Mapping` representing the found mapping.
    fn walk_addr_lvl0(page: &mut PTPage<A, P>, vaddr: VirtAddr) -> Mapping<'_, A> {
        let idx = vaddr.to_pgtbl_idx::<0>();
        Mapping::new(PageLevel::Level0, &mut page[idx])
    }

    /// Walks a page table at level 1 to find a mapping.
    ///
    /// # Parameters
    /// - `page`: A mutable reference to the root page table.
    /// - `vaddr`: The virtual address to find a mapping for.
    ///
    /// # Returns
    /// A `Mapping` representing the found mapping.
    fn walk_addr_lvl1<'a>(page: &'a mut PTPage<A, P>, vaddr: VirtAddr) -> Mapping<'a, A> {
        let idx = vaddr.to_pgtbl_idx::<1>();
        let entry = page[idx];
        match PTPage::from_entry(entry) {
            Some(next) => Self::walk_addr_lvl0(next, vaddr),
            None => Mapping::new(PageLevel::Level1, &mut page[idx]),
        }
    }

    /// Walks a page table at level 2 to find a mapping.
    ///
    /// # Parameters
    /// - `page`: A mutable reference to the root page table.
    /// - `vaddr`: The virtual address to find a mapping for.
    ///
    /// # Returns
    /// A `Mapping` representing the found mapping.
    fn walk_addr_lvl2<'a>(page: &'a mut PTPage<A, P>, vaddr: VirtAddr) -> Mapping<'a, A> {
        let idx = vaddr.to_pgtbl_idx::<2>();
        let entry = page[idx];
        match PTPage::from_entry(entry) {
            Some(next) => Self::walk_addr_lvl1(next, vaddr),
            None => Mapping::new(PageLevel::Level2, &mut page[idx]),
        }
    }

    /// Walks the page table to find a mapping for a given virtual address.
    ///
    /// # Parameters
    /// - `page`: A mutable reference to the root page table.
    /// - `vaddr`: The virtual address to find a mapping for.
    ///
    /// # Returns
    /// A `Mapping` representing the found mapping.
    fn walk_addr_lvl3<'a>(page: &'a mut PTPage<A, P>, vaddr: VirtAddr) -> Mapping<'a, A> {
        let idx = vaddr.to_pgtbl_idx::<3>();
        let entry = page[idx];
        match PTPage::from_entry(entry) {
            Some(next) => Self::walk_addr_lvl2(next, vaddr),
            None => Mapping::new(PageLevel::Level3, &mut page[idx]),
        }
    }

    /// Walk the virtual address and return the corresponding mapping.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address to find a mapping for.
    ///
    /// # Returns
    /// A `Mapping` representing the found mapping.
    pub fn walk_addr(&mut self, vaddr: VirtAddr) -> Mapping<'_, A> {
        match L::TOP_LEVEL {
            PageLevel::Level3 => Self::walk_addr_lvl3(&mut self.root, vaddr),
            PageLevel::Level2 => Self::walk_addr_lvl2(&mut self.root, vaddr),
            _ => unreachable!(),
        }
    }

    /// Allocate from level 3 (PML4E) down to the target level.
    fn alloc_pte_lvl3<'a>(
        entry: &'a mut PTEntry<A>,
        vaddr: VirtAddr,
        size: PageSize,
        parent_flags: A::PTFlags,
    ) -> Mapping<'a, A> {
        if entry.flags().contains(A::PTFlags::PRESENT) {
            return Mapping::new(PageLevel::Level3, entry);
        }

        let Ok((page, paddr)) = PTPage::<A, P>::alloc() else {
            return Mapping::new(PageLevel::Level3, entry);
        };

        entry.set(A::make_private_address(paddr), parent_flags);

        let idx = vaddr.to_pgtbl_idx::<2>();
        Self::alloc_pte_lvl2(&mut page[idx], vaddr, size, parent_flags)
    }

    /// Allocate from level 2 (PDPTE) down to the target level.
    fn alloc_pte_lvl2<'a>(
        entry: &'a mut PTEntry<A>,
        vaddr: VirtAddr,
        size: PageSize,
        parent_flags: A::PTFlags,
    ) -> Mapping<'a, A> {
        if entry.flags().contains(A::PTFlags::PRESENT) {
            return Mapping::new(PageLevel::Level2, entry);
        }

        let Ok((page, paddr)) = PTPage::<A, P>::alloc() else {
            return Mapping::new(PageLevel::Level2, entry);
        };

        entry.set(A::make_private_address(paddr), parent_flags);

        let idx = vaddr.to_pgtbl_idx::<1>();
        Self::alloc_pte_lvl1(&mut page[idx], vaddr, size, parent_flags)
    }

    /// Allocate from level 1 (PDE) down to level 0.
    /// Returns at level 1 if `size` is `Huge` (2 MiB page).
    fn alloc_pte_lvl1<'a>(
        entry: &'a mut PTEntry<A>,
        vaddr: VirtAddr,
        size: PageSize,
        parent_flags: A::PTFlags,
    ) -> Mapping<'a, A> {
        let flags = entry.flags();
        if size == PageSize::Huge || flags.contains(A::PTFlags::PRESENT) {
            return Mapping::new(PageLevel::Level1, entry);
        }

        let Ok((page, paddr)) = PTPage::<A, P>::alloc() else {
            return Mapping::new(PageLevel::Level1, entry);
        };

        entry.set(A::make_private_address(paddr), parent_flags);

        let idx = vaddr.to_pgtbl_idx::<0>();
        Mapping::new(PageLevel::Level0, &mut page[idx])
    }

    /// Allocates a 4KB page table entry for a given virtual address.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address for which to allocate the PTE.
    /// - `parent_flags`: The flags to apply to the allocated page table entries.
    ///
    /// # Returns
    /// A `Mapping` representing the allocated or existing PTE for the address.
    fn alloc_pte_4k(&mut self, vaddr: VirtAddr, parent_flags: A::PTFlags) -> Mapping<'_, A> {
        let m = self.walk_addr(vaddr);
        match m.level {
            PageLevel::Level0 => m,
            PageLevel::Level1 => Self::alloc_pte_lvl1(m.entry, vaddr, PageSize::Regular, parent_flags),
            PageLevel::Level2 => Self::alloc_pte_lvl2(m.entry, vaddr, PageSize::Regular, parent_flags),
            PageLevel::Level3 => Self::alloc_pte_lvl3(m.entry, vaddr, PageSize::Regular, parent_flags),
        }
    }

    /// Allocates a 2MB page table entry for a given virtual address.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address for which to allocate the PTE.
    /// - `parent_flags`: The flags to apply to the allocated page table entries.
    ///
    /// # Returns
    /// A `Mapping` representing the allocated or existing PTE for the address.
    fn alloc_pte_2m(&mut self, vaddr: VirtAddr, parent_flags: A::PTFlags) -> Mapping<'_, A> {
        let m = self.walk_addr(vaddr);
        match m.level {
            PageLevel::Level0 | PageLevel::Level1 => m,
            PageLevel::Level2 => Self::alloc_pte_lvl2(m.entry, vaddr, PageSize::Huge, parent_flags),
            PageLevel::Level3 => Self::alloc_pte_lvl3(m.entry, vaddr, PageSize::Huge, parent_flags),
        }
    }

    /// Splits a 2MB page into 4KB pages.
    ///
    /// # Parameters
    /// - `entry`: The 2M page table entry to split.
    ///
    /// # Returns
    /// A result indicating success or an error [`PagingError`] in failure.
    fn do_split_4k(entry: &mut PTEntry<A>) -> Result<(), PagingError> {
        let (page, paddr) = PTPage::<A, P>::alloc()?;
        let mut flags = entry.flags();

        assert!(flags.contains(A::PTFlags::HUGE));

        let addr_2m = PhysAddr::from(entry.address().bits() & 0x000f_ffff_fff0_0000);

        flags.remove(A::PTFlags::HUGE);

        // Prepare PTE leaf page
        for (i, e) in page.entries.iter_mut().enumerate() {
            let addr_4k = addr_2m + (i * PAGE_SIZE);
            e.clear();
            e.set(A::make_private_address(addr_4k), flags);
        }

        entry.set(A::make_private_address(paddr), flags);

        A::flush_tlb_global();

        Ok(())
    }

    /// Splits a page into 4KB pages if it is part of a larger mapping.
    ///
    /// # Parameters
    /// - `mapping`: The mapping to split.
    ///
    /// # Returns
    /// A result indicating success or an error [`PagingError`].
    pub fn split_4k(mapping: Mapping<'_, A>) -> Result<(), PagingError> {
        match mapping.level {
            PageLevel::Level0 => Ok(()),
            PageLevel::Level1 => Self::do_split_4k(mapping.entry),
            _ => Err(PagingError::NotMapped),
        }
    }

    fn make_pte_shared(entry: &mut PTEntry<A>) {
        let flags = entry.flags();
        let addr = entry.address();

        // entry.address() returned with c-bit clear already
        entry.set(A::make_shared_address(addr), flags);
    }

    fn make_pte_private(entry: &mut PTEntry<A>) {
        let flags = entry.flags();
        let addr = entry.address();

        // entry.address() returned with c-bit clear already
        entry.set(A::make_private_address(addr), flags);
    }

    /// Sets the shared state for a 4KB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address of the page.
    ///
    /// # Returns
    /// A result indicating success or an error [`PagingError`] if the
    /// operation fails.
    pub fn set_shared_4k(&mut self, vaddr: VirtAddr) -> Result<(), PagingError> {
        let mapping = self.walk_addr(vaddr);
        Self::split_4k(mapping)?;

        if let Mapping {
            level: PageLevel::Level0,
            entry,
        } = self.walk_addr(vaddr)
        {
            Self::make_pte_shared(entry);
            Ok(())
        } else {
            Err(PagingError::NotMapped)
        }
    }

    /// Sets the encryption state for a 4KB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address of the page.
    ///
    /// # Returns
    /// A result indicating success or an error [`PagingError`].
    pub fn set_encrypted_4k(&mut self, vaddr: VirtAddr) -> Result<(), PagingError> {
        let mapping = self.walk_addr(vaddr);
        Self::split_4k(mapping)?;

        if let Mapping {
            level: PageLevel::Level0,
            entry,
        } = self.walk_addr(vaddr)
        {
            Self::make_pte_private(entry);
            Ok(())
        } else {
            Err(PagingError::NotMapped)
        }
    }

    /// Maps a 2MB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address to map.
    /// - `paddr`: The physical address to map to.
    /// - `flags`: The flags to apply to the mapping.
    /// - `shared`: Indicates whether the mapping is shared.
    ///
    /// # Returns
    /// A result indicating success or failure ([`PagingError`]).
    ///
    /// # Panics
    /// Panics if either `vaddr` or `paddr` is not aligned to a 2MB boundary.
    pub fn map_2m_with_parent_flags(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
        parent_flags: A::PTFlags
    ) -> Result<(), PagingError> {
        assert!(vaddr.is_aligned(PAGE_SIZE_2M));
        assert!(paddr.is_aligned(PAGE_SIZE_2M));
        let mapping = self.alloc_pte_2m(vaddr, parent_flags);
        let addr = if !shared {
            A::make_private_address(paddr)
        } else {
            A::make_shared_address(paddr)
        };

        if let Mapping {
            level: PageLevel::Level1,
            entry,
        } = mapping
        {
            entry.set(addr, flags | A::PTFlags::HUGE);
            Ok(())
        } else {
            Err(PagingError::AllocFrame)
        }
    }


    /// Maps a 2MB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address to map.
    /// - `paddr`: The physical address to map to.
    /// - `flags`: The flags to apply to the mapping.
    /// - `shared`: Indicates whether the mapping is shared.
    ///
    /// # Returns
    /// A result indicating success or failure ([`PagingError`]).
    ///
    /// # Panics
    /// Panics if either `vaddr` or `paddr` is not aligned to a 2MB boundary.
    pub fn map_2m(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
    ) -> Result<(), PagingError> {
        self.map_2m_with_parent_flags(vaddr, paddr, flags, shared, A::PTFlags::parent_flags())
    }

    /// Unmaps a 2MB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address of the mapping to unmap.
    ///
    /// # Panics
    /// Panics if `vaddr` is not aligned to a 2MB boundary.
    pub fn unmap_2m(&mut self, vaddr: VirtAddr) -> Option<PTEntry<A>> {
        assert!(vaddr.is_aligned(PAGE_SIZE_2M));

        let mapping = self.walk_addr(vaddr);

        match mapping.level {
            PageLevel::Level0 => unreachable!(),
            PageLevel::Level1 => {
                let entry = *mapping.entry;
                mapping.entry.clear();
                Some(entry)
            }
            _ => {
                assert!(!mapping.entry.present());
                None
            }
        }
    }

    /// Maps a 4KB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address to map.
    /// - `paddr`: The physical address to map to.
    /// - `flags`: The flags to apply to the mapping.
    /// - `shared`: Indicates whether the mapping is shared.
    /// - `parent_flags`: The flags to apply to parent page table entries if new page tables
    ///    need to be allocated.
    ///
    /// # Returns
    /// A result indicating success or failure ([`PagingError`]).
    pub fn map_4k_with_parent_flags(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
        parent_flags: A::PTFlags,
    ) -> Result<(), PagingError> {
        let mapping = self.alloc_pte_4k(vaddr, parent_flags);
        let addr = if !shared {
            A::make_private_address(paddr)
        } else {
            A::make_shared_address(paddr)
        };

        if let Mapping {
            level: PageLevel::Level0,
            entry,
        } = mapping
        {
            entry.set(addr, flags);
            Ok(())
        } else {
            Err(PagingError::AllocFrame)
        }
    }

    /// Maps a 4KB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address to map.
    /// - `paddr`: The physical address to map to.
    /// - `flags`: The flags to apply to the mapping.
    /// - `shared`: Indicates whether the mapping is shared.
    ///
    /// # Returns
    /// A result indicating success or failure ([`PagingError`]).
    pub fn map_4k(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
    ) -> Result<(), PagingError> {
        self.map_4k_with_parent_flags(vaddr, paddr, flags, shared, A::PTFlags::parent_flags())
    }

    /// Unmaps a 4KB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address of the mapping to unmap.
    pub fn unmap_4k(&mut self, vaddr: VirtAddr) -> Option<PTEntry<A>> {
        let mapping = self.walk_addr(vaddr);

        match mapping.level {
            PageLevel::Level0 => {
                let entry = *mapping.entry;
                mapping.entry.clear();
                Some(entry)
            }
            _ => {
                assert!(!mapping.entry.present());
                None
            }
        }
    }

    /// Retrieves the physical address of a mapping.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address to query.
    ///
    /// # Returns
    /// The physical address of the mapping if present; otherwise, an error
    /// ([`PagingError`]).
    pub fn phys_addr(&mut self, vaddr: VirtAddr) -> Result<PhysAddr, PagingError> {
        let mapping = self.walk_addr(vaddr);

        match mapping.level {
            PageLevel::Level0 => {
                let offset = vaddr.page_offset();
                let entry = mapping.entry;
                if !entry.flags().contains(A::PTFlags::PRESENT) {
                    return Err(PagingError::NotMapped);
                }
                Ok(entry.address() + offset)
            }
            PageLevel::Level1 => {
                let offset = vaddr.bits() & (PAGE_SIZE_2M - 1);
                let entry = mapping.entry;
                if !entry.flags().contains(A::PTFlags::PRESENT)
                    || !entry.flags().contains(A::PTFlags::HUGE)
                {
                    return Err(PagingError::NotMapped);
                }
                Ok(entry.address() + offset)
            }
            _ => Err(PagingError::NotMapped),
        }
    }
}
