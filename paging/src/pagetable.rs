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
//! page table hierarchy (512 entries per page), supporting three page
//! sizes: 4 KiB, 2 MiB, and 1 GiB. This covers x86_64 (PML4) and
//! ARM64 with 4 KiB granule. Other granule sizes (16 KiB, 64 KiB on
//! ARM64) are not supported.

use crate::address::{Address, PhysAddr, VirtAddr};
use crate::sizes::{PAGE_SIZE, PAGE_SIZE_1G, PAGE_SIZE_2M, PageSize};
use bitflags::Flags;
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use zerocopy::{FromBytes, FromZeros};

/// Number of entries in a page table (4KB/8B).
pub const ENTRY_COUNT: usize = 512;

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
/// `feature_mask`) have default implementations derived from the masks.
/// Override only if the platform requires non-standard behaviour.
///
/// # Implementer requirements
///
/// * All methods must be stateless (no `&self`) — the type is used as a
///   zero-sized marker and never instantiated.
/// * `private_pte_mask` and `shared_pte_mask` must not share any set bits —
///   a page is either private *or* shared.
/// * `make_private_address` and `make_shared_address` must be idempotent.
pub trait ArchPagingMeta: 'static + Copy + FromBytes {
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

    /// Returns a bitmask of [`PTEntryFlags`] that the hardware supports.
    ///
    /// [`PTEntry::set`] filters its `flags` argument through this mask so
    /// that unsupported bits (e.g., `GLOBAL` before CR4.PGE is enabled)
    /// are silently cleared. The default allows all flags.
    fn feature_mask() -> Self::PTFlags {
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
/// * **`allocate_frame`** — every successful call returns a *unique*,
///   page-aligned, *zeroed* physical frame whose address is *clean* (no
///   encryption/confidentiality bits). The frame remains valid until a
///   matching `deallocate_frame` call.
///
/// * **`deallocate_frame`** — `paddr` is a value previously returned by
///   `allocate_frame` that has not yet been freed.
///
/// # Cross-method invariant
///
/// `paddr_to_vaddr` must return a valid, writable mapping for every
/// address returned by `allocate_frame`. The generic page table code
/// calls `allocate_frame` and immediately passes the result to
/// `paddr_to_vaddr` in order to zero-initialise and populate newly
/// allocated page table pages. This invariant can be satisfied either
/// by a linear map of all physical memory (so `paddr_to_vaddr` works
/// for any physical address) or by having `allocate_frame` return only
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
    fn allocate_frame() -> Result<PhysAddr, PagingError>;

    /// Deallocate a page-table frame previously returned by
    /// [`allocate_frame`](Self::allocate_frame).
    ///
    /// # Safety
    ///
    /// `paddr` must be a clean physical address previously returned by
    /// `allocate_frame` and not yet freed.
    unsafe fn deallocate_frame(paddr: PhysAddr);
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
/// * `pte_base()` must return the virtual base address of that self-map
///   window. For a self-map installed at PML4 index *N*, this is
///   `sign_extend(N << 39 | N << 30 | N << 21 | N << 12)`.
/// * The self-map entry must be installed before any method in the
///   `impl<..., P: PagingHandler + SelfMap>` block is called.
pub trait SelfMap {
    /// Returns the virtual base address of the PTE self-map region.
    fn pte_base() -> VirtAddr;
}

pub trait GenericPageTableFlags:
    bitflags::Flags<Bits = usize>
    + core::ops::BitAnd<Output = Self>
    + core::ops::BitOr<Output = Self>
    + Copy
    + Clone
{
    const PRESENT: Self;
    const WRITABLE: Self;
    const USER: Self;
    const ACCESSED: Self;
    const DIRTY: Self;
    const HUGE: Self;
    const GLOBAL: Self;
    const NX: Self;
}

/// A single page table entry, parameterised by [`ArchPagingMeta`].
///
/// `P` determines how encryption/confidentiality bits are interpreted.
/// [`address()`](Self::address) strips both the private and shared bits,
/// [`page_frame()`](Self::page_frame) strips only the private bits, and
/// [`set()`](Self::set) filters flags through
/// [`P::feature_mask()`](ArchPagingMeta::feature_mask).
///
/// `PhantomData<fn() -> P>` makes `P` covariant and keeps `PTEntry`
/// zero-cost (no runtime storage for `P`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes)]
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

    /// Set the page table entry with the specified address and flags,
    /// bypassing the [`ArchPagingMeta::feature_mask`] filter.
    ///
    /// Use [`set`](Self::set) for normal mappings. This variant is for
    /// cases where the caller has already applied the correct mask or
    /// needs to write flags that `feature_mask` would otherwise strip.
    ///
    /// # Panics
    ///
    /// Panics if `addr` has any bits set outside the 52-bit physical
    /// address field (`0x000f_ffff_ffff_f000`).
    pub fn set_unrestricted(&mut self, addr: PhysAddr, flags: A::PTFlags) {
        let addr = addr.bits();
        assert_eq!(addr & !A::address_mask(), 0);
        self.entry = PhysAddr::from(addr | flags.bits());
    }

    /// Set the page table entry with the specified address, with flags
    /// constrained to the supported feature flags.
    ///
    /// addr: for bits [51:12] of the entry, including C/shared bits.
    pub fn set(&mut self, addr: PhysAddr, flags: A::PTFlags) {
        self.set_unrestricted(addr, flags & A::feature_mask());
    }

    /// Get the raw physical address field from the entry.
    ///
    /// Returns bits `[51:12]` of the entry — the address *including* any
    /// encryption/confidentiality bits the hardware stores in the upper
    /// physical address bits.
    pub fn raw_address(&self) -> PhysAddr {
        PhysAddr::from(self.raw() & A::address_mask())
    }

    /// Get the page frame address with the private (confidentiality) bits
    /// stripped but the shared bit retained.
    ///
    /// Useful when the caller needs to distinguish shared vs. private
    /// mappings (e.g., when re-writing the encryption state of a PTE).
    pub fn page_frame(&self) -> PhysAddr {
        A::strip_confidentiality_bits(self.raw_address())
    }

    /// Get the clean physical address — both confidentiality and shared
    /// bits are stripped.
    ///
    /// This is the address suitable for passing to
    /// [`PagingHandler::paddr_to_vaddr`] or for arithmetic on physical
    /// memory ranges.
    pub fn address(&self) -> PhysAddr {
        A::strip_shared_address_bits(self.page_frame())
    }

    /// Inserts the private address mask if the page is present.
    pub fn make_private_if_present(&mut self) {
        if self.flags().contains(A::PTFlags::PRESENT) {
            self.entry = A::make_private_address(self.entry);
        }
    }

    /// Returns `true` if this entry's address has the shared mask applied.
    pub fn is_shared(&self) -> bool {
        A::is_shared_address(self.raw_address())
    }

    /// Read a page table entry from the specified virtual address.
    ///
    /// # Safety
    ///
    /// Reads from an arbitrary virtual address, making this essentially a
    /// raw pointer read. The caller must be certain to calculate the correct
    /// address.
    pub unsafe fn read_pte(vaddr: VirtAddr) -> Self {
        // SAFETY: When the method's safety requirements are met, the raw
        // pointer read is safe.
        unsafe { *vaddr.as_ptr::<Self>() }
    }
}

/// A page of page table entries (4 KiB / 512 entries).
///
/// Generic over `A` ([`ArchPagingMeta`]) for encryption-aware entry
/// interpretation, and `P` ([`PagingHandler`]) for frame allocation and
/// physical-to-virtual translation. `PhantomData<fn() -> P>` keeps the
/// type zero-cost at runtime.
#[repr(C)]
#[derive(Debug, FromBytes)]
pub struct PTPage<A: ArchPagingMeta, P: PagingHandler> {
    pub entries: [PTEntry<A>; ENTRY_COUNT],
    _phantom: PhantomData<fn() -> P>,
}

impl<A: ArchPagingMeta, P: PagingHandler> PTPage<A, P> {
    /// Allocate a new zeroed page-table page, returning a mutable reference
    /// and its clean physical address (no encryption bits).
    fn alloc() -> Result<(&'static mut Self, PhysAddr), PagingError> {
        let paddr = P::allocate_frame()?;
        let vaddr = P::paddr_to_vaddr(paddr);
        // SAFETY: allocate_frame returns a unique, zeroed frame and
        // paddr_to_vaddr returns a valid virtual mapping for it.
        let page = unsafe { Self::from_vaddr(vaddr) };
        Ok((page, paddr))
    }

    /// Converts a pagetable entry to a mutable reference to a [`PTPage`],
    /// if the entry is present and not huge. Uses `P::paddr_to_vaddr`
    /// to translate the physical address.
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
    ///
    /// # Safety
    ///
    /// The caller must ensure that the virtual address is a valid page table.
    pub unsafe fn from_vaddr(vaddr: VirtAddr) -> &'static mut Self {
        // SAFETY: the caller guarantees the correctness of the virtual address.
        unsafe { &mut *vaddr.as_mut_ptr::<Self>() }
    }

    /// Recursively free all child page table pages at the given `level`.
    ///
    /// Walks every present, non-huge entry and frees the referenced page.
    /// For levels > 1, descends into each child page first.
    /// Recursion depth is bounded by the page table depth (max 4 levels).
    pub fn free_pages(&self, level: usize) {
        if level == 0 {
            return;
        }
        for entry in self.entries.iter() {
            if let Some(child) = Self::from_entry(*entry) {
                child.free_pages(level - 1);
                let paddr = A::strip_confidentiality_bits(entry.raw_address());
                // SAFETY: the page was allocated via PagingHandler::allocate_frame.
                unsafe { P::deallocate_frame(paddr) };
            }
        }
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

/// Mapping level returned by the page table walk functions.
///
/// Indicates how deep the walk descended before finding a leaf entry
/// or an absent entry:
///
/// * `Level3` — PML4E (no PDPT page)
/// * `Level2` — PDPTE (no PD page, or 1 GiB huge page)
/// * `Level1` — PDE   (no PT page, or 2 MiB huge page)
/// * `Level0` — PTE   (4 KiB page)
#[derive(Debug)]
pub enum Mapping<'a, P: ArchPagingMeta> {
    Level3(&'a mut PTEntry<P>),
    Level2(&'a mut PTEntry<P>),
    Level1(&'a mut PTEntry<P>),
    Level0(&'a mut PTEntry<P>),
}

/// A physical address tagged with its page size (4 KiB, 2 MiB, or 1 GiB).
///
/// Returned by [`GenericPageTable::virt_to_frame`]; the stored address
/// retains the shared bit (via [`PTEntry::page_frame`]) and includes any
/// sub-page offset from the original virtual address.
///
/// * [`page_frame()`](Self::page_frame) — strips only the confidentiality
///   (private) bits, keeping the shared bit.
/// * [`address()`](Self::address) — strips both confidentiality and shared
///   bits, returning a clean physical address.
///
/// Type Invariant:
///     Inner pa is a valid page frame without C-bit.
#[derive(Clone, Copy, Debug)]
pub enum PageFrame<A: ArchPagingMeta> {
    Size4K(PhysAddr),
    Size2M(PhysAddr),
    Size1G(PhysAddr),
    #[doc(hidden)]
    _Phantom(PhantomData<fn() -> A>),
}

impl<A: ArchPagingMeta> PageFrame<A> {
    /// Get the address with confidentiality bits stripped but shared bit
    /// retained.
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

    /// Get the clean address with both confidentiality and shared bits
    /// stripped.
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

/// Root page table structure (PML4).
///
/// Generic over:
/// * `A: ArchPagingMeta` — encryption masks, TLB flush, feature mask.
/// * `P: PagingHandler` — frame allocation and paddr→vaddr translation.
///
/// Internally contains a single [`PTPage`] as the level-4 (PML4) page.
/// Lower-level pages are allocated on demand via [`PagingHandler::allocate_frame`].
#[repr(C)]
#[derive(Debug, FromZeros)]
pub struct GenericPageTable<A: ArchPagingMeta, P: PagingHandler> {
    root: PTPage<A, P>,
}

impl<A: ArchPagingMeta, P: PagingHandler> Default for GenericPageTable<A, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: ArchPagingMeta, P: PagingHandler> Drop for GenericPageTable<A, P> {
    fn drop(&mut self) {
        // Hardcoded to 4-level (PML4). The root page itself is inline
        // and freed when this struct is dropped.
        self.root.free_pages(3);
    }
}

impl<A: ArchPagingMeta, P: PagingHandler> GenericPageTable<A, P> {
    /// Create a new page table with zeroed root.
    pub fn new() -> Self {
        Self {
            root: PTPage {
                entries: [PTEntry {
                    entry: PhysAddr::null(),
                    _phantom: PhantomData,
                }; ENTRY_COUNT],
                _phantom: PhantomData,
            },
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
    pub fn index<const L: usize>(vaddr: VirtAddr) -> usize {
        vaddr.to_pgtbl_idx::<L>()
    }

    /// Copy an entry `entry` from another [`GenericPageTable`].
    pub fn copy_entry(&mut self, other: &Self, entry: usize) {
        self.root.entries[entry] = other.root.entries[entry];
    }

    /// Walk the page table hierarchy for `vaddr` and return the deepest
    /// [`Mapping`] reached.
    pub fn walk_addr(&mut self, vaddr: VirtAddr) -> Mapping<'_, A> {
        Self::walk_addr_lvl3(&mut self.root, vaddr)
    }

    /// Walk starting at a level-3 (PML4) page.
    fn walk_addr_lvl3<'a>(page: &'a mut PTPage<A, P>, vaddr: VirtAddr) -> Mapping<'a, A> {
        let idx = Self::index::<3>(vaddr);
        let entry = page[idx];
        match PTPage::from_entry(entry) {
            Some(page) => Self::walk_addr_lvl2(page, vaddr),
            None => Mapping::Level3(&mut page[idx]),
        }
    }

    /// Walk starting at a level-2 (PDPT) page.
    pub fn walk_addr_lvl2<'a>(page: &'a mut PTPage<A, P>, vaddr: VirtAddr) -> Mapping<'a, A> {
        let idx = Self::index::<2>(vaddr);
        let entry = page[idx];
        match PTPage::from_entry(entry) {
            Some(page) => Self::walk_addr_lvl1(page, vaddr),
            None => Mapping::Level2(&mut page[idx]),
        }
    }

    /// Walk starting at a level-1 (PD) page.
    fn walk_addr_lvl1<'a>(page: &'a mut PTPage<A, P>, vaddr: VirtAddr) -> Mapping<'a, A> {
        let idx = Self::index::<1>(vaddr);
        let entry = page[idx];
        match PTPage::from_entry(entry) {
            Some(page) => Self::walk_addr_lvl0(page, vaddr),
            None => Mapping::Level1(&mut page[idx]),
        }
    }

    /// Walk starting at a level-0 (PT) page — always returns `Level0`.
    fn walk_addr_lvl0(page: &mut PTPage<A, P>, vaddr: VirtAddr) -> Mapping<'_, A> {
        let idx = Self::index::<0>(vaddr);
        Mapping::Level0(&mut page[idx])
    }

    /// Walk and allocate intermediate page table levels for a 4 KiB mapping.
    ///
    /// Returns `Mapping::Level0` on success. If allocation fails at any
    /// level the walk stops and the level at which it failed is returned.
    fn alloc_pte_4k(&mut self, vaddr: VirtAddr) -> Mapping<'_, A> {
        let m = Self::walk_addr_lvl3(&mut self.root, vaddr);

        match m {
            Mapping::Level0(entry) => Mapping::Level0(entry),
            Mapping::Level1(entry) => Self::alloc_pte_lvl1(entry, vaddr, PageSize::Regular),
            Mapping::Level2(entry) => Self::alloc_pte_lvl2(entry, vaddr, PageSize::Regular),
            Mapping::Level3(entry) => Self::alloc_pte_lvl3(entry, vaddr, PageSize::Regular),
        }
    }

    /// Walk and allocate intermediate page table levels for a 2 MiB mapping.
    ///
    /// Returns `Mapping::Level1` on success (the PDE for the huge page).
    fn alloc_pte_2m(&mut self, vaddr: VirtAddr) -> Mapping<'_, A> {
        let m = Self::walk_addr_lvl3(&mut self.root, vaddr);

        match m {
            Mapping::Level0(entry) => Mapping::Level0(entry),
            Mapping::Level1(entry) => Mapping::Level1(entry),
            Mapping::Level2(entry) => Self::alloc_pte_lvl2(entry, vaddr, PageSize::Huge),
            Mapping::Level3(entry) => Self::alloc_pte_lvl3(entry, vaddr, PageSize::Huge),
        }
    }

    pub fn alloc_pte_lvl3<'a>(
        entry: &'a mut PTEntry<A>,
        vaddr: VirtAddr,
        size: PageSize,
    ) -> Mapping<'a, A> {
        let flags = entry.flags();

        if flags.contains(A::PTFlags::PRESENT) {
            return Mapping::Level3(entry);
        }

        let Ok((page, paddr)) = PTPage::<A, P>::alloc() else {
            return Mapping::Level3(entry);
        };

        let flags =
            A::PTFlags::PRESENT | A::PTFlags::WRITABLE | A::PTFlags::USER | A::PTFlags::ACCESSED;
        entry.set(A::make_private_address(paddr), flags);

        let idx = Self::index::<2>(vaddr);
        Self::alloc_pte_lvl2(&mut page[idx], vaddr, size)
    }

    pub fn alloc_pte_lvl2<'a>(
        entry: &'a mut PTEntry<A>,
        vaddr: VirtAddr,
        size: PageSize,
    ) -> Mapping<'a, A> {
        let flags = entry.flags();

        if flags.contains(A::PTFlags::PRESENT) {
            return Mapping::Level2(entry);
        }

        let Ok((page, paddr)) = PTPage::<A, P>::alloc() else {
            return Mapping::Level2(entry);
        };

        let flags =
            A::PTFlags::PRESENT | A::PTFlags::WRITABLE | A::PTFlags::USER | A::PTFlags::ACCESSED;
        entry.set(A::make_private_address(paddr), flags);

        let idx = Self::index::<1>(vaddr);
        Self::alloc_pte_lvl1(&mut page[idx], vaddr, size)
    }

    pub fn alloc_pte_lvl1<'a>(
        entry: &'a mut PTEntry<A>,
        vaddr: VirtAddr,
        size: PageSize,
    ) -> Mapping<'a, A> {
        let flags = entry.flags();

        if size == PageSize::Huge || flags.contains(A::PTFlags::PRESENT) {
            return Mapping::Level1(entry);
        }

        let Ok((page, paddr)) = PTPage::<A, P>::alloc() else {
            return Mapping::Level1(entry);
        };

        let flags =
            A::PTFlags::PRESENT | A::PTFlags::WRITABLE | A::PTFlags::USER | A::PTFlags::ACCESSED;
        entry.set(A::make_private_address(paddr), flags);

        let idx = Self::index::<0>(vaddr);
        Mapping::Level0(&mut page[idx])
    }

    /// Map a 4 KiB page at `vaddr` → `paddr`.
    ///
    /// `paddr` must be a *clean* physical address (no encryption bits).
    /// When `shared` is `false`, [`ArchPagingMeta::make_private_address`]
    /// is applied; when `true`, [`ArchPagingMeta::make_shared_address`].
    /// Flags are filtered through [`ArchPagingMeta::feature_mask`].
    pub fn map_4k(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
    ) -> Result<(), PagingError> {
        let mapping = self.alloc_pte_4k(vaddr);
        let addr = if !shared {
            A::make_private_address(paddr)
        } else {
            A::make_shared_address(paddr)
        };

        if let Mapping::Level0(entry) = mapping {
            entry.set(addr, flags);
            Ok(())
        } else {
            Err(PagingError::AllocFrame)
        }
    }

    /// Map a 2 MiB huge page at `vaddr` → `paddr`.
    ///
    /// `paddr` must be a *clean*, 2 MiB-aligned physical address.
    /// Encryption bits and the `HUGE` flag are applied automatically.
    ///
    /// # Panics
    ///
    /// Panics if either `vaddr` or `paddr` is not 2 MiB-aligned.
    pub fn map_2m(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
    ) -> Result<(), PagingError> {
        assert!(vaddr.is_aligned(PAGE_SIZE_2M));
        assert!(paddr.is_aligned(PAGE_SIZE_2M));

        let mapping = self.alloc_pte_2m(vaddr);
        let addr = if !shared {
            A::make_private_address(paddr)
        } else {
            A::make_shared_address(paddr)
        };

        if let Mapping::Level1(entry) = mapping {
            entry.set(addr, flags | A::PTFlags::HUGE);
            Ok(())
        } else {
            Err(PagingError::AllocFrame)
        }
    }

    /// Unmap a 4 KiB page. Returns the old entry if mapped at level 0,
    /// or `None` if the address was not mapped.
    pub fn unmap_4k(&mut self, vaddr: VirtAddr) -> Option<PTEntry<A>> {
        let mapping = self.walk_addr(vaddr);

        match mapping {
            Mapping::Level0(entry) => {
                let old = *entry;
                entry.clear();
                Some(old)
            }
            Mapping::Level1(entry) | Mapping::Level2(entry) | Mapping::Level3(entry) => {
                assert!(!entry.present());
                None
            }
        }
    }

    /// Unmap a 2 MiB huge page. Returns the old entry if mapped at level 1,
    /// or `None` if the address was not mapped.
    ///
    /// # Panics
    ///
    /// Panics if `vaddr` is not 2 MiB-aligned.
    pub fn unmap_2m(&mut self, vaddr: VirtAddr) -> Option<PTEntry<A>> {
        assert!(vaddr.is_aligned(PAGE_SIZE_2M));

        let mapping = self.walk_addr(vaddr);

        match mapping {
            Mapping::Level0(_) => unreachable!(),
            Mapping::Level1(entry) => {
                let old = *entry;
                entry.clear();
                Some(old)
            }
            Mapping::Level2(entry) | Mapping::Level3(entry) => {
                assert!(!entry.present());
                None
            }
        }
    }

    /// Returns the clean physical address for a mapped `vaddr`, or `None`
    /// if `vaddr` is not mapped at level 0 or level 1.
    pub fn check_mapping(&mut self, vaddr: VirtAddr) -> Option<PhysAddr> {
        match self.walk_addr(vaddr) {
            Mapping::Level0(entry) => Some(entry.address()),
            Mapping::Level1(entry) => Some(entry.address()),
            _ => None,
        }
    }

    /// Translate `vaddr` to a clean physical address including the
    /// intra-page offset. Returns [`PagingError::NotMapped`] if no valid
    /// mapping exists.
    pub fn phys_addr(&mut self, vaddr: VirtAddr) -> Result<PhysAddr, PagingError> {
        let mapping = self.walk_addr(vaddr);

        match mapping {
            Mapping::Level0(entry) => {
                let offset = vaddr.page_offset();
                if !entry.flags().contains(A::PTFlags::PRESENT) {
                    return Err(PagingError::NotMapped);
                }
                Ok(entry.address() + offset)
            }
            Mapping::Level1(entry) => {
                let offset = vaddr.bits() & (PAGE_SIZE_2M - 1);
                if !entry.flags().contains(A::PTFlags::PRESENT)
                    || !entry.flags().contains(A::PTFlags::HUGE)
                {
                    return Err(PagingError::NotMapped);
                }
                Ok(entry.address() + offset)
            }
            Mapping::Level2(_) | Mapping::Level3(_) => Err(PagingError::NotMapped),
        }
    }
}

/// Methods that modify live page table entries and flush the TLB.
impl<A: ArchPagingMeta, P: PagingHandler> GenericPageTable<A, P> {
    /// Split a 2 MiB huge page into 512 × 4 KiB pages.
    ///
    /// Allocates a new PT page via [`PagingHandler::allocate_frame`],
    /// populates it with 4 KiB entries covering the same physical range,
    /// and replaces the original PDE. Flushes the TLB afterwards.
    ///
    /// # Panics
    ///
    /// Panics if `entry` does not have the `HUGE` flag set.
    pub fn do_split_4k(entry: &mut PTEntry<A>) -> Result<(), PagingError> {
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

    /// If the mapping is a 2 MiB huge page, split it into 4 KiB pages.
    /// Level 0 (already 4 KiB) is a no-op; levels 2–3 are unmapped errors.
    fn split_4k(mapping: Mapping<'_, A>) -> Result<(), PagingError> {
        match mapping {
            Mapping::Level0(_entry) => Ok(()),
            Mapping::Level1(entry) => Self::do_split_4k(entry),
            Mapping::Level2(_entry) => Err(PagingError::NotMapped),
            Mapping::Level3(_entry) => Err(PagingError::NotMapped),
        }
    }

    /// Rewrite a PTE to use the shared (plaintext) encryption state.
    fn make_pte_shared(entry: &mut PTEntry<A>) {
        let flags = entry.flags();
        let addr = entry.address();
        // entry.address() returned with c-bit clear already
        entry.set(A::make_shared_address(addr), flags);
    }

    /// Rewrite a PTE to use the private (encrypted) encryption state.
    fn make_pte_private(entry: &mut PTEntry<A>) {
        let flags = entry.flags();
        let addr = entry.address();
        // entry.address() returned with c-bit clear already
        entry.set(A::make_private_address(addr), flags);
    }

    /// Mark a 4 KiB page as shared (plaintext).
    ///
    /// If the page is currently part of a 2 MiB mapping it is split first.
    pub fn set_shared_4k(&mut self, vaddr: VirtAddr) -> Result<(), PagingError> {
        let mapping = self.walk_addr(vaddr);
        Self::split_4k(mapping)?;

        if let Mapping::Level0(entry) = self.walk_addr(vaddr) {
            Self::make_pte_shared(entry);
            Ok(())
        } else {
            Err(PagingError::NotMapped)
        }
    }

    /// Mark a 4 KiB page as private (encrypted).
    ///
    /// If the page is currently part of a 2 MiB mapping it is split first.
    pub fn set_encrypted_4k(&mut self, vaddr: VirtAddr) -> Result<(), PagingError> {
        let mapping = self.walk_addr(vaddr);
        Self::split_4k(mapping)?;

        if let Mapping::Level0(entry) = self.walk_addr(vaddr) {
            Self::make_pte_private(entry);
            Ok(())
        } else {
            Err(PagingError::NotMapped)
        }
    }
}

/// Methods that use the self-map to inspect the *active* page table via
/// the [`SelfMap`] trait.
impl<A: ArchPagingMeta, P: PagingHandler + SelfMap> GenericPageTable<A, P> {
    /// Compute the virtual address of the level-0 PTE that maps `vaddr`
    /// in the self-map region.
    fn get_pte_address(vaddr: VirtAddr) -> VirtAddr {
        P::pte_base() + ((usize::from(vaddr) & 0x0000_FFFF_FFFF_F000) >> 9)
    }

    /// Translate `vaddr` to a [`PageFrame`] by reading the self-map of the
    /// currently active page table.
    ///
    /// Walks the PTE hierarchy top-down through self-map addresses. Returns
    /// `None` if any level is not present. For huge pages the returned
    /// address includes the sub-page offset from `vaddr`.
    ///
    /// The returned address retains the shared bit (via
    /// [`PTEntry::page_frame`]) so callers can distinguish shared from
    /// private mappings.
    pub fn virt_to_frame(vaddr: VirtAddr) -> Option<PageFrame<A>> {
        let pte_addr = Self::get_pte_address(vaddr);
        let pde_addr = Self::get_pte_address(pte_addr);
        let pdpe_addr = Self::get_pte_address(pde_addr);
        let pml4e_addr = Self::get_pte_address(pdpe_addr);

        // SAFETY: Reading PTE hierarchy top-down through self-map addresses.
        // Each level is checked for presence before reading the next.
        let pml4e: PTEntry<A> = unsafe { PTEntry::read_pte(pml4e_addr) };
        if !pml4e.present() {
            return None;
        }

        // SAFETY: The PML4E was checked to be present, so the PDPE exists
        // and can be read safely via the self-map.
        let pdpe: PTEntry<A> = unsafe { PTEntry::read_pte(pdpe_addr) };
        if !pdpe.present() {
            return None;
        }
        if pdpe.huge() {
            let pa = pdpe.page_frame() + (usize::from(vaddr) & 0x3FFF_FFFF);
            return Some(PageFrame::Size1G(pa));
        }

        // SAFETY: The PDPE was checked to be present and not huge,
        // so the PDE exists and can be read safely via the self-map.
        let pde: PTEntry<A> = unsafe { PTEntry::read_pte(pde_addr) };
        if !pde.present() {
            return None;
        }
        if pde.huge() {
            let pa = pde.page_frame() + (usize::from(vaddr) & 0x001F_FFFF);
            return Some(PageFrame::Size2M(pa));
        }

        // SAFETY: The PDE was checked to be present and not huge,
        // so the PTE exists and can be read safely via the self-map.
        let pte: PTEntry<A> = unsafe { PTEntry::read_pte(pte_addr) };
        if pte.present() {
            let pa = pte.page_frame() + (usize::from(vaddr) & 0xFFF);
            Some(PageFrame::Size4K(pa))
        } else {
            None
        }
    }
}
