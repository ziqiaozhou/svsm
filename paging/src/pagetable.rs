// SPDX-License-Identifier: MIT OR Apache-2.0

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
pub use crate::traits::{
    ArchPagingMeta, GenericPageTableFlags, PageLevel, PagingError, PagingHandler, PagingLevel,
    PagingLevel2, PagingLevel3, SelfMap,
};
use bitflags::Flags;
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use zerocopy::FromZeros;

/// Number of virtual-address bits indexed by a single page-table level.
/// Assume 8-byte page table entries and page granularity = 1 << PAGE_SHIFT
pub(crate) const PTE_SHIFT: usize = PAGE_SHIFT - 3;

/// Number of entries in a page table (4KB/8B).
pub(crate) const ENTRY_COUNT: usize = 1 << PTE_SHIFT;

pub(crate) const fn virt_from_lvl_idx(idx: usize, level: PageLevel) -> VirtAddr {
    VirtAddr::new(idx << ((level as usize * PTE_SHIFT) + PAGE_SHIFT))
}

const _: () = assert!(
    core::mem::size_of::<PhysAddr>() == 8,
    "Only supports 8 bytes PTE entry",
);

/// Represents a page table entry.
#[repr(C)]
#[derive(Copy, Clone, Debug, FromZeros)]
pub struct PTEntry<A: ArchPagingMeta> {
    entry: PhysAddr,
    _phantom: PhantomData<A>,
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
        self.flags().present()
    }

    /// Check if the page table entry is huge.
    pub fn huge(&self) -> bool {
        self.flags().huge()
    }

    /// Check if the page table entry is user-accessible.
    pub fn user(&self) -> bool {
        self.flags().user()
    }

    /// Get the raw bits (`usize`) of the page table entry.
    pub fn raw(&self) -> usize {
        self.entry.bits()
    }

    /// Get the flags of the page table entry.
    pub fn flags(&self) -> A::PTFlags {
        A::PTFlags::from_bits_truncate(self.entry.bits())
    }

    pub fn new(addr: PhysAddr, flags: A::PTFlags) -> Self {
        let addr = addr.bits();
        assert_eq!(addr & !A::address_mask(), 0);
        Self {
            entry: PhysAddr::from(addr | flags.bits()),
            _phantom: PhantomData,
        }
    }

    pub fn empty() -> Self {
        Self::new(PhysAddr::null(), A::PTFlags::empty())
    }

    /// Set the page table entry with the specified address, with flags
    /// constrained to the supported feature flags.
    pub fn set(&mut self, addr: PhysAddr, flags: A::PTFlags) {
        *self = Self::new(addr, flags);
    }

    fn make_pte_shared(&mut self) {
        let flags = self.flags();
        let addr = self.address();

        // entry.address() returned with c-bit clear already
        self.set(A::make_shared_address(addr), flags);
    }

    fn make_pte_private(&mut self) {
        let flags = self.flags();
        let addr = self.address();

        // entry.address() returned with c-bit clear already
        self.set(A::make_private_address(addr), flags);
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
    pub unsafe fn read_pte(entry: *const Self) -> Self {
        // SAFETY: the caller guarantees `entry` is valid. A volatile read
        // tolerates concurrent MMU writes to the accessed/dirty bits without
        // forming a reference to the frame.
        unsafe { entry.read_volatile() }
    }

    /// Volatile write of `e` through a raw entry pointer.
    ///
    /// # Safety
    /// `entry` must be a valid, aligned pointer to a live [`PTEntry`]
    /// (e.g. obtained from [`PTPage::entry_ptr`]).
    pub unsafe fn write_pte(entry: *mut Self, e: Self) {
        // SAFETY: as in `read_pte`.
        unsafe { entry.write_volatile(e) }
    }
}

/// A pagetable page with multiple entries.
#[repr(C)]
#[derive(Debug, FromZeros)]
pub struct PTPage<A: ArchPagingMeta, P: PagingHandler> {
    entries: [PTEntry<A>; ENTRY_COUNT],
    _phantom: PhantomData<P>,
}

impl<A: ArchPagingMeta, P: PagingHandler> PTPage<A, P> {
    /// Allocates a zeroed pagetable page and returns a mutable reference to
    /// it, plus its physical address.
    ///
    /// # Errors
    ///
    /// Returns [`PagingError`] if the page cannot be allocated.
    fn alloc() -> Result<(&'static mut PTPage<A, P>, PhysAddr), PagingError> {
        let paddr = P::allocate_physical_page()?;
        let vaddr = P::paddr_to_vaddr(paddr);
        // SAFETY: allocate_physical_page returns a unique, zeroed frame and
        // paddr_to_vaddr returns a valid virtual mapping for it.
        let page = unsafe { Self::from_vaddr(vaddr) };
        Ok((page, paddr))
    }

    /// Converts a pagetable entry to a mutable reference to a [`PTPage`],
    /// if the entry is present and not huge.
    /// # Safety
    /// The caller must ensure that the entry is valid and points to a valid page table.
    unsafe fn from_entry(entry: &PTEntry<A>) -> Option<&'_ Self> {
        Self::from_entry_raw(entry).map(|ptr| unsafe { &*ptr })
    }

    /// Generates a `PTPage` from a virtual address.
    /// # Safety
    /// The caller must ensure that the virtual address is a valid page table.
    pub unsafe fn from_vaddr(vaddr: VirtAddr) -> &'static mut Self {
        // SAFETY: the caller guarantees the correctness of the virtual
        // address.
        unsafe { &mut *vaddr.as_mut_ptr::<Self>() }
    }

    pub(crate) fn from_entry_raw(entry: &PTEntry<A>) -> Option<*mut Self> {
        if !entry.present() || entry.huge() {
            return None;
        }

        let address = P::paddr_to_vaddr(entry.address());
        // SAFETY: Every PTEntry points to a previously allocated page-table
        // page, so this pointer dereference is safe.
        Some(address.as_mut_ptr())
    }

    /// `*mut` pointer to the entry at `idx` within `page`.
    pub(crate) fn entry_ptr(page: *const Self, idx: usize) -> *const PTEntry<A> {
        (page as *const PTEntry<A>).wrapping_add(idx)
    }

    /// Recursively free all children page table pages starting from the level.
    ///
    /// # Safety
    /// The caller must ensure that children tables are not in use by any other thread.
    pub(crate) unsafe fn free_lvl(&self, level: PageLevel) {
        if level <= PageLevel::Level0 {
            return;
        }
        for entry in self.entries.iter() {
            // SAFETY: `entry` belongs to this inactive table which, per the
            // `free_lvl` contract, is not in use by any other thread, so the
            // child page it points to is valid and uniquely owned.
            if let Some(child) = unsafe { Self::from_entry(entry) } {
                if let Some(next) = level.next_down() {
                    // SAFETY: The child page table is not in use by any other thread.
                    unsafe { child.free_lvl(next) };
                }
                let paddr = entry.address();
                // SAFETY: the page was allocated via PagingHandler::allocate_physical_page.
                unsafe { P::deallocate_physical_page(paddr) };
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

/// Mapping of an inactive page table entry at a specific level.
#[derive(Debug)]
pub(crate) struct Mapping<'a, A: ArchPagingMeta> {
    pub level: PageLevel,
    pub entry: &'a mut PTEntry<A>,
}

impl<'a, A: ArchPagingMeta> Mapping<'a, A> {
    /// Construct a `Mapping` at the given level.
    fn new(level: PageLevel, entry: &'a mut PTEntry<A>) -> Self {
        Self { level, entry }
    }
}

/// A physical address within a page frame
#[derive(Clone, Copy, Debug)]
pub enum PageFrame<A: ArchPagingMeta> {
    Size4K(PhysAddr),
    Size2M(PhysAddr),
    Size1G(PhysAddr, PhantomData<A>),
}

impl<A: ArchPagingMeta> PageFrame<A> {
    /// Get the address from the page frame, including the shared bit.
    pub fn page_frame(&self) -> PhysAddr {
        let paddr = match *self {
            Self::Size4K(pa) => pa,
            Self::Size2M(pa) => pa,
            Self::Size1G(pa, _) => pa,
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
            Self::Size1G(_, _) => PAGE_SIZE_1G,
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

/// Methods that use the self-map to inspect the *active* top-level page table.
///
/// Self-mapped address calculations assume a PML4-rooted table.
impl<A: ArchPagingMeta, P: PagingHandler + SelfMap> PTPage<A, P> {
    /// Install the self-map entry in a freshly zeroed page table.
    ///
    /// The self-map PML4 entry is written at [`SelfMap::SELFMAP_IDX`] and points
    /// at this page table's own root page.
    pub fn init_self_map(&mut self) {
        let paddr = P::vaddr_to_paddr(core::ptr::addr_of!(self).into());
        let flags = A::PTFlags::self_map_table_flags();
        self[P::SELFMAP_IDX].set(A::make_private_address(paddr), flags);
    }
}

impl<A: ArchPagingMeta, P: PagingHandler> PTPage<A, P> {
    /// Allocate from level 3 (PML4E) down to the target level.
    fn alloc_pte_lvl3(
        entry: &mut PTEntry<A>,
        vaddr: VirtAddr,
        size: PageSize,
        parent_flags: A::PTFlags,
    ) -> Mapping<'_, A> {
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
    fn alloc_pte_lvl2(
        entry: &mut PTEntry<A>,
        vaddr: VirtAddr,
        size: PageSize,
        parent_flags: A::PTFlags,
    ) -> Mapping<'_, A> {
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
    fn alloc_pte_lvl1(
        entry: &mut PTEntry<A>,
        vaddr: VirtAddr,
        size: PageSize,
        parent_flags: A::PTFlags,
    ) -> Mapping<'_, A> {
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
    fn do_alloc_pte_4k(
        map: Mapping<'_, A>,
        vaddr: VirtAddr,
        parent_flags: A::PTFlags,
    ) -> Mapping<'_, A> {
        match map.level {
            PageLevel::Level0 => map,
            PageLevel::Level1 => {
                Self::alloc_pte_lvl1(map.entry, vaddr, PageSize::Regular, parent_flags)
            }
            PageLevel::Level2 => {
                Self::alloc_pte_lvl2(map.entry, vaddr, PageSize::Regular, parent_flags)
            }
            PageLevel::Level3 => {
                Self::alloc_pte_lvl3(map.entry, vaddr, PageSize::Regular, parent_flags)
            }
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
    fn do_alloc_pte_2m(
        map: Mapping<'_, A>,
        vaddr: VirtAddr,
        parent_flags: A::PTFlags,
    ) -> Mapping<'_, A> {
        match map.level {
            PageLevel::Level0 | PageLevel::Level1 => map,
            PageLevel::Level2 => {
                Self::alloc_pte_lvl2(map.entry, vaddr, PageSize::Huge, parent_flags)
            }
            PageLevel::Level3 => {
                Self::alloc_pte_lvl3(map.entry, vaddr, PageSize::Huge, parent_flags)
            }
        }
    }

    /// Splits a 2MB page into 4KB pages.
    ///
    /// # Parameters
    /// - `entry`: The 2M page table entry to split.
    ///
    /// # Returns
    /// A result containing the newly allocated page table page, or an error
    /// [`PagingError`] in failure.
    fn do_split_4k(entry: &mut PTEntry<A>) -> Result<&'static mut PTPage<A, P>, PagingError> {
        let (page, paddr) = PTPage::<A, P>::alloc()?;
        let mut flags = entry.flags();

        assert!(flags.huge());

        let addr_2m = PhysAddr::from(entry.address().bits() & 0x000f_ffff_fff0_0000);

        flags.remove(A::PTFlags::HUGE);

        // Prepare PTE leaf page
        for (i, e) in page.entries.iter_mut().enumerate() {
            let addr_4k = addr_2m + (i * PAGE_SIZE);
            e.clear();
            e.set(A::make_private_address(addr_4k), flags);
        }

        entry.set(A::make_private_address(paddr), flags);

        Ok(page)
    }

    /// Splits a page into 4KB pages for a specific virtual address.
    ///
    /// # Parameters
    /// - `mapping`: The mapping to split.
    /// - `vaddr`: The virtual address for which to split the page.
    ///
    /// # Returns
    /// A result containing the updated mapping for the virtual address, or an error
    /// [`PagingError`] in failure.
    fn split_4k(
        mapping: Mapping<'_, A>,
        vaddr: VirtAddr,
    ) -> Result<Mapping<'_, A>, PagingError> {
        match mapping.level {
            PageLevel::Level0 => Ok(mapping),
            PageLevel::Level1 => {
                let next = Self::do_split_4k(mapping.entry)?;
                let idx = vaddr.to_pgtbl_idx::<0>();
                Ok(Mapping::new(PageLevel::Level0, &mut next.entries[idx]))
            }
            _ => Err(PagingError::NotMapped),
        }
    }

    /// Sets the shared state for a 4KB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address of the page.
    ///
    /// # Returns
    /// A result indicating success or an error [`PagingError`] if the
    /// operation fails.
    pub(crate) fn do_set_shared_4k(
        mapping: Mapping<'_, A>,
        vaddr: VirtAddr,
    ) -> Result<(), PagingError> {
        if let Mapping {
            level: PageLevel::Level0,
            entry,
        } = Self::split_4k(mapping, vaddr)?
        {
            entry.make_pte_shared();
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
    /// A result indicating success or an error [`PagingError`] if the
    /// operation fails.
    pub(crate) fn do_set_encrypted_4k(
        mapping: Mapping<'_, A>,
        vaddr: VirtAddr,
    ) -> Result<(), PagingError> {
        if let Mapping {
            level: PageLevel::Level0,
            entry,
        } = Self::split_4k(mapping, vaddr)?
        {
            entry.make_pte_private();
            Ok(())
        } else {
            Err(PagingError::NotMapped)
        }
    }

    pub(crate) fn do_map_2m_with_parent_flags(
        old_mapping: Mapping<'_, A>,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
        parent_flags: A::PTFlags,
    ) -> Result<(), PagingError> {
        assert!(vaddr.is_aligned(PAGE_SIZE_2M));
        assert!(paddr.is_aligned(PAGE_SIZE_2M));
        let mapping = Self::do_alloc_pte_2m(old_mapping, vaddr, parent_flags);
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

    pub(crate) fn do_unmap_2m(mapping: Mapping<'_, A>) -> Option<PTEntry<A>> {
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

    pub(crate) fn do_map_4k_with_parent_flags(
        old_mapping: Mapping<'_, A>,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
        parent_flags: A::PTFlags,
    ) -> Result<(), PagingError> {
        assert!(vaddr.is_aligned(PAGE_SIZE));
        assert!(paddr.is_aligned(PAGE_SIZE));
        let mapping = Self::do_alloc_pte_4k(old_mapping, vaddr, parent_flags);
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

    pub(crate) fn do_unmap_4k(mapping: Mapping<'_, A>) -> Option<PTEntry<A>> {
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
}
