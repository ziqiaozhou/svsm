use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

use zerocopy::FromZeros;

use crate::address::{Address, PhysAddr, VirtAddr};
use crate::pagetable::{
    GenericPageTable, Mapping, PTE_SHIFT, PTEntry, PTPage, PageFrame, virt_from_lvl_idx,
};
use crate::sizes::{PAGE_SHIFT, PAGE_SIZE_2M};
use crate::tlb::{MayNeedFlush, TlbOps};
use crate::traits::{
    ArchPagingMeta, GenericPageTableFlags, NextLevel, PageLevel, PagingError, PagingHandler,
    PagingLevel, PagingLevel3, SelfMap,
};

impl<A: ArchPagingMeta + TlbOps, P: PagingHandler> PTPage<A, P> {
    fn from_entry_raw(entry: PTEntry<A>) -> Option<*mut Self> {
        if !entry.present() || entry.huge() {
            return None;
        }

        let address = P::paddr_to_vaddr(entry.address());
        // SAFETY: Every PTEntry points to a previously allocated page-table
        // page, so this pointer dereference is safe.
        Some(address.as_mut_ptr())
    }

    /// `*mut` pointer to the entry at `idx` within `page`.
    fn entry_ptr(page: *const Self, idx: usize) -> *const PTEntry<A> {
        (page as *const PTEntry<A>).wrapping_add(idx)
    }
}

/// The resolved result of a page-table walk on an *active* (installed)
/// page table: the level at which the walk terminated, together with a
/// raw pointer to the page-table entry found there.
///
/// `ActiveMapping` must be updated and accessed via volatile ptr operations.
///
/// The ActiveMapping is an abstract type that is only created
/// during a mutable page-table walk on an active page table.
#[derive(Clone, Copy, Debug)]
pub struct ActiveMapping<'a, P: ArchPagingMeta> {
    level: PageLevel,
    entry: *mut PTEntry<P>,
    _lifetime: PhantomData<&'a PTEntry<P>>,
}

/// An immutable version of `ActiveMapping`.
/// This type is used to represent a mapping from an readable active page table.
#[derive(Clone, Copy, Debug)]
struct ImmutableActiveMapping<'a, P: ArchPagingMeta> {
    level: PageLevel,
    entry: *const PTEntry<P>,
    _lifetime: PhantomData<&'a PTEntry<P>>,
}

impl<'a, A: ArchPagingMeta> From<Mapping<'a, A>> for ActiveMapping<'a, A> {
    fn from(mapping: Mapping<'a, A>) -> Self {
        Self {
            level: mapping.level,
            entry: mapping.entry as _,
            _lifetime: PhantomData,
        }
    }
}

impl<A: ArchPagingMeta + TlbOps> ImmutableActiveMapping<'_, A> {
    pub fn read(self) -> PTEntry<A> {
        // SAFETY: A valid ImmutableActiveMapping must point to a valid PTEntry.
        // See more in the documentation for `ImmutableActiveMapping`.
        unsafe { PTEntry::read_pte(self.entry) }
    }

    pub fn get_level_entry(self) -> (PageLevel, PTEntry<A>) {
        (self.level, self.read())
    }
}

impl<A: ArchPagingMeta + TlbOps> ActiveMapping<'_, A> {
    /// Volatile read of the entry this mapping points at.
    pub fn read(self) -> PTEntry<A> {
        // SAFETY: A valid ActiveMapping must point to a valid PTEntry.
        // See more in the documentation for `ActiveMapping`.
        unsafe { PTEntry::read_pte(self.entry) }
    }

    /// Create a new `ActiveMapping` with the given level and entry.
    pub fn update(self, entry: PTEntry<A>) -> MayNeedFlush<A> {
        // SAFETY: A valid ActiveMapping must point to a valid PTEntry.
        // See more in the documentation for `ActiveMapping`.
        unsafe {
            PTEntry::write_pte(self.entry, entry);
        }
        MayNeedFlush(self.level, PhantomData)
    }

    /// The level at which the walk that produced this mapping terminated.
    pub fn level(self) -> PageLevel {
        self.level
    }
}

/// ActivePageTable is a wrapper around an active page table.
/// It uses UnsafeCell to disable the Rust's assumption of aliasing, so that compiler
/// will not optimize away the reads and writes to the page table entries.
/// Since page table might be updated by hardware, we cannot use UnsafeCell::get_mut().
/// Accessed to the entries must be done through volatile or even atomic reads and writes.
#[repr(transparent)]
#[derive(Debug, FromZeros)]
pub struct ActivePageTable<A: ArchPagingMeta + TlbOps, P: PagingHandler, L: PagingLevel> {
    // store the inactive table here to prevent it from being used as inactive.
    /// The root page table.
    root: core::cell::UnsafeCell<PTPage<A, P>>,
    _level: PhantomData<(A, P, L)>,
}

// SAFETY: the node identifies an installed page table by the address of its
// backing page and accesses entries only through volatile reads/writes, which
// are valid from any CPU. Shared (`&`) access performs no mutation, so the node
// is safe to share across threads; exclusive mutation is serialized through
// `&mut` (and the locks that hold the node).
unsafe impl<A: ArchPagingMeta + TlbOps, P: PagingHandler, L: PagingLevel> Sync
    for ActivePageTable<A, P, L>
{
}

impl<A: ArchPagingMeta + TlbOps, P: PagingHandler, L: PagingLevel> ActivePageTable<A, P, L> {
    /// Wrap the to-be-installed page table.
    /// Transition it to a may-be active page table.
    /// # Safety
    /// The caller must guarantee that the root is a valid, page-aligned pointer
    /// to a root page-table page that remains valid for the lifetime of the implementor.
    pub fn new(inactive: GenericPageTable<A, P, L>) -> Self {
        Self {
            root: core::cell::UnsafeCell::new(inactive.root),
            _level: PhantomData,
        }
    }

    pub fn index<const LVL: usize>(vaddr: VirtAddr) -> usize {
        GenericPageTable::<A, P, L>::index::<LVL>(vaddr)
    }

    /// Get a mutable view of the active page table.
    pub fn get_view(&mut self) -> ActivePageTableView<'_, A, P, L> {
        ActivePageTableView::new(self as _)
    }

    fn root_page(&self) -> *const PTPage<A, P> {
        self.root.get()
    }

    pub fn root_pa(&self) -> PhysAddr {
        P::vaddr_to_paddr(self.root_page().into())
    }

    /// Borrow the wrapped table as an inactive table.
    ///
    /// # Safety
    /// The caller must guarantee the table is not installed, i.e. not
    /// reachable by the MMU on any CPU, for the duration of the borrow.
    /// Inactive access forms `&mut` references into the table and would be
    /// unsound on a live table.
    pub unsafe fn as_inactive(&mut self) -> &mut GenericPageTable<A, P, L> {
        // SAFETY: `GenericPageTable` and `ActivePageTableNode` share the same
        // backing root page layout, and the caller guarantees the table is not
        // installed, so forming a `&mut` into it is sound.
        unsafe { &mut *(self.root_page() as *mut GenericPageTable<A, P, L>) }
    }

    /// Get the next level page table view at the given index.
    /// Returns None if the entry is not present or is a huge page.
    pub fn next_table(&self, idx: usize) -> Option<ActivePageTableView<'_, A, P, L::Next>>
    where
        L: NextLevel,
    {
        let mapping = ImmutableActiveMapping {
            level: L::TOP_LEVEL,
            entry: PTPage::entry_ptr(self.root_page(), idx),
            _lifetime: PhantomData,
        };
        let entry = mapping.read();
        if !entry.present() || entry.huge() {
            return None;
        }
        Some(ActivePageTableView {
            ptr: P::paddr_to_vaddr(entry.address()).as_mut_ptr(),
            _level: PhantomData,
        })
    }

    /// Walk the installed table for `vaddr`, returning a mutable
    /// [`ActiveMapping`] (level and raw entry pointer) at which the walk
    /// terminated.
    /// This is the only entry to get an [`ActiveMapping`] from an active page table.
    pub fn walk_mut(&mut self, vaddr: VirtAddr) -> ActiveMapping<'_, A> {
        let tbl_view = ActivePageTableView::<A, P, L> {
            ptr: self,
            _level: PhantomData,
        };
        let m = tbl_view.walk(vaddr);
        ActiveMapping {
            level: m.level,
            entry: m.entry as *mut PTEntry<A>,
            _lifetime: PhantomData,
        }
    }

    /// Mutable view of the top-level entry at `idx`.
    ///
    /// Takes `&mut self` even though it only reads the root pointer: the
    /// returned [`ActiveMapping`] can write the entry, so it must be tied to an
    /// exclusive borrow of the table.
    #[allow(clippy::needless_pass_by_ref_mut)]
    fn top_entry_mut(&mut self, idx: usize) -> ActiveMapping<'_, A> {
        ActiveMapping {
            level: L::TOP_LEVEL,
            entry: PTPage::entry_ptr(self.root_page(), idx) as *mut PTEntry<A>,
            _lifetime: PhantomData,
        }
    }

    /// Walk the installed table for `vaddr`, returning the level at which the
    /// walk stopped and a raw pointer to the entry there.
    ///
    /// The walk descends through present, non-huge entries and stops at the
    /// leaf (`Level0`), at the first absent entry, or at a huge entry.
    fn walk<'a>(&'a self, vaddr: VirtAddr) -> ImmutableActiveMapping<'a, A> {
        let page = self.root_page();
        let level = L::TOP_LEVEL;
        let idx = (vaddr.as_usize() >> (PAGE_SHIFT + level as usize * PTE_SHIFT)) & 0x1FF;
        let mut mapping = ImmutableActiveMapping {
            level,
            entry: PTPage::entry_ptr(page, idx),
            _lifetime: PhantomData,
        };
        for _ in 0..L::TOP_LEVEL as usize {
            match PTPage::<A, P>::from_entry_raw(mapping.read()) {
                Some(next) => {
                    let level = mapping
                        .level
                        .next_down()
                        .expect("non-leaf level has a child");
                    mapping = ImmutableActiveMapping {
                        level,
                        entry: PTPage::entry_ptr(next, idx),
                        _lifetime: PhantomData,
                    };
                }
                None => {
                    return mapping;
                }
            }
        }
        mapping
    }

    /// Walk the installed table for `vaddr` to set up a read-modify-write of
    /// the terminal entry.
    ///
    /// Copies the live terminal entry into the caller-owned `entry` staging
    /// buffer and returns two handles to it:
    /// - a [`Mapping`] over the staged copy, used to edit the entry in place
    ///   (e.g. via the inactive `do_*` helpers) without touching the live table;
    /// - an [`ActiveMapping`] commit handle, used to volatile-write the edited
    ///   copy back to the live entry once.
    ///
    /// Takes `&mut self` even though the walk only reads: the returned
    /// [`ActiveMapping`] can write the entry, so it must be tied to an
    /// exclusive borrow of the table.
    fn walk_for_update<'a, 'b>(
        &'a mut self,
        vaddr: VirtAddr,
        entry: &'b mut PTEntry<A>,
    ) -> (Mapping<'b, A>, ActiveMapping<'a, A>) {
        let m = self.walk_mut(vaddr);
        *entry = m.read();

        let level = m.level;
        (Mapping { level, entry }, m)
    }

    /// Maps a 4KB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address to map.
    /// - `paddr`: The physical address to map to.
    /// - `flags`: The flags to apply to the mapping.
    /// - `shared`: Indicates whether the mapping is shared.
    /// - `parent_flags`: The flags to apply to the allocated parent page table entries.
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
        let mut entry = PTEntry::empty();
        let (mapping, active_m) = self.walk_for_update(vaddr, &mut entry);
        GenericPageTable::<A, P, L>::do_map_4k_with_parent_flags(
            mapping,
            vaddr,
            paddr,
            flags,
            shared,
            parent_flags,
        )?;
        active_m
            .update(entry)
            .ignore("non-present -> present does not need TLB flush");
        Ok(())
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
        parent_flags: A::PTFlags,
    ) -> Result<(), PagingError> {
        let mut entry = PTEntry::empty();
        let (mapping, active_m) = self.walk_for_update(vaddr, &mut entry);
        GenericPageTable::<A, P, L>::do_map_2m_with_parent_flags(
            mapping,
            vaddr,
            paddr,
            flags,
            shared,
            parent_flags,
        )?;
        active_m
            .update(entry)
            .ignore("non-present -> present does not need TLB flush");
        Ok(())
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

    /// Unmaps a 4KB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address of the mapping to unmap.
    ///
    /// # Returns
    /// A copy of the entry that mapped the address before it was cleared (if a
    /// leaf was present) together with the TLB-flush obligation for the
    /// now-stale translation.
    pub fn unmap_4k(&mut self, vaddr: VirtAddr) -> Option<(PTEntry<A>, MayNeedFlush<A>)> {
        let mut entry = PTEntry::empty();
        let (m, active_m) = self.walk_for_update(vaddr, &mut entry);
        GenericPageTable::<A, P, L>::do_unmap_4k(m).map(|e| (e, active_m.update(entry)))
    }

    /// Unmaps a 2MB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address of the mapping to unmap.
    ///
    /// # Returns
    /// A copy of the entry that mapped the address before it was cleared (if a
    /// huge leaf was present) together with the TLB-flush obligation for the
    /// now-stale translation.
    ///
    /// # Panics
    /// Panics if `vaddr` is not aligned to a 2MB boundary.
    pub fn unmap_2m(&mut self, vaddr: VirtAddr) -> Option<(PTEntry<A>, MayNeedFlush<A>)> {
        let mut entry = PTEntry::empty();
        let (m, active_m) = self.walk_for_update(vaddr, &mut entry);
        GenericPageTable::<A, P, L>::do_unmap_2m(m).map(|e| (e, active_m.update(entry)))
    }

    /// Sets the shared state for a 4KB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address of the page.
    ///
    /// # Returns
    /// A result indicating success or an error [`PagingError`] if the
    /// operation fails.
    pub fn set_shared_4k(&mut self, vaddr: VirtAddr) -> Result<MayNeedFlush<A>, PagingError> {
        let mut entry = PTEntry::empty();
        let (m, active_m) = self.walk_for_update(vaddr, &mut entry);
        GenericPageTable::<A, P, L>::do_set_shared_4k(m, vaddr)?;
        Ok(active_m.update(entry))
    }

    /// Sets the encryption state for a 4KB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address of the page.
    ///
    /// # Returns
    /// A result indicating success or an error [`PagingError`].
    pub fn set_encrypted_4k(&mut self, vaddr: VirtAddr) -> Result<MayNeedFlush<A>, PagingError> {
        let mut entry = PTEntry::empty();
        let (m, active_m) = self.walk_for_update(vaddr, &mut entry);
        GenericPageTable::<A, P, L>::do_set_encrypted_4k(m, vaddr)?;
        Ok(active_m.update(entry))
    }

    /// Retrieves the physical address of a mapping.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address to query.
    ///
    /// # Returns
    /// The physical address of the mapping if present; otherwise, an error
    /// ([`PagingError`]).
    pub fn phys_addr(&self, vaddr: VirtAddr) -> Result<PhysAddr, PagingError> {
        let (level, entry) = self.walk(vaddr).get_level_entry();
        match level {
            PageLevel::Level0 => {
                let offset = vaddr.page_offset();
                if !entry.present() {
                    return Err(PagingError::NotMapped);
                }
                Ok(entry.address() + offset)
            }
            PageLevel::Level1 => {
                let offset = vaddr.bits() & (PAGE_SIZE_2M - 1);
                if !entry.present() || !entry.huge() {
                    return Err(PagingError::NotMapped);
                }
                Ok(entry.address() + offset)
            }
            _ => Err(PagingError::NotMapped),
        }
    }

    pub fn populate(&mut self, idx: usize, child: &ActivePageTable<A, P, L::Next>) -> bool 
    where L: NextLevel
    {
        let desired = PTEntry::new(
            A::make_private_address(child.root_pa()),
            A::PTFlags::parent_flags(),
        );
        let mapping = self.top_entry_mut(idx);
        let old = mapping.read();
        if old.raw() == desired.raw() {
            return false;
        }
        assert!(old.is_clear() || !old.present());
        mapping.update(desired).ignore("TLB flush is not needed since the caller assumes the old entry is empty or non-present");
        true
    }
}

impl<A: ArchPagingMeta + TlbOps, P: PagingHandler + SelfMap> ActivePageTable<A, P, PagingLevel3> {
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
    /// This reads page-table entries through the self-map, which only resolves
    /// to *this* table's entries while the table is the one installed in the
    /// MMU. It is therefore an active-table operation.
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
        let pml4e = unsafe { PTEntry::<A>::read_pte(pml4e_addr.as_ptr()) };
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
        let pdpe = unsafe { PTEntry::<A>::read_pte(pdpe_addr.as_ptr()) };
        if !pdpe.present() {
            return None;
        }
        if pdpe.huge() {
            let pa = pdpe.page_frame() + (usize::from(vaddr) & 0x3FFF_FFFF);
            return Some(PageFrame::Size1G(pa, PhantomData));
        }

        // SAFETY: The PDPE was checked to be present and not to be a huge
        // page. So the PDE exists and can be read safely.
        let pde = unsafe { PTEntry::<A>::read_pte(pde_addr.as_ptr()) };
        if !pde.present() {
            return None;
        }
        if pde.huge() {
            let pa = pde.page_frame() + (usize::from(vaddr) & 0x001F_FFFF);
            return Some(PageFrame::Size2M(pa));
        }

        // SAFETY: The PDE was checked to be present and not to be a huge
        // page. So the PTE exists and can be read safely.
        let pte = unsafe { PTEntry::<A>::read_pte(pte_addr.as_ptr()) };
        if pte.present() {
            let pa = pte.page_frame() + (usize::from(vaddr) & 0xFFF);
            Some(PageFrame::Size4K(pa))
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct ActivePageTableView<'a, A: ArchPagingMeta + TlbOps, P: PagingHandler, L: PagingLevel> {
    ptr: *mut ActivePageTable<A, P, L>,
    _level: PhantomData<(&'a (), A, P, L)>,
}

impl<A: ArchPagingMeta + TlbOps, P: PagingHandler, L: PagingLevel> Deref
    for ActivePageTableView<'_, A, P, L>
{
    type Target = ActivePageTable<A, P, L>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: `ptr` was constructed from a valid node reference or an
        // installed root page and stays valid for the view's lifetime `'a`.
        unsafe { self.ptr.as_ref().unwrap() }
    }
}

impl<A: ArchPagingMeta + TlbOps, P: PagingHandler, L: PagingLevel> DerefMut
    for ActivePageTableView<'_, A, P, L>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: see `deref`; the exclusive borrow of the view grants
        // exclusive access to the pointed-to node.
        unsafe { self.ptr.as_mut().unwrap() }
    }
}

impl<A: ArchPagingMeta + TlbOps, P: PagingHandler, L: PagingLevel>
    ActivePageTableView<'_, A, P, L>
{
    pub(crate) fn new(ptr: *mut ActivePageTable<A, P, L>) -> Self {
        Self {
            ptr,
            _level: PhantomData,
        }
    }
}
