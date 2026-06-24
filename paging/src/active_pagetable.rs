use core::marker::PhantomData;

use crate::address::{PhysAddr, VirtAddr};
use crate::pagetable::{GenericPageTable, Mapping, PTEntry, PTPage};
use crate::tlb::{MayNeedFlush, TlbOps};
use crate::traits::{
    ArchPagingMeta, GenericPageTableFlags, PageLevel, PagingError, PagingHandler, PagingLevel,
};

#[cfg(feature = "ignore_ad")]
unsafe fn ptr_read<T: Copy>(ptr: *const T) -> T {
    // SAFETY: the caller guarantees `ptr` is a valid, aligned pointer.
    unsafe { *ptr }
}

#[cfg(not(feature = "ignore_ad"))]
unsafe fn ptr_volatile_read<T>(ptr: *const T) -> T {
    // SAFETY: the caller guarantees `ptr` is a valid, aligned pointer.
    unsafe { ptr.read_volatile() }
}

/// Volatile read of a page-table entry through a raw entry pointer.
///
/// The entry pointer is the granularity at which a future Verus
/// `PointsTo<PTEntry>` permission would be attached.
///
/// # Safety
/// `entry` must be a valid, aligned pointer to a live [`PTEntry`]
/// (e.g. obtained from [`PTPage::entry_ptr`]).
unsafe fn read_entry<A: ArchPagingMeta>(entry: *const PTEntry<A>) -> PTEntry<A> {
    // SAFETY: the caller guarantees `entry` is valid. A volatile read
    // tolerates concurrent MMU writes to the accessed/dirty bits without
    // forming a reference to the frame.
    #[cfg(feature = "ignore_ad")]
    unsafe {
        ptr_read(entry)
    }
    #[cfg(not(feature = "ignore_ad"))]
    // SAFETY: as above; the caller guarantees `entry` is valid.
    unsafe {
        ptr_volatile_read(entry)
    }
}

/// Volatile write of `e` through a raw entry pointer.
///
/// # Safety
/// `entry` must be a valid, aligned pointer to a live [`PTEntry`]
/// (e.g. obtained from [`PTPage::entry_ptr`]).
unsafe fn write_entry<A: ArchPagingMeta>(entry: *mut PTEntry<A>, e: PTEntry<A>) {
    // SAFETY: as in `read_entry`.
    unsafe { entry.write_volatile(e) }
}

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
    fn entry_ptr(page: *mut Self, idx: usize) -> *mut PTEntry<A> {
        (page as *mut PTEntry<A>).wrapping_add(idx)
    }

    /// Volatile read of the entry at `idx` within the (possibly live) page
    /// `page`.
    ///
    /// # Safety
    /// `page` must point to a valid [`PTPage`] and `idx < ENTRY_COUNT`.
    unsafe fn read_entry_at(page: *const Self, idx: usize) -> PTEntry<A> {
        // SAFETY: delegated to the caller; the volatile read forms no
        // reference into the frame.
        unsafe { read_entry((page as *const PTEntry<A>).wrapping_add(idx)) }
    }
}

/// The resolved result of a page-table walk on an *active* (installed)
/// page table: the level at which the walk terminated, together with a
/// *copy* of the page-table entry found there.
///
/// Unlike [`pagetable::Mapping`](crate::pagetable::Mapping) which holds a
/// `&mut PTEntry` borrow, `ActiveMapping` holds a copy because the MMU may
/// concurrently mutate the entry (e.g. setting accessed/dirty bits).
/// Mutations must go through the `map_*`, `unmap_*`, `set_*`, and
/// `split_4k` methods of [`ActivePageTableNode`], which read-modify-write
/// the entry with volatile accesses.
#[must_use = "ActiveMapping must be read/write to be useful"]
#[derive(Clone, Copy, Debug)]
pub struct ActiveMapping<P: ArchPagingMeta> {
    level: PageLevel,
    entry: *mut PTEntry<P>,
}

impl<A: ArchPagingMeta> From<Mapping<'_, A>> for ActiveMapping<A> {
    fn from(mapping: Mapping<'_, A>) -> Self {
        Self {
            level: mapping.level,
            entry: mapping.entry as _,
        }
    }
}

impl<A: ArchPagingMeta + TlbOps> ActiveMapping<A> {
    /// Create a new `ActiveMapping` with the given level and entry.
    pub fn update(self, entry: PTEntry<A>) -> MayNeedFlush<A> {
        // SAFETY: A valid ActiveMapping must point to a valid PTEntry.
        unsafe {
            write_entry(self.entry, entry);
        }
        MayNeedFlush(self.level, PhantomData)
    }

    pub fn read(self) -> PTEntry<A> {
        // SAFETY: A valid ActiveMapping must point to a valid PTEntry.
        unsafe { read_entry(self.entry) }
    }

    /// The level at which the walk that produced this mapping terminated.
    pub fn level(self) -> PageLevel {
        self.level
    }
}

/// The root of an *active* (installed) page table.
///
/// Provides a raw pointer to the root [`PTPage`]. The pointer is `*mut`
/// because an active table is mutated in place through volatile writes; it
/// must never be turned into a `&PTPage`/`&mut PTPage` reference.
///
/// # Safety
/// `root_ptr` must return a valid, page-aligned pointer to a live root
/// page-table page that remains valid for the lifetime of the implementor.
pub unsafe trait PageTableRoot<A: ArchPagingMeta, P: PagingHandler> {
    fn root_pa(&self) -> PhysAddr;
}

/// An active page table, which is a page table that is currently in use.
/// When a page table is active, we cannot safetyly access it via
/// Rust references, since the MMU can read and write to the page table
/// concurrently.
///
/// Every entry access therefore goes through a raw `*mut PTEntry` and a
/// volatile read/write of a *local* [`PTEntry`] copy; this type never forms
/// a `&PTPage`, `&PTEntry`, or `&mut` reference into the installed table.
#[derive(Debug)]
pub struct ActivePageTableNode<
    R: PageTableRoot<A, P>,
    A: ArchPagingMeta + TlbOps,
    P: PagingHandler,
    L: PagingLevel,
> {
    // store the inactive table here to prevent it from being used as inactive.
    root: R, 
    _level: PhantomData<(A, P, L)>,
}

impl<R: PageTableRoot<A, P>, A: ArchPagingMeta + TlbOps, P: PagingHandler, L: PagingLevel>
    ActivePageTableNode<R, A, P, L>
{
    /// Wrap the to-be-installed page table.
    ///
    /// # Safety
    /// `root` must point to a page-table page that is the root of an
    /// `L::TOP_LEVEL` hierarchy and that remains valid for the lifetime of
    /// the returned node. The caller must also coordinate updates if the
    /// table (or any sub-tree) is shared with other page tables.
    pub unsafe fn new(root: R) -> Self {
        Self {
            root,
            _level: PhantomData,
        }
    }

    /// Borrow the wrapped table as an inactive table.
    ///
    /// # Safety
    /// The caller must guarantee the table (and any of its sub-trees) is not
    /// installed in any active page table, i.e. not reachable by the MMU on
    /// any CPU, for the duration of the borrow. Inactive access forms `&mut`
    /// references into the table and would be unsound on a live table.
    pub unsafe fn as_inactive(&mut self) -> &mut R {
        &mut self.root
    }

    /// Walks an active page table at level 0 to find a mapping.
    fn walk_addr_lvl0(page: *mut PTPage<A, P>, vaddr: VirtAddr) -> ActiveMapping<A> {
        let idx = vaddr.to_pgtbl_idx::<0>();
        ActiveMapping {
            level: PageLevel::Level0,
            entry: PTPage::entry_ptr(page, idx),
        }
    }

    /// Walks an active page table at level 1, descending if the entry refers
    /// to a present, non-huge sub-table.
    fn walk_addr_lvl1(page: *mut PTPage<A, P>, vaddr: VirtAddr) -> ActiveMapping<A> {
        let idx = vaddr.to_pgtbl_idx::<1>();
        // SAFETY: `page`/`idx` address a live installed entry; the volatile
        // read tolerates concurrent MMU updates.
        let value = unsafe { PTPage::<A, P>::read_entry_at(page, idx) };
        match PTPage::<A, P>::from_entry_raw(value) {
            Some(next) => Self::walk_addr_lvl0(next, vaddr),
            None => ActiveMapping {
                level: PageLevel::Level1,
                entry: PTPage::entry_ptr(page, idx),
            },
        }
    }

    /// Walks an active page table at level 2, descending if the entry refers
    /// to a present, non-huge sub-table.
    fn walk_addr_lvl2(page: *mut PTPage<A, P>, vaddr: VirtAddr) -> ActiveMapping<A> {
        let idx = vaddr.to_pgtbl_idx::<2>();
        // SAFETY: `page`/`idx` address a live installed entry; the volatile
        // read tolerates concurrent MMU updates.
        let value = unsafe { PTPage::<A, P>::read_entry_at(page, idx) };
        match PTPage::<A, P>::from_entry_raw(value) {
            Some(next) => Self::walk_addr_lvl1(next, vaddr),
            None => ActiveMapping {
                level: PageLevel::Level2,
                entry: PTPage::entry_ptr(page, idx),
            },
        }
    }

    /// Walks an active page table at level 3, descending if the entry refers
    /// to a present, non-huge sub-table.
    fn walk_addr_lvl3(page: *mut PTPage<A, P>, vaddr: VirtAddr) -> ActiveMapping<A> {
        let idx = vaddr.to_pgtbl_idx::<3>();
        // SAFETY: `page`/`idx` address a live installed entry; the volatile
        // read tolerates concurrent MMU updates.
        let value = unsafe { PTPage::<A, P>::read_entry_at(page, idx) };
        match PTPage::<A, P>::from_entry_raw(value) {
            Some(next) => Self::walk_addr_lvl2(next, vaddr),
            None => ActiveMapping {
                level: PageLevel::Level3,
                entry: PTPage::entry_ptr(page, idx),
            },
        }
    }

    /// Walk the installed table for `vaddr`, returning the level at which the
    /// walk stopped and a raw pointer to the entry there.
    ///
    /// The walk descends through present, non-huge entries and stops at the
    /// leaf (`Level0`), at the first absent entry, or at a huge entry.
    fn walk_addr_raw(&self, vaddr: VirtAddr) -> ActiveMapping<A> {
        let root = P::paddr_to_vaddr(self.root.root_pa()).as_mut_ptr();
        match L::TOP_LEVEL {
            PageLevel::Level3 => Self::walk_addr_lvl3(root, vaddr),
            PageLevel::Level2 => Self::walk_addr_lvl2(root, vaddr),
            _ => unreachable!(),
        }
    }

    /// Walk the installed table for `vaddr`, copying the leaf entry into `entry`.
    /// Returns a copy of `Mapping` of the entry and the `ActiveMapping` of the walk.
    fn walk_addr<'a>(
        &self,
        vaddr: VirtAddr,
        entry: &'a mut PTEntry<A>,
    ) -> (Mapping<'a, A>, ActiveMapping<A>) {
        let m = self.walk_addr_raw(vaddr);
        // SAFETY: `m.entry` points at a live entry produced by the walk; a
        // volatile read copies it without forming a reference into the table.
        *entry = unsafe { read_entry(m.entry) };
        (
            Mapping {
                level: m.level,
                entry,
            },
            m,
        )
    }

    pub fn cr3_value(&self) -> PhysAddr {
        self.root.root_pa()
    }

    /// Physical address of the root page-table page of this (sub-)tree.
    pub fn root_pa(&self) -> PhysAddr {
        self.root.root_pa()
    }

    /// Walk the installed table for `vaddr`, returning the [`ActiveMapping`]
    /// (level and raw entry pointer) at which the walk terminated.
    pub fn walk(&self, vaddr: VirtAddr) -> ActiveMapping<A> {
        self.walk_addr_raw(vaddr)
    }

    /// Return the [`ActiveMapping`] for the top-level entry at index `idx`.
    ///
    /// Used to read or update a single root-table entry (e.g. linking a
    /// sub-tree) without walking a virtual address.
    fn top_entry(&self, idx: usize) -> ActiveMapping<A> {
        let root: *mut PTPage<A, P> = P::paddr_to_vaddr(self.root.root_pa()).as_mut_ptr();
        ActiveMapping {
            level: L::TOP_LEVEL,
            entry: PTPage::entry_ptr(root, idx),
        }
    }

    pub fn copy_entry_to(&self, other: &mut GenericPageTable<A, P, L>, idx: usize) {
        let entry = self.top_entry(idx).read();
        other.set_entry(idx, entry.paddr_field(), entry.flags());
    }

    /// Set the entry at the specified index.
    ///
    /// Returns `true` if the entry was updated, `false` otherwise.
    pub fn set_entry(&mut self, idx: usize, paddr: PhysAddr, flags: A::PTFlags) -> bool {
        let mut desired = PTEntry::empty();
        desired.set(A::make_private_address(paddr), flags);
        let mapping = self.top_entry(idx);
        if mapping.read().raw() == desired.raw() {
            return false;
        }
        mapping
            .update(desired)
            .ignore("populating a new sub-tree does not require a TLB flush");
        true
    }

    /// Translate `vaddr` to a physical address through the installed table.
    pub fn phys_addr(&self, vaddr: VirtAddr) -> Result<PhysAddr, PagingError> {
        let mut entry = PTEntry::empty();
        let (mapping, _active_m) = self.walk_addr(vaddr, &mut entry);
        GenericPageTable::<A, P, L>::get_phys_addr(mapping, vaddr)
    }

    /// Maps a 4 KiB page, allocating parent tables with `parent_flags`.
    pub fn map_4k_with_parent_flags(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
        parent_flags: A::PTFlags,
    ) -> Result<(), PagingError> {
        let mut entry = PTEntry::empty();
        let (mapping, active_m) = self.walk_addr(vaddr, &mut entry);
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

    /// Maps a 4 KiB page using the default parent flags.
    pub fn map_4k(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
    ) -> Result<(), PagingError> {
        self.map_4k_with_parent_flags(vaddr, paddr, flags, shared, A::PTFlags::parent_flags())
    }

    /// Maps a 2 MiB page, allocating parent tables with `parent_flags`.
    ///
    /// # Panics
    /// Panics if either `vaddr` or `paddr` is not aligned to a 2 MiB boundary.
    pub fn map_2m_with_parent_flags(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
        parent_flags: A::PTFlags,
    ) -> Result<(), PagingError> {
        let mut entry = PTEntry::empty();
        let (mapping, active_m) = self.walk_addr(vaddr, &mut entry);
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

    /// Maps a 2 MiB page using the default parent flags.
    ///
    /// # Panics
    /// Panics if either `vaddr` or `paddr` is not aligned to a 2 MiB boundary.
    pub fn map_2m(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
    ) -> Result<(), PagingError> {
        self.map_2m_with_parent_flags(vaddr, paddr, flags, shared, A::PTFlags::parent_flags())
    }

    /// Unmaps a 4 KiB page.
    ///
    /// Returns a copy of the cleared entry (if a leaf was present) together
    /// with the TLB-flush obligation for the now-stale translation.
    pub fn unmap_4k(&self, vaddr: VirtAddr) -> Option<(PTEntry<A>, MayNeedFlush<A>)> {
        let mut entry = PTEntry::empty();
        let (m, active_m) = self.walk_addr(vaddr, &mut entry);
        GenericPageTable::<A, P, L>::do_unmap_4k(m)?;
        Some((entry, active_m.update(entry)))
    }

    /// Unmaps a 2 MiB page.
    ///
    /// Returns a copy of the cleared entry (if a huge leaf was present)
    /// together with the TLB-flush obligation for the now-stale translation.
    ///
    /// # Panics
    /// Panics if `vaddr` is not aligned to a 2 MiB boundary.
    pub fn unmap_2m(&self, vaddr: VirtAddr) -> Option<(PTEntry<A>, MayNeedFlush<A>)> {
        let mut entry = PTEntry::empty();
        let (m, active_m) = self.walk_addr(vaddr, &mut entry);
        GenericPageTable::<A, P, L>::do_unmap_2m(m)?;
        Some((entry, active_m.update(entry)))
    }

    /// Marks the 4 KiB page mapping `vaddr` as shared, splitting any
    /// enclosing 2 MiB page first.
    pub fn set_shared_4k(&self, vaddr: VirtAddr) -> Result<MayNeedFlush<A>, PagingError> {
        let mut entry = PTEntry::empty();
        let (m, active_m) = self.walk_addr(vaddr, &mut entry);
        GenericPageTable::<A, P, L>::do_set_shared_4k(m, vaddr)?;
        Ok(active_m.update(entry))
    }

    /// Marks the 4 KiB page mapping `vaddr` as encrypted (private), splitting
    /// any enclosing 2 MiB page first.
    pub fn set_encrypted_4k(&self, vaddr: VirtAddr) -> Result<MayNeedFlush<A>, PagingError> {
        let mut entry = PTEntry::empty();
        let (m, active_m) = self.walk_addr(vaddr, &mut entry);
        GenericPageTable::<A, P, L>::do_set_encrypted_4k(m, vaddr)?;
        Ok(active_m.update(entry))
    }
}
