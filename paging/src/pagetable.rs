// SPDX-License-Identifier: MIT OR Apache-2.0

//! This module provides state-aware page table types for 4 KiB granule paging.
//!
//! They are designed for different OS or architectures.
//!
//! The implementation assumes a 4 KiB base page granule with a 4-level
//! page table hierarchy (8-byte PTEntry and 512 entries per page), supporting three page
//! sizes: 4 KiB, 2 MiB, and 1 GiB. This covers x86_64 (PML4) and
//! ARM64 with 4 KiB granule. Other granule sizes (16 KiB, 64 KiB on
//! ARM64) are not supported.
use core::cmp::min;
use core::marker::PhantomData;

use crate::address::{Address, PhysAddr, VirtAddr};
use crate::ptpage::{
    ENTRY_COUNT, Mapping, PTE_SHIFT, PTEntry, PTPage, PageFrame, virt_from_lvl_idx,
};
use crate::sizes::{PAGE_SIZE, PAGE_SIZE_1G, PAGE_SIZE_2M};
use crate::tlb::MayNeedFlush;
use crate::traits::{
    ArchPagingMeta, GenericPageTableFlags, NextLevel, PageLevel, PagingError, PagingHandler,
    PagingLevel, PagingLevel3, SelfMap,
};

/// Read-only *behavior* of a state-aware page table entry mapping handle.
pub trait MappingRefOps<'a, A>: Sized
where
    A: ArchPagingMeta,
{
    /// The level at which the entry terminated.
    fn level(&self) -> PageLevel;

    /// Read the entry (volatile for installed, plain for inactive).
    fn read(&self) -> PTEntry<A>;
}

/// Read + commit *behavior* of a state-aware mutable mapping handle.
pub trait MappingMutOps<'a, A>: Sized
where
    A: ArchPagingMeta,
{
    /// The level at which the mapping terminated.
    fn level(&self) -> PageLevel;

    /// Read the entry (volatile for installed, plain for inactive).
    fn read(&self) -> PTEntry<A>;

    /// A scratch [`Mapping`] the `do_*` helpers edit before committing.
    fn staged(&mut self) -> Mapping<'_, A>;

    /// Commit the staged entry, returning any TLB-flush obligation.
    fn commit(self) -> MayNeedFlush<A::TlbFlushTok>;

    /// Commit an entry for which **no TLB flush is owed**: either the previous
    /// entry was non-present (installed) or the table is not installed.
    fn commit_no_flush<F, O>(self, update: F) -> Result<O, PagingError>
    where
        F: FnOnce(Mapping<'_, A>) -> Result<O, PagingError>;
}

/// Mutable mapping handle for an **installed** table.
///
/// Its constructor is private to this module; it is only produced by a
/// writable walk.
#[derive(Debug)]
pub struct InstalledMapping<'a, A: ArchPagingMeta> {
    vaddr: Option<VirtAddr>,
    level: PageLevel,
    entry: *mut PTEntry<A>,
    staged: Option<PTEntry<A>>,
    _lifetime: PhantomData<&'a mut PTEntry<A>>,
}

impl<'a, A: ArchPagingMeta> MappingMutOps<'a, A> for InstalledMapping<'a, A> {
    fn level(&self) -> PageLevel {
        self.level
    }

    fn read(&self) -> PTEntry<A> {
        // SAFETY: `entry` points at a valid PTEntry; installed reads are volatile.
        unsafe { PTEntry::read_pte(self.entry) }
    }

    fn staged(&mut self) -> Mapping<'_, A> {
        let entry = self.entry;
        let staged = self.staged.get_or_insert_with(||
            // SAFETY: `entry` points at a valid PTEntry; installed reads are
            // volatile. First stage seeds the buffer from the live entry.
            unsafe { PTEntry::read_pte(entry) });
        Mapping::new(self.level, staged)
    }

    fn commit(self) -> MayNeedFlush<A::TlbFlushTok> {
        if let Some(staged) = self.staged {
            // SAFETY: `entry` points at a valid live PTEntry; installed writes are volatile.
            unsafe { PTEntry::write_pte(self.entry, staged) };
        }
        match self.vaddr {
            Some(vaddr) => MayNeedFlush::new(vaddr, self.level),
            None => MayNeedFlush::all(),
        }
    }

    fn commit_no_flush<F, O>(mut self, update: F) -> Result<O, PagingError>
    where
        F: FnOnce(Mapping<'_, A>) -> Result<O, PagingError>,
    {
        let staged = self.staged();
        if staged.entry.present() {
            return Err(PagingError::EntryAlreadyPresent);
        }
        let ret = update(staged)?;
        // SAFETY: InstalledMapping guarantees `entry` points at a valid PTE.
        unsafe { PTEntry::write_pte(self.entry, *self.staged().entry) };
        Ok(ret)
    }
}

/// Read-only mapping handle for an **installed** table.
///
/// Its constructor is private to this module; it is only produced by a
/// read-only walk. The lifetime `'a` is bounded by the table's lifetime. Only
/// `A` is stored; `P` is a `new`-only generic used to interpret the incoming
/// `*const PTPage<A, P>`.
#[derive(Debug)]
pub struct InstalledMappingRef<'a, A: ArchPagingMeta> {
    level: PageLevel,
    entry: *const PTEntry<A>,
    _lifetime: PhantomData<&'a PTEntry<A>>,
}

impl<'a, A: ArchPagingMeta> MappingRefOps<'a, A> for InstalledMappingRef<'a, A> {
    fn level(&self) -> PageLevel {
        self.level
    }

    fn read(&self) -> PTEntry<A> {
        // SAFETY: `entry` points at a valid PTEntry; installed reads are volatile.
        unsafe { PTEntry::read_pte(self.entry) }
    }
}

/// Read-only mapping handle for an **inactive** table: a plain shared borrow.
#[derive(Debug)]
pub struct InactiveMappingRef<'a, A: ArchPagingMeta, P: PagingHandler> {
    level: PageLevel,
    entry: &'a PTEntry<A>,
    _handler: PhantomData<P>,
}

impl<'a, A: ArchPagingMeta, P: PagingHandler> MappingRefOps<'a, A>
    for InactiveMappingRef<'a, A, P>
{
    fn level(&self) -> PageLevel {
        self.level
    }

    fn read(&self) -> PTEntry<A> {
        *self.entry
    }
}

impl<'a, A: ArchPagingMeta> MappingMutOps<'a, A> for Mapping<'a, A> {
    fn level(&self) -> PageLevel {
        self.level
    }

    fn read(&self) -> PTEntry<A> {
        *self.entry
    }

    fn staged(&mut self) -> Mapping<'_, A> {
        Mapping::new(self.level, self.entry)
    }

    fn commit(self) -> MayNeedFlush<A::TlbFlushTok> {
        MayNeedFlush::none()
    }

    fn commit_no_flush<F, O>(self, update: F) -> Result<O, PagingError>
    where
        F: FnOnce(Mapping<'_, A>) -> Result<O, PagingError>,
    {
        let ret = update(self)?;
        Ok(ret)
    }
}

/// Per-state associated types. Parameterized over the arch `A` and handler `P`
/// so the associated items only carry a lifetime, keeping every `S::…`
/// reference free of `A`/`P` turbofish.
pub trait PageTableTypesByState<A: ArchPagingMeta, P: PagingHandler> {
    /// TLB-flush obligation of this state's mutable commit
    /// (Installed: `MayNeedFlush<A::TlbFlushTok>`, Inactive: `()`).
    type Flush;

    /// Read handle type (behavior only; construction is via `new_mapping_ref`).
    type MappingRef<'a>: MappingRefOps<'a, A>;

    /// Mutable handle type; the `Flush` equality bound ties
    /// `MappingMutOpMayNeedFlush<A::TlbFlushTok>` to `Self::Flush`.
    type MappingMut<'a>: MappingMutOps<'a, A>;
}

/// The state-specific construction + descent primitives. This trait is
/// **sealed**: its supertrait [`PageTableState`] can only be implemented inside
/// this crate (via the private `SealedState`), so external crates may name it
/// as a bound but never implement it.
pub trait PageTableStateSealed<A: ArchPagingMeta, P: PagingHandler>:
    PageTableState + PageTableTypesByState<A, P>
{
    /// Construct a read handle to entry `index` of `page` at `level`.
    ///
    /// `page` must point at a valid page-table page owned by the table; each
    /// state wraps the resulting entry pointer as its own read handle (raw for
    /// installed, a bounded `&PTEntry` for inactive).
    fn new_mapping_ref<'a>(
        level: PageLevel,
        page: *const PTPage<A, P>,
        index: usize,
    ) -> Self::MappingRef<'a>;

    /// Construct a mutable handle to entry `index` of `page` at `level`.
    ///
    /// `page` must point at a valid page-table page owned by the table.
    fn new_mapping_mut<'a>(
        vaddr: Option<VirtAddr>,
        level: PageLevel,
        page: *mut PTPage<A, P>,
        index: usize,
    ) -> Self::MappingMut<'a>;

    /// Descend to the read child page of `entry` (present, non-huge).
    fn child(entry: &PTEntry<A>) -> Option<*const PTPage<A, P>> {
        PTPage::<A, P>::from_entry_raw(entry).map(|p| p.cast_const())
    }

    /// Descend to the mutable child page of `entry` (present, non-huge).
    fn child_mut(entry: &PTEntry<A>) -> Option<*mut PTPage<A, P>> {
        PTPage::<A, P>::from_entry_raw(entry)
    }
}

/// Private, non-generic seal for the public [`PageTableState`] marker.
trait SealedState {}
impl SealedState for Installed {}
impl SealedState for Inactive {}
/// Marker trait for the lifecycle state of a [`GenericPageTable`].
///
/// A table is either [`Installed`] (may be live in `CR3` on some CPU, so its
/// entries must only be touched through volatile accessors) or [`Inactive`]
/// (not reachable by any MMU walker, so plain `&mut PTPage` access is sound).
#[allow(private_bounds)]
pub trait PageTableState: SealedState {
    const IS_INSTALLED: bool = true;
}

/// The table may be installed in `CR3`; entries are accessed volatile-only.
#[derive(Debug)]
pub struct Installed;
impl PageTableState for Installed {
    const IS_INSTALLED: bool = true;
}

impl<A: ArchPagingMeta, P: PagingHandler> PageTableTypesByState<A, P> for Installed {
    type Flush = MayNeedFlush<A::TlbFlushTok>;
    type MappingRef<'a> = InstalledMappingRef<'a, A>;
    type MappingMut<'a> = InstalledMapping<'a, A>;
}

impl<A: ArchPagingMeta, P: PagingHandler> PageTableStateSealed<A, P> for Installed {
    fn new_mapping_ref<'a>(
        level: PageLevel,
        page: *const PTPage<A, P>,
        index: usize,
    ) -> Self::MappingRef<'a> {
        InstalledMappingRef {
            level,
            entry: PTPage::<A, P>::entry_ptr(page, index),
            _lifetime: PhantomData,
        }
    }

    fn new_mapping_mut<'a>(
        vaddr: Option<VirtAddr>,
        level: PageLevel,
        page: *mut PTPage<A, P>,
        index: usize,
    ) -> Self::MappingMut<'a> {
        InstalledMapping {
            vaddr,
            level,
            entry: PTPage::<A, P>::entry_ptr(page, index) as _,
            staged: None,
            _lifetime: PhantomData,
        }
    }
}

/// The table is not reachable by any MMU walker and may be edited directly.
#[derive(Debug)]
pub struct Inactive;

impl PageTableState for Inactive {
    const IS_INSTALLED: bool = false;
}

impl<A: ArchPagingMeta, P: PagingHandler> PageTableTypesByState<A, P> for Inactive {
    type Flush = ();
    type MappingRef<'a> = InactiveMappingRef<'a, A, P>;
    type MappingMut<'a> = Mapping<'a, A>;
}

impl<A: ArchPagingMeta, P: PagingHandler> PageTableStateSealed<A, P> for Inactive {
    fn new_mapping_ref<'a>(
        level: PageLevel,
        page: *const PTPage<A, P>,
        index: usize,
    ) -> Self::MappingRef<'a> {
        let entry = PTPage::<A, P>::entry_ptr(page, index);
        InactiveMappingRef {
            level,
            // SAFETY: `page` is a valid inactive page-table page and
            // `index < ENTRY_COUNT`, so `entry` points at a live `PTEntry` that
            // no MMU walker can alias while the table is inactive.
            entry: unsafe { &*entry },
            _handler: PhantomData,
        }
    }

    fn new_mapping_mut<'a>(
        _vaddr: Option<VirtAddr>,
        level: PageLevel,
        page: *mut PTPage<A, P>,
        index: usize,
    ) -> Self::MappingMut<'a> {
        let entry = PTPage::<A, P>::entry_ptr(page, index).cast_mut();
        // SAFETY: `page` is a valid inactive page-table page and
        // `index < ENTRY_COUNT`, so `entry` points at a live `PTEntry` uniquely
        // owned by this handle while the table is inactive.
        Mapping::new(level, unsafe { &mut *entry })
    }
}

/// An abstract type representing a page table at level `L`.
///
/// The root represents a concrete page-table node whose level is described
/// by `L`. This can represent either a complete top-level page table, such as
/// a PML4-rooted table, or a lower-level subtree, such as a PDPT-rooted table
/// that is installed into a top-level page table.
///
/// The ownership of top-level page is held by the `GenericPageTable`.
///
/// Ownership of lower-level subtrees are left to the OS-specific code.
/// If a lower-level subtree is shared between multiple top-level page tables,
/// all users of that subtree must coordinate updates to avoid concurrent
/// modifications to the same page-table entries.
///
/// We use state type parameters to track the state of page table, indicating
/// whether it is installed in a CPU or not. We rely on a conservative transition rule:
///
/// 1. The transition from inactive to installed is always safe since the operation
///    on an installed page table is still valid on an inactive page table. A clear
///    transition point is when we first load the table into cr3 in one CPU.
///
/// 2. The transition from installed to inactive is always unsafe since we never
///    track the cr3 values across different CPUs. In practice, this is difficult
///    since page table switch may happen in assembly code and thus it is not
///    possible to provide a single trusted function to track cr3 values. Thus,
///    we left it to the developer to decide when it is safe to use an installed
///    table as inactive.
///
/// For InstalledPageTable, there are several requirements:
/// a. compiler should not optimize the write even if no read happens after the write.
/// b. compiler should not optimize the read even if no write happens since the OS may
///    need to track A/D bits.
/// c. software should not observe torn writes, when MMU may update the entry (A/D bits)
///
/// All requirements can be satisfied if page table entries are accessed via
/// 64-bit volatile read and write.
///
/// Why AtomicUsize is not necessary?
/// a. Page table rely on Lock to ensure concurrent software accesses are safe
///    and synced and so it should not encounter relaxed order issue.
/// b. 64-bit aligned volatile access on 64 bits is atomic in x86_64 and Arm.
/// c. It is tolerable to lose some A/D bit updates by MMU. For example,
///    software read old -> hardware set A/D -> software update new = old | some_flag.
///
/// For InactivePageTable, we use normal memory accesses since it is not installed in any CPU.
#[repr(transparent)]
#[derive(Debug)]
pub struct GenericPageTable<A: ArchPagingMeta, P: PagingHandler, L: PagingLevel, S: PageTableState>
{
    // store the inactive table here to prevent it from being used as inactive.
    /// The root page table.
    root: VirtAddr,
    _level: PhantomData<(A, P, L, S)>,
}

/// Stateless methods for managing a generic page table.
impl<A: ArchPagingMeta, P: PagingHandler, L: PagingLevel, S: PageTableState>
    GenericPageTable<A, P, L, S>
{
    /// Allocate a new, zeroed root page table.
    /// # Errors
    /// Returns [`PagingError`] if the backing page cannot be allocated.
    pub fn alloc() -> Result<Self, PagingError> {
        let (page, _paddr) = PTPage::<A, P>::alloc()?;
        Ok(Self {
            root: page.into(),
            _level: PhantomData,
        })
    }

    /// Physical address of the root page (state-independent).
    pub fn root_pa(&self) -> PhysAddr {
        P::vaddr_to_paddr(self.root)
    }

    /// Virtual address of the root page (state-independent).
    pub fn root_vaddr(&self) -> VirtAddr {
        self.root
    }

    /// Wrap the page table as installed or inactive one.
    /// # Safety
    /// The caller must guarantee that
    /// 1. the root is a valid, page-aligned pointer
    ///    to a root page-table page that remains valid for the lifetime of the implementor.
    /// 2. the state of the page table is consistent with the type parameter `S` (i.e., either `Installed` or `Inactive`).
    pub unsafe fn from_ptr(root_page: *mut PTPage<A, P>) -> Self {
        Self {
            root: root_page.into(),
            _level: PhantomData,
        }
    }

    /// Computes the index within a page table at the given level for a
    /// virtual address `vaddr`.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address to compute the index for.
    ///
    /// # Returns
    /// The index within the page table.
    pub const fn index<const LVL: usize>(vaddr: VirtAddr) -> usize {
        vaddr.to_pgtbl_idx::<LVL>()
    }

    /// Computes the index within a page table at a runtime `level` for a
    /// virtual address `vaddr`.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address to compute the index for.
    /// - `level`: The page-table level to index into.
    ///
    /// # Returns
    /// The index within the page table.
    pub const fn index_at(vaddr: VirtAddr, level: PageLevel) -> usize {
        match level {
            PageLevel::Level0 => Self::index::<0>(vaddr),
            PageLevel::Level1 => Self::index::<1>(vaddr),
            PageLevel::Level2 => Self::index::<2>(vaddr),
            PageLevel::Level3 => Self::index::<3>(vaddr),
        }
    }
}

impl<A: ArchPagingMeta, P: PagingHandler, L: PagingLevel, S: PageTableState> Drop
    for GenericPageTable<A, P, L, S>
{
    fn drop(&mut self) {
        // Only free the root page for an inactive table. An installed table may
        // still be reachable by the MMU, so we intentionally leak its root
        // rather than free a page that hardware might still walk. Callers must
        // transition the table to `Inactive` (via `as_inactive`) and tear it
        // down explicitly to reclaim it.
        if !S::IS_INSTALLED {
            // SAFETY: GenericPageTable owns the root page, `root` was allocated by `alloc()`.
            unsafe { P::deallocate_physical_page(self.root_pa()) };
        }
    }
}

impl<A: ArchPagingMeta, P: PagingHandler, L: PagingLevel + NextLevel, S: PageTableStateSealed<A, P>>
    GenericPageTable<A, P, L, S>
{
    /// Get the physical address of the next level page table at the given index.
    /// Returns None if the entry is not present or is a huge page.
    pub fn next_table_pa(&self, idx: usize) -> Option<PhysAddr>
    where
        L: NextLevel,
    {
        let page = self.root.as_ptr();
        let entry = S::new_mapping_ref(L::TOP_LEVEL, page, idx).read();
        if !entry.present() || entry.huge() {
            return None;
        }
        Some(entry.address())
    }

    /// Shared read walk for `vaddr` (one body for all states). Descends through
    /// present, non-huge entries and stops at the leaf (`Level0`), at the first
    /// absent entry, or at a huge entry. `page` is a raw pointer to the current
    /// level's page; the next pointer comes from the read `entry`.
    pub fn walk(&self, vaddr: VirtAddr) -> S::MappingRef<'_> {
        let page: *const PTPage<A, P> = self.root.as_ptr();
        let level = L::TOP_LEVEL;
        let mut mapping = S::new_mapping_ref(level, page, Self::index_at(vaddr, level));
        for _ in 0..L::TOP_LEVEL as usize {
            let entry = mapping.read();
            match PTPage::<A, P>::from_entry_raw(&entry) {
                Some(next) => {
                    let level = mapping
                        .level()
                        .next_down()
                        .expect("non-leaf level has a child");
                    mapping = S::new_mapping_ref(level, next, Self::index_at(vaddr, level));
                }
                None => {
                    return mapping;
                }
            }
        }
        mapping
    }

    /// Shared mutable walk for `vaddr` (one body for all states), mirroring
    /// [`walk`](Self::walk). `page` is a raw pointer to the current level's
    /// page and the next pointer (`S::child_mut`) points at a disjoint child
    /// page, so no two mutable entry handles ever alias.
    pub fn walk_mut(&mut self, vaddr: VirtAddr) -> S::MappingMut<'_> {
        let page: *mut PTPage<A, P> = self.root.as_mut_ptr();
        let level = L::TOP_LEVEL;
        let mut mapping =
            S::new_mapping_mut(Some(vaddr), level, page, Self::index_at(vaddr, level));
        for _ in 0..L::TOP_LEVEL as usize {
            let entry = mapping.read();
            match PTPage::<A, P>::from_entry_raw(&entry) {
                Some(next) => {
                    let level = mapping
                        .level()
                        .next_down()
                        .expect("non-leaf level has a child");
                    mapping =
                        S::new_mapping_mut(Some(vaddr), level, next, Self::index_at(vaddr, level));
                }
                None => {
                    return mapping;
                }
            }
        }
        mapping
    }

    /// Retrieves the physical address of a mapping.
    ///
    /// # Returns
    /// The physical address of the mapping if present; otherwise, an error
    /// ([`PagingError`]).
    pub fn translate(&self, vaddr: VirtAddr) -> Result<PageFrame<A>, PagingError> {
        let mapping = self.walk(vaddr);
        let entry = mapping.read();
        match mapping.level() {
            PageLevel::Level0 => {
                let offset = vaddr.page_offset();
                if !entry.present() {
                    return Err(PagingError::NotMapped);
                }
                Ok(PageFrame::Size4K(entry.page_frame() + offset))
            }
            PageLevel::Level1 => {
                let offset = vaddr.bits() & (PAGE_SIZE_2M - 1);
                if !entry.present() || !entry.huge() {
                    return Err(PagingError::NotMapped);
                }
                Ok(PageFrame::Size2M(entry.page_frame() + offset))
            }
            PageLevel::Level2 => {
                let offset = vaddr.bits() & (PAGE_SIZE_1G - 1);
                if !entry.present() || !entry.huge() {
                    return Err(PagingError::NotMapped);
                }
                Ok(PageFrame::Size1G(entry.page_frame() + offset, PhantomData))
            }
            _ => Err(PagingError::NotMapped),
        }
    }

    pub fn phys_addr(&self, vaddr: VirtAddr) -> Result<PhysAddr, PagingError> {
        self.translate(vaddr).map(|pf| pf.address())
    }

    /// Maps a 4KB page, applying `parent_flags` to any allocated parent PTEs.
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
    ) -> Result<(), PagingError>
    where
        L: NextLevel,
    {
        let mapping = self.walk_mut(vaddr);
        mapping.commit_no_flush(|m| {
            PTPage::<A, P>::do_map_4k_with_parent_flags(
                m,
                vaddr,
                paddr,
                flags,
                shared,
                parent_flags,
            )
        })?;
        Ok(())
    }

    /// Maps a 4KB page.
    ///
    /// # Returns
    /// A result indicating success or failure ([`PagingError`]).
    pub fn map_4k(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
    ) -> Result<(), PagingError>
    where
        L: NextLevel,
    {
        self.map_4k_with_parent_flags(vaddr, paddr, flags, shared, A::PTFlags::parent_flags())
    }

    /// Maps a 2MB page, applying `parent_flags` to any allocated parent PTEs.
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
    ) -> Result<(), PagingError>
    where
        L: NextLevel,
    {
        self.walk_mut(vaddr).commit_no_flush(|m| {
            PTPage::<A, P>::do_map_2m_with_parent_flags(
                m,
                vaddr,
                paddr,
                flags,
                shared,
                parent_flags,
            )
        })?;
        Ok(())
    }

    /// Maps a 2MB page.
    ///
    /// # Panics
    /// Panics if either `vaddr` or `paddr` is not aligned to a 2MB boundary.
    pub fn map_2m(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
    ) -> Result<(), PagingError>
    where
        L: NextLevel,
    {
        self.map_2m_with_parent_flags(vaddr, paddr, flags, shared, A::PTFlags::parent_flags())
    }

    /// Unmaps a 4KB page.
    ///
    /// # Returns
    /// A copy of the entry that mapped the address before it was cleared (if a
    /// leaf was present) together with the TLB-flush obligation.
    pub fn unmap_4k(
        &mut self,
        vaddr: VirtAddr,
    ) -> (Option<PTEntry<A>>, MayNeedFlush<A::TlbFlushTok>) {
        let mut m = self.walk_mut(vaddr);
        (PTPage::<A, P>::do_unmap_4k(m.staged()), m.commit())
    }

    /// Unmaps a 2MB page.
    ///
    /// # Returns
    /// A copy of the entry that mapped the address before it was cleared (if a
    /// huge leaf was present) together with the TLB-flush obligation.
    ///
    /// # Panics
    /// Panics if `vaddr` is not aligned to a 2MB boundary.
    pub fn unmap_2m(
        &mut self,
        vaddr: VirtAddr,
    ) -> (Option<PTEntry<A>>, MayNeedFlush<A::TlbFlushTok>) {
        let mut m = self.walk_mut(vaddr);
        (PTPage::<A, P>::do_unmap_2m(m.staged()), m.commit())
    }

    pub fn unmap(&mut self, vaddr: VirtAddr) -> (Option<PageLevel>, MayNeedFlush<A::TlbFlushTok>) {
        let mut m = self.walk_mut(vaddr);
        (PTPage::<A, P>::do_unmap(m.staged()), m.commit())
    }

    /// Sets the shared state for a 4KB page.
    ///
    /// # Returns
    /// A result with the TLB-flush obligation on success or a [`PagingError`].
    pub fn set_shared_4k(
        &mut self,
        vaddr: VirtAddr,
    ) -> Result<MayNeedFlush<A::TlbFlushTok>, PagingError> {
        let mut m = self.walk_mut(vaddr);
        PTPage::<A, P>::do_set_shared_4k(m.staged(), vaddr)?;
        Ok(m.commit())
    }

    /// Sets the encryption state for a 4KB page.
    ///
    /// # Returns
    /// A result with the TLB-flush obligation on success or a [`PagingError`].
    pub fn set_encrypted_4k(
        &mut self,
        vaddr: VirtAddr,
    ) -> Result<MayNeedFlush<A::TlbFlushTok>, PagingError> {
        let mut m = self.walk_mut(vaddr);
        PTPage::<A, P>::do_set_encrypted_4k(m.staged(), vaddr)?;
        Ok(m.commit())
    }

    /// Populates the page table entry at the given index
    /// TLB flush is not required since the caller assumes the old entry is
    /// non-present or the table is not installed.
    ///
    /// # Parameters
    /// - `idx`: The index within the page table to populate.
    /// - `subpage_pa`: The physical address of the subpage to map.
    ///
    /// # Returns
    /// `true` if the entry was updated, `false` if the entry already contained the desired mapping.
    pub fn populate(&mut self, idx: usize, subpage_pa: PhysAddr) -> Result<bool, PagingError>
    where
        L: NextLevel,
    {
        let desired = PTEntry::new(
            A::make_private_address(subpage_pa),
            A::PTFlags::parent_flags(),
        );
        let page = self.root.as_mut_ptr();
        let mut mapping = S::new_mapping_mut(None, L::TOP_LEVEL, page, idx);
        let staged = mapping.staged().entry;
        if staged.raw() == desired.raw() {
            return Ok(false);
        }
        mapping.commit_no_flush(|m| {
            *m.entry = desired;
            Ok(true)
        })
    }

    /// Maps the half-open virtual range `[start, end)` using 4KB pages,
    /// starting at physical address `phys`.
    ///
    /// # Returns
    /// A result indicating success or failure ([`PagingError`]).
    pub fn map_region_4k(
        &mut self,
        start: VirtAddr,
        end: VirtAddr,
        phys: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
    ) -> Result<(), PagingError> {
        let mut vaddr = start;
        while vaddr < end {
            let offset = vaddr - start;
            self.map_4k(vaddr, phys + offset, flags, shared)?;
            vaddr = vaddr + PAGE_SIZE;
        }
        Ok(())
    }

    /// Unmaps the half-open virtual range `[start, end)` mapped with 4KB
    /// pages, returning the combined [`MayNeedFlush`] TLB-flush obligation.
    pub fn unmap_region_4k(
        &mut self,
        start: VirtAddr,
        end: VirtAddr,
    ) -> MayNeedFlush<A::TlbFlushTok> {
        let mut flush = MayNeedFlush::none();
        let mut vaddr = start;
        while vaddr < end {
            let (_, f) = self.unmap_4k(vaddr);
            flush = flush.and(f);
            vaddr = vaddr + PAGE_SIZE;
        }
        flush
    }

    /// Maps the half-open virtual range `[start, end)` using 2MB pages,
    /// starting at physical address `phys`.
    ///
    /// # Returns
    /// A result indicating success or failure ([`PagingError`]).
    pub fn map_region_2m(
        &mut self,
        start: VirtAddr,
        end: VirtAddr,
        phys: PhysAddr,
        flags: A::PTFlags,
        shared: bool,
    ) -> Result<(), PagingError>
    where
        L: NextLevel,
    {
        let mut vaddr = start;
        while vaddr < end {
            let offset = vaddr - start;
            self.map_2m(vaddr, phys + offset, flags, shared)?;
            vaddr = vaddr + PAGE_SIZE_2M;
        }
        Ok(())
    }

    /// Unmaps the half-open virtual range `[start, end)` mapped with 2MB
    /// pages, returning the combined [`MayNeedFlush`] TLB-flush obligation.
    ///
    /// The range must be 2MB-aligned and correspond to a set of huge mappings.
    pub fn unmap_region_2m(
        &mut self,
        start: VirtAddr,
        end: VirtAddr,
    ) -> MayNeedFlush<A::TlbFlushTok>
    where
        L: NextLevel,
    {
        let mut flush = MayNeedFlush::none();
        let mut vaddr = start;
        while vaddr < end {
            let (_, f) = self.unmap_2m(vaddr);
            flush = flush.and(f);
            vaddr = vaddr + PAGE_SIZE_2M;
        }
        flush
    }

    /// Maps the half-open virtual range `[start, end)` to physical memory
    /// starting at `phys`, preferring 2MB pages where alignment and size
    /// allow and falling back to 4KB pages otherwise.
    ///
    /// # Returns
    /// A result indicating success or failure ([`PagingError`]).
    pub fn map_region(
        &mut self,
        start: VirtAddr,
        end: VirtAddr,
        phys: PhysAddr,
        flags: A::PTFlags,
    ) -> Result<(), PagingError> {
        let mut vaddr = start;
        let mut paddr = phys;

        while vaddr < end {
            if vaddr.is_aligned(PAGE_SIZE_2M)
                && paddr.is_aligned(PAGE_SIZE_2M)
                && vaddr + PAGE_SIZE_2M <= end
                && self.map_2m(vaddr, paddr, flags, false).is_ok()
            {
                vaddr = vaddr + PAGE_SIZE_2M;
                paddr = paddr + PAGE_SIZE_2M;
                continue;
            }

            self.map_4k(vaddr, paddr, flags, false)?;
            vaddr = vaddr + PAGE_SIZE;
            paddr = paddr + PAGE_SIZE;
        }

        Ok(())
    }

    /// Unmaps the half-open virtual range `[start, end)`, clearing both 4KB
    /// and 2MB leaf entries.
    ///
    /// # Returns
    /// 1. whether every page in the range was mapped (`true`) or not (`false`).
    /// 2. a [`MayNeedFlush`] indicating which TLB entries may need to be flushed.
    ///    All mapped pages in the range are unmapped regardless.
    pub fn unmap_region(
        &mut self,
        start: VirtAddr,
        end: VirtAddr,
    ) -> (bool, MayNeedFlush<A::TlbFlushTok>) {
        let mut flush = MayNeedFlush::none();
        let mut vaddr = start;
        let mut all_mapped = true;
        while vaddr < end {
            let (level, f) = self.unmap(vaddr);
            flush = flush.and(f);
            vaddr = match level {
                Some(l) => vaddr + l.size(),
                _ => {
                    all_mapped = false;
                    vaddr + PAGE_SIZE
                }
            };
        }
        (all_mapped, flush)
    }

    /// Returns `true` if no entry in `page` (holding `level` entries) is present.
    fn page_is_empty(page: *const PTPage<A, P>, level: PageLevel) -> bool {
        (0..ENTRY_COUNT).all(|idx| !S::new_mapping_ref(level, page, idx).read().present())
    }

    fn free_pt_after_unmap(
        page: *mut PTPage<A, P>,
        level: PageLevel,
        start_vaddr: VirtAddr,
        end_vaddr: VirtAddr,
    ) -> bool {
        // Leaf entry representing mapped frame which should not be freed here.
        if level.next_down().is_none() {
            return Self::page_is_empty(page, level);
        }
        let start = Self::index_at(start_vaddr, level);
        let end = Self::index_at(end_vaddr, level);
        let mut next_start_vaddr = start_vaddr;
        let next_level = level.next_down().unwrap();
        let next_end_vaddr = (next_start_vaddr + next_level.size()).align_down(next_level.size());
        let mut next_end_vaddr = min(next_end_vaddr, end_vaddr);
        for index in start..=end {
            let mut mapping = S::new_mapping_mut(None, level, page, index);
            let entry = mapping.read();
            if let Some(next) = PTPage::<A, P>::from_entry_raw(&entry) {
                if Self::free_pt_after_unmap(
                    next,
                    level.next_down().unwrap(),
                    next_start_vaddr,
                    next_end_vaddr,
                ) {
                    mapping.staged().entry.clear();
                    // SAFETY: We do not need flush the TLB,
                    // since the old leaf mappings were non-present.
                    unsafe {
                        mapping.commit().ignore();
                    }
                    // SAFETY: the PT page is not reachable and can be safely deallocated.
                    unsafe {
                        P::deallocate_physical_page(entry.address());
                    }
                }
                next_start_vaddr = next_end_vaddr;
                next_end_vaddr = min(next_end_vaddr + next_level.size(), end_vaddr);
            }
        }
        Self::page_is_empty(page, level)
    }

    /// Frees the page table pages that are no longer reachable after unmapping
    pub fn free_page_table_by_range(&mut self, start_vaddr: VirtAddr, end_vaddr: VirtAddr) {
        Self::free_pt_after_unmap(
            self.root.as_mut_ptr(),
            PagingLevel3::TOP_LEVEL,
            start_vaddr,
            end_vaddr,
        );
    }
}

impl<A: ArchPagingMeta, P: PagingHandler + SelfMap, S: PageTableStateSealed<A, P>>
    GenericPageTable<A, P, PagingLevel3, S>
{
    /// Install the self-map entry in a freshly zeroed page table.
    ///
    /// The self-map PML4 entry is written at [`SelfMap::SELFMAP_IDX`] and points
    /// at this page table's own root page.
    pub fn init_self_map(&mut self, root_pa: PhysAddr) {
        let flags = A::PTFlags::self_map_table_flags();
        let mapping = S::new_mapping_mut(
            None,
            PagingLevel3::TOP_LEVEL,
            self.root.as_mut_ptr(),
            P::SELFMAP_IDX,
        );
        mapping
            .commit_no_flush(|m| {
                m.entry.set(A::make_private_address(root_pa), flags);
                Ok(())
            })
            .expect("fails to init self mapping");
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
    /// This reads page-table entries through the self mapping.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the a self-mapped page table is installed and
    /// no concurrent update to the page table entries for the vaddr.
    ///
    /// Risk behinds the lockless translation:
    /// 1. dangling pointer: a PTPage is already deallocated.
    /// 3. Wrong translation: another thread remaps or modifies mapping.
    ///
    /// # Parameters
    /// - `vaddr': The virtual address to translate.
    ///
    /// # Returns
    /// Some(PageFrame) if the virtual address is valid.
    /// None if the virtual address is not valid.
    pub unsafe fn virt_to_frame(vaddr: VirtAddr) -> Option<PageFrame<A>> {
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
            let pa = pdpe.page_frame() + (vaddr.bits() & (PAGE_SIZE_1G - 1));
            return Some(PageFrame::Size1G(pa, PhantomData));
        }

        // SAFETY: The PDPE was checked to be present and not to be a huge
        // page. So the PDE exists and can be read safely.
        let pde = unsafe { PTEntry::<A>::read_pte(pde_addr.as_ptr()) };
        if !pde.present() {
            return None;
        }
        if pde.huge() {
            let pa = pde.page_frame() + (vaddr.bits() & (PAGE_SIZE_2M - 1));
            return Some(PageFrame::Size2M(pa));
        }

        // SAFETY: The PDE was checked to be present and not to be a huge
        // page. So the PTE exists and can be read safely.
        let pte = unsafe { PTEntry::<A>::read_pte(pte_addr.as_ptr()) };
        if pte.present() {
            let pa = pte.page_frame() + (vaddr.page_offset());
            Some(PageFrame::Size4K(pa))
        } else {
            None
        }
    }
}

pub type InactivePageTable<A, P, L> = GenericPageTable<A, P, L, Inactive>;

impl<A: ArchPagingMeta, P: PagingHandler, L: PagingLevel> InactivePageTable<A, P, L> {
    /// Transition this table to the [`Installed`] state.
    ///
    /// Safe because [`Installed`] only ever accesses entries through volatile
    /// operations, which is a sound superset of what an inactive table allows.
    /// The actual `CR3` load is performed separately by OS-specific code.
    pub fn install(self) -> InstalledPageTable<A, P, L> {
        let root = self.root;
        // Transfer ownership of the root page to the new handle without
        // running `Drop` (which would free it) on the old one.
        core::mem::forget(self);
        InstalledPageTable {
            root,
            _level: PhantomData,
        }
    }

    /// # Safety
    /// The caller must guarantee that the page table's children are not in use and can be safely freed.
    pub unsafe fn free_children(&mut self) {
        // SAFETY: `self.root` is this inactive table's valid root page; the
        // caller guarantees its children are not in use and can be safely
        // freed.
        unsafe { PTPage::<A, P>::from_vaddr(self.root).free_lvl(L::TOP_LEVEL) };
    }
}

pub type InstalledPageTable<A, P, L> = GenericPageTable<A, P, L, Installed>;

impl<A: ArchPagingMeta, P: PagingHandler, L: PagingLevel> InstalledPageTable<A, P, L> {
    pub fn cr3_value(&self) -> PhysAddr {
        self.root_pa()
    }

    /// Borrow the wrapped table as an inactive table.
    ///
    /// # Safety
    /// The caller must guarantee the table is not installed, i.e. not
    /// reachable by the MMU on any CPU, for the duration of the borrow.
    /// Inactive access forms `&mut` references into the page table page.
    pub unsafe fn as_inactive(&mut self) -> &mut InactivePageTable<A, P, L> {
        // SAFETY: The caller guarantees that the table is not installed, so forming a `&mut` reference to an inactive table is sound.
        unsafe { &mut *(self as *mut _ as *mut InactivePageTable<A, P, L>) }
    }

    /// Deallocates the root page of the page table.
    ///
    /// # Safety
    /// The caller must guarantee that the page table is inactive and no longer in use.
    pub unsafe fn dealloc(&mut self) {
        // SAFETY: GenericPageTable owns the root page, `root` was allocated by `alloc()`.
        unsafe { P::deallocate_physical_page(self.root_pa()) };
    }
}
