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
use crate::sizes::{PAGE_SHIFT, PAGE_SIZE, PAGE_SIZE_1G, PAGE_SIZE_2M};
pub use crate::tlb::{MayNeedFlush, TlbOps};
pub use crate::traits::{
    ArchPagingMeta, GenericPageTableFlags, PageLevel, PagingError, PagingHandler, PagingLevel,
    PagingLevel2, PagingLevel3, SelfMap,
};
use bitflags::Flags;
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use zerocopy::{FromBytes, FromZeros};

/// Number of virtual-address bits indexed by a single page-table level.
/// Assume 8-byte page table entries and page granularity = 1 << PAGE_SHIFT
pub(crate) const PTE_SHIFT: usize = PAGE_SHIFT - 3;

/// Number of entries in a page table (4KB/8B).
pub(crate) const ENTRY_COUNT: usize = 1 << PTE_SHIFT;

const fn virt_from_lvl_idx(idx: usize, level: PageLevel) -> VirtAddr {
    VirtAddr::new(idx << ((level as usize * PTE_SHIFT) + PAGE_SHIFT))
}

const _: () = assert!(
    core::mem::size_of::<PhysAddr>() == 8,
    "Only supports 8 bytes PTE entry",
);

/// Represents a page table entry.
#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes)]
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

    /// Inserts the private address mask if the page is present.
    pub fn make_private_if_present(&mut self) {
        if self.flags().contains(A::PTFlags::PRESENT) {
            self.entry = A::make_private_address(self.entry);
        }
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
    _phantom: PhantomData<P>,
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
    unsafe { entry.read_volatile() }
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
        // SAFETY: `allocate_physical_page` returns a unique, zeroed frame that
        // is not yet installed in any page table, satisfying `from_vaddr`'s
        // uniquely-owned contract.
        let page = unsafe { Self::from_vaddr(vaddr) };
        Ok((page, paddr))
    }

    /// Returns a raw pointer to the child [`PTPage`] referenced by `entry`,
    /// or `None` if the entry is absent or a huge mapping.
    fn from_entry(entry: PTEntry<A>) -> Option<*mut Self> {
        if !entry.present() || entry.huge() {
            return None;
        }

        let address = P::paddr_to_vaddr(entry.address());
        Some(address.as_mut_ptr::<Self>())
    }

    /// `*const` pointer to the entry at `idx` within `page`.
    ///
    /// Pure pointer arithmetic: computing the pointer is always sound
    /// (only *dereferencing* an out-of-range pointer is UB, which is the
    /// contract of [`read_entry`]). `entries` is the first field of
    /// `#[repr(C)]` [`PTPage`], so `page` and `&entries[0]` share an
    /// address.
    fn entry_ptr_const(page: *const Self, idx: usize) -> *const PTEntry<A> {
        (page as *const PTEntry<A>).wrapping_add(idx)
    }

    /// `*mut` pointer to the entry at `idx` within `page`.
    ///
    /// See [`entry_ptr_const`](Self::entry_ptr_const); identical arithmetic
    /// for a mutable page pointer.
    pub fn entry_ptr(page: *mut Self, idx: usize) -> *mut PTEntry<A> {
        (page as *mut PTEntry<A>).wrapping_add(idx)
    }

    /// Forms a `&mut` to the [`PTPage`] living at `vaddr`.
    ///
    /// # Safety
    /// `vaddr` must map a valid page-table frame that is **not installed** in
    /// any live page-table hierarchy — i.e. uniquely owned by the caller (e.g.
    /// freshly allocated and not yet published into a parent entry). Forming a
    /// `&mut` to an *installed* table is UB, because the hardware page-table
    /// walker concurrently reads/writes it (accessed/dirty bits). Callers
    /// touching a live table must instead use the volatile element accessors
    /// [`read_entry_at`](Self::read_entry_at) /
    /// [`write_entry_at`](Self::write_entry_at) on a raw pointer.
    pub unsafe fn from_vaddr(vaddr: VirtAddr) -> &'static mut Self {
        // SAFETY: the caller guarantees the frame is uniquely owned and not
        // yet installed, so the exclusive reference is sound.
        unsafe { &mut *vaddr.as_mut_ptr::<Self>() }
    }

    /// Volatile read of the entry at `idx` within the (possibly live) page
    /// `page`.
    ///
    /// # Safety
    /// `page` must point to a valid [`PTPage`] and `idx < ENTRY_COUNT`.
    pub unsafe fn read_entry_at(page: *const Self, idx: usize) -> PTEntry<A> {
        // SAFETY: delegated to the caller; the volatile read forms no
        // reference into the frame.
        unsafe { read_entry(Self::entry_ptr_const(page, idx)) }
    }

    /// Volatile write of `e` to the entry at `idx` within the (possibly live)
    /// page `page`.
    ///
    /// # Safety
    /// `page` must point to a valid [`PTPage`] and `idx < ENTRY_COUNT`.
    pub unsafe fn write_entry_at(page: *mut Self, idx: usize, e: PTEntry<A>) {
        // SAFETY: delegated to the caller; the volatile write forms no
        // reference into the frame.
        unsafe { write_entry(Self::entry_ptr(page, idx), e) }
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

/// The resolved result of a page-table walk: the level at which the walk
/// terminated, together with a *copy* of the page-table entry found there.
///
/// Unlike a raw `&mut PTEntry`, a `Mapping` carries no aliasing borrow into
/// the live page table, so it is safe to inspect after the walk completes.
/// Mutations must go through [`GenericPageTable::walk_update_with`].
#[derive(Clone, Copy, Debug)]
pub struct Mapping<P: ArchPagingMeta> {
    pub level: PageLevel,
    pub entry: PTEntry<P>,
}

impl<P: ArchPagingMeta> Mapping<P> {
    /// Construct a `Mapping` at the given level.
    pub fn new(level: PageLevel, entry: PTEntry<P>) -> Self {
        Self { level, entry }
    }
}

/// Per-level decision returned by the [`GenericPageTable::walk_with`]
/// closure.
///
/// The closure is consulted at each level to decide whether the walk
/// should descend further or stop. Either way `walk_with` returns the
/// [`Mapping`] (level + entry copy) at which the walk terminated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkAction {
    /// Stop at the current level; `walk_with` returns this `(level, entry)`.
    Stop,
    /// Descend into the sub-table referenced by the current entry.
    ///
    /// If the entry is absent or huge, or the level is already
    /// [`PageLevel::Level0`], descent is impossible and the walk stops,
    /// returning the current `(level, entry)`.
    Descend,
}

/// Outcome of a single step of [`GenericPageTable::walk_update_with`].
///
/// Like [`WalkAction`] but the closure may also request an in-place
/// replacement of the visited PTE.
#[derive(Debug)]
pub enum WalkUpdate<A: ArchPagingMeta, R> {
    /// Terminate the walk without modifying the entry; yield `R`.
    Stop(R),
    /// Replace the current entry with `new`, terminate, and yield `R`.
    Replace(PTEntry<A>, R),
    /// Descend into the sub-table referenced by the current entry.
    /// Same conditions as [`WalkAction::Descend`].
    Descend,
    /// Step into the next-lower level, allocating a fresh sub-table with
    /// the supplied parent flags if the current entry is absent.
    ///
    /// Fails the walk (yields `None`) if the current entry is a huge
    /// mapping, if the current level is already `Level0`, or if the
    /// sub-table allocation fails.
    DescendOrAlloc(A::PTFlags),
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

/// Methods for page table hierarchies, independent of the self-map.
impl<A: ArchPagingMeta, P: PagingHandler, L: PagingLevel> GenericPageTable<A, P, L> {
    /// Recursively free all page table pages starting from the root page.
    ///
    /// # Safety
    /// `page` must point to a valid [`PTPage`] at `level` that is no longer
    /// installed in any CR3 and not concurrently accessed (teardown only).
    unsafe fn free_lvl(page: *const PTPage<A, P>, level: PageLevel) {
        if level <= PageLevel::Level0 {
            return;
        }
        for idx in 0..ENTRY_COUNT {
            // SAFETY: `idx < ENTRY_COUNT` and `page` is valid per the
            // function contract.
            let entry = unsafe { read_entry(PTPage::<A, P>::entry_ptr_const(page, idx)) };
            if let Some(child) = PTPage::<A, P>::from_entry(entry) {
                if let Some(next) = level.next_down() {
                    // SAFETY: `child` is a present, non-huge sub-table one
                    // level below `level`.
                    unsafe { Self::free_lvl(child, next) };
                }
                let paddr = entry.address();
                // SAFETY: the page was allocated via PagingHandler::allocate_physical_page.
                unsafe { P::deallocate_physical_page(paddr) };
            }
        }
    }

    /// Free all page table pages starting from the root page.
    /// We do not set it as a destructor because a page table may contain
    /// shared sub-trees that may be in use by other root table.
    ///
    /// # Safety
    /// The caller must ensure that the page table is not in use by any other thread.
    pub fn free(&self) {
        // SAFETY: per this function's documented contract the table is no
        // longer in use; `&raw const self.root` is a valid pointer to the
        // owned root frame.
        unsafe { Self::free_lvl(&raw const self.root, L::TOP_LEVEL) };
    }

    /// Get a copy of the entry at the specified index.
    pub fn entry(&mut self, idx: usize) -> PTEntry<A> {
        self.root.entries[idx]
    }

    /// Set the entry at the specified index.
    ///
    /// Returns `true` if the entry was updated, `false` otherwise.
    pub fn set_entry(&mut self, idx: usize, addr: PhysAddr, flags: A::PTFlags) -> bool {
        let old_entry = self.root.entries[idx];
        self.root.entries[idx].set(addr, flags);
        old_entry.raw() != self.root.entries[idx].raw()
    }

    /// Copy an entry at `entry` from another `GenericPageTable`.
    pub fn copy_entry(&mut self, other: &Self, entry: usize) {
        self.root.entries[entry] = other.root.entries[entry];
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

    /// Walk the page table for `vaddr` and return a copy of the deepest
    /// existing entry, together with the level at which it sits.
    ///
    /// This is the read-only replacement for the old `walk_addr`: it
    /// follows present, non-huge sub-tables downward and stops at the
    /// first entry that is absent, huge, or at `Level0`. The returned
    /// [`Mapping`] holds the entry *by value*, so no mutable borrow into
    /// the live page table escapes. Use
    /// [`walk_update_with`](Self::walk_update_with) to mutate entries.
    ///
    /// # Returns
    /// A [`Mapping`] describing where the walk terminated.
    pub fn translate(&self, vaddr: VirtAddr) -> Mapping<A> {
        // Always request `Descend`; `walk_with` stops on its own at the
        // first absent/huge/Level0 entry and reports that level.
        self.walk_with(vaddr, |_level, _entry| WalkAction::Descend)
    }

    /// Computes the page-table index for `vaddr` at the given runtime level.
    ///
    /// Dispatches to the const-generic [`VirtAddr::to_pgtbl_idx`] so the
    /// underlying shift remains constant-folded per level.
    fn pgtbl_idx_for(level: PageLevel, vaddr: VirtAddr) -> usize {
        match level {
            PageLevel::Level0 => vaddr.to_pgtbl_idx::<0>(),
            PageLevel::Level1 => vaddr.to_pgtbl_idx::<1>(),
            PageLevel::Level2 => vaddr.to_pgtbl_idx::<2>(),
            PageLevel::Level3 => vaddr.to_pgtbl_idx::<3>(),
        }
    }

    /// Recursive driver for [`walk_with`](Self::walk_with).
    ///
    /// # Safety
    /// `page` must point to a valid [`PTPage`] at `level`.
    unsafe fn walk_with_inner(
        page: *const PTPage<A, P>,
        level: PageLevel,
        vaddr: VirtAddr,
        f: &mut impl FnMut(PageLevel, PTEntry<A>) -> WalkAction,
    ) -> Mapping<A> {
        let idx = Self::pgtbl_idx_for(level, vaddr);
        // SAFETY: `idx < ENTRY_COUNT` and `page` is valid per contract.
        let entry = unsafe { read_entry(PTPage::<A, P>::entry_ptr_const(page, idx)) };
        match f(level, entry) {
            WalkAction::Stop => Mapping::new(level, entry),
            WalkAction::Descend => {
                // Descend only if the entry is a present, non-huge table
                // pointer and we are not already at Level0; otherwise the
                // current entry is the terminal mapping.
                match (level.next_down(), PTPage::<A, P>::from_entry(entry)) {
                    (Some(next_level), Some(child)) => {
                        // SAFETY: `child` is a valid sub-table one level down.
                        unsafe { Self::walk_with_inner(child, next_level, vaddr, f) }
                    }
                    _ => Mapping::new(level, entry),
                }
            }
        }
    }

    /// Walk the page table for `vaddr`, letting the closure decide at each
    /// level whether to stop or keep descending.
    ///
    /// At every visited level, the closure receives the current
    /// [`PageLevel`] and a *copy* of the [`PTEntry`] at the corresponding
    /// index, and returns a [`WalkAction`]. No mutable reference into the
    /// table ever escapes.
    ///
    /// The walk terminates — and returns the current `(level, entry)` as a
    /// [`Mapping`] — when either:
    /// * the closure returns [`WalkAction::Stop`], or
    /// * the closure returns [`WalkAction::Descend`] but descent is
    ///   impossible because the entry is absent or huge, or the level is
    ///   already [`PageLevel::Level0`].
    ///
    /// Because the terminal level is always reported, the closure does not
    /// need to detect leaf/huge/absent entries itself — it can simply
    /// request `Descend` and inspect the returned level.
    ///
    /// # Example
    /// ```ignore
    /// // Resolve the deepest existing entry for `vaddr`.
    /// let mapping = table.walk_with(vaddr, |level, entry| {
    ///     if level != PageLevel::Level0 && entry.present() && !entry.huge() {
    ///         WalkAction::Descend
    ///     } else {
    ///         WalkAction::Stop
    ///     }
    /// });
    /// ```
    pub fn walk_with(
        &self,
        vaddr: VirtAddr,
        mut f: impl FnMut(PageLevel, PTEntry<A>) -> WalkAction,
    ) -> Mapping<A> {
        // SAFETY: `&raw const self.root` is a valid pointer to the owned
        // root frame.
        unsafe { Self::walk_with_inner(&raw const self.root, L::TOP_LEVEL, vaddr, &mut f) }
    }

    /// Recursive driver for [`walk_update_with`](Self::walk_update_with).
    ///
    /// # Safety
    /// `page` must point to a valid [`PTPage`] at `level`.
    unsafe fn walk_update_with_inner<R>(
        page: *mut PTPage<A, P>,
        level: PageLevel,
        vaddr: VirtAddr,
        f: &mut impl FnMut(PageLevel, PTEntry<A>) -> WalkUpdate<A, R>,
    ) -> (Option<R>, MayNeedFlush<A>) {
        let idx = Self::pgtbl_idx_for(level, vaddr);
        let entry_ptr = PTPage::<A, P>::entry_ptr(page, idx);
        // SAFETY: `entry_ptr` is valid per the above.
        let entry = unsafe { read_entry(entry_ptr) };
        match f(level, entry) {
            WalkUpdate::Stop(r) => (Some(r), MayNeedFlush(PhantomData)),
            WalkUpdate::Replace(new, r) => {
                // SAFETY: as above.
                unsafe { write_entry(entry_ptr, new) };
                // The caller always receives a `MayNeedFlush`. A fresh
                // (not-present -> present) mapping technically owes no flush,
                // but the zero-sized token cannot encode that distinction, so
                // the caller decides and justifies skipping via
                // `.ignore(reason)`.
                (Some(r), MayNeedFlush(PhantomData))
            }
            WalkUpdate::Descend => {
                let Some(next_level) = level.next_down() else {
                    return (None, MayNeedFlush(PhantomData));
                };
                let Some(child) = PTPage::<A, P>::from_entry(entry) else {
                    return (None, MayNeedFlush(PhantomData));
                };
                // SAFETY: `child` is a valid sub-table one level down.
                unsafe { Self::walk_update_with_inner(child, next_level, vaddr, f) }
            }
            WalkUpdate::DescendOrAlloc(parent_flags) => {
                let Some(next_level) = level.next_down() else {
                    return (None, MayNeedFlush(PhantomData));
                };
                // A huge mapping cannot be descended into.
                if entry.huge() {
                    return (None, MayNeedFlush(PhantomData));
                }
                // Allocate a sub-table if the slot is empty.
                if !entry.present() {
                    let Ok((_page, paddr)) = PTPage::<A, P>::alloc() else {
                        return (None, MayNeedFlush(PhantomData));
                    };
                    let mut new = entry;
                    new.set(A::make_private_address(paddr), parent_flags);
                    // SAFETY: as above. Writing a fresh parent entry over an
                    // absent slot needs no flush.
                    unsafe { write_entry(entry_ptr, new) };
                }
                // SAFETY: re-read the (possibly newly allocated) entry.
                let entry = unsafe { read_entry(entry_ptr) };
                let Some(child) = PTPage::<A, P>::from_entry(entry) else {
                    return (None, MayNeedFlush(PhantomData));
                };
                // SAFETY: `child` is a valid sub-table one level down.
                unsafe { Self::walk_update_with_inner(child, next_level, vaddr, f) }
            }
        }
    }

    /// Walk the page table for `vaddr`, letting the closure decide at each
    /// level whether to stop (optionally replacing the entry) or keep
    /// descending.
    ///
    /// Same control-flow contract as [`walk_with`](Self::walk_with), but
    /// the closure may also request an in-place replacement of the visited
    /// entry without ever holding a mutable reference into the table.
    ///
    /// The closure returns a [`WalkUpdate`]:
    /// * [`WalkUpdate::Stop(r)`](WalkUpdate::Stop) — terminate without
    ///   modifying the entry; return `r`.
    /// * [`WalkUpdate::Replace(new, r)`](WalkUpdate::Replace) — write
    ///   `new` to the current entry and terminate; return `r`. The caller
    ///   is responsible for invariant preservation (e.g. not orphaning a
    ///   present sub-table by overwriting it with a leaf, not setting
    ///   `HUGE` at `Level0`, etc.).
    /// * [`WalkUpdate::Descend`] — step into the next-lower level. Same
    ///   conditions as [`WalkAction::Descend`].
    /// * [`WalkUpdate::DescendOrAlloc(flags)`](WalkUpdate::DescendOrAlloc)
    ///   — descend, allocating a fresh sub-table (with `flags` as the
    ///   parent-entry flags) if the slot is empty.
    ///
    /// # Returns
    /// A pair `(Option<R>, MayNeedFlush)`:
    /// * `Some(r)` — the closure stopped (with or without replacing);
    ///   `None` — descent was requested where it is impossible, or a
    ///   `DescendOrAlloc` allocation failed.
    /// * a [`MayNeedFlush`] obligation. It is `#[must_use]` (a hint only, not
    ///   an enforced flush). Either flush the page covering `vaddr` or
    ///   discharge it with [`ignore`](MayNeedFlush::ignore), stating why.
    pub fn walk_update_with<R>(
        &mut self,
        vaddr: VirtAddr,
        mut f: impl FnMut(PageLevel, PTEntry<A>) -> WalkUpdate<A, R>,
    ) -> (Option<R>, MayNeedFlush<A>) {
        // SAFETY: `&raw mut self.root` is a valid pointer to the owned root
        // frame; `&mut self` guarantees exclusive software access.
        unsafe { Self::walk_update_with_inner(&raw mut self.root, L::TOP_LEVEL, vaddr, &mut f) }
    }
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
        let entry = &mut self.root[P::SELFMAP_IDX];
        let flags = A::PTFlags::self_map_table_flags();
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
        let pml4e = unsafe { PTEntry::<A>::read_pte(pml4e_addr) };
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
        let pdpe = unsafe { PTEntry::<A>::read_pte(pdpe_addr) };
        if !pdpe.present() {
            return None;
        }
        if pdpe.huge() {
            let pa = pdpe.page_frame() + (usize::from(vaddr) & 0x3FFF_FFFF);
            return Some(PageFrame::Size1G(pa, PhantomData));
        }

        // SAFETY: The PDPE was checked to be present and not to be a huge
        // page. So the PDE exists and can be read safely.
        let pde = unsafe { PTEntry::<A>::read_pte(pde_addr) };
        if !pde.present() {
            return None;
        }
        if pde.huge() {
            let pa = pde.page_frame() + (usize::from(vaddr) & 0x001F_FFFF);
            return Some(PageFrame::Size2M(pa));
        }

        // SAFETY: The PDE was checked to be present and not to be a huge
        // page. So the PTE exists and can be read safely.
        let pte = unsafe { PTEntry::<A>::read_pte(pte_addr) };
        if pte.present() {
            let pa = pte.page_frame() + (usize::from(vaddr) & 0xFFF);
            Some(PageFrame::Size4K(pa))
        } else {
            None
        }
    }
}

impl<A: ArchPagingMeta, P: PagingHandler, L: PagingLevel> GenericPageTable<A, P, L> {
    /// Splits a 2 MiB huge mapping into 4 KiB pages, mutating `entry` in
    /// place to point at the freshly built leaf table.
    ///
    /// The huge mapping described by `entry` is expanded into 512 leaf
    /// entries covering the same 2 MiB region with identical flags (minus
    /// `HUGE`). On success `entry` is rewritten to the non-huge table
    /// pointer for the new leaf page.
    ///
    /// # Returns
    /// `Ok(())` on success, or [`PagingError`] if the leaf page could not
    /// be allocated.
    fn do_split_4k(entry: &mut PTEntry<A>) -> Result<(), PagingError> {
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
        Ok(())
    }

    /// Splits the 2 MiB mapping for `vaddr` into 4 KiB pages, if needed.
    ///
    /// If `vaddr` is already mapped at 4 KiB granularity this is a no-op.
    /// If it is mapped by a 2 MiB huge entry, a leaf page table is
    /// allocated, populated, and installed in place of the huge entry.
    ///
    /// # Returns
    /// On success, a [`MayNeedFlush`] obligation. A pure split publishes
    /// identical translations, so it never invalidates a live TLB entry;
    /// callers normally discharge this with
    /// [`MayNeedFlush::ignore`]. It is `#[must_use]` as a hint only — it does
    /// not force a flush.
    pub fn split_4k(&mut self, vaddr: VirtAddr) -> Result<MayNeedFlush<A>, PagingError> {
        // Single walk: classify and (if needed) split in one descent.
        let (outcome, flush) = self.walk_update_with(vaddr, |level, entry| match level {
            // Already mapped at 4 KiB granularity: nothing to do.
            PageLevel::Level0 => WalkUpdate::Stop(Ok(())),
            // 2 MiB huge mapping: build a leaf table and install it.
            PageLevel::Level1 if entry.huge() => {
                // Work on a copy, then publish it atomically via `Replace`
                // so no `&mut` into the live table escapes.
                let mut new = entry;
                match Self::do_split_4k(&mut new) {
                    Ok(()) => WalkUpdate::Replace(new, Ok(())),
                    Err(e) => WalkUpdate::Stop(Err(e)),
                }
            }
            // A huge mapping above Level1 cannot be split to 4 KiB here.
            _ if entry.huge() => WalkUpdate::Stop(Err(PagingError::NotMapped)),
            // Present table pointer: descend toward the leaf.
            _ => WalkUpdate::Descend,
        });

        outcome.unwrap_or(Err(PagingError::NotMapped))?;
        // The token already carries the split's 2 MiB range (or is empty on
        // a no-op); hand the flush decision to the caller instead of forcing
        // a global flush here.
        Ok(flush)
    }

    /// Sets the shared state for a 4KB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address of the page.
    ///
    /// # Returns
    /// On success, a [`MayNeedFlush`] obligation for the leaf page covering
    /// `vaddr`: changing the C-bit of a present mapping invalidates a live
    /// translation, so the caller must flush it. Returns [`PagingError`] if
    /// the operation fails.
    pub fn set_shared_4k(&mut self, vaddr: VirtAddr) -> Result<MayNeedFlush<A>, PagingError> {
        // A pure split publishes identical translations, so it owes no
        // flush; only the leaf C-bit change below does.
        self.split_4k(vaddr)?
            .ignore("pure split republishes identical translations");
        let (outcome, flush) = self.walk_update_with(vaddr, |level, entry| match level {
            PageLevel::Level0 => {
                let mut new = entry;
                // entry.address() is returned with the C-bit cleared already.
                new.set(A::make_shared_address(entry.address()), entry.flags());
                WalkUpdate::Replace(new, Ok(()))
            }
            _ if entry.present() && !entry.huge() => WalkUpdate::Descend,
            _ => WalkUpdate::Stop(Err(PagingError::NotMapped)),
        });
        outcome.unwrap_or(Err(PagingError::NotMapped))?;
        Ok(flush)
    }

    /// Sets the encryption state for a 4KB page.
    ///
    /// # Parameters
    /// - `vaddr`: The virtual address of the page.
    ///
    /// # Returns
    /// On success, a [`MayNeedFlush`] obligation for the leaf page covering
    /// `vaddr`: changing the C-bit of a present mapping invalidates a live
    /// translation, so the caller must flush it. Returns [`PagingError`] if
    /// the operation fails.
    pub fn set_encrypted_4k(&mut self, vaddr: VirtAddr) -> Result<MayNeedFlush<A>, PagingError> {
        // A pure split publishes identical translations, so it owes no
        // flush; only the leaf C-bit change below does.
        self.split_4k(vaddr)?
            .ignore("pure split republishes identical translations");
        let (outcome, flush) = self.walk_update_with(vaddr, |level, entry| match level {
            PageLevel::Level0 => {
                let mut new = entry;
                // entry.address() is returned with the C-bit cleared already.
                new.set(A::make_private_address(entry.address()), entry.flags());
                WalkUpdate::Replace(new, Ok(()))
            }
            _ if entry.present() && !entry.huge() => WalkUpdate::Descend,
            _ => WalkUpdate::Stop(Err(PagingError::NotMapped)),
        });
        outcome.unwrap_or(Err(PagingError::NotMapped))?;
        Ok(flush)
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
        assert!(vaddr.is_aligned(PAGE_SIZE_2M));
        assert!(paddr.is_aligned(PAGE_SIZE_2M));
        let addr = if !shared {
            A::make_private_address(paddr)
        } else {
            A::make_shared_address(paddr)
        };

        let (outcome, flush) = self.walk_update_with(vaddr, |level, entry| match level {
            PageLevel::Level1 => {
                // A present, non-huge entry means the 2 MiB region is
                // already split into 4 KiB pages; refuse to clobber it.
                if entry.present() && !entry.huge() {
                    WalkUpdate::Stop(Err(PagingError::AllocFrame))
                } else {
                    let mut new = entry;
                    new.set(addr, flags | A::PTFlags::HUGE);
                    WalkUpdate::Replace(new, Ok(()))
                }
            }
            // A huge mapping above Level1 (e.g. a 1 GiB page) blocks the
            // 2 MiB mapping.
            _ if entry.present() && entry.huge() => WalkUpdate::Stop(Err(PagingError::AllocFrame)),
            _ => WalkUpdate::DescendOrAlloc(parent_flags),
        });
        // A map assumes the VA was previously unmapped, so it only publishes a
        // fresh translation and never invalidates a live TLB entry (x86 has no
        // negative caching); discharge the obligation here rather than
        // threading it to the caller.
        flush.ignore("map_* assumes a previously-unmapped VA, so no flush is needed");
        outcome.unwrap_or(Err(PagingError::AllocFrame))
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
    /// # Returns
    /// The previous [`PTEntry`] (if the page was mapped) together with a
    /// [`MayNeedFlush`] obligation the caller must discharge.
    ///
    /// # Panics
    /// Panics if `vaddr` is not aligned to a 2MB boundary.
    pub fn unmap_2m(&mut self, vaddr: VirtAddr) -> (Option<PTEntry<A>>, MayNeedFlush<A>) {
        assert!(vaddr.is_aligned(PAGE_SIZE_2M));

        let (outcome, flush) = self.walk_update_with(vaddr, |level, entry| match level {
            PageLevel::Level0 => unreachable!(),
            PageLevel::Level1 => {
                let mut cleared = entry;
                cleared.clear();
                WalkUpdate::Replace(cleared, Some(entry))
            }
            _ if entry.present() && !entry.huge() => WalkUpdate::Descend,
            _ => {
                assert!(!entry.present());
                WalkUpdate::Stop(None)
            }
        });
        (outcome.flatten(), flush)
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
        let addr = if !shared {
            A::make_private_address(paddr)
        } else {
            A::make_shared_address(paddr)
        };

        let (outcome, flush) = self.walk_update_with(vaddr, |level, entry| match level {
            PageLevel::Level0 => {
                let mut new = entry;
                new.set(addr, flags);
                WalkUpdate::Replace(new, Ok(()))
            }
            // A huge mapping (2 MiB at Level1, 1 GiB at Level2) blocks the
            // 4 KiB mapping.
            _ if entry.present() && entry.huge() => WalkUpdate::Stop(Err(PagingError::AllocFrame)),
            _ => WalkUpdate::DescendOrAlloc(parent_flags),
        });
        // A map assumes the VA was previously unmapped, so it only publishes a
        // fresh translation and never invalidates a live TLB entry (x86 has no
        // negative caching); discharge the obligation here rather than
        // threading it to the caller.
        flush.ignore("map_* assumes a previously-unmapped VA, so no flush is needed");
        outcome.unwrap_or(Err(PagingError::AllocFrame))
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
    ///
    /// # Returns
    /// The previous [`PTEntry`] (if the page was mapped) together with a
    /// [`MayNeedFlush`] obligation the caller must discharge.
    pub fn unmap_4k(&mut self, vaddr: VirtAddr) -> (Option<PTEntry<A>>, MayNeedFlush<A>) {
        let (outcome, flush) = self.walk_update_with(vaddr, |level, entry| match level {
            PageLevel::Level0 => {
                let mut cleared = entry;
                cleared.clear();
                WalkUpdate::Replace(cleared, Some(entry))
            }
            _ if entry.present() && !entry.huge() => WalkUpdate::Descend,
            _ => {
                assert!(!entry.present());
                WalkUpdate::Stop(None)
            }
        });
        (outcome.flatten(), flush)
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
        let Mapping { level, entry } = self.translate(vaddr);
        match level {
            PageLevel::Level0 if entry.present() => Ok(entry.address() + vaddr.page_offset()),
            PageLevel::Level1 if entry.present() && entry.huge() => {
                let offset = vaddr.bits() & (PAGE_SIZE_2M - 1);
                Ok(entry.address() + offset)
            }
            _ => Err(PagingError::NotMapped),
        }
    }
}
