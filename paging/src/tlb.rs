// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) Microsoft Corporation
//
// Author: Ziqiao Zhou <ziqiaozhou@microsoft.com>

//! TLB invalidation hooks and the [`MayNeedFlush`] obligation token.

use crate::address::VirtAddr;
use crate::sizes::PageSize;
use crate::traits::PageLevel;
use core::marker::PhantomData;

/// Architecture/OS hooks for TLB invalidation.
///
/// Split out from [`ArchPagingMeta`](crate::traits::ArchPagingMeta) (which
/// lists it as a supertrait) so the page-table code can discharge a
/// [`MayNeedFlush`] without touching address-encoding concerns. All methods
/// are stateless associated functions whose names mirror the kernel's
/// `cpu::tlb` free functions.
pub trait TlbOps {
    /// Flush the whole TLB on all CPUs. Coarsest discharge of a
    /// [`MayNeedFlush`]; prefer the scoped hooks below when the address is
    /// known.
    fn flush_tlb_global_sync();

    /// Flush the single `page_size` page at `vaddr` on all CPUs.
    fn flush_tlb_global_sync_page(vaddr: VirtAddr, page_size: PageSize);

    /// Flush the `len`-byte region at `start` (mapped with `page_size` PTEs)
    /// on all CPUs.
    fn flush_tlb_global_sync_range(start: VirtAddr, len: usize, page_size: PageSize);

    /// Flush the whole TLB, **including global pages**, on the current CPU
    /// only (no cross-CPU sync).
    fn flush_tlb_global_percpu();

    /// Flush the non-global TLB entries on the current CPU only (no global
    /// pages, no cross-CPU sync).
    fn flush_tlb_percpu();

    /// Flush the single page containing `vaddr` on the current CPU only.
    fn flush_address_percpu(vaddr: VirtAddr);

    /// Flush the `len`-byte region at `start` (mapped with `page_size` PTEs)
    /// on the current CPU only (no cross-CPU sync).
    fn flush_range_percpu(start: VirtAddr, len: usize, page_size: PageSize);
}

/// A `#[must_use]` marker meaning the caller *may* still owe a TLB
/// invalidation for the page it just modified.
///
/// `#[must_use]` is **not** a hard check. It does not guarantee a flush ever
/// happens and it is trivially silenced; it is purely a quick hint that nudges
/// the developer to handle the obligation at the call site. Treat it as a lint,
/// not as a verified safety property.
///
/// PageLevel: The level of the page that needs to be flushed.
/// Specifically, some updates may involve updates in parent page levels.
/// For example, set_encryption flags for 4k page may require flushing the
/// corresponding 2M page if the target 4k page was in a huge page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "this page-table mutation may have invalidated a live TLB entry; \
flush the affected page or discharge the obligation with `.ignore(reason)`"]
pub struct MayNeedFlush<A: TlbOps>(pub(crate) PageLevel, pub(crate) PhantomData<A>);

impl<A: TlbOps> MayNeedFlush<A> {
    /// Creates a new obligation token for a mutation observed at `level`.
    ///
    /// Most callers get their token back from a `paging`-crate mutation
    /// helper (e.g. `unmap_4k`, `set_shared_4k`) that already captured the
    /// level a walk stopped at. Use this constructor only when a caller
    /// outside of `paging` performs its own low-level page-table edit (e.g.
    /// looping over a region and writing entries directly) and must
    /// therefore vouch for the level itself.
    pub fn new(level: PageLevel) -> Self {
        MayNeedFlush(level, PhantomData)
    }

    fn check_page_size(&self, page_size: PageSize) -> bool {
        match page_size {
            PageSize::Regular => self.0 == PageLevel::Level0,
            PageSize::Huge => self.0 <= PageLevel::Level1,
        }
    }
    /// Discharge by flushing the whole TLB on all CPUs (coarsest option).
    ///
    /// Dispatches to [`TlbOps::flush_tlb_global_sync`].
    pub fn flush_tlb_global_sync(self) {
        A::flush_tlb_global_sync();
    }

    /// Discharge by flushing the single `page_size` page at `vaddr` on all
    /// CPUs.
    ///
    /// Dispatches to [`TlbOps::flush_tlb_global_sync_page`].
    fn flush_tlb_global_sync_page(self, vaddr: VirtAddr, page_size: PageSize) {
        assert!(self.check_page_size(page_size));
        A::flush_tlb_global_sync_page(vaddr, page_size);
    }

    /// Discharge by flushing the `len`-byte region at `start`, mapped with
    /// `page_size` PTEs, on all CPUs.
    ///
    /// Dispatches to [`TlbOps::flush_tlb_global_sync_range`].
    pub fn flush_tlb_global_sync_range(self, start: VirtAddr, len: usize, page_size: PageSize) {
        assert!(self.check_page_size(page_size));
        A::flush_tlb_global_sync_range(start, len, page_size);
    }

    /// Discharge by flushing the single `page_size` page at `vaddr` on all
    /// CPUs.
    ///
    /// Dispatches to [`TlbOps::flush_tlb_global_sync_page`] if the page size is compatible with the
    /// level of the mutation that caused this obligation.
    /// Otherwise, returns the obligation back to the caller for further handling.
    pub fn try_flush_tlb_global_sync_page(
        self,
        vaddr: VirtAddr,
        page_size: PageSize,
    ) -> Result<(), Self> {
        if self.check_page_size(page_size) {
            self.flush_tlb_global_sync_page(vaddr, page_size);
            Ok(())
        } else {
            Err(self)
        }
    }

    /// Discharge by flushing the whole TLB (including global pages) on the
    /// **current CPU only**, without a cross-CPU IPI.
    ///
    /// Dispatches to [`TlbOps::flush_tlb_global_percpu`]. Use only when the
    /// changed mapping cannot be live on any other CPU.
    pub fn flush_tlb_global_percpu(self) {
        A::flush_tlb_global_percpu();
    }

    /// Discharge by flushing the non-global TLB entries on the **current CPU
    /// only**.
    ///
    /// Dispatches to [`TlbOps::flush_tlb_percpu`]. Does **not** evict
    /// global-page entries; only sound when the changed mapping is non-global
    /// and local to this CPU.
    pub fn flush_tlb_percpu(self) {
        A::flush_tlb_percpu();
    }

    /// Discharge by flushing the single page containing `vaddr` on the
    /// **current CPU only**.
    ///
    /// Dispatches to [`TlbOps::flush_address_percpu`].
    pub fn flush_address_percpu(self, vaddr: VirtAddr) {
        self.check_page_size(PageSize::Regular);
        A::flush_address_percpu(vaddr);
    }

    /// Discharge by flushing the `len`-byte region at `start`, mapped with
    /// `page_size` PTEs, on the **current CPU only** (no cross-CPU IPI).
    ///
    /// Dispatches to [`TlbOps::flush_range_percpu`]. Use only when the changed
    /// mappings cannot be live on any other CPU (e.g. a per-CPU scratch range
    /// being torn down).
    pub fn flush_range_percpu(self, start: VirtAddr, len: usize, page_size: PageSize) {
        self.check_page_size(page_size);
        A::flush_range_percpu(start, len, page_size);
    }

    /// Discharge the flush obligation **without** flushing, recording why a
    /// flush is unnecessary here.
    ///
    /// The `reason` is neither stored nor checked at runtime; requiring it
    /// simply forces every "no flush needed" decision to be written down and
    /// auditable at the call site. Use only when the caller knows a flush is
    /// genuinely unnecessary (e.g. a pure split that republishes identical
    /// translations) or is performed elsewhere (e.g. a broader flush issued
    /// by the surrounding code).
    pub fn ignore(self, reason: &'static str) {
        let _ = reason;
    }
}
