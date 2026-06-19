// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) Microsoft Corporation
//
// Author: Ziqiao Zhou <ziqiaozhou@microsoft.com>

//! TLB invalidation hooks and the [`MayNeedFlush`] obligation token.

use crate::address::VirtAddr;
use crate::sizes::PageSize;
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
/// `MayNeedFlush<A>` is a zero-sized type (it only carries a `PhantomData<A>`
/// tying it to the architecture's [`TlbOps`]). It deliberately carries no
/// address or page size: the caller already knows which page it asked to
/// change, so threading that information back would be redundant. The type's
/// only job is to force the caller to make a conscious decision about the TLB
/// — either
///
/// * discharge it by flushing through one of the `flush_*` methods (which
///   dispatch to the arch's [`TlbOps`] hooks), or
/// * discharge the obligation with [`MayNeedFlush::ignore`], **stating why**
///   no flush is needed.
///
/// `#[must_use]` is **not** a hard check. It does not guarantee a flush ever
/// happens and it is trivially silenced; it is purely a quick, zero-cost
/// compiler hint that nudges the developer to handle the obligation at the
/// call site. Treat it as a lint, not as a verified safety property.
///
/// Future verification: a tracked/ghost field can be threaded through here to
/// tie the obligation to a tracked TLB permission, turning "must flush before
/// the page is reused" into a checked property instead of a lint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "this page-table mutation may have invalidated a live TLB entry; \
flush the affected page or discharge the obligation with `.ignore(reason)`"]
pub struct MayNeedFlush<A: TlbOps>(pub(crate) PhantomData<A>);

impl<A: TlbOps> MayNeedFlush<A> {
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
    pub fn flush_tlb_global_sync_page(self, vaddr: VirtAddr, page_size: PageSize) {
        A::flush_tlb_global_sync_page(vaddr, page_size);
    }

    /// Discharge by flushing the `len`-byte region at `start`, mapped with
    /// `page_size` PTEs, on all CPUs.
    ///
    /// Dispatches to [`TlbOps::flush_tlb_global_sync_range`].
    pub fn flush_tlb_global_sync_range(self, start: VirtAddr, len: usize, page_size: PageSize) {
        A::flush_tlb_global_sync_range(start, len, page_size);
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
        A::flush_address_percpu(vaddr);
    }

    /// Discharge by flushing the `len`-byte region at `start`, mapped with
    /// `page_size` PTEs, on the **current CPU only** (no cross-CPU IPI).
    ///
    /// Dispatches to [`TlbOps::flush_range_percpu`]. Use only when the changed
    /// mappings cannot be live on any other CPU (e.g. a per-CPU scratch range
    /// being torn down).
    pub fn flush_range_percpu(self, start: VirtAddr, len: usize, page_size: PageSize) {
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
