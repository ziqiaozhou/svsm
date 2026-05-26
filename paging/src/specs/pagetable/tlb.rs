use vstd::prelude::*;
use crate::specs::pagetable::cpu::CpuIdentifier;

verus!{


pub struct TLBEntryView<VirtAddr, Entry> {
    addr: VirtAddr,
    entry: Entry,
}

#[verifier::external_body]
#[verifier::reject_recursive_types(VirtAddr)]
#[verifier::reject_recursive_types(Entry)]
pub tracked struct TLBEntry<VirtAddr, Entry> {
    phantom: core::marker::PhantomData<(VirtAddr, Entry)>,
    no_copy: NoCopy,
}

impl<VirtAddr, Entry> View for TLBEntry<VirtAddr, Entry> {
    type V = TLBEntryView<VirtAddr, Entry>;
    uninterp spec fn view(&self) -> TLBEntryView<VirtAddr, Entry>;
}

/// Represents the state of the TLB.
/// Conceptually, the TLB is a cache of the most recently used page table
/// entries, where the PFN is same but the other fields are the effective
/// permissions by considering parent and leaf PTEs instead of the exact one in
/// leaf PTE.
#[verifier::reject_recursive_types(VirtAddr)]
#[verifier::reject_recursive_types(Entry)]
struct TLBState<VirtAddr, Entry> {
    entries: Map<CpuIdentifier, Map<VirtAddr, TLBEntry<VirtAddr, Entry>>>,
}
}