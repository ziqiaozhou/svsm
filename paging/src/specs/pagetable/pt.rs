use vstd::prelude::*;

verus!{
/// Represents the state of a page table.
enum PTState {
    Installed, // The page table is installed in CPUs
    NotInstalled, // The page table is not installed in any CPU
}

pub(super) ghost struct EntryPropertiesSpec {
    valid: bool,
    writable: bool,
    user: bool,
}

pub(super) trait EntrySpec {
    type PhyAddr;
    spec fn address(&self) -> Self::PhyAddr;
    spec fn properties(&self) -> EntryPropertiesSpec;

    spec fn get_effective_properties(&self, parent: &Self) -> EntryPropertiesSpec;
}

/// Represents the tracked state of a memory permission for page table entry.
pub struct PTPermView<Entry> {
    value: Entry,
    state: PTState,
}

#[verifier::external_body]
#[verifier::reject_recursive_types(Entry)]
pub tracked struct PTEntry<Entry> {
    phantom: core::marker::PhantomData<Entry>,
    no_copy: NoCopy,
}

impl<Entry> View for PTEntry<Entry> {
    type V = PTPermView<Entry>;
    uninterp spec fn view(&self) -> PTPermView<Entry>;
}

/// A flatten view of a page table tree.
#[verifier::reject_recursive_types(VirtAddr)]
#[verifier::reject_recursive_types(Entry)]
struct PageTableState<VirtAddr, Entry> {
    entries: Map<VirtAddr, Seq<PTPermView<Entry>>>, // Each virtual address can have multiple page table entries 
}
}