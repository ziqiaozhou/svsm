use vstd::prelude::*;
use crate::traits::PageLevel;

verus!{
/// Represents the state of a page table.
enum PTState {
    Installed, // The page table is installed in CPUs
    NotInstalled, // The page table is not installed in any CPU
}

pub(super) ghost struct EntryPropertiesSpec {
    pub valid: bool,
    pub writable: bool,
    pub user: bool,
    pub huge: bool,
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

pub(super) trait PTableSpec {
    type Entry: EntrySpec;

    spec fn index(&self, i: int) -> Self::Entry;
}

pub(super) trait PTVAddrSpec {
    spec fn entry_index(&self, level: PageLevel) -> int;
}
}