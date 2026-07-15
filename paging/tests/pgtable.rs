// SPDX-License-Identifier: MIT OR Apache-2.0
use paging::address::{Address, PhysAddr, VirtAddr};
use paging::pagetable::{InactivePageTable, MappingMutOps, MappingRefOps};
use paging::sizes::PageSize;
use paging::tlb::TlbOps;
use paging::traits::{
    ArchPagingMeta, PageLevel, PagingError, PagingHandler, PagingLevel0, PagingLevel1,
    PagingLevel2, PagingLevel3,
};
use paging::x86_64::PTEntryFlags;
use std::alloc::{Layout, alloc_zeroed};

/// Minimal architecture metadata for the walk test: identity address
/// masking, no private/shared bits, and no-op TLB hooks.
#[derive(Copy, Clone, Debug)]
struct TestArch;

impl ArchPagingMeta for TestArch {
    type PTFlags = PTEntryFlags;
    fn private_pte_mask() -> usize {
        0
    }
    fn shared_pte_mask() -> usize {
        0
    }
    fn address_mask() -> usize {
        0x000f_ffff_ffff_f000
    }
    fn flush_tlb_global() {}
}

impl TlbOps for TestArch {
    fn flush_tlb_global_sync() {}
    fn flush_tlb_global_sync_page(_: VirtAddr, _: PageSize) {}
    fn flush_tlb_global_sync_range(_: VirtAddr, _: usize, _: PageSize) {}
    fn flush_tlb_global_percpu() {}
    fn flush_tlb_percpu() {}
    fn flush_address_percpu(_: VirtAddr) {}
    fn flush_range_percpu(_: VirtAddr, _: usize, _: PageSize) {}
}

/// Identity-mapped handler backed by leaked, page-aligned host allocations
/// so `paddr == vaddr` and frames stay valid for the test's lifetime.
#[derive(Copy, Clone, Debug, zerocopy::FromBytes)]
struct TestHandler;

// SAFETY: `paddr_to_vaddr`/`vaddr_to_paddr` are identity and
// `allocate_physical_page` returns freshly-allocated, page-aligned, zeroed
// frames, satisfying the trait's contract for the test.
unsafe impl PagingHandler for TestHandler {
    fn paddr_to_vaddr(paddr: PhysAddr) -> VirtAddr {
        assert_ne!(
            paddr.bits(),
            LEAF_PA,
            "walk resolved a leaf entry's data frame as a page-table page"
        );
        VirtAddr::from(paddr.bits())
    }
    fn vaddr_to_paddr(vaddr: VirtAddr) -> PhysAddr {
        PhysAddr::from(usize::from(vaddr))
    }
    fn allocate_physical_page() -> Result<PhysAddr, PagingError> {
        let layout = Layout::from_size_align(0x1000, 0x1000).unwrap();
        // SAFETY: `layout` is non-zero and validly aligned; the returned
        // frame is intentionally leaked for the duration of the test.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(PagingError::AllocFrame);
        }
        Ok(PhysAddr::from(ptr as usize))
    }
    unsafe fn deallocate_physical_page(_paddr: PhysAddr) {}
}

/// Sentinel physical address used as a 4 KiB leaf's data frame. It is never a
/// real allocation, so translating it as a page-table page is a bug.
const LEAF_PA: usize = 0x0000_dead_0000;

type L3Table = InactivePageTable<TestArch, TestHandler, PagingLevel3>;
type L2Table = InactivePageTable<TestArch, TestHandler, PagingLevel2>;
type L1Table = InactivePageTable<TestArch, TestHandler, PagingLevel1>;
type L0Table = InactivePageTable<TestArch, TestHandler, PagingLevel0>;

#[test]
fn test_uninstalled_populate() {
    let mut l3 = L3Table::alloc().unwrap();
    let l2 = L2Table::alloc().unwrap();
    let l22 = L1Table::alloc().unwrap();

    // Overwriting a present entry of an *installed* table with a different
    // sub-table trips the `populate` assert (`!old.present() ||
    // !S::IS_INSTALLED`).
    l3.populate(0, l2.root_pa()).unwrap(); // Ok: slot 0 was empty.
    l3.populate(0, l22.root_pa()).unwrap(); // OK: slot 0 was not empty but l3 is not installed.
}

#[test]
fn invalid_ops_for_installed() {
    let mut l3 = L3Table::alloc().unwrap().install();
    let l2 = L2Table::alloc().unwrap();
    let l22 = L1Table::alloc().unwrap();

    l3.populate(0, l2.root_pa()).unwrap(); // Ok: slot 0 was empty.
    let result = l3.populate(0, l22.root_pa()); // Error: slot 0 already present.
    assert!(result.is_err());
}

#[test]
fn walk_reaches_leaf_with_distinct_indices() {
    let mut l3 = L3Table::alloc().unwrap();
    let mut l2 = L2Table::alloc().unwrap();
    let mut l1 = L1Table::alloc().unwrap();
    let l0 = L0Table::alloc().unwrap();

    // Distinct index per level so a single-index walk diverges immediately.
    let (i3, i2, i1, i0) = (1usize, 2usize, 3usize, 4usize);
    let vaddr = VirtAddr::from((i3 << 39) | (i2 << 30) | (i1 << 21) | (i0 << 12));
    assert_eq!(level_indices(vaddr), (i3, i2, i1, i0));

    // Link each level to the next via `populate` (needs `L: NextLevel`).
    l3.populate(i3, l2.root_pa()).unwrap();
    l2.populate(i2, l1.root_pa()).unwrap();
    l1.populate(i1, l0.root_pa()).unwrap();

    // The leaf level has no child level, so write its entry directly
    // through the safe inactive accessor.
    let leaf_pa = PhysAddr::from(LEAF_PA);

    // Transition the root to the installed state before walking.
    let mut pgtbl = l3.install();
    assert!(pgtbl.next_table_pa(i3) == Some(l2.root_pa()));

    let mapping = pgtbl.walk(vaddr);
    assert_eq!(mapping.level(), PageLevel::Level0);
    let entry = mapping.read();
    assert!(!entry.present());

    pgtbl
        .map_4k(vaddr, leaf_pa, PTEntryFlags::PRESENT, false)
        .expect("map 4k");
    let mapping = pgtbl.walk(vaddr);
    let entry = mapping.read();
    assert_eq!(entry.address(), leaf_pa);
}

#[test]
fn map_and_then_walk() {
    let mut pgtbl = L3Table::alloc().unwrap().install();

    let vaddr = VirtAddr::from(0x0000_beef_0000usize);
    let leaf_pa = PhysAddr::from(LEAF_PA);
    pgtbl
        .map_4k(vaddr, leaf_pa, PTEntryFlags::data_ro(), false)
        .expect("map 4k");

    let mapping = pgtbl.walk(vaddr);
    assert_eq!(mapping.level(), PageLevel::Level0);
    let entry = mapping.read();
    assert!(entry.present());
    assert_eq!(entry.address(), leaf_pa);

    let translated = pgtbl.phys_addr(vaddr);
    assert_eq!(translated, Ok(leaf_pa));
}

#[test]
fn walk_does_not_resolve_leaf_child() {
    type GuardTable = InactivePageTable<TestArch, TestHandler, PagingLevel3>;
    let mut pgtbl = GuardTable::alloc().unwrap().install();

    let vaddr = VirtAddr::from(0x0000_beef_0000usize);
    let leaf_pa = PhysAddr::from(LEAF_PA);
    pgtbl
        .map_4k(vaddr, leaf_pa, PTEntryFlags::PRESENT, false)
        .expect("map 4k");

    let mapping = pgtbl.walk(vaddr);
    assert_eq!(mapping.level(), PageLevel::Level0);
    assert_eq!(mapping.read().address(), leaf_pa);
}

/// An absent top-level entry stops the walk immediately at the root level.
#[test]
fn walk_stops_at_absent_entry() {
    let mut table = L3Table::alloc().unwrap().install();
    let vaddr = VirtAddr::from(5usize << 39);

    let mapping = table.walk_mut(vaddr);
    assert_eq!(mapping.level(), PageLevel::Level3);
    assert!(!mapping.read().present());
}

/// `Inactive` -> `Installed` (safe `install`) and back (`unsafe as_inactive`)
/// preserve the root page and let both states reach the same entry.
#[test]
fn state_transitions_preserve_root() {
    let mut inactive = L3Table::alloc().unwrap();
    let root_pa = inactive.root_pa();
    inactive
        .populate(7, PhysAddr::from(0x0000_beef_0000usize))
        .unwrap();

    let mut installed = inactive.install();
    assert_eq!(installed.cr3_value(), root_pa);

    // SAFETY: `installed` was never loaded into CR3, so reborrowing it as
    // inactive is sound for this single-threaded test.
    let inactive = unsafe { installed.as_inactive() };
    assert_eq!(inactive.root_pa(), root_pa);
    assert!(inactive.next_table_pa(7).is_some());
}

/// Sanity check that `index_at` selects the expected 9-bit slice per level.
fn level_indices(vaddr: VirtAddr) -> (usize, usize, usize, usize) {
    (
        L3Table::index_at(vaddr, PageLevel::Level3),
        L3Table::index_at(vaddr, PageLevel::Level2),
        L3Table::index_at(vaddr, PageLevel::Level1),
        L3Table::index_at(vaddr, PageLevel::Level0),
    )
}
