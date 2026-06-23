// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2022-2023 SUSE LLC
//
// Author: Joerg Roedel <jroedel@suse.de>

use crate::BIT_MASK;
use crate::address::{Address, PhysAddr, VirtAddr};
use crate::cpu::control_regs::write_cr3;
use crate::cpu::flush_tlb_global_sync;
use crate::cpu::idt::common::PageFaultError;
use crate::cpu::registers::RFlags;
use crate::error::SvsmError;
use crate::mm::{
    PGTABLE_LVL3_IDX_PTE_SELFMAP, PGTABLE_LVL3_IDX_SHARED, PageBox, phys_to_virt, virt_to_phys,
};
use crate::platform::SvsmPlatform;
use crate::types::{PAGE_SIZE, PAGE_SIZE_2M, PageSize};
use crate::utils::MemoryRegion;
use crate::utils::immut_after_init::{ImmutAfterInitCell, ImmutAfterInitResult};
use bitflags::bitflags;
use core::cmp;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use cpuarch::x86::CR0Flags;
use cpuarch::x86::CR4Flags;
use cpuarch::x86::EFERFlags;
use paging::tlb::MayNeedFlush;
use zerocopy::FromBytes;

// Re-export types from the paging crate.
pub use paging::pagetable::{
    ArchPagingMeta, GenericPageTableFlags, InactivePageTable, PageLevel, PagingError,
    PagingHandler, PagingLevel, SelfMap,
};
pub use paging::x86_64::{PTEntryFlags, PdptLevel, Pml4Level};

use paging::active_pagetable::ActivePageTableNode;

/// Mask for private page table entry.
static PRIVATE_PTE_MASK: ImmutAfterInitCell<usize> = ImmutAfterInitCell::uninit();

/// Mask for shared page table entry.
static SHARED_PTE_MASK: ImmutAfterInitCell<usize> = ImmutAfterInitCell::uninit();

/// Maximum physical address supported by the system.
static MAX_PHYS_ADDR: ImmutAfterInitCell<u64> = ImmutAfterInitCell::uninit();

/// Maximum physical address bits supported by the system.
static PHYS_ADDR_SIZE: ImmutAfterInitCell<u32> = ImmutAfterInitCell::uninit();

/// Physical address for the Launch VMSA (Virtual Machine Saving Area).
pub const LAUNCH_VMSA_ADDR: PhysAddr = PhysAddr::new(0xFFFFFFFFF000);

/// Feature mask for page table entry flags.
static FEATURE_MASK: ImmutAfterInitCell<PTEntryFlags> = ImmutAfterInitCell::uninit();

/// Initializes paging settings.
pub fn paging_init(platform: &dyn SvsmPlatform, suppress_global: bool) -> ImmutAfterInitResult<()> {
    init_encrypt_mask(platform)?;

    let mut feature_mask = PTEntryFlags::all();
    if suppress_global {
        feature_mask.remove(PTEntryFlags::GLOBAL);
    }
    FEATURE_MASK.init(feature_mask)
}

/// Initializes the encrypt mask.
fn init_encrypt_mask(platform: &dyn SvsmPlatform) -> ImmutAfterInitResult<()> {
    let masks = platform.get_page_encryption_masks();

    PRIVATE_PTE_MASK.init(masks.private_pte_mask)?;
    SHARED_PTE_MASK.init(masks.shared_pte_mask)?;

    let guest_phys_addr_size = (masks.phys_addr_sizes >> 16) & 0xff;
    let host_phys_addr_size = masks.phys_addr_sizes & 0xff;
    let phys_addr_size = if guest_phys_addr_size == 0 {
        // When [GuestPhysAddrSize] is zero, refer to the PhysAddrSize field
        // for the maximum guest physical address size.
        // - APM3, E.4.7 Function 8000_0008h - Processor Capacity Parameters and Extended Feature Identification
        host_phys_addr_size
    } else {
        guest_phys_addr_size
    };

    PHYS_ADDR_SIZE.init(phys_addr_size)?;

    // If the C-bit is a physical address bit however, the guest physical
    // address space is effectively reduced by 1 bit.
    // - APM2, 15.34.6 Page Table Support
    let effective_phys_addr_size = cmp::min(masks.addr_mask_width, phys_addr_size);

    let max_addr = 1 << effective_phys_addr_size;
    MAX_PHYS_ADDR.init(max_addr)
}

/// Returns the private encrypt mask value.
pub fn private_pte_mask() -> usize {
    *PRIVATE_PTE_MASK
}

/// Returns the shared encrypt mask value.
fn shared_pte_mask() -> usize {
    *SHARED_PTE_MASK
}

/// Returns the exclusive end of the physical address space.
pub fn max_phys_addr() -> PhysAddr {
    PhysAddr::from(*MAX_PHYS_ADDR)
}

/// Set address as private via mask.
pub fn make_private_address(paddr: PhysAddr) -> PhysAddr {
    SvsmPaging::make_private_address(paddr)
}

/// The SVSM page table provider: frame mapping, allocation, and encryption masks.
#[derive(Debug, Clone, Copy, FromBytes)]
pub struct SvsmPaging;

impl ArchPagingMeta for SvsmPaging {
    type PTFlags = PTEntryFlags;

    fn private_pte_mask() -> usize {
        private_pte_mask()
    }

    fn shared_pte_mask() -> usize {
        shared_pte_mask()
    }

    fn address_mask() -> usize {
        0x000f_ffff_ffff_f000
    }

    fn flush_tlb_global() {
        flush_tlb_global_sync();
    }

    fn supported_flags() -> PTEntryFlags {
        *FEATURE_MASK
    }
}

// SAFETY trait: stateless TLB hooks dispatching to the kernel's `cpu::tlb`
// free functions; used to discharge `MayNeedFlush` obligations.
impl paging::tlb::TlbOps for SvsmPaging {
    fn flush_tlb_global_sync() {
        crate::cpu::flush_tlb_global_sync();
    }

    fn flush_tlb_global_sync_page(vaddr: VirtAddr, page_size: PageSize) {
        crate::cpu::flush_tlb_global_sync_page(vaddr, page_size);
    }

    fn flush_tlb_global_sync_range(start: VirtAddr, len: usize, page_size: PageSize) {
        crate::cpu::flush_tlb_global_sync_range(MemoryRegion::new(start, len), page_size);
    }

    fn flush_tlb_global_percpu() {
        crate::cpu::flush_tlb_global_percpu();
    }

    fn flush_tlb_percpu() {
        crate::cpu::flush_tlb_percpu();
    }

    fn flush_address_percpu(vaddr: VirtAddr) {
        crate::cpu::flush_address_percpu(vaddr);
    }

    fn flush_range_percpu(start: VirtAddr, len: usize, page_size: PageSize) {
        let region = MemoryRegion::new(start, len);
        for page in region.iter_pages(page_size) {
            crate::cpu::flush_address_percpu(page);
        }
    }
}

// SAFETY: paddr_to_vaddr correctly maps physical addresses via phys_to_virt,
// and allocate_physical_page returns unique zeroed frames via PageBox.
unsafe impl PagingHandler for SvsmPaging {
    fn paddr_to_vaddr(paddr: PhysAddr) -> VirtAddr {
        phys_to_virt(paddr)
    }

    fn vaddr_to_paddr(vaddr: VirtAddr) -> PhysAddr {
        virt_to_phys(vaddr)
    }

    fn allocate_physical_page() -> Result<PhysAddr, PagingError> {
        let page = pt_page_alloc_box().map_err(|_| PagingError::AllocFrame)?;
        let paddr = virt_to_phys(page.vaddr());
        let _ = PageBox::leak(page);
        Ok(paddr)
    }

    unsafe fn deallocate_physical_page(paddr: PhysAddr) {
        let vaddr = phys_to_virt(paddr);
        // SAFETY: paddr was returned by allocate_physical_page (via PageBox::leak),
        // so reconstructing the PageBox from the same pointer is valid.
        unsafe {
            let ptr = NonNull::new(vaddr.as_mut_ptr::<PTPage>()).unwrap();
            let _ = PageBox::from_raw(ptr);
        }
    }
}

impl SelfMap for SvsmPaging {
    const SELFMAP_IDX: usize = PGTABLE_LVL3_IDX_PTE_SELFMAP;
}

/// Represents paging mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagingMode {
    // Paging mode is disabled
    NoPaging,
    // 32bit legacy paging mode
    NonPAE,
    // 32bit PAE paging mode
    PAE,
    // 4 level paging mode
    PML4,
    // 5 level paging mode
    PML5,
}

impl PagingMode {
    pub fn new(efer: EFERFlags, cr0: CR0Flags, cr4: CR4Flags) -> Self {
        if !cr0.contains(CR0Flags::PG) {
            // Paging is disabled
            PagingMode::NoPaging
        } else if efer.contains(EFERFlags::LMA) {
            // Long mode is activated
            if cr4.contains(CR4Flags::LA57) {
                PagingMode::PML5
            } else {
                PagingMode::PML4
            }
        } else if cr4.contains(CR4Flags::PAE) {
            // PAE mode
            PagingMode::PAE
        } else {
            // Non PAE mode
            PagingMode::NonPAE
        }
    }
}

/// Represents a page table entry.
pub type PTEntry = paging::pagetable::PTEntry<SvsmPaging>;

/// A pagetable page with multiple entries.
pub type PTPage = paging::pagetable::PTPage<SvsmPaging, SvsmPaging>;

/// A physical address within a page frame
pub type PageFrame = paging::pagetable::PageFrame<SvsmPaging>;

trait PTEntryExt {
    /// Check if the page table entry has reserved bits set.
    fn has_reserved_bits(&self, pm: PagingMode, level: usize) -> bool;
}

impl PTEntryExt for PTEntry {
    fn has_reserved_bits(&self, pm: PagingMode, level: usize) -> bool {
        let reserved_mask = match pm {
            PagingMode::NoPaging => unreachable!("NoPaging does not have page table"),
            PagingMode::NonPAE => {
                match level {
                    // No reserved bits in 4k PTE.
                    0 => 0,
                    1 => {
                        if self.huge() {
                            // Bit21 is reserved in 4M PDE.
                            BIT_MASK!(21, 21)
                        } else {
                            // No reserved bits in PDE.
                            0
                        }
                    }
                    _ => unreachable!("Invalid NonPAE page table level"),
                }
            }
            PagingMode::PAE => {
                // Bit62 ~ MAXPHYSADDR are reserved for each
                // level in PAE page table.
                BIT_MASK!(62, *PHYS_ADDR_SIZE)
                    | match level {
                        // No additional reserved bits in 4k PTE.
                        0 => 0,
                        1 => {
                            if self.huge() {
                                // Bit20 ~ Bit13 are reserved in 2M PDE.
                                BIT_MASK!(20, 13)
                            } else {
                                // No additional reserved bits in PDE.
                                0
                            }
                        }
                        // Bit63 and Bit8 ~ Bit5 are reserved in PDPTE.
                        2 => BIT_MASK!(63, 63) | BIT_MASK!(8, 5),
                        _ => unreachable!("Invalid PAE page table level"),
                    }
            }
            PagingMode::PML4 | PagingMode::PML5 => {
                // Bit51 ~ MAXPHYSADDR are reserved for each level
                // in PML4 and PML5 page table.
                let common = if *PHYS_ADDR_SIZE > 51 {
                    0
                } else {
                    // Remove the encryption mask bit as this bit is not reserved
                    BIT_MASK!(51, *PHYS_ADDR_SIZE)
                        & !((shared_pte_mask() | private_pte_mask()) as u64)
                };

                common
                    | match level {
                        // No additional reserved bits in 4k PTE.
                        0 => 0,
                        1 => {
                            if self.huge() {
                                // Bit20 ~ Bit13 are reserved in 2M PDE.
                                BIT_MASK!(20, 13)
                            } else {
                                // No additional reserved bits in PDE.
                                0
                            }
                        }
                        2 => {
                            if self.huge() {
                                // Bit29 ~ Bit13 are reserved in 1G PDPTE.
                                BIT_MASK!(29, 13)
                            } else {
                                // No additional reserved bits in PDPTE.
                                0
                            }
                        }
                        // Bit8 ~ Bit7 are reserved in PML4E.
                        3 => BIT_MASK!(8, 7),
                        4 => {
                            if pm == PagingMode::PML4 {
                                unreachable!("Invalid PML4 page table level");
                            } else {
                                // Bit8 ~ Bit7 are reserved in PML5E.
                                BIT_MASK!(8, 7)
                            }
                        }
                        _ => unreachable!("Invalid PML4/PML5 page table level"),
                    }
            }
        };

        self.raw() & reserved_mask as usize != 0
    }
}

fn pt_page_alloc_box() -> Result<PageBox<PTPage>, SvsmError> {
    PageBox::try_new_zeroed()
}

/// Page table structure containing a root page with multiple entries.
pub type PageTable = InactivePageTable<SvsmPaging, SvsmPaging, Pml4Level>;

type ActivePageTablePart = ActivePageTableNode<SvsmPaging, SvsmPaging, PdptLevel>;

type SvsmActivePageTable = ActivePageTableNode<SvsmPaging, SvsmPaging, Pml4Level>;
/// An *active* (installed, or about-to-be-installed) page table.
///
/// Once a [`PageBox<PageTable>`] is stored for later use (per-CPU or
/// per-task), it must be assumed to be concurrently walked by the MMU.
/// All entry accesses therefore go through [`ActivePageTableNode`], which
/// uses volatile raw-pointer reads/writes and never forms a `&PTPage`/
/// `&PTEntry` reference into the live table.
///
/// Uninstalled page tables are built through [`PageTable`] (and its
/// [`GenericPageTable`] methods) and wrapped into an `ActivePageTable` via
/// [`ActivePageTable::new`] when they are stored.
#[derive(Debug)]
pub struct ActivePageTable {
    node: SvsmActivePageTable,
}

impl ActivePageTable {
    /// Wrap an (about-to-be) installed page table.
    pub fn new(pgtable: PageBox<PageTable>) -> Self {
        let pgtable = PageBox::leak(pgtable);
        let pgtable_phys = virt_to_phys((pgtable as *const PageTable).into());
        // SAFETY: the page table is leaked and will not be deallocated until the active node is dropped.
        let node = unsafe { SvsmActivePageTable::new(pgtable_phys) };
        Self { node }
    }

    /// Load this page table into the CR3 register.
    ///
    /// # Safety
    ///
    /// The caller must ensure to take other actions to make sure a memory safe
    /// execution state is warranted (e.g. changing the stack and register state)
    pub unsafe fn load(&self) {
        // SAFETY: demanded to the caller
        unsafe {
            write_cr3(self.node.cr3_value());
        }
    }

    /// Translate a virtual address using the self-map of the *currently
    /// installed* page table.
    ///
    /// This reads page-table entries through the self-map, so it only resolves
    /// correctly while this table is the one installed in the MMU.
    pub fn virt_to_frame(vaddr: VirtAddr) -> Option<PageFrame> {
        SvsmActivePageTable::virt_to_frame(vaddr)
    }

    /// Clone the shared part of the page table into a fresh, uninstalled
    /// page table; excluding the private parts.
    ///
    /// # Errors
    /// Returns [`SvsmError`] if the page cannot be allocated.
    pub fn clone_shared(&self) -> Result<PageBox<PageTable>, SvsmError> {
        let mut pgtable = allocate_pgtable()?;
        self.node
            .next_table(PGTABLE_LVL3_IDX_SHARED)
            .map(|next| pgtable.populate(PGTABLE_LVL3_IDX_SHARED, next));
        //self.copy_entry_to(&mut pgtable, PGTABLE_LVL3_IDX_SHARED);
        Ok(pgtable)
    }

    /// Maps a region of memory using 4KB pages. See [`PageTable::map_region_4k`].
    pub fn map_region_4k(
        &mut self,
        vregion: MemoryRegion<VirtAddr>,
        phys: PhysAddr,
        flags: PTEntryFlags,
        shared: bool,
    ) -> Result<(), SvsmError> {
        for addr in vregion.iter_pages(PageSize::Regular) {
            let offset = addr - vregion.start();
            self.map_4k(addr, phys + offset, flags, shared)?;
        }
        Ok(())
    }

    /// Maps a region of memory using 2MB pages. See [`PageTable::map_region_2m`].
    pub fn map_region_2m(
        &mut self,
        vregion: MemoryRegion<VirtAddr>,
        phys: PhysAddr,
        flags: PTEntryFlags,
        shared: bool,
    ) -> Result<(), SvsmError> {
        for addr in vregion.iter_pages(PageSize::Huge) {
            let offset = addr - vregion.start();
            self.node.map_2m(addr, phys + offset, flags, shared)?;
        }
        Ok(())
    }

    /// Maps a memory region, choosing 2MB or 4KB pages. See [`PageTable::map_region`].
    pub fn map_region(
        &mut self,
        region: MemoryRegion<VirtAddr>,
        phys: PhysAddr,
        flags: PTEntryFlags,
    ) -> Result<(), SvsmError> {
        let mut vaddr = region.start();
        let end = region.end();
        let mut paddr = phys;

        while vaddr < end {
            if vaddr.is_aligned(PAGE_SIZE_2M)
                && paddr.is_aligned(PAGE_SIZE_2M)
                && vaddr + PAGE_SIZE_2M <= end
                && self.node.map_2m(vaddr, paddr, flags, false).is_ok()
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

    /// Unmaps a region of 4KB pages. See [`PageTable::unmap_region_4k`].
    pub fn unmap_region_4k(&mut self, vregion: MemoryRegion<VirtAddr>) {
        for addr in vregion.iter_pages(PageSize::Regular) {
            if let Some((_, flush)) = self.node.unmap_4k(addr) {
                flush.ignore("caller flushes the TLB explicitly");
            }
        }
    }

    /// Unmaps a region of 2MB pages. See [`PageTable::unmap_region_2m`].
    pub fn unmap_region_2m(&mut self, vregion: MemoryRegion<VirtAddr>) {
        for addr in vregion.iter_pages(PageSize::Huge) {
            if let Some((_, flush)) = self.node.unmap_2m(addr) {
                flush.ignore("caller flushes the TLB explicitly");
            }
        }
    }

    /// Populates this page table with the contents of the given subtree in
    /// `part`. See [`PageTable::populate_pgtbl_part`].
    ///
    /// Returns `true` if the PTE contents were updated.
    pub fn populate_pgtbl_part(&mut self, part: &PageTablePart) -> bool {
        let Some(table) = &part.raw else {
            return false;
        };
        let idx: usize = part.index();
        self.populate(idx, table.get_view())
    }

    /// Makes the memory region pages read-only.
    /// This method is meant for global pages only.
    ///
    /// # Safety
    ///
    /// The caller should verify that `region` can be made read-only, i.e. that
    /// no write can happen or that a #PF raised by any tentative write is
    /// expected.
    /// The caller must also ensure that the region start and size are 4k
    /// aligned.
    ///
    /// # Returns
    /// The [`MayNeedFlush`] TLB-flush obligation for the now-stale
    /// translations on success, or a [`SvsmError`] on failure.
    pub unsafe fn make_region_ro_4k(
        &mut self,
        region: MemoryRegion<VirtAddr>,
    ) -> Result<MayNeedFlush<SvsmPaging>, SvsmError> {
        let mut flush = MayNeedFlush::new(PageLevel::Level0);
        for page in region.iter_pages(PageSize::Regular) {
            let mapping = self.node.walk_mut(page);
            let entry = mapping.read();
            match mapping.level() {
                PageLevel::Level0 => {
                    if !entry.present() || !entry.flags().global() {
                        return Err(SvsmError::Mem);
                    }

                    let mut new_entry = PTEntry::empty();
                    new_entry.set(entry.paddr_field(), PTEntryFlags::data_ro());
                    // The caller flushes the whole region at once below, via
                    // the token returned after the loop.
                    flush = mapping.update(new_entry);
                }
                PageLevel::Level1 | PageLevel::Level2 => {
                    // Ensure we never fell on a huge page while iterating over the region pages.
                    if entry.huge() {
                        return Err(SvsmError::Mem);
                    }
                }
                _ => {}
            }
        }

        Ok(flush)
    }
}

impl Deref for ActivePageTable {
    type Target = SvsmActivePageTable;
    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl DerefMut for ActivePageTable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.node
    }
}

fn allocate_pgtable() -> Result<PageBox<PageTable>, SvsmError> {
    let mut pgtable: PageBox<PageTable> = PageBox::try_new_zeroed()?;
    pgtable.init_self_map();
    Ok(pgtable)
}

/// Sub-tree of a page table that can be populated at the top-level
/// used for virtual memory management
#[derive(Debug)]
pub struct PageTablePart {
    /// The root of the page-table sub-tree
    /// We conservatively assume that the sub-tree might be active,
    /// since we never track whether the sub-tree is populated and
    /// whether the top table it populated to has been installed.
    raw: Option<ActivePageTablePart>,
    /// The top-level index this PageTablePart is populated at
    idx: usize,
}

impl Drop for PageTablePart {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.as_mut() {
            // SAFETY: the sub-tree is being dropped, so it must already have
            // been removed from any active page table and is no longer
            // reachable by the MMU; inactive reference access is therefore
            // sound.
            unsafe { raw.as_inactive().free() }
        }
    }
}

impl PageTablePart {
    /// Create a new PageTablePart and allocate a root page for the page-table sub-tree.
    ///
    /// # Arguments
    ///
    /// - `start`: Virtual start address this PageTablePart maps
    ///
    /// # Returns
    ///
    /// A new instance of PageTablePart
    pub fn new(start: VirtAddr) -> Self {
        PageTablePart {
            raw: None,
            idx: PageTable::index::<3>(start),
        }
    }

    pub fn alloc(&mut self) -> Result<(), SvsmError> {
        self.get_or_init_mut()?;
        Ok(())
    }

    fn get_or_init_mut(&mut self) -> Result<&mut ActivePageTablePart, SvsmError> {
        if self.raw.is_none() {
            // A freshly allocated sub-tree that might be linked to an active
            // page table; treat it as active from creation so all entry
            // accesses use volatile reads/writes.
            self.raw = Some(ActivePageTablePart::try_new().map_err(|_| SvsmError::Mem)?);
        }
        Ok(self.raw.as_mut().unwrap())
    }

    fn get_mut(&mut self) -> Option<&mut ActivePageTablePart> {
        self.raw.as_mut()
    }

    fn get(&self) -> Option<&ActivePageTablePart> {
        self.raw.as_ref()
    }

    /// Request PageTable index to populate this instance to
    ///
    /// # Returns
    ///
    /// Index of the top-level PageTable this sub-tree is populated to
    pub fn index(&self) -> usize {
        self.idx
    }

    /// Populate the PageTablePart to an inactive page table.
    pub fn populate_by_inactive(&self, pgtbl: &mut PageTable) -> bool {
        self.get()
            .map(|p| pgtbl.populate(self.index(), p.get_view()))
            .unwrap_or(false)
    }

    /// Map a 4KiB page in the page table sub-tree
    ///
    /// # Arguments
    ///
    /// * `vaddr` - Virtual address to create the mapping. Must be aligned to 4KiB.
    /// * `paddr` - Physical address to map. Must be aligned to 4KiB.
    /// * `flags` - PTEntryFlags used for the mapping
    /// * `shared` - Defines whether the page is mapped shared or private
    ///
    /// # Returns
    ///
    /// OK(()) on Success, Err(SvsmError::Mem) on error.
    ///
    /// This function can fail when there not enough memory to allocate pages for the mapping.
    ///
    /// # Panics
    ///
    /// This method panics when either `vaddr` or `paddr` are not aligned to 4KiB.
    pub fn map_4k(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: PTEntryFlags,
        shared: bool,
    ) -> Result<(), SvsmError> {
        assert!(PageTable::index::<3>(vaddr) == self.idx);

        self.get_or_init_mut()?
            .map_4k(vaddr, paddr, flags, shared)
            .map_err(|_| SvsmError::Mem)
    }

    /// Unmaps a 4KiB page from the page table sub-tree
    ///
    /// # Arguments
    ///
    /// * `vaddr` - The virtual address to unmap. Must be aligned to 4KiB.
    ///
    /// # Returns
    ///
    /// A copy of the [`PTEntry`] that mapped the virtual address together with
    /// the [`MayNeedFlush`] TLB-flush obligation for the now-stale translation,
    /// or [`None`] if no leaf was mapped.
    ///
    /// # Panics
    ///
    /// This method panics when `vaddr` is not aligned to 4KiB.
    pub fn unmap_4k(&mut self, vaddr: VirtAddr) -> Option<(PTEntry, MayNeedFlush<SvsmPaging>)> {
        assert!(PageTable::index::<3>(vaddr) == self.idx);
        self.get_mut()?.unmap_4k(vaddr)
    }

    /// Map a 2MiB page in the page table sub-tree
    ///
    /// # Arguments
    ///
    /// * `vaddr` - Virtual address to create the mapping. Must be aligned to 2MiB.
    /// * `paddr` - Physical address to map. Must be aligned to 2MiB.
    /// * `flags` - PTEntryFlags used for the mapping
    /// * `shared` - Defines whether the page is mapped shared or private
    ///
    /// # Returns
    ///
    /// OK(()) on Success, Err(SvsmError::Mem) on error.
    ///
    /// This function can fail when there not enough memory to allocate pages for the mapping.
    ///
    /// # Panics
    ///
    /// This method panics when either `vaddr` or `paddr` are not aligned to 2MiB.
    pub fn map_2m(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: PTEntryFlags,
        shared: bool,
    ) -> Result<(), SvsmError> {
        assert!(PageTable::index::<3>(vaddr) == self.idx);

        self.get_or_init_mut()?
            .map_2m(vaddr, paddr, flags, shared)
            .map_err(|_| SvsmError::Mem)
    }

    /// Unmaps a 2MiB page from the page table sub-tree
    ///
    /// # Arguments
    ///
    /// * `vaddr` - The virtual address to unmap. Must be aligned to 2MiB.
    ///
    /// # Returns
    ///
    /// A copy of the [`PTEntry`] that mapped the virtual address together with
    /// the [`MayNeedFlush`] TLB-flush obligation for the now-stale translation,
    /// or [`None`] if no huge leaf was mapped.
    ///
    /// # Panics
    ///
    /// This method panics when `vaddr` is not aligned to 2MiB.
    pub fn unmap_2m(&mut self, vaddr: VirtAddr) -> Option<(PTEntry, MayNeedFlush<SvsmPaging>)> {
        assert!(PageTable::index::<3>(vaddr) == self.idx);
        self.get_mut()?.unmap_2m(vaddr)
    }
}

bitflags! {
    /// Flags to represent how memory is accessed, e.g. write data to the
    /// memory or fetch code from the memory.
    #[derive(Clone, Copy, Debug)]
    pub struct MemAccessMode: u32 {
        const WRITE     = 1 << 0;
        const FETCH     = 1 << 1;
    }
}

/// Attributes to determin Whether a memory access (write/fetch) is permitted
/// by a translation which includes the paging-mode modifiers in CR0, CR4 and
/// EFER; EFLAGS.AC; and the supervisor/user mode access.
#[derive(Clone, Copy, Debug)]
pub struct PTWalkAttr {
    cr0: CR0Flags,
    cr4: CR4Flags,
    efer: EFERFlags,
    flags: RFlags,
    user_mode_access: bool,
    pm: PagingMode,
}

impl PTWalkAttr {
    /// Creates a new `PTWalkAttr` instance with the specified attributes.
    ///
    /// # Arguments
    ///
    /// * `cr0`, `cr4`, and `efer` - Represent the control register
    ///   flags for CR0, CR4, and EFER respectively.
    /// * `flags` - Represents the CPU Flags.
    /// * `user_mode_access` - Indicates whether the access is in user mode.
    ///
    /// Returns a new `PTWalkAttr` instance.
    pub fn new(
        cr0: CR0Flags,
        cr4: CR4Flags,
        efer: EFERFlags,
        flags: RFlags,
        user_mode_access: bool,
    ) -> Self {
        Self {
            cr0,
            cr4,
            efer,
            flags,
            user_mode_access,
            pm: PagingMode::new(efer, cr0, cr4),
        }
    }

    /// Checks the access rights for a page table entry.
    ///
    /// # Arguments
    ///
    /// * `entry` - The page table entry to check.
    /// * `mem_am` - Indicates how to access the memory.
    /// * `last_level` - Indicates whether the entry is at the last level
    ///   of the page table.
    /// * `pteflags` - The PTE flags to indicate if the corresponding page
    ///   table entry allows the access rights.
    ///
    /// # Returns
    ///
    /// Returns `Ok((entry, leaf))` if the access rights are valid, where
    /// `entry` is the modified page table entry and `leaf` is a boolean
    /// indicating whether the entry is a leaf node, or `Err(PageFaultError)`
    /// to indicate the page fault error code if the access rights are invalid.
    pub fn check_access_rights(
        &self,
        entry: PTEntry,
        mem_am: MemAccessMode,
        level: usize,
        pteflags: &mut PTEntryFlags,
    ) -> Result<(PTEntry, bool), PageFaultError> {
        let pf_err = self.default_pf_err(mem_am) | PageFaultError::P;

        if !entry.present() {
            // Entry is not present.
            return Err(pf_err & !PageFaultError::P);
        }

        if entry.has_reserved_bits(self.pm, level) {
            // Reserved bits have been set.
            return Err(pf_err | PageFaultError::R);
        }

        // SDM 4.6.1 Determination of Access Rights:
        // If the U/S flag (bit 2) is 0 in at least one of the
        // paging-structure entries, the address is a supervisor-mode
        // address. Otherwise, the address is a user-mode address.
        // So by-default assume the address is user mode address.
        if !entry.user() {
            *pteflags &= !PTEntryFlags::USER;
        }

        // SDM 4.6.1 Determination of Access Rights:
        // R/W flag (bit 1) is 1 in every paging-structure entry controlling
        // the translation and with a protection key for which write access is
        // permitted; data may not be written to any supervisor-mode
        // address with a translation for which the R/W flag is 0 in any
        // paging-structure entry controlling the translation.
        // The same for user mode address
        if !entry.flags().writable() {
            *pteflags &= !PTEntryFlags::WRITABLE;
        }

        // SDM 4.6.1 Determination of Access Rights:
        // For non 32-bit paging modes with IA32_EFER.NXE = 1, instructions
        // may be fetched from any supervisormode address with a translation
        // for which the XD flag (bit 63) is 0 in every paging-structure entry
        // controlling the translation; instructions may not be fetched from
        // any supervisor-mode address with a translation for which the XD flag
        // is 1 in any paging-structure entry controlling the translation
        if self.efer.contains(EFERFlags::NXE) && entry.flags().nx() {
            *pteflags |= PTEntryFlags::NX;
        } else if !self.efer.contains(EFERFlags::NXE) && entry.flags().nx() {
            // XD bit must be 0 if efer.NXE = 0
            return Err(pf_err | PageFaultError::R);
        }

        let leaf = if level == 0 || entry.huge() {
            // User mode cannot access any supervisor mode addresses
            if self.user_mode_access && !pteflags.contains(PTEntryFlags::USER) {
                return Err(pf_err);
            }

            // Always check for reading. For the case of supervisor mode read user
            // mode addresses, do special checking. For other cases, read is allowed.
            if !self.user_mode_access && pteflags.contains(PTEntryFlags::USER) {
                // Read not allowed with SMAP = 1 && flags.ac = 0
                if self.cr4.contains(CR4Flags::SMAP) && !self.flags.contains(RFlags::AC) {
                    return Err(pf_err);
                }
            }

            if mem_am.contains(MemAccessMode::WRITE) {
                if !self.user_mode_access && pteflags.contains(PTEntryFlags::USER) {
                    // Check supervisor mode write user mode addresses
                    if !self.cr0.contains(CR0Flags::WP) {
                        // Check write with CR0.WP = 0
                        if self.cr4.contains(CR4Flags::SMAP) && !self.flags.contains(RFlags::AC) {
                            // Write not allowed with SMAP = 1 && flags.ac = 0
                            return Err(pf_err);
                        }
                    } else {
                        // Check write with CR0.WP = 1
                        if !self.cr4.contains(CR4Flags::SMAP) {
                            // SMAP = 0
                            if !pteflags.contains(PTEntryFlags::WRITABLE) {
                                // Write not allowed R/W = 0
                                return Err(pf_err);
                            }
                        } else {
                            // SMAP = 1
                            if !self.flags.contains(RFlags::AC)
                                || !pteflags.contains(PTEntryFlags::WRITABLE)
                            {
                                // Write not allowed with flags.AC = 0 || R/W = 0
                                return Err(pf_err);
                            }
                        }
                    }
                } else if !self.user_mode_access && !pteflags.contains(PTEntryFlags::USER) {
                    // Check supervisor mode write supervisor mode addresses
                    if self.cr0.contains(CR0Flags::WP) && !pteflags.contains(PTEntryFlags::WRITABLE)
                    {
                        // Write not allowed with CR0.WP = 1 && R/W = 0
                        return Err(pf_err);
                    }
                } else if self.user_mode_access && pteflags.contains(PTEntryFlags::USER) {
                    // Check user mode write user mode addresses
                    if !pteflags.contains(PTEntryFlags::WRITABLE) {
                        // Write not allowed R/W = 0
                        return Err(pf_err);
                    }
                }
                // User mode write supervisor mode addresses is checked already
            }

            if mem_am.contains(MemAccessMode::FETCH) {
                // For instruction fetch, the rule is the same except for the case of
                // supervisor mode fetch user mode addresses
                if !self.user_mode_access && pteflags.contains(PTEntryFlags::USER) {
                    // Fetch not allowed with SMEP = 1
                    if self.cr4.contains(CR4Flags::SMEP) {
                        return Err(pf_err);
                    }
                }

                // For non-32bit paging mode, fetch not allowed with efer.NXE = 1 && XD = 1
                if self.cr4.contains(CR4Flags::PAE)
                    && self.efer.contains(EFERFlags::NXE)
                    && pteflags.contains(PTEntryFlags::NX)
                {
                    return Err(pf_err);
                }
            }
            true
        } else {
            false
        };

        Ok((entry, leaf))
    }

    fn default_pf_err(&self, mem_am: MemAccessMode) -> PageFaultError {
        let mut err = PageFaultError::empty();

        if mem_am.contains(MemAccessMode::WRITE) {
            err |= PageFaultError::W;
        }

        if mem_am.contains(MemAccessMode::FETCH) {
            err |= PageFaultError::I;
        }

        if self.user_mode_access {
            err |= PageFaultError::U;
        }

        err
    }
}
