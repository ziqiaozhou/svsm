// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) Microsoft Corporation
//
// Author: Jon Lange (jlange@microsoft.com)

use crate::address::PhysAddr;
use crate::address::VirtAddr;
use crate::mm::pagetable::PTEntry;
use crate::mm::pagetable::PTEntryFlags;
use crate::mm::pagetable::PTPage;
use crate::mm::pagetable::PageTable;
use crate::mm::pagetable::make_private_address;
use crate::platform::PageValidateOp;
use crate::platform::SvsmPlatform;
use crate::types::PAGE_SIZE;
use crate::types::PAGE_SIZE_2M;
use crate::utils::MemoryRegion;
use crate::utils::align_down;
use bootdefs::kernel_launch::KernelLaunchInfo;
use core::mem;
use core::ptr;

pub fn new_kernel_region(launch_info: &KernelLaunchInfo) -> MemoryRegion<PhysAddr> {
    let start = PhysAddr::from(launch_info.kernel_region_phys_start);
    let end = PhysAddr::from(launch_info.kernel_region_phys_end);
    MemoryRegion::from_addresses(start, end)
}

/// # Safety
/// The caller is required to ensure that the launch information accurately
/// describes the state of the kernel heap since its page will be consumed
/// for page table usage by this function.
pub unsafe fn expand_kernel_heap(
    launch_info: &mut KernelLaunchInfo,
    region_end_paddr: PhysAddr,
    platform: &dyn SvsmPlatform,
) -> usize {
    // This code is not elegantly integrated with its caller, nor does it
    // make effective use of existing page table code.  However, this routine
    // will cease to exist once the direct map is removed, so there is little
    // point in building it for long-term maintainability.

    // Calculate the bounds of the existing direct map so it can be expanded.
    let map_size =
        (launch_info.kernel_region_phys_end - launch_info.kernel_region_phys_start) as usize;
    let mut paddr = PhysAddr::from(launch_info.kernel_region_phys_start) + map_size;
    let mut vaddr = VirtAddr::from(launch_info.kernel_direct_map_vaddr) + map_size;

    // Calculate an offset from physical address to virtual address in the
    // direct map.  Since kernel virtual addresses have their upper bits set
    // and physical addresses do not, this can be a simple subtraction.
    let phys_virt_diff = launch_info.kernel_direct_map_vaddr - launch_info.kernel_region_phys_start;

    // Determine the next page in the heap available for allocation in case
    // additional page table page allocation is necessary.
    let mut allocated_pages = launch_info.heap_area_allocated as usize;
    let heap_phys_start =
        PhysAddr::from(launch_info.kernel_region_phys_start + launch_info.heap_area_offset);

    // Calculate the bounds of the expanded kernel region.  It must be aligned
    // to a 2 MB boundary.  If this restriction prohibits expansion, then
    // simply return without doing work.
    let kernel_region_end = align_down(u64::from(region_end_paddr), PAGE_SIZE_2M as u64);
    if kernel_region_end <= launch_info.kernel_region_phys_end {
        return allocated_pages;
    }

    let mut expansion_size = kernel_region_end - launch_info.kernel_region_phys_end;

    // Locate the PML4E that maps the kernel region.
    let pml4e_vaddr = VirtAddr::from(launch_info.kernel_page_table_vaddr)
        + PageTable::index::<3>(vaddr) * mem::size_of::<PTEntry>();
    // SAFETY: the launch info is trusted to provide a valid address for the
    // kernel paging root.
    let pml4e = unsafe { PTEntry::read_pte(pml4e_vaddr) };

    // Obtain the virtual address of the page that is described by the PML4E.
    // Because no global physical/virtual mappings have been established yet,
    // this must be done manually.
    let pdpe_vaddr = VirtAddr::from(u64::from(pml4e.address()) + phys_virt_diff);

    // Capture a raw pointer to the (live) page directory pointer table page.
    // `from_vaddr` cannot be used here: the table is installed, so it must be
    // accessed through the volatile element accessors rather than a `&mut`.
    let pdpt = pdpe_vaddr.as_mut_ptr::<PTPage>();

    // Calculate the page table entry flags that will be used for intermediate
    // entries and for leaf entries.
    let pxe_flags = PTEntryFlags::PRESENT
        | PTEntryFlags::ACCESSED
        | PTEntryFlags::WRITABLE
        | PTEntryFlags::DIRTY;
    let pte_flags = PTEntryFlags::PRESENT
        | PTEntryFlags::ACCESSED
        | PTEntryFlags::WRITABLE
        | PTEntryFlags::DIRTY
        | PTEntryFlags::HUGE
        | PTEntryFlags::GLOBAL
        | PTEntryFlags::NX;

    // Capture a reference to the current page directory page.  It is initially
    // unknown.
    let mut pdt_vaddr = VirtAddr::null();

    while expansion_size != 0 {
        // If the next address falls on a PDPE boundary, then the page
        // directory table address will need to be recalculated.
        let pde_index = PageTable::index::<1>(vaddr);
        if pde_index == 0 {
            pdt_vaddr = VirtAddr::null();
        }

        if pdt_vaddr == VirtAddr::null() {
            // Calculate the address of the page directory page that will
            // contain the large page PDE, allocating pages if necessary.
            let pdpe_index = PageTable::index::<2>(vaddr);
            // SAFETY: `pdpt` points at the live page directory pointer table
            // and `pdpe_index < ENTRY_COUNT`. A volatile read forms no
            // reference into the live frame.
            let mut pdpe = unsafe { PTPage::read_entry_at(pdpt, pdpe_index) };
            if !pdpe.present() {
                // Allocate a new page to use as the page directory page.
                let new_pdpe_paddr = heap_phys_start + allocated_pages * PAGE_SIZE;
                let new_pdpe_vaddr = VirtAddr::from(u64::from(new_pdpe_paddr) + phys_virt_diff);
                allocated_pages += 1;

                // Ensure that the newly allocated page is filled with zeroes.
                // SAFETY: the newly allocated page is known to be unused
                // because it is outside the bounds of allocated hep pages, and
                // therefore it can be written safely.
                unsafe {
                    ptr::write_bytes(new_pdpe_vaddr.as_mut_ptr::<u8>(), 0, PAGE_SIZE);
                }

                pdpe.set(make_private_address(new_pdpe_paddr), pxe_flags);
                // SAFETY: as above; publish the new PDPE via a volatile write.
                unsafe { PTPage::write_entry_at(pdpt, pdpe_index, pdpe) };
            }
            pdt_vaddr = VirtAddr::from(u64::from(pdpe.address()) + phys_virt_diff);
        }

        // Raw pointer to the (live) page directory page; accessed via the
        // volatile element accessors, not a `&mut`.
        let pdt = pdt_vaddr.as_mut_ptr::<PTPage>();

        // Fill in the PDE for the next direct map address in sequence.
        // SAFETY: `pdt` points at the live page directory and
        // `pde_index < ENTRY_COUNT`; the volatile read/write form no
        // reference into the live frame.
        let mut pde = unsafe { PTPage::read_entry_at(pdt, pde_index) };
        pde.set(make_private_address(paddr), pte_flags);
        // SAFETY: as above.
        unsafe { PTPage::write_entry_at(pdt, pde_index, pde) };

        // By the time kernel region expansion is performed, the heap area has
        // been fully validated.  Now that a new large page has been added,
        // that memory must be validated as well.
        // SAFETY: the newly mapped page is being used for the first
        // time, so it can be validated safely.
        unsafe {
            platform
                .validate_virtual_page_range(
                    MemoryRegion::new(vaddr, PAGE_SIZE_2M),
                    PageValidateOp::Validate,
                )
                .expect("Failed to validate heap memory");
        }

        paddr = paddr + PAGE_SIZE_2M;
        vaddr = vaddr + PAGE_SIZE_2M;
        expansion_size -= PAGE_SIZE_2M as u64;
    }

    log::info!(
        "Kernel region expanded to {:#018x}-{paddr:#018x}",
        launch_info.kernel_region_phys_start
    );

    // Modify the launch info block to indicate the fully expanded size of the
    // kernel region.
    launch_info.kernel_region_phys_end = paddr.into();

    allocated_pages
}
