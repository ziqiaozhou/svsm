// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2022-2023 SUSE LLC
//
// Author: Joerg Roedel <jroedel@suse.de>

//! Generic page table types and structures for x86_64 paging.
//!
//! This module provides architecture-level page table types that are
//! independent of any specific kernel or OS implementation. Encryption
//! mask handling and kernel-specific allocation remain in the kernel crate.

use crate::address::{Address, PhysAddr, VirtAddr};
use crate::types::{PAGE_SIZE, PAGE_SIZE_1G, PAGE_SIZE_2M, PageSize};
use bitflags::bitflags;
use core::ops::{Index, IndexMut};
use registers::{CR0Flags, CR4Flags, EFERFlags};
use zerocopy::FromBytes;

/// Number of entries in a page table (4KB/8B).
pub const ENTRY_COUNT: usize = 512;

/// Trait for translating physical addresses to virtual addresses.
///
/// # Safety
///
/// Implementer must guarantee that the returned virtual address validly
/// maps the given physical address.
pub unsafe trait PageTableFrameMapping {
    fn paddr_to_vaddr(&self, paddr: PhysAddr) -> VirtAddr;
}

/// Trait for allocating and deallocating page table frames.
///
/// # Safety
///
/// Implementer must guarantee that `allocate_frame` returns unique, zeroed
/// frames suitable as page table pages.
pub unsafe trait FrameAllocator {
    type Error;
    fn allocate_frame(&mut self) -> Result<PhysAddr, Self::Error>;
    /// # Safety
    ///
    /// `paddr` must have been returned by `allocate_frame` and not yet freed.
    unsafe fn deallocate_frame(&mut self, paddr: PhysAddr);
}

bitflags! {
    #[derive(Copy, Clone, Debug, Default)]
    pub struct PTEntryFlags: u64 {
        const PRESENT       = 1 << 0;
        const WRITABLE      = 1 << 1;
        const USER          = 1 << 2;
        const ACCESSED      = 1 << 5;
        const DIRTY         = 1 << 6;
        const HUGE          = 1 << 7;
        const GLOBAL        = 1 << 8;
        const NX            = 1 << 63;
    }
}

impl PTEntryFlags {
    pub fn exec() -> Self {
        Self::PRESENT | Self::GLOBAL | Self::ACCESSED
    }

    pub fn data() -> Self {
        Self::PRESENT | Self::GLOBAL | Self::WRITABLE | Self::NX | Self::ACCESSED | Self::DIRTY
    }

    pub fn data_ro() -> Self {
        Self::PRESENT | Self::GLOBAL | Self::NX | Self::ACCESSED
    }

    pub fn task_exec() -> Self {
        Self::PRESENT | Self::ACCESSED
    }

    pub fn task_data() -> Self {
        Self::PRESENT | Self::WRITABLE | Self::NX | Self::ACCESSED | Self::DIRTY
    }

    pub fn task_data_ro() -> Self {
        Self::PRESENT | Self::NX | Self::ACCESSED
    }
}

/// Represents paging mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagingMode {
    /// Paging mode is disabled
    NoPaging,
    /// 32bit legacy paging mode
    NonPAE,
    /// 32bit PAE paging mode
    PAE,
    /// 4 level paging mode
    PML4,
    /// 5 level paging mode
    PML5,
}

impl PagingMode {
    pub fn new(efer: EFERFlags, cr0: CR0Flags, cr4: CR4Flags) -> Self {
        if !cr0.contains(CR0Flags::PG) {
            PagingMode::NoPaging
        } else if efer.contains(EFERFlags::LMA) {
            if cr4.contains(CR4Flags::LA57) {
                PagingMode::PML5
            } else {
                PagingMode::PML4
            }
        } else if cr4.contains(CR4Flags::PAE) {
            PagingMode::PAE
        } else {
            PagingMode::NonPAE
        }
    }
}

/// Represents a page table entry.
#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes)]
pub struct PTEntry(pub PhysAddr);

impl PTEntry {
    /// Check if the page table entry is clear (null).
    pub fn is_clear(&self) -> bool {
        self.0.is_null()
    }

    /// Clear the page table entry.
    pub fn clear(&mut self) {
        self.0 = PhysAddr::null();
    }

    /// Check if the page table entry is present.
    pub fn present(&self) -> bool {
        self.flags().contains(PTEntryFlags::PRESENT)
    }

    /// Check if the page table entry is huge.
    pub fn huge(&self) -> bool {
        self.flags().contains(PTEntryFlags::HUGE)
    }

    /// Check if the page table entry is writable.
    pub fn writable(&self) -> bool {
        self.flags().contains(PTEntryFlags::WRITABLE)
    }

    /// Check if the page table entry is NX (no-execute).
    pub fn nx(&self) -> bool {
        self.flags().contains(PTEntryFlags::NX)
    }

    /// Check if the page table entry is user-accessible.
    pub fn user(&self) -> bool {
        self.flags().contains(PTEntryFlags::USER)
    }

    /// Check if the page table entry is global.
    pub fn global(&self) -> bool {
        self.flags().contains(PTEntryFlags::GLOBAL)
    }

    /// Get the raw bits (`u64`) of the page table entry.
    pub fn raw(&self) -> u64 {
        self.0.bits() as u64
    }

    /// Get the flags of the page table entry.
    pub fn flags(&self) -> PTEntryFlags {
        PTEntryFlags::from_bits_truncate(self.0.bits() as u64)
    }

    /// Set the page table entry with the specified address and flags.
    /// This is the raw set — no flag filtering or encryption handling.
    pub fn set(&mut self, addr: PhysAddr, flags: PTEntryFlags) {
        let addr = addr.bits() as u64;
        assert_eq!(addr & !0x000f_ffff_ffff_f000, 0);
        self.0 = PhysAddr::from(addr | flags.bits());
    }

    /// Get the raw physical address from the page table entry.
    /// Does NOT strip any encryption/confidentiality bits.
    pub fn address(&self) -> PhysAddr {
        PhysAddr::from(self.raw() & 0x000f_ffff_ffff_f000)
    }

    /// Read a page table entry from the specified virtual address.
    ///
    /// # Safety
    ///
    /// Reads from an arbitrary virtual address, making this essentially a
    /// raw pointer read. The caller must be certain to calculate the correct
    /// address.
    pub unsafe fn read_pte(vaddr: VirtAddr) -> Self {
        // SAFETY: When the method's safety requirements are met, the raw
        // pointer read is safe.
        unsafe { *vaddr.as_ptr::<Self>() }
    }
}

/// A pagetable page with multiple entries.
#[repr(C)]
#[derive(Debug, FromBytes)]
pub struct PTPage {
    pub entries: [PTEntry; ENTRY_COUNT],
}

impl PTPage {
    /// Converts a pagetable entry to a mutable reference to a [`PTPage`],
    /// if the entry is present and not huge. Uses the provided mapping
    /// to translate the physical address.
    pub fn from_entry<A: PageTableFrameMapping>(
        entry: PTEntry,
        mapping: &A,
    ) -> Option<&'static mut Self> {
        let flags = entry.flags();
        if !flags.contains(PTEntryFlags::PRESENT) || flags.contains(PTEntryFlags::HUGE) {
            return None;
        }

        let address = mapping.paddr_to_vaddr(entry.address());
        // SAFETY: Every PTEntry points to a previously allocated page-table
        // page, so this pointer dereference is safe.
        Some(unsafe { Self::from_vaddr(address) })
    }

    /// Generates a `PTPage` from a virtual address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the virtual address is a valid page table.
    pub unsafe fn from_vaddr(vaddr: VirtAddr) -> &'static mut Self {
        // SAFETY: the caller guarantees the correctness of the virtual address.
        unsafe { &mut *vaddr.as_mut_ptr::<PTPage>() }
    }
}

/// Can be used to access page table entries by index.
impl Index<usize> for PTPage {
    type Output = PTEntry;

    fn index(&self, index: usize) -> &PTEntry {
        &self.entries[index]
    }
}

/// Can be used to modify page table entries by index.
impl IndexMut<usize> for PTPage {
    fn index_mut(&mut self, index: usize) -> &mut PTEntry {
        &mut self.entries[index]
    }
}

/// Mapping levels of page table entries.
#[derive(Debug)]
pub enum Mapping<'a> {
    Level3(&'a mut PTEntry),
    Level2(&'a mut PTEntry),
    Level1(&'a mut PTEntry),
    Level0(&'a mut PTEntry),
}

/// A physical address within a page frame.
#[derive(Clone, Copy, Debug)]
pub enum PageFrame {
    Size4K(PhysAddr),
    Size2M(PhysAddr),
    Size1G(PhysAddr),
}

impl PageFrame {
    /// Get the raw address from the page frame (no encryption stripping).
    pub fn address(&self) -> PhysAddr {
        match *self {
            Self::Size4K(pa) => pa,
            Self::Size2M(pa) => pa,
            Self::Size1G(pa) => pa,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Self::Size4K(_) => PAGE_SIZE,
            Self::Size2M(_) => PAGE_SIZE_2M,
            Self::Size1G(_) => PAGE_SIZE_1G,
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

/// Page table structure containing a root page with multiple entries.
/// Generic over the allocator/mapping implementation.
#[repr(C)]
#[derive(Debug)]
pub struct PageTable<A: PageTableFrameMapping + FrameAllocator> {
    root: PTPage,
    alloc: A,
}

impl<A: PageTableFrameMapping + FrameAllocator> PageTable<A> {
    /// Create a new page table with zeroed root and the given allocator.
    pub fn new(alloc: A) -> Self {
        Self {
            root: PTPage {
                entries: [PTEntry(PhysAddr::null()); ENTRY_COUNT],
            },
            alloc,
        }
    }

    /// Get a reference to the root page.
    pub fn root(&self) -> &PTPage {
        &self.root
    }

    /// Get a mutable reference to the root page.
    pub fn root_mut(&mut self) -> &mut PTPage {
        &mut self.root
    }

    /// Get a reference to the allocator.
    pub fn alloc(&self) -> &A {
        &self.alloc
    }

    /// Get a mutable reference to the allocator.
    pub fn alloc_mut(&mut self) -> &mut A {
        &mut self.alloc
    }

    /// Computes the index within a page table at the given level for a
    /// virtual address `vaddr`.
    pub fn index<const L: usize>(vaddr: VirtAddr) -> usize {
        vaddr.to_pgtbl_idx::<L>()
    }

    /// Copy an entry `entry` from another [`PageTable`].
    pub fn copy_entry(&mut self, other: &Self, entry: usize) {
        self.root.entries[entry] = other.root.entries[entry];
    }

    /// Walk the virtual address and return the corresponding mapping.
    pub fn walk_addr(&mut self, vaddr: VirtAddr) -> Mapping<'_> {
        Self::walk_addr_lvl3(&mut self.root, &self.alloc, vaddr)
    }

    /// Walks the page table at level 3.
    pub fn walk_addr_lvl3<'a>(page: &'a mut PTPage, alloc: &A, vaddr: VirtAddr) -> Mapping<'a> {
        let idx = Self::index::<3>(vaddr);
        let entry = page[idx];
        match PTPage::from_entry(entry, alloc) {
            Some(page) => Self::walk_addr_lvl2(page, alloc, vaddr),
            None => Mapping::Level3(&mut page[idx]),
        }
    }

    /// Walks the page table at level 2.
    pub fn walk_addr_lvl2<'a>(page: &'a mut PTPage, alloc: &A, vaddr: VirtAddr) -> Mapping<'a> {
        let idx = Self::index::<2>(vaddr);
        let entry = page[idx];
        match PTPage::from_entry(entry, alloc) {
            Some(page) => Self::walk_addr_lvl1(page, alloc, vaddr),
            None => Mapping::Level2(&mut page[idx]),
        }
    }

    /// Walks the page table at level 1.
    pub fn walk_addr_lvl1<'a>(page: &'a mut PTPage, alloc: &A, vaddr: VirtAddr) -> Mapping<'a> {
        let idx = Self::index::<1>(vaddr);
        let entry = page[idx];
        match PTPage::from_entry(entry, alloc) {
            Some(page) => Self::walk_addr_lvl0(page, vaddr),
            None => Mapping::Level1(&mut page[idx]),
        }
    }

    /// Walks the page table at level 0.
    pub fn walk_addr_lvl0(page: &mut PTPage, vaddr: VirtAddr) -> Mapping<'_> {
        let idx = Self::index::<0>(vaddr);
        Mapping::Level0(&mut page[idx])
    }

    /// Allocates page table levels down to a 4KB PTE.
    pub fn alloc_pte_4k(&mut self, vaddr: VirtAddr) -> Mapping<'_> {
        let m = Self::walk_addr_lvl3(&mut self.root, &self.alloc, vaddr);

        match m {
            Mapping::Level0(entry) => Mapping::Level0(entry),
            Mapping::Level1(entry) => {
                Self::alloc_pte_lvl1(entry, vaddr, PageSize::Regular, &mut self.alloc)
            }
            Mapping::Level2(entry) => {
                Self::alloc_pte_lvl2(entry, vaddr, PageSize::Regular, &mut self.alloc)
            }
            Mapping::Level3(entry) => {
                Self::alloc_pte_lvl3(entry, vaddr, PageSize::Regular, &mut self.alloc)
            }
        }
    }

    /// Allocates page table levels down to a 2MB PTE.
    pub fn alloc_pte_2m(&mut self, vaddr: VirtAddr) -> Mapping<'_> {
        let m = Self::walk_addr_lvl3(&mut self.root, &self.alloc, vaddr);

        match m {
            Mapping::Level0(entry) => Mapping::Level0(entry),
            Mapping::Level1(entry) => Mapping::Level1(entry),
            Mapping::Level2(entry) => {
                Self::alloc_pte_lvl2(entry, vaddr, PageSize::Huge, &mut self.alloc)
            }
            Mapping::Level3(entry) => {
                Self::alloc_pte_lvl3(entry, vaddr, PageSize::Huge, &mut self.alloc)
            }
        }
    }

    /// Allocate at level 3, descend to level 2.
    pub fn alloc_pte_lvl3<'a>(
        entry: &'a mut PTEntry,
        vaddr: VirtAddr,
        size: PageSize,
        alloc: &mut A,
    ) -> Mapping<'a> {
        let flags = entry.flags();

        if flags.contains(PTEntryFlags::PRESENT) {
            return Mapping::Level3(entry);
        }

        let Ok(paddr) = alloc.allocate_frame() else {
            return Mapping::Level3(entry);
        };

        let flags = PTEntryFlags::PRESENT
            | PTEntryFlags::WRITABLE
            | PTEntryFlags::USER
            | PTEntryFlags::ACCESSED;
        entry.set(paddr, flags);

        let vaddr_page = alloc.paddr_to_vaddr(entry.address());
        // SAFETY: We just allocated a zeroed frame at this address.
        let page = unsafe { PTPage::from_vaddr(vaddr_page) };

        let idx = Self::index::<2>(vaddr);
        Self::alloc_pte_lvl2(&mut page[idx], vaddr, size, alloc)
    }

    /// Allocate at level 2, descend to level 1.
    pub fn alloc_pte_lvl2<'a>(
        entry: &'a mut PTEntry,
        vaddr: VirtAddr,
        size: PageSize,
        alloc: &mut A,
    ) -> Mapping<'a> {
        let flags = entry.flags();

        if flags.contains(PTEntryFlags::PRESENT) {
            return Mapping::Level2(entry);
        }

        let Ok(paddr) = alloc.allocate_frame() else {
            return Mapping::Level2(entry);
        };

        let flags = PTEntryFlags::PRESENT
            | PTEntryFlags::WRITABLE
            | PTEntryFlags::USER
            | PTEntryFlags::ACCESSED;
        entry.set(paddr, flags);

        let vaddr_page = alloc.paddr_to_vaddr(entry.address());
        // SAFETY: We just allocated a zeroed frame at this address.
        let page = unsafe { PTPage::from_vaddr(vaddr_page) };

        let idx = Self::index::<1>(vaddr);
        Self::alloc_pte_lvl1(&mut page[idx], vaddr, size, alloc)
    }

    /// Allocate at level 1, descend to level 0.
    pub fn alloc_pte_lvl1<'a>(
        entry: &'a mut PTEntry,
        vaddr: VirtAddr,
        size: PageSize,
        alloc: &mut A,
    ) -> Mapping<'a> {
        let flags = entry.flags();

        if size == PageSize::Huge || flags.contains(PTEntryFlags::PRESENT) {
            return Mapping::Level1(entry);
        }

        let Ok(paddr) = alloc.allocate_frame() else {
            return Mapping::Level1(entry);
        };

        let flags = PTEntryFlags::PRESENT
            | PTEntryFlags::WRITABLE
            | PTEntryFlags::USER
            | PTEntryFlags::ACCESSED;
        entry.set(paddr, flags);

        let vaddr_page = alloc.paddr_to_vaddr(entry.address());
        // SAFETY: We just allocated a zeroed frame at this address.
        let page = unsafe { PTPage::from_vaddr(vaddr_page) };

        let idx = Self::index::<0>(vaddr);
        Mapping::Level0(&mut page[idx])
    }

    /// Maps a 4KB page with raw address and flags (no encryption handling).
    pub fn map_4k(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: PTEntryFlags,
    ) -> Result<(), A::Error> {
        let mapping = self.alloc_pte_4k(vaddr);

        match mapping {
            Mapping::Level0(entry) => {
                entry.set(paddr, flags);
                Ok(())
            }
            _ => {
                // Allocation failed — we need an error. Since we can't create
                // A::Error generically, attempt one more allocation to get the error.
                Err(self.alloc.allocate_frame().unwrap_err())
            }
        }
    }

    /// Maps a 2MB page with raw address and flags (no encryption handling).
    pub fn map_2m(
        &mut self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: PTEntryFlags,
    ) -> Result<(), A::Error> {
        assert!(vaddr.is_aligned(PAGE_SIZE_2M));
        assert!(paddr.is_aligned(PAGE_SIZE_2M));

        let mapping = self.alloc_pte_2m(vaddr);

        match mapping {
            Mapping::Level1(entry) => {
                entry.set(paddr, flags | PTEntryFlags::HUGE);
                Ok(())
            }
            _ => Err(self.alloc.allocate_frame().unwrap_err()),
        }
    }

    /// Unmaps a 4KB page.
    pub fn unmap_4k(&mut self, vaddr: VirtAddr) {
        let mapping = self.walk_addr(vaddr);

        match mapping {
            Mapping::Level0(entry) => entry.clear(),
            Mapping::Level1(entry) => assert!(!entry.present()),
            Mapping::Level2(entry) => assert!(!entry.present()),
            Mapping::Level3(entry) => assert!(!entry.present()),
        }
    }

    /// Unmaps a 2MB page.
    pub fn unmap_2m(&mut self, vaddr: VirtAddr) {
        assert!(vaddr.is_aligned(PAGE_SIZE_2M));

        let mapping = self.walk_addr(vaddr);

        match mapping {
            Mapping::Level0(_) => unreachable!(),
            Mapping::Level1(entry) => entry.clear(),
            Mapping::Level2(entry) => assert!(!entry.present()),
            Mapping::Level3(entry) => assert!(!entry.present()),
        }
    }

    /// Gets the physical address for a mapped `vaddr` or `None` if
    /// no such mapping exists.
    pub fn check_mapping(&mut self, vaddr: VirtAddr) -> Option<PhysAddr> {
        match self.walk_addr(vaddr) {
            Mapping::Level0(entry) => Some(entry.address()),
            Mapping::Level1(entry) => Some(entry.address()),
            _ => None,
        }
    }

    /// Retrieves the physical address of a mapping, including page offset.
    pub fn phys_addr(&mut self, vaddr: VirtAddr) -> Option<PhysAddr> {
        let mapping = self.walk_addr(vaddr);

        match mapping {
            Mapping::Level0(entry) => {
                let offset = vaddr.page_offset();
                if !entry.flags().contains(PTEntryFlags::PRESENT) {
                    return None;
                }
                Some(entry.address() + offset)
            }
            Mapping::Level1(entry) => {
                let offset = vaddr.bits() & (PAGE_SIZE_2M - 1);
                if !entry.flags().contains(PTEntryFlags::PRESENT)
                    || !entry.flags().contains(PTEntryFlags::HUGE)
                {
                    return None;
                }
                Some(entry.address() + offset)
            }
            Mapping::Level2(_) | Mapping::Level3(_) => None,
        }
    }
}
