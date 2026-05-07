// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2022-2023 SUSE LLC
//
// Author: Ziqiao Zhou <ziqiaozhou@microsoft.com>
//
// Generic verified x86_64 page table structures.

#![allow(unexpected_cfgs)]
#![no_std]

pub mod address;
pub mod pagetable;
pub mod types;
pub mod util;

pub use address::{Address, PhysAddr, VirtAddr};
pub use pagetable::{
    ENTRY_COUNT, Mapping, PTEntry, PTEntryFlags, PTPage, PageFrame, PageTable, PageTableOps,
    PagingHandler, PagingMode, SelfMap,
};
pub use types::{
    PAGE_SHIFT, PAGE_SHIFT_1G, PAGE_SHIFT_2M, PAGE_SIZE, PAGE_SIZE_1G, PAGE_SIZE_2M, PageSize,
};
pub use util::{
    align_down, align_up, bit_mask, is_aligned, overlap, page_align_up, page_offset, round_to_pages,
};
