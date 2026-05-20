// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2022-2023 SUSE LLC
//
// Author: Ziqiao Zhou <ziqiaozhou@microsoft.com>
//
// Generic page table structures.

#![no_std]

pub mod address;
pub mod sizes;
pub mod traits;
pub mod util;
pub mod x86_64;

pub use address::{Address, PhysAddr, VirtAddr};
pub use sizes::{
    PAGE_SHIFT, PAGE_SHIFT_1G, PAGE_SHIFT_2M, PAGE_SIZE, PAGE_SIZE_1G, PAGE_SIZE_2M, PageSize,
};
pub use traits::{
    ArchPagingMeta, GenericPageTableFlags, PageLevel, PagingError, PagingHandler, PagingLevel,
    SelfMap,
};
pub use util::{
    align_down, align_up, bit_mask, is_aligned, overlap, page_align_up, page_offset, round_to_pages,
};
pub use x86_64::{PTEntryFlags, PdptLevel, Pml4Level};
