// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2022-2023 SUSE LLC
//
// Author: Carlos López <carlos.lopez@suse.com>

// Re-export all address types from the paging crate.
pub use paging::address::*;

use crate::mm::virt_to_phys;

#[derive(Clone, Copy, Debug)]
pub struct VirtPhysPair {
    pub vaddr: VirtAddr,
    pub paddr: PhysAddr,
}

impl VirtPhysPair {
    pub fn new(vaddr: VirtAddr) -> Self {
        Self {
            vaddr,
            paddr: virt_to_phys(vaddr),
        }
    }
}
