// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2022-2023 SUSE LLC
//
// Author: Joerg Roedel <jroedel@suse.de>

//! x86_64-specific page table entry flags.

use bitflags::bitflags;

use crate::pagetable::GenericPageTableFlags;

bitflags! {
    /// x86_64 page table entry flags.
    ///
    /// Bit positions follow the Intel/AMD architecture manuals for
    /// 4-level (PML4) and 5-level (PML5) paging modes.
    #[derive(Copy, Clone, Debug, Default)]
    pub struct PTEntryFlags: usize {
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

impl GenericPageTableFlags for PTEntryFlags {
    const PRESENT: Self = Self::PRESENT;
    const WRITABLE: Self = Self::WRITABLE;
    const USER: Self = Self::USER;
    const ACCESSED: Self = Self::ACCESSED;
    const DIRTY: Self = Self::DIRTY;
    const HUGE: Self = Self::HUGE;
    const GLOBAL: Self = Self::GLOBAL;
    const NX: Self = Self::NX;
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
