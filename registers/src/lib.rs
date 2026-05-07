// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2022-2023 SUSE LLC
//
// Author: Joerg Roedel <jroedel@suse.de>
//
// x86_64 CPU register flag definitions.

#![no_std]

pub mod control_regs;
pub mod registers;

pub use control_regs::{CR0Flags, CR4Flags, EFERFlags};
pub use registers::RFlags;
