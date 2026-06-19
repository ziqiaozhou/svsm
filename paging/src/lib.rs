// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) Microsoft Corporation
//
// Author: Ziqiao Zhou <ziqiaozhou@microsoft.com>

//! This crate provides page table–related functions and data structures.

#![no_std]

pub mod address;
pub mod pagetable;
pub mod sizes;
pub mod tlb;
pub mod traits;
pub mod util;
pub mod x86_64;
