// SPDX-License-Identifier: MIT

//! This crate provides page table–related functions and data structures.

#![no_std]

pub mod address;
pub mod pagetable;
pub mod sizes;
pub mod traits;
pub mod util;
pub mod x86_64;
