// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) Microsoft Corporation
//
// Author: Ziqiao Zhou <ziqiaozhou@microsoft.com>

#![no_std]
#![allow(unused_braces)]
#![allow(unexpected_cfgs)]
#![allow(missing_debug_implementations)]
use verus_builtin_macros::*;

pub mod bits;
#[cfg(verus_only)]
pub mod frac_perm;
#[cfg(verus_only)]
pub mod frac_ptr;
#[cfg(verus_only)]
pub mod nonlinear;
#[cfg(verus_only)]
pub mod set;
#[cfg(verus_only)]
pub mod sum;

verus! {

global size_of usize == 8;

} // verus!
