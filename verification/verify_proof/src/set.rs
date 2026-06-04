// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) Microsoft Corporation
//
// Author: Ziqiao Zhou <ziqiaozhou@microsoft.com>
use vstd::prelude::*;

verus! {

pub open spec fn set_usize_range(start: usize, end: int) -> Set<usize> {
    Set::<usize>::full().unwrap().filter(|u: usize| start <= u < end)
}

} // verus!
