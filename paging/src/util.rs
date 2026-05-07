// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2022-2023 SUSE LLC
//
// Author: Joerg Roedel <jroedel@suse.de>

use crate::types::PAGE_SIZE;
use core::ops::{Add, BitAnd, Not, Sub};

use verus_stub::*;

#[cfg(verus_keep_ghost)]
use verify_proof::align::*;

#[verus_spec(ret =>
    requires
        align_up_requires((addr, align)),
    ensures
        align_up_ens((addr, align), ret),
)]
pub fn align_up<T>(addr: T, align: T) -> T
where
    T: Add<Output = T> + Sub<Output = T> + BitAnd<Output = T> + Not<Output = T> + From<u8> + Copy,
{
    let mask: T = align - T::from(1u8);
    (addr + mask) & !mask
}

#[verus_spec(ret =>
    requires
        align_down_requires((addr, align)),
    ensures
        align_down_ens((addr, align), ret),
)]
pub fn align_down<T>(addr: T, align: T) -> T
where
    T: Sub<Output = T> + Not<Output = T> + BitAnd<Output = T> + From<u8> + Copy,
{
    addr & !(align - T::from(1u8))
}

#[verus_spec(ret =>
    requires
        is_aligned_requires((addr, align)),
    ensures
        is_aligned_ens((addr, align), ret)
)]
pub fn is_aligned<T>(addr: T, align: T) -> bool
where
    T: Sub<Output = T> + BitAnd<Output = T> + PartialEq + From<u8>,
{
    (addr & (align - T::from(1u8))) == T::from(0u8)
}

pub fn page_align_up(x: usize) -> usize {
    align_up(x, PAGE_SIZE)
}

pub fn round_to_pages(x: usize) -> usize {
    page_align_up(x) / PAGE_SIZE
}

pub fn page_offset(x: usize) -> usize {
    x & (PAGE_SIZE - 1)
}

pub fn overlap<T>(x1: T, x2: T, y1: T, y2: T) -> bool
where
    T: PartialOrd,
{
    x1 <= y2 && y1 <= x2
}
