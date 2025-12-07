// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) Microsoft Corporation
//
// Author: Ziqiao Zhou <ziqiaozhou@microsoft.com>
use vstd::prelude::*;
use vstd::std_specs::convert::{FromSpec, IntoSpec};
verus! {

pub trait FromIntoInteger: Sized {
    spec fn from_spec(v: int) -> Self;

    spec fn into_spec(self) -> int;
}

macro_rules! def_primitive_from{
    ($toty: ty; $($fromty: ty),*) => {verus!{
        $(
            impl FromIntoInteger for $fromty {
                open spec fn into_spec(self) -> int {
                    self as int
                }

                open spec fn from_spec(v: int) -> Self {
                    v as Self
                }
            }
        )*
    }}
}

def_primitive_from!{int; u8, u16, u32, u64, usize, u128, int, nat}

} // verus!
