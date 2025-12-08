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

#[verifier::external_type_specification]
#[verifier::external_body]
pub struct ExTryFromIntError(core::num::TryFromIntError);

#[verifier::external_trait_specification]
#[verifier::external_trait_extension(TryFromSpec via TryFromSpecImpl)]
pub trait ExTryFrom<T>: Sized {
    type ExternalTraitSpecificationFor: core::convert::TryFrom<T>;

    type Error;

    spec fn obeys_try_from_spec() -> bool;

    spec fn try_from_spec(v: T) -> Result<Self, Self::Error>;

    fn try_from(v: T) -> (ret: Result<Self, Self::Error>)
        ensures
            Self::obeys_try_from_spec() ==> ret == Self::try_from_spec(v),
    ;
}

#[verifier::external_trait_specification]
#[verifier::external_trait_extension(TryIntoSpec via TryIntoSpecImpl)]
pub trait ExTryInto<T>: Sized {
    type ExternalTraitSpecificationFor: core::convert::TryInto<T>;

    type Error;

    spec fn obeys_try_into_spec() -> bool;

    spec fn try_into_spec(self) -> Result<T, Self::Error>;

    fn try_into(self) -> (ret: Result<T, Self::Error>)
        ensures
            Self::obeys_try_into_spec() ==> ret == Self::try_into_spec(self),
    ;
}

impl<T, U: TryFrom<T>> TryIntoSpecImpl<U> for T {
    open spec fn obeys_try_into_spec() -> bool {
        <U as TryFromSpec<Self>>::obeys_try_from_spec()
    }

    open spec fn try_into_spec(self) -> Result<U, U::Error> {
        <U as TryFromSpec<Self>>::try_from_spec(self)
    }
}

pub assume_specification<T, U: TryFrom<T>>[ <T as TryInto<U>>::try_into ](a: T) -> (ret: Result<
    U,
    U::Error,
>)
    ensures
        call_ensures(U::try_from, (a,), ret),
;

macro_rules! impl_try_from_spec {
    ($from:ty => [$($to:ty)*]) => {
        verus!{
        $(
            pub assume_specification[ <$to as core::convert::TryFrom<$from>>::try_from ](a: $from) -> (ret: Result<$to, <$to as core::convert::TryFrom<$from>>::Error>);

        impl TryFromSpecImpl<$from> for $to {
            open spec fn obeys_try_from_spec() -> bool {
                true
            }

            #[verifier::inline]
            open spec fn try_from_spec(v: $from) -> Result<Self, Self::Error> {
                if $to::MIN <= v <= $to::MAX {
                    Ok(v as $to)
                } else {
                    Err(arbitrary())
                }
            }
        }
        )*
        }
    }
}

impl_try_from_spec! { u16 => [u8 i8] }

impl_try_from_spec! { u32 => [u8 u16 i8 i16 usize isize] }

impl_try_from_spec! { u64 => [u8 u16 u32 i8 i16 i32 usize isize] }

impl_try_from_spec! { u128 => [u8 u16 u32 u64 i8 i16 i32 i64 usize isize] }

impl_try_from_spec! { usize => [u8 u16 u32 u64 u128 i8 i16 i32 i64] }

} // verus!
