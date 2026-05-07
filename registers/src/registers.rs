// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2022-2023 SUSE LLC
//
// Author: Joerg Roedel <jroedel@suse.de>

use bitflags::bitflags;

bitflags! {
    #[derive(Clone, Copy, Debug)]
    pub struct RFlags: usize {
        const CF    = 1 << 0;
        const FIXED = 1 << 1;
        const PF    = 1 << 2;
        const AF    = 1 << 4;
        const ZF    = 1 << 6;
        const SF    = 1 << 7;
        const TF    = 1 << 8;
        const IF    = 1 << 9;
        const DF    = 1 << 10;
        const OF    = 1 << 11;
        const IOPL  = 3 << 12;
        const NT    = 1 << 14;
        const MD    = 1 << 15;
        const RF    = 1 << 16;
        const VM    = 1 << 17;
        const AC    = 1 << 18;
        const VIF   = 1 << 19;
        const VIP   = 1 << 20;
        const ID    = 1 << 21;
    }
}
