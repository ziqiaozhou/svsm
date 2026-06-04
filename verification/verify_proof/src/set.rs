// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) Microsoft Corporation
//
// Author: Ziqiao Zhou <ziqiaozhou@microsoft.com>
use vstd::prelude::*;

verus! {

/// Proves that the set of usizes in the range [lo, hi) is finite with length hi - lo.
pub proof fn lemma_usize_range_finite(lo: usize, hi: int)
    requires
        lo <= hi <= usize::MAX + 1,
    ensures
        ISet::new(|k: usize| lo <= k < hi).finite(),
        ISet::new(|k: usize| lo <= k < hi).len() == hi - lo,
{
    vstd::iset_lib::lemma_int_range(lo as int, hi);
    let iset_int = vstd::iset_lib::set_int_range(lo as int, hi);
    let iset_usize: ISet<usize> = ISet::new(|k: usize| lo <= k < hi);
    // Map from the finite int range to the usize range
    let g = |k: int| k as usize;
    assert forall|k: int| iset_int.contains(k) implies iset_usize.contains(
        #[trigger] g(k),
    ) by {}
    assert forall|k: usize| iset_usize.contains(k) implies iset_int.contains(k as int) by {}
    assert(iset_int.map(g) =~= iset_usize) by {
        assert forall|u: usize| #[trigger] iset_usize.contains(u) implies iset_int.map(g).contains(u) by {
            assert(iset_int.contains(u as int));
            assert(g(u as int) == u);
        }
    }
    assert(iset_int.injective_on(g));
    vstd::iset_lib::lemma_map_size(iset_int, iset_usize, g);
}

} // verus!
