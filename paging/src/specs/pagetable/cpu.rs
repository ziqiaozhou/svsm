use vstd::prelude::*;

verus!{
#[verifier::external_body]
pub tracked struct CpuIdentifier{
    no_copy: NoCopy,
}

impl View for CpuIdentifier {
    type V = int;
    uninterp spec fn view(&self) -> int;
}
}