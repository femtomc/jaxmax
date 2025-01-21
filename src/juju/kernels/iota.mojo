import compiler
from math import iota
from utils.index import IndexList
from tensor_utils import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr


@compiler.register("iota", num_dps_outputs=1)
struct Iota:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](out: ManagedTensorSlice, ctx: MojoCallContextPtr,):
        @parameter
        @always_inline
        fn load_iota[
            width: Int
        ](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            var v = iota[out.type, width]()
            return v[idx[0]]

        foreach[load_iota, synchronous, target](out, ctx)
