import compiler
from math import acos
from utils.index import IndexList
from max.tensor import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr


@compiler.register("acos", num_dps_outputs=1)
struct Acos:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice,
        x: ManagedTensorSlice[out.type, out.rank],
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn _acos[width: Int](idx: IndexList[x.rank]) -> SIMD[x.type, width]:
            return acos(x.load[width](idx))

        foreach[_acos, synchronous, target](out, ctx)
