import compiler
from math import iota
from utils.index import IndexList
from max.tensor import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr


@compiler.register("iota", num_dps_outputs=1)
struct Iota:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](out: ManagedTensorSlice, len: Int, ctx: MojoCallContextPtr,):
        iota[out.type](out.unsafe_ptr(), len, 0)
