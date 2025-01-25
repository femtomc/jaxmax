import compiler
from utils.index import IndexList
from max.tensor import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr


@compiler.register("random_split", num_dps_outputs=1)
struct RandomSplit:
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
        fn random_split[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.type, width]:
            return x.load[width](idx) + 1

        foreach[random_split, synchronous, target](out, ctx)

    @staticmethod
    fn shape(
        x: ManagedTensorSlice,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"
