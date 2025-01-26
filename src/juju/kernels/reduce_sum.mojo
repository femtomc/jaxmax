from gpu import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from math import ceildiv
from utils.index import IndexList
from max.tensor import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr


fn _reduce_sum_cpu(
    out: ManagedTensorSlice,
    x: ManagedTensorSlice[out.type, out.rank],
    axis: Int,
    ctx: MojoCallContextPtr,
):
    var vector_length = out.dim_size(axis)
    for i in range(vector_length):
        var idx = IndexList[out.rank](i)
        var result = lhs.load[1](idx) + rhs.load[1](idx)
        out.store[1](idx, result)


fn _reduce_sum_gpu(
    out: ManagedTensorSlice,
    x: ManagedTensorSlice[out.type, out.rank],
    axis: Int,
    ctx: MojoCallContextPtr,
) raises:
    alias BLOCK_SIZE = 16
    var gpu_ctx = ctx.get_device_context()
    var vector_length = out.dim_size(0)

    @parameter
    fn reduce_sum_gpu_kernel(length: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < length:
            var idx = IndexList[out.rank](tid)
            var result = lhs.load[1](idx) + rhs.load[1](idx)
            out.store[1](idx, result)

    var gpu_func = gpu_ctx.compile_function[reduce_sum_gpu_kernel]()
    var num_blocks = ceildiv(vector_length, BLOCK_SIZE)
    gpu_ctx.enqueue_function(
        gpu_func,
        vector_length,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


@compiler.register("reduce_sum", num_dps_outputs=1)
struct ReduceSum:
    @staticmethod
    fn execute[
        target: StringLiteral,
    ](
        out: ManagedTensorSlice[rank=1],
        x: ManagedTensorSlice[out.type, out.rank],
        axis: Int,
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _reduce_sum_cpu(out, lhs, rhs, ctx)
        elif target == "gpu":
            _reduce_sum_gpu(out, lhs, rhs, ctx)
        else:
            raise Error("No known target:", target)
