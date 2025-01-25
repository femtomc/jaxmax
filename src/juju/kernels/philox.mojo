import compiler
from complex import ComplexSIMD
from math import iota
from utils.index import IndexList
from max.tensor import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr

alias BLOCK_SIZE = (256, 256)
alias K_HI_32 = 0x9E3779B9
alias K_LO_32 = 0xBB67AE85
alias MUL_A = 0xCD9E8D57
alias MUL_B = 0xD2511F53


fn mul32_hi_lo(x: Float32, y: Float32) -> Tuple[Float32, Float32]:
    var xhi = x >> 16
    var yhi = y >> 16
    var xlo = x & 0xFFFF
    var ylo = y & 0xFFFF

    var xy_hi = xhi * yhi
    var xy_lo = xlo * ylo
    var cross_xy = xhi * ylo
    var cross_yx = xlo * yhi
    var carry = (cross_xy & 0xFFFF) + (cross_yx & 0xFFFF) + (xy_lo >> 16)
    return xy_hi + (cross_xy >> 16) + (cross_yx >> 16) + (carry >> 16), xy_lo


fn philox_4x32(
    owned hi0: Float32,
    owned lo0: Float32,
    owned hi1: Float32,
    owned lo1: Float32,
    owned k_hi: UInt32,
    owned k_lo: UInt32,
    rounds: Int32 = 10,
):
    pass


@compiler.register("philox", num_dps_outputs=1)
struct Philox:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice,
        key: ManagedTensorSlice[DType.uint32, 1],
        offset: UInt32,
        unpadded_shape: List[Int],
        ctx: MojoCallContextPtr,
    ):
        pass
