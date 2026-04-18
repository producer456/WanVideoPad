// Transformer layers for Wan2.1 DiT
// Self-attention, cross-attention, and attention blocks with fused norm+modulate

import MLX
import MLXNN
import MLXFast
import Foundation

// MARK: - Self Attention

class WanSelfAttention: Module {
    let dim: Int
    let numHeads: Int
    let headDim: Int
    let qkv: Linear
    let o: Linear
    let normQ: RMSNorm
    let normK: RMSNorm

    init(dim: Int, numHeads: Int, eps: Float = 1e-6) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.qkv = Linear(dim, dim * 3)
        self.o = Linear(dim, dim)
        self.normQ = RMSNorm(dimensions: dim, eps: eps)
        self.normK = RMSNorm(dimensions: dim, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, gridSizes: [[Int]]) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let C = x.dim(2)
        let n = numHeads
        let d = headDim

        let qkvOut = qkv(x)
        let parts = split(qkvOut, parts: 3, axis: -1)
        var q = normQ(parts[0])
        var k = normK(parts[1])
        let v = parts[2]

        q = q.reshaped([B, L, n, d])
        k = k.reshaped([B, L, n, d])
        let vr = v.reshaped([B, L, n, d])

        // Apply 3D RoPE
        q = ropeApply(q, gridSizes: gridSizes, headDim: d)
        k = ropeApply(k, gridSizes: gridSizes, headDim: d)

        // Transpose to [B, n, L, d]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        let vt = vr.transposed(0, 2, 1, 3)

        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: vt,
            scale: pow(Float(d), -0.5), mask: nil
        )
        return o(attn.transposed(0, 2, 1, 3).reshaped([B, L, C]))
    }
}

// MARK: - Cross Attention

class WanCrossAttention: Module {
    let dim: Int
    let numHeads: Int
    let headDim: Int
    let q: Linear
    let kv: Linear
    let o: Linear
    let normQ: RMSNorm
    let normK: RMSNorm

    init(dim: Int, numHeads: Int, eps: Float = 1e-6) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.q = Linear(dim, dim)
        self.kv = Linear(dim, dim * 2)
        self.o = Linear(dim, dim)
        self.normQ = RMSNorm(dimensions: dim, eps: eps)
        self.normK = RMSNorm(dimensions: dim, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L1 = x.dim(1)
        let L2 = context.dim(1)
        let n = numHeads
        let d = headDim

        let qOut = normQ(q(x))
        let kvOut = kv(context)
        let kvParts = split(kvOut, parts: 2, axis: -1)
        let kOut = normK(kvParts[0])
        let vOut = kvParts[1]

        let qt = qOut.reshaped([B, L1, n, d]).transposed(0, 2, 1, 3)
        let kt = kOut.reshaped([B, L2, n, d]).transposed(0, 2, 1, 3)
        let vt = vOut.reshaped([B, L2, n, d]).transposed(0, 2, 1, 3)

        let attn = MLXFast.scaledDotProductAttention(
            queries: qt, keys: kt, values: vt,
            scale: pow(Float(d), -0.5), mask: nil
        )
        let out = attn.transposed(0, 2, 1, 3).reshaped([B, L1, dim])
        return o(out)
    }
}

// MARK: - Attention Block

class WanAttentionBlock: Module {
    let dim: Int
    let eps: Float
    let norm3: LayerNorm?
    let selfAttn: WanSelfAttention
    let crossAttn: WanCrossAttention
    let ffn: Sequential
    @ParameterInfo(key: "modulation") var modulation: MLXArray

    init(dim: Int, ffnDim: Int, numHeads: Int, crossAttnNorm: Bool = false, eps: Float = 1e-6) {
        self.dim = dim
        self.eps = eps
        self.norm3 = crossAttnNorm ? LayerNorm(dimensions: dim, eps: eps) : nil
        self.selfAttn = WanSelfAttention(dim: dim, numHeads: numHeads, eps: eps)
        self.crossAttn = WanCrossAttention(dim: dim, numHeads: numHeads, eps: eps)
        self.ffn = Sequential {
            Linear(dim, ffnDim)
            GELU(approximation: .tanh)
            Linear(ffnDim, dim)
        }
        self._modulation.wrappedValue = MLXArray.zeros([1, 6, dim])
    }

    func callAsFunction(_ x: MLXArray, e: MLXArray, gridSizes: [[Int]], context: MLXArray) -> MLXArray {
        let eLocal = modulation + e

        // Self-attention: fused LayerNorm(x, scale=e[:,1], shift=e[:,0])
        let xNormed = MLXFast.layerNorm(x, weight: eLocal[0, 1], bias: eLocal[0, 0], eps: eps)
        let y = selfAttn(xNormed, gridSizes: gridSizes)
        // x = x + y * gate (e[:,2])
        var xOut = x + y * eLocal[0..., 2]

        // Cross-attention
        let xForCross = norm3 != nil ? norm3!(xOut) : xOut
        xOut = xOut + crossAttn(xForCross, context: context)

        // FFN: fused LayerNorm(x, scale=e[:,4], shift=e[:,3])
        let xNormedFFN = MLXFast.layerNorm(xOut, weight: eLocal[0, 4], bias: eLocal[0, 3], eps: eps)
        let yFFN = ffn(xNormedFFN)
        xOut = xOut + yFFN * eLocal[0..., 5]

        return xOut
    }
}

// MARK: - Output Head

class Head: Module {
    let dim: Int
    let eps: Float
    let linear: Linear
    @ParameterInfo(key: "modulation") var modulation: MLXArray

    init(dim: Int, outDim: Int, patchSize: (Int, Int, Int), eps: Float = 1e-6) {
        self.dim = dim
        self.eps = eps
        let outFeatures = patchSize.0 * patchSize.1 * patchSize.2 * outDim
        self.linear = Linear(dim, outFeatures)
        self._modulation.wrappedValue = MLXArray.zeros([1, 2, dim])
    }

    func callAsFunction(_ x: MLXArray, tEmb: MLXArray) -> MLXArray {
        let e = modulation + tEmb.expandedDimensions(axis: 1)
        let xNormed = MLXFast.layerNorm(x, weight: e[0, 1], bias: e[0, 0], eps: eps)
        return linear(xNormed)
    }
}
