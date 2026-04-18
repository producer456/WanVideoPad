// T5-XXL Encoder for Wan2.1 text conditioning
// UMT5-XXL: 4096 dim, 24 layers, 64 heads, gated GELU FFN, per-layer relative position embeddings

import MLX
import MLXNN
import MLXFast
import Foundation

// MARK: - T5 Relative Position Embedding

class T5RelativeEmbedding: Module {
    let numBuckets: Int
    let numHeads: Int
    let bidirectional: Bool
    let maxDist: Int
    let embedding: Embedding

    init(numBuckets: Int = 32, numHeads: Int = 64, bidirectional: Bool = true, maxDist: Int = 128) {
        self.numBuckets = numBuckets
        self.numHeads = numHeads
        self.bidirectional = bidirectional
        self.maxDist = maxDist
        self.embedding = Embedding(embeddingCount: numBuckets, dimensions: numHeads)
    }

    func callAsFunction(_ lq: Int, _ lk: Int) -> MLXArray {
        let queryPos = MLXArray(0..<Int32(lq)).reshaped([lq, 1])
        let keyPos = MLXArray(0..<Int32(lk)).reshaped([1, lk])
        let relPos = keyPos - queryPos
        let relBuckets = relativePositionBucket(relPos)
        let embeds = embedding(relBuckets)
        // [lq, lk, numHeads] -> [1, numHeads, lq, lk]
        return embeds.transposed(2, 0, 1).expandedDimensions(axis: 0)
    }

    private func relativePositionBucket(_ relPos: MLXArray) -> MLXArray {
        let numBucketsLocal: Int
        var relBuckets: MLXArray
        let absRelPos: MLXArray

        if bidirectional {
            numBucketsLocal = numBuckets / 2
            relBuckets = (relPos .> 0).asType(.int32) * Int32(numBucketsLocal)
            absRelPos = abs(relPos)
        } else {
            numBucketsLocal = numBuckets
            relBuckets = MLXArray.zeros(like: relPos).asType(.int32)
            absRelPos = -minimum(relPos, MLXArray.zeros(like: relPos))
        }

        let maxExact = numBucketsLocal / 2
        let isSmall = absRelPos .< Int32(maxExact)
        let scale = Float(numBucketsLocal - maxExact) / log(Float(maxDist) / Float(maxExact))
        let relPosLarge = Int32(maxExact) + (
            log(absRelPos.asType(.float32) / Float(maxExact)) * scale
        ).asType(.int32)
        let clampedLarge = minimum(relPosLarge, MLXArray(Int32(numBucketsLocal - 1)))
        relBuckets = relBuckets + which(isSmall, absRelPos.asType(.int32), clampedLarge)
        return relBuckets
    }
}

// MARK: - T5 Attention

class T5Attention: Module {
    let numHeads: Int
    let headDim: Int
    let q: Linear
    let k: Linear
    let v: Linear
    let o: Linear

    init(dim: Int, dimAttn: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dimAttn / numHeads
        self.q = Linear(dim, dimAttn, bias: false)
        self.k = Linear(dim, dimAttn, bias: false)
        self.v = Linear(dim, dimAttn, bias: false)
        self.o = Linear(dimAttn, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil, mask: MLXArray? = nil, posBias: MLXArray? = nil) -> MLXArray {
        let ctx = context ?? x
        let b = x.dim(0)
        let n = numHeads
        let c = headDim

        let qArr = q(x).reshaped([b, x.dim(1), n, c]).transposed(0, 2, 1, 3)
        let kArr = k(ctx).reshaped([b, ctx.dim(1), n, c]).transposed(0, 2, 1, 3)
        let vArr = v(ctx).reshaped([b, ctx.dim(1), n, c]).transposed(0, 2, 1, 3)

        // Build attention bias
        var attnBias = MLXArray.zeros([b, n, qArr.dim(2), kArr.dim(2)], dtype: x.dtype)
        if let posBias = posBias {
            attnBias = attnBias + posBias
        }
        if let mask = mask {
            let expandedMask: MLXArray
            if mask.ndim == 2 {
                expandedMask = mask.expandedDimensions(axes: [1, 2])
            } else {
                expandedMask = mask.expandedDimensions(axis: 1)
            }
            attnBias = which(expandedMask .== 0, MLXArray(Float(-1e9)), attnBias)
        }

        // T5 does NOT use sqrt(d) scaling - scale=1.0
        let out = MLXFast.scaledDotProductAttention(
            queries: qArr, keys: kArr, values: vArr,
            scale: 1.0, mask: attnBias
        )
        let reshaped = out.transposed(0, 2, 1, 3).reshaped([b, x.dim(1), n * c])
        return o(reshaped)
    }
}

// MARK: - T5 Feed Forward (Gated GELU)

class T5FeedForward: Module {
    let gate: Linear
    let fc1: Linear
    let fc2: Linear

    init(dim: Int, dimFfn: Int) {
        self.gate = Linear(dim, dimFfn, bias: false)
        self.fc1 = Linear(dim, dimFfn, bias: false)
        self.fc2 = Linear(dimFfn, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return fc2(fc1(x) * geluApproximate(gate(x)))
    }
}

// MARK: - T5 Self Attention Layer

class T5SelfAttentionLayer: Module {
    let sharedPos: Bool
    let norm1: RMSNorm
    let attn: T5Attention
    let norm2: RMSNorm
    let ffn: T5FeedForward
    let posEmbedding: T5RelativeEmbedding?

    init(dim: Int, dimAttn: Int, dimFfn: Int, numHeads: Int, numBuckets: Int, sharedPos: Bool = true) {
        self.sharedPos = sharedPos
        self.norm1 = RMSNorm(dimensions: dim, eps: 1e-6)
        self.attn = T5Attention(dim: dim, dimAttn: dimAttn, numHeads: numHeads)
        self.norm2 = RMSNorm(dimensions: dim, eps: 1e-6)
        self.ffn = T5FeedForward(dim: dim, dimFfn: dimFfn)
        self.posEmbedding = sharedPos ? nil : T5RelativeEmbedding(numBuckets: numBuckets, numHeads: numHeads, bidirectional: true)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, posBias: MLXArray? = nil) -> MLXArray {
        let e = sharedPos ? posBias : posEmbedding!(x.dim(1), x.dim(1))
        var out = x + attn(norm1(x), mask: mask, posBias: e)
        out = out + ffn(norm2(out))
        return out
    }
}

// MARK: - T5 Encoder

class T5Encoder: Module {
    let dim: Int
    let numLayers: Int
    let sharedPos: Bool
    let tokenEmbedding: Embedding
    let posEmbedding: T5RelativeEmbedding?
    let blocks: [T5SelfAttentionLayer]
    let norm: RMSNorm

    init(
        vocabSize: Int = 256384,
        dim: Int = 4096,
        dimAttn: Int = 4096,
        dimFfn: Int = 10240,
        numHeads: Int = 64,
        numLayers: Int = 24,
        numBuckets: Int = 32,
        sharedPos: Bool = false
    ) {
        self.dim = dim
        self.numLayers = numLayers
        self.sharedPos = sharedPos
        self.tokenEmbedding = Embedding(embeddingCount: vocabSize, dimensions: dim)
        self.posEmbedding = sharedPos ? T5RelativeEmbedding(numBuckets: numBuckets, numHeads: numHeads, bidirectional: true) : nil
        self.blocks = (0..<numLayers).map { _ in
            T5SelfAttentionLayer(dim: dim, dimAttn: dimAttn, dimFfn: dimFfn, numHeads: numHeads, numBuckets: numBuckets, sharedPos: sharedPos)
        }
        self.norm = RMSNorm(dimensions: dim, eps: 1e-6)
    }

    func callAsFunction(_ ids: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var x = tokenEmbedding(ids)
        let seqLen = x.dim(1)
        let e = sharedPos ? posEmbedding!(seqLen, seqLen) : nil

        for block in blocks {
            x = block(x, mask: mask, posBias: e)
        }
        return norm(x)
    }
}
