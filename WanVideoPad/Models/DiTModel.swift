// Wan2.1 DiT (Diffusion Transformer) for text-to-video
// 1.3B model: dim=1536, ffn_dim=8960, num_heads=12, num_layers=30

import MLX
import MLXNN
import MLXFast
import Foundation

/// Sinusoidal positional embedding for timesteps
func sinusoidalEmbedding1D(dim: Int, position: MLXArray) -> MLXArray {
    let half = dim / 2
    let pos = position.asType(.float32)
    let freqs = exp(-log(Float(10000)) * MLXArray(0..<Int32(half)).asType(.float32) / Float(half))
    let sinusoid = pos.expandedDimensions(axis: -1) * freqs.expandedDimensions(axis: 0)
    return concatenated([cos(sinusoid), sin(sinusoid)], axis: 1).asType(position.dtype)
}

class WanModel: Module {
    let patchSize: (Int, Int, Int)
    let dim: Int
    let freqDim: Int
    let patchEmbedding: Conv3d
    let textEmbedding: Sequential
    let timeEmbedding: Sequential
    let timeProjection: Sequential
    let blocks: [WanAttentionBlock]
    let head: Head

    init(
        patchSize: (Int, Int, Int) = (1, 2, 2),
        textLen: Int = 512,
        inDim: Int = 16,
        dim: Int = 2048,
        ffnDim: Int = 8192,
        freqDim: Int = 256,
        textDim: Int = 4096,
        outDim: Int = 16,
        numHeads: Int = 16,
        numLayers: Int = 32,
        crossAttnNorm: Bool = true,
        eps: Float = 1e-6
    ) {
        self.patchSize = patchSize
        self.dim = dim
        self.freqDim = freqDim

        self.patchEmbedding = Conv3d(
            inputChannels: inDim, outputChannels: dim,
            kernelSize: [patchSize.0, patchSize.1, patchSize.2],
            stride: [patchSize.0, patchSize.1, patchSize.2],
            bias: true
        )

        self.textEmbedding = Sequential {
            Linear(textDim, dim)
            GELU(approximation: .tanh)
            Linear(dim, dim)
        }

        self.timeEmbedding = Sequential {
            Linear(freqDim, dim)
            SiLU()
            Linear(dim, dim)
        }

        self.timeProjection = Sequential {
            SiLU()
            Linear(dim, 6 * dim)
        }

        self.blocks = (0..<numLayers).map { _ in
            WanAttentionBlock(dim: dim, ffnDim: ffnDim, numHeads: numHeads, crossAttnNorm: crossAttnNorm, eps: eps)
        }

        self.head = Head(dim: dim, outDim: outDim, patchSize: patchSize, eps: eps)
    }

    func callAsFunction(
        _ x: MLXArray,
        t: MLXArray,
        context: MLXArray
    ) -> (MLXArray, MLXArray) {
        // Patchify: [F, H, W, C] -> [1, F, H, W, C] -> conv3d -> [1, Fp, Hp, Wp, dim]
        let xPatched = patchEmbedding(x.expandedDimensions(axis: 0))
        let Fp = xPatched.dim(1)
        let Hp = xPatched.dim(2)
        let Wp = xPatched.dim(3)
        let gridSizes = [[Fp, Hp, Wp]]
        var xFlat = xPatched.reshaped([1, Fp * Hp * Wp, dim])

        // Embed context: [L, textDim] -> [1, L, dim]
        let ctxEmb = textEmbedding(context.expandedDimensions(axis: 0))

        // Time embedding
        let sinEmb = sinusoidalEmbedding1D(dim: freqDim, position: t)
        let tEmb = timeEmbedding(sinEmb)
        var e = timeProjection(tEmb)
        e = e.reshaped([1, 6, dim])

        // Store input for residual
        let xIn = xFlat

        // Transformer blocks
        for block in blocks {
            xFlat = block(xFlat, e: e, gridSizes: gridSizes, context: ctxEmb)
        }
        let newResidual = xFlat - xIn

        // Output head
        xFlat = head(xFlat, tEmb: tEmb)

        // Unpatchify: [1, seq_len, patch_features] -> [F, H, W, C]
        let pt = patchSize.0
        let ph = patchSize.1
        let pw = patchSize.2
        // x[0] shape: [Fp*Hp*Wp, pt*ph*pw*outDim]
        let xUnpatch = xFlat[0]
            .reshaped([Fp, Hp, Wp, pt, ph, pw, 16])
            .transposed(0, 3, 1, 4, 2, 5, 6)  // [Fp, pt, Hp, ph, Wp, pw, C]
            .reshaped([Fp * pt, Hp * ph, Wp * pw, 16])

        return (xUnpatch, newResidual)
    }

    /// Forward pass with precomputed time embedding and optional cached residual (for TeaCache)
    func callWithCache(
        _ x: MLXArray,
        t: MLXArray,
        context: MLXArray,
        blockResidual: MLXArray? = nil,
        precomputedTime: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, MLXArray) {
        // Patchify
        let xPatched = patchEmbedding(x.expandedDimensions(axis: 0))
        let Fp = xPatched.dim(1)
        let Hp = xPatched.dim(2)
        let Wp = xPatched.dim(3)
        let gridSizes = [[Fp, Hp, Wp]]
        var xFlat = xPatched.reshaped([1, Fp * Hp * Wp, dim])

        let ctxEmb = textEmbedding(context.expandedDimensions(axis: 0))

        // Time embedding
        let tEmb: MLXArray
        let e: MLXArray
        if let precomputed = precomputedTime {
            tEmb = precomputed.0
            e = precomputed.1.reshaped([1, 6, dim])
        } else {
            let sinEmb = sinusoidalEmbedding1D(dim: freqDim, position: t)
            tEmb = timeEmbedding(sinEmb)
            e = timeProjection(tEmb).reshaped([1, 6, dim])
        }

        // Apply blocks or cached residual
        let newResidual: MLXArray
        if let blockResidual = blockResidual {
            xFlat = xFlat + blockResidual
            newResidual = blockResidual
        } else {
            let xIn = xFlat
            for block in blocks {
                xFlat = block(xFlat, e: e, gridSizes: gridSizes, context: ctxEmb)
            }
            newResidual = xFlat - xIn
        }

        // Head + unpatchify
        xFlat = head(xFlat, tEmb: tEmb)
        let pt = patchSize.0
        let ph = patchSize.1
        let pw = patchSize.2
        let output = xFlat[0]
            .reshaped([Fp, Hp, Wp, pt, ph, pw, 16])
            .transposed(0, 3, 1, 4, 2, 5, 6)
            .reshaped([Fp * pt, Hp * ph, Wp * pw, 16])

        return (output, newResidual)
    }
}
