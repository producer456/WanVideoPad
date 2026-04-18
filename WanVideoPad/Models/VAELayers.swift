// VAE building blocks for Wan2.1
// CausalConv3d, ResidualBlock, AttentionBlock, Resample
// All use channels-last format (NTHWC)

import MLX
import MLXNN
import MLXFast
import Foundation

let CACHE_T = 2

/// Build temporal cache from the last CACHE_T frames
func createCacheEntry(_ x: MLXArray, existingCache: MLXArray? = nil) -> MLXArray {
    let t = x.dim(1)
    if t >= CACHE_T {
        return x[0..., (t - CACHE_T)..., 0..., 0..., 0...]
    } else {
        let cacheX = x[0..., (t - t)..., 0..., 0..., 0...]
        if let cache = existingCache {
            let oldFrames = cache[0..., (cache.dim(1) - (CACHE_T - t))..., 0..., 0..., 0...]
            return concatenated([oldFrames, cacheX], axis: 1)
        } else {
            let padT = CACHE_T - t
            let zeros = MLXArray.zeros([x.dim(0), padT, x.dim(2), x.dim(3), x.dim(4)], dtype: x.dtype)
            return concatenated([zeros, cacheX], axis: 1)
        }
    }
}

// MARK: - Causal Conv3d

class CausalConv3d: Module {
    let inChannels: Int
    let outChannels: Int
    let kernelSize: (Int, Int, Int)
    let stride: (Int, Int, Int)
    let padding: (Int, Int, Int)
    let temporalPad: Int
    let spatialPadH: Int
    let spatialPadW: Int
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?

    convenience init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int = 0, bias: Bool = true) {
        self.init(inChannels: inChannels, outChannels: outChannels,
                  kernelSize: (kernelSize, kernelSize, kernelSize),
                  stride: (stride, stride, stride),
                  padding: (padding, padding, padding), hasBias: bias)
    }

    init(inChannels: Int, outChannels: Int, kernelSize: (Int, Int, Int), stride: (Int, Int, Int) = (1,1,1), padding: (Int, Int, Int) = (0,0,0), hasBias: Bool = true) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.temporalPad = padding.0 * 2
        self.spatialPadH = padding.1
        self.spatialPadW = padding.2

        let scale = 1.0 / sqrt(Float(inChannels * kernelSize.0 * kernelSize.1 * kernelSize.2))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale,
            [outChannels, kernelSize.0, kernelSize.1, kernelSize.2, inChannels]
        )
        self._bias.wrappedValue = hasBias ? MLXArray.zeros([outChannels]) : nil
    }

    func callAsFunction(_ x: MLXArray, cacheX: MLXArray? = nil) -> MLXArray {
        var inp = x
        var tempPad = temporalPad

        if let cache = cacheX, temporalPad > 0 {
            inp = concatenated([cache, inp], axis: 1)
            tempPad = max(0, temporalPad - cache.dim(1))
        }

        if tempPad > 0 {
            inp = padded(inp, widths: [[0, 0], [tempPad, 0], [0, 0], [0, 0], [0, 0]])
        }

        if spatialPadH > 0 || spatialPadW > 0 {
            inp = padded(inp, widths: [
                [0, 0], [0, 0],
                [spatialPadH, spatialPadH],
                [spatialPadW, spatialPadW],
                [0, 0]
            ])
        }

        var y = conv3d(inp, weight, stride: [stride.0, stride.1, stride.2], padding: .init(integerLiteral: 0))
        if let b = bias {
            y = y + b
        }
        return y
    }
}

// MARK: - Resample

class Resample: Module {
    let dim: Int
    let mode: String
    let upsample: Upsample?
    let conv: Conv2d?
    let timeConv: CausalConv3d?

    init(dim: Int, mode: String) {
        self.dim = dim
        self.mode = mode

        switch mode {
        case "upsample2d":
            self.upsample = Upsample(scaleFactor: [2.0, 2.0], mode: .nearest)
            self.conv = Conv2d(inputChannels: dim, outputChannels: dim / 2, kernelSize: 3, stride: 1, padding: 0, bias: true)
            self.timeConv = nil
        case "upsample3d":
            self.upsample = Upsample(scaleFactor: [2.0, 2.0], mode: .nearest)
            self.conv = Conv2d(inputChannels: dim, outputChannels: dim / 2, kernelSize: 3, stride: 1, padding: 0, bias: true)
            self.timeConv = CausalConv3d(inChannels: dim, outChannels: dim * 2, kernelSize: (3, 1, 1), padding: (1, 0, 0))
        case "downsample2d", "downsample3d":
            self.upsample = nil
            self.conv = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 2, padding: 0, bias: true)
            self.timeConv = mode == "downsample3d" ?
                CausalConv3d(inChannels: dim, outChannels: dim, kernelSize: (3, 1, 1), stride: (2, 1, 1), padding: (0, 0, 0)) : nil
        default:
            fatalError("Unknown resample mode: \(mode)")
        }
    }

    func callAsFunction(_ x: MLXArray, cache: MLXArray? = nil) -> (MLXArray, MLXArray?) {
        let b = x.dim(0)
        let t = x.dim(1)
        let h = x.dim(2)
        let w = x.dim(3)
        let c = x.dim(4)
        var result = x
        var newCache: MLXArray? = nil

        if mode == "upsample3d" {
            if cache == nil {
                newCache = MLXArray.zeros([b, CACHE_T, h, w, c], dtype: x.dtype)
            } else {
                let cacheInput = result
                result = timeConv!(result, cacheX: cache)
                newCache = createCacheEntry(cacheInput, existingCache: cache)
                result = result.reshaped([b, t, h, w, 2, c])
                    .transposed(0, 1, 4, 2, 3, 5)
                    .reshaped([b, t * 2, h, w, c])
            }
        }

        let tOut = result.dim(1)
        let cOut = result.dim(4)
        result = result.reshaped([b * tOut, result.dim(2), result.dim(3), cOut])

        if mode == "upsample2d" || mode == "upsample3d" {
            result = upsample!(result)
            result = padded(result, widths: [[0, 0], [1, 1], [1, 1], [0, 0]])
            result = conv!(result)
        } else if mode == "downsample2d" || mode == "downsample3d" {
            result = padded(result, widths: [[0, 0], [0, 1], [0, 1], [0, 0]])
            result = conv!(result)
        }

        result = result.reshaped([b, tOut, result.dim(1), result.dim(2), result.dim(3)])

        if mode == "downsample3d" {
            if cache == nil {
                newCache = result
            } else {
                let lastFrame = cache![0..., (-1)..., 0..., 0..., 0...]
                let xWithCache = concatenated([lastFrame, result], axis: 1)
                newCache = result[0..., (-1)..., 0..., 0..., 0...]
                result = timeConv!(xWithCache, cacheX: nil)
            }
        }

        return (result, newCache)
    }
}

// MARK: - Residual Block

class ResidualBlock: Module {
    let inDim: Int
    let outDim: Int
    let norm1: RMSNorm
    let conv1: CausalConv3d
    let norm2: RMSNorm
    let conv2: CausalConv3d
    let shortcut: CausalConv3d?

    init(inDim: Int, outDim: Int) {
        self.inDim = inDim
        self.outDim = outDim
        self.norm1 = RMSNorm(dimensions: inDim, eps: 1e-12)
        self.conv1 = CausalConv3d(inChannels: inDim, outChannels: outDim, kernelSize: 3, padding: 1)
        self.norm2 = RMSNorm(dimensions: outDim, eps: 1e-12)
        self.conv2 = CausalConv3d(inChannels: outDim, outChannels: outDim, kernelSize: 3, padding: 1)
        self.shortcut = inDim != outDim ? CausalConv3d(inChannels: inDim, outChannels: outDim, kernelSize: 1) : nil
    }

    func callAsFunction(_ x: MLXArray, cache1: MLXArray?, cache2: MLXArray?) -> (MLXArray, MLXArray, MLXArray) {
        let h = shortcut != nil ? shortcut!(x) : x

        var residual = norm1(x)
        residual = silu(residual)
        let cacheInput1 = residual
        residual = conv1(residual, cacheX: cache1)
        let newCache1 = createCacheEntry(cacheInput1, existingCache: cache1)

        residual = norm2(residual)
        residual = silu(residual)
        let cacheInput2 = residual
        residual = conv2(residual, cacheX: cache2)
        let newCache2 = createCacheEntry(cacheInput2, existingCache: cache2)

        return (h + residual, newCache1, newCache2)
    }
}

// MARK: - Attention Block (Spatial, per-frame)

class VAEAttentionBlock: Module {
    let dim: Int
    let norm: RMSNorm
    let toQkv: Linear
    let proj: Linear

    init(dim: Int) {
        self.dim = dim
        self.norm = RMSNorm(dimensions: dim, eps: 1e-12)
        self.toQkv = Linear(dim, dim * 3)
        self.proj = Linear(dim, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let identity = x
        let b = x.dim(0)
        let t = x.dim(1)
        let h = x.dim(2)
        let w = x.dim(3)
        let c = x.dim(4)

        var xr = x.reshaped([b * t, h, w, c])
        xr = norm(xr)
        let qkv = toQkv(xr).reshaped([b * t, h * w, 3, c])
        let q = qkv[0..., 0..., 0, 0...].reshaped([b * t, 1, h * w, c])
        let k = qkv[0..., 0..., 1, 0...].reshaped([b * t, 1, h * w, c])
        let v = qkv[0..., 0..., 2, 0...].reshaped([b * t, 1, h * w, c])

        let scale = pow(Float(c), -0.5)
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil
        )
        let out = proj(attn.squeezed(axis: 1).reshaped([b * t, h, w, c]))
        return out.reshaped([b, t, h, w, c]) + identity
    }
}
