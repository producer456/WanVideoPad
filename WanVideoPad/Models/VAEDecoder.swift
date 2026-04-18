// Wan2.1 VAE Decoder
// Decodes latents [F, H/8, W/8, 16] to video [F*4, H, W, 3]
// Frame-by-frame decode with temporal cache for memory efficiency

import MLX
import MLXNN
import Foundation

// MARK: - Decoder3d

class Decoder3d: Module {
    let dim: Int
    let zDim: Int
    let conv1: CausalConv3d
    let middleRes1: ResidualBlock
    let middleAttn: VAEAttentionBlock
    let middleRes2: ResidualBlock
    let upsamples: [[Module]]
    let headNorm: RMSNorm
    let headConv: CausalConv3d
    let numCacheSlots: Int

    init(
        dim: Int = 96,
        zDim: Int = 16,
        dimMult: [Int] = [1, 2, 4, 4],
        numResBlocks: Int = 2,
        temporalUpsample: [Bool] = [true, true, false]
    ) {
        self.dim = dim
        self.zDim = zDim

        let dims = [dim * dimMult.last!] + dimMult.reversed().map { dim * $0 }

        self.conv1 = CausalConv3d(inChannels: zDim, outChannels: dims[0], kernelSize: 3, padding: 1)
        self.middleRes1 = ResidualBlock(inDim: dims[0], outDim: dims[0])
        self.middleAttn = VAEAttentionBlock(dim: dims[0])
        self.middleRes2 = ResidualBlock(inDim: dims[0], outDim: dims[0])

        var stages: [[Module]] = []
        for i in 0..<dims.count - 1 {
            var inDim = dims[i]
            let outDim = dims[i + 1]
            var stage: [Module] = []

            // Stages 1,2,3 halve the input dim
            if i == 1 || i == 2 || i == 3 {
                inDim = inDim / 2
            }

            for j in 0...(numResBlocks) {
                stage.append(ResidualBlock(inDim: j == 0 ? inDim : outDim, outDim: outDim))
            }

            if i != dimMult.count - 1 {
                let mode = temporalUpsample[i] ? "upsample3d" : "upsample2d"
                stage.append(Resample(dim: outDim, mode: mode))
            }
            stages.append(stage)
        }
        self.upsamples = stages

        self.headNorm = RMSNorm(dimensions: dims.last!, eps: 1e-12)
        self.headConv = CausalConv3d(inChannels: dims.last!, outChannels: 3, kernelSize: 3, padding: 1)

        // Count cache slots
        var n = 1 + 2 + 2 // conv1 + middleRes1 + middleRes2
        for stage in stages {
            for layer in stage {
                if layer is ResidualBlock { n += 2 }
                else if layer is Resample { n += 1 }
            }
        }
        n += 1 // headConv
        self.numCacheSlots = n
    }

    func callAsFunction(_ x: MLXArray, featCache: [MLXArray?]) -> (MLXArray, [MLXArray?]) {
        var cacheIdx = 0
        var newCache: [MLXArray?] = []
        var out = x

        // conv1
        let cacheInput0 = out
        out = conv1(out, cacheX: featCache[cacheIdx])
        newCache.append(createCacheEntry(cacheInput0, existingCache: featCache[cacheIdx]))
        cacheIdx += 1

        // middle
        let (mr1, c1a, c1b) = middleRes1(out, cache1: featCache[cacheIdx], cache2: featCache[cacheIdx + 1])
        out = mr1
        newCache.append(c1a); newCache.append(c1b)
        cacheIdx += 2

        out = middleAttn(out)

        let (mr2, c2a, c2b) = middleRes2(out, cache1: featCache[cacheIdx], cache2: featCache[cacheIdx + 1])
        out = mr2
        newCache.append(c2a); newCache.append(c2b)
        cacheIdx += 2

        // Upsample stages
        for stage in upsamples {
            for layer in stage {
                if let resBlock = layer as? ResidualBlock {
                    let (r, ca, cb) = resBlock(out, cache1: featCache[cacheIdx], cache2: featCache[cacheIdx + 1])
                    out = r
                    newCache.append(ca); newCache.append(cb)
                    cacheIdx += 2
                } else if let attnBlock = layer as? VAEAttentionBlock {
                    out = attnBlock(out)
                } else if let resample = layer as? Resample {
                    let (r, c) = resample(out, cache: featCache[cacheIdx])
                    out = r
                    newCache.append(c)
                    cacheIdx += 1
                }
            }
        }

        // Head
        out = headNorm(out)
        out = silu(out)
        let cacheInputHead = out
        out = headConv(out, cacheX: featCache[cacheIdx])
        newCache.append(createCacheEntry(cacheInputHead, existingCache: featCache[cacheIdx]))
        cacheIdx += 1

        return (out, newCache)
    }
}

// MARK: - WanVAE

class WanVAE: Module {
    let decoder: Decoder3d
    let conv2: CausalConv3d
    let zDim: Int

    // Latent normalization constants
    static let mean: [Float] = [
        -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
         0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921
    ]
    static let std: [Float] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ]

    override init() {
        self.decoder = Decoder3d()
        self.conv2 = CausalConv3d(inChannels: 16, outChannels: 16, kernelSize: 1)
        self.zDim = 16
    }

    func decode(_ z: MLXArray) -> MLXArray {
        // Add batch dim
        var x = z.expandedDimensions(axis: 0)

        // Unscale latents
        let scale = MLXArray(WanVAE.std.map { 1.0 / $0 }).reshaped([1, 1, 1, 1, zDim])
        let meanArr = MLXArray(WanVAE.mean).reshaped([1, 1, 1, 1, zDim])
        x = x / scale + meanArr

        // Pre-decoder conv
        x = conv2(x)

        // Decode frame by frame
        let numFrames = x.dim(1)
        var featCache: [MLXArray?] = Array(repeating: nil, count: decoder.numCacheSlots)
        var outputs: [MLXArray] = []

        for i in 0..<numFrames {
            let frame = x[0..., i..<(i+1), 0..., 0..., 0...]
            let (outFrame, newCache) = decoder(frame, featCache: featCache)
            eval(outFrame)
            // Also eval cache to release compute graph
            let nonNilCache = newCache.compactMap { $0 }
            if !nonNilCache.isEmpty {
                eval(nonNilCache)
            }
            featCache = newCache
            outputs.append(outFrame)
        }

        let out = concatenated(outputs, axis: 1)
        let clamped = clip(out, min: -1.0, max: 1.0)

        // Remove batch dim
        return clamped[0]
    }
}
