// Wan2.1 Text-to-Video Pipeline
// Orchestrates 3 sequential phases with memory management:
// Phase 1: T5 text encoding (11.4 GB peak) -> free T5
// Phase 2: DiT denoising (7.5 GB peak) -> free DiT
// Phase 3: VAE decode (13.3 GB peak) -> done

import MLX
import MLXNN
import MLXRandom
import Foundation

enum PipelinePhase: String {
    case idle = "Idle"
    case loadingT5 = "Loading T5 Encoder..."
    case encodingText = "Encoding Text..."
    case loadingDiT = "Loading DiT Model..."
    case denoising = "Denoising..."
    case loadingVAE = "Loading VAE Decoder..."
    case decodingVAE = "Decoding Video..."
    case savingVideo = "Saving Video..."
    case done = "Done!"
    case error = "Error"
}

struct GenerationConfig {
    var width: Int = 720
    var height: Int = 416
    var frames: Int = 17
    var steps: Int = 20
    var guidance: Float = 5.0
    var shift: Float = 5.0
    var seed: UInt64? = nil
    var negativePrompt: String = "Text, watermarks, blurry image, JPEG artifacts"
}

class WanPipeline {
    let config: GenerationConfig
    var onPhaseChange: ((PipelinePhase) -> Void)?
    var onProgress: ((Float) -> Void)?

    // VAE constants
    let vaeStride = (4, 8, 8)
    let zDim = 16

    init(config: GenerationConfig = GenerationConfig()) {
        self.config = config
    }

    func generate(prompt: String) async throws -> MLXArray {
        // Set GPU cache limit to 0 to reduce swap pressure
        GPU.set(cacheLimit: 0)

        if let seed = config.seed {
            MLXRandom.seed(seed)
        }

        // === Phase 1: T5 Text Encoding ===
        onPhaseChange?(.loadingT5)
        let tokenizer = try T5TokenizerWrapper(tokenizerPath: WeightLoader.localPath(.tokenizer))

        let t5 = T5Encoder()
        try WeightLoader.loadWeights(into: t5, from: .t5)

        onPhaseChange?(.encodingText)
        let context = encodeText(prompt, tokenizer: tokenizer, t5: t5)
        let contextNull = encodeText(config.negativePrompt, tokenizer: tokenizer, t5: t5)
        eval(context, contextNull)

        // Free T5
        // (In Swift, setting to nil + clearCache should release memory)
        GPU.clearCache()

        let peakT5 = GPU.peakMemory
        print("Peak memory T5: \(Float(peakT5) / 1e9) GB")
        GPU.resetPeakMemory()

        // === Phase 2: DiT Denoising ===
        onPhaseChange?(.loadingDiT)
        let dit = WanModel(
            patchSize: (1, 2, 2),
            inDim: 16, dim: 1536, ffnDim: 8960, freqDim: 256,
            textDim: 4096, outDim: 16, numHeads: 12, numLayers: 30,
            crossAttnNorm: true
        )
        try WeightLoader.loadWeights(into: dit, from: .dit)

        onPhaseChange?(.denoising)
        let latents = try denoise(dit: dit, context: context, contextNull: contextNull)
        eval(latents)

        // Free DiT
        GPU.clearCache()

        let peakDiT = GPU.peakMemory
        print("Peak memory DiT: \(Float(peakDiT) / 1e9) GB")
        GPU.resetPeakMemory()

        // === Phase 3: VAE Decode ===
        onPhaseChange?(.loadingVAE)
        let vae = WanVAE()
        try WeightLoader.loadWeights(into: vae, from: .vae)

        onPhaseChange?(.decodingVAE)
        let video = vae.decode(latents)
        eval(video)

        let peakVAE = GPU.peakMemory
        print("Peak memory VAE: \(Float(peakVAE) / 1e9) GB")

        GPU.clearCache()
        onPhaseChange?(.done)

        return video
    }

    private func encodeText(_ text: String, tokenizer: T5TokenizerWrapper, t5: T5Encoder) -> MLXArray {
        let tokens = tokenizer.encode(text)
        let embeddings = t5(tokens.inputIds, mask: tokens.attentionMask)
        let seqLen = Int(sum(tokens.attentionMask).item(Int.self))
        var context = embeddings[0, ..<seqLen, 0...]
        if seqLen < 512 {
            let padding = MLXArray.zeros([512 - seqLen, context.dim(-1)])
            context = concatenated([context, padding], axis: 0)
        }
        return context
    }

    private func denoise(dit: WanModel, context: MLXArray, contextNull: MLXArray) throws -> MLXArray {
        let W = config.width
        let H = config.height
        let targetShape = [
            (config.frames - 1) / vaeStride.0 + 1,
            H / vaeStride.1,
            W / vaeStride.2,
            zDim
        ]

        // Initial noise
        var xT = MLXRandom.normal(targetShape).asType(.bfloat16)

        // Set up sampler
        let sampler = FlowUniPCMultistepScheduler()
        sampler.setTimesteps(config.steps, shift: config.shift)

        guard let timesteps = sampler.timesteps else {
            throw PipelineError.samplerNotInitialized
        }

        // Denoising loop
        let numSteps = config.steps
        for stepIdx in 0..<numSteps {
            let t = timesteps[stepIdx].reshaped([1]).asType(.float32)

            // Conditional prediction
            let (noiseCond, _) = dit(xT, t: t, context: context)

            // Unconditional prediction (CFG)
            let noiseUncond: MLXArray
            if config.guidance > 1.0 {
                let (nu, _) = dit(xT, t: t, context: contextNull)
                noiseUncond = nu
            } else {
                noiseUncond = noiseCond
            }

            // CFG combine
            let noisePred = noiseUncond + config.guidance * (noiseCond - noiseUncond)

            // Sampler step
            xT = sampler.step(noisePred, timestep: t, sample: xT)
            eval(xT)

            let progress = Float(stepIdx + 1) / Float(numSteps)
            onProgress?(progress)
        }

        return xT
    }
}

enum PipelineError: Error {
    case samplerNotInitialized
    case generationFailed(String)
}
