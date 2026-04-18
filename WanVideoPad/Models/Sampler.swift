// FlowUniPCMultistepScheduler for Wan2.1 denoising
// Adapted from HuggingFace diffusers for flow matching prediction

import MLX
import Foundation

/// log(alpha/sigma) in float64 on CPU for numerical stability
private func lambda64(_ alpha: MLXArray, _ sigma: MLXArray) -> MLXArray {
    // Metal GPU doesn't support float64, compute on CPU
    let a = alpha.asType(.float64)
    let s = sigma.asType(.float64)
    return (log(a, stream: .cpu) - log(s, stream: .cpu)).asType(.float32)
}

class FlowUniPCMultistepScheduler {
    let numTrainTimesteps: Int
    let solverOrder: Int
    let predictX0: Bool
    let solverType: String
    let lowerOrderFinal: Bool

    var sigmas: MLXArray?
    var timesteps: MLXArray?
    var numInferenceSteps: Int = 0
    var modelOutputs: [MLXArray?]
    var timestepList: [MLXArray?]
    var lowerOrderNums: Int = 0
    var lastSample: MLXArray?
    var stepIndex: Int?
    var thisOrder: Int = 1

    init(numTrainTimesteps: Int = 1000, solverOrder: Int = 2) {
        self.numTrainTimesteps = numTrainTimesteps
        self.solverOrder = solverOrder
        self.predictX0 = true
        self.solverType = "bh2"
        self.lowerOrderFinal = true
        self.modelOutputs = Array(repeating: nil, count: solverOrder)
        self.timestepList = Array(repeating: nil, count: solverOrder)
    }

    func setTimesteps(_ numSteps: Int, shift: Float = 5.0) {
        let sigmaMax = shift * (1.0 - 1.0 / Float(numTrainTimesteps)) / (1.0 + (shift - 1.0) * (1.0 - 1.0 / Float(numTrainTimesteps)))
        let sigmaMin = shift * (1.0 / Float(numTrainTimesteps)) / (1.0 + (shift - 1.0) * (1.0 / Float(numTrainTimesteps)))

        var sigmaVals = MLXArray.linspace(sigmaMax, sigmaMin, count: numSteps + 1)[..<numSteps]
        sigmaVals = shift * sigmaVals / (1.0 + (shift - 1.0) * sigmaVals)

        let ts = sigmaVals * Float(numTrainTimesteps)
        let sigmaLast = MLXArray([Float(0.0)])
        self.sigmas = concatenated([sigmaVals, sigmaLast]).asType(.float32)
        self.timesteps = ts.asType(.int32)
        self.numInferenceSteps = numSteps
        self.modelOutputs = Array(repeating: nil, count: solverOrder)
        self.timestepList = Array(repeating: nil, count: solverOrder)
        self.lowerOrderNums = 0
        self.lastSample = nil
        self.stepIndex = nil
    }

    private func sigmaToAlphaSigma(_ sigma: MLXArray) -> (MLXArray, MLXArray) {
        return (1.0 - sigma, sigma)
    }

    private func convertModelOutput(_ modelOutput: MLXArray, sample: MLXArray) -> MLXArray {
        let sigmaT = sigmas![stepIndex!]
        return sample - sigmaT * modelOutput
    }

    private func initStepIndex(_ timestep: MLXArray) {
        let diff = abs(timesteps! - timestep)
        stepIndex = Int(argMin(diff).item(Int.self))
    }

    func step(_ modelOutput: MLXArray, timestep: MLXArray, sample: MLXArray) -> MLXArray {
        if stepIndex == nil {
            initStepIndex(timestep)
        }

        let useCorrector = stepIndex! > 0 && lastSample != nil

        let modelOutputConvert = convertModelOutput(modelOutput, sample: sample)

        var currentSample = sample
        if useCorrector {
            currentSample = multistepUniCBhUpdate(
                thisModelOutput: modelOutputConvert,
                lastSample: lastSample!,
                thisSample: currentSample,
                order: thisOrder
            )
        }

        // Shift model outputs
        for i in 0..<(solverOrder - 1) {
            modelOutputs[i] = modelOutputs[i + 1]
            timestepList[i] = timestepList[i + 1]
        }
        modelOutputs[solverOrder - 1] = modelOutputConvert
        timestepList[solverOrder - 1] = timestep

        var order: Int
        if lowerOrderFinal {
            order = min(solverOrder, numInferenceSteps - stepIndex!)
        } else {
            order = solverOrder
        }
        thisOrder = min(order, lowerOrderNums + 1)

        lastSample = currentSample
        let prevSample = multistepUniPBhUpdate(
            modelOutput: modelOutput,
            sample: currentSample,
            order: thisOrder
        )

        if lowerOrderNums < solverOrder {
            lowerOrderNums += 1
        }
        stepIndex! += 1

        return prevSample
    }

    private func multistepUniPBhUpdate(modelOutput: MLXArray, sample: MLXArray, order: Int) -> MLXArray {
        let m0 = modelOutputs.last!!
        let x = sample

        let sigmaT = sigmas![stepIndex! + 1]
        let sigmaS0 = sigmas![stepIndex!]
        let (alphaT, sigmaTVal) = sigmaToAlphaSigma(sigmaT)
        let (alphaS0, sigmaS0Val) = sigmaToAlphaSigma(sigmaS0)

        let lambdaT = lambda64(alphaT, sigmaTVal)
        let lambdaS0 = lambda64(alphaS0, sigmaS0Val)
        let h = lambdaT - lambdaS0

        var rks: [MLXArray] = []
        var D1s: [MLXArray] = []

        for i in 1..<order {
            let si = stepIndex! - i
            let mi = modelOutputs[modelOutputs.count - 1 - i]!
            let (alphaSi, sigmaSi) = sigmaToAlphaSigma(sigmas![si])
            let lambdaSi = lambda64(alphaSi, sigmaSi)
            let rk = (lambdaSi - lambdaS0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        }

        rks.append(MLXArray(Float(1.0)))
        let rksStacked = stacked(rks)

        let hh = -h
        let hPhi1 = expm1(hh)
        var hPhiK = hPhi1 / hh - 1.0
        var factorialI: Float = 1.0
        let Bh = expm1(hh)

        var R: [MLXArray] = []
        var b: [MLXArray] = []
        for i in 1...(order) {
            R.append(pow(rksStacked, i - 1))
            b.append(hPhiK * factorialI / Bh)
            factorialI *= Float(i + 1)
            hPhiK = hPhiK / hh - 1.0 / factorialI
        }

        if !D1s.isEmpty {
            let D1sStacked = stacked(D1s, axis: 1)
            let rhosP: MLXArray
            if order == 2 {
                rhosP = MLXArray([Float(0.5)]).asType(x.dtype)
            } else {
                rhosP = MLXArray([Float(0.5)]).asType(x.dtype)
            }

            let xT_ = sigmaTVal / sigmaS0Val * x - alphaT * hPhi1 * m0
            let predShape = [rhosP.dim(0)] + Array(repeating: 1, count: D1sStacked.ndim - 1)
            let predRes = sum(rhosP.reshaped(predShape) * D1sStacked, axis: 1)
            let xT = xT_ - alphaT * Bh * predRes
            return xT.asType(x.dtype)
        } else {
            let xT = sigmaTVal / sigmaS0Val * x - alphaT * hPhi1 * m0
            return xT.asType(x.dtype)
        }
    }

    private func multistepUniCBhUpdate(thisModelOutput: MLXArray, lastSample: MLXArray, thisSample: MLXArray, order: Int) -> MLXArray {
        let m0 = modelOutputs.last!!
        let x = lastSample
        let modelT = thisModelOutput

        let sigmaT = sigmas![stepIndex!]
        let sigmaS0 = sigmas![stepIndex! - 1]
        let (alphaT, sigmaTVal) = sigmaToAlphaSigma(sigmaT)
        let (alphaS0, sigmaS0Val) = sigmaToAlphaSigma(sigmaS0)

        let lambdaT = lambda64(alphaT, sigmaTVal)
        let lambdaS0 = lambda64(alphaS0, sigmaS0Val)
        let h = lambdaT - lambdaS0

        var rks: [MLXArray] = []
        var D1s: [MLXArray] = []

        for i in 1..<order {
            let si = stepIndex! - (i + 1)
            let mi = modelOutputs[modelOutputs.count - 1 - i]!
            let (alphaSi, sigmaSi) = sigmaToAlphaSigma(sigmas![si])
            let lambdaSi = lambda64(alphaSi, sigmaSi)
            let rk = (lambdaSi - lambdaS0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        }

        rks.append(MLXArray(Float(1.0)))

        let hh = -h
        let hPhi1 = expm1(hh)
        let Bh = expm1(hh)

        let rhosC: MLXArray
        if order == 1 {
            rhosC = MLXArray([Float(0.5)]).asType(x.dtype)
        } else {
            rhosC = MLXArray([Float(0.5)]).asType(x.dtype)
        }

        let xT_ = sigmaTVal / sigmaS0Val * x - alphaT * hPhi1 * m0
        let D1t = modelT - m0

        if !D1s.isEmpty {
            let D1sStacked = stacked(D1s, axis: 1)
            let corrShape = [rhosC.dim(0) - 1] + Array(repeating: 1, count: D1sStacked.ndim - 1)
            let corrRes = sum(rhosC[..<(rhosC.dim(0) - 1)].reshaped(corrShape) * D1sStacked, axis: 1)
            let xT = xT_ - alphaT * Bh * (corrRes + rhosC[rhosC.dim(0) - 1] * D1t)
            return xT.asType(x.dtype)
        } else {
            let xT = xT_ - alphaT * Bh * (rhosC[rhosC.dim(0) - 1] * D1t)
            return xT.asType(x.dtype)
        }
    }
}
