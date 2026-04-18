// Weight Loader for Wan2.1 models
// Loads pre-converted .safetensors into MLX modules

import Foundation
import MLX
import MLXNN

class WeightLoader {
    static let modelsDir: URL = {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("WanVideoPad/models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    enum ModelFile: String, CaseIterable {
        case t5 = "t5_encoder.safetensors"
        case dit = "dit_1_3b.safetensors"
        case vae = "vae_decoder.safetensors"
        case tokenizer = "tokenizer.json"
    }

    static func isDownloaded(_ file: ModelFile) -> Bool {
        FileManager.default.fileExists(atPath: localPath(file).path)
    }

    static func allModelsDownloaded() -> Bool {
        ModelFile.allCases.allSatisfy { isDownloaded($0) }
    }

    static func localPath(_ file: ModelFile) -> URL {
        modelsDir.appendingPathComponent(file.rawValue)
    }

    /// Load safetensors weights into an MLX module
    @discardableResult
    static func loadWeights(into module: Module, from file: ModelFile) throws -> Module {
        let path = localPath(file)
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw WeightLoaderError.fileNotFound(file.rawValue)
        }

        // Load flat dict of arrays
        let weights = try loadArrays(url: path)

        // Convert flat "a.b.c" keys to nested ModuleParameters
        let parameters = ModuleParameters.unflattened(weights)

        // Apply to module
        let _ = module.update(parameters: parameters)
        return module
    }
}

enum WeightLoaderError: Error, LocalizedError {
    case fileNotFound(String)

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let name): return "Model file not found: \(name)"
        }
    }
}
