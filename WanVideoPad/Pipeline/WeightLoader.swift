// Weight Loader for Wan2.1 models
// Downloads from GitHub Releases and loads .safetensors into MLX modules

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

    static let releaseBaseURL = "https://github.com/producer456/WanVideoPad/releases/download/v1.0-weights"

    enum ModelFile: String, CaseIterable {
        case t5 = "t5_encoder.safetensors"
        case dit = "dit_1_3b.safetensors"
        case vae = "vae_decoder.safetensors"
        case tokenizer = "tokenizer.json"

        /// Remote file parts for this model (split files for GitHub's 2GB limit)
        var remoteParts: [String] {
            switch self {
            case .t5:
                return (UnicodeScalar("a").value...UnicodeScalar("f").value).map {
                    "t5_encoder.safetensors.part-a\(String(UnicodeScalar($0)!))"
                }
            case .dit:
                return ["dit_1_3b.safetensors.part-aa", "dit_1_3b.safetensors.part-ab"]
            case .vae:
                return ["vae_decoder.safetensors"]
            case .tokenizer:
                return ["tokenizer.json"]
            }
        }

        /// Whether this file needs reassembly from parts
        var isSplit: Bool {
            remoteParts.count > 1
        }
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

    /// Download a model file from GitHub Releases
    /// Handles split files: downloads parts, reassembles, deletes parts
    static func download(
        _ file: ModelFile,
        onProgress: @Sendable @escaping (Float, String) -> Void = { _, _ in }
    ) async throws {
        let finalPath = localPath(file)
        if FileManager.default.fileExists(atPath: finalPath.path) { return }

        let parts = file.remoteParts
        var partPaths: [URL] = []

        for (i, partName) in parts.enumerated() {
            let partURL = URL(string: "\(releaseBaseURL)/\(partName)")!
            let localPartPath = modelsDir.appendingPathComponent(partName)

            if FileManager.default.fileExists(atPath: localPartPath.path) {
                partPaths.append(localPartPath)
                continue
            }

            let statusText = parts.count > 1
                ? "Downloading \(file.rawValue) part \(i + 1)/\(parts.count)..."
                : "Downloading \(file.rawValue)..."
            onProgress(Float(i) / Float(parts.count), statusText)

            // Download with URLSession
            let (tempURL, response) = try await URLSession.shared.download(from: partURL)
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                throw WeightLoaderError.downloadFailed("HTTP error downloading \(partName)")
            }

            try FileManager.default.moveItem(at: tempURL, to: localPartPath)
            partPaths.append(localPartPath)
        }

        // Reassemble split files
        if file.isSplit {
            onProgress(0.95, "Reassembling \(file.rawValue)...")
            try reassemble(parts: partPaths, into: finalPath)
            // Clean up part files
            for partPath in partPaths {
                try? FileManager.default.removeItem(at: partPath)
            }
        }

        onProgress(1.0, "\(file.rawValue) ready")
    }

    /// Concatenate part files into a single file
    private static func reassemble(parts: [URL], into destination: URL) throws {
        FileManager.default.createFile(atPath: destination.path, contents: nil)
        let handle = try FileHandle(forWritingTo: destination)
        defer { handle.closeFile() }

        for partURL in parts {
            let data = try Data(contentsOf: partURL)
            handle.write(data)
        }
    }

    /// Download all model files
    static func downloadAll(
        onProgress: @Sendable @escaping (ModelFile, Float, String) -> Void
    ) async throws {
        for file in ModelFile.allCases {
            try await download(file) { progress, status in
                onProgress(file, progress, status)
            }
        }
    }

    /// Load safetensors weights into an MLX module
    @discardableResult
    static func loadWeights(into module: Module, from file: ModelFile) throws -> Module {
        let path = localPath(file)
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw WeightLoaderError.fileNotFound(file.rawValue)
        }

        let weights = try loadArrays(url: path)
        let parameters = ModuleParameters.unflattened(weights)
        let _ = module.update(parameters: parameters)
        return module
    }
}

enum WeightLoaderError: Error, LocalizedError {
    case fileNotFound(String)
    case downloadFailed(String)

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let name): return "Model file not found: \(name)"
        case .downloadFailed(let msg): return "Download failed: \(msg)"
        }
    }
}
