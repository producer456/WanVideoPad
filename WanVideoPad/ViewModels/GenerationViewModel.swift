// Generation ViewModel
// Manages UI state and orchestrates the pipeline

import Foundation
import SwiftUI
import MLX

@MainActor
class GenerationViewModel: ObservableObject {
    @Published var prompt: String = "A golden retriever walking on grass"
    @Published var isGenerating: Bool = false
    @Published var phase: PipelinePhase = .idle
    @Published var progress: Float = 0.0
    @Published var statusText: String = "Ready"
    @Published var videoURL: URL?
    @Published var errorMessage: String?

    @Published var width: Int = 720
    @Published var height: Int = 416
    @Published var frames: Int = 17
    @Published var steps: Int = 20
    @Published var guidance: Float = 5.0
    @Published var seed: String = ""

    // Download state
    @Published var isDownloading: Bool = false
    @Published var downloadProgress: Float = 0.0
    @Published var downloadStatus: String = ""
    @Published var currentDownloadFile: String = ""

    var modelsReady: Bool {
        WeightLoader.allModelsDownloaded()
    }

    func downloadModels() {
        guard !isDownloading else { return }
        isDownloading = true
        downloadProgress = 0
        downloadStatus = "Starting download..."
        errorMessage = nil

        Task {
            do {
                for file in WeightLoader.ModelFile.allCases {
                    self.currentDownloadFile = file.rawValue
                    self.downloadStatus = "Downloading \(file.rawValue)..."
                    try await WeightLoader.download(file)
                    self.downloadProgress = Float(WeightLoader.ModelFile.allCases.firstIndex(of: file)! + 1) / Float(WeightLoader.ModelFile.allCases.count)
                }
                self.isDownloading = false
                self.downloadStatus = "All models downloaded!"
                self.objectWillChange.send()
            } catch {
                self.isDownloading = false
                self.errorMessage = "Download failed: \(error.localizedDescription)"
                self.downloadStatus = "Failed"
            }
        }
    }

    func generate() {
        guard !isGenerating else { return }
        guard modelsReady else {
            errorMessage = "Models not downloaded yet"
            return
        }

        isGenerating = true
        errorMessage = nil
        progress = 0
        videoURL = nil

        let config = GenerationConfig(
            width: width,
            height: height,
            frames: frames,
            steps: steps,
            guidance: guidance,
            shift: 5.0,
            seed: seed.isEmpty ? nil : UInt64(seed)
        )

        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let pipeline = WanPipeline(config: config)

                pipeline.onPhaseChange = { phase in
                    Task { @MainActor in
                        self?.phase = phase
                        self?.statusText = phase.rawValue
                    }
                }

                pipeline.onProgress = { progress in
                    Task { @MainActor in
                        self?.progress = progress
                    }
                }

                let video = try await pipeline.generate(prompt: self?.prompt ?? "")

                // Save video
                await MainActor.run { self?.phase = .savingVideo }
                let outputURL = VideoExporter.temporaryVideoURL()
                try await VideoExporter.export(frames: video, to: outputURL)

                await MainActor.run {
                    self?.videoURL = outputURL
                    self?.phase = .done
                    self?.isGenerating = false
                    self?.statusText = "Done!"
                }
            } catch {
                await MainActor.run {
                    self?.errorMessage = error.localizedDescription
                    self?.phase = .error
                    self?.isGenerating = false
                    self?.statusText = "Error: \(error.localizedDescription)"
                }
            }
        }
    }
}
