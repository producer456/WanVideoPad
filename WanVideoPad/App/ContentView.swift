import SwiftUI
import AVKit

struct ContentView: View {
    @StateObject private var viewModel = GenerationViewModel()
    @State private var showSettings = false

    var body: some View {
        VStack(spacing: 0) {
            // Video display area
            ZStack {
                Color.black

                if let url = viewModel.videoURL {
                    VideoPlayer(player: AVPlayer(url: url))
                } else if viewModel.isGenerating {
                    VStack(spacing: 20) {
                        ProgressView()
                            .scaleEffect(2)
                            .tint(.white)

                        Text(viewModel.statusText)
                            .font(.headline)
                            .foregroundStyle(.white)

                        if viewModel.phase == .denoising {
                            ProgressView(value: viewModel.progress)
                                .tint(.blue)
                                .frame(width: 300)
                            Text("Step \(Int(viewModel.progress * Float(viewModel.steps)))/\(viewModel.steps)")
                                .font(.caption)
                                .foregroundStyle(.white.opacity(0.7))
                        }
                    }
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "film")
                            .font(.system(size: 60))
                            .foregroundStyle(.white.opacity(0.3))
                        Text("Enter a prompt and tap Generate")
                            .foregroundStyle(.white.opacity(0.5))
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Bottom bar
            VStack(spacing: 12) {
                // Error display
                if let error = viewModel.errorMessage {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .lineLimit(2)
                }

                // Prompt input
                HStack(spacing: 12) {
                    TextField("Describe your video...", text: $viewModel.prompt, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...3)
                        .disabled(viewModel.isGenerating)

                    Button {
                        showSettings.toggle()
                    } label: {
                        Image(systemName: "gear")
                            .font(.title2)
                    }
                    .disabled(viewModel.isGenerating)

                    Button {
                        viewModel.generate()
                    } label: {
                        Image(systemName: "play.fill")
                            .font(.title2)
                            .frame(width: 44, height: 44)
                            .background(.blue)
                            .foregroundStyle(.white)
                            .clipShape(Circle())
                    }
                    .disabled(viewModel.isGenerating || !viewModel.modelsReady)
                }

                // Model status
                if !viewModel.modelsReady {
                    HStack {
                        Image(systemName: "exclamationmark.triangle")
                        Text("Models not found. Copy converted weights to App Support/WanVideoPad/models/")
                            .font(.caption)
                    }
                    .foregroundStyle(.orange)
                }
            }
            .padding()
            .background(.ultraThinMaterial)
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(viewModel: viewModel)
        }
    }
}

struct SettingsView: View {
    @ObservedObject var viewModel: GenerationViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Resolution") {
                    Picker("Size", selection: Binding(
                        get: { "\(viewModel.width)x\(viewModel.height)" },
                        set: { val in
                            let parts = val.split(separator: "x").compactMap { Int($0) }
                            if parts.count == 2 {
                                viewModel.width = parts[0]
                                viewModel.height = parts[1]
                            }
                        }
                    )) {
                        Text("720x416 (recommended)").tag("720x416")
                        Text("480x320 (fast)").tag("480x320")
                        Text("480x288 (low memory)").tag("480x288")
                    }
                }

                Section("Generation") {
                    Stepper("Frames: \(viewModel.frames)", value: $viewModel.frames, in: 5...41, step: 4)
                    Stepper("Steps: \(viewModel.steps)", value: $viewModel.steps, in: 10...50, step: 5)
                    HStack {
                        Text("Guidance: \(viewModel.guidance, specifier: "%.1f")")
                        Slider(value: $viewModel.guidance, in: 1...10)
                    }
                    TextField("Seed (empty = random)", text: $viewModel.seed)
                        .keyboardType(.numberPad)
                }

                Section("Info") {
                    LabeledContent("Model", value: "Wan2.1-1.3B")
                    LabeledContent("Resolution", value: "\(viewModel.width)x\(viewModel.height)")
                    LabeledContent("Est. Time", value: "~3-5 min")
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
