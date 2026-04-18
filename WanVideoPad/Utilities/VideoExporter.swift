// Video Exporter using AVFoundation
// Converts MLXArray frames to MP4 video (no ffmpeg needed)

import Foundation
import AVFoundation
import CoreVideo
import MLX

class VideoExporter {
    /// Export video frames to MP4
    /// - Parameters:
    ///   - frames: MLXArray [T, H, W, 3] in [-1, 1] range
    ///   - outputURL: destination file URL
    ///   - fps: frames per second (default 16 for Wan2.1)
    static func export(frames: MLXArray, to outputURL: URL, fps: Int = 16) async throws {
        // Remove existing file
        try? FileManager.default.removeItem(at: outputURL)

        // Evaluate and convert to numpy-like array
        eval(frames)
        let numFrames = frames.dim(0)
        let height = frames.dim(1)
        let width = frames.dim(2)

        // Convert from [-1, 1] to [0, 255] UInt8
        let normalized = clip((frames + 1.0) / 2.0 * 255.0, min: 0, max: 255).asType(.uint8)

        // Setup AVAssetWriter
        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 5_000_000,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ]
        ]

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        input.expectsMediaDataInRealTime = false

        let sourceAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: sourceAttributes
        )

        writer.add(input)
        guard writer.startWriting() else {
            throw ExporterError.writerFailed(writer.error?.localizedDescription ?? "Unknown")
        }
        writer.startSession(atSourceTime: .zero)

        for i in 0..<numFrames {
            // Wait for input to be ready
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 10_000_000) // 10ms
            }

            let frameData = normalized[i]
            eval(frameData)

            // Create pixel buffer
            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(
                nil, width, height,
                kCVPixelFormatType_32BGRA, nil,
                &pixelBuffer
            )
            guard status == kCVReturnSuccess, let pb = pixelBuffer else {
                throw ExporterError.pixelBufferFailed
            }

            CVPixelBufferLockBaseAddress(pb, [])
            defer { CVPixelBufferUnlockBaseAddress(pb, []) }

            let baseAddress = CVPixelBufferGetBaseAddress(pb)!
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)

            // Copy RGB to BGRA
            let frameArray: [UInt8] = frameData.asArray(UInt8.self)
            for y in 0..<height {
                let dstRow = baseAddress.advanced(by: y * bytesPerRow)
                for x in 0..<width {
                    let srcIdx = (y * width + x) * 3
                    let dstPtr = dstRow.advanced(by: x * 4).assumingMemoryBound(to: UInt8.self)
                    dstPtr[0] = frameArray[srcIdx + 2] // B
                    dstPtr[1] = frameArray[srcIdx + 1] // G
                    dstPtr[2] = frameArray[srcIdx + 0] // R
                    dstPtr[3] = 255                     // A
                }
            }

            let presentationTime = CMTime(value: Int64(i), timescale: Int32(fps))
            adaptor.append(pb, withPresentationTime: presentationTime)
        }

        input.markAsFinished()
        await writer.finishWriting()

        if writer.status == .failed {
            throw ExporterError.writerFailed(writer.error?.localizedDescription ?? "Unknown")
        }
    }

    /// Get a temporary URL for saving video
    static func temporaryVideoURL() -> URL {
        let tmpDir = FileManager.default.temporaryDirectory
        return tmpDir.appendingPathComponent("wan_output_\(Int(Date().timeIntervalSince1970)).mp4")
    }
}

enum ExporterError: Error, LocalizedError {
    case writerFailed(String)
    case pixelBufferFailed

    var errorDescription: String? {
        switch self {
        case .writerFailed(let msg): return "Video writer failed: \(msg)"
        case .pixelBufferFailed: return "Failed to create pixel buffer"
        }
    }
}
