// 3-axis Rotary Position Embedding for video transformers
// Splits head_dim into frame, height, width segments and applies RoPE independently

import MLX
import MLXFast
import Foundation

/// Get the dimension split for 3D RoPE
/// - frame: d - 4*(d//6)
/// - height: 2*(d//6)
/// - width: 2*(d//6)
func getRopeDimensions(_ headDim: Int) -> (frame: Int, height: Int, width: Int) {
    let d = headDim
    let frameDim = d - 4 * (d / 6)
    let heightDim = 2 * (d / 6)
    let widthDim = 2 * (d / 6)
    return (frameDim, heightDim, widthDim)
}

/// Apply 3D RoPE to input tensor
/// - Parameters:
///   - x: [B, L, H, D] tensor where L = f*h*w
///   - gridSizes: [[f, h, w]] grid dimensions
///   - headDim: dimension per attention head
///   - theta: RoPE base frequency
func ropeApply(_ x: MLXArray, gridSizes: [[Int]], headDim: Int, theta: Float = 10000.0) -> MLXArray {
    let f = gridSizes[0][0]
    let h = gridSizes[0][1]
    let w = gridSizes[0][2]
    let (frameDim, heightDim, widthDim) = getRopeDimensions(headDim)
    return rope3D(x, f: f, h: h, w: w, frameDim: frameDim, heightDim: heightDim, widthDim: widthDim, theta: theta)
}

/// Apply 3D RoPE with separate frame/height/width rotations
private func rope3D(_ x: MLXArray, f: Int, h: Int, w: Int, frameDim: Int, heightDim: Int, widthDim: Int, theta: Float) -> MLXArray {
    let B = x.dim(0)
    let n = x.dim(2) // num_heads
    let hw = h * w

    // Split along last dim
    let xFrame = x[0..., 0..., 0..., ..<frameDim]
    let xHeight = x[0..., 0..., 0..., frameDim..<(frameDim + heightDim)]
    let xWidth = x[0..., 0..., 0..., (frameDim + heightDim)...]

    // Frame RoPE: "B (f hw) n d -> (B hw) n f d"
    var frameR = xFrame.reshaped([B, f, hw, n, frameDim])
        .transposed(0, 2, 3, 1, 4)  // [B, hw, n, f, d]
        .reshaped([B * hw, n, f, frameDim])
    frameR = MLXFast.RoPE(frameR, dimensions: frameDim, traditional: true, base: theta, scale: 1.0, offset: 0)
    // "(B hw) n f d -> B (f hw) n d"
    frameR = frameR.reshaped([B, hw, n, f, frameDim])
        .transposed(0, 3, 1, 2, 4)  // [B, f, hw, n, d]
        .reshaped([B, f * hw, n, frameDim])

    // Height RoPE: "B (f h w) n d -> (B f w) n h d"
    var heightR = xHeight.reshaped([B, f, h, w, n, heightDim])
        .transposed(0, 1, 3, 4, 2, 5)  // [B, f, w, n, h, d]
        .reshaped([B * f * w, n, h, heightDim])
    heightR = MLXFast.RoPE(heightR, dimensions: heightDim, traditional: true, base: theta, scale: 1.0, offset: 0)
    // "(B f w) n h d -> B (f h w) n d"
    heightR = heightR.reshaped([B, f, w, n, h, heightDim])
        .transposed(0, 1, 4, 2, 3, 5)  // [B, f, h, w, n, d]
        .reshaped([B, f * h * w, n, heightDim])

    // Width RoPE: "B (f h w) n d -> (B f h) n w d"
    var widthR = xWidth.reshaped([B, f, h, w, n, widthDim])
        .transposed(0, 1, 2, 4, 3, 5)  // [B, f, h, n, w, d]
        .reshaped([B * f * h, n, w, widthDim])
    widthR = MLXFast.RoPE(widthR, dimensions: widthDim, traditional: true, base: theta, scale: 1.0, offset: 0)
    // "(B f h) n w d -> B (f h w) n d"
    widthR = widthR.reshaped([B, f, h, n, w, widthDim])
        .transposed(0, 1, 2, 4, 3, 5)  // [B, f, h, w, n, d]
        .reshaped([B, f * h * w, n, widthDim])

    return concatenated([frameR, heightR, widthR], axis: -1)
}
