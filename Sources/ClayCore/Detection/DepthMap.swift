import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import simd

/// A photo's embedded depth map (Portrait-mode, LiDAR or TrueDepth), rotated upright to match
/// `LoadedImage.cgImage` and converted to depth.
public struct DepthMap: @unchecked Sendable {
    public let width: Int
    public let height: Int
    /// Depth per pixel (metres when `isAbsolute`, otherwise only relative). ≤ 0 or non-finite = no data.
    public let values: [Float]
    /// LiDAR/TrueDepth maps are metric; dual-camera Portrait disparity usually isn't.
    public let isAbsolute: Bool

    /// Reads the depth (or disparity) auxiliary image of a photo, if it has one.
    init?(source: CGImageSource, orientation: CGImagePropertyOrientation) {
        for type in [kCGImageAuxiliaryDataTypeDepth, kCGImageAuxiliaryDataTypeDisparity] {
            guard let info = CGImageSourceCopyAuxiliaryDataInfoAtIndex(source, 0, type) as? [AnyHashable: Any],
                  let depth = try? AVDepthData(fromDictionaryRepresentation: info) else { continue }
            self.init(depth.applyingExifOrientation(orientation))
            return
        }
        return nil
    }

    init(_ depth: AVDepthData) {
        let d = depth.depthDataType == kCVPixelFormatType_DepthFloat32 ? depth
            : depth.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        isAbsolute = d.depthDataAccuracy == .absolute
        let map = d.depthDataMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        width = CVPixelBufferGetWidth(map)
        height = CVPixelBufferGetHeight(map)
        let stride = CVPixelBufferGetBytesPerRow(map) / MemoryLayout<Float>.stride
        let base = CVPixelBufferGetBaseAddress(map)!.assumingMemoryBound(to: Float.self)
        var v = [Float](repeating: 0, count: width * height)
        for y in 0..<height { for x in 0..<width { v[y * width + x] = base[y * stride + x] } }
        values = v
    }

    /// Median depth in a small window around an image point (robust at silhouette edges). nil if no data.
    func depth(at imagePoint: SIMD2<Double>, imageSize: SIMD2<Double>, radius: Int = 2) -> Double? {
        let cx = Int(imagePoint.x / imageSize.x * Double(width)), cy = Int(imagePoint.y / imageSize.y * Double(height))
        var samples: [Float] = []
        for y in (cy - radius)...(cy + radius) where y >= 0 && y < height {
            for x in (cx - radius)...(cx + radius) where x >= 0 && x < width {
                let d = values[y * width + x]
                if d.isFinite && d > 0 { samples.append(d) }
            }
        }
        guard !samples.isEmpty else { return nil }
        samples.sort()
        return Double(samples[samples.count / 2])
    }
}
