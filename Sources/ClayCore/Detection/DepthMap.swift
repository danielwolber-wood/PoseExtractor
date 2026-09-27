import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import simd

/// What the numbers in a depth map mean. Only `metricDepth` is in metres; the relative kinds need a
/// scale (and usually a shift) from somewhere else before they say anything about distance.
public enum DepthRepresentation: String, Codable, Sendable, CaseIterable {
    /// Metres along the camera's optical axis (z), from a calibrated sensor (LiDAR/TrueDepth) or a
    /// model whose output is metric given the focal length (Depth Pro).
    case metricDepth
    /// Affine-invariant inverse depth, disparity-like: value ≈ s / z + t with unknown s > 0 and t.
    /// **Larger = nearer.** Depth Anything V2's relative models output this.
    case relativeInverseDepth
    /// Depth with unknown scale (and possibly shift). **Larger = farther.** Dual-camera Portrait
    /// disparity converted to depth is of this kind.
    case relativeDepth

    public var isMetric: Bool { self == .metricDepth }
    public var largerIsNearer: Bool { self == .relativeInverseDepth }
}

/// A per-pixel depth map covering the whole upright image (`LoadedImage.cgImage`): either a photo's
/// embedded depth (Portrait-mode, LiDAR or TrueDepth) or a monocular estimate.
///
/// The map is usually smaller than the image; it's sampled in *image* pixel coordinates (top-left
/// origin), which are mapped to map pixels by the ratio of the sizes. Pixel centres are at +0.5.
public struct DepthMap: @unchecked Sendable {
    public let width: Int
    public let height: Int
    /// Per pixel, row-major, in `representation` units. ≤ 0 or non-finite = no data.
    public let values: [Float]
    public let representation: DepthRepresentation
    /// LiDAR/TrueDepth maps are metric; dual-camera Portrait disparity usually isn't.
    public var isAbsolute: Bool { representation.isMetric }

    public init(width: Int, height: Int, values: [Float], representation: DepthRepresentation) {
        precondition(width > 0 && height > 0 && values.count == width * height, "depth map size mismatch")
        self.width = width
        self.height = height
        self.values = values
        self.representation = representation
    }

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
        representation = d.depthDataAccuracy == .absolute ? .metricDepth : .relativeDepth
        let map = d.depthDataMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        width = CVPixelBufferGetWidth(map)
        height = CVPixelBufferGetHeight(map)
        let stride = CVPixelBufferGetBytesPerRow(map) / MemoryLayout<Float>.stride
        var v = [Float](repeating: 0, count: width * height)
        if let base = CVPixelBufferGetBaseAddress(map)?.assumingMemoryBound(to: Float.self) {
            for y in 0..<height { for x in 0..<width { v[y * width + x] = base[y * stride + x] } }
        }
        values = v
    }

    @inline(__always) static func isValid(_ v: Float) -> Bool { v.isFinite && v > 0 }

    /// Fraction of pixels with data.
    public var validFraction: Double {
        Double(values.reduce(0) { $0 + (Self.isValid($1) ? 1 : 0) }) / Double(max(values.count, 1))
    }

    /// Median depth in a small window around an image point (robust at silhouette edges). nil if no data.
    func depth(at imagePoint: SIMD2<Double>, imageSize: SIMD2<Double>, radius: Int = 2) -> Double? {
        windowStatistics(at: imagePoint, imageSize: imageSize, radius: radius)?.median
    }

    /// Median and median absolute deviation of the valid values in a (2r+1)² window of map pixels.
    func windowStatistics(at imagePoint: SIMD2<Double>, imageSize: SIMD2<Double>, radius: Int = 2)
        -> (median: Double, mad: Double, count: Int)? {
        let cx = Int(imagePoint.x / imageSize.x * Double(width)), cy = Int(imagePoint.y / imageSize.y * Double(height))
        var samples: [Float] = []
        samples.reserveCapacity((2 * radius + 1) * (2 * radius + 1))
        for y in (cy - radius)...(cy + radius) where y >= 0 && y < height {
            for x in (cx - radius)...(cx + radius) where x >= 0 && x < width {
                let d = values[y * width + x]
                if Self.isValid(d) { samples.append(d) }
            }
        }
        guard !samples.isEmpty else { return nil }
        samples.sort()
        let median = samples[samples.count / 2]
        var dev = samples.map { abs($0 - median) }
        dev.sort()
        return (Double(median), Double(dev[dev.count / 2]), samples.count)
    }

    /// Bilinearly interpolated value at an image point; invalid neighbours are left out (weights
    /// renormalised). nil when all four neighbours are invalid or the point is outside the image.
    public func sample(at imagePoint: SIMD2<Double>, imageSize: SIMD2<Double>) -> Float? {
        let u = imagePoint.x / imageSize.x * Double(width) - 0.5
        let v = imagePoint.y / imageSize.y * Double(height) - 0.5
        guard u > -1, v > -1, u < Double(width), v < Double(height) else { return nil }
        let x0 = Int(u.rounded(.down)), y0 = Int(v.rounded(.down))
        let fx = Float(u - Double(x0)), fy = Float(v - Double(y0))
        var sum: Float = 0, weight: Float = 0
        for (dx, dy, w) in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)] {
            let x = min(max(x0 + dx, 0), width - 1), y = min(max(y0 + dy, 0), height - 1)
            let d = values[y * width + x]
            if Self.isValid(d), w > 0 { sum += w * d; weight += w }
        }
        return weight > 1e-6 ? sum / weight : nil
    }

    /// The map bilinearly resampled to another size (e.g. the image's, for export or display).
    /// The fitter never needs this: it samples the native-resolution map directly.
    public func resampled(width w: Int, height h: Int) -> DepthMap {
        let size = SIMD2(Double(w), Double(h))
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                out[y * w + x] = sample(at: SIMD2(Double(x) + 0.5, Double(y) + 0.5), imageSize: size) ?? 0
            }
        }
        return DepthMap(width: w, height: h, values: out, representation: representation)
    }
}
