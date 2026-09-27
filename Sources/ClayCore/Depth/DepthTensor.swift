import Accelerate
import CoreML
import CoreVideo
import Foundation

/// Reading depth model outputs into row-major Float32 planes.
public enum DepthTensor {
    public struct Plane: Equatable {
        public var values: [Float]
        public var width: Int
        public var height: Int
    }

    enum ReadError: Error, CustomStringConvertible {
        case unsupported(String)
        var description: String { switch self { case .unsupported(let s): s } }
    }

    static func read(_ value: MLFeatureValue) throws -> Plane {
        if let array = value.multiArrayValue { return try read(array) }
        if let buffer = value.imageBufferValue { return try read(buffer) }
        throw ReadError.unsupported("output is neither a multi-array nor an image")
    }

    /// A `[…, H, W]` multi-array whose leading dimensions are all 1, honouring its strides.
    public static func read(_ array: MLMultiArray) throws -> Plane {
        let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
        guard shape.count >= 2, shape.dropLast(2).allSatisfy({ $0 == 1 }) else {
            throw ReadError.unsupported("multi-array shape \(shape) is not [..., H, W]")
        }
        let h = shape[shape.count - 2], w = shape[shape.count - 1]
        let sy = strides[strides.count - 2], sx = strides[strides.count - 1]
        var out = [Float](repeating: 0, count: w * h)
        switch array.dataType {
        case .float32:
            array.withUnsafeBufferPointer(ofType: Float.self) { p in
                for y in 0..<h { for x in 0..<w { out[y * w + x] = p[y * sy + x * sx] } }
            }
        case .double:
            array.withUnsafeBufferPointer(ofType: Double.self) { p in
                for y in 0..<h { for x in 0..<w { out[y * w + x] = Float(p[y * sy + x * sx]) } }
            }
        case .int32:
            array.withUnsafeBufferPointer(ofType: Int32.self) { p in
                for y in 0..<h { for x in 0..<w { out[y * w + x] = Float(p[y * sy + x * sx]) } }
            }
        case .float16:
            // Float16 → Float32 with vImage (works on every architecture, unlike Swift's Float16).
            array.withUnsafeBytes { raw in
                let p = raw.bindMemory(to: UInt16.self)
                var bits = [UInt16](repeating: 0, count: w * h)
                for y in 0..<h { for x in 0..<w { bits[y * w + x] = p[y * sy + x * sx] } }
                halfToFloat(bits, into: &out, width: w, height: h)
            }
        default:
            throw ReadError.unsupported("multi-array data type \(array.dataType.rawValue)")
        }
        return Plane(values: out, width: w, height: h)
    }

    /// A one-component pixel buffer (Float16, Float32, depth/disparity, or 8-bit).
    public static func read(_ buffer: CVPixelBuffer) throws -> Plane {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw ReadError.unsupported("pixel buffer has no memory") }
        var out = [Float](repeating: 0, count: w * h)
        switch CVPixelBufferGetPixelFormatType(buffer) {
        case kCVPixelFormatType_OneComponent16Half, kCVPixelFormatType_DepthFloat16, kCVPixelFormatType_DisparityFloat16:
            var src = vImage_Buffer(data: base, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: rowBytes)
            out.withUnsafeMutableBytes { dst in
                var d = vImage_Buffer(data: dst.baseAddress, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)
                _ = vImageConvert_Planar16FtoPlanarF(&src, &d, vImage_Flags(kvImageNoFlags))
            }
        case kCVPixelFormatType_OneComponent32Float, kCVPixelFormatType_DepthFloat32, kCVPixelFormatType_DisparityFloat32:
            for y in 0..<h {
                let row = (base + y * rowBytes).assumingMemoryBound(to: Float.self)
                for x in 0..<w { out[y * w + x] = row[x] }
            }
        case kCVPixelFormatType_OneComponent8:
            for y in 0..<h {
                let row = (base + y * rowBytes).assumingMemoryBound(to: UInt8.self)
                for x in 0..<w { out[y * w + x] = Float(row[x]) / 255 }
            }
        case let f:
            throw ReadError.unsupported("pixel format \(fourCC(f)) is not a one-component depth image")
        }
        return Plane(values: out, width: w, height: h)
    }

    /// First element of a (scalar) multi-array output, e.g. Depth Pro's field of view.
    static func scalar(_ value: MLFeatureValue?) -> Double? {
        guard let a = value?.multiArrayValue, a.count > 0 else { return nil }
        let v = a[0].doubleValue
        return v.isFinite ? v : nil
    }

    /// Crops `plane` to `x0..<x1, y0..<y1`.
    public static func crop(_ plane: Plane, x0: Int, y0: Int, x1: Int, y1: Int) -> Plane {
        if x0 == 0, y0 == 0, x1 == plane.width, y1 == plane.height { return plane }
        let w = x1 - x0, h = y1 - y0
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            let src = (y + y0) * plane.width + x0
            for x in 0..<w { out[y * w + x] = plane.values[src + x] }
        }
        return Plane(values: out, width: w, height: h)
    }

    /// Divides relative values by their 99th percentile (of the valid, positive ones) so they sit in
    /// roughly 0…1. A positive scale only: no shift, so ratios and the zero point survive — the
    /// scale estimator's shift-free model (inverse depth ∝ 1/z) depends on that. Non-positive and
    /// non-finite values become 0 (no data). Returns the divisor (1 if nothing was valid).
    public static func normalizeRelative(_ values: inout [Float]) -> Float {
        // A strided subsample (≤ ~20k values) gives the percentile to well under 1%, without sorting
        // a 2.4 M-value Depth Pro plane.
        let step = max(1, values.count / 20_000)
        let valid = stride(from: 0, to: values.count, by: step).map { values[$0] }.filter { $0.isFinite && $0 > 0 }.sorted()
        guard !valid.isEmpty else {
            for i in values.indices { values[i] = values[i].isFinite && values[i] > 0 ? values[i] : 0 }
            return 1
        }
        let p99 = valid[min(Int(Double(valid.count - 1) * 0.99), valid.count - 1)]
        let scale = p99 > 0 ? p99 : 1
        for i in values.indices {
            let v = values[i]
            values[i] = v.isFinite && v > 0 ? v / scale : 0
        }
        return scale
    }

    static func halfToFloat(_ bits: [UInt16], into out: inout [Float], width w: Int, height h: Int) {
        var bits = bits
        bits.withUnsafeMutableBytes { src in
            out.withUnsafeMutableBytes { dst in
                var s = vImage_Buffer(data: src.baseAddress, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 2)
                var d = vImage_Buffer(data: dst.baseAddress, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)
                _ = vImageConvert_Planar16FtoPlanarF(&s, &d, vImage_Flags(kvImageNoFlags))
            }
        }
    }

    static func fourCC(_ f: OSType) -> String {
        let chars = [24, 16, 8, 0].map { Character(UnicodeScalar(UInt8((f >> $0) & 0xff))) }
        return chars.allSatisfy { $0.isASCII && !$0.isWhitespace } ? String(chars) : "\(f)"
    }
}
