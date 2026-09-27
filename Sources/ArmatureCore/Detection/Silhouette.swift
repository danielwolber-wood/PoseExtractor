import CoreGraphics
import CoreVideo
import Foundation
import Vision
import simd

/// One person's segmentation mask, with the precomputed data the silhouette fit needs.
public struct PersonMask: @unchecked Sendable {
    public let width: Int
    public let height: Int
    /// Image pixels per mask pixel.
    public let scale: Double
    /// 1 where the person is.
    public let inside: [UInt8]
    /// Euclidean distance (mask pixels) to the nearest person pixel; 0 inside the mask.
    let distanceOutside: [Float]
    /// Evenly subsampled mask outline, in image pixels.
    let contour: [SIMD2<Double>]

    /// Builds a mask from per-pixel person probabilities (`width × height`, row-major).
    init(probabilities p: [Float], width: Int, height: Int, scale: Double) {
        self.width = width
        self.height = height
        self.scale = scale
        inside = p.map { $0 > 0.5 ? 1 : 0 }
        distanceOutside = euclideanDistance(to: inside, width: width, height: height)

        var edge: [SIMD2<Double>] = []
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) where inside[y * width + x] == 1 {
                let i = y * width + x
                if inside[i - 1] == 0 || inside[i + 1] == 0 || inside[i - width] == 0 || inside[i + width] == 0 {
                    edge.append(SIMD2(Double(x) + 0.5, Double(y) + 0.5) * scale)
                }
            }
        }
        let step = max(1, edge.count / 500)
        contour = stride(from: 0, to: edge.count, by: step).map { edge[$0] }
    }

    /// Distance to the mask (image pixels, 0 inside) and its gradient, bilinearly interpolated.
    func distance(at imagePoint: SIMD2<Double>) -> (Double, SIMD2<Double>) {
        let x = imagePoint.x / scale - 0.5, y = imagePoint.y / scale - 0.5
        func d(_ x: Double, _ y: Double) -> Double {
            let xi = min(max(Int(x.rounded(.down)), 0), width - 2), yi = min(max(Int(y.rounded(.down)), 0), height - 2)
            let fx = min(max(x - Double(xi), 0), 1), fy = min(max(y - Double(yi), 0), 1)
            let i = yi * width + xi
            let a = Double(distanceOutside[i]), b = Double(distanceOutside[i + 1])
            let c = Double(distanceOutside[i + width]), e = Double(distanceOutside[i + width + 1])
            return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + e * fx) * fy
        }
        let v = d(x, y)
        let g = SIMD2(d(x + 1, y) - d(x - 1, y), d(x, y + 1) - d(x, y - 1)) / 2
        return (v * scale, g)  // gradient: image px per image px == mask px per mask px
    }

    /// Compares a rasterised body (same size as the mask) with the mask.
    /// - Returns: intersection-over-union, and the fraction of the body lying outside the mask.
    ///   Clothing and hair make IoU understate a good fit; `outside` is the stricter signal.
    func overlap(with raster: [UInt8]) -> SilhouetteOverlap {
        var inter = 0, union = 0, body = 0
        for i in 0..<inside.count {
            let a = inside[i] == 1, b = raster[i] == 1
            if a && b { inter += 1 }
            if a || b { union += 1 }
            if b { body += 1 }
        }
        return SilhouetteOverlap(iou: union > 0 ? Double(inter) / Double(union) : 0,
                                 outside: body > 0 ? Double(body - inter) / Double(body) : 1)
    }

    /// Fills the projected triangles of a mesh (image-pixel vertices) at mask resolution.
    func rasterize(_ projected: [SIMD2<Double>], faces: [UInt32]) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height)
        let pts = projected.map { $0 / scale }
        for t in stride(from: 0, to: faces.count, by: 3) {
            let a = pts[Int(faces[t])], b = pts[Int(faces[t + 1])], c = pts[Int(faces[t + 2])]
            let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
            if abs(area) < 1e-9 { continue }
            let x0 = max(Int(min(a.x, b.x, c.x).rounded(.down)), 0), x1 = min(Int(max(a.x, b.x, c.x).rounded(.up)), width - 1)
            let y0 = max(Int(min(a.y, b.y, c.y).rounded(.down)), 0), y1 = min(Int(max(a.y, b.y, c.y).rounded(.up)), height - 1)
            if x0 > x1 || y0 > y1 { continue }
            for y in y0...y1 {
                for x in x0...x1 {
                    let p = SIMD2(Double(x) + 0.5, Double(y) + 0.5)
                    let w0 = (b.x - p.x) * (c.y - p.y) - (b.y - p.y) * (c.x - p.x)
                    let w1 = (c.x - p.x) * (a.y - p.y) - (c.y - p.y) * (a.x - p.x)
                    let w2 = (a.x - p.x) * (b.y - p.y) - (a.y - p.y) * (b.x - p.x)
                    if area > 0 ? (w0 >= 0 && w1 >= 0 && w2 >= 0) : (w0 <= 0 && w1 <= 0 && w2 <= 0) {
                        out[y * width + x] = 1
                    }
                }
            }
        }
        return out
    }

    /// Tinted, semi-transparent overlay of the mask for display.
    public func overlayImage(red: Double, green: Double, blue: Double, alpha: Double = 0.35) -> CGImage? {
        var px = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<inside.count where inside[i] == 1 {
            // Premultiplied RGBA.
            px[i * 4] = UInt8(red * alpha * 255)
            px[i * 4 + 1] = UInt8(green * alpha * 255)
            px[i * 4 + 2] = UInt8(blue * alpha * 255)
            px[i * 4 + 3] = UInt8(alpha * 255)
        }
        guard let provider = CGDataProvider(data: Data(px) as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}

public struct SilhouetteOverlap: Codable, Sendable {
    /// Intersection-over-union of the body's projection and the mask.
    public var iou: Double
    /// Fraction of the body's projection outside the mask.
    public var outside: Double
}

// MARK: - Segmentation

enum PersonSegmentation {
    /// Long side of the working mask. Plenty for shape fitting, and keeps the distance transform cheap.
    static let workingSize = 512

    /// Runs Vision's person instance segmentation and gives each detected person the mask their
    /// keypoints fall in. People with no matching instance get `nil`.
    static func masks(for image: LoadedImage, people: [DetectedPerson]) throws -> [PersonMask?] {
        let handler = VNImageRequestHandler(cgImage: image.cgImage, options: [:])
        let request = VNGeneratePersonInstanceMaskRequest()
        try handler.perform([request])
        guard let obs = request.results?.first else { return people.map { _ in nil } }

        // Vote with each person's confident keypoints on the low-res instance label map.
        let labels = obs.instanceMask
        CVPixelBufferLockBaseAddress(labels, .readOnly)
        let lw = CVPixelBufferGetWidth(labels), lh = CVPixelBufferGetHeight(labels)
        let stride = CVPixelBufferGetBytesPerRow(labels)
        let base = CVPixelBufferGetBaseAddress(labels)!.assumingMemoryBound(to: UInt8.self)
        let W = Double(image.width), H = Double(image.height)
        let assignment: [Int?] = people.map { p in
            var votes: [Int: Int] = [:]
            for j in BodyJoint.allCases where p.confidence2D[j.rawValue] > 0.3 || p.edited[j.rawValue] {
                let q = p.joints2D[j.rawValue]
                let x = Int(q.x / W * Double(lw)), y = Int(q.y / H * Double(lh))
                guard x >= 0, y >= 0, x < lw, y < lh else { continue }
                let label = Int(base[y * stride + x])
                if label != 0 { votes[label, default: 0] += 1 }
            }
            guard let best = votes.max(by: { $0.value < $1.value }), best.value >= 3 else { return nil }
            return best.key
        }
        CVPixelBufferUnlockBaseAddress(labels, .readOnly)

        var cache: [Int: PersonMask] = [:]
        return try assignment.map { label in
            guard let label else { return nil }
            if let m = cache[label] { return m }
            let buffer = try obs.generateScaledMaskForImage(forInstances: IndexSet(integer: label), from: handler)
            let m = downsample(buffer, imageWidth: image.width, imageHeight: image.height)
            cache[label] = m
            return m
        }
    }

    /// Box-filters Vision's full-resolution float mask down to the working size.
    private static func downsample(_ buffer: CVPixelBuffer, imageWidth: Int, imageHeight: Int) -> PersonMask {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let bw = CVPixelBufferGetWidth(buffer), bh = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer) / MemoryLayout<Float>.stride
        let src = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: Float.self)

        let scale = Double(max(imageWidth, imageHeight)) / Double(workingSize)
        let w = max(Int((Double(imageWidth) / scale).rounded()), 2), h = max(Int((Double(imageHeight) / scale).rounded()), 2)
        let sx = Double(bw) / Double(w), sy = Double(bh) / Double(h)
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            let y0 = Int(Double(y) * sy), y1 = max(min(Int(Double(y + 1) * sy), bh), y0 + 1)
            for x in 0..<w {
                let x0 = Int(Double(x) * sx), x1 = max(min(Int(Double(x + 1) * sx), bw), x0 + 1)
                var s: Float = 0
                for yy in y0..<y1 { for xx in x0..<x1 { s += src[yy * stride + xx] } }
                out[y * w + x] = s / Float((y1 - y0) * (x1 - x0))
            }
        }
        return PersonMask(probabilities: out, width: w, height: h, scale: Double(imageWidth) / Double(w))
    }
}

// MARK: - Distance transform

/// Exact Euclidean distance transform (Felzenszwalb & Huttenlocher): distance from each pixel to the
/// nearest pixel where `mask == 1`.
func euclideanDistance(to mask: [UInt8], width: Int, height: Int) -> [Float] {
    let inf = 1e20
    var grid = mask.map { $0 == 1 ? 0 : inf }
    var f = [Double](repeating: 0, count: max(width, height))
    var d = f
    var v = [Int](repeating: 0, count: max(width, height))
    var z = [Double](repeating: 0, count: max(width, height) + 1)

    func transform1D(_ n: Int) {
        var k = 0
        v[0] = 0
        z[0] = -inf; z[1] = inf
        for q in 1..<n {
            var s: Double
            repeat {
                let p = v[k]
                s = ((f[q] + Double(q * q)) - (f[p] + Double(p * p))) / Double(2 * q - 2 * p)
                if s <= z[k] { k -= 1 } else { break }
            } while k >= 0
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = inf
        }
        k = 0
        for q in 0..<n {
            while z[k + 1] < Double(q) { k += 1 }
            let p = v[k]
            d[q] = Double((q - p) * (q - p)) + f[p]
        }
    }

    for x in 0..<width {
        for y in 0..<height { f[y] = grid[y * width + x] }
        transform1D(height)
        for y in 0..<height { grid[y * width + x] = d[y] }
    }
    for y in 0..<height {
        for x in 0..<width { f[x] = grid[y * width + x] }
        transform1D(width)
        for x in 0..<width { grid[y * width + x] = d[x] }
    }
    return grid.map { $0 >= inf / 2 ? 1e6 : Float($0.squareRoot()) }
}
