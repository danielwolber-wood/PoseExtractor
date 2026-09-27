import CoreGraphics
import CoreML
import Foundation
import simd

/// An age from an age-estimation model.
public struct AgeEstimate: Codable, Sendable {
    public var years: Double
    /// Id of the model that produced it (e.g. "mivolo").
    public var model: String
}

/// A converted age model on disk (Models/age/<id>/, see tools/convert_age_models.py).
public struct AgeModelInfo: Identifiable, Hashable, Sendable {
    public let id: String
    public let displayName: String
    public let licence: String
    let order: Int

    public static func available(in modelsDirectory: URL) -> [AgeModelInfo] {
        let root = modelsDirectory.appendingPathComponent("age")
        let dirs = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        return dirs.compactMap { dir in
            guard let spec = try? AgeEstimator.Spec.load(dir) else { return nil }
            return AgeModelInfo(id: dir.lastPathComponent, displayName: spec.displayName, licence: spec.licence, order: spec.order ?? 99)
        }.sorted { ($0.order, $0.id) < ($1.order, $1.id) }
    }
}

/// Runs an open-source age estimator (MiVOLO v2, FaceAge, ...) natively through Core ML.
///
/// Each model's `age.json` says which crops it takes and how to prepare them, so any converted model
/// plugs in without code changes:
/// - `face`: the person's face box.
/// - `bodyFacesMasked`: the person's box with every face box (theirs included) and any overlapping people
///   blacked out — MiVOLO's recipe, so the body branch reads build and posture rather than the face.
public final class AgeEstimator: @unchecked Sendable {
    struct Spec: Decodable {
        struct Input: Decodable { let crop: String; let channels: [Int] }
        struct Normalise: Decodable { let mode: String; let mean: [Double]?; let std: [Double]? }
        let displayName: String
        let licence: String
        let order: Int?
        let inputs: [Input]
        let size: Int
        let resize: String          // "letterbox" | "stretch"
        let colour: String          // "RGB" | "BGR"
        let layout: String?         // "NCHW" (default) | "NHWC"
        let normalise: Normalise    // mode "fixed" (mean/std on 0–1 values) | "perImage" (standardise the crop)
        let input: String
        let output: String
        let validAges: [Double]?

        static func load(_ dir: URL) throws -> Spec {
            try JSONDecoder().decode(Spec.self, from: Data(contentsOf: dir.appendingPathComponent("age.json")))
        }
    }

    public let info: AgeModelInfo
    private let spec: Spec
    private let model: MLModel
    private let lock = NSLock()

    public convenience init(modelsDirectory: URL, id: String) throws {
        try self.init(directory: modelsDirectory.appendingPathComponent("age").appendingPathComponent(id))
    }

    public init(directory: URL) throws {
        spec = try Spec.load(directory)
        info = AgeModelInfo(id: directory.lastPathComponent, displayName: spec.displayName, licence: spec.licence,
                            order: spec.order ?? 99)
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try MLModel(contentsOf: try Self.compiled(directory.appendingPathComponent("model.mlpackage"),
                                                          id: info.id), configuration: config)
    }

    /// Compiles the .mlpackage once and caches the result (~/Library/Caches/ClayPose/age), keyed by the
    /// package's modification date so re-converted models are picked up.
    private static func compiled(_ package: URL, id: String) throws -> URL {
        let stamp = (try? FileManager.default.attributesOfItem(atPath: package.appendingPathComponent("Manifest.json").path)[.modificationDate] as? Date)
            .map { Int($0.timeIntervalSince1970) } ?? 0
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ClayPose/age", isDirectory: true)
        let cached = cacheDir.appendingPathComponent("\(id)-\(stamp).mlmodelc")
        if FileManager.default.fileExists(atPath: cached.path) { return cached }
        let tmp = try MLModel.compileModel(at: package)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: cached)
        try FileManager.default.moveItem(at: tmp, to: cached)
        return cached
    }

    /// Estimates one person's age. `everyone` is used to mask other people out of the body crop.
    /// Returns nil when the model needs a crop the person doesn't have (e.g. no face found).
    public func estimate(_ person: DetectedPerson, among everyone: [DetectedPerson], image: LoadedImage) throws -> AgeEstimate? {
        let S = spec.size
        let nhwc = spec.layout == "NHWC"
        let channels = (spec.inputs.map { $0.channels[1] }.max() ?? 3)
        let shape: [NSNumber] = nhwc ? [1, S, S, channels] as [NSNumber] : [1, channels, S, S] as [NSNumber]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)

        var anyCrop = false
        for input in spec.inputs {
            let pixels: [UInt8]?
            switch input.crop {
            case "face":
                pixels = person.faceBox.flatMap { crop(image, box: $0, masks: []) }
            case "bodyFacesMasked":
                pixels = person.bodyBox.flatMap { box in
                    // Black out every face, and people overlapping this one (MiVOLO's body-crop recipe).
                    var masks = everyone.compactMap(\.faceBox)
                    masks += everyone.compactMap(\.bodyBox).filter { $0 != box && $0.intersects(box) }
                        .map { snapToEdges($0.intersection(box), within: box) }
                    return crop(image, box: box, masks: masks, minRemaining: 0.4)
                }
            default:
                pixels = nil
            }
            if pixels != nil { anyCrop = true }
            write(pixels, channels: input.channels[0], into: ptr, channelCount: channels, nhwc: nhwc)
        }
        guard anyCrop else { return nil }

        lock.lock(); defer { lock.unlock() }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [spec.input: array]))
        guard let value = out.featureValue(for: spec.output)?.multiArrayValue, value.count > 0 else { return nil }
        var years = value[0].doubleValue
        if let range = spec.validAges, range.count == 2 { years = min(max(years, range[0]), range[1]) }
        return AgeEstimate(years: years, model: info.id)
    }

    /// MiVOLO snaps a masked neighbour's box to the crop's edge when within 30% of it.
    private func snapToEdges(_ r: CGRect, within box: CGRect) -> CGRect {
        var x0 = r.minX, y0 = r.minY, x1 = r.maxX, y1 = r.maxY
        if (y0 - box.minY) / box.height < 0.3 { y0 = box.minY }
        if (box.maxY - y1) / box.height < 0.3 { y1 = box.maxY }
        if (x0 - box.minX) / box.width < 0.3 { x0 = box.minX }
        if (box.maxX - x1) / box.width < 0.3 { x1 = box.maxX }
        return CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    }

    /// Crops, masks and resizes to S×S RGBA (letterboxed with black, or stretched). nil if too little remains.
    private func crop(_ image: LoadedImage, box: CGRect, masks: [CGRect], minRemaining: Double = 0) -> [UInt8]? {
        let bounds = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        let box = box.integral.intersection(bounds)
        guard box.width >= 8, box.height >= 8, let cropped = image.cgImage.cropping(to: box) else { return nil }
        let S = spec.size
        guard let ctx = CGContext(data: nil, width: S, height: S, bitsPerComponent: 8, bytesPerRow: S * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.interpolationQuality = .high
        ctx.setFillColor(CGColor(gray: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: S, height: S))
        // Target rectangle in top-left-origin coordinates.
        let target: CGRect
        if spec.resize == "letterbox" {
            let r = min(Double(S) / box.width, Double(S) / box.height)
            let w = (box.width * r).rounded(), h = (box.height * r).rounded()
            target = CGRect(x: ((Double(S) - w) / 2).rounded(.down), y: ((Double(S) - h) / 2).rounded(.down), width: w, height: h)
        } else {
            target = CGRect(x: 0, y: 0, width: S, height: S)
        }
        let flip = { (r: CGRect) in CGRect(x: r.minX, y: Double(S) - r.maxY, width: r.width, height: r.height) }
        ctx.draw(cropped, in: flip(target))
        // Masks, mapped from image pixels into the target rectangle.
        let sx = target.width / box.width, sy = target.height / box.height
        for m in masks {
            let r = m.intersection(box)
            guard !r.isNull, r.width > 0 else { continue }
            ctx.fill(flip(CGRect(x: target.minX + (r.minX - box.minX) * sx, y: target.minY + (r.minY - box.minY) * sy,
                                 width: r.width * sx, height: r.height * sy)))
        }
        guard let data = ctx.data else { return nil }
        let px = Array(UnsafeBufferPointer(start: data.assumingMemoryBound(to: UInt8.self), count: S * S * 4))
        if minRemaining > 0 {
            // Fraction of the (unpadded) crop that isn't masked out.
            var nonzero = 0, total = 0
            for y in Int(target.minY)..<Int(target.maxY) {
                for x in Int(target.minX)..<Int(target.maxX) {
                    let i = (y * S + x) * 4
                    total += 1
                    if px[i] != 0 || px[i + 1] != 0 || px[i + 2] != 0 { nonzero += 1 }
                }
            }
            if total == 0 || Double(nonzero) / Double(total) < minRemaining { return nil }
        }
        return px
    }

    /// Normalises RGBA pixels into three channels of the input tensor, starting at `first`.
    /// A missing crop becomes all-zero pixels, normalised (as MiVOLO does for an absent face or body).
    private func write(_ pixels: [UInt8]?, channels first: Int, into ptr: UnsafeMutablePointer<Float>, channelCount: Int, nhwc: Bool) {
        let S = spec.size, n = S * S
        let order = spec.colour == "BGR" ? [2, 1, 0] : [0, 1, 2]
        var mean = [Double](repeating: 0, count: 3), std = [Double](repeating: 1, count: 3)
        var scale = 1.0 / 255
        if spec.normalise.mode == "perImage" {
            // Standardise the crop as a whole: (x - mean) / std over all pixels and channels, on 0-255 values.
            scale = 1
            if let px = pixels {
                var s = 0.0, s2 = 0.0
                for i in 0..<n { for c in 0..<3 { let v = Double(px[i * 4 + c]); s += v; s2 += v * v } }
                let m = s / Double(3 * n)
                let sd = max(sqrt(max(s2 / Double(3 * n) - m * m, 0)), 1e-6)
                mean = [m, m, m]; std = [sd, sd, sd]
            }
        } else if let m = spec.normalise.mean, let s = spec.normalise.std {
            mean = m; std = s
        }
        for c in 0..<3 {
            let src = order[c]
            let mu = mean[c], sd = std[c]
            for i in 0..<n {
                let v = pixels.map { Double($0[i * 4 + src]) * scale } ?? 0
                let value = Float((v - mu) / sd)
                if nhwc { ptr[i * channelCount + first + c] = value } else { ptr[(first + c) * n + i] = value }
            }
        }
    }
}
