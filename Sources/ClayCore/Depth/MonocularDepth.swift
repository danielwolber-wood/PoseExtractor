import Foundation

/// A monocular (single-photo) depth estimator the app can run through Core ML.
public enum MonocularDepthBackend: String, CaseIterable, Codable, Sendable {
    /// Depth Anything V2 Small (24.8 M parameters, Apache-2.0): relative inverse depth, fast.
    case depthAnythingV2Small = "depth-anything-v2-small"
    /// Apple Depth Pro: metric depth plus a field-of-view estimate; large and slow.
    case depthPro = "depth-pro"

    public var displayName: String {
        switch self {
        case .depthAnythingV2Small: "Depth Anything V2 Small"
        case .depthPro: "Depth Pro"
        }
    }

    /// The order `automatic` tries installed backends in: the small, fast model first, so the large
    /// one is only ever loaded when it's the only one installed or explicitly selected.
    public static let automaticOrder: [MonocularDepthBackend] = [.depthAnythingV2Small, .depthPro]
}

/// Which monocular depth, if any, the pipeline runs when a photo has no embedded metric depth.
public enum MonocularDepthMode: Hashable, Sendable, CustomStringConvertible {
    case disabled
    /// The first installed backend in `MonocularDepthBackend.automaticOrder`.
    case automatic
    case backend(MonocularDepthBackend)

    /// Parses the CLI spelling: none | off | auto | depth-anything-v2-small | depth-pro.
    public init?(argument: String) {
        switch argument.lowercased() {
        case "none", "off", "disabled": self = .disabled
        case "auto", "automatic": self = .automatic
        default:
            guard let b = MonocularDepthBackend(rawValue: argument.lowercased()) else { return nil }
            self = .backend(b)
        }
    }

    public var description: String {
        switch self {
        case .disabled: "none"
        case .automatic: "auto"
        case .backend(let b): b.rawValue
        }
    }
}

public enum MonocularDepthError: LocalizedError, Equatable {
    /// No model file for the backend in any search location.
    case modelNotInstalled(MonocularDepthBackend, searched: [String])
    /// The model loaded, but its inputs/outputs don't match what the backend needs.
    case incompatibleModel(MonocularDepthBackend, path: String, reason: String)
    case loadFailed(MonocularDepthBackend, path: String, reason: String)
    case inferenceFailed(MonocularDepthBackend, reason: String)

    public var errorDescription: String? {
        switch self {
        case .modelNotInstalled(let b, let searched):
            return "\(b.displayName) is not installed. Put the Core ML model in Models/depth/\(b.rawValue)/ "
                + "(searched: \(searched.joined(separator: ", "))). See docs/depth.md."
        case .incompatibleModel(let b, let path, let reason):
            return "\(b.displayName) model at \(path) is not compatible: \(reason)"
        case .loadFailed(let b, let path, let reason):
            return "\(b.displayName) model at \(path) failed to load: \(reason)"
        case .inferenceFailed(let b, let reason):
            return "\(b.displayName) inference failed: \(reason)"
        }
    }
}

/// How an estimate's focal length was obtained (Depth Pro scales its output by it).
public enum FocalLengthSource: String, Codable, Sendable {
    /// From the photo's EXIF (35 mm-equivalent focal length) or the user's --focal-mm.
    case exif
    /// Predicted by the depth model (Depth Pro's field-of-view head).
    case predicted
    /// The pipeline's 50 mm-equivalent guess; a metric model scaled by it isn't trustworthy metric.
    case assumed
}

/// A monocular depth map for one photo, with where it came from and how it was computed.
public struct MonocularDepthEstimate: @unchecked Sendable {
    public let backend: MonocularDepthBackend
    /// Upright, covering the whole image. Relative kinds are normalised by a positive factor only
    /// (see `CoreMLDepthEstimator`), so ratios — and the zero point — of the raw output are kept.
    public let map: DepthMap
    /// Per-pixel confidence in 0…1 at map resolution. Neither backend predicts one; this is derived
    /// from local depth gradients (low at occlusion boundaries, where monocular depth bleeds).
    public let confidence: [Float]?
    public let confidenceIsDerived: Bool
    /// Focal length (image pixels) the model predicted, if it has a field-of-view output.
    public let estimatedFocalLengthPixels: Double?
    /// The focal length a metric output was scaled with, and its source (nil for relative output).
    public let focalLengthUsedPixels: Double?
    public let focalSource: FocalLengthSource?
    /// Diagnostics.
    public let modelPath: String?
    public let inputSize: SIMD2<Int>
    public let outputSize: SIMD2<Int>
    public let resize: DepthResizeMode
    /// Model load/compile time charged to this call (0 when the model was already loaded).
    public let loadSeconds: TimeInterval
    /// Core ML prediction time.
    public let inferenceSeconds: TimeInterval
    /// Resizing the photo in, and decoding / cropping / converting the output.
    public let processingSeconds: TimeInterval
    /// Whether this call loaded (and possibly compiled) the model.
    public let coldStart: Bool
    public let notes: [String]

    public init(backend: MonocularDepthBackend, map: DepthMap, confidence: [Float]? = nil, confidenceIsDerived: Bool = true,
                estimatedFocalLengthPixels: Double? = nil, focalLengthUsedPixels: Double? = nil,
                focalSource: FocalLengthSource? = nil, modelPath: String? = nil, inputSize: SIMD2<Int>? = nil,
                outputSize: SIMD2<Int>? = nil, resize: DepthResizeMode = .stretch, loadSeconds: TimeInterval = 0,
                inferenceSeconds: TimeInterval = 0, processingSeconds: TimeInterval = 0, coldStart: Bool = false,
                notes: [String] = []) {
        self.backend = backend
        self.map = map
        self.confidence = confidence ?? DepthConfidence.fromGradients(map)
        self.confidenceIsDerived = confidence == nil || confidenceIsDerived
        self.estimatedFocalLengthPixels = estimatedFocalLengthPixels
        self.focalLengthUsedPixels = focalLengthUsedPixels
        self.focalSource = focalSource
        self.modelPath = modelPath
        self.inputSize = inputSize ?? SIMD2(map.width, map.height)
        self.outputSize = outputSize ?? SIMD2(map.width, map.height)
        self.resize = resize
        self.loadSeconds = loadSeconds
        self.inferenceSeconds = inferenceSeconds
        self.processingSeconds = processingSeconds
        self.coldStart = coldStart
        self.notes = notes
    }

    /// Confidence at an image point (1 when there's no confidence map).
    func confidence(at p: SIMD2<Double>, imageSize: SIMD2<Double>) -> Double {
        guard let confidence else { return 1 }
        let x = min(max(Int(p.x / imageSize.x * Double(map.width)), 0), map.width - 1)
        let y = min(max(Int(p.y / imageSize.y * Double(map.height)), 0), map.height - 1)
        return Double(confidence[y * map.width + x])
    }

    public var confidenceSummary: (mean: Double, fractionAboveHalf: Double)? {
        guard let confidence, !confidence.isEmpty else { return nil }
        let mean = confidence.reduce(0) { $0 + Double($1) } / Double(confidence.count)
        let high = Double(confidence.filter { $0 > 0.5 }.count) / Double(confidence.count)
        return (mean, high)
    }
}

/// Something that turns a photo into a monocular depth estimate. The Core ML backends implement it;
/// tests plug in `SyntheticDepthEstimator`.
public protocol MonocularDepthEstimating: AnyObject, Sendable {
    var backend: MonocularDepthBackend { get }
    func estimate(_ image: LoadedImage) throws -> MonocularDepthEstimate
}

/// A fixed, caller-supplied depth map (tests, or depth computed elsewhere).
public final class SyntheticDepthEstimator: MonocularDepthEstimating, @unchecked Sendable {
    public let backend: MonocularDepthBackend
    private let make: (LoadedImage) throws -> MonocularDepthEstimate

    public init(backend: MonocularDepthBackend = .depthAnythingV2Small,
                _ make: @escaping (LoadedImage) throws -> MonocularDepthEstimate) {
        self.backend = backend
        self.make = make
    }

    public func estimate(_ image: LoadedImage) throws -> MonocularDepthEstimate { try make(image) }
}

enum DepthConfidence {
    /// 1 on smooth surfaces, falling towards 0 where the relative depth changes by more than ~10%
    /// between neighbouring map pixels (occlusion boundaries, hair against background).
    static func fromGradients(_ map: DepthMap) -> [Float] {
        let w = map.width, h = map.height, v = map.values
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let c = v[y * w + x]
                guard DepthMap.isValid(c) else { continue }
                var g: Float = 0
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let xx = x + dx, yy = y + dy
                    guard xx >= 0, yy >= 0, xx < w, yy < h else { continue }
                    let n = v[yy * w + xx]
                    g = max(g, DepthMap.isValid(n) ? abs(n - c) / max(c, n) : 1)
                }
                let k = g / 0.1
                out[y * w + x] = 1 / (1 + k * k)
            }
        }
        return out
    }
}

// MARK: - Source priority

/// Where the depth used for fitting comes from, in priority order: embedded metric depth, the
/// selected monocular backend, the automatic fallback, nothing.
public enum DepthSourceDecision: Equatable, Sendable {
    case embeddedMetric
    /// `fallbackFrom`: the backend that was asked for but isn't installed.
    case monocular(MonocularDepthBackend, fallbackFrom: MonocularDepthBackend?)
    case none(reason: String)

    /// - Parameters:
    ///   - embedded: the photo's own depth map, if any.
    ///   - installed: backends whose model files were found.
    ///   - fallback: when the selected backend isn't installed, use another installed one.
    public static func resolve(embedded: DepthMap?, mode: MonocularDepthMode, fallback: Bool,
                               installed: Set<MonocularDepthBackend>) -> DepthSourceDecision {
        if let embedded, embedded.isAbsolute { return .embeddedMetric }
        switch mode {
        case .disabled:
            return .none(reason: "monocular depth disabled")
        case .automatic:
            if let b = MonocularDepthBackend.automaticOrder.first(where: installed.contains) {
                return .monocular(b, fallbackFrom: nil)
            }
            return .none(reason: "no monocular depth model installed")
        case .backend(let b):
            if installed.contains(b) { return .monocular(b, fallbackFrom: nil) }
            if fallback, let other = MonocularDepthBackend.automaticOrder.first(where: { $0 != b && installed.contains($0) }) {
                return .monocular(other, fallbackFrom: b)
            }
            return .none(reason: "\(b.displayName) is not installed")
        }
    }
}
