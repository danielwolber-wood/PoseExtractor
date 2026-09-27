import CoreGraphics
import CoreML
import Foundation

/// How the photo is fitted into a fixed-size model input.
public enum DepthResizeMode: String, Codable, Sendable {
    /// Scale each axis independently to fill the input (Depth Pro's own preprocessing: a bilinear
    /// resize to 1536 × 1536 whatever the aspect ratio).
    case stretch
    /// Keep the aspect ratio and pad the rest with the photo's mean colour; the padding is cropped
    /// off the output. Used for Depth Anything, which was trained on aspect-preserving resizes but
    /// ships in Core ML with a fixed 518 × 392 input.
    case letterbox
}

/// What a model's depth output tensor contains.
public enum DepthOutputKind: String, Codable, Sendable {
    /// Metres (z).
    case metricDepth
    /// Depth Pro's "canonical inverse depth": inverse depth for a camera whose focal length equals
    /// the image width. Metric inverse depth = value × (imageWidth / focalPx); depth = 1 / that.
    case canonicalInverseDepth
    /// Affine-invariant inverse depth (larger = nearer); Depth Anything V2's relative models.
    case relativeInverseDepth
    /// Relative depth (larger = farther).
    case relativeDepth
}

/// The I/O contract of a depth model: which tensors to use and what their values mean.
///
/// Built-in defaults describe the published models; a `depth.json` next to the model overrides any
/// field, e.g. for a metric fine-tune of Depth Anything:
///
/// ```json
/// { "schemaVersion": 1, "outputKind": "metricDepth", "resize": "letterbox",
///   "input": "image", "output": "depth", "computeUnits": "all" }
/// ```
///
/// Input tensors: an **image** input (Core ML applies the model's own scale/bias; we supply sRGB
/// pixels at exactly the model's size) or a Float32 **multi-array** `[1, 3, H, W]` (RGB, values
/// `(pixel/255 − mean) / std`). Output tensors: a one-component image (Float16/Float32/UInt8) or a
/// multi-array `[…, H, W]` whose other dimensions are 1, of Float16/Float32/Float64/Int32, read with
/// its strides. The optional field-of-view output is a scalar in degrees (horizontal, across the width).
public struct DepthModelContract: Sendable, Equatable {
    public var input: String?
    public var output: String?
    public var fovOutput: String?
    public var outputKind: DepthOutputKind
    public var resize: DepthResizeMode
    /// For multi-array inputs: per-channel mean/std on 0…1 RGB.
    public var mean: [Float]
    public var std: [Float]
    public var computeUnits: MLComputeUnits

    public static func builtIn(_ backend: MonocularDepthBackend) -> DepthModelContract {
        switch backend {
        case .depthAnythingV2Small:
            // Apple's Core ML export: image input "image" (518 × 392), Grayscale16Half image output
            // "depth" — relative inverse depth. ImageNet normalisation is baked into the image input.
            return DepthModelContract(input: nil, output: nil, fovOutput: nil, outputKind: .relativeInverseDepth,
                                      resize: .letterbox, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225],
                                      computeUnits: .all)
        case .depthPro:
            // tools/convert_depth_models.py: image input "image" (1536 × 1536, scaled to [-1, 1]),
            // outputs "canonical_inverse_depth" (1 × 1 × 1536 × 1536) and "fov_deg" (1).
            return DepthModelContract(input: nil, output: "canonical_inverse_depth", fovOutput: "fov_deg",
                                      outputKind: .canonicalInverseDepth, resize: .stretch,
                                      mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], computeUnits: .all)
        }
    }

    struct Manifest: Decodable {
        let schemaVersion: Int?
        let backend: String?
        let input: String?
        let output: String?
        let fovOutput: String?
        let outputKind: DepthOutputKind?
        let resize: DepthResizeMode?
        let mean: [Float]?
        let std: [Float]?
        let computeUnits: String?
    }

    /// The built-in contract with a manifest's overrides applied.
    static func load(_ backend: MonocularDepthBackend, manifest: URL?) throws -> DepthModelContract {
        var c = builtIn(backend)
        guard let manifest else { return c }
        let m: Manifest
        do {
            m = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifest))
        } catch {
            throw MonocularDepthError.incompatibleModel(backend, path: manifest.path, reason: "unreadable depth.json: \(error.localizedDescription)")
        }
        if let v = m.schemaVersion, v != 1 {
            throw MonocularDepthError.incompatibleModel(backend, path: manifest.path, reason: "depth.json schemaVersion \(v) is not supported")
        }
        if let b = m.backend, b != backend.rawValue {
            throw MonocularDepthError.incompatibleModel(backend, path: manifest.path, reason: "depth.json is for backend '\(b)'")
        }
        if let v = m.input { c.input = v }
        if let v = m.output { c.output = v }
        if let v = m.fovOutput { c.fovOutput = v.isEmpty ? nil : v }
        if let v = m.outputKind { c.outputKind = v }
        if let v = m.resize { c.resize = v }
        if let v = m.mean, v.count == 3 { c.mean = v }
        if let v = m.std, v.count == 3, v.allSatisfy({ $0 > 0 }) { c.std = v }
        switch m.computeUnits {
        case "cpuOnly": c.computeUnits = .cpuOnly
        case "cpuAndGPU": c.computeUnits = .cpuAndGPU
        case "cpuAndNeuralEngine": c.computeUnits = .cpuAndNeuralEngine
        case "all", nil: break
        case let other?:
            throw MonocularDepthError.incompatibleModel(backend, path: manifest.path, reason: "unknown computeUnits '\(other)'")
        }
        return c
    }
}

/// The contract checked against a loaded model: concrete tensor names, input size and kinds.
struct ResolvedDepthIO {
    enum InputKind { case image(MLImageConstraint), multiArray(shape: [Int]) }
    let input: String
    let inputKind: InputKind
    /// Model input size (width, height) in pixels.
    let inputSize: SIMD2<Int>
    let output: String
    let fovOutput: String?

    /// Validates `contract` against `description`, choosing an input size for `imageSize` when the
    /// model accepts several.
    static func resolve(_ contract: DepthModelContract, description d: MLModelDescription, imageSize: SIMD2<Int>,
                        backend: MonocularDepthBackend, path: String) throws -> ResolvedDepthIO {
        func incompatible(_ reason: String) -> MonocularDepthError {
            .incompatibleModel(backend, path: path, reason: reason)
        }
        let inputs = d.inputDescriptionsByName
        let inputName: String
        if let name = contract.input {
            guard inputs[name] != nil else { throw incompatible("no input named '\(name)' (has \(inputs.keys.sorted()))") }
            inputName = name
        } else {
            let usable = inputs.filter { $0.value.type == .image || $0.value.type == .multiArray }
            guard usable.count == 1, let only = usable.first else {
                throw incompatible("expected one image input, found \(inputs.keys.sorted()); name it in depth.json")
            }
            inputName = only.key
        }
        guard let inDesc = inputs[inputName] else { throw incompatible("missing input") }
        let kind: InputKind
        let size: SIMD2<Int>
        switch inDesc.type {
        case .image:
            guard let constraint = inDesc.imageConstraint else { throw incompatible("image input without a size constraint") }
            kind = .image(constraint)
            size = chooseSize(constraint, imageSize: imageSize, multipleOf: backend == .depthAnythingV2Small ? 14 : 1)
        case .multiArray:
            guard let shape = inDesc.multiArrayConstraint?.shape.map(\.intValue), shape.count >= 3,
                  shape[shape.count - 3] == 3, shape.dropLast(3).allSatisfy({ $0 == 1 }) else {
                throw incompatible("multi-array input must be [1, 3, H, W]")
            }
            kind = .multiArray(shape: shape)
            size = SIMD2(shape[shape.count - 1], shape[shape.count - 2])
        default:
            throw incompatible("input '\(inputName)' is neither an image nor a multi-array")
        }
        guard size.x >= 8, size.y >= 8 else { throw incompatible("input size \(size.x)×\(size.y) is too small") }

        let outputs = d.outputDescriptionsByName
        func isSpatial(_ o: MLFeatureDescription) -> Bool {
            if o.type == .image { return true }
            guard o.type == .multiArray, let shape = o.multiArrayConstraint?.shape.map(\.intValue), shape.count >= 2 else { return false }
            return shape.dropLast(2).allSatisfy { $0 == 1 } && shape[shape.count - 1] > 1 && shape[shape.count - 2] > 1
        }
        let outputName: String
        if let name = contract.output {
            guard let o = outputs[name] else { throw incompatible("no output named '\(name)' (has \(outputs.keys.sorted()))") }
            guard isSpatial(o) else { throw incompatible("output '\(name)' is not a 2D depth map") }
            outputName = name
        } else {
            let spatial = outputs.filter { $0.key != contract.fovOutput && isSpatial($0.value) }.map(\.key).sorted()
            guard spatial.count == 1, let only = spatial.first else {
                throw incompatible("expected one 2D depth output, found \(outputs.keys.sorted()); name it in depth.json")
            }
            outputName = only
        }
        if contract.outputKind == .canonicalInverseDepth, let fov = contract.fovOutput, let o = outputs[fov], o.type != .multiArray {
            throw incompatible("field-of-view output '\(fov)' is not a multi-array")
        }
        let fov = contract.fovOutput.flatMap { outputs[$0] != nil ? $0 : nil }
        return ResolvedDepthIO(input: inputName, inputKind: kind, inputSize: size, output: outputName, fovOutput: fov)
    }

    /// Fixed-size models: their size. Flexible ones: the enumerated size closest in aspect ratio to the
    /// photo, or, for a range, the default size's long side at the photo's aspect (a multiple of `m`).
    static func chooseSize(_ c: MLImageConstraint, imageSize: SIMD2<Int>, multipleOf m: Int) -> SIMD2<Int> {
        let fixed = SIMD2(c.pixelsWide, c.pixelsHigh)
        let aspect = Double(imageSize.x) / Double(max(imageSize.y, 1))
        let sc = c.sizeConstraint
        switch sc.type {
        case .enumerated:
            let sizes = sc.enumeratedImageSizes.map { SIMD2($0.pixelsWide, $0.pixelsHigh) }
            return sizes.min { abs(log(Double($0.x) / Double($0.y) / aspect)) < abs(log(Double($1.x) / Double($1.y) / aspect)) } ?? fixed
        case .range:
            let long = Double(max(fixed.x, fixed.y))
            func round(_ v: Double, _ r: NSRange) -> Int {
                let snapped = max(Int((v / Double(m)).rounded()) * m, m)
                return min(max(snapped, r.location), r.location + max(r.length - 1, 0))
            }
            let w = aspect >= 1 ? long : long * aspect, h = aspect >= 1 ? long / aspect : long
            return SIMD2(round(w, sc.pixelsWideRange), round(h, sc.pixelsHighRange))
        default:
            return fixed
        }
    }
}

/// Where the photo lands inside the model input, and so which part of the output belongs to it.
public struct DepthInputLayout: Equatable, Sendable {
    public let imageSize: SIMD2<Int>
    public let inputSize: SIMD2<Int>
    /// The photo's rectangle in the input tensor, in input pixels (top-left origin, integral).
    public let content: CGRect

    public init(imageSize: SIMD2<Int>, inputSize: SIMD2<Int>, mode: DepthResizeMode) {
        self.imageSize = imageSize
        self.inputSize = inputSize
        let iw = Double(inputSize.x), ih = Double(inputSize.y)
        switch mode {
        case .stretch:
            content = CGRect(x: 0, y: 0, width: iw, height: ih)
        case .letterbox:
            let r = min(iw / Double(imageSize.x), ih / Double(imageSize.y))
            let w = min(max((Double(imageSize.x) * r).rounded(), 1), iw), h = min(max((Double(imageSize.y) * r).rounded(), 1), ih)
            content = CGRect(x: ((iw - w) / 2).rounded(.down), y: ((ih - h) / 2).rounded(.down), width: w, height: h)
        }
    }

    /// The content rectangle in an output of `outputSize` (which may differ from the input's),
    /// as integer bounds `x0..<x1`, `y0..<y1`, never empty.
    public func contentBounds(outputSize: SIMD2<Int>) -> (x0: Int, y0: Int, x1: Int, y1: Int) {
        let sx = Double(outputSize.x) / Double(inputSize.x), sy = Double(outputSize.y) / Double(inputSize.y)
        let x0 = min(max(Int((content.minX * sx).rounded()), 0), outputSize.x - 1)
        let y0 = min(max(Int((content.minY * sy).rounded()), 0), outputSize.y - 1)
        let x1 = min(max(Int((content.maxX * sx).rounded()), x0 + 1), outputSize.x)
        let y1 = min(max(Int((content.maxY * sy).rounded()), y0 + 1), outputSize.y)
        return (x0, y0, x1, y1)
    }
}
