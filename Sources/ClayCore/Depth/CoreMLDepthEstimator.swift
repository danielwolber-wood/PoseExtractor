import CoreGraphics
import CoreML
import CoreVideo
import CryptoKit
import Foundation

/// Runs a monocular depth model through Core ML (Neural Engine / GPU on Apple Silicon).
///
/// Pipeline per photo, all on the caller's thread (the app calls it off the main actor):
/// 1. **Input:** the upright photo (`LoadedImage.cgImage`, EXIF orientation already applied) is drawn
///    into a BGRA pixel buffer of the model's input size — stretched or letterboxed per the contract
///    (`DepthInputLayout`) — or into a normalised `[1, 3, H, W]` Float32 array for multi-array inputs.
/// 2. **Output:** the depth tensor is read with its strides (`DepthTensor`), the letterbox padding is
///    cropped off, and the result is kept at the model's output resolution. It still covers exactly
///    the whole photo, so `DepthMap` sampling maps image pixels onto it; no full-resolution copy is made.
/// 3. **Semantics** (`DepthOutputKind`): metric output stays in metres. Depth Pro's canonical inverse
///    depth becomes metres using the photo's EXIF focal length, else the model's own field-of-view
///    estimate; with neither (only the pipeline's 50 mm guess) it is labelled *relative* inverse
///    depth instead, because its scale would just be that guess. Relative outputs are divided by
///    their 99th percentile and labelled relative — never metres.
public final class CoreMLDepthEstimator: MonocularDepthEstimating, @unchecked Sendable {
    public let backend: MonocularDepthBackend
    public let location: DepthModelLocation
    public let contract: DepthModelContract
    /// Seconds spent compiling (first use of a package) and loading the model.
    public let loadSeconds: TimeInterval
    private let model: MLModel
    private let lock = NSLock()
    private var predictions = 0

    public init(location: DepthModelLocation, computeUnits: MLComputeUnits? = nil) throws {
        backend = location.backend
        self.location = location
        var contract = try DepthModelContract.load(location.backend, manifest: location.manifestURL)
        if let computeUnits { contract.computeUnits = computeUnits }
        self.contract = contract
        let t0 = Date()
        let config = MLModelConfiguration()
        config.computeUnits = contract.computeUnits
        do {
            model = try MLModel(contentsOf: try Self.compiled(location), configuration: config)
        } catch let e as MonocularDepthError {
            throw e
        } catch {
            throw MonocularDepthError.loadFailed(location.backend, path: location.modelURL.path, reason: error.localizedDescription)
        }
        loadSeconds = Date().timeIntervalSince(t0)
        // Fail early on an incompatible model rather than on the first photo.
        _ = try ResolvedDepthIO.resolve(contract, description: model.modelDescription, imageSize: SIMD2(640, 480),
                                        backend: backend, path: location.modelURL.path)
    }

    /// Compiled models load directly; packages are compiled once and cached in
    /// ~/Library/Caches/ClayPose/depth, keyed by path, size and modification date.
    static func compiled(_ location: DepthModelLocation) throws -> URL {
        let url = location.modelURL
        if url.pathExtension.lowercased() == "mlmodelc" { return url }
        let fm = FileManager.default
        let stampFile = url.pathExtension.lowercased() == "mlpackage" ? url.appendingPathComponent("Manifest.json") : url
        let attrs = try? fm.attributesOfItem(atPath: stampFile.path)
        let stamp = "\(url.standardizedFileURL.path)|\((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)|\(attrs?[.size] ?? 0)"
        let key = SHA256.hash(data: Data(stamp.utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
        let cacheDir = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("ClayPose/depth", isDirectory: true)
        let cached = cacheDir.appendingPathComponent("\(location.backend.rawValue)-\(key).mlmodelc")
        if fm.fileExists(atPath: cached.path) { return cached }
        try validatePackage(location)
        let tmp: URL
        do {
            tmp = try MLModel.compileModel(at: url)
        } catch {
            throw MonocularDepthError.loadFailed(location.backend, path: url.path, reason: "compilation failed: \(error.localizedDescription)")
        }
        try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        do { try fm.moveItem(at: tmp, to: cached) } catch {
            // Another process may have won the race; use theirs, else the temporary copy.
            if !fm.fileExists(atPath: cached.path) { return tmp }
        }
        return cached
    }

    /// Checks an .mlpackage's structure before compiling it: Core ML's compiler aborts the whole process
    /// (an uncaught C++ exception) on a malformed Manifest.json rather than throwing.
    static func validatePackage(_ location: DepthModelLocation) throws {
        let url = location.modelURL
        guard url.pathExtension.lowercased() == "mlpackage" else { return }
        func broken(_ why: String) -> MonocularDepthError { .loadFailed(location.backend, path: url.path, reason: why) }
        guard let data = try? Data(contentsOf: url.appendingPathComponent("Manifest.json")),
              let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw broken("Manifest.json is missing or not JSON (incomplete download?)")
        }
        guard manifest["fileFormatVersion"] is String, let root = manifest["rootModelIdentifier"] as? String,
              let items = manifest["itemInfoEntries"] as? [String: Any], let rootItem = items[root] as? [String: Any],
              let path = rootItem["path"] as? String else {
            throw broken("Manifest.json does not describe a Core ML model package")
        }
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent("Data").appendingPathComponent(path).path) else {
            throw broken("the package is missing \(path) (incomplete download?)")
        }
    }

    public func estimate(_ image: LoadedImage) throws -> MonocularDepthEstimate {
        let t0 = Date()
        let imageSize = SIMD2(image.width, image.height)
        let io = try ResolvedDepthIO.resolve(contract, description: model.modelDescription, imageSize: imageSize,
                                             backend: backend, path: location.modelURL.path)
        let layout = DepthInputLayout(imageSize: imageSize, inputSize: io.inputSize, mode: contract.resize)
        let inputValue = try makeInput(image.cgImage, io: io, layout: layout)

        lock.lock()
        let cold = predictions == 0
        predictions += 1
        let tPredict = Date()
        let out: MLFeatureProvider
        do {
            out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [io.input: inputValue]))
        } catch {
            lock.unlock()
            throw MonocularDepthError.inferenceFailed(backend, reason: error.localizedDescription)
        }
        let inference = Date().timeIntervalSince(tPredict)
        lock.unlock()

        guard let value = out.featureValue(for: io.output) else {
            throw MonocularDepthError.inferenceFailed(backend, reason: "no '\(io.output)' output")
        }
        let raw: DepthTensor.Plane
        do { raw = try DepthTensor.read(value) } catch {
            throw MonocularDepthError.incompatibleModel(backend, path: location.modelURL.path, reason: "\(error)")
        }
        let b = layout.contentBounds(outputSize: SIMD2(raw.width, raw.height))
        var plane = DepthTensor.crop(raw, x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1)

        var notes: [String] = []
        let fovDegrees = io.fovOutput.flatMap { DepthTensor.scalar(out.featureValue(for: $0)) }
        let predictedFocal = fovDegrees.flatMap { fov -> Double? in
            guard fov > 1, fov < 179 else { return nil }
            // Depth Pro's field of view is horizontal: f = (W / 2) / tan(fov / 2).
            return 0.5 * Double(image.width) / tan(0.5 * fov * .pi / 180)
        }
        let representation: DepthRepresentation
        var focalUsed: Double?, focalSource: FocalLengthSource?
        switch contract.outputKind {
        case .metricDepth:
            representation = .metricDepth
            for i in plane.values.indices where !(plane.values[i].isFinite && plane.values[i] > 0) { plane.values[i] = 0 }
        case .canonicalInverseDepth:
            if image.focalFromEXIF {
                (focalUsed, focalSource) = (image.focalLengthPixels, .exif)
            } else if let predictedFocal {
                (focalUsed, focalSource) = (predictedFocal, .predicted)
            } else {
                (focalUsed, focalSource) = (image.focalLengthPixels, .assumed)
            }
            let f = focalUsed ?? image.focalLengthPixels
            if focalSource == .assumed {
                // Scale = the pipeline's lens guess: keep it as relative inverse depth (larger = nearer).
                representation = .relativeInverseDepth
                _ = DepthTensor.normalizeRelative(&plane.values)
                notes.append("no EXIF focal length and no field-of-view output: Depth Pro output treated as relative")
            } else {
                // Depth Pro: inverse depth = canonical × (W / f); depth = 1 / clamp(inverse, 1e-4, 1e4).
                representation = .metricDepth
                let k = Float(Double(image.width) / f)
                for i in plane.values.indices {
                    let v = plane.values[i]
                    plane.values[i] = v.isFinite && v > 0 ? 1 / min(max(v * k, 1e-4), 1e4) : 0
                }
            }
        case .relativeInverseDepth, .relativeDepth:
            representation = contract.outputKind == .relativeInverseDepth ? .relativeInverseDepth : .relativeDepth
            let scale = DepthTensor.normalizeRelative(&plane.values)
            notes.append(String(format: "relative output normalised by its 99th percentile (%.4g)", scale))
        }
        if let fovDegrees { notes.append(String(format: "model field of view %.1f°", fovDegrees)) }

        let map = DepthMap(width: plane.width, height: plane.height, values: plane.values, representation: representation)
        return MonocularDepthEstimate(
            backend: backend, map: map, estimatedFocalLengthPixels: predictedFocal, focalLengthUsedPixels: focalUsed,
            focalSource: focalSource, modelPath: location.modelURL.path, inputSize: io.inputSize,
            outputSize: SIMD2(raw.width, raw.height), resize: contract.resize, loadSeconds: cold ? loadSeconds : 0,
            inferenceSeconds: inference, processingSeconds: Date().timeIntervalSince(t0) - inference, coldStart: cold,
            notes: notes)
    }

    // MARK: Input

    private func makeInput(_ image: CGImage, io: ResolvedDepthIO, layout: DepthInputLayout) throws -> MLFeatureValue {
        let w = io.inputSize.x, h = io.inputSize.y
        switch io.inputKind {
        case .image(let constraint):
            if let buffer = Self.pixelBuffer(image, layout: layout, format: constraint.pixelFormatType) {
                return MLFeatureValue(pixelBuffer: buffer)
            }
            // Other pixel formats: draw at the exact input size, and let Core ML convert (no rescale).
            guard let resized = Self.render(image, layout: layout) else {
                throw MonocularDepthError.inferenceFailed(backend, reason: "could not resize the photo")
            }
            do {
                return try MLFeatureValue(cgImage: resized, pixelsWide: w, pixelsHigh: h,
                                          pixelFormatType: constraint.pixelFormatType, options: nil)
            } catch {
                throw MonocularDepthError.inferenceFailed(backend, reason: "image input conversion failed: \(error.localizedDescription)")
            }
        case .multiArray(let shape):
            guard let ctx = Self.context(width: w, height: h) else {
                throw MonocularDepthError.inferenceFailed(backend, reason: "could not allocate the input image")
            }
            Self.draw(image, in: ctx, layout: layout)
            guard let data = ctx.data else { throw MonocularDepthError.inferenceFailed(backend, reason: "no input pixels") }
            let px = data.assumingMemoryBound(to: UInt8.self)
            let array: MLMultiArray
            do { array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32) } catch {
                throw MonocularDepthError.inferenceFailed(backend, reason: "could not allocate the input tensor")
            }
            let rowBytes = ctx.bytesPerRow
            array.withUnsafeMutableBufferPointer(ofType: Float.self) { dst, strides in
                let sc = strides[strides.count - 3], sy = strides[strides.count - 2], sx = strides[strides.count - 1]
                for c in 0..<3 {
                    let mean = contract.mean[c], std = contract.std[c]
                    for y in 0..<h {
                        for x in 0..<w {
                            dst[c * sc + y * sy + x * sx] = (Float(px[y * rowBytes + x * 4 + c]) / 255 - mean) / std
                        }
                    }
                }
            }
            return MLFeatureValue(multiArray: array)
        }
    }

    /// RGBA8 sRGB context (premultiplied; photos are opaque).
    static func context(width: Int, height: Int) -> CGContext? {
        CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    }

    /// Draws the photo into `layout.content`, padding the rest with the photo's mean colour (a neutral
    /// fill: black or white bars would read as a strong depth edge at the letterbox boundary).
    static func draw(_ image: CGImage, in ctx: CGContext, layout: DepthInputLayout) {
        let H = CGFloat(layout.inputSize.y)
        if layout.content.size != CGSize(width: layout.inputSize.x, height: layout.inputSize.y) {
            ctx.setFillColor(meanColour(image))
            ctx.fill(CGRect(x: 0, y: 0, width: layout.inputSize.x, height: layout.inputSize.y))
        }
        ctx.interpolationQuality = .high
        // CGContext has a bottom-left origin; `content` is top-left.
        let r = layout.content
        ctx.draw(image, in: CGRect(x: r.minX, y: H - r.maxY, width: r.width, height: r.height))
    }

    static func render(_ image: CGImage, layout: DepthInputLayout) -> CGImage? {
        guard let ctx = context(width: layout.inputSize.x, height: layout.inputSize.y) else { return nil }
        draw(image, in: ctx, layout: layout)
        return ctx.makeImage()
    }

    /// A BGRA (or ARGB) pixel buffer at the input size with the photo drawn in; nil for other formats.
    static func pixelBuffer(_ image: CGImage, layout: DepthInputLayout, format: OSType) -> CVPixelBuffer? {
        let info: UInt32
        switch format {
        case kCVPixelFormatType_32BGRA: info = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        case kCVPixelFormatType_32ARGB: info = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        default: return nil
        }
        var buffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                     kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, layout.inputSize.x, layout.inputSize.y, format, attrs, &buffer) == kCVReturnSuccess,
              let buffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: layout.inputSize.x, height: layout.inputSize.y,
                                  bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: info) else { return nil }
        draw(image, in: ctx, layout: layout)
        return buffer
    }

    static func meanColour(_ image: CGImage) -> CGColor {
        var px = [UInt8](repeating: 128, count: 4)
        if let ctx = CGContext(data: &px, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                               space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
            ctx.interpolationQuality = .medium
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        }
        return CGColor(srgbRed: CGFloat(px[0]) / 255, green: CGFloat(px[1]) / 255, blue: CGFloat(px[2]) / 255, alpha: 1)
    }
}
