import ArmatureCore
import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

// Deterministic checks of the monocular depth plumbing. Needs no downloaded model and no body model:
//   swift run -c release armature-depth-selftest
// The Core ML path (compile, load, predict, tensor decoding) runs on tiny generated fixtures
// (Fixtures.swift, from tools/make_depth_test_fixtures.py) that mimic the real models' I/O contracts.
//
//   swift run -c release armature-depth-selftest --bench photo.jpg [--runs 5]
// instead times each *installed* backend on a photo: model load (compile on first use), the first
// (cold) prediction and warm predictions, and summarises its output.

if let i = CommandLine.arguments.firstIndex(of: "--bench"), i + 1 < CommandLine.arguments.count {
    let url = URL(fileURLWithPath: CommandLine.arguments[i + 1])
    let runs = CommandLine.arguments.firstIndex(of: "--runs").flatMap { Int(CommandLine.arguments[$0 + 1]) } ?? 5
    let image = try LoadedImage(url: url)
    let models = ArmaturePipeline.defaultModelsDirectory() ?? URL(fileURLWithPath: "Models")
    let installed = ArmaturePipeline(modelsDirectory: models).installedDepthModels()
    print("\(url.lastPathComponent): \(image.width)×\(image.height), focal \(Int(image.focalLengthPixels)) px (\(image.focalFromEXIF ? "EXIF" : "assumed"))")
    if installed.isEmpty { print("no depth models installed (see docs/depth.md)") }
    for b in MonocularDepthBackend.allCases {
        guard let loc = installed[b] else { print("\(b.displayName): not installed"); continue }
        let t0 = Date()
        let estimator = try CoreMLDepthEstimator(location: loc)
        let load = Date().timeIntervalSince(t0)
        var times: [Double] = []
        var last: MonocularDepthEstimate?
        for _ in 0..<max(runs, 2) {
            let t = Date()
            last = try estimator.estimate(image)
            times.append(Date().timeIntervalSince(t))
        }
        guard let e = last else { continue }
        let warm = times.dropFirst().sorted()[times.count / 2 - 1]
        let valid = e.map.values.filter { $0 > 0 && $0.isFinite }.sorted()
        print("\(b.displayName): \(loc.modelURL.lastPathComponent), compute units \(["cpuOnly", "cpuAndGPU", "all", "cpuAndNeuralEngine"][min(estimator.contract.computeUnits.rawValue, 3)])")
        print(String(format: "  load %.0f ms, cold prediction %.0f ms, warm %.0f ms (median of %d, incl. resize + decode)",
                     load * 1000, times[0] * 1000, warm * 1000, times.count - 1))
        print("  input \(e.inputSize.x)×\(e.inputSize.y) (\(e.resize.rawValue)), output \(e.outputSize.x)×\(e.outputSize.y), map \(e.map.width)×\(e.map.height), \(e.map.representation.rawValue)")
        if let lo = valid.first, let hi = valid.last {
            print(String(format: "  valid %.0f%%, range %.3g–%.3g (median %.3g)%@", e.map.validFraction * 100, lo, hi, valid[valid.count / 2],
                         e.map.isAbsolute ? " m" : ""))
        }
        if let f = e.estimatedFocalLengthPixels { print(String(format: "  predicted focal %.0f px", f)) }
        if let f = e.focalLengthUsedPixels, let src = e.focalSource { print(String(format: "  scaled with focal %.0f px (%@)", f, src.rawValue)) }
        for n in e.notes { print("  \(n)") }
    }
    exit(0)
}

var failures: [String] = []
var passed = 0
func check(_ condition: @autoclosure () throws -> Bool, _ message: String) {
    let ok: Bool
    do { ok = try condition() } catch { ok = false; print("error in '\(message)': \(error)") }
    if ok { passed += 1 } else { failures.append(message); print("FAIL: \(message)") }
}
func close(_ a: Double, _ b: Double, _ tol: Double) -> Bool { abs(a - b) <= tol }

let fm = FileManager.default
let tmp = fm.temporaryDirectory.appendingPathComponent("armature-depth-selftest-\(UUID().uuidString)")
try fm.createDirectory(at: tmp, withIntermediateDirectories: true)
defer { try? fm.removeItem(at: tmp) }

func touchPackage(_ url: URL) throws {
    try fm.createDirectory(at: url, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: url.appendingPathComponent("Manifest.json"))
}
func writeFixture(_ files: [String: String], to package: URL) throws {
    for (path, b64) in files {
        let url = package.appendingPathComponent(path)
        if path.hasSuffix("/") {
            try fm.createDirectory(at: url, withIntermediateDirectories: true)
            continue
        }
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let data = Data(base64Encoded: b64) else { throw CocoaError(.fileReadCorruptFile) }
        try data.write(to: url)
    }
}

// MARK: Model discovery

do {
    let env = tmp.appendingPathComponent("env"), models = tmp.appendingPathComponent("models"),
        bundle = tmp.appendingPathComponent("bundle"), support = tmp.appendingPathComponent("support")
    let roots = DepthModelLocator.searchRoots(modelsDirectory: models, environment: ["ARMATURE_MODELS": env.path, "CLAY_MODELS": "/legacy"],
                                              bundleResources: bundle, currentDirectory: models.deletingLastPathComponent(),
                                              executable: nil, applicationSupport: support)
    check(roots.first == env, "ARMATURE_MODELS is searched first, ahead of the pre-rename CLAY_MODELS")
    check(ModelLocations.environmentDirectory(["CLAY_MODELS": env.path]) == env, "CLAY_MODELS still works on its own")
    check(roots.count == Set(roots.map(\.path)).count, "search roots are de-duplicated")
    check(roots.contains(bundle.appendingPathComponent("Models")) && Array(roots.suffix(2)) == [support.appendingPathComponent("Armature/Models"), support.appendingPathComponent("ClayStudio/Models")],
          "bundle and Application Support are searched, Application Support (new, then pre-rename) last")

    let da = models.appendingPathComponent("depth/depth-anything-v2-small")
    try touchPackage(da.appendingPathComponent("DepthAnythingV2SmallF32.mlpackage"))
    check(DepthModelLocator.locate(.depthAnythingV2Small, roots: roots)?.modelURL.lastPathComponent == "DepthAnythingV2SmallF32.mlpackage",
          "finds Apple's Float32 package")
    try touchPackage(da.appendingPathComponent("DepthAnythingV2SmallF16.mlpackage"))
    check(DepthModelLocator.locate(.depthAnythingV2Small, roots: roots)?.modelURL.lastPathComponent == "DepthAnythingV2SmallF16.mlpackage",
          "prefers Float16 over Float32")
    try touchPackage(da.appendingPathComponent("DepthAnythingV2SmallF16.mlmodelc"))
    check(DepthModelLocator.locate(.depthAnythingV2Small, roots: roots)?.modelURL.lastPathComponent == "DepthAnythingV2SmallF16.mlmodelc",
          "prefers a compiled model over its package")
    check(DepthModelLocator.locate(.depthPro, roots: roots) == nil, "Depth Pro absent")
    let envDA = env.appendingPathComponent("depth/DepthAnythingV2SmallF16.mlpackage")
    try touchPackage(envDA)
    check(DepthModelLocator.locate(.depthAnythingV2Small, roots: roots)?.modelURL.standardizedFileURL.path == envDA.standardizedFileURL.path,
          "an earlier root wins; Apple's file name is accepted directly in Models/depth/")
    let pro = support.appendingPathComponent("Armature/Models/depth/depth-pro")
    try touchPackage(pro.appendingPathComponent("model.mlpackage"))
    try Data(#"{"schemaVersion": 1}"#.utf8).write(to: pro.appendingPathComponent("depth.json"))
    let proLoc = DepthModelLocator.locate(.depthPro, roots: roots)
    check(proLoc?.modelURL.lastPathComponent == "model.mlpackage" && proLoc?.manifestURL?.lastPathComponent == "depth.json",
          "Depth Pro found in Application Support with its manifest")
    check(DepthModelLocator.explicit(.depthPro, url: pro)?.modelURL.path == proLoc?.modelURL.path, "explicit directory")
    check(DepthModelLocator.explicit(.depthPro, url: pro.appendingPathComponent("model.mlpackage")) != nil, "explicit package")
    check(DepthModelLocator.explicit(.depthPro, url: tmp.appendingPathComponent("nope.mlpackage")) == nil, "explicit missing path")
}

// MARK: Source priority and modes

do {
    let metric = DepthMap(width: 2, height: 2, values: [1, 1, 1, 1], representation: .metricDepth)
    let portrait = DepthMap(width: 2, height: 2, values: [1, 1, 1, 1], representation: .relativeDepth)
    let both: Set<MonocularDepthBackend> = [.depthAnythingV2Small, .depthPro]
    typealias D = DepthSourceDecision
    check(D.resolve(embedded: metric, mode: .backend(.depthPro), fallback: true, installed: both) == .embeddedMetric,
          "embedded metric depth beats a selected backend")
    check(D.resolve(embedded: portrait, mode: .automatic, fallback: true, installed: both) == .monocular(.depthAnythingV2Small, fallbackFrom: nil),
          "relative embedded depth doesn't block monocular; auto prefers Depth Anything")
    check(D.resolve(embedded: nil, mode: .automatic, fallback: true, installed: [.depthPro]) == .monocular(.depthPro, fallbackFrom: nil),
          "auto uses Depth Pro when it's the only one")
    check(D.resolve(embedded: nil, mode: .backend(.depthPro), fallback: true, installed: both) == .monocular(.depthPro, fallbackFrom: nil),
          "a selected backend wins over auto order")
    check(D.resolve(embedded: nil, mode: .backend(.depthPro), fallback: true, installed: [.depthAnythingV2Small])
          == .monocular(.depthAnythingV2Small, fallbackFrom: .depthPro), "fallback to the installed backend")
    if case .none = D.resolve(embedded: nil, mode: .backend(.depthPro), fallback: false, installed: [.depthAnythingV2Small]) {
        passed += 1
    } else { check(false, "no fallback when disabled") }
    if case .none = D.resolve(embedded: nil, mode: .automatic, fallback: true, installed: []) { passed += 1 } else { check(false, "auto, nothing installed") }
    if case .none = D.resolve(embedded: nil, mode: .disabled, fallback: true, installed: both) { passed += 1 } else { check(false, "disabled") }
    check(MonocularDepthMode(argument: "none") == .disabled && MonocularDepthMode(argument: "auto") == .automatic
          && MonocularDepthMode(argument: "depth-pro") == .backend(.depthPro)
          && MonocularDepthMode(argument: "depth-anything-v2-small") == .backend(.depthAnythingV2Small)
          && MonocularDepthMode(argument: "midas") == nil, "CLI mode spellings")
}

// MARK: Depth maps: labelling, sampling, resizing, normalisation

do {
    let m = DepthMap(width: 2, height: 2, values: [1, 2, 3, 4], representation: .relativeInverseDepth)
    check(!m.isAbsolute && m.representation.largerIsNearer, "relative inverse depth: not metric, larger = nearer")
    check(!DepthRepresentation.relativeDepth.largerIsNearer && DepthRepresentation.metricDepth.isMetric, "labels")
    let size = SIMD2(20.0, 20.0)
    check(close(Double(m.sample(at: SIMD2(10, 10), imageSize: size) ?? 0), 2.5, 1e-5), "bilinear centre")
    check(close(Double(m.sample(at: SIMD2(5, 5), imageSize: size) ?? 0), 1, 1e-5), "bilinear at a pixel centre")
    let holes = DepthMap(width: 2, height: 2, values: [1, 0, .nan, 3], representation: .metricDepth)
    check(close(Double(holes.sample(at: SIMD2(10, 10), imageSize: size) ?? 0), 2, 1e-5), "invalid neighbours are skipped")
    check(DepthMap(width: 1, height: 1, values: [0], representation: .metricDepth).sample(at: SIMD2(0.5, 0.5), imageSize: SIMD2(1, 1)) == nil,
          "no data → nil")
    check(close(holes.validFraction, 0.5, 1e-9), "valid fraction")
    let up = m.resampled(width: 4, height: 6)
    check(up.width == 4 && up.height == 6 && up.representation == .relativeInverseDepth, "resampling keeps the label")
    check(close(Double(up.sample(at: SIMD2(10, 10), imageSize: size) ?? 0), 2.5, 0.2), "resampled values")

    var raw: [Float] = (1...100).map(Float.init) + [-1, .nan, 0, .infinity]
    let scale = DepthTensor.normalizeRelative(&raw)
    check(scale == 99, "normalised by the 99th percentile")
    check(raw[0] == 1 / 99 && close(Double(raw[49] / raw[9]), 5, 1e-6) && raw[100...].allSatisfy { $0 == 0 },
          "positive scale only: ratios and zero kept, invalid → 0")
}

// MARK: Input layout (resizing) and tensor decoding

do {
    let exact = DepthInputLayout(imageSize: SIMD2(1036, 784), inputSize: SIMD2(518, 392), mode: .letterbox)
    check(exact.content == CGRect(x: 0, y: 0, width: 518, height: 392), "same aspect: no padding")
    let portrait = DepthInputLayout(imageSize: SIMD2(1000, 1500), inputSize: SIMD2(518, 392), mode: .letterbox)
    check(portrait.content == CGRect(x: 128, y: 0, width: 261, height: 392), "portrait letterboxed into landscape")
    let b = portrait.contentBounds(outputSize: SIMD2(259, 196))
    check(b.x0 == 64 && b.x1 == 195 && b.y0 == 0 && b.y1 == 196, "content bounds at half output resolution")
    let wide = DepthInputLayout(imageSize: SIMD2(4000, 1000), inputSize: SIMD2(518, 392), mode: .letterbox)
    check(wide.content.minX == 0 && wide.content.width == 518 && close(wide.content.height, 130, 1), "panorama letterboxed")
    let stretch = DepthInputLayout(imageSize: SIMD2(1000, 1500), inputSize: SIMD2(1536, 1536), mode: .stretch)
    check(stretch.content == CGRect(x: 0, y: 0, width: 1536, height: 1536), "stretch fills the input")

    // Float32 with padded row stride.
    var storage: [Float] = [1, 2, 3, -9, 4, 5, 6, -9]
    let padded = try storage.withUnsafeMutableBytes { p in
        try MLMultiArray(dataPointer: p.baseAddress!, shape: [1, 1, 2, 3], dataType: .float32, strides: [8, 8, 4, 1])
    }
    let plane = try DepthTensor.read(padded)
    check(plane.width == 3 && plane.height == 2 && plane.values == [1, 2, 3, 4, 5, 6], "multi-array strides honoured")
    // Float16 (bit patterns 1.0, 0.5, 2.0, 0).
    let half = try MLMultiArray(shape: [1, 2, 2], dataType: .float16)
    half.withUnsafeMutableBytes { p, _ in
        let h = p.bindMemory(to: UInt16.self)
        h[0] = 0x3C00; h[1] = 0x3800; h[2] = 0x4000; h[3] = 0
    }
    check(try DepthTensor.read(half).values == [1, 0.5, 2, 0], "Float16 multi-array")
    var bad = false
    do { _ = try DepthTensor.read(try MLMultiArray(shape: [2, 2, 2], dataType: .float32)) } catch { bad = true }
    check(bad, "non-singleton leading dimension rejected")
    // One-component Float16 pixel buffer.
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(nil, 2, 1, kCVPixelFormatType_OneComponent16Half, nil, &pb)
    if let pb {
        CVPixelBufferLockBaseAddress(pb, [])
        let p = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt16.self)
        p[0] = 0x3C00; p[1] = 0x4000
        CVPixelBufferUnlockBaseAddress(pb, [])
        check(try DepthTensor.read(pb).values == [1, 2], "Grayscale16Half image output")
    }
}

// MARK: Scale estimation

do {
    let s = 0.25
    let z1 = [4.0, 4.05, 4.1]
    let single = DepthCalibration.estimate(representation: .relativeInverseDepth,
                                           anchors: [DepthAnchor(values: z1.map { 1 / (s * $0) }, bodyDepths: z1)])
    check(single.method == .perPersonScale && close(single.personScales[0] ?? 0, s, 1e-9), "one person: shift-free scale")
    check(!single.constrainsDistance && single.scaleEstimated, "one person can't observe distance")
    check(close(single.metres(1 / (s * 4.5), person: 0) ?? 0, 4.5, 1e-9), "calibrated metres")

    let z2 = [6.0, 6.1, 5.95]
    let shared = DepthCalibration.estimate(representation: .relativeInverseDepth, anchors: [
        DepthAnchor(values: z1.map { 1 / (s * $0) }, bodyDepths: z1),
        DepthAnchor(values: z2.map { 1 / (s * $0) }, bodyDepths: z2.map { $0 * 1.03 }),
    ])
    check(shared.method == .sharedScale && shared.constrainsDistance, "two people agreeing: shared scale")

    // With a shift, 1/z = s·v + t: per-person shift-free scales disagree, the affine fit recovers it.
    let t = 0.1, zNear = [2.0, 2.02, 2.05], zFar = [6.0, 6.05, 6.1]
    let affine = DepthCalibration.estimate(representation: .relativeInverseDepth, anchors: [
        DepthAnchor(values: zNear.map { (1 / $0 - t) / s }, bodyDepths: zNear),
        DepthAnchor(values: zFar.map { (1 / $0 - t) / s }, bodyDepths: zFar),
    ])
    check(affine.method == .affine && close(affine.scale, s, 0.01) && close(affine.shift, t, 0.005), "affine scale and shift")

    let noisy = DepthCalibration.estimate(representation: .relativeInverseDepth,
                                          anchors: [DepthAnchor(values: [1, 0.3, 2.2, 0.9], bodyDepths: [4, 4.05, 4.1, 4])])
    check(noisy.method == .unusable && !noisy.usable, "inconsistent torso depth rejected")
    check(DepthCalibration.estimate(representation: .relativeInverseDepth, anchors: [nil]).method == .unusable, "no anchors")

    let metric = DepthCalibration.estimate(representation: .metricDepth,
                                           anchors: [DepthAnchor(values: [4.1, 4.2, 4.0], bodyDepths: [4.0, 4.1, 4.05])])
    check(metric.method == .metric && !metric.scaleEstimated && metric.constrainsDistance && close(metric.metricAgreement ?? 0, 0.99, 0.03),
          "metric map agreeing with the prior")
    let off = DepthCalibration.estimate(representation: .metricDepth,
                                        anchors: [DepthAnchor(values: [8.1, 8.2, 8.0], bodyDepths: [4.0, 4.1, 4.05])])
    check(off.method == .metric && !off.constrainsDistance, "metric map 2× off the prior: relative use only")
}

// MARK: Core ML backends (fixtures)

/// A photo with a known red-channel pattern, written with an EXIF orientation and loaded upright.
func orientedPhoto(rawWidth w: Int, rawHeight h: Int, orientation: CGImagePropertyOrientation) throws -> LoadedImage {
    let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    let px = ctx.data!.assumingMemoryBound(to: UInt8.self)
    for y in 0..<h {
        for x in 0..<w {
            // Asymmetric in both axes, smooth enough to survive resampling.
            let r = 30 + 180 * Double(x) / Double(w - 1) * (0.4 + 0.6 * Double(y) / Double(h - 1))
            px[(y * w + x) * 4] = UInt8(r)
            px[(y * w + x) * 4 + 1] = 90
            px[(y * w + x) * 4 + 2] = 40
        }
    }
    let url = tmp.appendingPathComponent("photo-\(orientation.rawValue).png")
    let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, ctx.makeImage()!, [kCGImagePropertyOrientation: orientation.rawValue] as CFDictionary)
    CGImageDestinationFinalize(dest)
    return try LoadedImage(url: url)
}

func redChannel(_ image: CGImage) -> [Double] {
    let w = image.width, h = image.height
    let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    let px = ctx.data!.assumingMemoryBound(to: UInt8.self)
    return (0..<(w * h)).map { Double(px[$0 * 4]) }
}

/// Worst relative error of the depth map against the photo's red channel, after the best single scale
/// (relative maps are known only up to scale) or, with `absolute`, as is. Samples a grid inside the image.
func mismatch(_ map: DepthMap, _ image: LoadedImage, absolute: Bool = false, expected: (Double) -> Double) -> Double {
    let red = redChannel(image.cgImage)
    let size = SIMD2(Double(image.width), Double(image.height))
    var pairs: [(Double, Double)] = []
    for gy in 1..<8 {
        for gx in 1..<8 {
            let p = SIMD2(size.x * Double(gx) / 8, size.y * Double(gy) / 8)
            guard let v = map.sample(at: p, imageSize: size) else { return .infinity }
            pairs.append((Double(v), expected(red[Int(p.y) * image.width + Int(p.x)])))
        }
    }
    let k = absolute ? 1 : pairs.map { $0.0 * $0.1 }.reduce(0, +) / pairs.map { $0.1 * $0.1 }.reduce(0, +)
    return pairs.map { abs($0.0 - k * $0.1) / max(k * $0.1, 1e-6) }.max() ?? .infinity
}

do {
    let root = tmp.appendingPathComponent("coreml")
    let daDir = root.appendingPathComponent("depth/depth-anything-v2-small")
    try writeFixture(Fixtures.relative, to: daDir.appendingPathComponent("DepthAnythingV2SmallF16.mlpackage"))
    guard let daLoc = DepthModelLocator.locate(.depthAnythingV2Small, roots: [root]) else { throw CocoaError(.fileNoSuchFile) }
    let da = try CoreMLDepthEstimator(location: daLoc)
    check(da.contract.resize == .letterbox && da.contract.outputKind == .relativeInverseDepth, "Depth Anything built-in contract")

    // Upright portrait from a landscape file with EXIF orientation 6 (rotate 90° clockwise to view).
    let photo = try orientedPhoto(rawWidth: 180, rawHeight: 120, orientation: .right)
    check(photo.width == 120 && photo.height == 180, "EXIF orientation applied (upright portrait)")
    let e1 = try da.estimate(photo)
    check(e1.map.representation == .relativeInverseDepth && !e1.map.isAbsolute, "Depth Anything output labelled relative, not metric")
    check(e1.focalSource == nil && e1.focalLengthUsedPixels == nil, "relative output has no focal scaling")
    check(e1.inputSize == SIMD2(56, 42) && e1.outputSize == SIMD2(56, 42), "fixture input/output size")
    check(e1.map.width == 28 && e1.map.height == 42, "letterbox padding cropped from the output (28×42 of 56×42)")
    check(mismatch(e1.map, photo, expected: { $0 / 255 }) < 0.06, "depth aligns with the upright photo (orientation + letterbox)")
    check(e1.coldStart && e1.loadSeconds > 0, "first prediction is a cold start")
    let e2 = try da.estimate(photo)
    check(!e2.coldStart && e2.loadSeconds == 0, "second prediction is warm")
    print(String(format: "fixture timings: load %.1f ms, cold %.2f ms, warm %.2f ms",
                 da.loadSeconds * 1000, e1.inferenceSeconds * 1000, e2.inferenceSeconds * 1000))

    // The raw (un-rotated) pattern must NOT match: guards against ignoring orientation.
    let rawPhoto = try LoadedImage(cgImage: try orientedPhoto(rawWidth: 180, rawHeight: 120, orientation: .up).cgImage)
    check(mismatch(e1.map, rawPhoto, expected: { $0 / 255 }) > 0.1, "wrong orientation would be detected")

    // Stretch via a manifest.
    try Data(#"{"schemaVersion": 1, "resize": "stretch"}"#.utf8).write(to: daDir.appendingPathComponent("depth.json"))
    let stretched = try CoreMLDepthEstimator(location: DepthModelLocator.locate(.depthAnythingV2Small, roots: [root])!).estimate(photo)
    check(stretched.map.width == 56 && stretched.resize == .stretch && mismatch(stretched.map, photo, expected: { $0 / 255 }) < 0.06,
          "stretched input maps back to the photo")
    // A manifest declaring metric output is honoured (e.g. a metric fine-tune).
    try Data(#"{"schemaVersion": 1, "outputKind": "metricDepth"}"#.utf8).write(to: daDir.appendingPathComponent("depth.json"))
    check(try CoreMLDepthEstimator(location: DepthModelLocator.locate(.depthAnythingV2Small, roots: [root])!).estimate(photo).map.isAbsolute,
          "metric only when the manifest says so")
    try fm.removeItem(at: daDir.appendingPathComponent("depth.json"))

    // Depth Pro-like: canonical inverse depth + field of view.
    let proDir = root.appendingPathComponent("depth/depth-pro")
    try writeFixture(Fixtures.pro, to: proDir.appendingPathComponent("model.mlpackage"))
    let pro = try CoreMLDepthEstimator(location: DepthModelLocator.locate(.depthPro, roots: [root])!)
    let square = try LoadedImage(cgImage: photo.cgImage, focalLength35mm: 26)
    let withExif = try pro.estimate(square)
    let f = square.focalLengthPixels, W = Double(square.width)
    check(withExif.map.isAbsolute && withExif.focalSource == .exif && withExif.focalLengthUsedPixels == f, "Depth Pro + EXIF → metric")
    // canonical = 0.5·red/255 + 0.05 → depth = 1 / (canonical · W / f).
    check(mismatch(withExif.map, square, absolute: true, expected: { 1 / ((0.5 * $0 / 255 + 0.05) * W / f) }) < 0.06,
          "Depth Pro metric conversion, in metres without any rescaling")
    let predictedF = 0.5 * W / tan(30 * Double.pi / 180)
    check(close(withExif.estimatedFocalLengthPixels ?? 0, predictedF, 0.5), "predicted focal length kept alongside EXIF")
    let noExif = try pro.estimate(photo)
    check(noExif.map.isAbsolute && noExif.focalSource == .predicted && close(noExif.focalLengthUsedPixels ?? 0, predictedF, 0.5),
          "no EXIF: Depth Pro's own field of view")
    try Data(#"{"schemaVersion": 1, "fovOutput": ""}"#.utf8).write(to: proDir.appendingPathComponent("depth.json"))
    let guessed = try CoreMLDepthEstimator(location: DepthModelLocator.locate(.depthPro, roots: [root])!).estimate(photo)
    check(!guessed.map.isAbsolute && guessed.map.representation == .relativeInverseDepth && guessed.focalSource == .assumed,
          "no EXIF, no field of view: not called metric")
    try fm.removeItem(at: proDir.appendingPathComponent("depth.json"))

    // Incompatible and broken models fail with explicit errors, never a crash.
    let wrongDir = root.appendingPathComponent("wrong/depth/depth-pro")
    try writeFixture(Fixtures.relative, to: wrongDir.appendingPathComponent("model.mlpackage"))
    do {
        _ = try CoreMLDepthEstimator(location: DepthModelLocator.locate(.depthPro, roots: [root.appendingPathComponent("wrong")])!)
        check(false, "incompatible model must be rejected")
    } catch let e as MonocularDepthError {
        if case .incompatibleModel = e { passed += 1 } else { check(false, "incompatible model error kind: \(e)") }
    }
    let brokenDir = root.appendingPathComponent("broken/depth/depth-anything-v2-small/model.mlpackage")
    try touchPackage(brokenDir)
    do {
        _ = try CoreMLDepthEstimator(location: DepthModelLocator.locate(.depthAnythingV2Small, roots: [root.appendingPathComponent("broken")])!)
        check(false, "corrupt model must be rejected")
    } catch let e as MonocularDepthError {
        if case .loadFailed = e { passed += 1 } else { check(false, "corrupt model error kind: \(e)") }
    }

    // Pipeline: missing models are warnings; a broken selected model falls back; embedded metric wins.
    let pipeline = ArmaturePipeline(modelsDirectory: tmp.appendingPathComponent("no-models"))
    for b in MonocularDepthBackend.allCases { pipeline.depthModelPaths[b] = tmp.appendingPathComponent("missing-\(b.rawValue)") }
    pipeline.monocularDepthMode = .backend(.depthPro)
    pipeline.monocularDepthFallback = false
    let missing = pipeline.estimateDepth(for: photo)
    check(missing.estimate == nil && !missing.warnings.isEmpty, "missing model: warning, no depth, no throw")
    if case .none = missing.decision { passed += 1 } else { check(false, "missing model decision") }
    var threw = false
    do { _ = try pipeline.depthEstimator(.depthPro) } catch MonocularDepthError.modelNotInstalled { threw = true }
    check(threw, "depthEstimator reports modelNotInstalled")

    pipeline.depthModelPaths[.depthPro] = brokenDir
    pipeline.depthModelPaths[.depthAnythingV2Small] = daDir
    pipeline.monocularDepthFallback = true
    let fellBack = pipeline.estimateDepth(for: photo)
    check(fellBack.decision == .monocular(.depthAnythingV2Small, fallbackFrom: .depthPro) && fellBack.estimate != nil
          && fellBack.warnings.contains { $0.contains("failed to load") }, "broken selected model falls back to the other")

    pipeline.monocularDepthMode = .automatic
    let metricPhoto = try LoadedImage(cgImage: photo.cgImage, depth: DepthMap(width: 4, height: 6,
        values: [Float](repeating: 3, count: 24), representation: .metricDepth))
    let embedded = pipeline.estimateDepth(for: metricPhoto)
    check(embedded.decision == .embeddedMetric && embedded.estimate == nil, "embedded metric depth: no monocular inference")

    let synthetic = SyntheticDepthEstimator(backend: .depthPro) { image in
        MonocularDepthEstimate(backend: .depthPro, map: DepthMap(width: 2, height: 2, values: [2, 2, 2, 2], representation: .metricDepth))
    }
    pipeline.depthEstimatorOverride = synthetic
    pipeline.monocularDepthMode = .backend(.depthPro)
    let injected = pipeline.estimateDepth(for: photo)
    check(injected.estimate?.map.values == [2, 2, 2, 2] && injected.summary.source == "monocular", "synthetic backend injection")
    check(injected.summary.headline.contains("Depth Pro"), "summary headline")
}

if failures.isEmpty {
    print("Depth self-test passed: \(passed) checks (discovery, priority, sampling, resizing, tensors, scale, Core ML fixtures, errors)")
} else {
    FileHandle.standardError.write(Data("Depth self-test failed: \(failures.count) of \(passed + failures.count) checks\n".utf8))
    exit(1)
}
