import AVFoundation
import ClayCore
import CoreGraphics
import Foundation
import ImageIO
import simd
import UniformTypeIdentifiers

// Round-trip regression test for any body model:
//   clay-selftest [--model <id>] [--pose raise|reach|walk] [--age <years>] [--out <dir>]
// --age (models with an age-aware shape space, i.e. Anny): the truth body is that age's average shape,
// and the fit is compared with and without being told the age.
// Poses the model in a known asymmetric pose with a heavier-than-average build, renders it as clay,
// runs the whole pipeline on the render, and reports how well pose, shape, placement, the silhouette
// stage, keypoint editing and metric depth recover the truth.

var modelID = ClayPipeline.defaultModelID
var poseName = "raise"
var truthAge: Double?
var outDir = URL(fileURLWithPath: "out/selftest")
var argv = Array(CommandLine.arguments.dropFirst())
while !argv.isEmpty {
    let a = argv.removeFirst()
    if a == "--model", !argv.isEmpty { modelID = argv.removeFirst() }
    if a == "--pose", !argv.isEmpty { poseName = argv.removeFirst() }
    if a == "--age", !argv.isEmpty { truthAge = Double(argv.removeFirst()) }
    if a == "--out", !argv.isEmpty { outDir = URL(fileURLWithPath: argv.removeFirst()) }
}
guard let modelsURL = ClayPipeline.defaultModelsDirectory() else { print("models not found"); exit(1) }
outDir.appendPathComponent(modelID + (poseName == "raise" ? "" : "_" + poseName) + (truthAge.map { "_age\(Int($0))" } ?? ""))
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

let pipeline = ClayPipeline(modelsDirectory: modelsURL)
let model = try pipeline.model(modelID)
print("model: \(model.info.displayName) — \(model.jointCount) joints, \(model.vertexCount) vertices, \(model.betaCount) shape coefficients")

func joint(_ name: String) -> Int {
    guard let j = model.joint(name) else { fatalError("model has no \(name)") }
    return j
}

// MARK: Ground truth

var pose = model.defaultPose
// Upright in camera space (180° about x), then turned 20° about the vertical.
let orient = simd_quatd(angle: 0.35, axis: SIMD3(0, 1, 0)) * simd_quatd(angle: .pi, axis: SIMD3(1, 0, 0))
pose[joint("pelvis")] = orient.axis * orient.angle
let rest = model.restJoints(betas: [])
/// Local rotation pointing a limb (from `j` to `child`) in a model-frame direction (parents are at rest).
func aim(_ j: Int, _ child: Int, _ dir: SIMD3<Double>) {
    let q = simd_quatd(from: simd_normalize(rest[child] - rest[j]), to: simd_normalize(dir))
    pose[j] = q.axis * q.angle
}
let midSpine = model.chain("spine")[model.chain("spine").count / 2]
func bend(_ name: String, _ angle: Double) { pose[joint(name)] = model.flexAxis(joint: joint(name))! * angle }
switch poseName {
case "reach":  // both arms reaching forward, torso twisted, knees slightly bent, head tilted
    aim(joint("leftShoulder"), joint("leftElbow"), SIMD3(0.3, 0.1, 0.95))
    aim(joint("rightShoulder"), joint("rightElbow"), SIMD3(-0.2, -0.2, 0.96))
    bend("leftElbow", 0.5); bend("rightElbow", 0.9)
    bend("leftKnee", 0.35); bend("rightKnee", 0.35)
    pose[midSpine] = SIMD3(0.15, 0.3, 0)
    pose[joint("head")] = SIMD3(0.2, 0, 0.15)
case "walk":  // mid-stride, arms swinging, head turned
    aim(joint("leftHip"), joint("leftKnee"), SIMD3(0, -0.9, 0.44))
    aim(joint("rightHip"), joint("rightKnee"), SIMD3(0, -0.94, -0.34))
    bend("rightKnee", 0.6)
    aim(joint("leftShoulder"), joint("leftElbow"), SIMD3(0.25, -0.9, -0.35))
    aim(joint("rightShoulder"), joint("rightElbow"), SIMD3(-0.25, -0.85, 0.46))
    bend("leftElbow", 0.3); bend("rightElbow", 0.8)
    pose[joint("head")] = SIMD3(0, 0.45, 0)
default:  // "raise": left arm raised, right elbow bent, right thigh lifted
    aim(joint("leftShoulder"), joint("leftElbow"), SIMD3(0.6, 0.8, 0))
    aim(joint("rightShoulder"), joint("rightElbow"), SIMD3(-0.35, -0.94, 0))
    aim(joint("rightHip"), joint("rightKnee"), SIMD3(0, -0.83, 0.56))
    bend("rightElbow", 1.4); bend("rightKnee", 1.0)
    pose[midSpine] = SIMD3(0.1, 0, 0.08)
}
// A noticeably heavier-than-average build, so the silhouette stage has something to recover.
var betas = [Double](repeating: 0, count: model.betaCount)
betas[0] = 0.5; betas[1] = 2.0
if let a = truthAge {
    guard let prior = model.shapePrior(ageYears: a) else { print("\(modelID) has no age-aware shape space"); exit(1) }
    betas = prior.mean
}
let truth = FittedBody(model: modelID, pose: pose, betas: betas, translation: SIMD3(0, 0.1, 4.2))
let mesh = model.vertices(pose: pose, betas: betas, translation: truth.translation)
if let p = model.phenotype(betas: betas) {
    print(String(format: "truth body: %.2f m, ~%.0f years", model.height(betas: betas), p["ageYears"] ?? 0))
}

// Render on a plain backdrop.
let W = 1024, H = 1536
let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: 0,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
ctx.setFillColor(CGColor(srgbRed: 0.55, green: 0.6, blue: 0.62, alpha: 1))
ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))
let backdrop = try LoadedImage(cgImage: ctx.makeImage()!)
var style = ClayStyle(); style.palette = .terracotta
let gtScene = ClayScene(bodies: [truth], meshes: [mesh], faces: model.faces, image: backdrop, style: style)
guard let render = gtScene.render(.photo) else { print("render failed"); exit(1) }
try ClayExport.writePNG(render, to: outDir.appendingPathComponent("input_synthetic.png"))

// MARK: Recover

let image = try LoadedImage(cgImage: render)
let result = try pipeline.run(image: image, model: modelID)
guard let fit = result.bodies.first else { print("no person detected"); exit(1) }
let f = image.focalLengthPixels, c = SIMD2(Double(W) / 2, Double(H) / 2)

let gtK = model.kinematics(pose: pose, betas: betas)
let evalJoints = ["leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle", "leftShoulder", "rightShoulder",
                  "leftElbow", "rightElbow", "leftWrist", "rightWrist", "head"].map(joint) + [model.chain("neck")[0]]
/// Mean per-joint position error, root-relative (pose + proportions, independent of placement).
func mpjpe(_ b: FittedBody) -> Double {
    let k = model.kinematics(pose: b.pose, betas: b.betas)
    let r = joint("pelvis")
    return evalJoints.map { simd_distance(gtK.joints[$0] - gtK.joints[r], k.joints[$0] - k.joints[r]) }.reduce(0, +)
        / Double(evalJoints.count)
}
/// Mean posed-vertex error, pelvis-relative: pose and shape together, and sensitive to surface folds
/// that joint error can't see (e.g. a spine creased by opposite bends that cancel out at the joints).
func surfaceError(_ b: FittedBody) -> Double {
    let truthV = model.vertices(pose: pose, betas: betas), fitV = model.vertices(pose: b.pose, betas: b.betas)
    let tr = gtK.joints[joint("pelvis")], fr = model.kinematics(pose: b.pose, betas: b.betas).joints[joint("pelvis")]
    return zip(truthV, fitV).map { simd_distance(SIMD3<Double>($0) - tr, SIMD3<Double>($1) - fr) }.reduce(0, +) / Double(truthV.count)
}
/// Rest-pose vertex RMS: body shape only.
func shapeError(_ b: FittedBody) -> Double {
    let zero = [SIMD3<Double>](repeating: .zero, count: model.jointCount)
    let a = model.vertices(pose: zero, betas: betas), g = model.vertices(pose: zero, betas: b.betas)
    return sqrt(zip(a, g).map { Double(simd_distance_squared($0, $1)) }.reduce(0, +) / Double(a.count))
}

let found = BodyJoint.allCases.filter { $0.is2DOnly && result.people[0].confidence2D[$0.rawValue] > 0 }
print("keypoints: \(found.filter(\.isFace).count) face, \(found.filter(\.isHand).count) hand")
print(String(format: "fit residuals: 3D %.1f cm rms vs Vision, 2D %.1f px rms", fit.rms3D * 100, fit.rms2D))

let fitK = model.kinematics(pose: fit.pose, betas: fit.betas)
// Handedness: the fitted left wrist should be nearer the true left wrist than the true right one.
let lwFit = fitK.joints[joint("leftWrist")] - fitK.joints[joint("pelvis")]
let handOK = simd_distance(lwFit, gtK.joints[joint("leftWrist")] - gtK.joints[joint("pelvis")])
    < simd_distance(lwFit, gtK.joints[joint("rightWrist")] - gtK.joints[joint("pelvis")])
print("handedness OK: \(handOK)")

pipeline.useSilhouette = false
let noSil = try pipeline.fit(people: result.people, image: image, model: modelID).bodies[0]
pipeline.useSilhouette = true
print(String(format: "MPJPE        : without silhouette %.1f cm, with %.1f cm", mpjpe(noSil) * 100, mpjpe(fit) * 100))
print(String(format: "shape error  : without silhouette %.1f cm, with %.1f cm", shapeError(noSil) * 100, shapeError(fit) * 100))
print(String(format: "surface error: without silhouette %.1f cm, with %.1f cm", surfaceError(noSil) * 100, surfaceError(fit) * 100))
if let b = fit.silhouetteBefore, let a = fit.silhouette {
    print(String(format: "silhouette   : body outside mask %.1f%% → %.1f%%, IoU %.3f → %.3f", b.outside * 100, a.outside * 100, b.iou, a.iou))
}

// Editing: drag the right wrist 150 px up and to the side; the fitted wrist should follow.
var edited = result.people[0]
let target = edited.joints2D[BodyJoint.rightWrist.rawValue] + SIMD2(-150, -150)
edited.move(.rightWrist, to: target)
let t0 = Date()
let refitted = try pipeline.refit(result, person: 0, with: edited, model: modelID).bodies[0]
let rk = model.kinematics(pose: refitted.pose, betas: refitted.betas)
let rw = rk.joints[joint("rightWrist")] + refitted.translation
print(String(format: "editing      : wrist lands %.1f px from the dragged target (full refit %.0f ms)",
             simd_distance(SIMD2(f * rw.x / rw.z, f * rw.y / rw.z) + c, target), Date().timeIntervalSince(t0) * 1000))
// The live path used while dragging: warm-started, no silhouette.
let t1 = Date()
let live = try pipeline.refit(result, person: 0, with: edited, model: modelID, silhouette: false).bodies[0]
let lk = model.kinematics(pose: live.pose, betas: live.betas)
let lw = lk.joints[joint("rightWrist")] + live.translation
print(String(format: "live editing : wrist lands %.1f px from the target (%.0f ms per update)",
             simd_distance(SIMD2(f * lw.x / lw.z, f * lw.y / lw.z) + c, target), Date().timeIntervalSince(t1) * 1000))

// MARK: Age conditioning

if let a = truthAge {
    var told = result.people
    for i in told.indices { told[i].ageOverride = a }
    let withAge = try pipeline.fit(people: told, image: image, model: modelID).bodies[0]
    for (label, b) in [("age not given", fit), ("age given    ", withAge)] {
        print(String(format: "%@: height %.2f m (truth %.2f), body reads %.0f years (truth %.0f), shape error %.1f cm, MPJPE %.1f cm",
                     label, model.height(betas: b.betas), model.height(betas: betas), model.phenotype(betas: b.betas)?["ageYears"] ?? 0,
                     a, shapeError(b) * 100, mpjpe(b) * 100))
    }
}

// MARK: Depth round trip

// Write the render as a HEIC with a metric depth map (as LiDAR photos carry), load it back through the
// normal path, and check that distance and body size improve.
func syntheticDepth(width dw: Int, height dh: Int) -> [Float] {
    var z = [Float](repeating: 6, count: dw * dh)   // back wall at 6 m
    let s = Double(dw) / Double(W)
    let p = mesh.map { v -> SIMD3<Double> in
        let q = SIMD3<Double>(v)
        return SIMD3((f * q.x / q.z + c.x) * s, (f * q.y / q.z + c.y) * s, q.z)
    }
    for t in stride(from: 0, to: model.faces.count, by: 3) {
        let a = p[Int(model.faces[t])], b = p[Int(model.faces[t + 1])], e = p[Int(model.faces[t + 2])]
        let area = (b.x - a.x) * (e.y - a.y) - (b.y - a.y) * (e.x - a.x)
        if abs(area) < 1e-12 { continue }
        for y in max(Int(min(a.y, b.y, e.y)), 0)...min(Int(max(a.y, b.y, e.y)) + 1, dh - 1) {
            for x in max(Int(min(a.x, b.x, e.x)), 0)...min(Int(max(a.x, b.x, e.x)) + 1, dw - 1) {
                let px = Double(x) + 0.5, py = Double(y) + 0.5
                let w0 = ((b.x - px) * (e.y - py) - (b.y - py) * (e.x - px)) / area
                let w1 = ((e.x - px) * (a.y - py) - (e.y - py) * (a.x - px)) / area
                let w2 = 1 - w0 - w1
                if w0 < 0 || w1 < 0 || w2 < 0 { continue }
                z[y * dw + x] = min(z[y * dw + x], Float(w0 * a.z + w1 * b.z + w2 * e.z))
            }
        }
    }
    return z
}

let (dw, dh) = (256, 384)
var depthValues = syntheticDepth(width: dw, height: dh)
let meta = CGImageMetadataCreateMutable()
CGImageMetadataRegisterNamespaceForPrefix(meta, "http://ns.apple.com/depthData/1.0/" as CFString, "depthData" as CFString, nil)
for (k, v) in [("Accuracy", "absolute"), ("Filtered", "true"), ("Quality", "high")] {
    CGImageMetadataSetValueWithPath(meta, nil, "depthData:\(k)" as CFString, v as CFString)
}
let depthInfo: [AnyHashable: Any] = [
    kCGImageAuxiliaryDataInfoData: Data(bytes: &depthValues, count: depthValues.count * 4),
    kCGImageAuxiliaryDataInfoDataDescription: ["Width": dw, "Height": dh, "BytesPerRow": dw * 4,
                                               "PixelFormat": kCVPixelFormatType_DepthFloat32],
    kCGImageAuxiliaryDataInfoMetadata: meta,
]
let heicURL = outDir.appendingPathComponent("input_synthetic_depth.heic")
if let avDepth = try? AVDepthData(fromDictionaryRepresentation: depthInfo),
   let dest = CGImageDestinationCreateWithURL(heicURL as CFURL, UTType.heic.identifier as CFString, 1, nil) {
    var auxType: NSString?
    let dict = avDepth.dictionaryRepresentation(forAuxiliaryDataType: &auxType)
    CGImageDestinationAddImage(dest, render, nil)
    CGImageDestinationAddAuxiliaryDataInfo(dest, (auxType ?? kCGImageAuxiliaryDataTypeDepth) as CFString, dict as CFDictionary? ?? [:] as CFDictionary)
    _ = CGImageDestinationFinalize(dest)
}
let depthImage = try LoadedImage(url: heicURL)
let withDepth = try pipeline.run(image: depthImage, model: modelID).bodies[0]
let trueZ = gtK.joints[joint("pelvis")].z + truth.translation.z
for (label, b) in [("without depth", fit), ("with depth   ", withDepth)] {
    let k = model.kinematics(pose: b.pose, betas: b.betas)
    print(String(format: "%@: pelvis distance %.2f m (truth %.2f), height %.2f m (truth %.2f)", label,
                 k.joints[joint("pelvis")].z + b.translation.z, trueZ, model.height(betas: b.betas), model.height(betas: betas)))
}

let scene = result.makeScene(style: ClayStyle())
if let img = scene.render(.photo) { try ClayExport.writePNG(img, to: outDir.appendingPathComponent("recovered_photo.png")) }
if let img = scene.render(.studio) { try ClayExport.writePNG(img, to: outDir.appendingPathComponent("recovered_studio.png")) }
