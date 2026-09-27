import ArmatureCore
import Foundation

let usage = """
usage: armature <image> [options]
       armature quality <image-or-folder> [-o scores.csv|scores.json] [--models Models]
                        [--metrics musiq,hyperiqa,nima,brisque,clipiqa,niqe,arniqa,liqe]

Detects people in <image>, fits SMPL bodies and writes clay renders + meshes.

options:
  -o, --out <dir>        output directory (default: ./out/<image name>)
  -b, --body <id>        body model (default: smpl_neutral); see --list-models
  -g, --gender <g>       shorthand for SMPL: neutral | male | female
      --list-models      list the converted body and age models and exit
      --age <a>          age in years, for everyone (e.g. 34) or per person (e.g. 0=34,1=6); conditions
                         Anny's body shape and overrides estimated ages
      --age-model <m>    age estimator: mivolo | faceage | none (default: first installed)
  -m, --material <m>     clay | smooth | plastic | ceramic | marble | bronze | chrome | wood | wireframe
                         (default: clay)
  -p, --palette <p>      plasticine | terracotta | stone (default: plasticine)
  -s, --subdivision <n>  GPU subdivision level 0-3 (default: 2)
      --size <px>        long edge of the rendered images (default: 2048)
      --transparent      render images without the photo/backdrop (PNG with alpha, shadows kept)
      --focal-mm <mm>    35 mm-equivalent lens focal length (default: EXIF, else 50)
      --models <dir>     converted model directory (default: ./Models or $ARMATURE_MODELS)
      --no-silhouette    skip fitting body shape to the person segmentation mask
      --depth-backend <b> monocular depth when the photo has no LiDAR/TrueDepth depth:
                         none | auto | depth-anything-v2-small | depth-pro (default: auto — the first
                         installed; nothing if none is). A missing model is a warning, not an error.
      --no-monocular-depth  same as --depth-backend none
      --depth-model <path>  Core ML depth model (.mlpackage/.mlmodelc or its folder) for --depth-backend
      --no-depth-fallback   don't substitute another installed backend for a missing one
      --no-usdz          skip USDZ export
"""

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
    exit(1)
}

var args = Array(CommandLine.arguments.dropFirst())
if args.isEmpty || args.contains("-h") || args.contains("--help") { print(usage); exit(args.isEmpty ? 1 : 0) }
if args.first == "quality" {
    do { try runQualityCommand(Array(args.dropFirst())); exit(0) }
    catch { fail(error.localizedDescription) }
}
if args.contains("--list-models") {
    guard let dir = ArmaturePipeline.defaultModelsDirectory() else { print("no converted models found"); exit(1) }
    print("body models:")
    for m in BodyModelInfo.available(in: dir) {
        print("  \(m.id.padding(toLength: 16, withPad: " ", startingAt: 0)) \(m.displayName)  [\(m.licence)]")
    }
    print("age models:")
    for m in AgeModelInfo.available(in: dir) {
        print("  \(m.id.padding(toLength: 16, withPad: " ", startingAt: 0)) \(m.displayName)  [\(m.licence)]")
    }
    print("depth models:")
    let depthModels = ArmaturePipeline(modelsDirectory: dir).installedDepthModels()
    for b in MonocularDepthBackend.allCases {
        let where_ = depthModels[b].map { $0.modelURL.path } ?? "not installed (Models/depth/\(b.rawValue)/)"
        print("  \(b.rawValue.padding(toLength: 24, withPad: " ", startingAt: 0)) \(b.displayName): \(where_)")
    }
    exit(0)
}

var inputPath: String?
var outDir: String?
var bodyModel = ArmaturePipeline.defaultModelID
var ages = ArmaturePipeline.AgeInput.none
var ageModel: String??  // nil: default; .some(nil): none
var style = ClayStyle()
var modelsPath: String?
var writeUSDZ = true
var useSilhouette = true
var renderSize = 2048.0
var focalMM: Double?
var background = ClayScene.Background.scene
var depthMode = MonocularDepthMode.automatic
var depthModelPath: String?
var depthFallback = true

while !args.isEmpty {
    let a = args.removeFirst()
    func value() -> String {
        if args.isEmpty { fail("\(a) needs a value") }
        return args.removeFirst()
    }
    switch a {
    case "-o", "--out": outDir = value()
    case "-b", "--body": bodyModel = value()
    case "--age":
        let v = value()
        if let a = Double(v) {
            ages = .everyone(a)
        } else {
            var map: [Int: Double] = [:]
            for part in v.split(separator: ",") {
                let kv = part.split(separator: "=")
                guard kv.count == 2, let i = Int(kv[0]), let a = Double(kv[1]) else { fail("bad --age (use 34 or 0=34,1=6)") }
                map[i] = a
            }
            ages = .perPerson(map)
        }
    case "--age-model":
        let v = value()
        ageModel = .some(v == "none" ? nil : v)
    case "-g", "--gender":
        let g = value()
        guard ["neutral", "male", "female"].contains(g) else { fail("unknown gender") }
        bodyModel = "smpl_\(g)"
    case "-p", "--palette":
        guard let p = ClayStyle.Palette(rawValue: value()) else { fail("unknown palette") }
        style.palette = p
    case "-s", "--subdivision":
        guard let n = Int(value()) else { fail("bad subdivision level") }
        style.subdivision = min(max(n, 0), 3)
    case "-m", "--material":
        guard let f = ClayStyle.Finish(rawValue: value()) else { fail("unknown material") }
        style.finish = f
    case "--smooth": style.finish = .smooth
    case "--size":
        guard let n = Double(value()), n >= 64, n <= 8192 else { fail("bad size (64-8192)") }
        renderSize = n
    case "--transparent": background = .transparent
    case "--focal-mm":
        guard let f = Double(value()), f > 5, f < 1000 else { fail("bad focal length") }
        focalMM = f
    case "--models": modelsPath = value()
    case "--no-usdz": writeUSDZ = false
    case "--no-silhouette": useSilhouette = false
    case "--depth-backend":
        let v = value()
        guard let m = MonocularDepthMode(argument: v) else {
            fail("unknown depth backend '\(v)' (none | auto | \(MonocularDepthBackend.allCases.map(\.rawValue).joined(separator: " | ")))")
        }
        depthMode = m
    case "--no-monocular-depth": depthMode = .disabled
    case "--depth-model": depthModelPath = value()
    case "--no-depth-fallback": depthFallback = false
    default:
        if a.hasPrefix("-") { fail("unknown option \(a)") }
        inputPath = a
    }
}

guard let inputPath else { fail("no input image") }
let inputURL = URL(fileURLWithPath: inputPath)
let modelsURL = modelsPath.map { URL(fileURLWithPath: $0) } ?? ArmaturePipeline.defaultModelsDirectory()
guard let modelsURL else {
    fail("converted SMPL models not found. Run: uv run --with numpy --with scipy tools/convert_models.py")
}
let out = URL(fileURLWithPath: outDir ?? "out/\(inputURL.deletingPathExtension().lastPathComponent)")

do {
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let t0 = Date()
    let image = try LoadedImage(url: inputURL, focalLength35mm: focalMM)
    let pipeline = ArmaturePipeline(modelsDirectory: modelsURL)
    pipeline.useSilhouette = useSilhouette
    pipeline.ageModel = ageModel ?? pipeline.availableAgeModels.first?.id
    if let id = pipeline.ageModel, !pipeline.availableAgeModels.contains(where: { $0.id == id }) {
        fail("unknown age model '\(id)' (see --list-models)")
    }
    pipeline.detector.segmentPeople = useSilhouette
    pipeline.monocularDepthMode = depthMode
    pipeline.monocularDepthFallback = depthFallback
    if let depthModelPath {
        guard case .backend(let b) = depthMode else { fail("--depth-model needs --depth-backend depth-anything-v2-small or depth-pro") }
        pipeline.depthModelPaths[b] = URL(fileURLWithPath: depthModelPath)
    }
    let result = try pipeline.run(image: image, model: bodyModel, ages: ages)
    let model = try pipeline.model(bodyModel)
    print("body model: \(model.info.displayName)")

    print("image \(image.width)×\(image.height), focal \(Int(image.focalLengthPixels))px\(image.focalFromEXIF ? " (EXIF)" : " (assumed)")")
    if let d = image.depth {
        print("depth map: \(d.width)×\(d.height), \(d.isAbsolute ? "metric" : "relative (no scale, not used)")")
    }
    if let depth = result.depth {
        let d = depth.summary
        // Quiet when nothing was asked for and nothing ran (e.g. auto with no depth model installed).
        let asked: Bool
        if case .backend = depthMode { asked = true } else { asked = false }
        if d.source == "monocular" || asked || !d.warnings.isEmpty { print("monocular depth: \(d.headline)") }
        if let c = depth.calibration {
            var line = "  scale: \(c.method.rawValue)"
            if c.usable {
                line += c.scaleEstimated ? " (estimated from the body-size prior)" : " (from the model)"
                if let a = c.metricAgreement { line += String(format: ", agrees with body prior to ×%.2f", a) }
                if !c.constrainsDistance { line += ", relative depth within each body only" }
            } else {
                line += " (depth not used for fitting)"
            }
            print(line)
            for n in c.notes { print("  note: \(n)") }
        }
        if let e = depth.estimate {
            var line = String(format: "  valid %.0f%%, model output %d×%d", e.map.validFraction * 100, e.outputSize.x, e.outputSize.y)
            if let f = e.estimatedFocalLengthPixels { line += String(format: ", model focal %.0f px", f) }
            if let f = e.focalLengthUsedPixels, let s = e.focalSource { line += String(format: ", scaled with %.0f px (%@)", f, s.rawValue) }
            if let c = e.confidenceSummary { line += String(format: ", confidence %.2f%@", c.mean, e.confidenceIsDerived ? " (derived)" : "") }
            if e.coldStart { line += String(format: ", cold start: load %.0f ms", e.loadSeconds * 1000) }
            print(line)
        }
        for w in d.warnings { print("warning: \(w)") }
    }
    if let h = result.horizonAngle { print(String(format: "horizon: tilted %.1f° (counter-clockwise +)", -h * 180 / .pi)) }
    print("found \(result.bodies.count) \(result.bodies.count == 1 ? "person" : "people")")
    for (i, b) in result.bodies.enumerated() {
        var line = String(format: "  person %d: depth %.2fm, 3D fit %.1fcm rms, reprojection %.1fpx rms",
                          i, b.translation.z, b.rms3D * 100, b.rms2D)
        if b.usedDepth {
            line += String(format: "\n    depth: distance and size from the depth map; height %.2f m", model.height(betas: b.betas))
        }
        if let m = b.monocularDepth {
            if m.accepted {
                line += String(format: "\n    monocular depth: used (%d samples%@), depth disagreement %.1f → %.1f cm",
                               m.samples, m.usedAbsoluteDistance ? ", distance" : ", relative only",
                               (m.depthRMSBefore ?? 0) * 100, (m.depthRMSAfter ?? 0) * 100)
            } else {
                line += "\n    monocular depth: not used — \(m.reason)"
            }
        }
        let person = result.people[i]
        if let given = person.ageOverride {
            line += String(format: "\n    age %.0f (given)", given)
        } else if let e = person.estimatedAge {
            line += String(format: "\n    age ~%.0f (estimated by %@)", e.years, e.model)
        }
        if let p = model.phenotype(betas: b.betas), let age = p["ageYears"] {
            line += String(format: "\n    body looks about %.0f years old, %.2f m tall%@", max(age, 1), model.height(betas: b.betas),
                           model.supportsAge && person.targetAge != nil ? " (shape conditioned on age)" : "")
        }
        if let before = b.silhouetteBefore, let after = b.silhouette {
            line += String(format: "\n    silhouette: body outside mask %.0f%% → %.0f%%, IoU %.2f → %.2f",
                           before.outside * 100, after.outside * 100, before.iou, after.iou)
        } else if useSilhouette {
            line += ", no segmentation mask"
        }
        print(line)
    }
    if result.bodies.isEmpty { fail("no people detected") }

    for (i, mesh) in result.meshes.enumerated() {
        try ArmatureExport.obj(vertices: mesh, faces: result.faces)
            .write(to: out.appendingPathComponent("person_\(i).obj"), atomically: true, encoding: .utf8)
    }
    try ArmatureExport.parametersJSON(result, pipeline: pipeline).write(to: out.appendingPathComponent("body_params.json"))

    let t1 = Date()
    let scene = result.makeScene(style: style)
    for view in ClayScene.View.allCases {
        guard let img = scene.render(view, size: scene.defaultSize(view, maxDimension: renderSize), background: background) else {
            fail("rendering failed")
        }
        try ArmatureExport.writePNG(img, to: out.appendingPathComponent("clay_\(view.rawValue).png"))
    }
    if writeUSDZ && !scene.exportUSDZ(to: out.appendingPathComponent("clay.usdz")) {
        print("warning: USDZ export failed")
    }
    let renderTime = Date().timeIntervalSince(t1)

    for (label, t) in result.timings { print(String(format: "  %-18@ %6.0f ms", label, t * 1000)) }
    print(String(format: "  %-18@ %6.0f ms", "render + export", renderTime * 1000))
    print(String(format: "done in %.2fs → %@", Date().timeIntervalSince(t0), out.path))
} catch {
    fail(error.localizedDescription)
}
