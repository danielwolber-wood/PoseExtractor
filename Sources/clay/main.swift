import ClayCore
import Foundation

let usage = """
usage: clay <image> [options]

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
      --models <dir>     converted model directory (default: ./Models or $CLAY_MODELS)
      --no-silhouette    skip fitting body shape to the person segmentation mask
      --no-usdz          skip USDZ export
"""

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
    exit(1)
}

var args = Array(CommandLine.arguments.dropFirst())
if args.isEmpty || args.contains("-h") || args.contains("--help") { print(usage); exit(args.isEmpty ? 1 : 0) }
if args.contains("--list-models") {
    guard let dir = ClayPipeline.defaultModelsDirectory() else { print("no converted models found"); exit(1) }
    print("body models:")
    for m in BodyModelInfo.available(in: dir) {
        print("  \(m.id.padding(toLength: 16, withPad: " ", startingAt: 0)) \(m.displayName)  [\(m.licence)]")
    }
    print("age models:")
    for m in AgeModelInfo.available(in: dir) {
        print("  \(m.id.padding(toLength: 16, withPad: " ", startingAt: 0)) \(m.displayName)  [\(m.licence)]")
    }
    exit(0)
}

var inputPath: String?
var outDir: String?
var bodyModel = ClayPipeline.defaultModelID
var ages = ClayPipeline.AgeInput.none
var ageModel: String??  // nil: default; .some(nil): none
var style = ClayStyle()
var modelsPath: String?
var writeUSDZ = true
var useSilhouette = true
var renderSize = 2048.0
var focalMM: Double?
var background = ClayScene.Background.scene

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
    default:
        if a.hasPrefix("-") { fail("unknown option \(a)") }
        inputPath = a
    }
}

guard let inputPath else { fail("no input image") }
let inputURL = URL(fileURLWithPath: inputPath)
let modelsURL = modelsPath.map { URL(fileURLWithPath: $0) } ?? ClayPipeline.defaultModelsDirectory()
guard let modelsURL else {
    fail("converted SMPL models not found. Run: uv run --with numpy --with scipy tools/convert_models.py")
}
let out = URL(fileURLWithPath: outDir ?? "out/\(inputURL.deletingPathExtension().lastPathComponent)")

do {
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let t0 = Date()
    let image = try LoadedImage(url: inputURL, focalLength35mm: focalMM)
    let pipeline = ClayPipeline(modelsDirectory: modelsURL)
    pipeline.useSilhouette = useSilhouette
    pipeline.ageModel = ageModel ?? pipeline.availableAgeModels.first?.id
    if let id = pipeline.ageModel, !pipeline.availableAgeModels.contains(where: { $0.id == id }) {
        fail("unknown age model '\(id)' (see --list-models)")
    }
    pipeline.detector.segmentPeople = useSilhouette
    let result = try pipeline.run(image: image, model: bodyModel, ages: ages)
    let model = try pipeline.model(bodyModel)
    print("body model: \(model.info.displayName)")

    print("image \(image.width)×\(image.height), focal \(Int(image.focalLengthPixels))px\(image.focalFromEXIF ? " (EXIF)" : " (assumed)")")
    if let d = image.depth {
        print("depth map: \(d.width)×\(d.height), \(d.isAbsolute ? "metric" : "relative (no scale, not used)")")
    }
    if let h = result.horizonAngle { print(String(format: "horizon: tilted %.1f° (counter-clockwise +)", -h * 180 / .pi)) }
    print("found \(result.bodies.count) \(result.bodies.count == 1 ? "person" : "people")")
    for (i, b) in result.bodies.enumerated() {
        var line = String(format: "  person %d: depth %.2fm, 3D fit %.1fcm rms, reprojection %.1fpx rms",
                          i, b.translation.z, b.rms3D * 100, b.rms2D)
        if b.usedDepth {
            line += String(format: "\n    depth: distance and size from the depth map; height %.2f m", model.height(betas: b.betas))
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
        try ClayExport.obj(vertices: mesh, faces: result.faces)
            .write(to: out.appendingPathComponent("person_\(i).obj"), atomically: true, encoding: .utf8)
    }
    try ClayExport.parametersJSON(result, pipeline: pipeline).write(to: out.appendingPathComponent("body_params.json"))

    let t1 = Date()
    let scene = result.makeScene(style: style)
    for view in ClayScene.View.allCases {
        guard let img = scene.render(view, size: scene.defaultSize(view, maxDimension: renderSize), background: background) else {
            fail("rendering failed")
        }
        try ClayExport.writePNG(img, to: out.appendingPathComponent("clay_\(view.rawValue).png"))
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
