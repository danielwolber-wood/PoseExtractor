import ArmatureCore
import Foundation

// A/B evaluation of the fitter's anatomical priors on real photos (no ground truth needed):
//   armature-eval <image folder> [--model <id>] [--limit <n>] [--out <dir>]
// Detects people once per photo, fits them with and without the anatomical priors (range-of-motion
// limits, self-collision, flipped-limb hypotheses) on the same detections, and compares how many fits
// are anatomically impossible, how well they match the photo (2D keypoints, silhouette), and fit time.
// With --out, writes before/after renders of every person whose fit changed from impossible to possible.

var folder: URL?
var modelID = ArmaturePipeline.defaultModelID
var limit = Int.max
var outDir: URL?
var argv = Array(CommandLine.arguments.dropFirst())
while !argv.isEmpty {
    let a = argv.removeFirst()
    switch a {
    case "--model" where !argv.isEmpty: modelID = argv.removeFirst()
    case "--limit" where !argv.isEmpty: limit = Int(argv.removeFirst()) ?? limit
    case "--out" where !argv.isEmpty: outDir = URL(fileURLWithPath: argv.removeFirst())
    default: folder = URL(fileURLWithPath: a)
    }
}
guard let folder, let modelsURL = ArmaturePipeline.defaultModelsDirectory() else {
    print("usage: armature-eval <image folder> [--model <id>] [--limit <n>] [--out <dir>]"); exit(1)
}
if let outDir { try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true) }

let pipeline = ArmaturePipeline(modelsDirectory: modelsURL)
pipeline.monocularDepthMode = .disabled
pipeline.ageModel = nil
let model = try pipeline.model(modelID)
let images = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
    .filter { ["jpg", "jpeg", "png", "heic"].contains($0.pathExtension.lowercased()) }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }.prefix(limit)

/// "Gross" errors: clearly impossible, not borderline.
let grossDegrees = 15.0, grossCentimetres = 3.0
struct Side {
    var gross = 0, limitCases = 0, collisionCases = 0
    var rms2D: [Double] = [], iou: [Double] = [], seconds = 0.0
    var joints: [String: Int] = [:], pairs: [String: Int] = [:]
}
var before = Side(), after = Side()
var people = 0, fixed = 0, broke = 0
var flips: [String: Int] = [:]

func record(_ b: FittedBody, into s: inout Side) -> Bool {
    let p = model.plausibility(pose: b.pose, betas: b.betas, minDegrees: grossDegrees, minCentimetres: grossCentimetres)
    s.rms2D.append(b.rms2D)
    if let o = b.silhouette { s.iou.append(o.iou) }
    for k in p.beyondLimits.keys { s.joints[k, default: 0] += 1 }
    for k in p.penetrations.keys { s.pairs[k, default: 0] += 1 }
    if !p.beyondLimits.isEmpty { s.limitCases += 1 }
    if !p.penetrations.isEmpty { s.collisionCases += 1 }
    if !p.isClean { s.gross += 1 }
    return !p.isClean
}

for url in images {
    guard let image = try? LoadedImage(url: url), let detected = try? pipeline.detector.detect(in: image),
          !detected.isEmpty else { continue }
    pipeline.anatomicalPriors = false
    var t = Date()
    guard let a = try? pipeline.fit(people: detected, image: image, model: modelID) else { continue }
    before.seconds += Date().timeIntervalSince(t)
    pipeline.anatomicalPriors = true
    t = Date()
    guard let b = try? pipeline.fit(people: detected, image: image, model: modelID) else { continue }
    after.seconds += Date().timeIntervalSince(t)
    for i in a.bodies.indices {
        people += 1
        let wasBad = record(a.bodies[i], into: &before), isBad = record(b.bodies[i], into: &after)
        for l in b.bodies[i].plausibility?.flippedLimbs ?? [] { flips[l, default: 0] += 1 }
        if wasBad && !isBad { fixed += 1 }
        if !wasBad && isBad { broke += 1 }
        let pa = model.plausibility(pose: a.bodies[i].pose, betas: a.bodies[i].betas, minDegrees: grossDegrees, minCentimetres: grossCentimetres)
        if wasBad || isBad {
            let pb = model.plausibility(pose: b.bodies[i].pose, betas: b.bodies[i].betas, minDegrees: grossDegrees, minCentimetres: grossCentimetres)
            func show(_ p: PlausibilityReport) -> String {
                (p.beyondLimits.sorted { $0.key < $1.key }.map { String(format: "%@ +%.0f°", $0.key, $0.value) }
                 + p.penetrations.sorted { $0.key < $1.key }.map { String(format: "%@ %.0f cm", $0.key, $0.value) }).joined(separator: ", ")
            }
            print(String(format: "%@ #%d  before: [%@] 2D %.1f px  →  after: [%@] 2D %.1f px%@", url.lastPathComponent, i, show(pa),
                         a.bodies[i].rms2D, show(pb), b.bodies[i].rms2D,
                         (b.bodies[i].plausibility?.flippedLimbs.isEmpty == false ? " flipped \(b.bodies[i].plausibility!.flippedLimbs)" : "")))
        }
        if let outDir, wasBad != isBad {
            let stem = url.deletingPathExtension().lastPathComponent + "_\(i)"
            for (label, r) in [("before", a), ("after", b)] {
                let scene = ClayScene(bodies: [r.bodies[i]], meshes: [r.meshes[i]], faces: model.faces, image: image, style: ClayStyle())
                // The studio view shows depth errors (a leg bent backwards looks fine from the camera).
                for view in ClayScene.View.allCases {
                    if let img = scene.render(view) {
                        try? ArmatureExport.writePNG(img, to: outDir.appendingPathComponent("\(stem)_\(label)_\(view).png"))
                    }
                }
            }
        }
    }
}

func median(_ v: [Double]) -> Double { v.isEmpty ? .nan : v.sorted()[v.count / 2] }
func mean(_ v: [Double]) -> Double { v.isEmpty ? .nan : v.reduce(0, +) / Double(v.count) }
func pct(_ n: Int) -> String { String(format: "%d (%.1f%%)", n, 100 * Double(n) / Double(max(people, 1))) }
print("\n\(model.info.displayName): \(people) people in \(images.count) photos")
print("                          without priors      with priors")
print("impossible poses          \(pct(before.gross).padding(toLength: 20, withPad: " ", startingAt: 0))\(pct(after.gross))")
print("  past range of motion    \(pct(before.limitCases).padding(toLength: 20, withPad: " ", startingAt: 0))\(pct(after.limitCases))")
print("  body parts intersecting \(pct(before.collisionCases).padding(toLength: 20, withPad: " ", startingAt: 0))\(pct(after.collisionCases))")
print(String(format: "2D keypoint error        median %.1f / mean %.1f px   median %.1f / mean %.1f px",
             median(before.rms2D), mean(before.rms2D), median(after.rms2D), mean(after.rms2D)))
print(String(format: "silhouette IoU           mean %.3f            mean %.3f", mean(before.iou), mean(after.iou)))
print(String(format: "fit time per person      %.0f ms               %.0f ms", 1000 * before.seconds / Double(max(people, 1)),
             1000 * after.seconds / Double(max(people, 1))))
print("fixed \(fixed), newly impossible \(broke)")
print("joints past range (before): \(before.joints.sorted { $0.value > $1.value }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))")
print("parts intersecting (before): \(before.pairs.sorted { $0.value > $1.value }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))")
if !after.joints.isEmpty || !after.pairs.isEmpty {
    print("left after: \((after.joints.sorted { $0.value > $1.value } + after.pairs.sorted { $0.value > $1.value }).map { "\($0.key) \($0.value)" }.joined(separator: ", "))")
}
if !flips.isEmpty { print("flipped limbs: \(flips.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))") }
