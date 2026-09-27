import AppKit
import ArmatureCore
import SceneKit
import SwiftUI
import UniformTypeIdentifiers

/// A joint of one detected person.
struct JointRef: Equatable {
    let person: Int
    let joint: BodyJoint
}

@MainActor
final class StudioModel: ObservableObject {
    enum Phase: Equatable { case idle, working(String), done, failed(String) }

    @Published var phase: Phase = .idle
    @Published var sourceImage: NSImage?
    /// Detections as edited by the user (drives the skeleton overlay).
    @Published private(set) var people: [DetectedPerson] = []
    @Published private(set) var result: ArmatureResult?
    @Published private(set) var clayScene: ClayScene?
    /// Id of the body model to fit (see `availableModels`). Switching re-fits the current detections.
    @Published var bodyModel = ArmaturePipeline.defaultModelID { didSet { if bodyModel != oldValue { refitAll() } } }
    /// Converted body models found on disk.
    var availableModels: [BodyModelInfo] { pipeline?.availableModels ?? [] }

    struct ModelFamily: Identifiable { let id: String; let models: [BodyModelInfo] }
    /// Available models grouped by family (SMPL, SMPL-X, Anny), in display order.
    var modelFamilies: [ModelFamily] {
        var out: [ModelFamily] = []
        for m in availableModels {
            if let i = out.firstIndex(where: { $0.id == m.family }) {
                out[i] = ModelFamily(id: m.family, models: out[i].models + [m])
            } else {
                out.append(ModelFamily(id: m.family, models: [m]))
            }
        }
        return out
    }
    @Published var style = ClayStyle() { didSet { clayScene?.applyStyle(style) } }
    @Published var view: ClayScene.View = .photo
    @Published var showSkeleton = true
    @Published var showMasks = false
    /// Refine body shape and pose against each person's segmentation mask.
    @Published var useSilhouette = true {
        didSet {
            guard useSilhouette != oldValue else { return }
            pipeline?.useSilhouette = useSilhouette
            refitAll()
        }
    }
    /// Tinted segmentation masks for the overlay, per person.
    @Published private(set) var maskOverlays: [CGImage?] = []
    @Published var dropTargeted = false
    @Published private(set) var hovered: JointRef?
    @Published private(set) var dragging: JointRef?

    let modelsDirectory = ArmaturePipeline.defaultModelsDirectory()
    private lazy var pipeline: ArmaturePipeline? = modelsDirectory.map { dir in
        let p = ArmaturePipeline(modelsDirectory: dir)
        p.ageModel = ageModel == Self.noAgeModel ? nil : ageModel
        p.monocularDepthMode = MonocularDepthMode(argument: depthMode) ?? .automatic
        return p
    }

    /// Monocular depth for photos without LiDAR/TrueDepth depth: "none", "auto" or a backend id.
    /// Switching re-estimates depth (off the main thread) and re-fits, keeping edits.
    @Published var depthMode = "auto" {
        didSet {
            guard depthMode != oldValue, let pipeline else { return }
            pipeline.monocularDepthMode = MonocularDepthMode(argument: depthMode) ?? .automatic
            reestimateDepth()
        }
    }
    /// Depth backends whose model files were found (refreshed when a photo is opened).
    @Published private(set) var installedDepthBackends: Set<MonocularDepthBackend> = []
    private func refreshInstalledDepthBackends() {
        installedDepthBackends = Set(pipeline.map { Array($0.installedDepthModels().keys) } ?? [])
    }
    /// What depth did for the current result, for the status line (nil when there's nothing to say).
    var depthStatus: (text: String, warning: Bool)? {
        guard let depth = result?.depth else { return nil }
        let d = depth.summary
        if let w = d.warnings.first { return (w, true) }
        guard d.source == "monocular" else { return nil }
        let reports = result?.bodies.compactMap(\.monocularDepth) ?? []
        let used = reports.filter(\.accepted).count
        return ("Depth: \(d.headline) · used for \(used) of \(reports.count)", false)
    }

    static let noAgeModel = "none"
    /// Age estimator id, or `noAgeModel`. Defaults to the first installed (MiVOLO). Switching re-estimates
    /// everyone's age and re-fits (Anny's shape is conditioned on age).
    @Published var ageModel: String = {
        ArmaturePipeline.defaultModelsDirectory().flatMap { AgeModelInfo.available(in: $0).first?.id } ?? "none"
    }() {
        didSet {
            guard ageModel != oldValue, let pipeline, let result else { return }
            pipeline.ageModel = ageModel == Self.noAgeModel ? nil : ageModel
            pipeline.estimateAges(&people, image: result.image)
            refitAll()
        }
    }
    var availableAgeModels: [AgeModelInfo] { pipeline?.availableAgeModels ?? [] }
    @Published private(set) var qualityReport: ImageQualityReport?
    @Published private(set) var qualityStatus = ""
    @Published private(set) var qualityRunning = false
    private var qualityTask: Task<Void, Never>?
    private var qualityGeneration = UUID()
    private let qualityAnalyzer = ImageQualityAnalyzer(modelsDirectory: ImageQualityAnalyzer.defaultModelsDirectory())

    func analyzeQuality() {
        guard let url = currentURL, !qualityRunning else { return }
        qualityTask?.cancel()
        let generation = UUID()
        qualityGeneration = generation
        qualityRunning = true
        qualityStatus = "Analyzing image quality…"
        let analyzer = qualityAnalyzer
        qualityTask = Task.detached(priority: .utility) {
            do {
                let report = try analyzer.analyze(url: url, cancelled: { Task.isCancelled }, progress: { metric in
                    Task { @MainActor in
                        guard self.qualityGeneration == generation, self.qualityRunning else { return }
                        self.qualityStatus = "Scoring \(metric.uppercased())…"
                    }
                })
                await MainActor.run {
                    guard self.qualityGeneration == generation else { return }
                    self.qualityReport = report
                    self.qualityRunning = false
                    self.qualityStatus = ""
                }
            } catch {
                await MainActor.run {
                    guard self.qualityGeneration == generation else { return }
                    self.qualityRunning = false
                    self.qualityStatus = error is CancellationError ? "Cancelled" : error.localizedDescription
                }
            }
        }
    }

    func cancelQuality() {
        qualityTask?.cancel()
        qualityGeneration = UUID()
        qualityRunning = false
        qualityStatus = "Cancelled"
    }

    func exportQuality() {
        guard let report = qualityReport else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.commaSeparatedText, .json]
        panel.nameFieldStringValue = "quality_output.csv"
        if panel.runModal() == .OK, let url = panel.url {
            do {
                if url.pathExtension.lowercased() == "json" {
                    try ImageQualityReport.json([report]).write(to: url, options: .atomic)
                } else {
                    try ImageQualityReport.csv([report]).write(to: url, atomically: true, encoding: .utf8)
                }
            } catch { qualityStatus = error.localizedDescription }
        }
    }

    private var currentURL: URL?
    private var originalPeople: [DetectedPerson] = []
    private var dragSessionActive = false
    private var lastLiveRefit = Date.distantPast
    /// The on-screen SCNView, so image export can render from wherever the user has orbited to.
    weak var liveView: SCNView?

    var modelsMissing: Bool { modelsDirectory == nil }

    func open(_ url: URL) {
        cancelQuality()
        qualityReport = nil
        qualityStatus = ""
        currentURL = url
        sourceImage = NSImage(contentsOf: url)
        refreshInstalledDepthBackends()
        detect()
    }

    // MARK: Detection and fitting

    func detect() {
        guard let url = currentURL else { return }
        guard let pipeline else { phase = .failed("Body models are not installed. Image quality analysis is available below."); return }
        phase = .working("Finding people…")
        result = nil
        clayScene = nil
        people = []
        let bodyModel = self.bodyModel
        Task.detached(priority: .userInitiated) {
            do {
                let image = try LoadedImage(url: url)
                let result = try pipeline.run(image: image, model: bodyModel)
                await MainActor.run {
                    guard self.currentURL == url else { return }
                    self.originalPeople = result.people
                    self.install(result)
                }
            } catch {
                await MainActor.run {
                    guard self.currentURL == url else { return }
                    self.phase = .failed(error.localizedDescription)
                }
            }
        }
    }

    private func install(_ result: ArmatureResult) {
        self.result = result
        people = result.people
        maskOverlays = result.people.enumerated().map { i, p in
            guard let mask = p.silhouette, let c = style.color(forPerson: i).usingColorSpace(.sRGB) else { return nil }
            return mask.overlayImage(red: c.redComponent, green: c.greenComponent, blue: c.blueComponent, alpha: 0.45)
        }
        if result.bodies.isEmpty {
            clayScene = nil
            phase = .failed("No people found in this image.")
        } else {
            let scene = result.makeScene(style: style)
            scene.setView(view)
            clayScene = scene
            phase = .done
        }
    }

    /// Re-fits every body with the current (possibly edited) detections, e.g. after a body-model change.
    /// Reuses the result's cached depth estimate.
    private func refitAll() {
        guard let pipeline, let result else { return }
        do {
            install(try pipeline.fit(people: people, image: result.image, model: bodyModel, horizonAngle: result.horizonAngle,
                                     depth: result.depth))
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// Runs the (newly selected) depth backend on the current photo in the background, then re-fits.
    private func reestimateDepth() {
        guard let pipeline, let result else { return }
        let (image, people, bodyModel, horizon, url) = (result.image, self.people, self.bodyModel, result.horizonAngle, currentURL)
        phase = .working("Estimating depth…")
        Task.detached(priority: .userInitiated) {
            let depth = pipeline.estimateDepth(for: image)
            do {
                let refit = try pipeline.fit(people: people, image: image, model: bodyModel, horizonAngle: horizon, depth: depth)
                await MainActor.run {
                    guard self.currentURL == url else { return }
                    self.install(refit)
                }
            } catch {
                await MainActor.run {
                    guard self.currentURL == url else { return }
                    self.phase = .failed(error.localizedDescription)
                }
            }
        }
    }

    /// `live` skips the silhouette stage (~100 ms) so dragging stays smooth; it runs on release.
    private func refit(_ index: Int, live: Bool = false) {
        guard let pipeline, let result else { return }
        do {
            let updated = try pipeline.refit(result, person: index, with: people[index], model: bodyModel, silhouette: !live)
            self.result = updated
            clayScene?.updateBody(index, mesh: updated.meshes[index], faces: updated.faces)
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// Short description of a fitted body: height, and apparent age for models that have one (Anny).
    func summary(of body: FittedBody) -> String? {
        guard let m = try? pipeline?.model(body.model) else { return nil }
        let h = String(format: "%.2f m", m.height(betas: body.betas))
        if let age = m.phenotype(betas: body.betas)?["ageYears"] { return h + String(format: " · ~%.0f y", max(age, 1)) }
        return h
    }

    // MARK: Skeleton editing

    /// Nearest joint to `point` (image pixels) within `radius` pixels.
    func joint(near point: CGPoint, radius: Double) -> JointRef? {
        var best: (JointRef, Double)?
        for (i, p) in people.enumerated() {
            for j in BodyJoint.allCases where p.isVisible(j) {
                let d = simd_distance(p.joints2D[j.rawValue], SIMD2(point.x, point.y))
                if d <= radius, d < (best?.1 ?? .infinity) { best = (JointRef(person: i, joint: j), d) }
            }
        }
        return best?.0
    }

    func hover(_ point: CGPoint?, radius: Double) {
        let h = point.flatMap { joint(near: $0, radius: radius) }
        if h != hovered { hovered = h }
    }

    func drag(from start: CGPoint, to point: CGPoint, radius: Double) {
        if !dragSessionActive {
            dragSessionActive = true
            dragging = joint(near: start, radius: radius)
        }
        guard let d = dragging, let result else { return }
        let clamped = SIMD2(min(max(point.x, 0), Double(result.image.width)),
                            min(max(point.y, 0), Double(result.image.height)))
        people[d.person].move(d.joint, to: clamped)
        // A fit takes ~10 ms; refit live, but no more often than the display can use it.
        if Date().timeIntervalSince(lastLiveRefit) > 1.0 / 30 {
            lastLiveRefit = Date()
            refit(d.person, live: true)
        }
    }

    func endDrag() {
        if let d = dragging { refit(d.person) }
        dragging = nil
        dragSessionActive = false
    }

    func swapLeftRight(_ index: Int) {
        people[index].swapLeftRight()
        refit(index)
    }

    /// Sets (or, with nil, clears) a person's age. Overrides the estimate; re-fits.
    func setAge(_ index: Int, _ years: Double?) {
        guard people.indices.contains(index) else { return }
        people[index].ageOverride = years.map { min(max($0, 0), 100) }
        refit(index)
    }

    func resetEdits(_ index: Int) {
        let (estimate, given) = (people[index].estimatedAge, people[index].ageOverride)
        people[index] = originalPeople[index]
        people[index].estimatedAge = estimate
        people[index].ageOverride = given
        refit(index)
    }

    func remove(_ index: Int) {
        guard var r = result else { return }
        r.remove(person: index)
        originalPeople.remove(at: index)
        install(r)
    }

    // MARK: Export

    func exportImage() {
        guard let scene = clayScene else { return }
        let options = ImageExportOptions()
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "clay_\(view.rawValue)"
        options.attach(to: panel)
        guard panel.runModal() == .OK, let url = panel.url else { return }

        // Match what's on screen: the photo's aspect in photo view, the viewport's aspect in studio view.
        let aspect: CGFloat
        if view == .photo {
            aspect = scene.imageSize.width / scene.imageSize.height
        } else if let v = liveView, v.bounds.height > 0 {
            aspect = v.bounds.width / v.bounds.height
        } else {
            aspect = 1
        }
        let long = options.longEdge(current: max(liveView?.bounds.width ?? 1600, liveView?.bounds.height ?? 1600)
                                        * (liveView?.window?.backingScaleFactor ?? 2))
        let size = aspect >= 1 ? CGSize(width: long, height: (long / aspect).rounded())
                               : CGSize(width: (long * aspect).rounded(), height: long)
        let pov = liveView?.pointOfView
        guard let image = scene.render(view, size: size, background: options.background, pointOfView: pov) else {
            phase = .failed("Rendering failed.")
            return
        }
        do {
            try ArmatureExport.writeImage(image, to: url, type: options.format)
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func exportUSDZ() {
        guard let scene = clayScene else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.usdz]
        panel.nameFieldStringValue = "clay.usdz"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        if !scene.exportUSDZ(to: url) { phase = .failed("USDZ export failed.") }
        scene.setView(view)
    }

    func exportOBJ() {
        guard let result else { return }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Export Here"
        panel.message = "Choose a folder for the OBJ meshes and SMPL parameters"
        guard panel.runModal() == .OK, let dir = panel.url else { return }
        do {
            for (i, mesh) in result.meshes.enumerated() {
                try ArmatureExport.obj(vertices: mesh, faces: result.faces)
                    .write(to: dir.appendingPathComponent("person_\(i).obj"), atomically: true, encoding: .utf8)
            }
            try ArmatureExport.parametersJSON(result, pipeline: pipeline).write(to: dir.appendingPathComponent("body_params.json"))
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }
}
