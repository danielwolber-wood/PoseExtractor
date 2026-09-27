import Foundation

/// The depth side of one pipeline run: where depth came from, the (cached) monocular estimate, its
/// calibration and any warnings. Kept on `ClayResult`, so re-fits reuse it instead of re-running a model.
public struct DepthReport: @unchecked Sendable {
    public var requested: MonocularDepthMode
    public var decision: DepthSourceDecision
    public var estimate: MonocularDepthEstimate?
    public var calibration: DepthCalibration?
    /// E.g. a requested backend that isn't installed. Never fatal.
    public var warnings: [String]
    /// Wall-clock time of the depth stage (model load + inference + conversion).
    public var seconds: TimeInterval

    public var summary: DepthSummary { DepthSummary(self) }
}

/// A `DepthReport` as plain values, for JSON and display.
public struct DepthSummary: Codable, Sendable {
    /// "embedded-metric", "monocular" or "none".
    public var source: String
    public var requestedMode: String
    public var backend: String?
    public var fallbackFrom: String?
    public var reason: String?
    public var modelPath: String?
    public var inputSize: [Int]?
    public var outputSize: [Int]?
    public var resize: String?
    public var representation: String?
    public var scaleEstimated: Bool?
    public var calibration: DepthCalibration?
    public var focalLengthPredictedPixels: Double?
    public var focalLengthUsedPixels: Double?
    public var focalSource: String?
    public var validFraction: Double?
    public var confidenceMean: Double?
    public var confidenceAboveHalf: Double?
    public var confidenceIsDerived: Bool?
    public var loadMilliseconds: Double?
    public var inferenceMilliseconds: Double?
    public var processingMilliseconds: Double?
    public var coldStart: Bool?
    public var warnings: [String]
    public var notes: [String]

    init(_ r: DepthReport) {
        requestedMode = r.requested.description
        warnings = r.warnings
        switch r.decision {
        case .embeddedMetric: source = "embedded-metric"
        case .monocular(let b, let from):
            source = "monocular"
            backend = b.rawValue
            fallbackFrom = from?.rawValue
        case .none(let why):
            source = "none"
            reason = why
        }
        calibration = r.calibration
        notes = r.estimate?.notes ?? []
        guard let e = r.estimate else { return }
        modelPath = e.modelPath
        inputSize = [e.inputSize.x, e.inputSize.y]
        outputSize = [e.outputSize.x, e.outputSize.y]
        resize = e.resize.rawValue
        representation = e.map.representation.rawValue
        scaleEstimated = !e.map.representation.isMetric
        focalLengthPredictedPixels = e.estimatedFocalLengthPixels
        focalLengthUsedPixels = e.focalLengthUsedPixels
        focalSource = e.focalSource?.rawValue
        validFraction = e.map.validFraction
        if let c = e.confidenceSummary { confidenceMean = c.mean; confidenceAboveHalf = c.fractionAboveHalf }
        confidenceIsDerived = e.confidenceIsDerived
        loadMilliseconds = e.loadSeconds * 1000
        inferenceMilliseconds = e.inferenceSeconds * 1000
        processingMilliseconds = e.processingSeconds * 1000
        coldStart = e.coldStart
    }

    /// One line for the CLI / app, e.g. "Depth Anything V2 Small · relative inverse depth · 518×392 · 41 ms".
    public var headline: String {
        switch source {
        case "embedded-metric": return "embedded metric depth (LiDAR/TrueDepth)"
        case "monocular":
            var parts = [MonocularDepthBackend(rawValue: backend ?? "")?.displayName ?? backend ?? "?"]
            if let r = representation {
                parts.append(r == DepthRepresentation.metricDepth.rawValue ? "metric (\(focalSource ?? "?") focal length)"
                             : r == DepthRepresentation.relativeInverseDepth.rawValue ? "relative inverse depth" : "relative depth")
            }
            if let i = inputSize { parts.append("\(i[0])×\(i[1])") }
            if let ms = inferenceMilliseconds { parts.append(String(format: "%.0f ms%@", ms + (processingMilliseconds ?? 0),
                                                                    coldStart == true ? " (cold)" : "")) }
            if let c = calibration { parts.append("scale: \(c.method.rawValue)") }
            return parts.joined(separator: " · ")
        default: return "none (\(reason ?? "not requested"))"
        }
    }
}

extension ClayPipeline {
    /// Depth backends whose model files are present (nothing is loaded).
    public func installedDepthModels() -> [MonocularDepthBackend: DepthModelLocation] {
        let roots = DepthModelLocator.searchRoots(modelsDirectory: modelsDirectory)
        var out: [MonocularDepthBackend: DepthModelLocation] = [:]
        for b in MonocularDepthBackend.allCases {
            if let url = depthModelPaths[b] {
                if let loc = DepthModelLocator.explicit(b, url: url) { out[b] = loc }
            } else if let loc = DepthModelLocator.locate(b, roots: roots) {
                out[b] = loc
            }
        }
        return out
    }

    /// The (lazily loaded, cached) estimator for a backend. Only the requested backend is loaded.
    public func depthEstimator(_ backend: MonocularDepthBackend) throws -> any MonocularDepthEstimating {
        if let o = depthEstimatorOverride, o.backend == backend { return o }
        depthLock.lock(); defer { depthLock.unlock() }
        if let e = depthEstimators[backend] { return e }
        guard let location = installedDepthModels()[backend] else {
            let searched = depthModelPaths[backend].map { [$0.path] }
                ?? DepthModelLocator.searchRoots(modelsDirectory: modelsDirectory).map { $0.appendingPathComponent("depth").path }
            throw MonocularDepthError.modelNotInstalled(backend, searched: searched)
        }
        let e = try CoreMLDepthEstimator(location: location)
        depthEstimators[backend] = e
        return e
    }

    /// Frees loaded depth models (Depth Pro holds about a gigabyte).
    public func unloadDepthModels() {
        depthLock.lock(); defer { depthLock.unlock() }
        depthEstimators.removeAll()
    }

    /// Decides the depth source for a photo and, when that's a monocular backend, runs it once.
    /// Never throws: a missing, incompatible or failing model becomes a warning and no depth.
    public func estimateDepth(for image: LoadedImage) -> DepthReport {
        let t0 = Date()
        var warnings: [String] = []
        if let d = image.depth, !d.isAbsolute {
            warnings.append("the photo's embedded depth is relative (Portrait disparity); it isn't used")
        }
        var installed = Set(installedDepthModels().keys)
        if let o = depthEstimatorOverride { installed.insert(o.backend) }
        var decision = DepthSourceDecision.resolve(embedded: image.depth, mode: monocularDepthMode,
                                                   fallback: monocularDepthFallback, installed: installed)
        if case .backend(let b) = monocularDepthMode, !installed.contains(b), decision != .embeddedMetric {
            warnings.append("\(b.displayName) is not installed (expected in Models/depth/\(b.rawValue)/; see docs/depth.md)")
            if case .monocular(let other, _) = decision { warnings.append("using \(other.displayName) instead") }
        }
        var estimate: MonocularDepthEstimate?
        // Try the chosen backend; if it fails to load or run, fall back (when allowed) to another one.
        var tried: Set<MonocularDepthBackend> = []
        while case .monocular(let b, let from) = decision, !tried.contains(b) {
            tried.insert(b)
            do {
                estimate = try depthEstimator(b).estimate(image)
                break
            } catch {
                warnings.append(error.localizedDescription)
                let others = MonocularDepthBackend.automaticOrder.filter { !tried.contains($0) && installed.contains($0) }
                let allowed = monocularDepthMode == .automatic || monocularDepthFallback
                if allowed, let next = others.first {
                    decision = .monocular(next, fallbackFrom: from ?? b)
                } else {
                    decision = .none(reason: "\(b.displayName) failed")
                }
            }
        }
        return DepthReport(requested: monocularDepthMode, decision: decision, estimate: estimate, calibration: nil,
                           warnings: warnings, seconds: Date().timeIntervalSince(t0))
    }

    /// Calibrates the monocular estimate against the fits made without it, then re-fits everyone with
    /// it, keeping each depth-enhanced fit only where it's no worse. Runs people in parallel.
    func applyMonocularDepth(_ depth: inout DepthReport, bodies: inout [FittedBody], people: [DetectedPerson],
                             image: LoadedImage, fitter: BodyFitter) {
        guard case .monocular(let backend, _) = depth.decision, let estimate = depth.estimate,
              bodies.count == people.count, !people.isEmpty else { return }
        let anchors = zip(bodies, people).map { fitter.depthAnchor(for: $0, person: $1, image: image, estimate: estimate) }
        let calibration = DepthCalibration.estimate(representation: estimate.map.representation, anchors: anchors)
        depth.calibration = calibration
        guard calibration.usable else {
            for i in bodies.indices {
                bodies[i].monocularDepth = .skipped(backend, "depth calibration unusable: \(calibration.notes.first ?? "")")
            }
            return
        }
        var out = bodies
        let useSilhouette = self.useSilhouette
        // Not `depthLock`: that one is held while a model loads (tens of seconds on first use), and this
        // runs on the main thread when the app re-fits.
        let resultLock = NSLock()
        DispatchQueue.concurrentPerform(iterations: people.count) { i in
            let refined: FittedBody
            if let cue = fitter.monocularCue(for: people[i], index: i, image: image, estimate: estimate, calibration: calibration,
                                             body: bodies[i]) {
                refined = fitter.refine(bodies[i], person: people[i], image: image, cue: cue, useSilhouette: useSilhouette)
            } else {
                var b = bodies[i]
                b.monocularDepth = .skipped(backend, "too few reliable depth samples on this person")
                refined = b
            }
            resultLock.lock(); out[i] = refined; resultLock.unlock()
        }
        bodies = out
    }

    /// The cached estimate and calibration as depth evidence for one (possibly edited) person.
    func cachedDepthCue(_ result: ClayResult, index: Int, person: DetectedPerson, fitter: BodyFitter, body: FittedBody)
        -> (cue: MonocularDepthCue?, skipped: MonocularDepthFitReport?) {
        guard let depth = result.depth, case .monocular(let backend, _) = depth.decision, let estimate = depth.estimate,
              let calibration = depth.calibration else { return (nil, nil) }
        if let cue = fitter.monocularCue(for: person, index: index, image: result.image, estimate: estimate, calibration: calibration,
                                         body: body) {
            return (cue, nil)
        }
        return (nil, .skipped(backend, calibration.usable ? "too few reliable depth samples on this person" : "depth calibration unusable"))
    }
}
