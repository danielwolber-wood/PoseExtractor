import Foundation
import simd

/// One person's monocular depth evidence, ready for the fitter: calibrated depths (metres) of the
/// body's front surface at some of its keypoints.
///
/// It enters the fit as soft, robust residuals:
/// - **relative** (always): each keypoint's depth relative to the torso, `(z_i − z_torso) − (d_i − d_torso)`,
///   which says which limbs are in front without saying how far away the person is;
/// - **absolute** (only when the calibration can observe distance): the torso's distance, `z_torso − d_torso`,
///   as a fraction of distance — much softer than the LiDAR term.
/// Both use a Cauchy loss, so samples that land on hair, loose clothing or background barely pull.
public struct MonocularDepthCue: Sendable {
    struct Sample: Sendable {
        /// Index into `BodyFitter.targets`.
        let target: Int
        let joint: BodyJoint
        /// Calibrated depth of the body's front surface (metres).
        let depth: Double
        let weight: Double
        /// Joint-to-front-surface distance: depth maps see skin and clothes, the fitter moves joints.
        let offset: Double
        /// Torso points define the reference depth.
        let reference: Bool
    }

    let backend: MonocularDepthBackend
    let samples: [Sample]
    let referenceDepth: Double
    let absolute: Bool
    /// Let body shape (and so size) change: only with metric depth that agrees with the body prior.
    let freeShape: Bool
    /// Absolute term: 1σ as a fraction of the distance.
    let sigmaAbsolute: Double
    /// Relative terms: 1σ in metres.
    let sigmaRelative: Double
    /// Weighted correlation between the map's depth ordering of the limbs (relative to the torso) and
    /// Vision's 3D pose's; nil when either has too little depth variation to say. Vision's 3D is an
    /// independent (if rough) witness: a map that systematically contradicts it is likely wrong.
    let visionCorrelation: Double?

    public var sampleCount: Int { samples.count }

    private var referenceWeight: Double { samples.filter(\.reference).map(\.weight).reduce(0, +) }

    /// Weighted mean front-surface depth of the body's torso samples.
    func bodyReference(_ pts: [SIMD3<Double>], tz: Double) -> Double {
        var s = 0.0
        for p in samples where p.reference { s += p.weight * (pts[p.target].z + tz - p.offset) }
        return s / max(referenceWeight, 1e-9)
    }

    /// Cauchy-robust residual: ½r² = ½·log(1 + e²). Linear near zero, flat for outliers.
    @inline(__always) static func robust(_ e: Double) -> Double {
        (e < 0 ? -1 : 1) * log1p(e * e).squareRoot()
    }

    func residuals(_ pts: [SIMD3<Double>], tz: Double, into r: inout [Double]) {
        let ref = bodyReference(pts, tz: tz)
        if absolute { r.append(Self.robust((ref - referenceDepth) / (sigmaAbsolute * referenceDepth))) }
        for p in samples {
            let e = ((pts[p.target].z + tz - p.offset - ref) - (p.depth - referenceDepth)) / sigmaRelative
            r.append(p.weight.squareRoot() * Self.robust(e))
        }
    }

    /// Weighted RMS (metres) of the relative depth disagreement — for reporting and acceptance.
    func relativeRMS(_ pts: [SIMD3<Double>], tz: Double) -> Double {
        let ref = bodyReference(pts, tz: tz)
        var s = 0.0, w = 0.0
        for p in samples {
            let e = (pts[p.target].z + tz - p.offset - ref) - (p.depth - referenceDepth)
            s += p.weight * e * e; w += p.weight
        }
        return (s / max(w, 1e-9)).squareRoot()
    }

    /// The robust objective alone (sum of squared residuals).
    func cost(_ pts: [SIMD3<Double>], tz: Double) -> Double {
        var r: [Double] = []
        residuals(pts, tz: tz, into: &r)
        return r.reduce(0) { $0 + $1 * $1 }
    }
}

/// What monocular depth did to one person's fit.
public struct MonocularDepthFitReport: Codable, Sendable {
    public var backend: String
    public var samples: Int
    public var usedAbsoluteDistance: Bool
    public var freedShape: Bool
    /// The depth-enhanced fit replaced the fit without depth.
    public var accepted: Bool
    /// "accepted", or why the depth-enhanced fit was rejected (or never attempted).
    public var reason: String
    /// Relative depth disagreement (weighted RMS, metres) without and with depth.
    public var depthRMSBefore: Double?
    public var depthRMSAfter: Double?
    public var rms2DBefore: Double?
    public var rms2DAfter: Double?
    public var distanceBefore: Double?
    public var distanceAfter: Double?
    /// Correlation of the map's limb depth ordering with Vision's 3D pose (see `MonocularDepthCue`).
    public var visionCorrelation: Double?

    static func skipped(_ backend: MonocularDepthBackend, _ reason: String) -> MonocularDepthFitReport {
        MonocularDepthFitReport(backend: backend.rawValue, samples: 0, usedAbsoluteDistance: false, freedShape: false,
                                accepted: false, reason: reason)
    }
}

extension BodyFitter {
    /// Keypoints sampled for depth: a fallback joint-to-front-surface offset (used only when the body's
    /// own surface can't be ray-cast), whether it's a torso reference point, and how far inside the mask
    /// (fraction of the usual margin) the sample must be.
    static let depthKeypoints: [(joint: BodyJoint, offset: Double, reference: Bool, margin: Double)] = [
        (.leftShoulder, 0.08, true, 1), (.rightShoulder, 0.08, true, 1),
        (.leftHip, 0.08, true, 1), (.rightHip, 0.08, true, 1), (.root, 0.08, true, 1),
        (.leftElbow, 0.04, false, 0.7), (.rightElbow, 0.04, false, 0.7),
        (.leftWrist, 0.03, false, 0.5), (.rightWrist, 0.03, false, 0.5),
        (.leftKnee, 0.05, false, 0.7), (.rightKnee, 0.05, false, 0.7),
        (.leftAnkle, 0.04, false, 0.5), (.rightAnkle, 0.04, false, 0.5),
        // The nose, not the head top: hair makes the top of the head unreliable in depth.
        (.nose, 0.0, false, 1),
    ]

    struct RawDepthSample {
        let target: Int
        let joint: BodyJoint
        let value: Double
        let weight: Double
        let offset: Double
        let reference: Bool
    }

    /// Robust samples of the raw map at the person's keypoints. Skips keypoints that are guessed,
    /// user-edited (their dragged position may sit on background), outside the photo, too close to
    /// the mask's edge (depth bleeds across occlusion boundaries), on a depth edge, or — judging by
    /// `body` — hidden behind another part of the body.
    ///
    /// The map sees the body's front surface; the fitter moves joints. Each sample's joint-to-surface
    /// offset comes from `body`'s own mesh, ray-cast through the joint's projection, so it fits the
    /// model's rig, the build and the pose (constant offsets mis-placed Anny's limbs on the self-test).
    func rawDepthSamples(_ person: DetectedPerson, image: LoadedImage, estimate: MonocularDepthEstimate,
                         body: FittedBody?) -> [RawDepthSample] {
        let map = estimate.map
        let imageSize = SIMD2(Double(image.width), Double(image.height))
        let seenY = BodyJoint.allCases.filter { person.confidence2D[$0.rawValue] > 0.2 }.map { person.joints2D[$0.rawValue].y }
        let personPx = max((seenY.max() ?? 0) - (seenY.min() ?? 0), 20)
        let margin = max(0.015 * personPx, 3)
        // A window of about 1% of the person's height (at least 3×3 map pixels).
        let radius = max(1, Int((0.01 * personPx / imageSize.y * Double(map.height)).rounded()))
        let offsets = body.flatMap { surfaceOffsets($0, image: image) } ?? [:]
        return Self.depthKeypoints.compactMap { k in
            let j = k.joint
            guard let i = targets.firstIndex(where: { $0.joint == j }), !person.edited[j.rawValue],
                  person.confidence2D[j.rawValue] > 0.3 else { return nil }
            let p = person.joints2D[j.rawValue]
            guard p.x >= 0, p.y >= 0, p.x < imageSize.x, p.y < imageSize.y else { return nil }
            if let mask = person.silhouette {
                let d = margin * k.margin
                for o in [SIMD2(0.0, 0), SIMD2(d, 0), SIMD2(-d, 0), SIMD2(0, d), SIMD2(0, -d)] where !mask.contains(p + o) {
                    return nil
                }
            }
            guard let stats = map.windowStatistics(at: p, imageSize: imageSize, radius: radius),
                  stats.count * 2 > (2 * radius + 1) * (2 * radius + 1), stats.median > 0 else { return nil }
            let spread = stats.mad / stats.median
            guard spread < 0.1 else { return nil }
            let w = person.confidence2D[j.rawValue] * estimate.confidence(at: p, imageSize: imageSize)
                / (1 + (spread / 0.03) * (spread / 0.03))
            guard w > 0.05 else { return nil }
            var offset = k.offset
            if let o = offsets[i] {
                // More than 25 cm of body in front of the joint: something (an arm) occludes it.
                guard o > -0.02, o < 0.25 else { return nil }
                offset = max(o, 0)
            }
            return RawDepthSample(target: i, joint: j, value: stats.median, weight: min(w, 1), offset: offset, reference: k.reference)
        }
    }

    /// Distance from each depth keypoint's joint to the body's first surface along the camera ray through
    /// it (target index → metres; missing where the ray misses the mesh). A z-buffer of the posed mesh,
    /// evaluated only at the ~14 joint projections.
    func surfaceOffsets(_ body: FittedBody, image: LoadedImage) -> [Int: Double]? {
        guard body.model == model.info.id, body.pose.count == model.jointCount else { return nil }
        let f = image.focalLengthPixels, c = SIMD2(Double(image.width) / 2, Double(image.height) / 2)
        let pts = points(of: body)
        let wanted = Self.depthKeypoints.compactMap { k in targets.firstIndex { $0.joint == k.joint } }
        let rays = wanted.compactMap { i -> (index: Int, pixel: SIMD2<Double>, z: Double)? in
            let p = pts[i]
            guard p.z > 0.1 else { return nil }
            return (i, SIMD2(f * p.x / p.z, f * p.y / p.z) + c, p.z)
        }
        guard !rays.isEmpty else { return nil }
        let verts = model.vertices(pose: body.pose, betas: body.betas, translation: body.translation)
        let proj = verts.map { v -> SIMD3<Double> in
            let z = max(Double(v.z), 0.05)
            return SIMD3(f * Double(v.x) / z + c.x, f * Double(v.y) / z + c.y, z)
        }
        var nearest = [Double](repeating: .infinity, count: rays.count)
        let faces = model.faces
        for t in stride(from: 0, to: faces.count, by: 3) {
            let a = proj[Int(faces[t])], b = proj[Int(faces[t + 1])], e = proj[Int(faces[t + 2])]
            let area = (b.x - a.x) * (e.y - a.y) - (b.y - a.y) * (e.x - a.x)
            guard abs(area) > 1e-9 else { continue }
            let x0 = min(a.x, b.x, e.x), x1 = max(a.x, b.x, e.x), y0 = min(a.y, b.y, e.y), y1 = max(a.y, b.y, e.y)
            for (k, r) in rays.enumerated() where r.pixel.x >= x0 && r.pixel.x <= x1 && r.pixel.y >= y0 && r.pixel.y <= y1 {
                let px = r.pixel.x, py = r.pixel.y
                let w0 = ((b.x - px) * (e.y - py) - (b.y - py) * (e.x - px)) / area
                let w1 = ((e.x - px) * (a.y - py) - (e.y - py) * (a.x - px)) / area
                let w2 = 1 - w0 - w1
                guard w0 >= 0, w1 >= 0, w2 >= 0 else { continue }
                nearest[k] = min(nearest[k], w0 * a.z + w1 * b.z + w2 * e.z)
            }
        }
        var out: [Int: Double] = [:]
        for (k, r) in rays.enumerated() where nearest[k].isFinite { out[r.index] = r.z - nearest[k] }
        return out
    }

    /// Posed target points of a fitted body (camera space, translation included).
    func points(of body: FittedBody) -> [SIMD3<Double>] {
        let k = model.kinematics(pose: body.pose, betas: body.betas)
        return targets.map { model.point($0.target, k, betas: body.betas) + body.translation }
    }

    /// Calibration evidence from a fit made without monocular depth: raw values at the torso paired
    /// with the body's front-surface depth there.
    public func depthAnchor(for body: FittedBody, person: DetectedPerson, image: LoadedImage,
                            estimate: MonocularDepthEstimate) -> DepthAnchor? {
        guard body.model == model.info.id else { return nil }
        let pts = points(of: body)
        let torso = rawDepthSamples(person, image: image, estimate: estimate, body: body).filter(\.reference)
        guard torso.count >= 2 else { return nil }
        return DepthAnchor(values: torso.map(\.value), bodyDepths: torso.map { pts[$0.target].z - $0.offset })
    }

    /// The fitter's depth evidence for person `index`, or nil when there isn't enough of it.
    /// - Parameter body: a current fit of the person, whose surface gives the joint-to-surface offsets.
    public func monocularCue(for person: DetectedPerson, index: Int, image: LoadedImage, estimate: MonocularDepthEstimate,
                             calibration: DepthCalibration, body: FittedBody?) -> MonocularDepthCue? {
        guard calibration.usable else { return nil }
        var samples = rawDepthSamples(person, image: image, estimate: estimate, body: body).compactMap { s -> MonocularDepthCue.Sample? in
            guard let d = calibration.metres(s.value, person: index), d > 0.2, d < 100 else { return nil }
            return .init(target: s.target, joint: s.joint, depth: d, weight: s.weight, offset: s.offset, reference: s.reference)
        }
        // The torso reference: a weighted median, then torso samples far from it (hair, a gap to the
        // background) are dropped, so one bad sample can't shift every relative residual.
        var ref = samples.filter(\.reference)
        guard ref.count >= 2, let median = Self.weightedMedian(ref.map(\.depth), weights: ref.map(\.weight)) else { return nil }
        samples = samples.filter { !$0.reference || abs($0.depth - median) < 0.25 }
        ref = samples.filter(\.reference)
        guard ref.count >= 2 else { return nil }
        let refDepth = ref.map { $0.weight * $0.depth }.reduce(0, +) / max(ref.map(\.weight).reduce(0, +), 1e-9)
        // A keypoint more than a metre in front of or behind the torso has hit something else.
        samples = samples.filter { abs($0.depth - refDepth) < 1.0 }
        let absolute = calibration.constrainsDistance
        guard samples.filter(\.reference).count >= 2, absolute || samples.contains(where: { !$0.reference }) else { return nil }
        let metric = calibration.method == .metric
        // Limb depth relative to the torso, from the map and from Vision's 3D joints.
        let visionZ = { (s: MonocularDepthCue.Sample) in person.joints3D[s.joint.rawValue].z }
        let refSamples = samples.filter(\.reference)
        let visionRef = refSamples.map { $0.weight * visionZ($0) }.reduce(0, +) / max(refSamples.map(\.weight).reduce(0, +), 1e-9)
        let limbs = samples.filter { !$0.reference && !$0.joint.is2DOnly && person.confidence3D[$0.joint.rawValue] > 0.1 }
        let visionCorrelation = Self.weightedCorrelation(limbs.map { $0.depth - refDepth }, limbs.map { visionZ($0) - visionRef },
                                                         weights: limbs.map(\.weight))
        return MonocularDepthCue(
            backend: estimate.backend, samples: samples, referenceDepth: refDepth, absolute: absolute,
            freeShape: metric && absolute,
            // Depth Pro's metric error is ~10% of distance (AbsRel on zero-shot benchmarks) — no better than
            // the adult body-size prior — and a scale shared across people is weaker still. A tighter 6% let a
            // 6%-biased map pull a correctly placed body away on the self-test.
            sigmaAbsolute: metric ? 0.10 : 0.12,
            // Monocular relative depth inside a body is good to a few centimetres up close, worse far away.
            sigmaRelative: 0.04 + 0.015 * refDepth,
            visionCorrelation: visionCorrelation)
    }

    static func weightedMedian(_ v: [Double], weights w: [Double]) -> Double? {
        let pairs = zip(v, w).filter { $0.1 > 0 }.sorted { $0.0 < $1.0 }
        let total = pairs.map(\.1).reduce(0, +)
        guard total > 0 else { return nil }
        var acc = 0.0
        for (value, weight) in pairs {
            acc += weight
            if acc >= total / 2 { return value }
        }
        return pairs.last?.0
    }

    /// Weighted Pearson correlation; nil with fewer than 3 points or when either side spans < 10 cm.
    static func weightedCorrelation(_ a: [Double], _ b: [Double], weights w: [Double]) -> Double? {
        guard a.count >= 3, let aMin = a.min(), let aMax = a.max(), let bMin = b.min(), let bMax = b.max(),
              aMax - aMin > 0.1, bMax - bMin > 0.1 else { return nil }
        let sw = w.reduce(0, +)
        guard sw > 0 else { return nil }
        let ma = zip(a, w).map { $0 * $1 }.reduce(0, +) / sw, mb = zip(b, w).map { $0 * $1 }.reduce(0, +) / sw
        var sab = 0.0, saa = 0.0, sbb = 0.0
        for i in a.indices {
            sab += w[i] * (a[i] - ma) * (b[i] - mb); saa += w[i] * (a[i] - ma) * (a[i] - ma); sbb += w[i] * (b[i] - mb) * (b[i] - mb)
        }
        return saa > 0 && sbb > 0 ? sab / (saa * sbb).squareRoot() : nil
    }

    /// Fits with monocular depth, starting from `baseline` (the fit without it), and keeps the result
    /// only if it's no worse on everything else: 2D keypoints, the silhouette, user-edited joints, the
    /// pose prior and plausible size — and actually agrees better with the depth.
    public func refine(_ baseline: FittedBody, person: DetectedPerson, image: LoadedImage, cue: MonocularDepthCue,
                       useSilhouette: Bool) -> FittedBody {
        var report = MonocularDepthFitReport(backend: cue.backend.rawValue, samples: cue.sampleCount,
                                             usedAbsoluteDistance: cue.absolute, freedShape: cue.freeShape,
                                             accepted: false, reason: "")
        guard baseline.model == model.info.id else {
            var b = baseline
            b.monocularDepth = .skipped(cue.backend, "baseline fit is for another model")
            return b
        }
        let candidate = fit(person, image: image, useSilhouette: useSilhouette, warmStart: baseline, monocular: cue,
                            refineIterations: 25)
        let before = points(of: baseline), after = points(of: candidate)
        report.depthRMSBefore = cue.relativeRMS(before, tz: 0)
        report.depthRMSAfter = cue.relativeRMS(after, tz: 0)
        report.rms2DBefore = baseline.rms2D
        report.rms2DAfter = candidate.rms2D
        let rootIndex = 0
        report.distanceBefore = before[rootIndex].z
        report.distanceAfter = after[rootIndex].z
        report.visionCorrelation = cue.visionCorrelation

        report.reason = rejection(baseline: baseline, candidate: candidate, before: before, after: after, person: person,
                                  image: image, cue: cue) ?? "accepted"
        if ProcessInfo.processInfo.environment["ARMATURE_DEBUG_DEPTH"] != nil {
            // Debug aid: each sample's depth relative to the torso — map, body before, body after (cm).
            let refBefore = cue.bodyReference(before, tz: 0), refAfter = cue.bodyReference(after, tz: 0)
            print(String(format: "depth samples (reference %.2f m, σ %.0f cm): %@", cue.referenceDepth, cue.sigmaRelative * 100,
                         report.reason))
            for s in cue.samples {
                print(String(format: "  %-14@ map %+6.1f  before %+6.1f  after %+6.1f  offset %4.1f  weight %.2f%@",
                             "\(s.joint)" as NSString, (s.depth - cue.referenceDepth) * 100,
                             (before[s.target].z - s.offset - refBefore) * 100, (after[s.target].z - s.offset - refAfter) * 100,
                             s.offset * 100, s.weight, s.reference ? " (torso)" : ""))
            }
        }
        report.accepted = report.reason == "accepted"
        var out = report.accepted ? candidate : baseline
        out.monocularDepth = report
        return out
    }

    /// Why `candidate` is worse than `baseline`, or nil if it's acceptable.
    func rejection(baseline: FittedBody, candidate: FittedBody, before: [SIMD3<Double>], after: [SIMD3<Double>],
                   person: DetectedPerson, image: LoadedImage, cue: MonocularDepthCue) -> String? {
        guard candidate.pose.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }),
              candidate.translation.z.isFinite, candidate.translation.z > 0.2 else { return "fit diverged" }
        if candidate.rms2D > baseline.rms2D * 1.1 + 1 {
            return String(format: "2D keypoint error grew (%.1f → %.1f px)", baseline.rms2D, candidate.rms2D)
        }
        if let a = baseline.silhouette, let b = candidate.silhouette,
           b.outside > a.outside + 0.02 || b.iou < a.iou - 0.02 {
            return String(format: "silhouette got worse (outside %.0f%% → %.0f%%, IoU %.2f → %.2f)",
                          a.outside * 100, b.outside * 100, a.iou, b.iou)
        }
        // The user's dragged joints must stay where they were put.
        let f = image.focalLengthPixels, c = SIMD2(Double(image.width) / 2, Double(image.height) / 2)
        func reprojection(_ p: SIMD3<Double>, _ i: Int) -> Double {
            simd_distance(SIMD2(f * p.x / p.z, f * p.y / p.z) + c, person.joints2D[targets[i].joint.rawValue])
        }
        for i in targets.indices where person.edited[targets[i].joint.rawValue] {
            if reprojection(after[i], i) > reprojection(before[i], i) + 2 {
                return "moved a user-edited joint (\(targets[i].joint))"
            }
        }
        if poseEnergy(candidate) > 1.5 * poseEnergy(baseline) + 0.5 { return "pose became implausible" }
        if cue.freeShape {
            let h0 = model.height(betas: baseline.betas), h1 = model.height(betas: candidate.betas)
            if abs(h1 - h0) > 0.2 * h0 { return String(format: "body height changed implausibly (%.2f → %.2f m)", h0, h1) }
        } else if abs(after[0].z - before[0].z) > 0.15 * before[0].z {
            return String(format: "distance moved too far (%.2f → %.2f m)", before[0].z, after[0].z)
        }
        if cue.cost(after, tz: 0) >= cue.cost(before, tz: 0) - 1e-6 { return "no better agreement with the depth map" }
        // A map no plausible body can match (on the self-test, depth-inverted maps stay 20–30 cm off; good
        // ones end at 4–11 cm) — or one whose limb ordering systematically contradicts Vision's 3D pose.
        let residual = cue.relativeRMS(after, tz: 0)
        if residual > 1.5 * cue.sigmaRelative {
            return String(format: "depth map still disagrees with the fitted body by %.0f cm", residual * 100)
        }
        if let r = cue.visionCorrelation, r < -0.5 {
            return String(format: "depth ordering contradicts Vision's 3D pose (correlation %.2f)", r)
        }
        return nil
    }

    /// Stiffness-weighted squared deviation of the pose from the model's default pose.
    func poseEnergy(_ body: FittedBody) -> Double {
        var e = 0.0
        for j in 1..<min(model.jointCount, body.pose.count) {
            e += model.stiffness[j] * model.stiffness[j] * simd_length_squared(body.pose[j] - model.defaultPose[j])
        }
        return e
    }
}

extension PersonMask {
    /// Whether an image point falls on the person.
    func contains(_ p: SIMD2<Double>) -> Bool {
        let x = Int(p.x / scale), y = Int(p.y / scale)
        guard x >= 0, y >= 0, x < width, y < height else { return false }
        return inside[y * width + x] == 1
    }
}
