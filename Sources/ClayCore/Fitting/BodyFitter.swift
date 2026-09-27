import Accelerate
import Foundation
import simd

/// A fitted body in camera space (x right, y down, z forward, metres).
public struct FittedBody: Codable, Sendable {
    /// Id of the body model it was fitted with (e.g. "smpl_neutral", "anny").
    public var model: String
    public var pose: [SIMD3<Double>]      // one axis-angle per model joint; [0] = global orientation
    public var betas: [Double]
    public var translation: SIMD3<Double>
    /// Root-mean-square 3D joint error after fitting (metres, root-relative).
    public var rms3D: Double
    /// Root-mean-square reprojection error (pixels) over keypoints visible in the photo.
    public var rms2D: Double
    /// Whether a metric depth map constrained the body's distance (and so its size).
    public var usedDepth = false
    /// What monocular depth did to this fit (nil when no monocular depth was run).
    public var monocularDepth: MonocularDepthFitReport?
    /// Overlap of the body's projection with the person's segmentation mask, before and after the
    /// silhouette stage (nil when no mask was available or the stage was skipped).
    public var silhouetteBefore: SilhouetteOverlap?
    public var silhouette: SilhouetteOverlap?

    public init(model: String, pose: [SIMD3<Double>], betas: [Double], translation: SIMD3<Double>, rms3D: Double = 0, rms2D: Double = 0) {
        self.model = model
        self.pose = pose
        self.betas = betas
        self.translation = translation
        self.rms3D = rms3D
        self.rms2D = rms2D
    }
}

/// Fits a body model's pose + shape to a `DetectedPerson`. Model-agnostic: which joint or surface point
/// each keypoint corresponds to, joint stiffness and hinge limits all come from the model's rig.
///
/// Stage 1 matches Vision's root-relative 3D joints (pose, shape, orientation, and a free scale —
/// Vision's metric scale is a 1.8 m reference guess, so body size comes from the shape prior).
/// Stage 2 places the body in the full image's camera. The 2D keypoints are the primary evidence
/// there; Vision's 3D joints act as a soft prior that resolves depth ambiguity.
public struct BodyFitter {
    public let model: BodyModel
    public var iterations = 40
    /// Keypoints this model can fit, with their target on the body and weight. `.root` is always first.
    let targets: [(joint: BodyJoint, target: BodyModel.Target, weight: Double)]

    public init(model: BodyModel) {
        self.model = model
        targets = BodyJoint.allCases.compactMap { j in model.targets[j].map { (j, $0, Self.weight(j)) } }
    }

    // Parameter vector layout.
    var nPose: Int { model.jointCount * 3 }
    var betaOffset: Int { nPose }
    var transOffset: Int { nPose + model.betaCount }
    var scaleIndex: Int { transOffset + 3 }
    var paramCount: Int { scaleIndex + 1 }

    /// How much each keypoint counts.
    static func weight(_ j: BodyJoint) -> Double {
        switch j {
        // Vision places hips on the outside of the pelvis, body models inside it: weight them down.
        case .leftHip, .rightHip: 0.3
        case .spine: 0.5
        case .centerHead, .topHead, .leftEar, .rightEar: 0.7
        case .mouthLeft, .mouthRight, .chin: 0.8
        default:
            // Fingers: many small, noisy points; knuckles (MCP) matter most, they orient the hand.
            if j.isHand { "\(j)".hasSuffix("MCP") ? 0.8 : 0.5 } else { 1.0 }
        }
    }

    func targetPoints(_ x: [Double]) -> (points: [SIMD3<Double>], kin: BodyModel.Kinematics) {
        let pose = (0..<model.jointCount).map { SIMD3(x[$0 * 3], x[$0 * 3 + 1], x[$0 * 3 + 2]) }
        let betas = Array(x[betaOffset..<betaOffset + model.betaCount])
        let k = model.kinematics(pose: pose, betas: betas)
        return (targets.map { model.point($0.target, k, betas: betas) }, k)
    }

    /// - Parameter useSilhouette: refine body shape against the person's segmentation mask (~50 ms).
    /// - Parameter warmStart: a previous fit of the same person to start from (interactive editing):
    ///   skips stage 1 and runs a short stage 2, so a re-fit takes a few milliseconds even for Anny.
    /// - Parameter monocular: monocular depth evidence (see `MonocularDepthCue`). Ignored when the photo
    ///   has embedded metric depth, which is stronger. Use `refine` to also guard against it making the fit worse.
    /// - Parameter refineIterations: stage-2 iterations when warm-started (default 10).
    public func fit(_ person: DetectedPerson, image: LoadedImage, useSilhouette: Bool = true,
                    warmStart: FittedBody? = nil, monocular: MonocularDepthCue? = nil,
                    refineIterations: Int? = nil) -> FittedBody {
        let warm = warmStart.flatMap { $0.model == model.info.id && $0.pose.count == model.jointCount ? $0 : nil }
        let obs3 = targets.map { person.joints3D[$0.joint.rawValue] }
        let obsRel = obs3.map { $0 - obs3[0] }
        let obs2 = targets.map { person.joints2D[$0.joint.rawValue] }
        let conf2 = targets.map { person.confidence2D[$0.joint.rawValue] }
        // User-edited joints: trust the dragged 2D position, drop Vision's (evidently wrong) 3D estimate.
        let w3 = targets.map { person.edited[$0.joint.rawValue] ? 0 : $0.weight * person.confidence3D[$0.joint.rawValue] }
        let w2 = targets.map { person.edited[$0.joint.rawValue] ? 3 * $0.weight : $0.weight }
        let f = image.focalLengthPixels
        let c = SIMD2(Double(image.width) / 2, Double(image.height) / 2)
        // Person's height in pixels, from keypoints actually seen (Vision's guesses can lie far off-frame).
        let seenY = zip(obs2, conf2).filter { $0.1 > 0.2 }.map(\.0.y)
        let personPx = max((seenY.max() ?? 0) - (seenY.min() ?? 0), 20)

        let betaStiffness = 1.0
        var x = [Double](repeating: 0, count: paramCount)
        let mean = model.defaultPose
        for (j, a) in mean.enumerated() { x[j * 3] = a.x; x[j * 3 + 1] = a.y; x[j * 3 + 2] = a.z }
        // Age conditioning (models with an age-aware shape space, i.e. Anny): start from that age's average
        // shape, pull towards that age's spread of shapes, and constrain the body's read-out age.
        let age = model.supportsAge ? person.targetAge : nil
        let agePrior = age.flatMap { model.shapePrior(ageYears: $0.years) }
        if let agePrior { for b in 0..<model.betaCount { x[betaOffset + b] = agePrior.mean[b] } }
        x[0..<3] = ArraySlice(initialOrientation(observedRelative: obsRel).scalars)

        func priorResiduals(_ x: [Double], into r: inout [Double]) {
            for j in 1..<model.jointCount {
                let s = model.stiffness[j]
                for d in 0..<3 { r.append(s * (x[j * 3 + d] - mean[j][d])) }
            }
            // Hinge limits (knees, elbows, fingers): no twist or sideways bend beyond the default pose,
            // no hyperextension.
            for h in model.hinges {
                let a = SIMD3(x[h.joint * 3], x[h.joint * 3 + 1], x[h.joint * 3 + 2])
                let d = a - mean[h.joint]
                r += [h.twist * simd_dot(d, h.twistAxis), h.side * simd_dot(d, h.sideAxis),
                      h.hyper * min(0, simd_dot(a, h.flexAxis))]
            }
            if let agePrior, let age {
                for b in 0..<model.betaCount {
                    r.append(betaStiffness * (x[betaOffset + b] - agePrior.mean[b]) / max(agePrior.std[b], 0.1))
                }
                // The read-out constraint complements the prior (self-test: 1.8 vs 2.2 cm shape error over ages 6–75).
                if let readout = model.ageReadout(betas: Array(x[betaOffset..<betaOffset + model.betaCount])) {
                    r.append((readout.years - age.years) / age.sigma)
                }
            } else {
                for b in 0..<model.betaCount { r.append(betaStiffness * x[betaOffset + b]) }
            }
        }

        func residuals3D(_ pts: [SIMD3<Double>], _ x: [Double], sigma: Double, into r: inout [Double]) {
            let root = pts[0]
            let s = exp(x[scaleIndex])
            for i in 1..<pts.count {
                let d = ((pts[i] - root) - s * obsRel[i]) * (w3[i] / sigma)
                r += [d.x, d.y, d.z]
            }
        }

        // Stage 1: articulated pose, shape, orientation and Vision's scale from root-relative 3D joints.
        // Only optimise joints that can move: the rig pins some at rest (stiffness ≥ 6: toes, twist bones,
        // eyes...), and fingers only move when that hand was detected. Anny has 104 joints; this keeps
        // the problem ~25–35 joints instead of all of them.
        let handSeen = [("left", true), ("right", false)].filter { _, left in
            BodyJoint.allCases.contains { $0.isHand && $0.isLeftHand == left && person.confidence2D[$0.rawValue] > 0.3 }
        }.map(\.0)
        let fingers = Set(["Thumb", "Index", "Middle", "Ring", "Little"].flatMap { f in
            ["left", "right"].filter { !handSeen.contains($0) }.flatMap { model.chain($0 + f) }
        })
        let movable = (0..<model.jointCount).filter { $0 == 0 || (model.stiffness[$0] < 6 && !fingers.contains($0)) }
        let poseParams = movable.flatMap { j in (0..<3).map { j * 3 + $0 } }
        let stage1 = poseParams + Array(betaOffset..<transOffset) + [scaleIndex]
        if let warm {
            for (j, a) in warm.pose.enumerated() { x[j * 3] = a.x; x[j * 3 + 1] = a.y; x[j * 3 + 2] = a.z }
            for (b, v) in warm.betas.enumerated() where b < model.betaCount { x[betaOffset + b] = v }
            x[transOffset..<transOffset + 3] = ArraySlice(warm.translation.scalars)
            // Vision's scale isn't stored with a fit; least-squares it from the current skeleton.
            let pts = targetPoints(x).points
            var num = 0.0, den = 0.0
            for i in 1..<pts.count where w3[i] > 0 {
                num += simd_dot(pts[i] - pts[0], obsRel[i]); den += simd_length_squared(obsRel[i])
            }
            if num > 0, den > 0 { x[scaleIndex] = log(num / den) }
        } else {
            x = LevenbergMarquardt.minimize(x, active: stage1, iterations: iterations) { x in
                var r: [Double] = []
                residuals3D(targetPoints(x).points, x, sigma: 0.03, into: &r)
                priorResiduals(x, into: &r)
                return r
            }
        }

        // Metric depth (LiDAR/TrueDepth): the torso's measured distance fixes the body's distance, and so
        // its real size. The map measures the body's front surface; torso joints sit ~8 cm behind it.
        let torso: [BodyJoint] = [.leftShoulder, .rightShoulder, .centerShoulder, .leftHip, .rightHip, .root]
        let imageSize = SIMD2(Double(image.width), Double(image.height))
        let depthTargets: [(index: Int, depth: Double)] = image.depth.map { map in
            guard map.isAbsolute else { return [] }
            return torso.compactMap { j in
                guard let i = targets.firstIndex(where: { $0.joint == j }), conf2[i] > 0.3,
                      let d = map.depth(at: obs2[i], imageSize: imageSize) else { return nil }
                return (i, d + 0.08)
            }
        } ?? []
        let useDepth = depthTargets.count >= 3
        let measuredDepth = useDepth ? depthTargets.map(\.depth).sorted()[depthTargets.count / 2] : 0
        /// Median camera distance of the body's torso joints.
        func torsoDepth(_ x: [Double]) -> Double {
            let pts = targetPoints(x).points
            return depthTargets.map { pts[$0.index].z + x[transOffset + 2] }.sorted()[depthTargets.count / 2]
        }
        // Monocular depth only when there's no embedded metric depth (priority: sensor > model).
        let mono = useDepth ? nil : monocular
        func depthResiduals(_ x: [Double], _ pts: [SIMD3<Double>], into r: inout [Double]) {
            if useDepth { r.append((torsoDepth(x) - measuredDepth) / 0.02) }
            mono?.residuals(pts, tz: x[transOffset + 2], into: &r)
        }

        // Stage 2: translation (closed form), then pose + translation against the 2D keypoints.
        if warm == nil {
            let pts1 = targetPoints(x).points
            let t0 = solveTranslation(points: pts1, observed: obs2, weights: conf2, f: f, c: c)
                // Fallback: Vision's own root position. Its camera-relative *translation* reports depth with the
                // opposite sign to its root-relative joints (checked against ground truth in clay-selftest).
                ?? (SIMD3(obs3[0].x, obs3[0].y, abs(obs3[0].z)) - pts1[0])
            x[transOffset..<transOffset + 3] = ArraySlice(t0.scalars)
        }

        let sigma2D = 0.015 * personPx
        // With metric depth, body size is observable: let the shape adjust too.
        let freeShape = useDepth || mono?.freeShape == true
        let stage2 = poseParams + Array(transOffset..<transOffset + 3)
            + (freeShape ? Array(betaOffset..<betaOffset + model.betaCount) : [])
        if useDepth { x[transOffset + 2] += measuredDepth - torsoDepth(x) }
        if let mono, mono.absolute {
            x[transOffset + 2] += mono.referenceDepth - mono.bodyReference(targetPoints(x).points, tz: x[transOffset + 2])
        }
        x = LevenbergMarquardt.minimize(x, active: stage2, iterations: warm == nil ? iterations : (refineIterations ?? 10)) { x in
            var r: [Double] = []
            let pts = targetPoints(x).points
            let t = SIMD3(x[transOffset], x[transOffset + 1], x[transOffset + 2])
            for (i, p) in pts.enumerated() {
                let q = p + t
                let z = max(q.z, 0.05)
                let u = SIMD2(f * q.x / z, f * q.y / z) + c
                let e = (u - obs2[i]) * (w2[i] * conf2[i] / sigma2D)
                r += [e.x, e.y]
            }
            residuals3D(pts, x, sigma: 0.08, into: &r)
            priorResiduals(x, into: &r)
            depthResiduals(x, pts, into: &r)
            return r
        }

        // Stage 3: body shape (and placement) from the silhouette, pose held fixed.
        var overlap: (before: SilhouetteOverlap, after: SilhouetteOverlap)?
        if useSilhouette, let mask = person.silhouette {
            let kw = (0..<obs2.count).map { w2[$0] * conf2[$0] / sigma2D }
            overlap = fitSilhouette(&x, mask: mask, keypoints: obs2, keypointWeights: kw, f: f, c: c,
                                imageSize: SIMD2(Double(image.width), Double(image.height)),
                                personPx: personPx, prior: { x, r in
                                    // A weak pull toward Vision's 3D: the silhouette can't see depth, so without
                                    // it the trunk and hips drift; stage 2's stronger 8 cm version over-constrains
                                    // (Vision's own 3D is ~18 cm off). Tuned on clay-selftest.
                                    let pts = targetPoints(x).points
                                    residuals3D(pts, x, sigma: 0.25, into: &r)
                                    priorResiduals(x, into: &r)
                                    depthResiduals(x, pts, into: &r)
                                })
        }

        let pts = targetPoints(x).points
        let t = SIMD3(x[transOffset], x[transOffset + 1], x[transOffset + 2])
        let s = exp(x[scaleIndex])
        // 3D error over the joints Vision actually estimated in 3D.
        let with3D = pts.indices.filter { w3[$0] > 0 }
        let rms3 = sqrt(with3D.map { simd_length_squared((pts[$0] - pts[0]) - s * obsRel[$0]) }.reduce(0, +)
                        / Double(max(with3D.count, 1)))
        // Reprojection error over the keypoints actually seen in the photo.
        let seen = pts.indices.filter { conf2[$0] > 0.2 }
        let rms2 = sqrt(seen.map { i in
            let q = pts[i] + t
            return simd_length_squared(SIMD2(f * q.x / q.z, f * q.y / q.z) + c - obs2[i])
        }.reduce(0, +) / Double(max(seen.count, 1)))

        var body = FittedBody(
            model: model.info.id,
            pose: (0..<model.jointCount).map { SIMD3(x[$0 * 3], x[$0 * 3 + 1], x[$0 * 3 + 2]) },
            betas: Array(x[betaOffset..<betaOffset + model.betaCount]),
            translation: t, rms3D: rms3, rms2D: rms2)
        body.usedDepth = useDepth
        body.silhouetteBefore = overlap?.before
        body.silhouette = overlap?.after
        return body
    }

    /// Kabsch alignment of the rest-pose torso onto the observed torso.
    private func initialOrientation(observedRelative obs: [SIMD3<Double>]) -> SIMD3<Double> {
        let rest = targetPoints([Double](repeating: 0, count: paramCount)).points
        let torso: [BodyJoint] = [.leftHip, .rightHip, .spine, .centerShoulder, .leftShoulder, .rightShoulder, .centerHead]
        let idx = torso.compactMap { j in targets.firstIndex { $0.joint == j } }
        let a = idx.map { rest[$0] - rest[0] }, b = idx.map { obs[$0] }
        var H = simd_double3x3()
        for (p, q) in zip(a, b) { H += simd_double3x3(columns: (p * q.x, p * q.y, p * q.z)) }  // Σ p qᵀ
        let R = rotationFromCovariance(H)
        return axisAngle(R)
    }

    /// Linear least squares for t such that the pinhole projection of (p + t) matches the keypoints.
    private func solveTranslation(points: [SIMD3<Double>], observed: [SIMD2<Double>], weights: [Double],
                                  f: Double, c: SIMD2<Double>) -> SIMD3<Double>? {
        var AtA = simd_double3x3(), Atb = SIMD3<Double>.zero
        for ((p, o), w) in zip(zip(points, observed), weights) {
            let du = o.x - c.x, dv = o.y - c.y
            for (row, rhs) in [(SIMD3(f, 0, -du), du * p.z - f * p.x), (SIMD3(0, f, -dv), dv * p.z - f * p.y)] {
                AtA += w * simd_double3x3(columns: (row * row.x, row * row.y, row * row.z))
                Atb += w * row * rhs
            }
        }
        guard abs(AtA.determinant) > 1e-12 else { return nil }
        let t = AtA.inverse * Atb
        return t.z > 0.3 ? t : nil
    }
}

/// Rotation R maximising tr(R·H) where H = Σ p qᵀ (so R·p ≈ q), via the polar decomposition.
func rotationFromCovariance(_ H: simd_double3x3) -> simd_double3x3 {
    // Polar decomposition of Hᵀ by Newton iteration; fix up reflections.
    var M = H.transpose
    if abs(M.determinant) < 1e-12 { return matrix_identity_double3x3 }
    for _ in 0..<30 {
        let next = 0.5 * (M + M.inverse.transpose)
        if simd_almost_equal_elements(next, M, 1e-12) { M = next; break }
        M = next
    }
    if M.determinant < 0 {
        // Nearest proper rotation: flip the axis with the smallest singular value via SVD-free fallback.
        M = simd_double3x3(columns: (M.columns.0, M.columns.1, -M.columns.2))
    }
    return M
}

extension SIMD3 where Scalar == Double {
    var scalars: [Double] { [x, y, z] }
}

// MARK: - Levenberg–Marquardt

enum LevenbergMarquardt {
    /// Minimises ½‖r(x)‖² over the `active` subset of parameters, forward-difference Jacobian.
    static func minimize(_ x0: [Double], active: [Int], iterations: Int, residuals: ([Double]) -> [Double]) -> [Double] {
        var x = x0
        var r = residuals(x)
        var cost = r.reduce(0) { $0 + $1 * $1 }
        var lambda = 1e-3
        let n = active.count
        let h = 1e-6

        for _ in 0..<iterations {
            let m = r.count
            var J = [Double](repeating: 0, count: m * n)  // column-major
            for (col, p) in active.enumerated() {
                var xp = x
                xp[p] += h
                let rp = residuals(xp)
                for i in 0..<m { J[col * m + i] = (rp[i] - r[i]) / h }
            }
            // JᵀJ and Jᵀr with BLAS (J is m×n column-major); symmetric fill of the upper triangle.
            var JtJ = [Double](repeating: 0, count: n * n)
            var Jtr = [Double](repeating: 0, count: n)
            cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, Int32(n), Int32(m), 1, J, Int32(m), 0, &JtJ, Int32(n))
            for a in 0..<n { for b in 0..<a { JtJ[b * n + a] = JtJ[a * n + b] } }
            cblas_dgemv(CblasColMajor, CblasTrans, Int32(m), Int32(n), 1, J, Int32(m), r, 1, 0, &Jtr, 1)

            var improved = false
            for _ in 0..<8 {
                var A = JtJ
                for d in 0..<n { A[d * n + d] += lambda * (JtJ[d * n + d] + 1e-6) }
                guard let delta = choleskySolve(A, Jtr.map { -$0 }, n: n) else { lambda *= 10; continue }
                var xn = x
                for (k, p) in active.enumerated() { xn[p] += delta[k] }
                let rn = residuals(xn)
                let cn = rn.reduce(0) { $0 + $1 * $1 }
                if cn < cost {
                    let rel = (cost - cn) / max(cost, 1e-12)
                    x = xn; r = rn; cost = cn
                    lambda = max(lambda / 3, 1e-7)
                    improved = true
                    if rel < 1e-7 { return x }
                    break
                }
                lambda *= 4
            }
            if !improved { break }
        }
        return x
    }

    static func choleskySolve(_ A: [Double], _ b: [Double], n: Int) -> [Double]? {
        var L = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0...i {
                var s = A[i * n + j]
                for k in 0..<j { s -= L[i * n + k] * L[j * n + k] }
                if i == j {
                    guard s > 0 else { return nil }
                    L[i * n + i] = sqrt(s)
                } else {
                    L[i * n + j] = s / L[j * n + j]
                }
            }
        }
        var y = b
        for i in 0..<n {
            for k in 0..<i { y[i] -= L[i * n + k] * y[k] }
            y[i] /= L[i * n + i]
        }
        for i in stride(from: n - 1, through: 0, by: -1) {
            for k in (i + 1)..<n { y[i] -= L[k * n + i] * y[k] }
            y[i] /= L[i * n + i]
        }
        return y
    }
}
