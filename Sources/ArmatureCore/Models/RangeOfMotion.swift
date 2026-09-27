import Foundation
import simd

// Anatomical plausibility: joint range-of-motion limits and self-collision capsules, both read from the
// rig description (see tools/convert_models.py). The fitter turns them into penalties; `plausibility`
// reports what a finished pose still violates.

extension BodyModel {
    /// Swing-twist limit of a ball joint (hip, shoulder, spine, neck, wrist, ankle...), relative to an
    /// anatomical neutral direction rather than the rest pose, so T- and A-posed rigs get the same limits.
    struct JointLimit {
        let joint: Int
        /// Neutral bone direction, in the parent's frame (the canonical frame at rest).
        let neutral: SIMD3<Double>
        /// Minimal rotation from the neutral direction to the rest bone (identity when they coincide).
        let restFromNeutral: simd_quatd
        /// Swinging about `flexAxis` by a positive angle is flexion; about `abductAxis`, abduction.
        let flexAxis: SIMD3<Double>, abductAxis: SIMD3<Double>
        let flex: Double, extend: Double, abduct: Double, adduct: Double, twist: Double

        init(_ l: RigDescription.Limit, rest: [SIMD3<Double>]) {
            joint = l.joint
            let n = simd_normalize(SIMD3(l.neutral[0], l.neutral[1], l.neutral[2]))
            neutral = n
            let bone = l.child.map { simd_normalize(rest[$0] - rest[l.joint]) } ?? n
            restFromNeutral = simd_quatd(from: n, to: bone)
            let f = simd_normalize(simd_cross(n, SIMD3(l.flexDir[0], l.flexDir[1], l.flexDir[2])))
            let a = simd_cross(n, SIMD3(l.abductDir[0], l.abductDir[1], l.abductDir[2]))
            flexAxis = f
            abductAxis = simd_normalize(a - simd_dot(a, f) * f)
            (flex, extend, abduct, adduct, twist) = (l.flex, l.extend, l.abduct, l.adduct, l.twist)
        }

        /// How far (radians) a local rotation (axis-angle, relative to the parent) swings and twists
        /// beyond the limits. The swing limit is an ellipse in each quadrant of (flexion, abduction).
        func excess(_ r: SIMD3<Double>) -> (swing: Double, twist: Double) {
            let angle = simd_length(r)
            let local = angle < 1e-12 ? simd_quatd(ix: 0, iy: 0, iz: 0, r: 1) : simd_quatd(angle: angle, axis: r / angle)
            var q = local * restFromNeutral
            if q.real < 0 { q = simd_quatd(vector: -q.vector) }
            // Twist about the neutral direction, then the swing that remains.
            let t = simd_dot(q.imag, neutral)
            let tau = 2 * atan2(t, q.real)
            let norm = (t * t + q.real * q.real).squareRoot()
            let twistQ = norm < 1e-12 ? simd_quatd(ix: 0, iy: 0, iz: 0, r: 1)
                : simd_quatd(ix: neutral.x * t / norm, iy: neutral.y * t / norm, iz: neutral.z * t / norm, r: q.real / norm)
            var s = q * twistQ.inverse
            if s.real < 0 { s = simd_quatd(vector: -s.vector) }
            let sAngle = 2 * atan2(simd_length(s.imag), s.real)
            let swing = sAngle < 1e-12 ? SIMD3<Double>.zero : simd_normalize(s.imag) * sAngle
            let sf = simd_dot(swing, flexAxis), sa = simd_dot(swing, abductAxis)
            let rho = ((sf / (sf >= 0 ? flex : extend)).squared + (sa / (sa >= 0 ? abduct : adduct)).squared).squareRoot()
            return (rho > 1 ? sAngle * (1 - 1 / rho) : 0, max(0, abs(tau) - twist))
        }
    }

    /// A capsule approximating one body part for self-collision: a segment between two points, each a
    /// joint plus an offset in that joint's frame, and a radius (all at the template's size).
    struct Capsule {
        let name: String
        let a: Int, b: Int
        let offsetA: SIMD3<Double>, offsetB: SIMD3<Double>
        let radius: Double
    }

    /// Body size relative to the template (pelvis-to-head distance), to scale capsules with shape.
    func capsuleScale(_ k: Kinematics) -> Double {
        guard let p = joint("pelvis"), let h = joint("head") else { return 1 }
        let template = simd_distance(jTemplate[h], jTemplate[p])
        return template > 0 ? simd_distance(k.restJoints[h], k.restJoints[p]) / template : 1
    }

    /// Overlap (metres) of each collision pair's capsules in a posed skeleton; 0 where they don't touch.
    func penetrations(_ k: Kinematics) -> [Double] {
        guard !collisionPairs.isEmpty else { return [] }
        let s = capsuleScale(k)
        let ends = capsules.map { c in
            (k.joints[c.a] + k.rotations[c.a] * (s * c.offsetA), k.joints[c.b] + k.rotations[c.b] * (s * c.offsetB))
        }
        return collisionPairs.map { i, j in
            let d = segmentDistance(ends[i].0, ends[i].1, ends[j].0, ends[j].1)
            return max(0, s * (capsules[i].radius + capsules[j].radius) - d)
        }
    }

    /// What a pose still gets wrong anatomically: joints past their range of motion (degrees beyond the
    /// limit, keyed by joint name) and body parts passing through each other (overlap in cm, keyed
    /// "part/part"). Only violations above `minDegrees` / `minCentimetres` are listed.
    public func plausibility(pose: [SIMD3<Double>], betas: [Double], minDegrees: Double = 3,
                             minCentimetres: Double = 1.5) -> PlausibilityReport {
        var report = PlausibilityReport()
        for l in limits {
            let e = l.excess(pose[l.joint])
            let deg = max(e.swing, e.twist) * 180 / .pi
            if deg > minDegrees { report.beyondLimits[jointLabel(l.joint), default: 0] = max(report.beyondLimits[jointLabel(l.joint)] ?? 0, deg) }
        }
        for h in hinges {
            // Past full flexion, or bent the wrong way (hyperextended).
            // (Wrapped to ±180°: an unconstrained axis-angle can wind past a full turn.)
            let raw = h.restFlex + simd_dot(pose[h.joint], h.flexAxis)
            let flex = atan2(sin(raw), cos(raw))
            let deg = max(flex - h.maxFlex, -flex, 0) * 180 / .pi
            if deg > minDegrees { report.beyondLimits[jointLabel(h.joint)] = deg }
        }
        let k = kinematics(pose: pose, betas: betas)
        for ((i, j), pen) in zip(collisionPairs, penetrations(k)) where pen * 100 > minCentimetres {
            report.penetrations["\(capsules[i].name)/\(capsules[j].name)"] = pen * 100
        }
        return report
    }

    /// A joint's semantic name where it has one ("leftKnee", "spine[1]"), else the model's own name.
    func jointLabel(_ j: Int) -> String {
        for (name, v) in semantic {
            switch v {
            case .joint(let i) where i == j: return name
            case .chain(let c) where c.contains(j): return c.count == 1 ? name : "\(name)[\(c.firstIndex(of: j)!)]"
            default: continue
            }
        }
        return j < jointNames.count ? jointNames[j] : "joint\(j)"
    }
}

/// Anatomical problems left in a fitted pose (see `BodyModel.plausibility`).
public struct PlausibilityReport: Codable, Sendable, Equatable {
    /// Joints past their range of motion: joint name → degrees beyond the limit.
    public var beyondLimits: [String: Double] = [:]
    /// Body parts passing through each other: "part/part" → overlap in centimetres.
    public var penetrations: [String: Double] = [:]
    /// Limbs whose depth the fitter flipped relative to Vision's 3D estimate (pointing towards the camera
    /// instead of away, or the reverse), because that fitted the photo more plausibly.
    public var flippedLimbs: [String] = []

    public var isClean: Bool { beyondLimits.isEmpty && penetrations.isEmpty }
}

/// Closest distance between segments p1–q1 and p2–q2 (Ericson, Real-Time Collision Detection 5.1.9).
func segmentDistance(_ p1: SIMD3<Double>, _ q1: SIMD3<Double>, _ p2: SIMD3<Double>, _ q2: SIMD3<Double>) -> Double {
    let d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2
    let a = simd_length_squared(d1), e = simd_length_squared(d2), f = simd_dot(d2, r)
    var s = 0.0, t = 0.0
    if a <= 1e-12 && e <= 1e-12 { return simd_distance(p1, p2) }
    if a <= 1e-12 {
        t = min(max(f / e, 0), 1)
    } else {
        let c = simd_dot(d1, r)
        if e <= 1e-12 {
            s = min(max(-c / a, 0), 1)
        } else {
            let b = simd_dot(d1, d2), denom = a * e - b * b
            s = denom > 1e-12 ? min(max((b * f - c * e) / denom, 0), 1) : 0
            t = (b * s + f) / e
            if t < 0 { t = 0; s = min(max(-c / a, 0), 1) } else if t > 1 { t = 1; s = min(max((b - c) / a, 0), 1) }
        }
    }
    return simd_distance(p1 + d1 * s, p2 + d2 * t)
}

private extension Double {
    var squared: Double { self * self }
}
