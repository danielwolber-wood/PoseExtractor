import Foundation

/// One person's evidence for calibrating a depth map: the map's values at their torso keypoints,
/// paired with the depth of the body's front surface at those points from the fit *without* depth.
/// That fit's distance comes from the 2D keypoints, the camera intrinsics and the body-size prior
/// (an average-sized person of that pixel height), so it's a prior, not a measurement.
public struct DepthAnchor: Sendable {
    public var values: [Double]
    public var bodyDepths: [Double]

    public init(values: [Double], bodyDepths: [Double]) {
        self.values = values
        self.bodyDepths = bodyDepths
    }
}

/// How a monocular depth map's values are turned into metres for fitting, and how far to trust that.
///
/// - `metric`: the map is already in metres (Depth Pro with a known or predicted focal length).
///   `metricAgreement` compares it with the body-size prior.
/// - `sharedScale`: relative map, one shift-free scale for everyone (inverse depth: 1/z = s·v; depth:
///   z = s·v). Needs ≥ 2 people agreeing on s, so it can place people relative to each other.
/// - `affine`: relative map, 1/z = s·v + t (or z = s·v + t) fitted through ≥ 2 people at clearly
///   different distances.
/// - `perPersonScale`: one shift-free scale per person, from that person's own prior distance. It can't
///   say anything about distance (it's circular), only about depth *within* the body — which limb is
///   in front. Shift-free assumes the model's zero is at infinity, which holds approximately for
///   Depth Anything (sky and far background go to ~0) and is why normalisation keeps the zero point.
/// - `unusable`: nothing consistent; the map isn't used for fitting.
public struct DepthCalibration: Codable, Sendable {
    public enum Method: String, Codable, Sendable { case metric, sharedScale, affine, perPersonScale, unusable }

    public var method: Method
    public var representation: DepthRepresentation
    public var scale: Double
    public var shift: Double
    /// Per-person shift-free scales (`perPersonScale`), indexed like the people; nil where unknown.
    public var personScales: [Double?]
    /// Whether the calibrated map may move a person's overall distance (not just their pose in depth).
    public var constrainsDistance: Bool
    /// True for relative maps: scale (and shift) were estimated here, not given by the model.
    public var scaleEstimated: Bool
    public var anchorCount: Int
    /// Relative spread (MAD / median) of the per-joint or per-person scale estimates.
    public var spread: Double?
    /// Metric maps: median of (prior body distance / map distance). ~1 means they agree.
    public var metricAgreement: Double?
    public var notes: [String]

    public var usable: Bool { method != .unusable }

    /// Metres for a raw map value, for person `index`. nil when the value can't be converted.
    public func metres(_ v: Double, person index: Int) -> Double? {
        guard v.isFinite, v > 0 else { return nil }
        switch method {
        case .unusable: return nil
        case .metric: return v
        case .sharedScale, .affine, .perPersonScale:
            let s = method == .perPersonScale ? (index < personScales.count ? personScales[index] : nil) : scale
            guard let s, s > 0 else { return nil }
            let t = method == .affine ? shift : 0
            if representation == .relativeInverseDepth {
                let inv = s * v + t
                return inv > 1e-3 ? 1 / inv : nil
            }
            let z = s * v + t
            return z > 0.05 ? z : nil
        }
    }

    static func unusable(_ representation: DepthRepresentation, people: Int, _ reason: String) -> DepthCalibration {
        DepthCalibration(method: .unusable, representation: representation, scale: 0, shift: 0,
                         personScales: [Double?](repeating: nil, count: people), constrainsDistance: false,
                         scaleEstimated: !representation.isMetric, anchorCount: 0, spread: nil, metricAgreement: nil,
                         notes: [reason])
    }

    /// Calibrates from each person's anchor (nil for people without enough valid samples).
    public static func estimate(representation: DepthRepresentation, anchors: [DepthAnchor?]) -> DepthCalibration {
        let people = anchors.count
        let valid = anchors.enumerated().compactMap { i, a -> (Int, DepthAnchor)? in
            guard let a, a.values.count >= 2, a.values.count == a.bodyDepths.count else { return nil }
            return (i, a)
        }
        guard !valid.isEmpty else { return unusable(representation, people: people, "no person had enough valid depth samples") }

        if representation.isMetric {
            // Metric already; check it against the body-size prior.
            let ratios = valid.flatMap { _, a in zip(a.bodyDepths, a.values).map { $0 / $1 } }
            let agreement = median(ratios)
            var c = DepthCalibration(method: .metric, representation: representation, scale: 1, shift: 0,
                                     personScales: [Double?](repeating: nil, count: people), constrainsDistance: true,
                                     scaleEstimated: false, anchorCount: valid.count, spread: relativeMAD(ratios),
                                     metricAgreement: agreement, notes: [])
            if agreement < 0.67 || agreement > 1.5 {
                // Either the model's scale or the prior is badly off (a wrong focal length, a child vs an adult
                // prior...). Don't let it move the body; its relative structure is still usable.
                c.constrainsDistance = false
                c.notes.append(String(format: "metric depth disagrees with the body-size prior (ratio %.2f); used for relative depth only", agreement))
            }
            return c
        }

        let inverse = representation == .relativeInverseDepth
        // Shift-free scale per joint: inverse depth s = 1/(z·v); depth s = z/v.
        func jointScales(_ a: DepthAnchor) -> [Double] {
            zip(a.bodyDepths, a.values).compactMap { z, v in
                guard z > 0.1, v > 0 else { return nil }
                return inverse ? 1 / (z * v) : z / v
            }
        }
        var personScales = [Double?](repeating: nil, count: people)
        var notes: [String] = []
        var intraSpreads: [Double] = []
        for (i, a) in valid {
            let s = jointScales(a)
            guard s.count >= 2 else { continue }
            let spread = relativeMAD(s)
            intraSpreads.append(spread)
            // Torso points sit within ~10 cm of the body's own depth there, so with the person's scale the
            // map must reproduce them to about that; if it doesn't, it's seeing hair, clothing or background
            // at the torso, and this person's calibration is unreliable.
            let scale = median(s)
            let implied = a.values.map { v in inverse ? 1 / (scale * v) : scale * v }
            let miss = median(zip(implied, a.bodyDepths).map { abs($0 - $1) })
            let tolerance = max(0.1, 0.03 * median(a.bodyDepths))
            if spread < 0.15, miss < tolerance { personScales[i] = scale } else {
                notes.append(String(format: "person %d: torso depth inconsistent (off by %.0f cm)", i, miss * 100))
            }
        }
        let calibrated = personScales.enumerated().compactMap { i, s in s.map { (i, $0) } }
        guard !calibrated.isEmpty else {
            return unusable(representation, people: people, notes.first ?? "torso depth samples inconsistent")
        }
        let spreadWithin = intraSpreads.isEmpty ? nil : median(intraSpreads)

        if calibrated.count >= 2 {
            let scales = calibrated.map(\.1)
            let across = relativeMAD(scales)
            if across < 0.12 {
                return DepthCalibration(method: .sharedScale, representation: representation, scale: median(scales), shift: 0,
                                        personScales: personScales, constrainsDistance: true, scaleEstimated: true,
                                        anchorCount: calibrated.count, spread: across, metricAgreement: nil, notes: notes)
            }
            // Try an affine fit through the people's median (value, depth) pairs.
            let pairs = calibrated.compactMap { i, _ -> (v: Double, z: Double)? in
                guard let a = anchors[i] else { return nil }
                return (median(a.values), median(a.bodyDepths))
            }
            let zs = pairs.map(\.z)
            if let zMin = zs.min(), let zMax = zs.max(), zMax / zMin >= 1.3,
               let (s, t) = linearFit(pairs.map(\.v), pairs.map { inverse ? 1 / $0.z : $0.z }), s > 0 {
                let predicted = pairs.map { p -> Double in
                    let y = s * p.v + t
                    return inverse ? (y > 0 ? 1 / y : .infinity) : y
                }
                let worst = zip(predicted, pairs).map { abs($0 - $1.z) / $1.z }.max() ?? .infinity
                if worst < 0.1 {
                    return DepthCalibration(method: .affine, representation: representation, scale: s, shift: t,
                                            personScales: personScales, constrainsDistance: true, scaleEstimated: true,
                                            anchorCount: pairs.count, spread: worst, metricAgreement: nil, notes: notes)
                }
            }
            notes.append(String(format: "people disagree on the depth scale (spread %.0f%%); using per-person scales", across * 100))
        }
        return DepthCalibration(method: .perPersonScale, representation: representation, scale: median(calibrated.map(\.1)),
                                shift: 0, personScales: personScales, constrainsDistance: false, scaleEstimated: true,
                                anchorCount: calibrated.count, spread: spreadWithin, metricAgreement: nil, notes: notes)
    }

    // MARK: Robust statistics

    static func median(_ v: [Double]) -> Double {
        guard !v.isEmpty else { return .nan }
        let s = v.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : 0.5 * (s[s.count / 2 - 1] + s[s.count / 2])
    }

    /// Median absolute deviation relative to the median (a robust coefficient of variation).
    static func relativeMAD(_ v: [Double]) -> Double {
        let m = median(v)
        guard m.isFinite, m != 0 else { return .infinity }
        return median(v.map { abs($0 - m) }) / abs(m)
    }

    static func linearFit(_ x: [Double], _ y: [Double]) -> (Double, Double)? {
        let n = Double(x.count)
        guard x.count >= 2 else { return nil }
        let mx = x.reduce(0, +) / n, my = y.reduce(0, +) / n
        let sxx = zip(x, x).map { ($0 - mx) * ($1 - mx) }.reduce(0, +)
        guard sxx > 1e-12 else { return nil }
        let sxy = zip(x, y).map { ($0 - mx) * ($1 - my) }.reduce(0, +)
        let s = sxy / sxx
        return (s, my - s * mx)
    }
}
