import Accelerate
import Foundation
import simd

public enum ModelError: LocalizedError {
    case missing(URL)
    case badArray(String)
    case unknownModel(String)

    public var errorDescription: String? {
        switch self {
        case .missing(let url):
            return "Model files not found at \(url.path). Run `uv run --with numpy --with scipy tools/convert_models.py` first."
        case .badArray(let name):
            return "Model array '\(name)' is missing or has an unexpected size."
        case .unknownModel(let id):
            return "No converted body model called '\(id)'."
        }
    }
}

/// A converted body model on disk (see tools/convert_models.py).
public struct BodyModelInfo: Identifiable, Hashable, Sendable {
    public let id: String
    public let displayName: String
    public let family: String
    public let licence: String
    let order: Int

    /// Models in `directory`, in display order.
    public static func available(in directory: URL) -> [BodyModelInfo] {
        let dirs = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return dirs.compactMap { dir in
            struct Header: Decodable { let family: String?; let displayName: String?; let licence: String?; let order: Int? }
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("meta.json")),
                  let h = try? JSONDecoder().decode(Header.self, from: data) else { return nil }
            let id = dir.lastPathComponent
            return BodyModelInfo(id: id, displayName: h.displayName ?? id, family: h.family ?? "",
                                 licence: h.licence ?? "", order: h.order ?? 99)
        }.sorted { ($0.order, $0.id) < ($1.order, $1.id) }
    }
}

/// Reads the flat `meta.json` + `data.bin` layout written by tools/convert_models.py.
struct ModelArchive {
    struct ArrayInfo: Decodable { let dtype: String; let shape: [Int]; let offset: Int }
    struct Meta: Decodable {
        let arrays: [String: ArrayInfo]
        let family: String?
        let displayName: String?
        let licence: String?
        let order: Int?
        let rig: BodyModel.RigDescription
    }

    let meta: Meta
    let data: Data

    init(directory: URL) throws {
        let metaURL = directory.appendingPathComponent("meta.json")
        let dataURL = directory.appendingPathComponent("data.bin")
        guard FileManager.default.fileExists(atPath: metaURL.path),
              FileManager.default.fileExists(atPath: dataURL.path) else { throw ModelError.missing(directory) }
        meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        data = try Data(contentsOf: dataURL, options: .alwaysMapped)
    }

    func shape(_ name: String) -> [Int]? { meta.arrays[name]?.shape }

    func array<T>(_ name: String, as type: T.Type) throws -> [T] {
        guard let info = meta.arrays[name] else { throw ModelError.badArray(name) }
        let count = info.shape.reduce(1, *)
        guard info.offset + count * MemoryLayout<T>.stride <= data.count else { throw ModelError.badArray(name) }
        return data.withUnsafeBytes { raw in
            Array(UnsafeBufferPointer(start: raw.baseAddress!.advanced(by: info.offset).assumingMemoryBound(to: T.self),
                                      count: count))
        }
    }
}

/// A linear-blend-skinned parametric body model: SMPL, SMPL-X, Anny, ...
///
/// All models share one canonical frame (y up, facing +z, +x = the body's left, metres) and one
/// formulation: `v(β, θ) = LBS(template + S·β + P·f(θ))`, with rest joints affine in β. What differs
/// between models — which joint is the knee, where the nose is on the mesh, how stiff each joint is —
/// comes from the rig description written by the converter, so the fitter stays model-agnostic.
public final class BodyModel: @unchecked Sendable {
    // MARK: Rig description (from meta.json)

    struct RigDescription: Decodable {
        struct TargetSpec: Decodable { let joint: Int?; let mid: [Int]?; let vertices: [[Double]]? }
        struct Hinge: Decodable {
            let joint: Int; let child: Int; let flex: [Double]; let twist: Double; let side: Double; let hyper: Double
            let restFlex: Double?; let maxFlex: Double?
        }
        struct Limit: Decodable {
            let joint: Int; let child: Int?; let neutral: [Double]; let flexDir: [Double]; let abductDir: [Double]
            let flex: Double; let extend: Double; let abduct: Double; let adduct: Double; let twist: Double
        }
        struct Collision: Decodable {
            struct Capsule: Decodable { let name: String; let a: Int; let b: Int; let offsetA: [Double]; let offsetB: [Double]; let radius: Double }
            let capsules: [Capsule]
            let pairs: [[Int]]
        }
        struct Phenotype: Decodable {
            struct AgeShape: Decodable { let ages: [Double]; let mean: [[Double]]; let std: [[Double]] }
            let labels: [String]
            let matrix: [[Double]]
            let ageShape: AgeShape?
        }
        enum JointOrChain: Decodable {
            case joint(Int), chain([Int])
            init(from decoder: Decoder) throws {
                let c = try decoder.singleValueContainer()
                if let i = try? c.decode(Int.self) { self = .joint(i) } else { self = .chain(try c.decode([Int].self)) }
            }
        }
        let jointNames: [String]
        let semantic: [String: JointOrChain]
        let targets: [String: TargetSpec]
        let stiffness: [Double]
        let hinges: [Hinge]
        /// Range-of-motion limits and self-collision capsules (absent in models converted before they existed).
        let limits: [Limit]?
        let collision: Collision?
        let silhouetteJoints: [Int]
        let phenotype: Phenotype?
        let poseMean: [[Double]]?
    }

    /// Where a keypoint lives on the body.
    enum Target {
        case joint(Int)
        case mid(Int, Int)
        case surface(SurfacePoint)
    }

    /// A weighted blend of mesh vertices, with what's needed to pose it without a full mesh pass.
    struct SurfacePoint {
        struct Vertex { let weight: Double; let template: SIMD3<Double>; let shapedirs: [SIMD3<Double>]; let skin: [(Int, Double)] }
        let vertices: [Vertex]
    }

    struct Hinge {
        let joint: Int
        let flexAxis: SIMD3<Double>, twistAxis: SIMD3<Double>, sideAxis: SIMD3<Double>
        let twist: Double, side: Double, hyper: Double
        /// Flexion already in the rest pose (Anny's A-pose has bent elbows) and the most the joint bends,
        /// both measured from straight (radians).
        let restFlex: Double, maxFlex: Double
    }

    // MARK: Data

    public let info: BodyModelInfo
    public let vertexCount: Int
    public let jointCount: Int
    public let betaCount: Int
    public let faces: [UInt32]
    public let parents: [Int]
    public let jointNames: [String]
    /// The model's natural default pose (identity everywhere except, e.g., SMPL-X's relaxed hands).
    /// Pose priors pull towards it and joints nothing observes stay at it.
    public let defaultPose: [SIMD3<Double>]

    let vTemplate: [Float]       // V*3
    let shapedirs: [Float]       // (V*3) x B, row-major
    let posedirs: [Float]?       // (V*3) x 9(J-1), row-major (nil: pure LBS, e.g. Anny)
    let weights: [Float]         // V x J, row-major
    let sparseWeights: [[(Int, Double)]]
    let jTemplate: [SIMD3<Double>]
    let jShapedirs: [[SIMD3<Double>]]   // J x B

    let targets: [BodyJoint: Target]
    let stiffness: [Double]
    let hinges: [Hinge]
    let limits: [JointLimit]
    let capsules: [Capsule]
    let collisionPairs: [(Int, Int)]
    let silhouetteJoints: [Int]
    let semantic: [String: RigDescription.JointOrChain]
    private let phenotypeModel: RigDescription.Phenotype?

    var poseFeatureCount: Int { 9 * (jointCount - 1) }

    public convenience init(modelsDirectory: URL, id: String) throws {
        try self.init(directory: modelsDirectory.appendingPathComponent(id))
    }

    public init(directory: URL) throws {
        let archive = try ModelArchive(directory: directory)
        let meta = archive.meta
        guard let vShape = archive.shape("v_template"), let sShape = archive.shape("shapedirs"),
              let wShape = archive.shape("weights") else { throw ModelError.badArray("v_template/shapedirs/weights") }
        let V = vShape[0], B = sShape[2], J = wShape[1]
        let id = directory.lastPathComponent
        info = BodyModelInfo(id: id, displayName: meta.displayName ?? id, family: meta.family ?? "",
                             licence: meta.licence ?? "", order: meta.order ?? 99)
        vertexCount = V
        jointCount = J
        betaCount = B
        vTemplate = try archive.array("v_template", as: Float.self)
        shapedirs = try archive.array("shapedirs", as: Float.self)
        posedirs = archive.shape("posedirs") != nil ? try archive.array("posedirs", as: Float.self) : nil
        weights = try archive.array("weights", as: Float.self)
        faces = try archive.array("faces", as: UInt32.self)
        parents = try archive.array("parents", as: UInt32.self).map { $0 >= UInt32(J) ? -1 : Int($0) }
        let jt = try archive.array("joints_template", as: Float.self)
        let js = try archive.array("joints_shapedirs", as: Float.self)
        jTemplate = (0..<J).map { SIMD3(Double(jt[$0 * 3]), Double(jt[$0 * 3 + 1]), Double(jt[$0 * 3 + 2])) }
        jShapedirs = (0..<J).map { j in
            (0..<B).map { b in SIMD3((0..<3).map { Double(js[(j * 3 + $0) * B + b]) }) }
        }
        let w = weights
        sparseWeights = (0..<V).map { v in
            (0..<J).compactMap { j in let x = Double(w[v * J + j]); return x > 1e-4 ? (j, x) : nil }
        }

        let rig = meta.rig
        jointNames = rig.jointNames
        defaultPose = (0..<J).map { j in
            guard let m = rig.poseMean, j < m.count, m[j].count == 3 else { return .zero }
            return SIMD3(m[j][0], m[j][1], m[j][2])
        }
        stiffness = rig.stiffness
        silhouetteJoints = rig.silhouetteJoints
        semantic = rig.semantic
        phenotypeModel = rig.phenotype

        // Resolve keypoint targets by BodyJoint name.
        let (vt, sd, sw) = (vTemplate, shapedirs, sparseWeights)
        var resolved: [BodyJoint: Target] = [:]
        for joint in BodyJoint.allCases {
            guard let spec = rig.targets["\(joint)"] else { continue }
            if let j = spec.joint {
                resolved[joint] = .joint(j)
            } else if let m = spec.mid, m.count == 2 {
                resolved[joint] = .mid(m[0], m[1])
            } else if let vs = spec.vertices {
                let total = vs.reduce(0) { $0 + $1[1] }
                resolved[joint] = .surface(SurfacePoint(vertices: vs.map { pair in
                    let v = Int(pair[0])
                    return .init(weight: pair[1] / total,
                                 template: SIMD3(Double(vt[v * 3]), Double(vt[v * 3 + 1]), Double(vt[v * 3 + 2])),
                                 shapedirs: (0..<B).map { b in SIMD3((0..<3).map { Double(sd[(v * 3 + $0) * B + b]) }) },
                                 skin: sw[v])
                }))
            }
        }
        targets = resolved

        // Hinge axes from the rest skeleton: flexing rotates the child towards `flex`.
        let rest = jTemplate
        hinges = rig.hinges.map { h in
            let bone = simd_normalize(rest[h.child] - rest[h.joint])
            let flex = simd_normalize(simd_cross(bone, SIMD3(h.flex[0], h.flex[1], h.flex[2])))
            return Hinge(joint: h.joint, flexAxis: flex, twistAxis: bone, sideAxis: simd_normalize(simd_cross(flex, bone)),
                         twist: h.twist, side: h.side, hyper: h.hyper, restFlex: h.restFlex ?? 0, maxFlex: h.maxFlex ?? .infinity)
        }
        limits = (rig.limits ?? []).map { JointLimit($0, rest: rest) }
        let v3 = { (a: [Double]) in SIMD3(a[0], a[1], a[2]) }
        capsules = (rig.collision?.capsules ?? []).map {
            Capsule(name: $0.name, a: $0.a, b: $0.b, offsetA: v3($0.offsetA), offsetB: v3($0.offsetB), radius: $0.radius)
        }
        collisionPairs = (rig.collision?.pairs ?? []).map { ($0[0], $0[1]) }
    }

    /// A semantic joint's index (e.g. "leftShoulder", "pelvis"); for chains, the first joint.
    public func joint(_ name: String) -> Int? {
        switch semantic[name] {
        case .joint(let j): j
        case .chain(let c): c.first
        case nil: nil
        }
    }

    /// Flexion axis of a hinge joint (knee, elbow, finger), in the rest frame: rotating about it by a
    /// positive angle bends the joint the natural way.
    public func flexAxis(joint: Int) -> SIMD3<Double>? {
        hinges.first { $0.joint == joint }?.flexAxis
    }

    public func chain(_ name: String) -> [Int] {
        switch semantic[name] {
        case .joint(let j): [j]
        case .chain(let c): c
        case nil: []
        }
    }

    // MARK: - Kinematics

    public struct Kinematics {
        public var restJoints: [SIMD3<Double>]
        public var rotations: [simd_double3x3]   // global rotation of each joint frame
        public var joints: [SIMD3<Double>]       // posed joint positions (before translation)
    }

    public func restJoints(betas: [Double]) -> [SIMD3<Double>] {
        var out = jTemplate
        for j in 0..<jointCount {
            for b in 0..<min(betas.count, betaCount) { out[j] += jShapedirs[j][b] * betas[b] }
        }
        return out
    }

    /// `pose` holds one axis-angle rotation per joint (root first), relative to the parent, in the
    /// canonical frame's axes (rest rotations are identity, as in SMPL).
    public func kinematics(pose: [SIMD3<Double>], betas: [Double]) -> Kinematics {
        let rest = restJoints(betas: betas)
        var rot = [simd_double3x3](repeating: matrix_identity_double3x3, count: jointCount)
        var pos = [SIMD3<Double>](repeating: .zero, count: jointCount)
        for j in 0..<jointCount {
            let local = rodrigues(pose[j])
            let p = parents[j]
            if p < 0 {
                rot[j] = local
                pos[j] = rest[j]
            } else {
                rot[j] = rot[p] * local
                pos[j] = pos[p] + rot[p] * (rest[j] - rest[p])
            }
        }
        return Kinematics(restJoints: rest, rotations: rot, joints: pos)
    }

    /// Posed position of a keypoint target (surface points use LBS without pose correctives).
    func point(_ target: Target, _ k: Kinematics, betas: [Double]) -> SIMD3<Double> {
        switch target {
        case .joint(let j): return k.joints[j]
        case .mid(let a, let b): return 0.5 * (k.joints[a] + k.joints[b])
        case .surface(let s):
            var out = SIMD3<Double>.zero
            for v in s.vertices {
                var p = v.template
                for b in 0..<min(betas.count, betaCount) { p += v.shapedirs[b] * betas[b] }
                var q = SIMD3<Double>.zero
                for (j, w) in v.skin { q += w * (k.rotations[j] * (p - k.restJoints[j]) + k.joints[j]) }
                out += v.weight * q
            }
            return out
        }
    }

    /// Standing height (metres) of the body with these shape coefficients, in the rest pose.
    public func height(betas: [Double]) -> Double {
        let v = vertices(pose: [SIMD3<Double>](repeating: .zero, count: jointCount), betas: betas)
        return Double((v.map(\.y).max() ?? 0) - (v.map(\.y).min() ?? 0))
    }

    /// For models with an interpretable shape space (Anny): phenotype read out from the shape
    /// coefficients, e.g. ["ageYears": 34, "gender": 0.8, ...].
    public func phenotype(betas: [Double]) -> [String: Double]? {
        guard let p = phenotypeModel else { return nil }
        var out: [String: Double] = [:]
        for (label, row) in zip(p.labels, p.matrix) {
            var s = row.last ?? 0
            for b in 0..<min(betas.count, row.count - 1) { s += row[b] * betas[b] }
            out[label] = s
        }
        return out
    }

    /// Whether the shape space can be conditioned on age (Anny).
    public var supportsAge: Bool { phenotypeModel?.ageShape != nil && phenotypeModel?.labels.contains("ageYears") == true }

    /// Shape prior for a given age: per-coefficient mean and spread (linearly interpolated over the
    /// converter's 1-year grid, clamped to 0–90). nil for models without an age-aware shape space.
    public func shapePrior(ageYears: Double) -> (mean: [Double], std: [Double])? {
        guard let a = phenotypeModel?.ageShape, let first = a.ages.first, let last = a.ages.last, a.ages.count > 1 else { return nil }
        let t = min(max(ageYears, first), last)
        let i = min(Int((t - first) / (a.ages[1] - a.ages[0])), a.ages.count - 2)
        let f = (t - a.ages[i]) / (a.ages[i + 1] - a.ages[i])
        let lerp = { (x: [Double], y: [Double]) in zip(x, y).map { $0 + ($1 - $0) * f } }
        return (lerp(a.mean[i], a.mean[i + 1]), lerp(a.std[i], a.std[i + 1]))
    }

    /// Apparent age (years) read out from shape coefficients, and its linear gradient with respect to them.
    func ageReadout(betas: [Double]) -> (years: Double, gradient: [Double])? {
        guard let p = phenotypeModel, let row = p.labels.firstIndex(of: "ageYears").map({ p.matrix[$0] }) else { return nil }
        var y = row.last ?? 0
        for b in 0..<min(betas.count, row.count - 1) { y += row[b] * betas[b] }
        return (y, Array(row.dropLast()))
    }

    // MARK: - Mesh

    /// Pose-corrective offsets (rest space) for every vertex at `pose`; nil for pure-LBS models.
    func poseCorrectives(pose: [SIMD3<Double>]) -> [SIMD3<Double>]? {
        guard let posedirs else { return nil }
        var feature = [Float](repeating: 0, count: poseFeatureCount)
        for j in 1..<jointCount {
            let r = rodrigues(pose[j])
            for row in 0..<3 {
                for col in 0..<3 { feature[(j - 1) * 9 + row * 3 + col] = Float(r[col][row] - (row == col ? 1 : 0)) }
            }
        }
        var out = [Float](repeating: 0, count: vertexCount * 3)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(vertexCount * 3), Int32(poseFeatureCount), 1, posedirs,
                    Int32(poseFeatureCount), &feature, 1, 0, &out, 1)
        return (0..<vertexCount).map { SIMD3(Double(out[$0 * 3]), Double(out[$0 * 3 + 1]), Double(out[$0 * 3 + 2])) }
    }

    /// Posed positions of a subset of vertices, with pose correctives supplied (they barely change
    /// during a refinement, so callers compute them once). Cheap enough for finite-difference Jacobians.
    func skinned(_ indices: [Int], kinematics k: Kinematics, betas: [Double], correctives: [SIMD3<Double>]?,
                 translation: SIMD3<Double>) -> [SIMD3<Double>] {
        let B = min(betas.count, betaCount)
        return indices.map { i in
            var v = SIMD3(Double(vTemplate[i * 3]), Double(vTemplate[i * 3 + 1]), Double(vTemplate[i * 3 + 2]))
            if let correctives { v += correctives[i] }
            for d in 0..<3 {
                var s = 0.0
                for b in 0..<B { s += Double(shapedirs[(i * 3 + d) * betaCount + b]) * betas[b] }
                v[d] += s
            }
            var out = translation
            for (j, w) in sparseWeights[i] { out += w * (k.rotations[j] * (v - k.restJoints[j]) + k.joints[j]) }
            return out
        }
    }

    /// Full forward pass. Returns vertex positions with `translation` applied.
    public func vertices(pose: [SIMD3<Double>], betas: [Double], translation: SIMD3<Double> = .zero) -> [SIMD3<Float>] {
        let V = vertexCount, J = jointCount, B = betaCount
        let k = kinematics(pose: pose, betas: betas)

        // v_shaped = v_template + shapedirs · β
        var shaped = vTemplate
        var fBetas = (0..<B).map { $0 < betas.count ? Float(betas[$0]) : 0 }
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(V * 3), Int32(B), 1, shapedirs, Int32(B), &fBetas, 1, 1, &shaped, 1)

        // + pose correctives: posedirs · (R_j - I), row-major flattening, joints 1..J-1
        if let posedirs {
            var feature = [Float](repeating: 0, count: poseFeatureCount)
            for j in 1..<J {
                let r = rodrigues(pose[j])
                for row in 0..<3 {
                    for col in 0..<3 { feature[(j - 1) * 9 + row * 3 + col] = Float(r[col][row] - (row == col ? 1 : 0)) }
                }
            }
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(V * 3), Int32(poseFeatureCount), 1, posedirs,
                        Int32(poseFeatureCount), &feature, 1, 1, &shaped, 1)
        }

        // Skinning transforms A_j = [R_j | t_j - R_j·J_rest_j] as J x 12 row-major.
        var A = [Float](repeating: 0, count: J * 12)
        for j in 0..<J {
            let R = k.rotations[j]
            let t = k.joints[j] - R * k.restJoints[j] + translation
            for row in 0..<3 {
                for col in 0..<3 { A[j * 12 + row * 4 + col] = Float(R[col][row]) }
                A[j * 12 + row * 4 + 3] = Float(t[row])
            }
        }
        // Per-vertex blended transforms T = W · A  (V x 12)
        var T = [Float](repeating: 0, count: V * 12)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(V), 12, Int32(J), 1, weights, Int32(J), A, 12, 0, &T, 12)

        var out = [SIMD3<Float>](repeating: .zero, count: V)
        T.withUnsafeBufferPointer { t in
            shaped.withUnsafeBufferPointer { s in
                for v in 0..<V {
                    let x = s[v * 3], y = s[v * 3 + 1], z = s[v * 3 + 2]
                    let m = v * 12
                    out[v] = SIMD3(t[m] * x + t[m + 1] * y + t[m + 2] * z + t[m + 3],
                                   t[m + 4] * x + t[m + 5] * y + t[m + 6] * z + t[m + 7],
                                   t[m + 8] * x + t[m + 9] * y + t[m + 10] * z + t[m + 11])
                }
            }
        }
        return out
    }
}

// MARK: - Rotation helpers

@inline(__always)
func skew(_ v: SIMD3<Double>) -> simd_double3x3 {
    simd_double3x3(rows: [SIMD3(0, -v.z, v.y), SIMD3(v.z, 0, -v.x), SIMD3(-v.y, v.x, 0)])
}

/// Axis-angle → rotation matrix.
func rodrigues(_ r: SIMD3<Double>) -> simd_double3x3 {
    let theta = simd_length(r)
    if theta < 1e-10 { return matrix_identity_double3x3 + skew(r) }
    let K = skew(r / theta)
    return matrix_identity_double3x3 + sin(theta) * K + (1 - cos(theta)) * (K * K)
}

/// Rotation matrix → axis-angle.
func axisAngle(_ R: simd_double3x3) -> SIMD3<Double> {
    let q = simd_quatd(R)
    let angle = q.angle
    if angle < 1e-10 { return .zero }
    var a = q.axis * angle
    if angle > .pi { a = q.axis * (angle - 2 * .pi) }
    return a
}
