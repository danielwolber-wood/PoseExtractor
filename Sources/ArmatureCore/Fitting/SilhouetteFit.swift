import Foundation
import simd

extension BodyFitter {
    /// Stage 3: refines shape, placement and trunk/limb pose so the body's outline matches the
    /// person's segmentation mask, ICP-style: each round picks the body's occluding-contour vertices
    /// and pairs mask-outline points with their nearest contour vertex, then runs Levenberg–Marquardt
    /// with those fixed. Keypoints and the pose/shape priors stay in the objective.
    ///
    /// Clothing and hair only ever make the mask *bigger* than the body, so the terms are asymmetric:
    /// the body's contour is pushed inside the mask, and a weaker, tightly capped term pulls it out
    /// to the mask's outline (overhang beyond the cap — a loose dress, hair — is ignored).
    ///
    /// - Returns: overlap with the mask before and after. The update is kept only if less of the body
    ///   lies outside the mask and IoU hasn't collapsed (a guard against the body shrinking away).
    func fitSilhouette(_ x: inout [Double], mask: PersonMask, keypoints obs2: [SIMD2<Double>],
                       keypointWeights kw: [Double], f: Double, c: SIMD2<Double>, imageSize: SIMD2<Double>,
                       personPx: Double, prior: ([Double], inout [Double]) -> Void)
        -> (before: SilhouetteOverlap, after: SilhouetteOverlap) {
        let B = model.betaCount
        func pose(_ x: [Double]) -> [SIMD3<Double>] {
            (0..<model.jointCount).map { SIMD3(x[$0 * 3], x[$0 * 3 + 1], x[$0 * 3 + 2]) }
        }
        func betas(_ x: [Double]) -> [Double] { Array(x[betaOffset..<betaOffset + B]) }
        func translation(_ x: [Double]) -> SIMD3<Double> { SIMD3(x[transOffset], x[transOffset + 1], x[transOffset + 2]) }
        func project(_ v: SIMD3<Double>) -> SIMD2<Double> {
            let z = max(v.z, 0.05)
            return SIMD2(f * v.x / z, f * v.y / z) + c
        }

        let correctives = model.poseCorrectives(pose: pose(x))
        let allVertices = Array(0..<model.vertexCount)
        func vertices(_ x: [Double], _ indices: [Int]) -> [SIMD3<Double>] {
            let p = pose(x), b = betas(x)
            return model.skinned(indices, kinematics: model.kinematics(pose: p, betas: b), betas: b,
                                 correctives: correctives, translation: translation(x))
        }
        func overlap(_ x: [Double]) -> SilhouetteOverlap {
            mask.overlap(with: mask.rasterize(vertices(x, allVertices).map(project), faces: model.faces))
        }

        /// Vertices on the occluding contour (surface grazing the view ray) that project into the photo.
        func contourVertices(_ verts: [SIMD3<Double>]) -> [Int] {
            var normals = [SIMD3<Double>](repeating: .zero, count: verts.count)
            let faces = model.faces
            for t in stride(from: 0, to: faces.count, by: 3) {
                let a = Int(faces[t]), b = Int(faces[t + 1]), cc = Int(faces[t + 2])
                let n = simd_cross(verts[b] - verts[a], verts[cc] - verts[a])
                normals[a] += n; normals[b] += n; normals[cc] += n
            }
            return verts.indices.filter { i in
                let v = verts[i]
                guard v.z > 0.1 else { return false }
                let p = project(v)
                guard p.x >= 0, p.y >= 0, p.x < imageSize.x, p.y < imageSize.y else { return false }
                return abs(simd_dot(simd_normalize(normals[i]), simd_normalize(v))) < 0.3
            }
        }

        let sigma = 0.01 * personPx
        let capInside = 0.08 * personPx, capCover = 0.03 * personPx
        let active = model.silhouetteJoints.flatMap { j in (0..<3).map { j * 3 + $0 } }
            + Array(betaOffset..<betaOffset + B) + Array(transOffset..<transOffset + 3)

        let x0 = x
        let before = overlap(x)
        for _ in 0..<3 {
            let all = vertices(x, allVertices)
            let rimAll = contourVertices(all)
            let rim = stride(from: 0, to: rimAll.count, by: max(1, rimAll.count / 700)).map { rimAll[$0] }
            guard !rim.isEmpty else { break }

            // Pair each mask-outline point with its nearest contour vertex (held fixed this round).
            let proj = rim.map { project(all[$0]) }
            var pairs: [(point: SIMD2<Double>, slot: Int)] = []
            for q in mask.contour {
                var best = 0, bestD = Double.infinity
                for (k, p) in proj.enumerated() {
                    let d = simd_distance_squared(p, q)
                    if d < bestD { bestD = d; best = k }
                }
                if bestD.squareRoot() < capCover { pairs.append((q, best)) }
            }
            let kIn = sqrt(12.0 / Double(rim.count)) / sigma
            let kCov = sqrt(5.0 / Double(max(mask.contour.count, 1))) / sigma

            x = LevenbergMarquardt.minimize(x, active: active, iterations: 8) { x in
                var r: [Double] = []
                r.reserveCapacity(rim.count + pairs.count + 200)
                let p = vertices(x, rim).map(project)
                // Saturating (not dropping) beyond the cap, so points can't escape the term by moving further out.
                for q in p { r.append(kIn * min(mask.distance(at: q).0, capInside)) }
                for (q, slot) in pairs { r.append(kCov * min(simd_distance(p[slot], q), capCover)) }
                let t = translation(x)
                for (i, j) in targetPoints(x).points.enumerated() where kw[i] > 0 {
                    let e = (project(j + t) - obs2[i]) * kw[i]
                    r += [e.x, e.y]
                }
                prior(x, &r)
                return r
            }
        }

        let after = overlap(x)
        if let path = ProcessInfo.processInfo.environment["ARMATURE_DEBUG_MASK"] {
            // Debug aid: mask (grey), body before (red) and after (green), as a PPM.
            let r0 = mask.rasterize(vertices(x0, allVertices).map(project), faces: model.faces)
            let r1 = mask.rasterize(vertices(x, allVertices).map(project), faces: model.faces)
            var ppm = Data("P6 \(mask.width) \(mask.height) 255\n".utf8)
            for i in 0..<mask.inside.count {
                let m: UInt8 = mask.inside[i] == 1 ? 110 : 20
                ppm += [r0[i] == 1 ? 230 : m, r1[i] == 1 ? 230 : m, m]
            }
            try? ppm.write(to: URL(fileURLWithPath: path))
        }
        guard after.outside < before.outside, after.iou > before.iou - 0.05 else {
            x = x0
            return (before, before)
        }
        return (before, after)
    }
}
