import AppKit
import Foundation
import Metal
import SceneKit
import simd

/// A SceneKit scene of clay figures sharing the photo's camera.
///
/// World space is SceneKit's (y up, camera looking down -z) with the photo camera at the origin,
/// so the "photo" view overlays the figures on the people they came from.
public final class ClayScene {
    public enum View: String, CaseIterable, Sendable { case photo, studio }

    public let scene = SCNScene()
    public let photoCamera = SCNNode()
    public let studioCamera = SCNNode()
    public private(set) var bodyNodes: [SCNNode] = []
    public let imageSize: CGSize
    let photo: CGImage
    private let floor = SCNNode()
    private var studioDistance: Float = 5
    private var style: ClayStyle

    /// - Parameter horizonAngle: Vision's horizon angle (the rotation that levels the photo), if found.
    public init(bodies: [FittedBody], meshes: [[SIMD3<Float>]], faces: [UInt32], image: LoadedImage,
                style: ClayStyle = ClayStyle(), horizonAngle: Double? = nil) {
        self.style = style
        photo = image.cgImage
        imageSize = CGSize(width: image.width, height: image.height)

        // CV camera space (x right, y down, z forward) → SceneKit (x right, y up, z backward).
        let meshesSK = meshes.map { $0.map { SIMD3($0.x, -$0.y, -$0.z) } }

        for (i, verts) in meshesSK.enumerated() {
            let node = SCNNode(geometry: Self.geometry(vertices: verts, faces: faces))
            node.name = "person_\(i)"
            node.castsShadow = true
            bodyNodes.append(node)
            scene.rootNode.addChildNode(node)
        }
        applyStyle(style)

        let camera = SCNCamera()
        camera.projectionDirection = .vertical
        camera.fieldOfView = 2 * atan(imageSize.height / 2 / image.focalLengthPixels) * 180 / .pi
        camera.zNear = 0.05
        camera.zFar = 200
        Self.configureLook(camera)
        photoCamera.camera = camera
        photoCamera.name = "photoCamera"
        scene.rootNode.addChildNode(photoCamera)

        setupGroundAndLights(bodies: bodies, meshes: meshesSK, horizonAngle: horizonAngle)
        setView(.photo)
    }

    private static func configureLook(_ camera: SCNCamera) {
        camera.wantsHDR = true
        camera.screenSpaceAmbientOcclusionIntensity = 2.0
        camera.screenSpaceAmbientOcclusionRadius = 0.12
        camera.screenSpaceAmbientOcclusionNormalThreshold = 0.3
        camera.screenSpaceAmbientOcclusionDepthThreshold = 0.1
        camera.wantsExposureAdaptation = false
        camera.exposureOffset = 0
        camera.whitePoint = 1.2
    }

    // MARK: - Geometry

    static func geometry(vertices: [SIMD3<Float>], faces: [UInt32]) -> SCNGeometry {
        var normals = [SIMD3<Float>](repeating: .zero, count: vertices.count)
        for t in stride(from: 0, to: faces.count, by: 3) {
            let a = Int(faces[t]), b = Int(faces[t + 1]), c = Int(faces[t + 2])
            let n = simd_cross(vertices[b] - vertices[a], vertices[c] - vertices[a])  // area-weighted
            normals[a] += n; normals[b] += n; normals[c] += n
        }
        let vSrc = SCNGeometrySource(vertices: vertices.map { SCNVector3($0.x, $0.y, $0.z) })
        let nSrc = SCNGeometrySource(normals: normals.map { let n = simd_normalize($0); return SCNVector3(n.x, n.y, n.z) })
        let element = SCNGeometryElement(indices: faces, primitiveType: .triangles)
        return SCNGeometry(sources: [vSrc, nSrc], elements: [element])
    }

    // MARK: - Style

    public func applyStyle(_ style: ClayStyle) {
        self.style = style
        for i in bodyNodes.indices { styleBody(i) }
    }

    private func styleBody(_ i: Int) {
        let node = bodyNodes[i]
        guard let g = node.geometry else { return }
        // Wireframe shows the actual SMPL topology, so skip subdivision there.
        g.subdivisionLevel = style.finish == .wireframe ? 0 : style.subdivision
        g.wantsAdaptiveSubdivision = false
        let (lo, hi) = node.boundingBox
        let centre = SIMD3<Float>(Float(lo.x + hi.x) / 2, Float(lo.y + hi.y) / 2, Float(lo.z + hi.z) / 2)
        g.materials = [style.makeMaterial(person: i, centre: centre)]
        node.castsShadow = style.finish != .wireframe
    }

    /// Replaces one person's mesh (after an edit + re-fit) without rebuilding the scene.
    public func updateBody(_ index: Int, mesh: [SIMD3<Float>], faces: [UInt32]) {
        guard bodyNodes.indices.contains(index) else { return }
        bodyNodes[index].geometry = Self.geometry(vertices: mesh.map { SIMD3($0.x, -$0.y, -$0.z) }, faces: faces)
        styleBody(index)
    }

    // MARK: - Ground, lights, cameras

    private func setupGroundAndLights(bodies: [FittedBody], meshes: [[SIMD3<Float>]], horizonAngle: Double?) {
        // "Up" = average body up-axis (SMPL +y through each global orientation), in SceneKit space.
        var up = SIMD3<Double>.zero
        for b in bodies {
            let u = rodrigues(b.pose[0]) * SIMD3(0, 1, 0)       // CV space
            up += SIMD3(u.x, -u.y, -u.z)
        }
        up = simd_length(up) > 1e-6 ? simd_normalize(up) : SIMD3(0, 1, 0)
        // The camera's roll: from the horizon if Vision found one (it reports the angle that levels the
        // photo, so a horizon tilted counter-clockwise by θ gives -θ), else assume a level camera.
        let camUp = horizonAngle.map { SIMD3(sin($0), cos($0), 0) } ?? SIMD3<Double>(0, 1, 0)
        if horizonAngle != nil {
            // Bodies lean and sit; the horizon is the better roll estimate. Keep only the bodies'
            // forward/back tilt, which tells us the camera's pitch.
            let side = SIMD3<Double>(camUp.y, -camUp.x, 0)      // camera right, levelled
            up = simd_normalize(up - simd_dot(up, side) * side)
        }
        // Bodies can be tilted (sitting, lying); don't let the ground tip past ~35° from the camera's up.
        if simd_dot(up, camUp) < cos(35 * Double.pi / 180) {
            let axis = simd_normalize(simd_cross(camUp, up) + SIMD3(1e-9, 0, 0))
            up = simd_quatd(angle: 35 * .pi / 180, axis: axis).act(camUp)
        }
        let upF = SIMD3<Float>(up)

        let all = meshes.flatMap { $0 }
        let lowest = all.min { simd_dot($0, upF) < simd_dot($1, upF) } ?? SIMD3(0, -1, -3)
        let centre = all.isEmpty ? SIMD3<Float>(0, 0, -3) : all.reduce(.zero, +) / Float(all.count)
        let radius = all.map { simd_distance($0, centre) }.max() ?? 1
        // Ground point directly under the group centre.
        let groundPoint = centre - simd_dot(centre - lowest, upF) * upF

        let floorGeo = SCNFloor()
        floorGeo.reflectivity = 0
        floorGeo.firstMaterial?.lightingModel = .shadowOnly
        floor.geometry = floorGeo
        floor.simdPosition = groundPoint
        floor.simdOrientation = simd_quatf(from: SIMD3(0, 1, 0), to: upF)
        floor.name = "ground"
        scene.rootNode.addChildNode(floor)

        // Key light: high, from the camera side and slightly left, casting soft shadows.
        let key = SCNLight()
        key.type = .directional
        key.intensity = 1800
        key.color = NSColor(srgbRed: 1.0, green: 0.96, blue: 0.9, alpha: 1)
        key.castsShadow = true
        key.shadowMode = .deferred
        key.shadowSampleCount = 24
        key.shadowRadius = 6
        key.shadowMapSize = CGSize(width: 4096, height: 4096)
        key.shadowColor = NSColor(white: 0, alpha: 0.6)
        // Fit the shadow frustum to the figures; automatic fitting would try to cover the infinite floor.
        key.automaticallyAdjustsShadowProjection = false
        key.orthographicScale = CGFloat(radius * 1.3)
        key.zNear = CGFloat(radius * 0.5)
        key.zFar = CGFloat(radius * 10)
        let keyNode = SCNNode()
        keyNode.light = key
        let side = simd_normalize(simd_cross(upF, SIMD3(0, 0, 1)))
        keyNode.simdPosition = centre + upF * radius * 3 + SIMD3(0, 0, 1) * radius * 2 - side * radius * 1.5
        keyNode.simdLook(at: centre, up: upF, localFront: SIMD3(0, 0, -1))
        scene.rootNode.addChildNode(keyNode)

        let fill = SCNLight()
        fill.type = .directional
        fill.intensity = 250
        fill.color = NSColor(srgbRed: 0.85, green: 0.9, blue: 1.0, alpha: 1)
        let fillNode = SCNNode()
        fillNode.light = fill
        fillNode.simdPosition = centre + upF * radius + SIMD3(0, 0, 1) * radius + side * radius * 2
        fillNode.simdLook(at: centre, up: upF, localFront: SIMD3(0, 0, -1))
        scene.rootNode.addChildNode(fillNode)

        let rim = SCNLight()
        rim.type = .directional
        rim.intensity = 700
        let rimNode = SCNNode()
        rimNode.light = rim
        rimNode.simdPosition = centre + upF * radius * 2 - SIMD3(0, 0, 1) * radius * 2
        rimNode.simdLook(at: centre, up: upF, localFront: SIMD3(0, 0, -1))
        scene.rootNode.addChildNode(rimNode)

        let ambient = SCNLight()
        ambient.type = .ambient
        ambient.intensity = 40
        let ambientNode = SCNNode()
        ambientNode.light = ambient
        scene.rootNode.addChildNode(ambientNode)

        scene.lightingEnvironment.contents = Self.skyGradient()
        scene.lightingEnvironment.intensity = 0.6

        // Studio camera: 30° around the group, a little higher than the photo camera.
        let studio = SCNCamera()
        studio.fieldOfView = 35
        studio.zNear = 0.05
        studio.zFar = 200
        Self.configureLook(studio)
        studioCamera.camera = studio
        studioCamera.name = "studioCamera"
        let toCam = simd_normalize(-centre)
        let dir = simd_normalize(simd_quatf(angle: 0.52, axis: upF).act(toCam) + upF * 0.25)
        let dist = radius / sin(Float(studio.fieldOfView) * .pi / 360) * 0.9
        studioDistance = dist
        studioCamera.simdPosition = centre + dir * dist
        studioCamera.simdLook(at: centre, up: upF, localFront: SIMD3(0, 0, -1))
        scene.rootNode.addChildNode(studioCamera)
    }

    public func setView(_ view: View) {
        switch view {
        case .photo:
            scene.background.contents = photo
            scene.fogEndDistance = 0
            floor.geometry?.firstMaterial?.lightingModel = .shadowOnly
        case .studio:
            // Warm grey sweep: the floor fades into the backdrop through fog, so there's no horizon line.
            let backdrop = NSColor(srgbRed: 0.76, green: 0.74, blue: 0.71, alpha: 1)
            scene.background.contents = backdrop
            scene.fogColor = backdrop
            scene.fogStartDistance = CGFloat(studioDistance * 1.2)
            scene.fogEndDistance = CGFloat(studioDistance * 3)
            let m = floor.geometry?.firstMaterial
            m?.lightingModel = .physicallyBased
            m?.diffuse.contents = NSColor(srgbRed: 0.6, green: 0.58, blue: 0.55, alpha: 1)
            m?.roughness.contents = 1.0
        }
    }

    /// Equirectangular studio environment: sky-to-floor gradient plus a few softboxes for reflections.
    private static func skyGradient() -> CGImage {
        let w = 1024, h = 512
        let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let colors = [NSColor(srgbRed: 0.92, green: 0.93, blue: 0.96, alpha: 1).cgColor,
                      NSColor(srgbRed: 0.70, green: 0.69, blue: 0.67, alpha: 1).cgColor,
                      NSColor(srgbRed: 0.55, green: 0.52, blue: 0.49, alpha: 1).cgColor,
                      NSColor(srgbRed: 0.36, green: 0.34, blue: 0.32, alpha: 1).cgColor] as CFArray
        let grad = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: colors,
                              locations: [0, 0.45, 0.55, 1])!
        ctx.drawLinearGradient(grad, start: CGPoint(x: 0, y: h), end: CGPoint(x: 0, y: 0), options: [])
        // Softboxes (CG origin is bottom-left, so the upper hemisphere is the top of the context).
        ctx.setFillColor(NSColor(white: 1, alpha: 1).cgColor)
        for (x, y, bw, bh) in [(0.18, 0.62, 0.10, 0.2), (0.62, 0.66, 0.16, 0.12), (0.88, 0.58, 0.05, 0.25)] {
            ctx.fill(CGRect(x: x * Double(w), y: y * Double(h), width: bw * Double(w), height: bh * Double(h)))
        }
        return ctx.makeImage()!
    }

    // MARK: - Output

    public enum Background: String, CaseIterable, Sendable { case scene, transparent }

    /// Default output size for a view: the photo's aspect for `.photo`, square for `.studio`.
    public func defaultSize(_ view: View, maxDimension: CGFloat = 2048) -> CGSize {
        switch view {
        case .photo:
            let scale = min(1, maxDimension / max(imageSize.width, imageSize.height))
            return CGSize(width: (imageSize.width * scale).rounded(), height: (imageSize.height * scale).rounded())
        case .studio:
            return CGSize(width: min(maxDimension, 1600), height: min(maxDimension, 1600))
        }
    }

    public func render(_ view: View, maxDimension: CGFloat = 2048) -> CGImage? {
        render(view, size: defaultSize(view, maxDimension: maxDimension))
    }

    /// Renders offscreen with Metal.
    /// - Parameters:
    ///   - pointOfView: camera to render from; defaults to the view's own camera. Pass the live
    ///     view's camera to export exactly what the user has orbited to.
    ///   - background: `.transparent` drops the photo/backdrop but keeps the figures' ground shadows.
    public func render(_ view: View, size: CGSize, background: Background = .scene,
                       pointOfView: SCNNode? = nil) -> CGImage? {
        setView(view)
        defer { setView(view) }
        if background == .transparent {
            scene.background.contents = NSColor.clear
            scene.fogEndDistance = 0
            floor.geometry?.firstMaterial?.lightingModel = .shadowOnly
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        let renderer = SCNRenderer(device: device, options: nil)
        renderer.scene = scene
        renderer.pointOfView = pointOfView ?? (view == .photo ? photoCamera : studioCamera)
        let image = renderer.snapshot(atTime: 0, with: size, antialiasingMode: .multisampling4X)
        return image.cgImage(forProposedRect: nil, context: nil, hints: nil)
    }

    public func exportUSDZ(to url: URL) -> Bool {
        setView(.studio)
        return scene.write(to: url, options: nil, delegate: nil, progressHandler: nil)
    }
}
