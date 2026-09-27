import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import Vision
import simd

/// Keypoints the app detects: Vision's 17 3D body joints, plus 2D-only face and hand keypoints.
/// Body models map these to their own joints and surface points (see tools/convert_models.py).
public enum BodyJoint: Int, CaseIterable, Codable, Sendable {
    case root, leftHip, leftKnee, leftAnkle, rightHip, rightKnee, rightAnkle, spine, centerShoulder,
         centerHead, topHead, leftShoulder, leftElbow, leftWrist, rightShoulder, rightElbow, rightWrist
    // 2D-only keypoints. Face: nose/eyes/ears from the body pose request, mouth and chin from face landmarks.
    case nose, leftEye, rightEye, leftEar, rightEar, mouthLeft, mouthRight, chin
    // Hands: all 20 non-wrist keypoints of Vision's hand pose request (models with fingers use them all;
    // SMPL, which has none, uses the index/middle/little knuckles to orient the hand).
    case leftThumbCMC, leftThumbMP, leftThumbIP, leftThumbTip,
         leftIndexMCP, leftIndexPIP, leftIndexDIP, leftIndexTip,
         leftMiddleMCP, leftMiddlePIP, leftMiddleDIP, leftMiddleTip,
         leftRingMCP, leftRingPIP, leftRingDIP, leftRingTip,
         leftLittleMCP, leftLittlePIP, leftLittleDIP, leftLittleTip
    case rightThumbCMC, rightThumbMP, rightThumbIP, rightThumbTip,
         rightIndexMCP, rightIndexPIP, rightIndexDIP, rightIndexTip,
         rightMiddleMCP, rightMiddlePIP, rightMiddleDIP, rightMiddleTip,
         rightRingMCP, rightRingPIP, rightRingDIP, rightRingTip,
         rightLittleMCP, rightLittlePIP, rightLittleDIP, rightLittleTip

    /// Keypoints with no 3D estimate from Vision.
    public var is2DOnly: Bool { rawValue >= BodyJoint.nose.rawValue }
    public var isFace: Bool { is2DOnly && rawValue <= BodyJoint.chin.rawValue }
    public var isHand: Bool { rawValue >= BodyJoint.leftThumbCMC.rawValue }
    var isLeftHand: Bool { isHand && rawValue < BodyJoint.rightThumbCMC.rawValue }

    /// Hand keypoints in Vision's hand pose request, per side (same order as the enum cases).
    static let handJoints: [(vision: VNHumanHandPoseObservation.JointName, left: BodyJoint, right: BodyJoint)] = {
        let names: [VNHumanHandPoseObservation.JointName] = [
            .thumbCMC, .thumbMP, .thumbIP, .thumbTip, .indexMCP, .indexPIP, .indexDIP, .indexTip,
            .middleMCP, .middlePIP, .middleDIP, .middleTip, .ringMCP, .ringPIP, .ringDIP, .ringTip,
            .littleMCP, .littlePIP, .littleDIP, .littleTip,
        ]
        return names.enumerated().map { i, n in
            (n, BodyJoint(rawValue: BodyJoint.leftThumbCMC.rawValue + i)!, BodyJoint(rawValue: BodyJoint.rightThumbCMC.rawValue + i)!)
        }
    }()

    /// Name in Vision's 3D request (nil for 2D-only keypoints).
    var visionName: VNHumanBodyPose3DObservation.JointName? {
        switch self {
        case .root: .root
        case .leftHip: .leftHip
        case .leftKnee: .leftKnee
        case .leftAnkle: .leftAnkle
        case .rightHip: .rightHip
        case .rightKnee: .rightKnee
        case .rightAnkle: .rightAnkle
        case .spine: .spine
        case .centerShoulder: .centerShoulder
        case .centerHead: .centerHead
        case .topHead: .topHead
        case .leftShoulder: .leftShoulder
        case .leftElbow: .leftElbow
        case .leftWrist: .leftWrist
        case .rightShoulder: .rightShoulder
        case .rightElbow: .rightElbow
        case .rightWrist: .rightWrist
        default: nil
        }
    }

    /// Equivalent joint in Vision's (more accurate) 2D body pose request, where one exists.
    var vision2DName: VNHumanBodyPoseObservation.JointName? {
        switch self {
        case .root: .root
        case .leftHip: .leftHip
        case .leftKnee: .leftKnee
        case .leftAnkle: .leftAnkle
        case .rightHip: .rightHip
        case .rightKnee: .rightKnee
        case .rightAnkle: .rightAnkle
        case .centerShoulder: .neck
        case .leftShoulder: .leftShoulder
        case .leftElbow: .leftElbow
        case .leftWrist: .leftWrist
        case .rightShoulder: .rightShoulder
        case .rightElbow: .rightElbow
        case .rightWrist: .rightWrist
        case .nose: .nose
        case .leftEye: .leftEye
        case .rightEye: .rightEye
        case .leftEar: .leftEar
        case .rightEar: .rightEar
        default: nil
        }
    }

    public static let bones: [(BodyJoint, BodyJoint)] = {
        var b: [(BodyJoint, BodyJoint)] = [
            (.root, .leftHip), (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
            (.root, .rightHip), (.rightHip, .rightKnee), (.rightKnee, .rightAnkle),
            (.root, .spine), (.spine, .centerShoulder), (.centerShoulder, .centerHead), (.centerHead, .topHead),
            (.centerShoulder, .leftShoulder), (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
            (.centerShoulder, .rightShoulder), (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
            (.nose, .leftEye), (.nose, .rightEye), (.leftEye, .leftEar), (.rightEye, .rightEar),
            (.mouthLeft, .mouthRight), (.mouthLeft, .chin), (.mouthRight, .chin),
        ]
        // Each finger: wrist → base → ... → tip; plus the knuckle line across the palm.
        for (wrist, first) in [(BodyJoint.leftWrist, BodyJoint.leftThumbCMC), (.rightWrist, .rightThumbCMC)] {
            for finger in 0..<5 {
                let base = first.rawValue + finger * 4
                b.append((wrist, BodyJoint(rawValue: base)!))
                for k in 0..<3 { b.append((BodyJoint(rawValue: base + k)!, BodyJoint(rawValue: base + k + 1)!)) }
            }
            for finger in 1..<4 {
                b.append((BodyJoint(rawValue: first.rawValue + finger * 4)!, BodyJoint(rawValue: first.rawValue + (finger + 1) * 4)!))
            }
        }
        return b
    }()
}

/// One detected person, in the full (upright) image's coordinate system.
public struct DetectedPerson: Sendable {
    /// Joint positions relative to the camera, in metres. Computer-vision convention:
    /// x right, y down, z forward (into the scene). Indexed by `BodyJoint.rawValue`.
    public var joints3D: [SIMD3<Double>]
    /// Joint projections in image pixels, top-left origin. Indexed by `BodyJoint.rawValue`.
    public var joints2D: [SIMD2<Double>]
    /// Confidence of each 2D keypoint (0...1). Joints seen by Vision's dedicated 2D pose request carry
    /// its confidence; the rest fall back to the 3D request's projection with a fixed low confidence.
    public var confidence2D: [Double]
    /// How far to trust Vision's 3D estimate of each joint (0...1). Vision invents plausible positions for
    /// joints it can't see (out of frame, occluded); those are down-weighted so the pose prior takes over.
    public var confidence3D: [Double]
    public var bodyHeight: Double
    public var heightIsMeasured: Bool
    public var confidence: Float
    /// Region (pixels, top-left origin) the pose was estimated from.
    public var region: CGRect
    /// Joints the user has dragged. The fitter trusts their 2D position and ignores Vision's 3D guess for them.
    public var edited = [Bool](repeating: false, count: BodyJoint.allCases.count)
    /// The person's segmentation mask, if Vision found one. Used to fit body shape to the silhouette.
    public var silhouette: PersonMask?
    /// Tight person box from the human detector (image pixels, top-left origin); the body crop for age models.
    public var bodyBox: CGRect?
    /// This person's face box (image pixels, top-left origin), if Vision found their face.
    public var faceBox: CGRect?
    /// Age from an age-estimation model (see `AgeEstimator`), if one ran.
    public var estimatedAge: AgeEstimate?
    /// Age set by the user, in years; overrides the estimate.
    public var ageOverride: Double?

    /// The age the fit is conditioned on — the user's, else the estimate — with its uncertainty (1σ, years).
    /// A stated age is trusted to ±1.5 years; estimates to roughly their typical error.
    public var targetAge: (years: Double, sigma: Double)? {
        if let a = ageOverride { return (a, 1.5) }
        if let e = estimatedAge { return (e.years, max(3, 0.12 * e.years)) }
        return nil
    }

    public var isEdited: Bool { edited.contains(true) }

    /// Whether a keypoint has any image evidence (face points Vision didn't find have none).
    public func isVisible(_ joint: BodyJoint) -> Bool {
        !joint.is2DOnly || edited[joint.rawValue] || confidence2D[joint.rawValue] > 0
    }

    /// Whether a keypoint is only Vision's guess (e.g. out of frame or hidden), not seen by the 2D detector.
    public func isGuessed(_ joint: BodyJoint) -> Bool {
        !edited[joint.rawValue] && confidence2D[joint.rawValue] <= 0.05
    }

    /// Moves a joint's 2D keypoint (image pixels) and marks it as user-edited.
    public mutating func move(_ joint: BodyJoint, to point: SIMD2<Double>) {
        // A hand follows its wrist.
        let hand = joint == .leftWrist ? BodyJoint.handJoints.map(\.left)
            : joint == .rightWrist ? BodyJoint.handJoints.map(\.right) : []
        let delta = point - joints2D[joint.rawValue]
        for k in hand where isVisible(k) { joints2D[k.rawValue] += delta }
        joints2D[joint.rawValue] = point
        confidence2D[joint.rawValue] = 1
        edited[joint.rawValue] = true
    }

    /// Fixes Vision's most common failure: the person's left and right labelled the wrong way round.
    public mutating func swapLeftRight() {
        let pairs: [(BodyJoint, BodyJoint)] = [(.leftHip, .rightHip), (.leftKnee, .rightKnee), (.leftAnkle, .rightAnkle),
                                               (.leftShoulder, .rightShoulder), (.leftElbow, .rightElbow),
                                               (.leftWrist, .rightWrist)]
            + BodyJoint.handJoints.map { ($0.left, $0.right) }
        for (a, b) in pairs {
            joints2D.swapAt(a.rawValue, b.rawValue)
            joints3D.swapAt(a.rawValue, b.rawValue)
            confidence2D.swapAt(a.rawValue, b.rawValue)
            confidence3D.swapAt(a.rawValue, b.rawValue)
            edited.swapAt(a.rawValue, b.rawValue)
        }
    }
}

public struct LoadedImage: @unchecked Sendable {
    public let cgImage: CGImage
    /// Focal length in pixels, from EXIF when available.
    public let focalLengthPixels: Double
    public let focalFromEXIF: Bool
    /// The photo's embedded depth map, if it has one (Portrait mode, LiDAR, TrueDepth).
    public let depth: DepthMap?
    public var width: Int { cgImage.width }
    public var height: Int { cgImage.height }

    /// Loads an image, bakes in its EXIF orientation and estimates the camera focal length.
    /// - Parameter focalLength35mm: overrides EXIF (35 mm-equivalent focal length in mm).
    public init(url: URL, focalLength35mm override: Double? = nil) throws {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let raw = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw CocoaError(.fileReadCorruptFile, userInfo: [NSFilePathErrorKey: url.path])
        }
        let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any] ?? [:]
        let orientation = (props[kCGImagePropertyOrientation] as? UInt32).flatMap(CGImagePropertyOrientation.init) ?? .up
        let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any] ?? [:]
        let f35 = exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? Double
        try self.init(cgImage: raw, orientation: orientation, focalLength35mm: override ?? f35,
                      depth: DepthMap(source: src, orientation: orientation))
    }

    public init(cgImage raw: CGImage, orientation: CGImagePropertyOrientation = .up, focalLength35mm: Double? = nil,
                depth: DepthMap? = nil) throws {
        cgImage = orientation == .up ? raw : try Self.upright(raw, orientation)
        self.depth = depth
        let longSide = Double(max(cgImage.width, cgImage.height))
        if let f35 = focalLength35mm, f35 > 0 {
            focalLengthPixels = f35 / 36.0 * longSide
            focalFromEXIF = true
        } else {
            // No metadata: assume a 50 mm-equivalent lens. Photos without EXIF are mostly edited or
            // professional shots, which skew longer than a phone's ~26 mm; 50 mm is a safe middle.
            focalLengthPixels = 50.0 / 36.0 * longSide
            focalFromEXIF = false
        }
    }

    private static func upright(_ image: CGImage, _ orientation: CGImagePropertyOrientation) throws -> CGImage {
        let ci = CIImage(cgImage: image).oriented(orientation)
        guard let out = CIContext().createCGImage(ci, from: ci.extent) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return out
    }
}

/// Finds people with Vision and estimates a 3D pose for each one.
///
/// Vision's 3D pose request is tuned for a single prominent subject, so we first find people
/// with the human-rectangles detector and run the 3D request on a padded crop around each one.
public struct PoseDetector: Sendable {
    public var maxPeople = 8
    public var minPersonHeightFraction = 0.12
    /// Also segment each person (needed for silhouette fitting).
    public var segmentPeople = true
    /// Tell Vision the camera intrinsics for each crop (focal length, and the principal point offset
    /// by the crop). Without it, Vision's 3D pose assumes a camera centred on the crop.
    public var passIntrinsics = true

    public init() {}

    public func detect(in image: LoadedImage) throws -> [DetectedPerson] {
        let W = Double(image.width), H = Double(image.height)
        let rectRequest = VNDetectHumanRectanglesRequest()
        rectRequest.upperBodyOnly = false
        try VNImageRequestHandler(cgImage: image.cgImage, options: [:]).perform([rectRequest])

        var regions: [CGRect] = (rectRequest.results ?? [])
            .sorted { $0.confidence > $1.confidence }
            .map { obs in
                let b = obs.boundingBox  // normalised, lower-left origin
                return CGRect(x: b.minX * W, y: (1 - b.maxY) * H, width: b.width * W, height: b.height * H)
            }
            .filter { $0.height >= minPersonHeightFraction * H }
        regions = Array(regions.prefix(maxPeople))

        var people: [DetectedPerson] = []
        for region in regions {
            if var p = try estimate(in: image, region: padded(region, W: W, H: H)) {
                p.bodyBox = region
                people.append(p)
            }
        }
        if people.isEmpty, let p = try estimate(in: image, region: CGRect(x: 0, y: 0, width: W, height: H)) {
            people.append(p)
        }
        people = dedupe(people)
        if segmentPeople, !people.isEmpty, let masks = try? PersonSegmentation.masks(for: image, people: people) {
            for i in people.indices { people[i].silhouette = masks[i] }
        }
        return people
    }

    private func padded(_ r: CGRect, W: Double, H: Double) -> CGRect {
        let pad = 0.2 * max(r.width, r.height)
        return r.insetBy(dx: -pad, dy: -pad).intersection(CGRect(x: 0, y: 0, width: W, height: H)).integral
    }

    private func estimate(in image: LoadedImage, region: CGRect) throws -> DetectedPerson? {
        guard let crop = image.cgImage.cropping(to: region) else { return nil }
        let request = VNDetectHumanBodyPose3DRequest()
        let request2D = VNDetectHumanBodyPoseRequest()
        let handRequest = VNDetectHumanHandPoseRequest()
        handRequest.maximumHandCount = 4
        let faceRequest = VNDetectFaceLandmarksRequest()
        var options: [VNImageOption: Any] = [:]
        if passIntrinsics {
            // Column-major, principal point relative to the crop's top-left corner.
            let f = Float(image.focalLengthPixels)
            let K = simd_float3x3(columns: (SIMD3(f, 0, 0), SIMD3(0, f, 0),
                                            SIMD3(Float(Double(image.width) / 2 - region.minX),
                                                  Float(Double(image.height) / 2 - region.minY), 1)))
            options[.cameraIntrinsics] = withUnsafeBytes(of: K) { Data($0) } as NSData
        }
        // Depth maps are deliberately not passed to Vision (VNImageRequestHandler(cvPixelBuffer:depthData:)):
        // with depth lacking camera calibration its 3D pose refiner hits an internal assertion that kills
        // the process. The fitter uses the depth map directly instead.
        try VNImageRequestHandler(cgImage: crop, options: options).perform([request, request2D, handRequest, faceRequest])
        guard let results = request.results, !results.isEmpty else { return nil }

        // If the crop contains several people, keep the one closest to the crop centre.
        func centreDistance(_ o: VNHumanBodyPose3DObservation) -> Double {
            guard let p = try? o.pointInImage(.root) else { return .infinity }
            return hypot(p.x - 0.5, p.y - 0.5)
        }
        guard let obs = results.min(by: { centreDistance($0) < centreDistance($1) }) else { return nil }

        var j3 = [SIMD3<Double>](repeating: .zero, count: BodyJoint.allCases.count)
        var j2 = [SIMD2<Double>](repeating: .zero, count: BodyJoint.allCases.count)
        for joint in BodyJoint.allCases {
            guard let visionName = joint.visionName else {
                // 2D-only keypoints: placeholder at the head / wrist until a detector supplies them.
                let anchor: BodyJoint = joint.isFace ? .centerHead : joint.isLeftHand ? .leftWrist : .rightWrist
                j3[joint.rawValue] = j3[anchor.rawValue]
                j2[joint.rawValue] = j2[anchor.rawValue]
                continue
            }
            let m = try obs.cameraRelativePosition(visionName)
            // Vision's camera space is y-up / looking down -z; convert to x-right, y-down, z-forward.
            let t = m.columns.3
            j3[joint.rawValue] = SIMD3(Double(t.x), -Double(t.y), -Double(t.z))
            let p = try obs.pointInImage(visionName)
            j2[joint.rawValue] = SIMD2(region.minX + p.x * region.width, region.minY + (1 - p.y) * region.height)
        }
        // Refine 2D keypoints with the 2D request's observation that best matches this body.
        var conf = [Double](repeating: 0.3, count: BodyJoint.allCases.count)
        var confirmed = [Bool](repeating: false, count: BodyJoint.allCases.count)
        func toImage(_ p: CGPoint) -> SIMD2<Double> {
            SIMD2(region.minX + p.x * region.width, region.minY + (1 - p.y) * region.height)
        }
        let scale = max(simd_distance(j2[BodyJoint.topHead.rawValue], j2[BodyJoint.root.rawValue]), 1)
        let match = (request2D.results ?? []).map { o -> (VNHumanBodyPoseObservation, Double) in
            let d = BodyJoint.allCases.compactMap { j -> Double? in
                guard let n = j.vision2DName, let p = try? o.recognizedPoint(n), p.confidence > 0.3 else { return nil }
                return simd_distance(toImage(p.location), j2[j.rawValue])
            }
            return (o, d.isEmpty ? .infinity : d.reduce(0, +) / Double(d.count))
        }.min { $0.1 < $1.1 }
        if let (o2, meanDist) = match, meanDist < 0.3 * scale {
            for j in BodyJoint.allCases {
                guard let n = j.vision2DName, let p = try? o2.recognizedPoint(n), p.confidence > 0.2 else { continue }
                j2[j.rawValue] = toImage(p.location)
                conf[j.rawValue] = Double(p.confidence)
                confirmed[j.rawValue] = p.confidence > 0.35
            }
        }

        // Hands: pair each detected hand with the body's nearer wrist.
        var handFor: [BodyJoint: (VNHumanHandPoseObservation, Double)] = [:]  // keyed by wrist
        for hand in handRequest.results ?? [] {
            guard let w = try? hand.recognizedPoint(.wrist), w.confidence > 0.3 else { continue }
            let wp = toImage(w.location)
            for wrist in [BodyJoint.leftWrist, .rightWrist] where confirmed[wrist.rawValue] {
                let d = simd_distance(wp, j2[wrist.rawValue])
                if d < 0.25 * scale, d < (handFor[wrist]?.1 ?? .infinity) { handFor[wrist] = (hand, d) }
            }
        }
        for (wrist, (hand, _)) in handFor {
            for (name, left, right) in BodyJoint.handJoints {
                guard let p = try? hand.recognizedPoint(name), p.confidence > 0.3 else { continue }
                let j = wrist == .leftWrist ? left : right
                j2[j.rawValue] = toImage(p.location)
                conf[j.rawValue] = Double(p.confidence)
                confirmed[j.rawValue] = true
            }
        }

        // This person's face: the detected face containing their nose (or, failing that, head centre).
        let cropSize = CGSize(width: region.width, height: region.height)
        func cropToImage(_ p: CGPoint) -> SIMD2<Double> {
            SIMD2(region.minX + p.x, region.minY + (region.height - p.y))  // lower-left crop px → image px
        }
        let headPoint = confirmed[BodyJoint.nose.rawValue] ? j2[BodyJoint.nose.rawValue] : j2[BodyJoint.centerHead.rawValue]
        var faceBox: CGRect?
        let face = (faceRequest.results ?? []).first { f in
            let b = VNImageRectForNormalizedRect(f.boundingBox, Int(region.width), Int(region.height))
            let tl = cropToImage(CGPoint(x: b.minX, y: b.maxY)), br = cropToImage(CGPoint(x: b.maxX, y: b.minY))
            guard headPoint.x >= tl.x && headPoint.x <= br.x && headPoint.y >= tl.y && headPoint.y <= br.y else { return false }
            faceBox = CGRect(x: tl.x, y: tl.y, width: br.x - tl.x, height: br.y - tl.y)
            return true
        }

        // Face landmarks: mouth corners and chin, which pin down head pitch and roll.
        // Left/right comes from the body pose's eyes, so it can't disagree with them.
        let le = BodyJoint.leftEye.rawValue, re = BodyJoint.rightEye.rawValue
        if confirmed[le], confirmed[re] {
            if let face, let lm = face.landmarks, let lips = lm.outerLips, let contour = lm.faceContour {
                let across = simd_normalize(j2[le] - j2[re])            // towards the person's left
                let eyesMid = (j2[le] + j2[re]) / 2
                let lipPts = lips.pointsInImage(imageSize: cropSize).map(cropToImage)
                let mouthMid = lipPts.reduce(.zero, +) / Double(max(lipPts.count, 1))
                let down = simd_normalize(mouthMid - eyesMid)
                let c = Double(lm.confidence)
                if let l = lipPts.max(by: { simd_dot($0, across) < simd_dot($1, across) }),
                   let r = lipPts.min(by: { simd_dot($0, across) < simd_dot($1, across) }),
                   let chin = contour.pointsInImage(imageSize: cropSize).map(cropToImage)
                       .max(by: { simd_dot($0 - eyesMid, down) < simd_dot($1 - eyesMid, down) }) {
                    for (j, p) in [(BodyJoint.mouthLeft, l), (.mouthRight, r), (.chin, chin)] {
                        j2[j.rawValue] = p
                        conf[j.rawValue] = c
                        confirmed[j.rawValue] = c > 0.3
                    }
                }
            }
        }

        // Trust in Vision's 3D joints: torso and head are inferred reliably from context; limbs only
        // when the 2D detector also sees them; anything projecting outside the photo is a guess.
        var conf3 = [Double](repeating: 1, count: BodyJoint.allCases.count)
        let W = Double(image.width), H = Double(image.height), margin = 0.02 * max(W, H)
        for j in BodyJoint.allCases {
            let limb: Set<BodyJoint> = [.leftElbow, .rightElbow, .leftWrist, .rightWrist,
                                        .leftKnee, .rightKnee, .leftAnkle, .rightAnkle]
            if j.is2DOnly {
                conf3[j.rawValue] = 0
                if !confirmed[j.rawValue] { conf[j.rawValue] = 0 }
            }
            if limb.contains(j) && !confirmed[j.rawValue] {
                conf3[j.rawValue] = 0.03
                conf[j.rawValue] = 0  // the 2D point is just Vision's 3D guess projected
            }
            if [.leftHip, .rightHip].contains(j) && !confirmed[j.rawValue] { conf3[j.rawValue] = 0.6 }
            // When the 2D detector sees the face, it beats Vision's 3D head points.
            if (j == .centerHead || j == .topHead) && confirmed[BodyJoint.nose.rawValue] {
                conf[j.rawValue] = 0.1
                conf3[j.rawValue] = 0.3
            }
            let p = j2[j.rawValue]
            if p.x < -margin || p.y < -margin || p.x > W + margin || p.y > H + margin {
                conf[j.rawValue] = 0
                conf3[j.rawValue] = min(conf3[j.rawValue], 0.02)
            }
        }

        return DetectedPerson(joints3D: j3, joints2D: j2, confidence2D: conf, confidence3D: conf3, bodyHeight: Double(obs.bodyHeight),
                              heightIsMeasured: obs.heightEstimation == .measured,
                              confidence: obs.confidence, region: region, faceBox: faceBox)
    }

    /// Roll of the camera from the photo's horizon (radians, counter-clockwise), if Vision finds one.
    public func horizonAngle(in image: LoadedImage) -> Double? {
        let request = VNDetectHorizonRequest()
        try? VNImageRequestHandler(cgImage: image.cgImage, options: [:]).perform([request])
        guard let h = request.results?.first, h.confidence > 0.3 else { return nil }
        return Double(h.angle)
    }

    /// Overlapping boxes can yield the same person twice; drop near-duplicates.
    private func dedupe(_ people: [DetectedPerson]) -> [DetectedPerson] {
        var kept: [DetectedPerson] = []
        for p in people {
            let scale = max(simd_distance(p.joints2D[BodyJoint.topHead.rawValue], p.joints2D[BodyJoint.root.rawValue]), 1)
            let dup = kept.contains { k in
                let d = zip(p.joints2D, k.joints2D).map { simd_distance($0, $1) }.reduce(0, +) / Double(p.joints2D.count)
                return d < 0.25 * scale
            }
            if !dup { kept.append(p) }
        }
        return kept
    }
}
