import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct ClayResult {
    public let image: LoadedImage
    public internal(set) var people: [DetectedPerson]
    public internal(set) var bodies: [FittedBody]
    public internal(set) var meshes: [[SIMD3<Float>]]   // camera space (CV convention), metres
    public let faces: [UInt32]
    public internal(set) var timings: [(String, TimeInterval)]
    /// Vision's horizon angle (radians; the rotation that levels the photo), if it found a horizon.
    public internal(set) var horizonAngle: Double? = nil

    /// Drops a person (e.g. a false detection).
    public mutating func remove(person index: Int) {
        people.remove(at: index)
        bodies.remove(at: index)
        meshes.remove(at: index)
    }

    public func makeScene(style: ClayStyle = ClayStyle()) -> ClayScene {
        ClayScene(bodies: bodies, meshes: meshes, faces: faces, image: image, style: style, horizonAngle: horizonAngle)
    }
}

/// Image → people → fitted SMPL bodies → meshes.
public final class ClayPipeline: @unchecked Sendable {
    public let modelsDirectory: URL
    private var models: [String: BodyModel] = [:]
    private let lock = NSLock()
    public var detector = PoseDetector()
    /// Refine body shape against each person's segmentation mask.
    public var useSilhouette = true
    /// Age-estimation model id (see `availableAgeModels`), or nil for none. Estimates condition Anny's
    /// shape on age and are shown for every model.
    public var ageModel: String?
    private var ageEstimators: [String: AgeEstimator] = [:]

    /// Converted age-estimation models, in display order.
    public var availableAgeModels: [AgeModelInfo] { AgeModelInfo.available(in: modelsDirectory) }

    public func ageEstimator(_ id: String) throws -> AgeEstimator {
        lock.lock(); defer { lock.unlock() }
        if let e = ageEstimators[id] { return e }
        let e = try AgeEstimator(modelsDirectory: modelsDirectory, id: id)
        ageEstimators[id] = e
        return e
    }

    /// Sets each person's `estimatedAge` with the current age model (clears them when there is none).
    public func estimateAges(_ people: inout [DetectedPerson], image: LoadedImage) {
        let estimator = ageModel.flatMap { try? ageEstimator($0) }
        let everyone = people
        for i in people.indices {
            people[i].estimatedAge = estimator.flatMap { try? $0.estimate(everyone[i], among: everyone, image: image) }
        }
    }

    public init(modelsDirectory: URL) {
        self.modelsDirectory = modelsDirectory
    }

    /// Looks for converted models next to the working directory, the executable, or in Application Support.
    public static func defaultModelsDirectory() -> URL? {
        var candidates: [URL] = []
        if let env = ProcessInfo.processInfo.environment["CLAY_MODELS"] { candidates.append(URL(fileURLWithPath: env)) }
        if let resources = Bundle.main.resourceURL { candidates.append(resources.appendingPathComponent("Models")) }
        candidates.append(URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("Models"))
        var dir = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        for _ in 0..<5 {
            candidates.append(dir.appendingPathComponent("Models"))
            dir.deleteLastPathComponent()
        }
        if let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            candidates.append(support.appendingPathComponent("ClayStudio/Models"))
        }
        return candidates.first {
            !BodyModelInfo.available(in: $0).isEmpty
        }
    }

    /// The default model: SMPL neutral, else whatever is installed first.
    public static let defaultModelID = "smpl_neutral"

    /// Converted body models in the models directory, in display order.
    public var availableModels: [BodyModelInfo] { BodyModelInfo.available(in: modelsDirectory) }

    public func model(_ id: String) throws -> BodyModel {
        lock.lock(); defer { lock.unlock() }
        if let m = models[id] { return m }
        guard FileManager.default.fileExists(atPath: modelsDirectory.appendingPathComponent(id).path) else {
            throw ModelError.unknownModel(id)
        }
        let m = try BodyModel(modelsDirectory: modelsDirectory, id: id)
        models[id] = m
        return m
    }

    /// Ages given by the caller; they override estimates.
    public enum AgeInput: Sendable {
        case none
        case everyone(Double)
        case perPerson([Int: Double])
    }

    public func run(image: LoadedImage, model modelID: String = ClayPipeline.defaultModelID,
                    ages: AgeInput = .none) throws -> ClayResult {
        let t0 = Date()
        var people = try detector.detect(in: image)
        let horizon = detector.horizonAngle(in: image)
        let tDetect = Date().timeIntervalSince(t0)
        let t1 = Date()
        estimateAges(&people, image: image)
        let tAge = Date().timeIntervalSince(t1)
        for i in people.indices {
            switch ages {
            case .none: break
            case .everyone(let a): people[i].ageOverride = a
            case .perPerson(let map): if let a = map[i] { people[i].ageOverride = a }
            }
        }
        var result = try fit(people: people, image: image, model: modelID, horizonAngle: horizon)
        result.timings.insert(("detect + 3D pose", tDetect), at: 0)
        if ageModel != nil { result.timings.insert(("age estimation", tAge), at: 1) }
        return result
    }

    /// Re-fits one person after their detection was edited. Fast enough (~10 ms) to call while dragging.
    /// Pass `silhouette: false` for interactive updates: skips the silhouette stage and warm-starts
    /// from the current fit (a few ms, even for Anny). The full re-fit runs when the edit ends.
    public func refit(_ result: ClayResult, person index: Int, with person: DetectedPerson, model modelID: String,
                      silhouette: Bool = true) throws -> ClayResult {
        let body3D = try model(modelID)
        var out = result
        // Interactive (no silhouette) re-fits warm-start from the current body: a small edit is a small change.
        let body = BodyFitter(model: body3D).fit(person, image: result.image, useSilhouette: silhouette && useSilhouette,
                                                 warmStart: silhouette ? nil : result.bodies[index])
        out.people[index] = person
        out.bodies[index] = body
        out.meshes[index] = body3D.vertices(pose: body.pose, betas: body.betas, translation: body.translation)
        return out
    }

    /// Fits bodies to existing detections (used for re-fitting with a different body model, keeping edits).
    public func fit(people: [DetectedPerson], image: LoadedImage, model modelID: String,
                    horizonAngle: Double? = nil) throws -> ClayResult {
        var timings: [(String, TimeInterval)] = []
        func timed<T>(_ label: String, _ body: () throws -> T) rethrows -> T {
            let t0 = Date()
            defer { timings.append((label, Date().timeIntervalSince(t0))) }
            return try body()
        }
        let body3D = try timed("load model") { try model(modelID) }
        let fitter = BodyFitter(model: body3D)
        let bodies = timed("fit body") {
            // Each fit is independent; run them in parallel.
            var out = [FittedBody?](repeating: nil, count: people.count)
            DispatchQueue.concurrentPerform(iterations: people.count) { i in
                let b = fitter.fit(people[i], image: image, useSilhouette: useSilhouette)
                lock.lock(); out[i] = b; lock.unlock()
            }
            return out.compactMap { $0 }
        }
        let meshes = timed("build meshes") {
            bodies.map { body3D.vertices(pose: $0.pose, betas: $0.betas, translation: $0.translation) }
        }
        return ClayResult(image: image, people: people, bodies: bodies, meshes: meshes, faces: body3D.faces,
                          timings: timings, horizonAngle: horizonAngle)
    }
}

// MARK: - Export

public enum ClayExport {
    /// Wavefront OBJ in a y-up, right-handed frame (camera at origin looking down -z).
    public static func obj(vertices: [SIMD3<Float>], faces: [UInt32]) -> String {
        var s = "# SMPL clay figure (y-up, metres)\n"
        s.reserveCapacity(vertices.count * 40 + faces.count * 20)
        for v in vertices { s += "v \(v.x) \(-v.y) \(-v.z)\n" }
        for t in stride(from: 0, to: faces.count, by: 3) {
            s += "f \(faces[t] + 1) \(faces[t + 1] + 1) \(faces[t + 2] + 1)\n"
        }
        return s
    }

    /// - Parameter pipeline: when given, adds each body's phenotype read-out (Anny: apparent age etc.).
    public static func parametersJSON(_ result: ClayResult, pipeline: ClayPipeline? = nil) throws -> Data {
        struct Person: Encodable {
            let index: Int
            let model: String
            let globalOrient: [Double]
            let bodyPose: [Double]
            let phenotype: [String: Double]?
            let betas: [Double]
            let translation: [Double]
            let rms3DMetres: Double
            let rms2DPixels: Double
            let silhouette: SilhouetteOverlap?
            let keypoints2D: [String: [Double]]
            let keypointConfidence: [String: Double]
            let editedJoints: [String]
            let estimatedAge: AgeEstimate?
            let givenAge: Double?
            let faceBox: [Double]?
            let bodyBox: [Double]?
        }
        struct Output: Encodable {
            let imageWidth: Int
            let imageHeight: Int
            let focalLengthPixels: Double
            let focalFromEXIF: Bool
            let coordinateSystem: String
            let people: [Person]
        }
        let people = zip(result.people, result.bodies).enumerated().map { i, pb in
            let (p, b) = pb
            return Person(
                index: i, model: b.model,
                globalOrient: b.pose[0].scalars,
                bodyPose: b.pose.dropFirst().flatMap(\.scalars),
                phenotype: (try? pipeline?.model(b.model))??.phenotype(betas: b.betas),
                betas: b.betas, translation: b.translation.scalars,
                rms3DMetres: b.rms3D, rms2DPixels: b.rms2D, silhouette: b.silhouette,
                keypoints2D: Dictionary(uniqueKeysWithValues: BodyJoint.allCases.map {
                    ("\($0)", [p.joints2D[$0.rawValue].x, p.joints2D[$0.rawValue].y])
                }),
                keypointConfidence: Dictionary(uniqueKeysWithValues: BodyJoint.allCases.map {
                    ("\($0)", p.confidence2D[$0.rawValue])
                }),
                editedJoints: BodyJoint.allCases.filter { p.edited[$0.rawValue] }.map { "\($0)" },
                estimatedAge: p.estimatedAge, givenAge: p.ageOverride,
                faceBox: p.faceBox.map { [$0.minX, $0.minY, $0.width, $0.height] },
                bodyBox: p.bodyBox.map { [$0.minX, $0.minY, $0.width, $0.height] })
        }
        let out = Output(imageWidth: result.image.width, imageHeight: result.image.height,
                         focalLengthPixels: result.image.focalLengthPixels, focalFromEXIF: result.image.focalFromEXIF,
                         coordinateSystem: "OpenCV camera: x right, y down, z forward; SMPL axis-angle",
                         people: people)
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try enc.encode(out)
    }

    public static func writePNG(_ image: CGImage, to url: URL) throws {
        try writeImage(image, to: url, type: .png)
    }

    /// Writes PNG, JPEG or HEIC (anything ImageIO can encode).
    public static func writeImage(_ image: CGImage, to url: URL, type: UTType, quality: Double = 0.92) throws {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: url.path])
        }
        CGImageDestinationAddImage(dest, image, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }
}
