import Foundation

/// A depth model found on disk.
public struct DepthModelLocation: Equatable, Sendable {
    public let backend: MonocularDepthBackend
    /// The .mlpackage, .mlmodel or compiled .mlmodelc.
    public let modelURL: URL
    /// Optional `depth.json` next to the model, overriding the backend's built-in I/O contract.
    public let manifestURL: URL?
}

/// Finds depth models. Layout, under any models directory:
///
///     Models/depth/depth-anything-v2-small/DepthAnythingV2SmallF16.mlpackage   (Apple's download)
///     Models/depth/depth-pro/model.mlpackage                                   (tools/convert_depth_models.py)
///
/// Each backend directory may also hold a `depth.json` manifest. Float16 variants are preferred
/// over Float32, and a compiled `.mlmodelc` over its package. Apple's Depth Anything file names are
/// also accepted directly in `Models/depth/`.
public enum DepthModelLocator {
    /// File names tried, in order, inside the backend's directory (and then `Models/depth/`).
    public static func candidateNames(for backend: MonocularDepthBackend) -> [String] {
        let stems: [String]
        switch backend {
        case .depthAnythingV2Small:
            stems = ["model", "DepthAnythingV2SmallF16", "DepthAnythingV2SmallF16P6", "DepthAnythingV2SmallF32"]
        case .depthPro:
            stems = ["model", "DepthProF16", "DepthPro"]
        }
        return stems.flatMap { ["\($0).mlmodelc", "\($0).mlpackage", "\($0).mlmodel"] }
    }

    /// Directories searched, in priority order, de-duplicated: `ARMATURE_MODELS` (or `CLAY_MODELS`), the pipeline's models
    /// directory, the app bundle, ./Models, Models next to (or above) the executable, and
    /// ~/Library/Application Support/Armature/Models (then ClayStudio/Models) — where large optional models can live
    /// without being bundled into the app.
    public static func searchRoots(modelsDirectory: URL?,
                                   environment: [String: String] = ProcessInfo.processInfo.environment,
                                   bundleResources: URL? = Bundle.main.resourceURL,
                                   currentDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
                                   executable: URL? = CommandLine.arguments.first.map { URL(fileURLWithPath: $0) },
                                   applicationSupport: URL? = FileManager.default.urls(for: .applicationSupportDirectory,
                                                                                       in: .userDomainMask).first)
        -> [URL] {
        var roots: [URL] = []
        if let env = ModelLocations.environmentDirectory(environment) { roots.append(env) }
        if let modelsDirectory { roots.append(modelsDirectory) }
        if let bundleResources { roots.append(bundleResources.appendingPathComponent("Models")) }
        roots.append(currentDirectory.appendingPathComponent("Models"))
        if let executable {
            var dir = executable.resolvingSymlinksInPath().deletingLastPathComponent()
            for _ in 0..<5 {
                roots.append(dir.appendingPathComponent("Models"))
                dir.deleteLastPathComponent()
            }
        }
        roots += ModelLocations.applicationSupportDirectories(applicationSupport)
        var seen = Set<String>()
        return roots.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }

    /// The first model for `backend` under `roots`, or nil.
    public static func locate(_ backend: MonocularDepthBackend, roots: [URL]) -> DepthModelLocation? {
        let fm = FileManager.default
        for root in roots {
            let depthDir = root.appendingPathComponent("depth")
            let backendDir = depthDir.appendingPathComponent(backend.rawValue)
            for dir in [backendDir, depthDir] {
                for name in candidateNames(for: backend) {
                    // Only the backend's own directory may use the generic "model.*" names.
                    if dir == depthDir && name.hasPrefix("model.") { continue }
                    let url = dir.appendingPathComponent(name)
                    if fm.fileExists(atPath: url.path) {
                        let manifest = backendDir.appendingPathComponent("depth.json")
                        return DepthModelLocation(backend: backend, modelURL: url,
                                                  manifestURL: dir == backendDir && fm.fileExists(atPath: manifest.path) ? manifest : nil)
                    }
                }
            }
        }
        return nil
    }

    /// A model given explicitly (e.g. `armature --depth-model path`): the file itself, or a directory
    /// laid out like a backend directory.
    public static func explicit(_ backend: MonocularDepthBackend, url: URL) -> DepthModelLocation? {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { return nil }
        let ext = url.pathExtension.lowercased()
        if ["mlpackage", "mlmodelc", "mlmodel"].contains(ext) {
            let manifest = url.deletingLastPathComponent().appendingPathComponent("depth.json")
            return DepthModelLocation(backend: backend, modelURL: url,
                                      manifestURL: FileManager.default.fileExists(atPath: manifest.path) ? manifest : nil)
        }
        guard isDir.boolValue else { return nil }
        for name in candidateNames(for: backend) {
            let candidate = url.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: candidate.path) {
                let manifest = url.appendingPathComponent("depth.json")
                return DepthModelLocation(backend: backend, modelURL: candidate,
                                          manifestURL: FileManager.default.fileExists(atPath: manifest.path) ? manifest : nil)
            }
        }
        return nil
    }
}
