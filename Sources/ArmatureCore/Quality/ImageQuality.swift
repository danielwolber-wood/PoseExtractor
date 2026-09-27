import CoreML
import CryptoKit
import Foundation

public enum QualityError: LocalizedError {
    case invalid(String)
    public var errorDescription: String? { switch self { case .invalid(let message): return message } }
}

public struct ImageQualityReport: Codable, Sendable {
    public struct Score: Codable, Sendable {
        public let metric: String
        public let raw: Double?
        public let scaled: Double?
        public let lowerBetter: Bool
        public let error: String?
    }
    public let filePath: String
    public let fileName: String
    public let width: Int
    public let height: Int
    public let aspectRatio: String
    public let decimal: Double
    public let totalPixels: Int
    public let megapixels: Double
    public let orientation: String
    public let standardRatioBucket: String
    public var scores: [Score]

    public init(url: URL, width: Int, height: Int) {
        filePath = url.path; fileName = url.lastPathComponent
        self.width = width; self.height = height
        var a = width, b = height
        while b != 0 { (a,b) = (b,a%b) }
        aspectRatio = "\(width/a):\(height/a)"
        decimal = Double(width)/Double(height)
        totalPixels = width*height; megapixels = Double(totalPixels)/1_000_000
        orientation = width == height ? "Square" : width > height ? "Landscape" : "Portrait"
        let ratios: [(String,Double)] = [("16:9",16.0/9),("3:2",1.5),("4:3",4.0/3),("5:4",1.25),("1:1",1),
                                         ("4:5",0.8),("3:4",0.75),("2:3",2.0/3),("9:16",9.0/16)]
        let ratio = Double(width)/Double(height)
        let closest = ratios.min { abs($0.1-ratio) < abs($1.1-ratio) }!
        standardRatioBucket = abs(closest.1-ratio) <= 0.05 ? closest.0 : "Custom"
        scores = []
    }

    public static func csv(_ reports: [ImageQualityReport]) -> String {
        let ids = ImageQualityAnalyzer.metricIDs
        let headers = ["file_path","file_name","width","height","aspect_ratio","decimal","total_pixels","megapixels","orientation","standard_ratio_bucket"]
            + ids.flatMap { ["\($0)_raw", "\($0)_scaled", "\($0)_error"] }
        func quote(_ s: String) -> String { "\"" + s.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }
        var rows = [headers.map(quote).joined(separator: ",")]
        for report in reports {
            var row = [report.filePath,report.fileName,String(report.width),String(report.height),report.aspectRatio,
                       String(report.decimal),String(report.totalPixels),String(report.megapixels),report.orientation,report.standardRatioBucket]
            for id in ids {
                let s = report.scores.first { $0.metric == id }
                row += [s?.raw.map { String($0) } ?? "", s?.scaled.map { String($0) } ?? "", s?.error ?? ""]
            }
            rows.append(row.map(quote).joined(separator: ","))
        }
        return rows.joined(separator: "\r\n") + "\r\n"
    }
    public static func json(_ reports: [ImageQualityReport]) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
        return try encoder.encode(reports)
    }
}

/// Independent of body models and person detection. Calls are serialized to bound memory usage.
public final class ImageQualityAnalyzer: @unchecked Sendable {
    public static let metricIDs = ["align-one","musiq","hyperiqa","nima","brisque","clipiqa","niqe","arniqa","liqe"]
    public let directory: URL
    private let lock = NSLock()
    struct Spec: Decodable {
        let id: String
        let schemaVersion: Int
        let backend: String
        let preprocessing: String?
        let lowerBetter: Bool
        let scoreRange: [Double]
        let input: String
        let output: String
    }
    public static func defaultModelsDirectory() -> URL {
        if let env = ModelLocations.environmentDirectory() { return env }
        let candidates = [Bundle.main.resourceURL?.appendingPathComponent("Models"),
                          URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("Models")].compactMap { $0 }
                          + ModelLocations.applicationSupportDirectories()
        return candidates.first { FileManager.default.fileExists(atPath: $0.appendingPathComponent("quality").path) }
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("Models")
    }
    public init(modelsDirectory: URL) { directory = modelsDirectory.appendingPathComponent("quality") }

    public func analyze(url: URL, metrics: [String] = ImageQualityAnalyzer.metricIDs,
                        cancelled: () -> Bool = { false }, progress: (String) -> Void = { _ in }) throws -> ImageQualityReport {
        lock.lock(); defer { lock.unlock() }
        if cancelled() { throw CancellationError() }
        let image = try LoadedImage(url: url)
        let pixels = try QualityPixels(image.cgImage)
        var report = ImageQualityReport(url: url, width: image.width, height: image.height)
        for id in metrics {
            if cancelled() { throw CancellationError() }
            progress(id)
            let lower = id == "brisque" || id == "niqe"
            do {
                let score: ImageQualityReport.Score = try autoreleasepool {
                    guard Self.metricIDs.contains(id) else { throw QualityError.invalid("Unknown quality metric") }
                    guard id != "align-one" else { throw QualityError.invalid("Not registered in PyIQA 0.1.14.1; no substitute selected") }
                    let root = directory.appendingPathComponent(id)
                    guard FileManager.default.fileExists(atPath: root.appendingPathComponent("quality.json").path) else {
                        throw QualityError.invalid("Model not installed; run tools/convert_quality_models.py \(id)")
                    }
                    let spec = try JSONDecoder().decode(Spec.self, from: Data(contentsOf: root.appendingPathComponent("quality.json")))
                    guard spec.schemaVersion == 1, spec.id == id, ["statistics", "coreml"].contains(spec.backend) else { throw QualityError.invalid("Unsupported quality model manifest") }
                    let raw: Double
                    if spec.backend == "statistics" {
                        raw = try QualityStatistics.score(pixels, id: id, parameters: root.appendingPathComponent("parameters.json"))
                    } else {
                        raw = try predict(pixels, spec: spec, root: root, cancelled: cancelled)
                    }
                    guard raw.isFinite else { throw QualityError.invalid("Metric produced a non-finite score") }
                    var scaled: Double?
                    if spec.scoreRange.count == 2, spec.scoreRange[1] > spec.scoreRange[0] {
                        let s = (raw-spec.scoreRange[0])/(spec.scoreRange[1]-spec.scoreRange[0])
                        scaled = spec.lowerBetter ? 1-s : s
                    }
                    return .init(metric: id, raw: raw, scaled: scaled, lowerBetter: spec.lowerBetter, error: nil)
                }
                report.scores.append(score)
            } catch {
                if error is CancellationError { throw error }
                report.scores.append(.init(metric: id, raw: nil, scaled: nil, lowerBetter: lower, error: error.localizedDescription))
            }
        }
        if cancelled() { throw CancellationError() }
        return report
    }

    private func predict(_ pixels: QualityPixels, spec: Spec, root: URL, cancelled: () -> Bool) throws -> Double {
        if spec.preprocessing == "liqe", min(pixels.width, pixels.height) < 224 {
            throw QualityError.invalid("LIQE needs both dimensions at least 224 pixels")
        }
        if ["arniqa", "clipiqa"].contains(spec.preprocessing ?? ""),
           min(pixels.width, pixels.height) < 32 || max(pixels.width, pixels.height) > 8192 {
            throw QualityError.invalid("Metric supports dimensions from 32 to 8192 pixels")
        }
        if spec.preprocessing == "musiq", ((pixels.width + 31)/32)*((pixels.height + 31)/32)+193 > 16384 {
            throw QualityError.invalid("MUSIQ supports at most 16,384 patches; image is too large")
        }
        let package = root.appendingPathComponent("model.mlpackage")
        let manifest = try Data(contentsOf: package.appendingPathComponent("Manifest.json"))
        let stamp = try Data(contentsOf: root.appendingPathComponent("quality.json"))
        let digest = SHA256.hash(data: manifest + stamp).map { String(format: "%02x", $0) }.joined()
        let cacheRoot = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("Armature/quality")
        let key = "\(spec.id)-\(digest)"
        let cached = cacheRoot.appendingPathComponent(key + ".mlmodelc")
        if !FileManager.default.fileExists(atPath: cached.path) {
            let compiled = try MLModel.compileModel(at: package)
            try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
            do { try FileManager.default.moveItem(at: compiled, to: cached) }
            catch { if !FileManager.default.fileExists(atPath: cached.path) { throw error } }
        }
        let config = MLModelConfiguration(); config.computeUnits = .all
        let model = try MLModel(contentsOf: cached, configuration: config)
        func run(_ array: MLMultiArray) throws -> [Double] {
            if cancelled() { throw CancellationError() }
            let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [spec.input: array]))
            guard let values = output.featureValue(for: spec.output)?.multiArrayValue else { throw QualityError.invalid("Missing score output") }
            return (0..<values.count).map { values[$0].doubleValue }
        }
        func scalar(_ p: QualityPixels) throws -> Double {
            let values = try run(p.tensor())
            guard values.count == 1 else { throw QualityError.invalid("Expected one score") }
            return values[0]
        }
        switch spec.preprocessing {
        case "musiq": return try run(pixels.musiqPatches())[0]
        case "nima":
            let ratio = 299.0 / Double(min(pixels.width,pixels.height))
            let resized = pixels.resized(width: Int(Double(pixels.width)*ratio), height: Int(Double(pixels.height)*ratio), antialias: true)
            return try scalar(resized.crop(x: Int((Double(resized.width-299)/2).rounded(.toNearestOrEven)),
                                           y: Int((Double(resized.height-299)/2).rounded(.toNearestOrEven)), width: 299, height: 299))
        case "hyperiqa":
            var p = pixels
            if min(p.width,p.height) <= 224 {
                let scale = 225.0/Double(min(p.width,p.height))
                p = p.resized(width: Int(Double(p.width)*scale), height: Int(Double(p.height)*scale), scaleFactor: scale)
            }
            var total = 0.0
            for y in 0..<5 { for x in 0..<5 { total += try scalar(p.crop(x: x*((p.width-224)/5), y: y*((p.height-224)/5), width: 224, height: 224)) } }
            return total/25
        case "liqe":
            guard min(pixels.width,pixels.height) >= 224 else { throw QualityError.invalid("LIQE needs both dimensions at least 224 pixels") }
            let cols = (pixels.width-224)/32+1, rows = (pixels.height-224)/32+1
            let count = min(15,cols*rows), step = max(1,cols*rows/count)
            var logits = [Double](repeating: 0,count: 5)
            for i in 0..<count {
                let index = i*step
                let result = try run(pixels.crop(x: (index%cols)*32, y: (index/cols)*32, width: 224, height: 224).tensor())
                guard result.count == 5 else { throw QualityError.invalid("Expected five LIQE logits") }
                for j in 0..<5 { logits[j] += result[j]/Double(count) }
            }
            let maxLogit = logits.max()!
            let probabilities = logits.map { exp($0-maxLogit) }
            return probabilities.enumerated().reduce(0) { $0+Double($1.offset+1)*$1.element }/probabilities.reduce(0,+)
        case "arniqa", "clipiqa":
            guard min(pixels.width,pixels.height) >= 32, max(pixels.width,pixels.height) <= 8192 else { throw QualityError.invalid("Metric supports dimensions from 32 to 8192 pixels") }
            return try scalar(pixels)
        default: throw QualityError.invalid("Unsupported quality preprocessing")
        }
    }
}
