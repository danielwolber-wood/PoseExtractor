import ArmatureCore
import Foundation

/// Independent command: no body models, pose detection, scene, or renderer required.
func runQualityCommand(_ arguments: [String]) throws {
    var args = arguments
    var path: String?, output = "quality_output.csv"
    var models: URL?
    var metrics = ImageQualityAnalyzer.metricIDs
    while !args.isEmpty {
        let arg = args.removeFirst()
        func value() throws -> String {
            guard !args.isEmpty else { throw QualityError.invalid("\(arg) requires a value") }
            return args.removeFirst()
        }
        switch arg {
        case "-o", "--out": output = try value()
        case "--models": models = URL(fileURLWithPath: try value())
        case "--metrics":
            metrics = try value().split(separator: ",").map(String.init)
            guard !metrics.isEmpty, metrics.allSatisfy(ImageQualityAnalyzer.metricIDs.contains) else {
                throw QualityError.invalid("Metrics: \(ImageQualityAnalyzer.metricIDs.joined(separator: ","))")
            }
        default:
            guard !arg.hasPrefix("-"), path == nil else { throw QualityError.invalid("Unexpected argument: \(arg)") }
            path = arg
        }
    }
    guard let path else { throw QualityError.invalid("Usage: armature quality <image-or-folder> [-o scores.csv|scores.json] [--models Models] [--metrics nima,brisque,...]") }
    let root = URL(fileURLWithPath: path)
    let destination = URL(fileURLWithPath: output)
    guard ["csv","json"].contains(destination.pathExtension.lowercased()) else { throw QualityError.invalid("Output must be .csv or .json") }
    let directory = models ?? ImageQualityAnalyzer.defaultModelsDirectory()
    let analyzer = ImageQualityAnalyzer(modelsDirectory: directory)
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory) else { throw QualityError.invalid("Input not found: \(path)") }
    let files: [URL]
    if isDirectory.boolValue {
        let extensions = Set(["jpg","jpeg","png","heic","heif","tif","tiff","bmp","webp"])
        let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])
        files = (enumerator?.allObjects as? [URL] ?? []).filter {
            extensions.contains($0.pathExtension.lowercased()) && (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }.sorted { $0.path < $1.path }
    } else { files = [root] }
    guard !files.isEmpty else { throw QualityError.invalid("No supported images found") }
    var reports: [ImageQualityReport] = [], failures: [String] = []
    for (i,file) in files.enumerated() {
        print("[\(i+1)/\(files.count)] \(file.lastPathComponent)")
        do {
            let report = try analyzer.analyze(url: file, metrics: metrics)
            reports.append(report)
            for score in report.scores {
                if let raw = score.raw { print(String(format: "  %@: %.6f", score.metric, raw)) }
                if let error = score.error { print("  \(score.metric): \(error)") }
            }
        } catch {
            failures.append("\(file.path): \(error.localizedDescription)")
            FileHandle.standardError.write(Data("warning: \(failures.last!)\n".utf8))
        }
    }
    try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
    if destination.pathExtension.lowercased() == "json" {
        try ImageQualityReport.json(reports).write(to: destination, options: .atomic)
    } else {
        try ImageQualityReport.csv(reports).write(to: destination, atomically: true, encoding: .utf8)
    }
    print("Wrote \(reports.count) images to \(destination.path); \(failures.count) unreadable images")
    if reports.isEmpty { throw QualityError.invalid("No images could be scored; an empty report was written") }
    let requested = metrics.filter { $0 != "align-one" }
    if !requested.isEmpty && !reports.contains(where: { $0.scores.contains(where: { $0.raw != nil }) }) {
        throw QualityError.invalid("All requested metrics failed; see the report's error fields")
    }
}
