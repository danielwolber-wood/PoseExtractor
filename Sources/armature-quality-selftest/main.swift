import ArmatureCore
import CoreGraphics
import Foundation

func check(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw QualityError.invalid(message) }
}

do {
    let report = ImageQualityReport(url: URL(fileURLWithPath: "/tmp/a,\"b.png"), width: 4032, height: 3024)
    try check(report.aspectRatio == "4:3" && report.standardRatioBucket == "4:3", "Aspect ratio")
    try check(report.orientation == "Landscape", "Orientation")
    try check(abs(report.megapixels - 12.192768) < 1e-9, "Megapixels")
    let csv = ImageQualityReport.csv([report])
    try check(csv.contains("\"a,\"\"b.png\""), "CSV quote escaping")
    try check(csv.contains("\"niqe_error\""), "Stable metric columns")
    try check(ImageQualityReport.csv([]).components(separatedBy: "\r\n").count == 2, "Empty CSV")
    let decoded = try JSONDecoder().decode([ImageQualityReport].self, from: ImageQualityReport.json([report]))
    try check(decoded.count == 1 && decoded[0].fileName == report.fileName, "JSON round trip")
    try check(ImageQualityReport(url: URL(fileURLWithPath: "/tmp/p"), width: 300, height: 700).standardRatioBucket == "Custom", "Custom ratio")
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let ctx = CGContext(data: nil, width: 32, height: 24, bitsPerComponent: 8, bytesPerRow: 128,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    let url = dir.appendingPathComponent("image.png")
    try ArmatureExport.writePNG(ctx.makeImage()!, to: url)
    let analyzer = ImageQualityAnalyzer(modelsDirectory: dir)
    let missing = try analyzer.analyze(url: url, metrics: ["nima", "align-one"])
    try check(missing.width == 32 && missing.scores.count == 2, "Metadata without body models")
    try check(missing.scores.allSatisfy { $0.raw == nil && $0.error != nil }, "Missing models must remain explicit")
    var cancelled = false
    do { _ = try analyzer.analyze(url: url, cancelled: { true }) } catch is CancellationError { cancelled = true }
    try check(cancelled, "Cancellation")
    var rejected = false
    do { _ = try analyzer.analyze(url: dir.appendingPathComponent("corrupt.jpg")) } catch { rejected = true }
    try check(rejected, "Unreadable images")
    print("Quality self-test passed: metadata, CSV, JSON, missing models, cancellation, unreadable images")
} catch {
    FileHandle.standardError.write(Data("Quality self-test failed: \(error)\n".utf8))
    exit(1)
}
