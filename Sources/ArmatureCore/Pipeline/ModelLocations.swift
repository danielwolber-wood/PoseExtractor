import Foundation

/// Model folders outside the working tree. `CLAY_MODELS` and Application Support/ClayStudio/Models are
/// the names from before the rename to Armature; they're still searched so existing installs keep working.
public enum ModelLocations {
    /// `ARMATURE_MODELS`, else `CLAY_MODELS`.
    public static func environmentDirectory(_ environment: [String: String] = ProcessInfo.processInfo.environment) -> URL? {
        for key in ["ARMATURE_MODELS", "CLAY_MODELS"] {
            if let path = environment[key], !path.isEmpty { return URL(fileURLWithPath: path) }
        }
        return nil
    }

    /// ~/Library/Application Support/Armature/Models, then the pre-rename ClayStudio/Models.
    public static func applicationSupportDirectories(
        _ applicationSupport: URL? = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
    ) -> [URL] {
        guard let applicationSupport else { return [] }
        return ["Armature/Models", "ClayStudio/Models"].map { applicationSupport.appendingPathComponent($0) }
    }
}
