// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ClayPose",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "ClayCore", targets: ["ClayCore"]),
        .executable(name: "clay", targets: ["clay"]),
        .executable(name: "ClayStudio", targets: ["ClayStudio"]),
    ],
    targets: [
        .target(
            name: "ClayCore",
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Vision"),
                .linkedFramework("SceneKit"),
            ]
        ),
        .executableTarget(name: "clay", dependencies: ["ClayCore"]),
        .executableTarget(name: "ClayStudio", dependencies: ["ClayCore"]),
        .executableTarget(name: "clay-selftest", dependencies: ["ClayCore"]),
        .executableTarget(name: "clay-icon", dependencies: ["ClayCore"]),
    ]
)
