// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "Armature",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "ArmatureCore", targets: ["ArmatureCore"]),
        .executable(name: "armature", targets: ["armature"]),
        .executable(name: "ArmatureApp", targets: ["ArmatureApp"]),
    ],
    targets: [
        .target(
            name: "ArmatureCore",
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Vision"),
                .linkedFramework("SceneKit"),
            ]
        ),
        .executableTarget(name: "armature", dependencies: ["ArmatureCore"]),
        .executableTarget(name: "ArmatureApp", dependencies: ["ArmatureCore"]),
        .executableTarget(name: "armature-selftest", dependencies: ["ArmatureCore"]),
        .executableTarget(name: "armature-eval", dependencies: ["ArmatureCore"]),
        .executableTarget(name: "armature-quality-selftest", dependencies: ["ArmatureCore"]),
        .executableTarget(name: "armature-depth-selftest", dependencies: ["ArmatureCore"]),
    ]
)
