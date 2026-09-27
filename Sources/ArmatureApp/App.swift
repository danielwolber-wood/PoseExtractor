import AppKit
import ArmatureCore
import SceneKit
import SwiftUI
import UniformTypeIdentifiers

@main
struct ArmatureApp: App {
    init() {
        // Running from `swift run` there is no bundle; make sure we still get a Dock icon and focus.
        NSApplication.shared.setActivationPolicy(.regular)
        DispatchQueue.main.async { NSApplication.shared.activate(ignoringOtherApps: true) }
    }

    var body: some Scene {
        WindowGroup("Armature") {
            ContentView()
                .frame(minWidth: 1000, minHeight: 620)
        }
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}

struct ContentView: View {
    // (No @State: its macro plugin ships with Xcode, and this builds with the Command Line Tools alone.)
    @StateObject private var model = StudioModel()

    var body: some View {
        Group {
            if model.modelsMissing && model.sourceImage == nil {
                MissingModelsView()
            } else if model.sourceImage == nil {
                DropZone(targeted: model.dropTargeted, open: openPanel)
            } else {
                HSplitView {
                    SourcePanel(model: model)
                        .frame(minWidth: 300, idealWidth: 400, maxWidth: 620)
                    ScenePanel(model: model)
                        .frame(minWidth: 480)
                }
            }
        }
        .onDrop(of: [.fileURL], isTargeted: $model.dropTargeted) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url { Task { @MainActor in model.open(url) } }
            }
            return true
        }
        .toolbar { toolbar }
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .navigation) {
            Button(action: openPanel) { Label("Open", systemImage: "photo.badge.plus") }
                .keyboardShortcut("o")
        }
        ToolbarItemGroup(placement: .principal) {
            Picker("View", selection: $model.view) {
                Text("Photo").tag(ClayScene.View.photo)
                Text("Studio").tag(ClayScene.View.studio)
            }
            .pickerStyle(.segmented)
            .onChange(of: model.view) { _, v in model.clayScene?.setView(v) }
        }
        ToolbarItemGroup(placement: .primaryAction) {
            Picker("Body model", selection: $model.bodyModel) {
                // Grouped by family (SMPL, SMPL-X, Anny); only models converted on this Mac appear.
                ForEach(model.modelFamilies) { family in
                    Section(family.id) {
                        ForEach(family.models) { m in Text(m.displayName).tag(m.id) }
                    }
                }
            }
            .help("Body model: SMPL, SMPL-X (fingers, jaw) or Anny (MakeHuman-based, all ages, Apache-2.0)")
            Picker("Material", selection: $model.style.finish) {
                ForEach(ClayStyle.Finish.allCases, id: \.self) { Text($0.displayName).tag($0) }
            }
            .help("Surface material")
            Picker("Age", selection: $model.ageModel) {
                Text("No age estimation").tag(StudioModel.noAgeModel)
                ForEach(model.availableAgeModels) { m in Text(m.displayName).tag(m.id) }
            }
            .help("Age estimator. Estimated (or typed-in) ages condition Anny's body shape on age.")
            Picker("Depth", selection: $model.depthMode) {
                Text("No monocular depth").tag("none")
                Text("Automatic depth").tag("auto")
                ForEach(MonocularDepthBackend.allCases, id: \.self) { b in
                    Text(b.displayName + (model.installedDepthBackends.contains(b) ? "" : " (not installed)")).tag(b.rawValue)
                }
            }
            .help("Monocular depth for photos without LiDAR/TrueDepth depth. Embedded metric depth always wins; "
                  + "a depth-guided fit is kept only when it doesn't make the fit worse.")
            Picker("Colours", selection: $model.style.palette) {
                ForEach(ClayStyle.Palette.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) }
            }
            .disabled(!model.style.finish.usesPalette)
            .help("Colour palette")
            Menu {
                Button("Image…", action: model.exportImage)
                    .keyboardShortcut("e")
                Divider()
                Button("USDZ Scene…", action: model.exportUSDZ)
                Button("OBJ Meshes + SMPL Parameters…", action: model.exportOBJ)
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .disabled(model.clayScene == nil)
        }
    }

    private func openPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK, let url = panel.url { model.open(url) }
    }
}

struct DropZone: View {
    let targeted: Bool
    let open: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "figure.stand")
                .font(.system(size: 56, weight: .light))
                .foregroundStyle(.secondary)
            Text("Drop a photo of people")
                .font(.title2)
            Text("Each person is detected, fitted with an SMPL body and sculpted in clay.")
                .foregroundStyle(.secondary)
            Button("Choose Photo…", action: open)
                .controlSize(.large)
                .padding(.top, 6)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [8]))
                .foregroundStyle(targeted ? Color.accentColor : Color.secondary.opacity(0.4))
                .padding(24)
        )
    }
}

struct MissingModelsView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("SMPL models not found", systemImage: "exclamationmark.triangle")
                .font(.title2)
            Text("Convert your SMPL download once, from the project folder:")
            Text("uv run --with numpy --with scipy tools/convert_models.py")
                .font(.system(.body, design: .monospaced))
                .textSelection(.enabled)
                .padding(8)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
            Text("Armature looks in ./Models, next to the app, in $ARMATURE_MODELS, and in ~/Library/Application Support/Armature/Models.")
                .foregroundStyle(.secondary)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct SourcePanel: View {
    @ObservedObject var model: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let image = model.sourceImage {
                GeometryReader { geo in
                    let fitted = fit(image.size, in: geo.size)
                    ZStack(alignment: .topLeading) {
                        Image(nsImage: image)
                            .resizable()
                            .frame(width: fitted.width, height: fitted.height)
                        if model.showMasks {
                            ForEach(Array(model.maskOverlays.enumerated()), id: \.offset) { _, overlay in
                                if let overlay {
                                    Image(decorative: overlay, scale: 1)
                                        .resizable()
                                        .frame(width: fitted.width, height: fitted.height)
                                        .allowsHitTesting(false)
                                }
                            }
                        }
                        if model.showSkeleton, let result = model.result {
                            SkeletonEditor(model: model, scale: fitted.width / CGFloat(result.image.width))
                                .frame(width: fitted.width, height: fitted.height)
                        }
                    }
                    .frame(width: geo.size.width, height: geo.size.height)
                }
                .padding(12)
            }
            Divider()
            VStack(alignment: .leading, spacing: 8) {
                QualityPanel(model: model)
                status
                if let depth = model.depthStatus {
                    Label(depth.text, systemImage: depth.warning ? "exclamationmark.triangle" : "cube.transparent")
                        .font(.caption)
                        .foregroundStyle(depth.warning ? AnyShapeStyle(.orange) : AnyShapeStyle(.secondary))
                        .lineLimit(2)
                        .help(depth.text)
                }
                HStack(spacing: 14) {
                    Toggle("Skeletons", isOn: $model.showSkeleton)
                        .toggleStyle(.checkbox)
                    Toggle("Masks", isOn: $model.showMasks)
                        .toggleStyle(.checkbox)
                        .help("Show Vision's person segmentation masks")
                    Toggle("Fit silhouette", isOn: $model.useSilhouette)
                        .toggleStyle(.checkbox)
                        .help("Refine body shape and pose so the body's outline matches the mask")
                    Spacer()
                    if model.showSkeleton && model.result != nil {
                        Text("Drag joints to fix the pose")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                if let result = model.result {
                    ForEach(Array(zip(result.bodies, model.people).enumerated()), id: \.offset) { i, pair in
                        PersonRow(model: model, index: i, body: pair.0, person: pair.1)
                    }
                    let total = result.timings.map(\.1).reduce(0, +)
                    Text(String(format: "Processed in %.0f ms", total * 1000))
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(12)
        }
    }

    @ViewBuilder private var status: some View {
        switch model.phase {
        case .idle, .done: EmptyView()
        case .working(let msg):
            HStack(spacing: 8) { ProgressView().controlSize(.small); Text(msg) }
        case .failed(let msg):
            Label(msg, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
        }
    }

    private func fit(_ size: CGSize, in box: CGSize) -> CGSize {
        let s = min(box.width / max(size.width, 1), box.height / max(size.height, 1))
        return CGSize(width: size.width * s, height: size.height * s)
    }
}

struct PersonRow: View {
    @ObservedObject var model: StudioModel
    let index: Int
    let body_: FittedBody
    let person: DetectedPerson

    init(model: StudioModel, index: Int, body: FittedBody, person: DetectedPerson) {
        self.model = model
        self.index = index
        self.body_ = body
        self.person = person
    }

    private var stats: String {
        var s = String(format: "%.1f m away · %.0f px", body_.translation.z, body_.rms2D)
        if let summary = model.summary(of: body_) { s = summary + " · " + s }
        if let sil = body_.silhouette { s += String(format: " · %.0f%% out", sil.outside * 100) }
        return s
    }

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(Color(nsColor: model.style.color(forPerson: index)))
                .frame(width: 10, height: 10)
            Text("Person \(index + 1)")
            if person.isEdited {
                Text("edited")
                    .font(.caption2)
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(.yellow.opacity(0.3), in: Capsule())
            }
            Spacer()
            AgeField(model: model, index: index, person: person)
            Text(stats)
                .foregroundStyle(.secondary)
                .monospacedDigit()
                .help("Distance from camera · keypoint reprojection error · share of the body outside its mask")
            Menu {
                Button("Swap Left and Right") { model.swapLeftRight(index) }
                Button("Reset Edits") { model.resetEdits(index) }
                    .disabled(!person.isEdited)
                Divider()
                Button("Remove Person", role: .destructive) { model.remove(index) }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
        }
        .font(.callout)
    }
}

/// Age in years: shows the estimate as a placeholder; typing an age overrides it, clearing reverts.
struct AgeField: View {
    @ObservedObject var model: StudioModel
    let index: Int
    let person: DetectedPerson

    var body: some View {
        HStack(spacing: 3) {
            Text("age").foregroundStyle(.secondary)
            TextField(person.estimatedAge.map { String(format: "~%.0f", $0.years) } ?? "auto",
                      value: Binding(get: { person.ageOverride }, set: { model.setAge(index, $0) }),
                      format: .number.precision(.fractionLength(0)))
                .frame(width: 38)
                .textFieldStyle(.roundedBorder)
                .help(person.estimatedAge.map { "Estimated \(String(format: "%.1f", $0.years)) by \($0.model). Type an age to override; clear to use the estimate." }
                      ?? "Type an age in years (conditions Anny's body shape).")
        }
        .font(.caption)
    }
}

struct ScenePanel: View {
    @ObservedObject var model: StudioModel

    var body: some View {
        ZStack {
            Color(nsColor: .underPageBackgroundColor)
            if let scene = model.clayScene {
                // In photo view the viewport must keep the photo's aspect so the figures line up with it.
                ClaySceneView(clayScene: scene, view: model.view, model: model)
                    .id(ObjectIdentifier(scene))
                    .aspectRatio(model.view == .photo ? scene.imageSize.width / scene.imageSize.height : nil,
                                 contentMode: .fit)
                VStack {
                    Spacer()
                    Text("Drag to orbit · scroll to zoom · double-click to reset · ⌘E to export an image")
                        .font(.caption)
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(10)
                }
            } else if case .working = model.phase {
                ProgressView()
            }
        }
    }
}

/// SCNView wrapper (SwiftUI's SceneView doesn't expose multisampling).
struct ClaySceneView: NSViewRepresentable {
    let clayScene: ClayScene
    let view: ClayScene.View
    let model: StudioModel

    final class Coordinator {
        var saved: [ObjectIdentifier: simd_float4x4] = [:]
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> SCNView {
        let v = SCNView()
        v.scene = clayScene.scene
        v.antialiasingMode = .multisampling4X
        v.allowsCameraControl = true
        v.rendersContinuously = false
        v.backgroundColor = .clear
        for cam in [clayScene.photoCamera, clayScene.studioCamera] {
            context.coordinator.saved[ObjectIdentifier(cam)] = cam.simdTransform
        }
        let reset = NSClickGestureRecognizer(target: ResetTarget.shared, action: #selector(ResetTarget.reset(_:)))
        reset.numberOfClicksRequired = 2
        v.addGestureRecognizer(reset)
        ResetTarget.shared.handlers[ObjectIdentifier(v)] = { [weak v] in
            guard let v, let pov = v.pointOfView,
                  let t = context.coordinator.saved[ObjectIdentifier(pov)] else { return }
            SCNTransaction.begin()
            SCNTransaction.animationDuration = 0.35
            pov.simdTransform = t
            SCNTransaction.commit()
        }
        model.liveView = v
        apply(to: v, context: context)
        return v
    }

    func updateNSView(_ v: SCNView, context: Context) {
        model.liveView = v
        apply(to: v, context: context)
    }

    private func apply(to v: SCNView, context: Context) {
        let cam = view == .photo ? clayScene.photoCamera : clayScene.studioCamera
        if v.pointOfView !== cam {
            if let t = context.coordinator.saved[ObjectIdentifier(cam)] { cam.simdTransform = t }
            v.pointOfView = cam
        }
        clayScene.setView(view)
    }
}

/// Target for the double-click-to-reset gesture (gesture recognisers need an NSObject target).
final class ResetTarget: NSObject {
    static let shared = ResetTarget()
    var handlers: [ObjectIdentifier: () -> Void] = [:]
    @objc func reset(_ g: NSGestureRecognizer) {
        if let v = g.view { handlers[ObjectIdentifier(v)]?() }
    }
}


struct QualityPanel: View {
    @ObservedObject var model: StudioModel
    var body: some View {
        DisclosureGroup("Image quality") {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Button(model.qualityReport == nil ? "Analyze Quality" : "Analyze Again", action: model.analyzeQuality)
                        .disabled(model.qualityRunning)
                    if model.qualityRunning {
                        ProgressView().controlSize(.small)
                        Button("Cancel", action: model.cancelQuality)
                    }
                    Spacer()
                    Button("Export Scores…", action: model.exportQuality).disabled(model.qualityReport == nil)
                }
                if !model.qualityStatus.isEmpty { Text(model.qualityStatus).font(.caption).foregroundStyle(.secondary) }
                if let report = model.qualityReport {
                    Text(String(format: "%d × %d · %.2f MP · %@ · %@", report.width, report.height, report.megapixels, report.orientation, report.standardRatioBucket))
                        .font(.caption)
                    ScrollView {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(report.scores, id: \.metric) { score in
                                HStack(alignment: .top) {
                                    Text(score.metric.uppercased()).frame(width: 90, alignment: .leading)
                                    if let raw = score.raw {
                                        Text(String(format: "%.4f", raw)).monospacedDigit()
                                        Spacer()
                                        Text(score.lowerBetter ? "Lower is better" : "Higher is better").foregroundStyle(.secondary)
                                    } else {
                                        Text(score.error ?? "Unavailable").foregroundStyle(.secondary).textSelection(.enabled)
                                    }
                                }.font(.caption)
                            }
                        }
                    }.frame(maxHeight: 180)
                    Text("Each metric has its own scale. These scores assess the photo, not pose accuracy.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }.padding(.vertical, 6)
        }
    }
}
