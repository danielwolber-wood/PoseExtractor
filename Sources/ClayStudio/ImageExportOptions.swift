import AppKit
import ClayCore
import UniformTypeIdentifiers

/// Accessory controls for the "Export Image" save panel: size, format and background.
@MainActor
final class ImageExportOptions: NSObject {
    private let sizePopup = NSPopUpButton()
    private let formatPopup = NSPopUpButton()
    private let backgroundPopup = NSPopUpButton()
    private weak var panel: NSSavePanel?

    private static let sizes: [(String, CGFloat?)] = [
        ("Window size", nil), ("1080 px", 1080), ("2048 px", 2048), ("4096 px", 4096),
    ]
    private static let formats: [(String, UTType)] = [("PNG", .png), ("JPEG", .jpeg), ("HEIC", .heic)]

    // Remember choices between exports.
    private static var lastSize = 2
    private static var lastFormat = 0
    private static var lastBackground = 0

    var format: UTType { Self.formats[formatPopup.indexOfSelectedItem].1 }
    var background: ClayScene.Background {
        backgroundPopup.indexOfSelectedItem == 1 && backgroundPopup.isEnabled ? .transparent : .scene
    }

    /// Long edge in pixels. `current` is the on-screen view's long edge in pixels.
    func longEdge(current: CGFloat) -> CGFloat {
        (Self.sizes[sizePopup.indexOfSelectedItem].1 ?? current).rounded()
    }

    func attach(to panel: NSSavePanel) {
        self.panel = panel
        sizePopup.addItems(withTitles: Self.sizes.map(\.0))
        formatPopup.addItems(withTitles: Self.formats.map(\.0))
        backgroundPopup.addItems(withTitles: ["Scene", "Transparent"])
        sizePopup.selectItem(at: Self.lastSize)
        formatPopup.selectItem(at: Self.lastFormat)
        backgroundPopup.selectItem(at: Self.lastBackground)
        for p in [sizePopup, formatPopup, backgroundPopup] {
            p.target = self
            p.action = #selector(changed)
        }

        let grid = NSGridView(views: [
            [NSTextField(labelWithString: "Size:"), sizePopup],
            [NSTextField(labelWithString: "Format:"), formatPopup],
            [NSTextField(labelWithString: "Background:"), backgroundPopup],
        ])
        grid.column(at: 0).xPlacement = .trailing
        grid.rowSpacing = 8
        let container = NSView()
        grid.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(grid)
        NSLayoutConstraint.activate([
            grid.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            grid.topAnchor.constraint(equalTo: container.topAnchor, constant: 12),
            grid.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -12),
        ])
        container.frame = NSRect(x: 0, y: 0, width: 360, height: 110)
        panel.accessoryView = container
        panel.isExtensionHidden = false
        objc_setAssociatedObject(panel, &Self.key, self, .OBJC_ASSOCIATION_RETAIN)  // live as long as the panel
        changed()
    }

    private static var key = 0

    @objc private func changed() {
        // JPEG has no alpha channel.
        backgroundPopup.isEnabled = format != .jpeg
        Self.lastSize = sizePopup.indexOfSelectedItem
        Self.lastFormat = formatPopup.indexOfSelectedItem
        Self.lastBackground = backgroundPopup.indexOfSelectedItem
        panel?.allowedContentTypes = [format]
    }
}
