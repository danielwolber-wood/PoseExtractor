import AppKit
import ClayCore
import SwiftUI

/// Draws each person's detected skeleton over the photo and lets the user drag joints to fix it.
struct SkeletonEditor: View {
    @ObservedObject var model: StudioModel
    /// View points per image pixel.
    let scale: CGFloat

    private var hitRadius: Double { 14 / scale }

    var body: some View {
        Canvas { ctx, _ in
            for (i, p) in model.people.enumerated() {
                let color = Color(nsColor: model.style.color(forPerson: i))
                func pt(_ j: BodyJoint) -> CGPoint {
                    CGPoint(x: p.joints2D[j.rawValue].x * scale, y: p.joints2D[j.rawValue].y * scale)
                }
                var path = Path()
                for (a, b) in BodyJoint.bones where p.isVisible(a) && p.isVisible(b) {
                    path.move(to: pt(a)); path.addLine(to: pt(b))
                }
                ctx.stroke(path, with: .color(.black.opacity(0.35)), style: StrokeStyle(lineWidth: 5, lineCap: .round))
                ctx.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 2.5, lineCap: .round))

                for j in BodyJoint.allCases where p.isVisible(j) {
                    let c = pt(j)
                    let ref = JointRef(person: i, joint: j)
                    let active = model.dragging == ref || (model.dragging == nil && model.hovered == ref)
                    let r: CGFloat = active ? 7 : (j.is2DOnly ? 3.5 : 4.5)
                    let dot = Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r))
                    // Edited joints are yellow so it's clear which keypoints the fit is now trusting.
                    // Guessed joints (out of frame / hidden) are hollow: a hint that they may need dragging.
                    if p.isGuessed(j) {
                        ctx.stroke(dot, with: .color(.white), style: StrokeStyle(lineWidth: 1.5, dash: [2, 2]))
                    } else {
                        ctx.fill(dot, with: .color(p.edited[j.rawValue] ? .yellow : .white))
                        ctx.stroke(dot, with: .color(.black.opacity(0.6)), lineWidth: 1)
                    }
                }
            }
        }
        .contentShape(Rectangle())
        .onContinuousHover { phase in
            switch phase {
            case .active(let loc):
                model.hover(image(loc), radius: hitRadius)
                (model.hovered != nil || model.dragging != nil ? NSCursor.openHand : NSCursor.arrow).set()
            case .ended:
                model.hover(nil, radius: hitRadius)
                NSCursor.arrow.set()
            }
        }
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { v in
                    model.drag(from: image(v.startLocation), to: image(v.location), radius: hitRadius)
                    if model.dragging != nil { NSCursor.closedHand.set() }
                }
                .onEnded { _ in
                    model.endDrag()
                    NSCursor.arrow.set()
                }
        )
    }

    private func image(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x / scale, y: p.y / scale) }
}
