import AppKit
import ClayCore
import simd

// Renders the app icon (1024×1024 PNG) with the app's own pipeline: a clay SMPL figure,
// star-jump pose, on a macOS-style rounded square. Usage: clay-icon <out.png>

guard let modelsURL = ClayPipeline.defaultModelsDirectory() else { print("models not found"); exit(1) }
let outURL = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "build/icon_1024.png")
let model = try BodyModel(modelsDirectory: modelsURL, id: ClayPipeline.defaultModelID)
func joint(_ name: String) -> Int { model.joint(name)! }

var pose = [SIMD3<Double>](repeating: .zero, count: model.jointCount)
let orient = simd_quatd(angle: -0.25, axis: SIMD3(0, 1, 0)) * simd_quatd(angle: .pi, axis: SIMD3(1, 0, 0))
pose[joint("pelvis")] = orient.axis * orient.angle
let rest = model.restJoints(betas: [])
func aim(_ j: Int, _ child: Int, _ dir: SIMD3<Double>) {
    let q = simd_quatd(from: simd_normalize(rest[child] - rest[j]), to: simd_normalize(dir))
    pose[j] = q.axis * q.angle
}
aim(joint("leftShoulder"), joint("leftElbow"), SIMD3(0.6, 0.8, 0))      // arms up in a V
aim(joint("rightShoulder"), joint("rightElbow"), SIMD3(-0.6, 0.8, 0))
aim(joint("leftHip"), joint("leftKnee"), SIMD3(0.27, -0.96, 0))         // legs apart
aim(joint("rightHip"), joint("rightKnee"), SIMD3(-0.27, -0.96, 0))
pose[joint("head")] = SIMD3(-0.15, 0, 0)                                 // chin up
let body = FittedBody(model: model.info.id, pose: pose, betas: [Double](repeating: 0, count: model.betaCount),
                      translation: SIMD3(0, 0, 4))
let mesh = model.vertices(pose: body.pose, betas: body.betas, translation: body.translation)

let blank = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
var style = ClayStyle()
style.palette = .terracotta
let scene = ClayScene(bodies: [body], meshes: [mesh], faces: model.faces,
                      image: try LoadedImage(cgImage: blank.makeImage()!), style: style)
guard let figure = scene.render(.studio, size: CGSize(width: 1024, height: 1024), background: .transparent) else {
    print("render failed"); exit(1)
}

// Compose on the macOS icon grid: 824 pt rounded square centred in 1024, with a soft drop shadow.
let S = 1024
let ctx = CGContext(data: nil, width: S, height: S, bitsPerComponent: 8, bytesPerRow: 0,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
let tile = CGRect(x: 100, y: 100, width: 824, height: 824)
let shape = CGPath(roundedRect: tile, cornerWidth: 185, cornerHeight: 185, transform: nil)

ctx.saveGState()
ctx.setShadow(offset: CGSize(width: 0, height: -12), blur: 28, color: CGColor(gray: 0, alpha: 0.35))
ctx.addPath(shape)
ctx.setFillColor(CGColor(srgbRed: 0.2, green: 0.4, blue: 0.5, alpha: 1))
ctx.fillPath()
ctx.restoreGState()

ctx.saveGState()
ctx.addPath(shape)
ctx.clip()
let gradient = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: [
    CGColor(srgbRed: 0.42, green: 0.68, blue: 0.76, alpha: 1),
    CGColor(srgbRed: 0.16, green: 0.33, blue: 0.45, alpha: 1),
] as CFArray, locations: [0, 1])!
ctx.drawLinearGradient(gradient, start: CGPoint(x: 0, y: tile.maxY), end: CGPoint(x: 0, y: tile.minY), options: [])
// Soft spotlight behind the figure.
let glow = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: [
    CGColor(srgbRed: 1, green: 0.95, blue: 0.85, alpha: 0.35), CGColor(srgbRed: 1, green: 0.95, blue: 0.85, alpha: 0),
] as CFArray, locations: [0, 1])!
ctx.drawRadialGradient(glow, startCenter: CGPoint(x: 512, y: 560), startRadius: 0,
                       endCenter: CGPoint(x: 512, y: 560), endRadius: 420, options: [])
// Crop the render to the figure (alpha > 0) and fit it into the tile.
let fig = crop(figure, alphaAbove: 40)
let h = 700.0, w = h * Double(fig.width) / Double(fig.height)
let figRect = CGRect(x: 512 - w / 2, y: 150, width: w, height: h)
// Contact shadow under the feet.
let floorShadow = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: [
    CGColor(gray: 0, alpha: 0.4), CGColor(gray: 0, alpha: 0),
] as CFArray, locations: [0, 1])!
ctx.saveGState()
ctx.translateBy(x: 512, y: figRect.minY + 8)
ctx.scaleBy(x: 1, y: 0.16)
ctx.drawRadialGradient(floorShadow, startCenter: .zero, startRadius: 0, endCenter: .zero, endRadius: w * 0.42, options: [])
ctx.restoreGState()
ctx.draw(fig, in: figRect)
ctx.restoreGState()

try ClayExport.writePNG(ctx.makeImage()!, to: outURL)
print("wrote \(outURL.path)")

func crop(_ image: CGImage, alphaAbove threshold: UInt8) -> CGImage {
    let w = image.width, h = image.height
    let c = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    c.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    let px = c.data!.assumingMemoryBound(to: UInt8.self)
    var minX = w, minY = h, maxX = 0, maxY = 0
    for y in 0..<h { for x in 0..<w where px[(y * w + x) * 4 + 3] > threshold {
        minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
    } }
    guard maxX > minX, maxY > minY else { return image }
    // Rows in the bitmap buffer are top-down, matching CGImage.cropping coordinates.
    return image.cropping(to: CGRect(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1)) ?? image
}
