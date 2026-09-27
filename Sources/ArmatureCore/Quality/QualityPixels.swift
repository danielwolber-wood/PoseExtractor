import CoreGraphics
import CoreML
import Foundation

/// Planar RGB in [0,1], with explicit PyTorch-compatible resampling.
struct QualityPixels {
    var values: [Double]
    let width: Int
    let height: Int
    let channels: Int

    init(_ image: CGImage) throws {
        width = image.width; height = image.height; channels = 3; values = []
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let ok = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard let ctx = CGContext(data: bytes.baseAddress, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.setFillColor(CGColor(gray: 1, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard ok else { throw QualityError.invalid("Cannot decode image pixels") }
        values = [Double](repeating: 0, count: width * height * 3)
        for c in 0..<3 { for i in 0..<(width * height) { values[c * width * height + i] = Double(rgba[i * 4 + c]) / 255 } }
    }

    init(values: [Double], width: Int, height: Int, channels: Int) {
        self.values = values; self.width = width; self.height = height; self.channels = channels
    }

    func crop(x: Int, y: Int, width w: Int, height h: Int) -> QualityPixels {
        var out = [Double](repeating: 0, count: w * h * channels)
        for c in 0..<channels { for row in 0..<h {
            let start = c * width * height + (y + row) * width + x
            out.replaceSubrange((c * w * h + row * w)..<(c * w * h + (row + 1) * w), with: values[start..<(start + w)])
        } }
        return QualityPixels(values: out, width: w, height: h, channels: channels)
    }

    enum Filter { case linear, cubic, matlab }
    func resized(width w: Int, height h: Int, filter: Filter = .linear, antialias: Bool = false, scaleFactor: Double? = nil) -> QualityPixels {
        if w == width && h == height { return self }
        func weights(_ source: Int, _ target: Int) -> [[(Int, Double)]] {
            let scale = scaleFactor.map { 1 / $0 } ?? (Double(source) / Double(target))
            let support = filter == .linear ? 1.0 : 2.0
            let aa = antialias ? max(1, scale) : 1
            func kernel(_ d: Double) -> Double {
                let t = abs(d)
                if filter == .linear { return max(0, 1 - t) }
                let a = filter == .matlab ? -0.5 : -0.75
                if t <= 1 { return (a + 2) * t * t * t - (a + 3) * t * t + 1 }
                if t < 2 { return a * t * t * t - 5 * a * t * t + 8 * a * t - 4 * a }
                return 0
            }
            return (0..<target).map { i in
                let center = (Double(i) + 0.5) * scale - 0.5
                let lo = Int(ceil(center - support * aa)), hi = Int(floor(center + support * aa))
                var result: [(Int, Double)] = []
                var total = 0.0
                for j in lo...hi {
                    let v = kernel((center - Double(j)) / aa)
                    var index = j
                    if filter == .matlab {
                        while index < 0 || index >= source { index = index < 0 ? -index - 1 : 2 * source - index - 1 }
                    } else { index = min(source - 1, max(0, index)) }
                    result.append((index, v)); total += v
                }
                return result.map { ($0.0, $0.1 / total) }
            }
        }
        let xs = weights(width, w), ys = weights(height, h)
        var tmp = [Double](repeating: 0, count: w * height * channels)
        var out = [Double](repeating: 0, count: w * h * channels)
        for c in 0..<channels { for y in 0..<height { for x in 0..<w {
            var sum = 0.0
            for (i, k) in xs[x] { sum += values[c * width * height + y * width + i] * k }
            tmp[c * w * height + y * w + x] = sum
        } } }
        for c in 0..<channels { for y in 0..<h { for x in 0..<w {
            var sum = 0.0
            for (i, k) in ys[y] { sum += tmp[c * w * height + i * w + x] * k }
            out[c * w * h + y * w + x] = sum
        } } }
        return QualityPixels(values: out, width: w, height: h, channels: channels)
    }

    func tensor() throws -> MLMultiArray { try Self.tensor(values, shape: [1, channels, height, width]) }
    static func tensor(_ values: [Double], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        let p = array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in values.indices { p[i] = Float(values[i]) }
        return array
    }

    func musiqPatches() throws -> MLMultiArray {
        let nativeCount = ((width + 31) / 32) * ((height + 31) / 32)
        guard nativeCount + 193 <= 16384 else { throw QualityError.invalid("MUSIQ supports at most 16,384 patches; image is too large") }
        var patches: [Double] = []
        patches.reserveCapacity((193 + nativeCount) * 3075)
        for scale in 0..<3 {
            let image: QualityPixels
            let count: Int
            if scale < 2 {
                let side = scale == 0 ? 224 : 384
                let ratio = Double(side) / Double(max(width, height))
                image = resized(width: max(1, Int((Double(width) * ratio).rounded(.toNearestOrEven))),
                                height: max(1, Int((Double(height) * ratio).rounded(.toNearestOrEven))), filter: .cubic)
                count = scale == 0 ? 49 : 144
            } else { image = self; count = nativeCount }
            let rows = (image.height + 31) / 32, cols = (image.width + 31) / 32
            let top = (rows * 32 - image.height) / 2, left = (cols * 32 - image.width) / 2
            for index in 0..<count {
                if index >= rows * cols { patches += [Double](repeating: 0, count: 3075); continue }
                let row = index / cols, col = index % cols
                for c in 0..<3 { for y in 0..<32 { for x in 0..<32 {
                    let iy = row * 32 + y - top, ix = col * 32 + x - left
                    let v = iy >= 0 && iy < image.height && ix >= 0 && ix < image.width
                        ? (image.values[c * image.width * image.height + iy * image.width + ix] - 0.5) * 2 : 0
                    patches.append(v)
                } } }
                patches += [Double((row * 10 / rows) * 10 + col * 10 / cols), Double(scale), 1]
            }
        }
        return try Self.tensor(patches, shape: [1, nativeCount + 193, 3075])
    }
}
