import Accelerate
import Foundation

/// BRISQUE (original) and NIQE (original) equations and parameters from PyIQA 0.1.14.1.
/// These statistical algorithms don't need neural-network conversion.
enum QualityStatistics {
    struct Parameters: Decodable {
        let sv: [[Double]]?
        let coefficients: [Double]?
        let gamma: Double?
        let rho: Double?
        let mean: [Double]?
        let covariance: [[Double]]?
    }
    static let alphas = (0...9800).map { 0.2 + Double($0) * 0.001 }
    static let ratios = alphas.map { exp(2 * lgamma(2 / $0) - lgamma(1 / $0) - lgamma(3 / $0)) }
    static let ranges: [(Double, Double)] = [
        (0.338,10),(0.017204,0.806612),(0.236,1.642),(-0.123884,0.20293),(0.000155,0.712298),(0.001122,0.470257),
        (0.244,1.641),(-0.123586,0.179083),(0.000152,0.710456),(0.000975,0.470984),
        (0.249,1.555),(-0.135687,0.100858),(0.000174,0.684173),(0.000913,0.534174),
        (0.258,1.561),(-0.143408,0.100486),(0.000179,0.685696),(0.000888,0.536508),
        (0.471,3.264),(0.012809,0.703171),(0.218,1.046),(-0.094876,0.187459),(0.000015,0.442057),(0.001272,0.40803),
        (0.222,1.042),(-0.115772,0.162604),(0.000016,0.444362),(0.001374,0.40243),
        (0.227,0.996),(-0.117188,0.098323),(0.00003,0.531903),(0.001122,0.369589),
        (0.228,0.99),(-0.12243,0.098658),(0.000028,0.530092),(0.001118,0.370399)]

    static func score(_ rgb: QualityPixels, id: String, parameters: URL) throws -> Double {
        let params = try JSONDecoder().decode(Parameters.self, from: Data(contentsOf: parameters))
        let n = rgb.width * rgb.height
        var luma = [Double](repeating: 0, count: n)
        for i in 0..<n {
            // PyIQA converts to float32 YIQ, then rounds, even for BRISQUE.
            let r = Float(rgb.values[i]), g = Float(rgb.values[n+i]), b = Float(rgb.values[2*n+i])
            luma[i] = Double(((r * 0.299 + g * 0.587 + b * 0.114) * 255).rounded(.toNearestOrEven))
        }
        var image = QualityPixels(values: luma, width: rgb.width, height: rgb.height, channels: 1)
        if id == "brisque" {
            guard image.width >= 4, image.height >= 4 else { throw QualityError.invalid("BRISQUE needs at least 4×4 pixels") }
            var features: [Double] = []
            for scale in 0..<2 {
                let normalized = normalize(image, replicate: false)
                features += nss(normalized, niqe: false)
                if scale == 0 { image = image.resized(width: (image.width+1)/2, height: (image.height+1)/2, filter: .matlab, antialias: true, scaleFactor: 0.5) }
            }
            guard features.allSatisfy(\.isFinite), let sv = params.sv, let coefficients = params.coefficients,
                  sv.count == coefficients.count, sv.allSatisfy({ $0.count == 36 }),
                  let gamma = params.gamma, let rho = params.rho else { throw QualityError.invalid("BRISQUE has degenerate features or invalid parameters") }
            let scaled = zip(features, ranges).map { -1 + 2 * ($0.0 - $0.1.0) / ($0.1.1 - $0.1.0) }
            var score = -rho
            for (vector, coefficient) in zip(sv, coefficients) {
                var distance = 0.0
                for i in 0..<36 { distance += pow(scaled[i] - vector[i], 2) }
                score += exp(-distance * gamma) * coefficient
            }
            return score
        }
        let rows = image.height / 96, cols = image.width / 96
        guard rows * cols >= 2 else { throw QualityError.invalid("NIQE needs at least two complete 96×96 blocks") }
        image = image.crop(x: 0, y: 0, width: cols * 96, height: rows * 96)
        var blocks = [[Double]](repeating: [], count: rows * cols)
        for scale in 0..<2 {
            let size = scale == 0 ? 96 : 48
            let normalized = normalize(image, replicate: true)
            for y in 0..<rows { for x in 0..<cols {
                blocks[y * cols + x] += nss(normalized.crop(x: x * size, y: y * size, width: size, height: size), niqe: true)
            } }
            if scale == 0 { image = image.resized(width: image.width / 2, height: image.height / 2, filter: .matlab, antialias: true, scaleFactor: 0.5) }
        }
        let valid = blocks.filter { $0.allSatisfy(\.isFinite) }
        guard valid.count >= 2, let reference = params.mean, reference.count == 36,
              let covariance = params.covariance, covariance.count == 36,
              covariance.allSatisfy({ $0.count == 36 }) else { throw QualityError.invalid("NIQE has insufficient valid blocks or invalid parameters") }
        let mean = (0..<36).map { i in valid.reduce(0) { $0 + $1[i] } / Double(valid.count) }
        var matrix = [Double](repeating: 0, count: 36 * 36)
        for i in 0..<36 { for j in 0..<36 {
            var sum = 0.0
            for block in valid { sum += (block[i] - mean[i]) * (block[j] - mean[j]) }
            matrix[j * 36 + i] = (covariance[i][j] + sum / Double(valid.count - 1)) / 2
        } }
        // Symmetric eigendecomposition gives the Moore–Penrose inverse, including singular covariances.
        var job: Int8 = 86, triangle: Int8 = 85, dim: Int32 = 36, lda: Int32 = 36, info: Int32 = 0
        var eigenvalues = [Double](repeating: 0, count: 36)
        var workSize: Int32 = -1, query = 0.0
        dsyev_(&job, &triangle, &dim, &matrix, &lda, &eigenvalues, &query, &workSize, &info)
        guard info == 0, query.isFinite, query > 0 else { throw QualityError.invalid("NIQE workspace query failed") }
        workSize = Int32(query)
        var work = [Double](repeating: 0, count: Int(workSize))
        dsyev_(&job, &triangle, &dim, &matrix, &lda, &eigenvalues, &work, &workSize, &info)
        guard info == 0 else { throw QualityError.invalid("NIQE covariance decomposition failed") }
        let threshold = (eigenvalues.map(abs).max() ?? 0) * 36 * Double.ulpOfOne
        var distance = 0.0
        for j in 0..<36 where abs(eigenvalues[j]) > threshold {
            var projection = 0.0
            for i in 0..<36 { projection += (reference[i] - mean[i]) * matrix[j * 36 + i] }
            distance += projection * projection / eigenvalues[j]
        }
        return sqrt(max(0, distance))
    }

    static func normalize(_ image: QualityPixels, replicate: Bool) -> QualityPixels {
        let w = image.width, h = image.height
        var kernel = (-3...3).map { exp(-Double($0 * $0) / (2 * pow(7.0/6, 2))) }
        let sum = kernel.reduce(0,+); kernel = kernel.map { $0/sum }
        func blur(_ values: [Double]) -> [Double] {
            var temp = values, out = values
            for y in 0..<h { for x in 0..<w {
                var s = 0.0
                for k in -3...3 {
                    let ix = x+k
                    if replicate || (ix >= 0 && ix < w) { s += values[y*w + min(w-1,max(0,ix))] * kernel[k+3] }
                }
                temp[y*w+x] = s
            } }
            for y in 0..<h { for x in 0..<w {
                var s = 0.0
                for k in -3...3 {
                    let iy = y+k
                    if replicate || (iy >= 0 && iy < h) { s += temp[min(h-1,max(0,iy))*w+x] * kernel[k+3] }
                }
                out[y*w+x] = s
            } }
            return out
        }
        let mean = blur(image.values), variance = blur(image.values.map { $0*$0 })
        let eps = replicate ? Double.ulpOfOne : Double(Float.ulpOfOne)
        let out = image.values.indices.map { (image.values[$0] - mean[$0]) / (sqrt(abs(variance[$0] - mean[$0]*mean[$0]) + eps) + 1) }
        return QualityPixels(values: out, width: w, height: h, channels: 1)
    }

    static func alpha(_ target: Double, reciprocal: Bool = false) -> Double {
        guard target.isFinite else { return 0.2 }
        var best = Double.infinity, index = 0
        for i in ratios.indices {
            let error = abs((reciprocal ? 1/ratios[i] : ratios[i]) - target)
            if error < best { best = error; index = i }
        }
        return alphas[index]
    }
    static func aggd(_ x: [Double]) -> (Double, Double, Double) {
        var left = 0.0, right = 0.0, nl = 0.0, nr = 0.0, absolute = 0.0
        for v in x {
            absolute += abs(v)
            if v < 0 { left += v*v; nl += 1 }
            if v > 0 { right += v*v; nr += 1 }
        }
        let l = sqrt(left/nl), r = sqrt(right/nr), g = l/r
        let rhat = absolute*absolute / (Double(x.count) * (left+right))
        return (alpha(rhat * (pow(g,3)+1)*(g+1)/pow(g*g+1,2)), l, r)
    }
    static func nss(_ image: QualityPixels, niqe: Bool) -> [Double] {
        let x = image.values, w = image.width, h = image.height
        var out: [Double]
        if niqe {
            let (a,l,r) = aggd(x)
            out = [a, (l+r)/2 * exp((lgamma(1/a)-lgamma(3/a))/2)]
        } else {
            let variance = x.reduce(0) { $0+$1*$1 }/Double(x.count)
            let mean = x.reduce(0) { $0+abs($1) }/Double(x.count)
            out = [alpha(variance/(mean*mean), reciprocal: true), variance]
        }
        let shifts = niqe ? [(0,1),(1,0),(1,1),(1,-1)] : [(0,1),(1,0),(1,1),(-1,1)]
        for (dy,dx) in shifts {
            var product = x
            for y in 0..<h { for col in 0..<w { product[y*w+col] *= x[((y-dy+h)%h)*w + (col-dx+w)%w] } }
            let (a,l,r) = aggd(product)
            if niqe {
                let factor = exp((lgamma(1/a)-lgamma(3/a))/2)
                out += [a, (r-l)*factor*exp(lgamma(2/a)-lgamma(1/a)), l*factor, r*factor]
            } else {
                out += [a, (r-l)*exp(lgamma(2/a)-(lgamma(1/a)+lgamma(3/a))/2), l*l, r*r]
            }
        }
        return out
    }
}
