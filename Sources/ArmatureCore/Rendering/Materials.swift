import AppKit
import SceneKit
import simd

public struct ClayStyle: Sendable {
    /// Colour set used by the finishes that take a colour (clay, smooth, plastic, ceramic, wireframe).
    public enum Palette: String, CaseIterable, Sendable {
        case terracotta, stone, plasticine
    }

    /// Surface finish of the figures.
    public enum Finish: String, CaseIterable, Sendable {
        case clay, smooth, plastic, ceramic, marble, bronze, chrome, wood, wireframe

        public var displayName: String {
            switch self {
            case .clay: "Clay"
            case .smooth: "Smooth Clay"
            case .plastic: "Plastic"
            case .ceramic: "Glazed Ceramic"
            case .marble: "Marble"
            case .bronze: "Bronze"
            case .chrome: "Chrome"
            case .wood: "Carved Wood"
            case .wireframe: "Wireframe"
            }
        }

        /// Whether the palette affects this finish.
        public var usesPalette: Bool {
            switch self {
            case .clay, .smooth, .plastic, .ceramic, .wireframe: true
            case .marble, .bronze, .chrome, .wood: false
            }
        }
    }

    public var palette: Palette = .plasticine
    public var finish: Finish = .clay
    /// Loop-subdivision level applied on the GPU (0 = raw SMPL mesh).
    public var subdivision = 2
    /// Strength of the procedural thumb/tool marks on the clay finish.
    public var toolMarks: Float = 1.0

    public init() {}

    public func color(forPerson index: Int) -> NSColor {
        switch palette {
        case .terracotta: return NSColor(srgbRed: 0.72, green: 0.40, blue: 0.28, alpha: 1)
        case .stone: return NSColor(srgbRed: 0.64, green: 0.62, blue: 0.58, alpha: 1)
        case .plasticine:
            let colors: [(CGFloat, CGFloat, CGFloat)] = [
                (0.80, 0.45, 0.32), (0.45, 0.60, 0.74), (0.58, 0.68, 0.48), (0.88, 0.70, 0.36),
                (0.80, 0.52, 0.58), (0.60, 0.52, 0.72), (0.40, 0.66, 0.64), (0.68, 0.64, 0.58),
            ]
            let c = colors[index % colors.count]
            return NSColor(srgbRed: c.0, green: c.1, blue: c.2, alpha: 1)
        }
    }

    /// `centre` is the figure's centre in scene space (used to align wood grain to the body).
    func makeMaterial(person: Int, centre: SIMD3<Float>) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        m.metalness.contents = 0.0
        let color = color(forPerson: person)
        switch finish {
        case .clay:
            m.diffuse.contents = color
            m.roughness.contents = 0.62
            if toolMarks > 0 {
                m.shaderModifiers = [.surface: Shaders.clay]
                m.setValue(NSNumber(value: 14.0), forKey: "clayNoiseScale")
                m.setValue(NSNumber(value: 0.3 * toolMarks), forKey: "clayNoiseStrength")
            }
        case .smooth:
            m.diffuse.contents = color
            m.roughness.contents = 0.55
        case .plastic:
            m.diffuse.contents = color.saturated(by: 1.25)
            m.roughness.contents = 0.32
            m.clearCoat.contents = 0.5
            m.clearCoatRoughness.contents = 0.12
        case .ceramic:
            m.diffuse.contents = color.blended(withFraction: 0.3, of: .white) ?? color
            m.roughness.contents = 0.25
            m.clearCoat.contents = 1.0
            m.clearCoatRoughness.contents = 0.03
        case .marble:
            m.diffuse.contents = NSColor(srgbRed: 0.93, green: 0.92, blue: 0.89, alpha: 1)
            m.roughness.contents = 0.22
            m.clearCoat.contents = 0.6
            m.clearCoatRoughness.contents = 0.05
            m.shaderModifiers = [.surface: Shaders.marble]
        case .bronze:
            m.diffuse.contents = NSColor(srgbRed: 0.66, green: 0.44, blue: 0.24, alpha: 1)
            m.metalness.contents = 1.0
            m.roughness.contents = 0.32
            m.shaderModifiers = [.surface: Shaders.bronze]
        case .chrome:
            m.diffuse.contents = NSColor(white: 0.96, alpha: 1)
            m.metalness.contents = 1.0
            m.roughness.contents = 0.06
        case .wood:
            m.diffuse.contents = NSColor(srgbRed: 0.62, green: 0.44, blue: 0.27, alpha: 1)
            m.roughness.contents = 0.5
            m.clearCoat.contents = 0.25
            m.clearCoatRoughness.contents = 0.3
            m.shaderModifiers = [.surface: Shaders.wood]
            m.setValue(NSValue(scnVector3: SCNVector3(centre.x, centre.y, centre.z)), forKey: "woodCentre")
        case .wireframe:
            m.lightingModel = .constant
            m.diffuse.contents = color.blended(withFraction: 0.35, of: .black) ?? color
            m.fillMode = .lines
            m.isDoubleSided = true
        }
        return m
    }
}

private extension NSColor {
    func saturated(by factor: CGFloat) -> NSColor {
        guard let c = usingColorSpace(.sRGB) else { return self }
        return NSColor(hue: c.hueComponent, saturation: min(1, c.saturationComponent * factor),
                       brightness: c.brightnessComponent, alpha: 1)
    }
}

/// SceneKit surface shader modifiers. Colours written in the shaders are linear, not sRGB. Helper functions must precede `#pragma arguments`,
/// or SceneKit parses them as uniforms. Positions are sampled in node space, so patterns stick to the body.
enum Shaders {
    static let noise = """
    float clayHash(float3 p) {
        p = fract(p * 0.3183099 + 0.1);
        p *= 17.0;
        return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
    }
    float clayNoise(float3 x) {
        float3 i = floor(x);
        float3 f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        return mix(mix(mix(clayHash(i + float3(0,0,0)), clayHash(i + float3(1,0,0)), f.x),
                       mix(clayHash(i + float3(0,1,0)), clayHash(i + float3(1,1,0)), f.x), f.y),
                   mix(mix(clayHash(i + float3(0,0,1)), clayHash(i + float3(1,0,1)), f.x),
                       mix(clayHash(i + float3(0,1,1)), clayHash(i + float3(1,1,1)), f.x), f.y), f.z);
    }
    float clayFbm(float3 p) {
        return 0.6 * clayNoise(p) + 0.3 * clayNoise(p * 2.07 + 7.1) + 0.1 * clayNoise(p * 5.3 + 3.3);
    }

    """

    static let nodePosition = """
    float3 p = (scn_node.inverseModelViewTransform * float4(_surface.position, 1.0)).xyz;

    """

    /// Dents the normals with fBm (thumb/tool marks) and mottles the albedo.
    static let clay = noise + """
    #pragma arguments
    float clayNoiseScale;
    float clayNoiseStrength;

    #pragma body

    """ + nodePosition + """
    float3 pw = p * clayNoiseScale;
    float e = 0.08;
    float n0 = clayFbm(pw);
    float3 g = float3(clayFbm(pw + float3(e, 0, 0)) - n0,
                      clayFbm(pw + float3(0, e, 0)) - n0,
                      clayFbm(pw + float3(0, 0, e)) - n0) / e;
    float3 gv = (scn_node.modelViewTransform * float4(g, 0.0)).xyz;
    float3 nrm = _surface.normal;
    _surface.normal = normalize(nrm - clayNoiseStrength * (gv - dot(gv, nrm) * nrm));
    _surface.diffuse.rgb *= 0.93 + 0.14 * n0;
    """

    /// Turbulent sine veins over a cloudy white base.
    static let marble = noise + """
    #pragma body

    """ + nodePosition + """
    float n = clayFbm(p * 5.0);
    float v = abs(sin((p.x * 0.8 + p.y + p.z * 0.4) * 7.0 + n * 7.0));
    float vein = 1.0 - smoothstep(0.0, 0.16, v);
    float fine = 1.0 - smoothstep(0.0, 0.03, abs(sin((p.x - p.z) * 23.0 + clayFbm(p * 11.0) * 9.0)));
    float3 base = _surface.diffuse.rgb * (0.94 + 0.08 * n);
    base = mix(base, float3(0.11, 0.115, 0.13), vein * 0.8);
    base = mix(base, float3(0.3, 0.3, 0.32), fine * 0.5);
    _surface.diffuse.rgb = base;
    """

    /// Polished bronze with verdigris patina settling in blotches.
    static let bronze = noise + """
    #pragma body

    """ + nodePosition + """
    float n = clayFbm(p * 6.0);
    float patina = smoothstep(0.58, 0.8, n) * 0.85;
    _surface.diffuse.rgb = mix(_surface.diffuse.rgb * (0.85 + 0.3 * clayNoise(p * 40.0)), float3(0.07, 0.19, 0.14), patina);
    _surface.metalness = mix(1.0, 0.05, patina);
    _surface.roughness = mix(_surface.roughness, 0.85, patina);
    """

    /// Growth rings around the body's vertical axis, as if carved from a single log.
    static let wood = noise + """
    #pragma arguments
    float3 woodCentre;

    #pragma body

    """ + nodePosition + """
    float3 q = p - woodCentre;
    float n = clayFbm(q * 3.0);
    float rings = fract(length(q.xz) * 30.0 + n * 2.5 + q.y * 0.6);
    float t = smoothstep(0.05, 0.2, rings) * (1.0 - smoothstep(0.75, 0.95, rings));
    float grain = clayNoise(q * float3(6.0, 90.0, 6.0));
    float3 dark = float3(0.09, 0.035, 0.012);
    float3 light = float3(0.4, 0.19, 0.07);
    _surface.diffuse.rgb = mix(dark, light, t) * (0.88 + 0.22 * grain);
    """
}
