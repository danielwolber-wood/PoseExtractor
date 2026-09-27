import os, math
OUT = "/Users/daniel/Temp/PoseEstimation/design/icon"
BG, CLAY, WIRE = "#1F4E5F", "#D9774B", "#F6EFE6"

def svg(body): return f'<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">{body}</svg>'
def circ(x, y, r):  # Icon Composer ignores <circle> cx/cy: draw as arcs
    return f"M{x-r} {y} a{r} {r} 0 1 0 {2*r} 0 a{r} {r} 0 1 0 {-2*r} 0 Z "
def diamond(x, y, r): return f"M{x} {y-r} L{x+r} {y} L{x} {y+r} L{x-r} {y} Z "
def square(x, y, r):
    return f"M{x-r} {y-r} L{x+r} {y-r} L{x+r} {y+r} L{x-r} {y+r} Z "
def octagon(x, y, r):
    pts = [(x + r*math.cos(math.radians(22.5+45*i)), y + r*math.sin(math.radians(22.5+45*i))) for i in range(8)]
    return "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in pts) + " Z "
def lines(*chains): return "".join("M" + " L".join(f"{x} {y}" for x, y in c) + " " for c in chains)
def stroke(d, c, w, cap="round", join="round"):
    return f'<path d="{d.strip()}" fill="none" stroke="{c}" stroke-width="{w}" stroke-linecap="{cap}" stroke-linejoin="{join}"/>'
def fill(d, c): return f'<path fill="{c}" d="{d.strip()}"/>'
def poly(pts): return "M" + " L".join(f"{x} {y}" for x, y in pts) + " Z "

# Shared pose (1024 canvas)
H  = (512, 240)                      # head centre
N  = (512, 372)                      # single neck / upper-chest node
SL, SR = (418, 372), (606, 372)      # shoulders
EL, ER = (338, 274), (686, 274)      # elbows
WL, WR = (280, 180), (744, 180)      # wrists
P  = (512, 604)                      # pelvis centre
HL, HR = (464, 604), (560, 604)      # hips
KL, KR = (430, 748), (594, 748)      # knees
AL, AR = (398, 866), (626, 866)      # ankles

arms = [(WL, EL, SL, SR, ER, WR)]
legs = [(AL, KL, HL), (HR, KR, AR)]

def write(name, clay, wire, notes):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    open(f"{d}/0-background.svg", "w").write(svg(f'<rect width="1024" height="1024" fill="{BG}"/>'))
    open(f"{d}/1-clay.svg", "w").write(svg(clay))
    open(f"{d}/2-wire.svg", "w").write(svg(wire))
    open(f"{d}/NOTES.txt", "w").write(notes + "\n")

# ---- A: Hip bar — two hip nodes + short connector, single neck node, capsule body slimmed
def A():
    clay = stroke(lines(*arms, *legs, (HL, HR)), CLAY, 96) \
         + fill(poly([(430, 350), (594, 350), (572, 612), (452, 612)]), CLAY) \
         + fill(circ(*H, 80), CLAY)
    w = stroke(lines(*arms, *legs, (HL, HR), (H, N, (512, 604))), WIRE, 22)
    w += fill("".join(circ(*p, 28) for p in [H, N, SL, SR, EL, ER, WL, WR, HL, HR, KL, KR, AL, AR]), WIRE)
    return clay, w
# ---- A2: Hip bar with a straight-sided (untapered) torso
def A2():
    clay = stroke(lines(*arms, *legs, (HL, HR)), CLAY, 96) \
         + fill(poly([(432, 350), (592, 350), (592, 612), (432, 612)]), CLAY) \
         + fill(circ(*H, 80), CLAY)
    w = stroke(lines(*arms, *legs, (HL, HR), (H, N, (512, 604))), WIRE, 22)
    w += fill("".join(circ(*p, 28) for p in [H, N, SL, SR, EL, ER, WL, WR, HL, HR, KL, KR, AL, AR]), WIRE)
    return clay, w
# ---- B: Central pelvis — pelvis node + two wider-set hips, trapezoid torso
def B():
    hl, hr = (444, 612), (580, 612)
    lg = [(AL, KL, hl), (hr, KR, AR)]
    clay = stroke(lines(*arms, *lg), CLAY, 96) \
         + fill(poly([(412, 336), (612, 336), (580, 640), (444, 640)]), CLAY) \
         + fill(circ(*H, 80), CLAY)
    w = stroke(lines(*arms, *lg, (hl, P, hr), (H, N, P)), WIRE, 22)
    w += fill("".join(circ(*p, 26) for p in [H, N, SL, SR, EL, ER, WL, WR, hl, hr, KL, KR, AL, AR]) + circ(*P, 34), WIRE)
    return clay, w
# ---- C: Faceted — constructed figure: mitred segments, octagon head, hexagonal torso
def C():
    clay = stroke(lines(*arms, *legs), CLAY, 96, cap="square", join="miter") \
         + fill(poly([(418, 332), (606, 332), (640, 372), (584, 628), (440, 628), (384, 372)]), CLAY) \
         + fill(octagon(*H, 84), CLAY)
    w = stroke(lines(*arms, *legs, (HL, HR), (H, N, (512, 604))), WIRE, 22, join="miter")
    w += fill("".join(circ(*p, 28) for p in [H, N, SL, SR, EL, ER, WL, WR, HL, HR, KL, KR, AL, AR]), WIRE)
    return clay, w
# ---- D: Keypoints — diamond joint markers (CV keypoint style), thin constructed body
def D():
    clay = stroke(lines(*arms, *legs, (HL, HR)), CLAY, 96, join="miter") \
         + fill(poly([(440, 356), (584, 356), (566, 612), (458, 612)]), CLAY) \
         + fill(circ(*H, 80), CLAY)
    w = stroke(lines(*arms, *legs, (HL, HR), (H, N, (512, 604))), WIRE, 22, join="miter")
    w += fill("".join(diamond(*p, 36) for p in [H, N, SL, SR, EL, ER, WL, WR, HL, HR, KL, KR, AL, AR]), WIRE)
    return clay, w
# ---- E: Rig-first — heavy skeleton, body as a flat block silhouette behind it
def E():
    clay = stroke(lines(*arms, *legs, (HL, HR), (N, (512, 604))), CLAY, 108, cap="square", join="miter") \
         + fill(poly([(418, 332), (606, 332), (606, 412), (560, 628), (464, 628), (418, 412)]), CLAY) \
         + fill(circ(*H, 84), CLAY)
    w = stroke(lines(*arms, *legs, (HL, HR), (H, N, (512, 604))), WIRE, 30)
    w += fill("".join(circ(*p, 32) for p in [H, N, SL, SR, EL, ER, WL, WR, HL, HR, KL, KR, AL, AR]), WIRE)
    return clay, w

V = {
 "armature-a-hipbar":   (A, "Two hip nodes + short connector; one neck node; wire 24/node r32; slimmer capsule limbs, straight-sided torso."),
 "armature-a2-hipbar-straight": (A2, "As A, but with a straight-sided rectangular torso (no taper)."),
 "armature-b-pelvis":   (B, "Central pelvis node (larger) + two hips; one neck node; trapezoid torso."),
 "armature-c-faceted":  (C, "Constructed/faceted body: square-capped mitred limbs, hexagonal torso, octagon head."),
 "armature-d-keypoint": (D, "Diamond keypoint markers (CV-tracker look); mitred body."),
 "armature-e-rig":      (E, "Skeleton-first: wire 30 / node r32 on a wider square-capped block body."),
}
for n, (f, note) in V.items():
    c, w = f(); write(n, c, w, note)

# preview sheet
rows = []
for n in ["armature"] + list(V):
    d = os.path.join(OUT, n)
    layers = "".join(open(f"{d}/{l}").read().split(">", 1)[1].rsplit("</svg>", 1)[0] for l in ["0-background.svg", "1-clay.svg", "2-wire.svg"])
    tile = lambda s: f'<svg width="{s}" height="{s}" viewBox="0 0 1024 1024" style="border-radius:{s*0.225}px">{layers}</svg>'
    rows.append(f'<div class="row"><h3>{n}</h3>{tile(320)}{tile(64)}{tile(32)}{tile(16)}</div>')
open("/private/tmp/claude-501/-Users-daniel-Temp-PoseEstimation/8dec879f-4641-469e-8744-043003b0077b/scratchpad/sheet.html","w").write(
 "<html><body style='background:#ddd;font-family:-apple-system;margin:16px'><style>.row{display:flex;align-items:end;gap:20px;margin-bottom:12px}h3{width:170px;font-size:13px}</style>"+"".join(rows)+"</body></html>")
