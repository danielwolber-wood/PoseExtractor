#!/usr/bin/env python3
"""One-time conversion of body models into the flat format the Swift app memory-maps.

Every model is written in one canonical frame -- y up, facing +z, +x on the body's left, metres -- as
a linear model (template + shape blend shapes [+ pose correctives]), a skeleton (rest joints that are
linear in the shape coefficients), dense skinning weights, and a *rig description* telling the fitter
which joint is which, where each keypoint lives on the body, how stiff each joint is, which joints
are hinges, each joint's anatomical range of motion, and capsules approximating the body for self-collision.

Models:
  SMPL    (SMPL_python_v.1.1.0.zip)   -- Loper et al. 2015, MPI non-commercial licence
  SMPL-X  (models_smplx_v1_1.zip)     -- Pavlakos et al. 2019, MPI non-commercial licence; adds fingers + jaw
  Anny    (pip package `anny`)        -- NAVER LABS 2025, Apache-2.0, built on CC0 MakeHuman assets;
                                         all ages. Its non-linear phenotype space (age, gender, weight, ...)
                                         is linearised here by PCA over realistic sampled bodies.

Output layout (one directory per model):
    Models/<id>/meta.json   {"arrays": {name: {"dtype", "shape", "offset"}}, "rig": {...}, ...}
    Models/<id>/data.bin    little-endian float32 / uint32 blobs

Usage:
    uv run --with numpy --with scipy tools/convert_models.py                 # SMPL (+ SMPL-X if its zip is present)
    uv run --python 3.12 --with anny --with numpy --with scipy tools/convert_models.py --anny
"""
import argparse
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

NUM_BETAS = 10
ROOT = Path(__file__).resolve().parent.parent
ARCHIVES = ROOT / "data" / "archives"

# ---------------------------------------------------------------------------------------------------
# Output


def write_model(out_dir: Path, arrays: dict, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"arrays": {}, **meta}
    offset = 0
    with open(out_dir / "data.bin", "wb") as fh:
        for name, arr in arrays.items():
            arr = np.ascontiguousarray(arr)
            if arr.dtype.kind == "f":
                arr, dtype = arr.astype("<f4"), "f32"
            else:
                arr, dtype = arr.astype("<u4"), "u32"
            fh.write(arr.tobytes())
            meta["arrays"][name] = {"dtype": dtype, "shape": list(arr.shape), "offset": offset}
            offset += arr.nbytes
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  wrote {out_dir} ({offset / 1e6:.1f} MB)")


def linear_model_arrays(v_template, shapedirs, joints_template, joints_shapedirs, weights, faces, parents,
                        posedirs=None):
    arrays = {
        "v_template": v_template,              # (V, 3)
        "shapedirs": shapedirs,                # (V, 3, B)
        "joints_template": joints_template,    # (J, 3)
        "joints_shapedirs": joints_shapedirs,  # (J, 3, B)
        "weights": weights,                    # (V, J) dense
        "faces": faces.astype(np.uint32),      # (F, 3)
        "parents": np.array([p if p >= 0 else 0xFFFFFFFF for p in parents], dtype=np.uint32),
    }
    if posedirs is not None:
        arrays["posedirs"] = posedirs          # (V, 3, 9 * (J - 1))
    return arrays


# ---------------------------------------------------------------------------------------------------
# Rig description


def spine_target(joints, spine_chain, pelvis, neck):
    """Vision's 'spine' point sits ~45% of the way from the hips to the neck; pick the nearest spine joint."""
    goal = joints[pelvis] + 0.45 * (joints[neck] - joints[pelvis])
    return int(min(spine_chain, key=lambda j: np.linalg.norm(joints[j] - goal)))


def top_of_head_vertex(v, weights, head_joints):
    head = weights[:, head_joints].sum(1) > 0.5
    idx = np.where(head)[0]
    return int(idx[np.argmax(v[idx, 1])])


def fingertip_vertex(v, weights, distal, proximal, joints):
    """Vertex furthest along the last finger segment, among those skinned to it."""
    d = joints[distal] - joints[proximal]
    d /= np.linalg.norm(d)
    idx = np.where(weights[:, distal] > 0.5)[0]
    return int(idx[np.argmax((v[idx] - joints[distal]) @ d)])


def similarity_transform(src, dst):
    """Umeyama: s, R, t with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def transfer_vertices(reference_points, anchors_ref, anchors_model, v_model, allowed):
    """Maps reference landmark positions into the model via the anchors' similarity transform, then snaps
    each to the nearest allowed model vertex."""
    s, R, t = similarity_transform(np.asarray(anchors_ref), np.asarray(anchors_model))
    idx = np.where(allowed)[0]
    out = []
    for p in reference_points:
        q = s * R @ p + t
        out.append(int(idx[np.argmin(np.linalg.norm(v_model[idx] - q, axis=1))]))
    return out


def surface(indices, weights=None):
    if weights is None:
        weights = [1.0] * len(indices)
    return {"vertices": [[int(i), float(w)] for i, w in zip(indices, weights) if w > 1e-6]}


def make_rig(joints, parents, names, sem, targets, extra_stiffness=None, hand_hinges=True):
    """Builds stiffness, hinges and silhouette joints from the semantic joint map.

    sem: semantic name -> joint index (or list for chains). targets: BodyJoint name -> spec.
    """
    J = len(parents)
    stiffness = np.full(J, 6.0)  # anything not named below (twist bones, toes, eyes, ...) stays near rest

    def setj(key, value):
        v = sem.get(key)
        if v is None:
            return
        for j in (v if isinstance(v, list) else [v]):
            stiffness[j] = value

    stiffness[sem["pelvis"]] = 0.0
    n = len(sem["spine"])
    setj("spine", 1.5 * np.sqrt(n / 3))           # same total stiffness to bend as SMPL's three spine joints
    setj("neck", 1.5 * np.sqrt(len(sem["neck"])))
    setj("head", 1.2)
    for side in ("left", "right"):
        setj(side + "Collar", 2.5)
        setj(side + "ShoulderBlade", 3.0)
        for k in ("Shoulder", "Elbow", "Hip", "Knee"):
            setj(side + k, 0.6)
        setj(side + "Wrist", 1.2)
        setj(side + "Ankle", 3.0)
        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            for i, j in enumerate(sem.get(side + finger, [])):
                stiffness[j] = 0.8 if i == 0 else 1.0
    # Jaw pinned closed: Vision's "chin" is the bottom of the face outline, below the mesh's chin landmark,
    # so a free jaw gets pulled open. (Fitting expressions would need proper mouth landmarks.)
    setj("jaw", 6.0)
    for j, v in (extra_stiffness or {}).items():
        stiffness[j] = v

    hinges = []
    forward = np.array([0.0, 0.0, 1.0])
    for side in ("left", "right"):
        hinges.append({"joint": sem[side + "Knee"], "child": sem[side + "Ankle"], "flex": (-forward).tolist(),
                       "twist": 4.0, "side": 4.0, "hyper": 8.0})
        hinges.append({"joint": sem[side + "Elbow"], "child": sem[side + "Wrist"], "flex": forward.tolist(),
                       "twist": 1.5, "side": 3.0, "hyper": 6.0})
        if hand_hinges and sem.get(side + "Index"):
            wrist = joints[sem[side + "Wrist"]]
            v1 = joints[sem[side + "Index"][0]] - wrist
            v2 = joints[sem[side + "Little"][0]] - wrist
            palmar = np.cross(v1, v2) * (-1 if side == "left" else 1)
            palmar /= np.linalg.norm(palmar)
            for finger in ("Index", "Middle", "Ring", "Little"):
                chain = sem[side + finger]
                for a, b in zip(chain, chain[1:] + [None]):
                    child = b if b is not None else a
                    if b is None:
                        continue
                    hinges.append({"joint": a, "child": child, "flex": palmar.tolist(),
                                   "twist": 2.0, "side": 1.0 if a == chain[0] else 3.0, "hyper": 3.0})

    for h in hinges:
        h["restFlex"] = rest_flex(joints, parents, h["joint"], h["child"], h["flex"])
        kind = next((k for k in MAX_FLEX if h["joint"] in (sem.get("left" + k), sem.get("right" + k))), None)
        h["maxFlex"] = round(float(np.radians(MAX_FLEX.get(kind, FINGER_MAX_FLEX))), 4)

    silhouette = [sem["pelvis"]] + sem["spine"] + sem["neck"] + [sem["head"]]
    for side in ("left", "right"):
        for k in ("Collar", "ShoulderBlade", "Shoulder", "Elbow", "Hip", "Knee"):
            if sem.get(side + k) is not None:
                silhouette.append(sem[side + k])

    return {
        "jointNames": names,
        "semantic": sem,
        "targets": targets,
        "stiffness": stiffness.round(3).tolist(),
        "hinges": hinges,
        "limits": joint_limits(joints, parents, sem),
        "silhouetteJoints": sorted(set(int(j) for j in silhouette)),
    }


# ---------------------------------------------------------------------------------------------------
# Range of motion and self-collision
#
# Anatomical limits, generous versions of clinical range-of-motion tables (AAOS): they exist to rule out
# impossible poses (a leg twisted backwards, the head turned 180°), not to judge athletic ones. Ball
# joints get a swing cone around an anatomical neutral direction -- four half-axes (flex/extend,
# abduct/adduct) joined by ellipse quadrants -- plus a symmetric twist limit about the bone. The neutral
# direction is anatomical, not the rest pose, so T-posed (SMPL) and A-posed (Anny) rigs get the same
# limits. Degrees here; radians in the rig.

MAX_FLEX = {"Knee": 155, "Elbow": 150}   # hinge flexion from straight
FINGER_MAX_FLEX = 110
# Whole-chain ranges, split over however many joints the model has in the chain: x1.5 so the fitter can
# bend one joint more than another, and at least 40% of the chain's range per joint (Anny has five spine
# joints). The chain total stays well short of impossible.
SPINE_ROM = {"flex": 90, "extend": 50, "lateral": 45, "twist": 60}
NECK_ROM = {"flex": 50, "extend": 60, "lateral": 45, "twist": 80}


def chain_share(n):
    return max(1.5 / n, 0.4)


def _as_list(v):
    return [] if v is None else (v if isinstance(v, list) else [v])


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def rest_flex(joints, parents, joint, child, flex_dir):
    """Hinge flexion already present in the rest pose (Anny's A-pose has bent elbows), in radians:
    the angle from the parent bone to the child bone, positive towards `flex_dir`."""
    p = _unit(joints[joint] - joints[parents[joint]])
    c = _unit(joints[child] - joints[joint])
    axis = np.cross(p, flex_dir)
    if np.linalg.norm(axis) < 1e-6:
        return 0.0
    return round(float(np.arctan2(np.dot(np.cross(p, c), _unit(axis)), np.dot(p, c))), 4)


def joint_limits(joints, parents, sem):
    """Swing-twist limits for ball joints. Each limit: joint, the rest bone (`child`, or `neutral` alone for
    a joint without one, e.g. SMPL's head), the anatomical neutral direction, the directions flexion and
    abduction move the bone towards (canonical frame: y up, +z forward, +x the body's left), and the angles."""
    up, fwd = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    rad = np.radians
    out = []

    def add(joint, child, neutral, flex_dir, abduct_dir, flex, extend, abduct, adduct, twist):
        neutral = _unit(neutral if neutral is not None else joints[child] - joints[joint])
        out.append({"joint": int(joint), "child": None if child is None else int(child),
                    "neutral": neutral.round(5).tolist(),
                    "flexDir": _unit(flex_dir).round(5).tolist(), "abductDir": _unit(abduct_dir).round(5).tolist(),
                    "flex": round(rad(flex), 4), "extend": round(rad(extend), 4),
                    "abduct": round(rad(abduct), 4), "adduct": round(rad(adduct), 4), "twist": round(rad(twist), 4)})

    def chain(js, next_joint, rom, lateral_dir):
        n = len(js)
        s = chain_share(n)
        for i, j in enumerate(js):
            child = js[i + 1] if i + 1 < n else next_joint
            add(j, child, None, fwd, lateral_dir, s * rom["flex"], s * rom["extend"], s * rom["lateral"],
                s * rom["lateral"], s * rom["twist"])

    x = np.array([1.0, 0.0, 0.0])
    # Spine and neck bones point up their chain; lateral bending is symmetric, so +x serves both sides.
    chain(sem["spine"], sem["neck"][0], SPINE_ROM, x)
    neck = sem["neck"] + [sem["head"]]
    n = len(neck)
    for i, j in enumerate(neck):
        s = chain_share(n)
        if j == sem["head"]:
            add(j, None, up, fwd, x, s * NECK_ROM["flex"], s * NECK_ROM["extend"], s * NECK_ROM["lateral"],
                s * NECK_ROM["lateral"], s * NECK_ROM["twist"])
        else:
            add(j, neck[i + 1], None, fwd, x, s * NECK_ROM["flex"], s * NECK_ROM["extend"], s * NECK_ROM["lateral"],
                s * NECK_ROM["lateral"], s * NECK_ROM["twist"])

    for side, lat in (("left", x), ("right", -x)):
        # Hip: neutral straight down; flexion swings the knee forward, abduction out to the side.
        add(sem[side + "Hip"], sem[side + "Knee"], -up, fwd, lat, 130, 40, 55, 35, 55)
        # Shoulder: neutral straight out to the side (the middle of its range). Forward/back is horizontal
        # flexion/extension, up/down is elevation. Wider than the glenohumeral joint alone: the arm reaches
        # further with the shoulder blade, and "back" must admit an arm hanging down behind the body.
        add(sem[side + "Shoulder"], sem[side + "Elbow"], lat, fwd, up, 140, 85, 100, 115, 105)
        # Collar (and Anny's shoulder blade): protraction/retraction and elevation/depression.
        for k in ("Collar", "ShoulderBlade"):
            j = sem.get(side + k)
            if j is not None:
                c = next(i for i, p in enumerate(parents) if p == j)
                add(j, c, None, fwd, up, 35, 30, 45, 20, 30)
        # Wrist: flexion towards the palm, abduction = radial deviation (towards the thumb).
        wrist = sem[side + "Wrist"]
        hand = sem[side + "Middle"][0] if sem.get(side + "Middle") else sem.get(side + "Hand")
        if hand is not None:
            if sem.get(side + "Index"):
                v1 = joints[sem[side + "Index"][0]] - joints[wrist]
                v2 = joints[sem[side + "Little"][0]] - joints[wrist]
                palmar = _unit(np.cross(v1, v2) * (-1 if side == "left" else 1))
                radial = _unit(v1 - v2)   # little finger -> index finger
            else:
                palmar, radial = -up, fwd  # SMPL: T-pose, palms down, thumbs forward
            add(wrist, hand, None, palmar, radial, 85, 80, 30, 45, 90)
        # Ankle: dorsiflexion lifts the toes; abduction turns them out; twist is inversion/eversion.
        ankle, foot = sem[side + "Ankle"], sem[side + "Foot"]
        add(ankle, foot, None, up, lat, 30, 55, 30, 30, 35)
    return out


def body_capsules(v, weights, joints, parents, sem, targets):
    """Capsules approximating the body for self-collision: torso segments along the spine, a head, and
    upper arm, forearm, hand, thigh, shin and foot on each side. Each end is a joint plus an offset in the
    rest frame; the radius is inscribed in the vertices skinned mostly to that segment, so capsules sit
    inside the flesh and ordinary contact (hands on hips, crossed arms) doesn't count as collision. Pairs
    already touching (or nearly) in the rest pose are excluded."""
    dominant = weights.argmax(1)

    def inscribed_radius(p, a, b, t_min=0.02):
        """Smallest of the (lower-quartile) surface distances in 8 directions around the segment: the capsule then
        stays inside the flesh on its thinnest side (a torso is much wider than it is deep)."""
        ab = b - a
        L2 = max(ab @ ab, 1e-12)
        t = np.clip(((p - a) @ ab) / L2, 0, 1)
        d = p - (a + t[:, None] * ab)
        axis = ab / np.sqrt(L2) if L2 > 1e-10 else np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        w = np.cross(axis, u)
        # Only vertices beside the segment, not beyond its ends (those are caps, or the next segment).
        mid = (t > t_min) & (t < 0.98) if L2 > 1e-10 else np.ones(len(p), bool)
        bins = ((np.arctan2(d @ w, d @ u) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        r = np.linalg.norm(d, axis=1)
        low = [np.percentile(r[mid & (bins == k)], 25) for k in range(8) if (mid & (bins == k)).sum() >= 5]
        return float(min(low)) if len(low) >= 6 else 0.03

    caps = []

    def add(name, ja, jb, owners, oa=None, ob=None, t_min=0.02):
        a = joints[ja] + (0 if oa is None else oa)
        b = joints[jb] + (0 if ob is None else ob)
        idx = np.where(np.isin(dominant, owners))[0]
        r = inscribed_radius(v[idx], a, b, t_min) if len(idx) else 0.03
        caps.append({"name": name, "a": int(ja), "b": int(jb),
                     "offsetA": (np.zeros(3) if oa is None else oa).round(4).tolist(),
                     "offsetB": (np.zeros(3) if ob is None else ob).round(4).tolist(), "radius": round(r, 4),
                     "_a": a, "_b": b})

    def descendants(j):
        out, frontier = [j], [j]
        while frontier:
            frontier = [i for i, p in enumerate(parents) if p in frontier]
            out += frontier
        return out

    # Torso: consecutive spine joints from the pelvis to the neck.
    spine = [sem["pelvis"]] + sem["spine"] + [sem["neck"][0]]
    spine = [j for i, j in enumerate(spine) if i == 0 or np.linalg.norm(joints[j] - joints[spine[i - 1]]) > 0.02]
    for i, (ja, jb) in enumerate(zip(spine, spine[1:])):
        add(f"torso{i}", ja, jb, [ja, jb])
    # One radius for the whole trunk: its segments are short and uneven (SMPL's spine2-spine3 is 5 cm), so
    # per-segment estimates scatter. Distances to the spine polyline, in 8 directions around the vertical,
    # from the pelvis up to the shoulder line (the neck and shoulders would pull it in).
    trunk = np.where(np.isin(dominant, spine[:-1]) & (v[:, 1] < joints[sem["leftShoulder"]][1]))[0]
    dist = np.min([np.linalg.norm(v[trunk] - (joints[a] + np.clip(((v[trunk] - joints[a]) @ (joints[b] - joints[a]))
                                                 / ((joints[b] - joints[a]) @ (joints[b] - joints[a])), 0, 1)[:, None]
                                                 * (joints[b] - joints[a])), axis=1)
                   for a, b in zip(spine, spine[1:])], axis=0)
    rel = v[trunk] - joints[sem["pelvis"]]
    bins = ((np.arctan2(rel[:, 2], rel[:, 0]) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    trunk_r = min(np.percentile(dist[bins == k], 25) for k in range(8) if (bins == k).sum() >= 5)
    for c in caps:
        c["radius"] = round(float(trunk_r), 4)
    # Head: from the head joint half way to the top of the head.
    head = sem["head"]
    top = v[targets["topHead"]["vertices"][0][0]]
    head_owners = [j for j in descendants(head)]
    add("head", head, head, head_owners, oa=0.35 * (top - joints[head]), ob=0.65 * (top - joints[head]))
    for side in ("left", "right"):
        s = sem[side + "Shoulder"]; e = sem[side + "Elbow"]; w = sem[side + "Wrist"]
        add(side + "UpperArm", s, e, [s])
        add(side + "Forearm", e, w, [e])
        hand_end = sem[side + "Middle"][0] if sem.get(side + "Middle") else sem.get(side + "Hand")
        hand_owners = [j for j in descendants(w)]
        # Hands reach past their end joint: extend to roughly the fingertips.
        tip = v[targets[side + "MiddleTip"]["vertices"][0][0]] if side + "MiddleTip" in targets else None
        ob = None if tip is None else 0.8 * (tip - joints[hand_end])
        if tip is None:
            ob = 0.8 * (joints[hand_end] - joints[w])
        add(side + "Hand", w, hand_end, hand_owners, ob=ob)
        h = sem[side + "Hip"]; k = sem[side + "Knee"]; a = sem[side + "Ankle"]; f = sem[side + "Foot"]
        # Thighs taper: measured over the lower two thirds, so knees can touch without colliding.
        add(side + "Thigh", h, k, [h], t_min=0.35)
        add(side + "Shin", k, a, [k])
        add(side + "Foot", a, f, descendants(a))

    # Pairs to test: everything not already overlapping in the rest pose, except legs against the trunk
    # (hip limits already bound them, and a deep squat presses thigh into belly).
    def cap_dist(p, q):
        # Sampled segment-segment distance (exact enough for this one-off test).
        ts = np.linspace(0, 1, 21)[:, None]
        P = p["_a"] + ts * (p["_b"] - p["_a"])
        Q = q["_a"] + ts * (q["_b"] - q["_a"])
        return np.linalg.norm(P[:, None] - Q[None], axis=2).min()

    pairs = []
    for i in range(len(caps)):
        for j in range(i + 1, len(caps)):
            p, q = caps[i], caps[j]
            leg_trunk = {p["name"][:5], q["name"][:5]} >= {"torso"} and any(
                n.endswith(("Thigh", "Shin", "Foot")) for n in (p["name"], q["name"]))
            if cap_dist(p, q) > p["radius"] + q["radius"] + 0.01 and not leg_trunk:
                pairs.append([i, j])
    for c in caps:
        del c["_a"], c["_b"]
        print(f"    capsule {c['name']:14} r={c['radius'] * 100:4.1f} cm")
    print(f"    {len(pairs)} collision pairs")
    return {"capsules": caps, "pairs": pairs}


def body_targets(sem, joints):
    """Keypoint targets shared by all models' skeletons."""
    t = {
        "root": {"mid": [sem["leftHip"], sem["rightHip"]]},
        "spine": {"joint": spine_target(joints, sem["spine"], sem["pelvis"], sem["neck"][0])},
        "centerShoulder": {"joint": sem["neck"][0]},
        "centerHead": {"joint": sem["head"]},
    }
    for side in ("left", "right"):
        for k in ("Hip", "Knee", "Ankle", "Shoulder", "Elbow", "Wrist"):
            t[side + k] = {"joint": sem[side + k]}
    return t


def finger_targets(sem, v, weights, joints):
    """Vision hand keypoints -> finger joints (MCP/PIP/DIP, thumb CMC/MP/IP) and fingertip vertices."""
    t = {}
    for side in ("left", "right"):
        thumb = sem[side + "Thumb"]
        t[side + "ThumbCMC"] = {"joint": thumb[0]}
        t[side + "ThumbMP"] = {"joint": thumb[1]}
        t[side + "ThumbIP"] = {"joint": thumb[2]}
        t[side + "ThumbTip"] = surface([fingertip_vertex(v, weights, thumb[2], thumb[1], joints)])
        for finger in ("Index", "Middle", "Ring", "Little"):
            c = sem[side + finger]
            t[side + finger + "MCP"] = {"joint": c[0]}
            t[side + finger + "PIP"] = {"joint": c[1]}
            t[side + finger + "DIP"] = {"joint": c[2]}
            t[side + finger + "Tip"] = surface([fingertip_vertex(v, weights, c[2], c[1], joints)])
    return t


# ---------------------------------------------------------------------------------------------------
# SMPL


class _ChumpyStub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return type(name, (_ChumpyStub,), {})
        if module.startswith("scipy.sparse"):
            return getattr(sp, name)
        return super().find_class(module, name)


def _array(v):
    if isinstance(v, _ChumpyStub):
        v = v._state["x"]
    if sp.issparse(v):
        v = v.toarray()
    return np.asarray(v)


SMPL_MEMBERS = {
    "neutral": "SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
    "male": "SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
    "female": "SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
}
SMPL_NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
              "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
              "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
              "left_hand", "right_hand"]
SMPL_SEMANTIC = {
    "pelvis": 0, "spine": [3, 6, 9], "neck": [12], "head": 15,
    "leftHip": 1, "rightHip": 2, "leftKnee": 4, "rightKnee": 5, "leftAnkle": 7, "rightAnkle": 8,
    "leftFoot": 10, "rightFoot": 11, "leftCollar": 13, "rightCollar": 14, "leftShoulder": 16,
    "rightShoulder": 17, "leftElbow": 18, "rightElbow": 19, "leftWrist": 20, "rightWrist": 21,
    "leftHand": 22, "rightHand": 23,
}
# Surface landmarks on the SMPL template (checked against the geometry: +x is the body's left).
SMPL_LANDMARKS = {
    "topHead": 411, "nose": 332, "leftEye": 2800, "rightEye": 6260, "leftEar": 583, "rightEar": 4071,
    "mouthLeft": 101, "mouthRight": 3612, "chin": 3161,
    "leftIndexMCP": 2133, "leftMiddleMCP": 2198, "leftLittleMCP": 2591,
    "rightIndexMCP": 5594, "rightMiddleMCP": 5737, "rightLittleMCP": 6052,
}
FACE_KEYS = ["topHead", "nose", "leftEye", "rightEye", "leftEar", "rightEar", "mouthLeft", "mouthRight", "chin"]


def load_pickle(zip_path: Path, member: str) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        return _Unpickler(io.BytesIO(z.read(member)), encoding="latin1").load()


def convert_smpl(zip_path: Path, out_root: Path):
    reference = None
    for gender, member in SMPL_MEMBERS.items():
        print(f"SMPL {gender}")
        d = load_pickle(zip_path, member)
        v = _array(d["v_template"])
        shapedirs = _array(d["shapedirs"])[:, :, :NUM_BETAS]
        jreg = _array(d["J_regressor"])
        joints = jreg @ v
        weights = _array(d["weights"])
        parents = [int(p) if p < 2 ** 31 else -1 for p in _array(d["kintree_table"])[0].astype(np.int64)]
        parents[0] = -1
        sem = SMPL_SEMANTIC
        targets = body_targets(sem, joints)
        for k, vid in SMPL_LANDMARKS.items():
            targets[k] = surface([vid])
        rig = make_rig(joints, parents, SMPL_NAMES, sem, targets, hand_hinges=False)
        rig["stiffness"][22] = rig["stiffness"][23] = 6.0
        rig["collision"] = body_capsules(v, weights, joints, parents, sem, targets)
        arrays = linear_model_arrays(v, shapedirs, joints, np.einsum("jv,vdb->jdb", jreg, shapedirs), weights,
                                     _array(d["f"]), parents, _array(d["posedirs"]))
        write_model(out_root / f"smpl_{gender}", arrays, {
            "family": "SMPL", "displayName": f"SMPL · {gender.capitalize()}", "order": 10 + list(SMPL_MEMBERS).index(gender),
            "licence": "SMPL Model License (MPI, non-commercial)", "rig": rig})
        if gender == "neutral":
            reference = {"v": v, "joints": joints, "landmarks": {k: v[i] for k, i in SMPL_LANDMARKS.items()}}
    return reference


# ---------------------------------------------------------------------------------------------------
# SMPL-X


SMPLX_SEMANTIC = {
    "pelvis": 0, "spine": [3, 6, 9], "neck": [12], "head": 15, "jaw": 22,
    "leftHip": 1, "rightHip": 2, "leftKnee": 4, "rightKnee": 5, "leftAnkle": 7, "rightAnkle": 8,
    "leftFoot": 10, "rightFoot": 11, "leftCollar": 13, "rightCollar": 14, "leftShoulder": 16,
    "rightShoulder": 17, "leftElbow": 18, "rightElbow": 19, "leftWrist": 20, "rightWrist": 21,
    # Hand joints: index 1-3, middle 1-3, pinky 1-3, ring 1-3, thumb 1-3 (left 25-39, right 40-54).
    "leftIndex": [25, 26, 27], "leftMiddle": [28, 29, 30], "leftLittle": [31, 32, 33], "leftRing": [34, 35, 36],
    "leftThumb": [37, 38, 39],
    "rightIndex": [40, 41, 42], "rightMiddle": [43, 44, 45], "rightLittle": [46, 47, 48], "rightRing": [49, 50, 51],
    "rightThumb": [52, 53, 54],
}
SMPLX_EXTRA = {23: 6.0, 24: 6.0}  # eyes


def find_smplx_zip(root: Path):
    for p in sorted(root.glob("*.zip")):
        try:
            with zipfile.ZipFile(p) as z:
                if any(n.lower().endswith("smplx_neutral.npz") for n in z.namelist()):
                    return p
        except zipfile.BadZipFile:
            pass
    return None


def convert_smplx(zip_path: Path, out_root: Path, reference):
    with zipfile.ZipFile(zip_path) as z:
        for gender in ("neutral", "male", "female"):
            member = next((n for n in z.namelist() if n.lower().endswith(f"smplx_{gender}.npz")), None)
            if member is None:
                continue
            print(f"SMPL-X {gender}")
            d = np.load(io.BytesIO(z.read(member)), allow_pickle=True)
            v = d["v_template"]
            shapedirs = d["shapedirs"][:, :, :NUM_BETAS]            # (first 300 are shape, then expression)
            jreg = d["J_regressor"]
            jreg = jreg.toarray() if sp.issparse(jreg) else np.asarray(jreg)
            joints = jreg @ v
            weights = d["weights"]
            posedirs = d["posedirs"]
            if posedirs.ndim == 2:                                  # stored as (P, V*3) in some releases
                posedirs = posedirs.T.reshape(v.shape[0], 3, -1)
            parents = [int(p) for p in d["kintree_table"][0].astype(np.int64)]
            parents = [p if 0 <= p < len(parents) else -1 for p in parents]
            parents[0] = -1
            sem = SMPLX_SEMANTIC
            targets = body_targets(sem, joints)
            targets.update(finger_targets(sem, v, weights, joints))
            # Face: SMPL-X ships a landmark embedding (51 iBUG points, 17..67, as barycentric coordinates on
            # mesh faces). Use it directly for nose, eyes and mouth corners -- transferring them from SMPL puts
            # them 2-3.5 cm too low, the heads' proportions differ. iBUG 36-41 is the subject's right eye.
            faces_idx, bary = d["lmk_faces_idx"], d["lmk_bary_coords"]
            def ibug(*points):
                blend = {}
                for i in points:
                    for vid, w in zip(d["f"][faces_idx[i - 17]], bary[i - 17]):
                        blend[int(vid)] = blend.get(int(vid), 0.0) + w / len(points)
                return surface(list(blend), list(blend.values()))
            targets["nose"] = ibug(30)
            targets["rightEye"] = ibug(36, 37, 38, 39, 40, 41)
            targets["leftEye"] = ibug(42, 43, 44, 45, 46, 47)
            targets["mouthRight"] = ibug(48)
            targets["mouthLeft"] = ibug(54)
            def pos(t):
                vs = np.array(t["vertices"])
                return (v[vs[:, 0].astype(int)] * vs[:, 1:2]).sum(0) / vs[:, 1].sum()
            # Chin and ears aren't in the embedding: transfer from SMPL via a similarity fit on the face.
            anchors = ["nose", "leftEye", "rightEye", "mouthLeft", "mouthRight"]
            head = weights[:, [15, 22, 23, 24]].sum(1) > 0.5
            ids = transfer_vertices([reference["landmarks"][k] for k in ("chin", "leftEar", "rightEar")],
                                    [reference["landmarks"][k] for k in anchors], [pos(targets[k]) for k in anchors],
                                    v, head)
            for k, vid in zip(("chin", "leftEar", "rightEar"), ids):
                targets[k] = surface([vid])
            targets["topHead"] = surface([top_of_head_vertex(v, weights, [15, 22, 23, 24])])
            # Untested against real files: print landmarks so the first conversion can be sanity-checked
            # (expect +x on the left, nose furthest forward in +z, mouth below the nose, chin below the mouth).
            for k in FACE_KEYS + ["leftIndexMCP", "leftIndexTip", "leftThumbTip"]:
                t = targets[k]
                p = joints[t["joint"]] if "joint" in t else pos(t)
                print(f"    {k:14} {np.round(p, 3)}")
            names = [f"joint{i}" for i in range(len(parents))]
            rig = make_rig(joints, parents, names, sem, targets, extra_stiffness=SMPLX_EXTRA)
            rig["collision"] = body_capsules(v, weights, joints, parents, sem, targets)
            # Default pose: SMPL-X's relaxed mean hand (its template hand is flat), so unobserved hands
            # look natural and the finger prior pulls towards a relaxed hand rather than a flat one.
            mean = np.zeros((len(parents), 3))
            mean[25:40] = d["hands_meanl"].reshape(15, 3)
            mean[40:55] = d["hands_meanr"].reshape(15, 3)
            rig["poseMean"] = mean.round(5).tolist()
            arrays = linear_model_arrays(v, shapedirs, joints, np.einsum("jv,vdb->jdb", jreg, shapedirs),
                                         weights, d["f"], parents, posedirs)
            write_model(out_root / f"smplx_{gender}", arrays, {
                "family": "SMPL-X", "displayName": f"SMPL-X · {gender.capitalize()}",
                "order": 20 + ("neutral", "male", "female").index(gender),
                "licence": "SMPL-X Model License (MPI, non-commercial)", "rig": rig})


# ---------------------------------------------------------------------------------------------------
# Anny


def height_of(v_template, shapedirs, betas):
    v = v_template + shapedirs @ betas
    return float(v[:, 1].max() - v[:, 1].min())


def convert_anny(out_root: Path, reference, samples=2000, components=16, seed=0):
    import torch
    import anny
    import anny.keypoints
    import anny.shape_distribution

    print(f"Anny: sampling {samples} bodies")
    torch.manual_seed(seed)
    model = anny.Anny().to(dtype=torch.float32)
    names = list(model.bone_labels)
    idx = {n: i for i, n in enumerate(names)}
    parents = [int(p) for p in model.bone_parents]

    # Anny is z-up and faces -y; rotate into the canonical frame (y up, facing +z). Proper rotation.
    to_canonical = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

    distribution = anny.shape_distribution.SimpleShapeDistribution(model)
    ages, phenotypes = distribution.sample(samples)
    labels = model.phenotype_labels
    rest_pose = torch.eye(4)[None, None].repeat(samples, model.bone_count, 1, 1)
    with torch.no_grad():
        out = model(pose_parameters=rest_pose, phenotype_kwargs={k: phenotypes[k] for k in labels})
    V = out["rest_vertices"].double().numpy() @ to_canonical.T          # (N, V, 3)
    H = out["rest_bone_heads"].double().numpy() @ to_canonical.T        # (N, J, 3)
    N, nv, nj = V.shape[0], V.shape[1], H.shape[1]

    # PCA over vertices and joints together, so joints stay consistent with the surface.
    X = np.concatenate([V.reshape(N, -1), H.reshape(N, -1)], axis=1)
    mean = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    std = S[:components] / np.sqrt(N - 1)
    basis = (Vt[:components] * std[:, None]).T                          # unit-variance coefficients
    explained = (S[:components] ** 2).sum() / (S ** 2).sum()
    print(f"  {components} components explain {explained * 100:.2f}% of shape variance")
    coeffs = (X - mean) @ Vt[:components].T / std                       # (N, B)

    v_template = mean[: nv * 3].reshape(nv, 3)
    joints_template = mean[nv * 3:].reshape(nj, 3)
    shapedirs = basis[: nv * 3].reshape(nv, 3, components)
    joints_shapedirs = basis[nv * 3:].reshape(nj, 3, components)

    # Dense skinning weights from Anny's sparse (up to 9 bones per vertex) weights.
    weights = np.zeros((nv, nj))
    bw, bi = model.vertex_bone_weights.numpy(), model.vertex_bone_indices.numpy()
    for k in range(bw.shape[1]):
        np.add.at(weights, (np.arange(nv), bi[:, k]), bw[:, k])
    weights /= weights.sum(1, keepdims=True)

    L, R = ".L", ".R"
    sem = {
        "pelvis": idx["root"], "spine": [idx[f"spine0{i}"] for i in (5, 4, 3, 2, 1)],
        "neck": [idx["neck01"], idx["neck02"], idx["neck03"]], "head": idx["head"],
    }
    for side, s in (("left", L), ("right", R)):
        sem.update({
            side + "Hip": idx["upperleg01" + s], side + "Knee": idx["lowerleg01" + s],
            side + "Ankle": idx["foot" + s], side + "Foot": idx["toe1-1" + s],
            side + "Collar": idx["clavicle" + s], side + "ShoulderBlade": idx["shoulder01" + s],
            side + "Shoulder": idx["upperarm01" + s], side + "Elbow": idx["lowerarm01" + s],
            side + "Wrist": idx["wrist" + s],
            side + "Thumb": [idx[f"finger1-{k}" + s] for k in (1, 2, 3)],
            side + "Index": [idx[f"finger2-{k}" + s] for k in (1, 2, 3)],
            side + "Middle": [idx[f"finger3-{k}" + s] for k in (1, 2, 3)],
            side + "Ring": [idx[f"finger4-{k}" + s] for k in (1, 2, 3)],
            side + "Little": [idx[f"finger5-{k}" + s] for k in (1, 2, 3)],
        })
    targets = body_targets(sem, joints_template)
    targets.update(finger_targets(sem, v_template, weights, joints_template))

    # Face: Anny's own COCO keypoint regressor for nose/eyes/ears (vertex blends on its mesh)...
    coco = anny.keypoints.KeypointsRegressor.coco(model)
    W = coco.regression_weights.numpy()
    coco_names = {"nose": "nose", "leftEye": "left_eye", "rightEye": "right_eye",
                  "leftEar": "left_ear", "rightEar": "right_ear"}
    face_pts = {}
    for key, label in coco_names.items():
        row = W[coco.labels.index(label)]
        face_pts[key] = row @ v_template
        # Keep the 12 heaviest vertices (the full blends span up to ~400, which makes the fitter slow).
        top = np.argsort(row)[::-1][:12]
        w = row[top] / row[top].sum()
        targets[key] = surface(top, w)
        print(f"  {key}: {np.count_nonzero(row > 1e-6)} -> 12 vertices, moves {np.linalg.norm(w @ v_template[top] - face_pts[key]) * 1000:.1f} mm")
    # ...and mouth corners + chin transferred from SMPL through a similarity fit on those five points.
    head_bones = [idx["head"], idx["eye.L"], idx["eye.R"]]
    head = weights[:, head_bones].sum(1) > 0.5
    anchors = list(coco_names)
    ids = transfer_vertices([reference["landmarks"][k] for k in ("mouthLeft", "mouthRight", "chin")],
                            [reference["landmarks"][k] for k in anchors], [face_pts[k] for k in anchors],
                            v_template, head)
    for k, vid in zip(("mouthLeft", "mouthRight", "chin"), ids):
        targets[k] = surface([vid])
    targets["topHead"] = surface([top_of_head_vertex(v_template, weights, head_bones)])

    extra = {idx["pelvis" + s]: 6.0 for s in (L, R)}
    rig = make_rig(joints_template, parents, names, sem, targets, extra_stiffness=extra)
    rig["collision"] = body_capsules(v_template, weights, joints_template, parents, sem, targets)

    # Linear read-out of the phenotype from the shape coefficients, for display (apparent age etc.).
    Y = np.stack([ages.double().numpy()] + [phenotypes[k].double().numpy() for k in labels], 1)
    A = np.concatenate([coeffs, np.ones((N, 1))], 1)
    M, *_ = np.linalg.lstsq(A, Y, rcond=None)
    pred = A @ M
    print("  phenotype read-out R²:", {n: round(1 - ((Y[:, i] - pred[:, i]) ** 2).mean() / Y[:, i].var(), 3)
                                       for i, n in enumerate(["ageYears"] + labels)})
    rig["phenotype"] = {"labels": ["ageYears"] + labels, "matrix": M.T.round(6).tolist()}

    # Age-conditioned shape prior: mean and spread of the shape coefficients at each age, by Gaussian
    # kernel regression over the samples (narrow for children, whose shape changes fast; wider for adults).
    years = ages.double().numpy()
    grid = np.arange(0, 91, 1.0)
    means, stds = [], []
    for a in grid:
        bw = 1.0 + 0.08 * a
        w = np.exp(-0.5 * ((years - a) / bw) ** 2)
        w /= w.sum()
        mu = w @ coeffs
        means.append(mu)
        stds.append(np.sqrt(np.maximum(w @ (coeffs - mu) ** 2, 0.05 ** 2)))
    rig["phenotype"]["ageShape"] = {"ages": grid.tolist(), "mean": np.round(means, 5).tolist(),
                                    "std": np.round(stds, 5).tolist()}
    for a in (3, 8, 15, 30, 70):
        i = int(a)
        print(f"  age {a:2d}: prior mean height {height_of(v_template, shapedirs, means[i]):.2f} m")

    arrays = linear_model_arrays(v_template, shapedirs, joints_template, joints_shapedirs, weights,
                                 model.faces.numpy(), parents)
    write_model(out_root / "anny", arrays, {
        "family": "Anny", "displayName": "Anny (MakeHuman, all ages)", "order": 30,
        "licence": "Anny: Apache-2.0 (NAVER); MakeHuman assets CC0", "rig": rig})


def main():
    ap = argparse.ArgumentParser()
    smpl_zip = ARCHIVES / "SMPL_python_v.1.1.0.zip"
    if not smpl_zip.exists():
        smpl_zip = ROOT / smpl_zip.name  # Compatibility with older checkouts.
    ap.add_argument("--smpl-zip", type=Path, default=smpl_zip)
    ap.add_argument("--smplx-zip", type=Path, default=None, help="default: a zip in data/archives (or the project root) containing SMPLX_NEUTRAL.npz")
    ap.add_argument("--anny", action="store_true", help="also convert Anny (needs the `anny` package)")
    ap.add_argument("--out", type=Path, default=ROOT / "Models")
    args = ap.parse_args()

    reference = convert_smpl(args.smpl_zip, args.out)
    smplx = args.smplx_zip or find_smplx_zip(ARCHIVES) or find_smplx_zip(ROOT)
    if smplx:
        convert_smplx(smplx, args.out, reference)
    else:
        print("SMPL-X: no zip found (download from smpl-x.is.tue.mpg.de and put it in data/archives/)")
    if args.anny:
        convert_anny(args.out, reference)


if __name__ == "__main__":
    main()
