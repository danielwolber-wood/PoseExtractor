#!/usr/bin/env python3
"""One-time conversion of body models into the flat format the Swift app memory-maps.

Every model is written in one canonical frame -- y up, facing +z, +x on the body's left, metres -- as
a linear model (template + shape blend shapes [+ pose correctives]), a skeleton (rest joints that are
linear in the shape coefficients), dense skinning weights, and a *rig description* telling the fitter
which joint is which, where each keypoint lives on the body, how stiff each joint is, and which joints
are hinges.

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
        "silhouetteJoints": sorted(set(int(j) for j in silhouette)),
    }


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
