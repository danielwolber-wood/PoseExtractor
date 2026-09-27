# How the fit works

Code: `Sources/ArmatureCore/Fitting/BodyFitter.swift`.

1. **Initial orientation:** a Kabsch alignment of the rest-pose torso onto Vision's torso.
2. **Stage 1 (3D):** solves for pose, shape and a free scale factor against Vision's root-relative joints. The scale is free because Vision reports every person at a nominal 1.8 m, so its scale carries no real information. Body size therefore comes from the shape prior.
3. **Stage 2 (2D):** a closed-form translation, then pose and translation refined against the 2D keypoints. Vision's 3D joints act as a soft prior for depth.
4. **Extra keypoints:**
   - **Hands:** knuckles from Vision's hand pose request orient each hand, and with it forearm twist. Hands are matched to the body by wrist position rather than Vision's left/right label.
   - **Face:** mouth corners and chin from face landmarks pin down head pitch and roll.
   - **Surface landmarks:** these are matched to SMPL surface vertices, found from the template geometry.
   - **Fingers:** SMPL's are rigid, so fingertips aren't used.
5. **Camera and scene:**
   - **Intrinsics:** each crop is sent to Vision with its camera intrinsics. This doesn't change Vision's pose, but moves its own distance estimate from 2.3 m to 4.3 m on the self-test (truth 4.2 m).
   - **Horizon:** Vision's horizon angle sets the floor's roll when a horizon is found; the bodies' tilt still sets pitch.
6. **Depth:**
   - **When it applies:** photos with an embedded *metric* depth map (LiDAR/TrueDepth). The torso's measured distance pins the body's distance, and body shape is then free to set its real size.
   - **Self-test:** distance error drops from 38 cm to 3 cm and height error from 10 cm to 3 cm, measured on a synthetic HEIC with embedded depth.
   - **Relative depth:** dual-camera Portrait disparity has no scale, so it's ignored.
   - **Monocular depth:** photos without embedded metric depth can use Depth Anything V2 Small or Depth Pro
     (optional Core ML models). Embedded metric depth always takes priority. Monocular depth refines the
     fit made without it: it adds robust residuals for each keypoint's depth relative to the torso and,
     only when distance is observable, a soft distance term. The refined fit is kept only if 2D keypoints,
     silhouette, edited joints and pose plausibility don't get worse. Relative depth is never treated as
     metres. See [Monocular depth](depth.md).
   - **Not passed to Vision:** with depth that lacks camera-calibration data, Vision's own depth-aware 3D request hits an internal assertion that kills the process. That path is deliberately not used.
7. **Keypoint trust:** Vision invents positions for joints it can't see. Limbs its 2D detector doesn't confirm, and anything projecting outside the photo, lose almost all weight, so truncated legs fall back to a natural pose. Face keypoints (nose, eyes, ears) come from the 2D detector and are matched to SMPL surface vertices; when they're found, they override Vision's 3D head.
8. **Silhouette (stage 3):** code in `Sources/ArmatureCore/Fitting/SilhouetteFit.swift`.
   - Vision's person instance segmentation gives each person a mask; the mask their keypoints fall in.
   - Body shape, placement, and trunk and limb pose are then refined ICP-style. Each round picks the body's outline vertices, the vertices whose surface grazes the view ray.
   - The body's outline is pushed inside the mask. A weaker term, capped at 3% of body height, pulls it out to the mask's edge. The asymmetry is because clothing and hair only ever make the mask bigger than the body.
   - The result is kept only if less of the body lies outside the mask without the overlap collapsing.
   - On the self-test (a heavier-than-average body), body-shape error drops from 4.9 cm to 1.8 cm.
   - In the app, this stage (about 100 ms) is skipped while you drag and runs when you let go.
   - For debugging, set `ARMATURE_DEBUG_MASK=/tmp/m.ppm` to dump the mask with the body before (red) and after (green).
9. **Priors:** the pose prior holds joints Vision can't see (hands, feet, collars, neck twist) near rest. It also spreads spine bending across the three spine joints and restricts knees and elbows to hinge motion.
10. **Anatomical limits:** see [Anatomical limits](#anatomical-limits) below.

## Anatomical limits

Fits used to end up in impossible poses: legs twisted backwards, heads turned 180°, forearms through the
chest. Three things now rule those out. `BodyFitter.anatomicalPriors` (on by default; the CLI's
`--no-anatomical-limits` turns it off) switches all three.

- **Range of motion.** Hips, shoulders, collars, spine, neck, head, wrists and ankles get a swing-twist
  limit. The swing limit is an ellipse in each quadrant of (flexion, abduction) around an anatomical neutral
  direction, plus a symmetric twist limit about the bone. Knees, elbows and fingers get a maximum flexion,
  and hyperextension is measured from straight rather than from the rest pose. (Anny's A-pose has elbows
  bent ~40°, which it previously couldn't straighten.) The limits are generous versions of clinical
  range-of-motion tables, in `tools/convert_models.py`. The spine and neck ranges are for the whole chain,
  shared over however many joints the model has. Inside the range there is no cost; beyond it, 20 residual
  units per radian.
- **Self-collision.** About 17 capsules (trunk segments, head, upper arm, forearm, hand, thigh, shin, foot)
  are inscribed in the flesh, so ordinary contact such as hands on hips doesn't count. Overlap beyond 1 cm
  is penalised. Pairs already touching in the rest pose, and legs against the trunk, aren't tested.
- **Limb-depth flips.** A bone pointing towards the camera projects like one pointing away, and Vision's
  3D sometimes picks wrong. For each limb bone that is seen in the photo and clearly out of the image
  plane, stages 1–2 are re-run with its depth flipped. The flip is kept if it lowers the stage-2 cost by
  more than 3%. When comparing, each joint's 3D term uses whichever depth reading the body is nearest, so
  Vision's guess gets no head start. Flips in different limbs are also tried together. Warm-started
  (interactive) re-fits skip this, and so do fits of people the user has edited.
- **An unconstrained start.** Limits and collision penalties can act as walls between Vision's starting
  pose and the right one; for example, a leg may need to pass through the other on its way. So stages 1–2
  also run once with stage 1 free of them (stage 2 still has them), and the better-scoring of the two is
  kept.

Each fit carries a `plausibility` report (also in the JSON export). It lists joints still past their range,
body parts still overlapping, and which limbs were flipped. `BodyModel.plausibility(pose:betas:)` checks
any pose.

`armature-eval <folder>` compares fits with and without these priors on real photos. It uses the same
detections for both, and reports impossible poses, 2D error, silhouette IoU and time.
`armature-selftest --plausibility-only` checks the limits on known good and broken poses.


Vision's 3D joints sit in slightly different anatomical places than SMPL's; hips are the worst, about 9 cm wider apart. Those joints are down-weighted rather than trusted.

