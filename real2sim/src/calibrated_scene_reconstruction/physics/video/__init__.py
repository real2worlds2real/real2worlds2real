"""Physics from one RGB video: something sets an object in motion, and the motion the object makes on its own afterwards
measures the physical parameters of its twin. See docs/video_physics.md.

Rigid bodies on a support
perceive   GPU: SAM 3 masks of the object, the support and whatever moves the object; MoGe-2 support plane, intrinsics
scene      CPU: the world frame of a monocular video (the support plane), rays, plane fits; reading a recording
stabilise  CPU: a camera that only shakes, warped onto one frame
track      CPU: the object's pose on the support in every frame from its masks
fit        CPU: the free phase and its closed-form fit by motion (meta.json "motion"): slide, roll
ballistic  drop: flights and bounces (restitution, the scene's scale)
incline    a support that is not level, or that tilts (kinetic and static friction)
turntable  a support that turns (static friction's bound, kinetic friction)
pendulum   what swings from a pivot (its length, or equivalent length, and damping)
carried    a support pulled along under the object (a bound on static friction from the track)
stickslip  a long body drawn along by its other end, its foot following in jumps (kinetic over static friction)
mesh       GPU: the object's mesh by SAM 3D, its place in the first frame; the clean plate
stage      Isaac python: the twin's USD stage (apps/video_physics.py replays and fits it in PhysX)
raster     silhouettes of meshes
visualise  summary.mp4: what was measured from what

Bodies that deform
soft       GPU: perception of a body that is on no support (a level surface gives gravity, MoGe-2 the body's depth;
           a strain needs no level surface)
beam       a slender body clamped at one end that sags and swings (modulus over density, viscoelastic relaxation)
           (apps/video_soft.py replays it in PhysX as an articulation of the chain's joints and renders it)
strain     a block pulled, pushed, sheared, twisted or bent by what holds it (the strain, Poisson's ratio; no modulus)
sheet      a cloth that hangs from an edge that is carried and set down (its mass per area, weighed by the air)

meta         meta.json checked before a run (a key or a motion that would be read as if it were not there is said),
             the kinds of motion and what each must show, and after a run what was made of what the agent wrote
readings     what a recording's physics.json gives, in one form whatever the motion (values, bounds, or why not)
"""
