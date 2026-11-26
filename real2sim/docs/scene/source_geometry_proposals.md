# Source geometry proposals

If a model family cannot represent the visible structure, record a source-based
`fit.representation_review` before extending its parameter search. A rejected
representation cannot pass normal fit or export. Keep its measurements and
inspect a replacement before changing that decision.

Calibrated source images can supply new geometry without task dimensions or an
old model. `scene.objects.source_stereo.reconstruct_pair` accepts two OpenCV BGR
images, their pinhole intrinsics, camera-to-world rotations and camera centres,
and independently reviewed source masks. It returns world points, RGB point
colours, correspondence diagnostics and rectified BGR views. This is a proposal
tool; it does not approve object geometry or change source thresholds.

The helper rectifies the complete source-mask area, rather than zooming into
the intersection of two full-image fields of view. Empty padding prevents the
stereo search band from silently cropping the object. It checks both disparity
directions and then checks the actual image patches separately. A textureless
surface can produce no useful points. Repeated patterns, transparency,
reflections and changing exposure still require source review; two-way matching
alone does not prove that a point belongs to the desired surface.

Keep each point's originating image and camera hashes. A point seen on a rear
surface in one image can project inside a foreground part's mask in another.
That projection does not make it a foreground observation. When reconstructing
a visible part, first use points measured from that image or independently
establish their visibility. Combine additional views only after checking part
identity and occlusion. Separate source patches from whole physical parts:
an isolated visible patch cannot establish hidden material, a total part count,
thickness or a connection.

Use source-supported curves, surfaces or mesh proposals to build an editable
Blender model. Review internal structure, full material quantity, rigid/flexible
choices and uncertain hidden connections. Then run the original source,
geometry, material, contact, task and delivery checks. A source-textured patch,
stereo point cloud or successful synthetic stereo control is not a substitute
for those checks.
