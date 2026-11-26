"""Agent-neutral capability catalog shared by external and managed sessions."""
from __future__ import annotations

from typing import Any

from .deformable_tools import schemas as deformable_schemas
from .observation_tools import OPERATIONS as OBSERVATION_TOOLS
from .observation_tools import schemas as observation_schemas
from .rigid_contact_tools import schemas as rigid_contact_schemas
from .scene_agent import SCENE_TOOLS

TASK_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "declare_asset_contract",
        "description": "For an automatic-input task, declare the asset type and essential semantic parts inferred from the reference evidence before building. Record observations and uncertainty. The declaration then stays fixed for this run; user-required parts are preserved.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["rigid", "articulated", "deformable", "hybrid", "fluid", "strand", "fracture", "solid"]},
                "required_parts": {
                    "type": "array", "minItems": 1, "maxItems": 80,
                    "items": {"type": "string"},
                },
                "observations": {"type": "string"},
                "uncertainty": {"type": "string"},
            },
            "required": ["kind", "required_parts", "observations", "uncertainty"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_blender",
        "description": "Run a Blender Python file from your workspace. Returns construction checks and semantic renders, not reference-appearance acceptance. Compare key proportions and structural features early; use corresponding reference views and appropriate camera/shape diagnostics for mismatches before investing in cosmetic detail. The host preserves editable code and artifacts.",
        "parameters": {
            "type": "object",
            "properties": {
                "script_path": {
                    "type": "string",
                    "description": "Workspace-relative path to a Python file defining build_scene().",
                },
                "plan": {
                    "type": "string",
                    "description": "Geometry/material strategy, supported structural priors, and the reference feature or demonstrated discrepancy this attempt addresses.",
                },
                "change_summary": {
                    "type": "string",
                    "description": "What changed and which observed discrepancy it addresses; use 'initial' first. Tool execution alone does not establish that the discrepancy is repaired.",
                },
            },
            "required": ["script_path", "plan", "change_summary"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "inspect_surface",
        "description": "Inspect a world-space surface region with three close-ups: source material/lighting, neutral clay without texture relief, and grazing diagnostic light. Helps distinguish geometric wrinkles from normal-map marks. Optional frame uses a passed native simulation cache; null inspects the initial saved pose. Read refine-deformable-surface for proxy and detail guidance.",
        "parameters": {
            "type": "object",
            "properties": {
                "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
                "center": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "size_m": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0}, "minItems": 3, "maxItems": 3},
                "view_direction": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "Direction from region center toward camera; need not be normalized."},
                "frame": {"type": ["integer", "null"]},
            },
            "required": ["attempt_id", "center", "size_m", "view_direction", "frame"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_deformation",
        "description": "Bake native Blender CLOTH/SOFT_BODY modifiers on a saved attempt. Returns actual shape-change diagnostics, seven rendered sample frames and a fresh-process cache replay check. Read review-deformation for local conventions. This validates numerical execution, not measured real-world material accuracy.",
        "parameters": {
            "type": "object",
            "properties": {
                "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
                "frames": {"type": "integer", "enum": [48, 72, 96, 120, 180]},
                "assumptions": {"type": "string", "description": "Chosen physical model, material/scale assumptions, constraints or excitation, and what this test can establish."},
            },
            "required": ["attempt_id", "frames", "assumptions"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_simulation",
        "description": "Export a saved rigid, revolute articulated, or cloth Blender attempt into PhysX/Newton and verify real target-backend motion and contacts. Optional during modeling; REQUIRED once with render_views=True before final acceptance. Inspect returned diagnostics and target-rendered images, correct defects and repeat until successful. Read review-simulation for parameter overrides. Particle liquids use run_fluid instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
                "steps": {"type": "integer", "enum": [60, 120, 240, 480]},
                "render_views": {"type": "boolean"},
                "parameters_json": {"type": ["string", "null"], "description": "Optional JSON object of overrides for physics_parameters.json; null uses source/default values."},
                "assumptions": {"type": "string"},
            },
            "required": ["attempt_id", "steps", "render_views", "parameters_json", "assumptions"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_fluid",
        "description": "Run Newton MPM on closed Blender material volumes and reconstruct their flowing surfaces. Returns actual motion/footprint statistics and seven Cycles samples using the original Blender materials. Read simulate-particle-materials for the generic volume metadata. Material defaults are assumed, not calibrated.",
        "parameters": {
            "type": "object",
            "properties": {
                "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
                "frames": {"type": "integer", "enum": [48, 72, 96, 120, 144, 192]},
                "assumptions": {"type": "string"},
            },
            "required": ["attempt_id", "frames", "assumptions"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_hybrid",
        "description": "Run a hybrid asset in Newton 1.7 with articulated rigid links, cloth and native two-way attachments. Returns motion samples, joint/attachment/stretch checks and Isaac RTX views. Required before hybrid delivery. Read simulate-articulated-cloth for generic metadata; correct failures and resubmit.",
        "parameters": {
            "type": "object",
            "properties": {
                "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
                "frames": {"type": "integer", "enum": [96, 120, 180, 240]},
                "parameters_json": {"type": ["string", "null"]},
                "assumptions": {"type": "string"},
            },
            "required": ["attempt_id", "frames", "parameters_json", "assumptions"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "render_final",
        "description": "Render one successful attempt with GPU Cycles, adaptive sampling and denoising. Returns presentation views, adding side views for elongated assets and a detail view when the source marks relevant geometry with real2sim_detail=True. Use after structural iteration and before acceptance.",
        "parameters": {
            "type": "object",
            "properties": {
                "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
                "resolution": {"type": "integer", "enum": [768, 1024, 1536]},
                "samples": {"type": "integer", "enum": [64, 128, 256]},
            },
            "required": ["attempt_id", "resolution", "samples"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "accept_result",
        "description": "Deliver one reviewed asset only after successful Blender rendering AND final target-simulator acceptance. For ordinary rigid/articulated/cloth assets first call run_simulation with render_views=True and inspect target images; a rigid contact-dependent assembly may explicitly select source-bound native SDF review with run_rigid_contact_behavior(record_asset_evidence=True). Fluid assets require run_fluid and surface replay. Failed or missing simulator evidence rejects this submission and leaves the session open for correction.",
        "parameters": {
            "type": "object",
            "properties": {
                "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
                "reason": {
                    "type": "string",
                    "description": "Evidence supporting visual and functional acceptance, including corresponding reference comparisons, key shape and finish features, structural priors and remaining uncertainty. Consequential unresolved visible defects require repair; numeric or physics success alone is insufficient.",
                },
                "cousin_request": {"type": ["string", "null"], "description": "For a twin, preserve the user's natural-language request for a post-twin cousin stage if one was requested; otherwise null. Interpret scope yourself. For a cousin child, always null."},
            },
            "required": ["attempt_id", "reason", "cousin_request"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_skills",
        "description": "Search concise project skills when specialized review guidance may help.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "read_skill",
        "description": "Read one exact project skill selected from the catalog or search results. Use resource=null for its main instructions and available Markdown resources; then select an exact listed resource only when needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "resource": {"type": ["string", "null"]},
            },
            "required": ["name", "resource"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "inspect_silhouette",
        "description": "Load detailed per-view silhouette metrics and mask images for one attempt on demand.",
        "parameters": {
            "type": "object",
            "properties": {"attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"}},
            "required": ["attempt_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "inspect_articulation",
        "description": "Load the full joint audit and limit-pose renders for one Blender attempt on demand.",
        "parameters": {
            "type": "object",
            "properties": {"attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"}},
            "required": ["attempt_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "inspect_references",
        "description": "Inspect reference image metadata, focal-length priors, duplicate frames and available geometry backends on demand.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "solve_reference_camera",
        "description": "Solve and persist one reference camera from metric model points paired with image pixels using OpenCV PnP.",
        "parameters": {
            "type": "object",
            "properties": {
                "reference_index": {"type": "integer", "minimum": 0},
                "object_points": {
                    "type": "array",
                    "minItems": 4,
                    "items": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {"type": "number"},
                    },
                },
                "image_points": {
                    "type": "array",
                    "minItems": 4,
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "number"},
                    },
                },
                "fx": {"type": "number", "exclusiveMinimum": 0},
                "fy": {"type": "number", "exclusiveMinimum": 0},
                "cx": {"type": "number"},
                "cy": {"type": "number"},
                "distortion": {"type": "array", "items": {"type": "number"}},
            },
            "required": [
                "reference_index",
                "object_points",
                "image_points",
                "fx",
                "fy",
                "cx",
                "cy",
                "distortion",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "inspect_camera_alignment",
        "description": "Compare a reference image with an attempt rendered from its solved camera, including the aligned silhouette.",
        "parameters": {
            "type": "object",
            "properties": {
                "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
                "reference_index": {"type": "integer", "minimum": 0},
            },
            "required": ["attempt_id", "reference_index"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_reference_geometry",
        "description": "Run a cached COLMAP multi-view reconstruction with classic SIFT or learned ALIKED-LightGlue features.",
        "parameters": {
            "type": "object",
            "properties": {
                "features": {"type": "string", "enum": ["sift", "aliked"]}
            },
            "required": ["features"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_reference_prior",
        "description": "Run the cached official VGGT-Omega camera/depth prior on all references and return visual input-depth-confidence evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "resolution": {"type": "integer", "enum": [256, 416, 512]}
            },
            "required": ["resolution"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "align_reference_prior",
        "description": "Turn VGGT-Omega intrinsics into one metric Blender reference camera using explicit model-point/image-pixel PnP correspondences, then stage it for the next Blender attempt.",
        "parameters": {
            "type": "object",
            "properties": {
                "resolution": {"type": "integer", "enum": [256, 416, 512]},
                "reference_index": {"type": "integer", "minimum": 0},
                "object_points": {
                    "type": "array",
                    "minItems": 4,
                    "items": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {"type": "number"},
                    },
                },
                "image_points": {
                    "type": "array",
                    "minItems": 4,
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "number"},
                    },
                },
                "max_reprojection_error_px": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": [
                "resolution",
                "reference_index",
                "object_points",
                "image_points",
                "max_reprojection_error_px",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


TASK_TOOLS.append({
    "type": "function", "name": "run_fracture",
    "description": "Cut a closed material volume with official Cell Fracture, then test actual contact-triggered native PhysX fragmentation, velocity handoff, collisions and settling. Return source-bound native feedback and material replay. Required for fracture delivery; read simulate-impact-fracture before authoring. This is prefracture with an editable impulse threshold, not calibrated crack propagation.",
    "parameters": {"type": "object", "properties": {
        "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
        "frames": {"type": "integer", "enum": [144, 192, 240]},
        "assumptions": {"type": "string"}
    }, "required": ["attempt_id", "frames", "assumptions"], "additionalProperties": False},
    "strict": True,
})


TASK_TOOLS.append({
    "type": "function", "name": "run_strands",
    "description": "Run native Newton rods/cables with stretch, shear, bend, twist, frictional contact and optional rigid endpoint bodies. Return actual motion diagnostics and source-material fresh-process replay. Required before strand delivery. Read simulate-slender-bodies for generic authoring metadata; choose material assumptions and an operation appropriate to the input.",
    "parameters": {"type": "object", "properties": {
        "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
        "frames": {"type": "integer", "enum": [96, 120, 144, 192, 240]},
        "assumptions": {"type": "string"}
    }, "required": ["attempt_id", "frames", "assumptions"], "additionalProperties": False},
    "strict": True,
})


TASK_TOOLS.extend([
    {
        "type": "function", "name": "request_cousins",
        "description": "If the user's natural-language twin request also asks for cousins, retain that request verbatim for an optional stage after twin acceptance. Do not request extra variants without user intent. Counts and scope are interpreted by Astra in the planning stage.",
        "parameters": {"type": "object", "properties": {"request": {"type": "string"}}, "required": ["request"], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "submit_cousin_plan",
        "description": "Planning-only: interpret the user's free-form request into an arbitrary positive count and one prompt per cousin. Preserve function and articulated topology. Default to meaningful geometry variation; geometry_change_expected may be false when the user specifically requests surface-only variation. If a required decision is missing, use count=null, variants=[] and a concise question instead of inventing a count.",
        "parameters": {"type": "object", "properties": {
            "count": {"type": ["integer", "null"], "minimum": 1},
            "variants": {"type": "array", "items": {"type": "object", "properties": {
                "prompt": {"type": "string"}, "geometry_change_expected": {"type": "boolean"},
                "layout_change_request": {"type": ["string", "null"], "description": "Scene cousins only: null by default to preserve recognizable parent layout. If the user requests layout changes in natural language, summarize their requested changes and scope here. Do not invent rearrangement for diversity. Object cousins use null."}
            }, "required": ["prompt", "geometry_change_expected", "layout_change_request"], "additionalProperties": False}},
            "shared_notes": {"type": "string"}, "question": {"type": ["string", "null"]},
            "execute": {"type": "boolean", "description": "Infer execution intent from the user's prose: false for planning-only requests, true when the user requests generation."},
        }, "required": ["count", "variants", "shared_notes", "question", "execute"], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "list_video_sources",
        "description": "List retained original reference videos and rendered candidate movies, with stable video IDs. Use inspect_video to view their time-indexed content, intervals and details.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "inspect_video",
        "description": "Read a registered video: overview samples, a dense [start,end) time interval, or exact original frame indices. Returns ordered time-stamped images and full-resolution frame/crop evidence. Interval sample_fps=null reads every original frame; requests above max_frames fail instead of silently dropping frames. Read inspect-video skill.",
        "parameters": {"type": "object", "properties": {
            "video_id": {"type": "string"},
            "mode": {"type": "string", "enum": ["overview", "interval", "frames"]},
            "start_seconds": {"type": ["number", "null"]},
            "end_seconds": {"type": ["number", "null"]},
            "sample_fps": {"type": ["number", "null"]},
            "frame_indices": {"type": ["array", "null"], "items": {"type": "integer"}},
            "crop": {"type": ["array", "null"], "items": {"type": "number"}, "description": "Optional normalized [left,top,right,bottom] crop."},
            "max_frames": {"type": "integer", "minimum": 1, "maximum": 120},
            "detail": {"type": ["string", "null"], "enum": ["overview", "full", None], "description": "null automatically returns full images for dense/frame requests up to 24 frames; full explicitly requests original decoded resolution, overview returns contact sheets. Split full-detail requests above 24 frames."},
        }, "required": ["video_id", "mode", "start_seconds", "end_seconds", "sample_fps", "frame_indices", "crop", "max_frames", "detail"], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "render_motion_video",
        "description": "Optionally render a simulation-verified candidate's real motion to a complete MP4/GIF. Native Newton strands, cloth, MPM fluids and hybrid assets replay retained caches, including material transitions. For rigid/articulated PhysX assets, first write workspace motion_plan.json with attempt_id and a physical control plan; see workspace physics_motion_review.md. Returns video IDs for inspect_video without accepting delivery. Use for final dynamic review when helpful, not every draft.",
        "parameters": {"type": "object", "properties": {
            "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
            "assumptions": {"type": "string"},
        }, "required": ["attempt_id", "assumptions"], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "promote_video_frames",
        "description": "Retain selected uncropped video frames as numbered reference images for existing multiview geometry and camera tools. Choose consistent object states when using multiview reconstruction; motion frames are not automatically one rigid scene.",
        "parameters": {"type": "object", "properties": {
            "video_id": {"type": "string"},
            "frame_indices": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 24},
            "reason": {"type": "string"},
        }, "required": ["video_id", "frame_indices", "reason"], "additionalProperties": False},
        "strict": True,
    },
])


def _measurement_tool(name, description, properties, required):
    return {'type': 'function', 'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': required, 'additionalProperties': False},
            'strict': set(required) == set(properties)}


TASK_TOOLS.extend([
    _measurement_tool('declare_measurements',
        'Before modeling, retain dimensions explicitly stated in the original user request. '
        'dimensions_json is an array of {part:null or exact name,axis:X/Y/Z,value_m,tolerance_m,source:exact user quote}. '
        'The quoted request must state the value and units. Estimated dimensions are not measured constraints. '
        'Existing retained measurement inputs cannot be replaced.',
        {'dimensions_json': {'type': 'string'}}, ['dimensions_json']),
    _measurement_tool('calibrate_measurement_camera',
        'Calibrate intrinsics and lens distortion from retained ChArUco views of the same camera and optics. '
        'Reports reprojection error, view diversity and uncertainty. Optional constrained model, subset sensitivity and small-marker upsampling complement the full model. Calibration is not object-accuracy certification.',
        {'model': {'type': 'string', 'enum': ['full', 'constrained']},
         'subset_trials': {'type': 'integer', 'minimum': 0, 'maximum': 32},
         'detector_upsample': {'type': 'integer', 'enum': [1, 2]}}, []),
    _measurement_tool('measure_reference_board',
        'Detect real ChArUco corners using retained known board dimensions and calibrated intrinsics. '
        'An explicit independently justified asset_to_board rigid 4x4 transform optionally registers a Blender camera. '
        'null returns board-frame pose only; it does not imply the object lies on the board.',
        {'reference_index': {'type': 'integer', 'minimum': 0},
         'asset_to_board': {'type': ['array', 'null'], 'items': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4}, 'minItems': 4, 'maxItems': 4}},
        ['reference_index', 'asset_to_board']),
    _measurement_tool('review_measurements',
        'Measure actual evaluated Blender geometry against retained metric dimensions or observed surface points. '
        'Returns dimension residuals and point-to-triangle errors; does not change or fit the geometry. '
        'Required for acceptance only when metric dimension/surface constraints were supplied.',
        {'attempt_id': {'type': 'string', 'pattern': '^attempt-[0-9]{3}$'}}, ['attempt_id']),
])


TASK_TOOLS.extend(SCENE_TOOLS)


TASK_TOOLS.extend(observation_schemas())


TASK_TOOLS.extend(deformable_schemas())


TASK_TOOLS.extend(rigid_contact_schemas())


TASK_TOOLS.append({
    "type": "function", "name": "run_solids",
    "description": "Convert a declared closed solid Blender control into tetrahedra, bind the source visuals, run native Newton FEM and return fresh-process source-material replay. Required before solid asset delivery. Read simulate-volumetric-solids; preserve topology and source hashes.",
    "parameters": {"type": "object", "properties": {
        "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
        "frames": {"type": "integer", "enum": [120, 144, 192, 240]},
        "assumptions": {"type": "string"}},
        "required": ["attempt_id", "frames", "assumptions"], "additionalProperties": False}, "strict": True,
})


def tool_enabled(mode, name):
    scene_names = {tool["name"] for tool in SCENE_TOOLS}
    if name == "build_scene_cousin_object":
        return mode == "scene_cousin"
    if name == "request_cousins":
        return mode in {"twin", "scene"}
    if mode in {"cousin_planning", "scene_cousin_planning"}:
        return name in {"submit_cousin_plan", "inspect_references", "list_video_sources", "inspect_video", "read_skill", "search_skills"}
    if name in OBSERVATION_TOOLS:
        return True
    if name in {"tetrahedralize_control_mesh", "run_deformable_behavior", "run_rigid_contact_behavior"}:
        return mode in {"twin", "cousin", "scene", "scene_cousin"}
    if mode in {"scene", "scene_cousin"}:
        if mode == "scene_cousin" and name == "build_scene_object":
            return False
        return name in scene_names | {"search_skills", "read_skill", "inspect_references", "list_video_sources", "inspect_video", "promote_video_frames", "run_reference_geometry", "run_reference_prior", "align_reference_prior", "solve_reference_camera", "calibrate_measurement_camera", "measure_reference_board"}
    return name not in scene_names
