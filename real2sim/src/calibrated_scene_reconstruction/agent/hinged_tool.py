"""Native attached-lid cloth-extraction tool adapter."""


def execute(s, obj, request, out):
    from calibrated_scene_reconstruction.agent.hand_cloth import run
    result=run(s,obj,request['candidate'],request['hand_plan'],out/'hand_cloth',
               criterion=request['criterion'],case=request.get('case','nominal'),
               fixture_proof=request.get('fixture_proof'),register=request.get('register',False),
               inside_finger=request.get('inside_finger','either'),
               minimum_normal_alignment=request.get('minimum_normal_alignment'),
               pinch_axis=request.get('pinch_axis'),
               pinch_material_axis=request.get('pinch_material_axis'),
               pinch_material_frame=request.get('pinch_material_frame'),
               minimum_pinch_axis_alignment=request.get('minimum_pinch_axis_alignment',.94),
               closure_descent_m=request.get('closure_descent_m'),
               closure_mode=request.get('closure_mode','open_descent'),
               open_descent_seconds=request.get('open_descent_seconds',.5),
               closure_contact_links=request.get('closure_contact_links'),
               closure_contact_basis=request.get('closure_contact_basis'),
               timing=request.get('timing'),material_contact_mode=request.get('material_contact_mode','same_component'),
               require_contact_free_approach=request.get('require_contact_free_approach',False),
               pinch_settle_seconds=request.get('pinch_settle_seconds',.4),
               track_pinch_material=request.get('track_pinch_material',False),
               particle_friction_proof=request.get('particle_friction_proof'))
    return result
