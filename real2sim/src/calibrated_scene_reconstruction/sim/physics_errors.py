"""Fail native checks on solver errors even when Kit returns process status zero."""
import json
import re
import shutil
from pathlib import Path


class PhysicsErrors:
    def __init__(self, output, *, log_path=None):
        self.output=Path(output)
        self.messages=[]
        self.count=0
        self.unknown_face_material_count=0
        self.contact_report_compatibility=None
        self.sdf_limits={}
        self.convex_cooking_rejections=[]
        if log_path is None:
            import carb.settings
            log_path=carb.settings.get_settings().get('/log/file')
        if not log_path:raise ValueError('Native Kit log is required for physics-error checks')
        self.log_path=Path(log_path)
        self.stream=self.log_path.open('r',encoding='utf-8',errors='replace')
        self.pending=''
        self.checked_bytes=0

    def poll(self):
        # Cooking threads can log while the main thread holds the GIL. Read
        # the native file instead of installing a Python logger callback.
        if self.stream is None:return
        content=self.pending+self.stream.read()
        lines=content.split('\n');self.pending=lines.pop()
        self.checked_bytes=self.stream.tell()
        for line in lines:
            if ('[Warning]' in line and '[omni.physx.plugin]' in line
                    and 'self collision filter pose' in line.lower()
                    and 'not supported' in line.lower()):
                self.count+=1
                if len(self.messages)<32:
                    self.messages.append({'source':'omni.physx.plugin','message':line.strip(),
                                          'reason':'Requested deformable collision-filter pose was rejected by the native runtime'})
            if ('[Warning]' in line and '[omni.physx.tensors.plugin]' in line
                    and 'non-root articulation links whose velocities cannot be set directly' in line):
                self.count+=1
                if len(self.messages)<32:
                    self.messages.append({'source':'omni.physx.tensors.plugin','message':line.strip(),
                                          'reason':'Requested native actuation was not applied to all selected bodies'})
            # Unknown mesh-face IDs can also be a report-metadata limitation,
            # not a missing solver material. Keep the raw warning; only an
            # explicit audited runtime/material profile may classify it.
            if ('[Warning]' in line and '[omni.physx.plugin]' in line
                    and 'PxShape::getMaterialFromInternalFaceIndex' in line
                    and 'returning NULL' in line):
                self.count+=1
                if 'received 0xFFFFffff as input - returning NULL.' in line:
                    self.unknown_face_material_count+=1
                if len(self.messages)<32:
                    self.messages.append({'source':'omni.physx.plugin','message':line.strip(),
                                          'reason':'Native contact material lookup failed'})
            convex=re.search(r'\[Warning\]\s+\[omni\.convexdecomposition\.plugin\]\s*(Invalid .*)',line)
            if convex and convex[1] not in self.convex_cooking_rejections:
                self.convex_cooking_rejections.append(convex[1])
            clamp=re.search(r'\[Warning\]\s+\[omni\.physx\.cooking\.plugin\].*Limited the SDF resolution to (\d+) for prim\(([^)]+)\)',line)
            if clamp:
                limit=int(clamp[1]);path=clamp[2]
                self.sdf_limits[path]=min(limit,self.sdf_limits.get(path,limit))
            match=re.search(r'\[(?:Error|Fatal)\]\s+\[([^\]]+)\]\s*(.*)',line)
            if match and any(name in match[1].lower() for name in ('physx','physics','cuda')):
                self.count+=1
                if len(self.messages)<32:self.messages.append({'source':match[1],'message':match[2]})

    def report(self):
        self.poll()
        effective=self.count-(self.unknown_face_material_count if self.contact_report_compatibility else 0)
        return {'pass':effective==0,'error_count':effective,'raw_error_warning_count':self.count,'messages':list(self.messages),
                'unknown_face_material_lookup_count':self.unknown_face_material_count,
                'contact_report_metadata_compatibility':self.contact_report_compatibility,
                'native_log':str(self.log_path),'checked_log_bytes':self.checked_bytes,
                'sdf_cooking_resolution_limits':dict(self.sdf_limits),
                'convex_cooking_rejections':list(self.convex_cooking_rejections),
                'scope':'Native solver errors and rejected actuation fail; an explicitly audited unknown-face reporting limitation retains its raw warnings separately'}

    def audit_contact_report_metadata(self,stage):
        from .contact_report_compat import audit
        self.contact_report_compatibility=audit(stage)
        return self.contact_report_compatibility

    def sdf_resolution(self,path,requested):
        """Use reported cooking limits in precision budgets, never as a proof.

        USD attributes retain the requested resolution after cooking clamps it.
        The resulting bound does not measure the SDF's interpolation error or
        certify contacts. A native functional check remains necessary.
        """
        if type(requested) is not int or requested<=0:
            raise ValueError('Positive integer requested SDF resolution required')
        self.poll();limit=self.sdf_limits.get(str(path))
        return {'prim':str(path),'requested_resolution':requested,
                'cooking_limit':limit,'effective_resolution_bound':min(requested,limit) if limit is not None else requested,
                'scope':'Requested resolution bounded by logged native cooking clamp; not an SDF accuracy or contact certificate'}

    def precision_audit(self):
        """A silent native fallback cannot approve the requested profile.

        Author a supported resolution and repeat the checks, or change the
        collider representation when the required detail exceeds that limit.
        """
        self.poll()
        return {'pass':not bool(self.sdf_limits or self.convex_cooking_rejections),'sdf_cooking_resolution_limits':dict(self.sdf_limits),
                'convex_cooking_rejections':list(self.convex_cooking_rejections),
                'failures':(['Native cooking reduced requested SDF precision; author a supported profile and repeat validation']
                            if self.sdf_limits else [])+list(self.convex_cooking_rejections),
                'scope':'Native cooking/profile agreement only; source accuracy and functional contacts remain separate required checks'}

    def check(self):
        self.poll()
        if self.count-(self.unknown_face_material_count if self.contact_report_compatibility else 0):
            self.output.parent.mkdir(parents=True,exist_ok=True)
            self.output.write_text(json.dumps(self.report(),indent=2)+'\n')
            raise RuntimeError('Native physics reported errors; preserve '+str(self.output))

    def guard_steps(self,world):
        """Guard initialization as well as explicit timesteps.

        A failed initial state can stop a probe before its first World.step.
        Cooking/reset errors must still reject that experiment immediately.
        """
        original=world.step
        def checked(*args,**kwargs):
            self.check()
            value=original(*args,**kwargs)
            self.check()
            return value
        world.step=checked
        reset=getattr(world,'reset',None)
        if reset is not None:
            def checked_reset(*args,**kwargs):
                self.check()
                value=reset(*args,**kwargs)
                self.check()
                return value
            world.reset=checked_reset

    def close(self):
        if self.stream is not None:
            self.poll();self.stream.close();self.stream=None
            self.output.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(self.log_path,self.output.with_suffix('.kit.log'))
