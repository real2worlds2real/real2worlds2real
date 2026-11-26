"""Read the source hand's actual convex colliders for penetration diagnostics."""
import numpy as np


def surface_contact(vertices,direction,origin=None):
    """Ray onto a source convex collider, returning an actual surface frame."""
    from scipy.spatial import ConvexHull
    vertices=np.asarray(vertices,float);direction=np.asarray(direction,float)
    origin=vertices.mean(0) if origin is None else np.asarray(origin,float)
    if (direction.shape!=(3,) or origin.shape!=(3,) or not np.isfinite([origin,direction]).all()
            or np.linalg.norm(direction)<1e-12):raise ValueError('Finite nonzero contact ray required')
    planes=ConvexHull(vertices).equations
    if (planes[:,:3]@origin+planes[:,3]).max()>1e-8:raise ValueError('Contact ray must start inside the source collider')
    direction=direction/np.linalg.norm(direction);denominator=planes[:,:3]@direction
    selected=planes[denominator>1e-10];denominator=denominator[denominator>1e-10]
    travel=-(selected[:,:3]@origin+selected[:,3])/denominator;index=int(np.argmin(travel))
    point=origin+travel[index]*direction
    return {'point':point,'normal':selected[index,:3].copy(),
            'source':'Ray intersection with actual source convex collision surface'}


def colliders(stage, root):
    from pxr import Usd,UsdGeom,UsdPhysics
    import trimesh
    cache=UsdGeom.XformCache();rows=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        if not prim.IsA(UsdGeom.Mesh):raise ValueError('Hand collider diagnostics require actual source mesh geometry')
        if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='convexHull':
            raise ValueError('Audit the cooked hand collider representation before interpreting penetration')
        parent=prim.GetParent()
        while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):parent=parent.GetParent()
        if not parent:raise ValueError('Hand collider has no rigid owner')
        matrix=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(parent)).T)@np.asarray(cache.GetLocalToWorldTransform(prim)).T
        vertices=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
        vertices=vertices@matrix[:3,:3].T+matrix[:3,3]
        hull=trimesh.convex.convex_hull(vertices)
        rows.append({'link':parent.GetName(),'path':str(prim.GetPath()),'V':np.asarray(hull.vertices),'F':np.asarray(hull.faces)})
    if not rows:raise ValueError('No hand colliders available')
    return rows


class Penetration:
    def __init__(self, rows):
        from scipy.spatial import ConvexHull
        self.rows=[]
        for row in rows:
            vertices=np.asarray(row['V'],float);planes=ConvexHull(vertices).equations
            # Imported STL hulls can contain near-duplicate triangulation
            # vertices that a visual-mesh cleanup turns into tiny holes. Native
            # convex collision depends on the hull, not that triangulation.
            _,indices=np.unique(np.round(planes,10),axis=0,return_index=True)
            self.rows.append((row['link'],vertices.min(0),vertices.max(0),planes[indices]))

    def check(self, points, transforms, names, tolerance,*,contact_range_m=None):
        from scipy.spatial.transform import Rotation
        contact_range_m=tolerance if contact_range_m is None else contact_range_m
        if not np.isfinite([tolerance,contact_range_m]).all() or min(tolerance,contact_range_m)<0:
            raise ValueError('Finite nonnegative penetration and contact-query distances required')
        query_margin=max(tolerance,contact_range_m)
        points=np.asarray(points)
        maximum=0.;by_link={};contacts={}
        cloud_bounds=(points.min(0),points.max(0)) if len(points) else None
        corners=np.array([[x,y,z] for x in [0,1] for y in [0,1] for z in [0,1]])
        for name,lo,hi,planes in self.rows:
            tf=transforms[names.index(name)];r=Rotation.from_quat(tf[3:]).as_matrix()
            # Conservatively enclose the original tolerance-expanded local
            # box in world coordinates. Far hulls never need a whole-sheet
            # coordinate transform; the exact local box/planes stay unchanged.
            expanded_lo,expanded_hi=lo-query_margin,hi+query_margin
            world=(expanded_lo+corners*(expanded_hi-expanded_lo))@r.T+tf[:3]
            # The original local subtraction can occur in float32. Reserve
            # that rounding here too; this only expands the broadphase, never
            # the final local bounds or the penetration tolerance.
            eps=max(np.finfo(float).eps,*[np.finfo(a.dtype).eps for a in [points,tf]
                if np.issubdtype(a.dtype,np.floating)])
            padding=64*eps*max(1.,float(abs(world).max()))
            world_lo,world_hi=world.min(0)-padding,world.max(0)+padding
            if (cloud_bounds is None or (cloud_bounds[1]<world_lo).any()
                    or (cloud_bounds[0]>world_hi).any()):
                by_link[name]=max(by_link.get(name,0.),0.);contacts.setdefault(name,0)
                continue
            near=((points>=world_lo)&(points<=world_hi)).all(1)
            local=(points[near]-tf[:3])@r
            selected=((local>=lo-query_margin)&(local<=hi+query_margin)).all(1)
            distances=np.max(local[selected]@planes[:,:3].T+planes[:,3],axis=1) if selected.any() else np.empty(0)
            depth=float(max(0.,-distances.min())) if len(distances) else 0.
            contacts[name]=contacts.get(name,0)+int((abs(distances)<=contact_range_m).sum())
            by_link[name]=max(by_link.get(name,0.),depth);maximum=max(maximum,depth)
        return {'pass':maximum<=tolerance,'maximum_penetration_m':maximum,'by_link_m':by_link,'near_surface_samples':contacts,
                'tolerance_m':tolerance,'near_surface_range_m':contact_range_m,
                'representation':'Source convex hull half-spaces in actual native link poses; exact interior depth, conservative exterior near-contact screen'}


class PlannedClearance:
    """Geometric proposal screen for a finite pinch; never a native contact proof."""
    def __init__(self,rows,model,opened,closed,*,samples=5):
        from calibrated_scene_reconstruction.assets.surface_distance import triangle_samples
        if type(samples) is not int or not 2<=samples<=16:raise ValueError('Bounded pinch-path sampling required')
        self.points=[]
        names={j.name for j in model.movable}
        for alpha in np.linspace(0,1,samples):
            state={name:(1-alpha)*opened.get(name,0.)+alpha*closed.get(name,0.) for name in names}
            transforms=model.transforms(state);points=[]
            for row in rows:
                tf=transforms[row['link']]
                points.append(triangle_samples(row['V'],row['F'])@tf[:3,:3].T+tf[:3,3])
            self.points.append(np.concatenate(points))

    def check(self,position,rotation,surfaces,poses,tolerance):
        return self.check_poses(np.tile(position,(len(self.points),1)),
                                np.tile(rotation,(len(self.points),1,1)),surfaces,poses,tolerance)

    def check_poses(self,positions,rotations,surfaces,poses,tolerance):
        """Screen coupled wrist motion and joint closure at matching samples."""
        return self._check(self.points,positions,rotations,surfaces,poses,tolerance)

    def check_open_path(self,positions,rotations,surfaces,poses,tolerance):
        """Screen the complete open hand along the descent, including turns."""
        if not 2<=len(positions)<=128:
            raise ValueError('Bounded open-hand approach sampling required')
        return self._check([self.points[0]]*len(positions),positions,rotations,surfaces,poses,tolerance)

    def check_closed_pose(self,position,rotation,surfaces,poses,tolerance):
        """Reject an impossible endpoint before testing many approach paths."""
        return self._check([self.points[-1]],np.asarray(position)[None],
                           np.asarray(rotation)[None],surfaces,poses,tolerance)

    @staticmethod
    def _check(point_sets,positions,rotations,surfaces,poses,tolerance):
        positions=np.asarray(positions,float);rotations=np.asarray(rotations,float)
        if (positions.shape!=(len(point_sets),3) or rotations.shape!=(len(point_sets),3,3)
                or not np.isfinite(positions).all() or not np.isfinite(rotations).all()
                or not np.allclose(rotations.transpose(0,2,1)@rotations,np.eye(3),atol=1e-7)
                or not np.allclose(np.linalg.det(rotations),1.,atol=1e-7)):
            raise ValueError('Matching finite proper wrist poses required for the closing samples')
        maximum={name:0. for name in surfaces};worst=None
        for index,points in enumerate(point_sets):
            world=points@rotations[index].T+positions[index]
            for name,surface in surfaces.items():
                p,r=poses[name];depth=surface.max_penetration((world-p)@r)
                if depth>maximum[name]:maximum[name]=depth;worst=index
        return {'pass':max(maximum.values(),default=0.)<=tolerance,
                'maximum_penetration_m':maximum,'worst_path_sample':worst,
                'tolerance_m':tolerance,'proposal_only':True,
                'scope':'Source hand convex surfaces against actual rigid triangles in sampled planned poses. Actual finite-force native validation remains required.'}


class OpenPinchReadiness:
    """Does current material lie in both real open fingertip cross-sections?

    Intersect one line through the material with actual source convex hulls.
    Opposing intersections show that closing can reach this material slice.
    Elapsed descent time, a commanded wrist height and proximity alone do not.
    This geometric condition does not prove loaded contact or task success.
    """
    def __init__(self,rows,pad_links):
        from scipy.spatial import ConvexHull
        if len(pad_links)!=2 or len(set(pad_links))!=2:
            raise ValueError('Two distinct source pad links required')
        self.links=list(pad_links);self.hulls={name:[] for name in self.links};self.vertices={name:[] for name in self.links}
        for row in rows:
            if row['link'] in self.hulls:
                self.hulls[row['link']].append(ConvexHull(row['V']).equations)
                self.vertices[row['link']].append(np.asarray(row['V'],float))
        if not all(self.hulls.values()):raise ValueError('Actual pad collider geometry required')

    def check(self,point,pads,transforms,names):
        from scipy.spatial.transform import Rotation
        point,pads,transforms=np.asarray(point,float),np.asarray(pads,float),np.asarray(transforms,float)
        if (point.shape!=(3,) or pads.shape!=(2,3) or transforms.shape!=(len(names),7)
                or len(set(names))!=len(names) or not set(self.links)<=set(names)
                or not all(np.isfinite(x).all() for x in [point,pads,transforms])):
            raise ValueError('Finite actual material, open pads and matching native body poses required')
        axis=pads[1]-pads[0];length=float(np.linalg.norm(axis))
        if length<1e-8:raise ValueError('Resolved open fingertip axis required')
        axis/=length;hits={}
        for index,name in enumerate(self.links):
            pose=transforms[list(names).index(name)];r=Rotation.from_quat(pose[3:]).as_matrix()
            origin=(point-pose[:3])@r;direction=axis@r;candidates=[]
            for planes in self.hulls[name]:
                den=planes[:,:3]@direction;num=-(planes[:,:3]@origin+planes[:,3])
                if np.any((abs(den)<1e-12)&(num< -1e-10)):continue
                lo=float(np.max(num[den< -1e-12]/den[den< -1e-12],initial=-np.inf))
                hi=float(np.min(num[den>1e-12]/den[den>1e-12],initial=np.inf))
                if lo>hi+1e-10:continue
                if index==0 and hi<=1e-10 and np.isfinite(hi):candidates.append(-hi)
                if index==1 and lo>=-1e-10 and np.isfinite(lo):candidates.append(lo)
            hits[name]=max(0.,min(candidates)) if candidates else None
        return {'pass':all(v is not None for v in hits.values()),
            'inner_surface_distance_to_material_m':hits,'actual_pad_reference_gap_m':length,
            'material_point_world':point.tolist(),'closing_axis_world':axis.tolist(),
            'scope':'Current material lies between actual open pad cross-sections; no contact force, friction or task approval.'}

    def lowering_candidates(self,point,pads,transforms,names,*,up,step_m,maximum_m,budget=257):
        """Read-only open-tool proposals from actual pad extent and material.

        No candidate moves the cloth or establishes rigid clearance. The caller
        must screen the open path and validate actual native arrival again.
        """
        from scipy.spatial.transform import Rotation
        self.check(point,pads,transforms,names)
        point,pads,poses=np.asarray(point,float),np.asarray(pads,float),np.asarray(transforms,float)
        up=np.asarray(up,float)
        if (up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)<1e-12
                or not np.isfinite([step_m,maximum_m]).all() or min(step_m,maximum_m)<=0
                or type(budget) is not int or not 2<=budget<=513):
            raise ValueError('Resolved lowering direction, positive distance bounds and bounded search required')
        up/=np.linalg.norm(up);bounds=[]
        for name in self.links:
            pose=poses[list(names).index(name)];r=Rotation.from_quat(pose[3:]).as_matrix()
            height=(np.concatenate(self.vertices[name])@r.T+pose[:3]-point)@up
            bounds.append([height.min(),height.max()])
        lo=max(0.,max(b[0] for b in bounds));hi=min(maximum_m,min(b[1]for b in bounds))
        if lo>hi:return []
        count=min(budget,max(2,int(np.ceil((hi-lo)/step_m))+1));result=[]
        for distance in np.linspace(lo,hi,count):
            shifted=poses.copy();shifted[:,:3]-=distance*up
            witness=self.check(point,pads-distance*up,shifted,names)
            if witness['pass']:result.append({'distance_m':float(distance),'capture':witness,'proposal_only':True})
        return result


class PlannedPinchContact:
    """Require a target inside both closed pad envelopes before native testing.

    A proposal can intentionally contact deformable material during approach.
    This check prevents accepting a collision-free pose that closes in empty air;
    it does not prove force transmission or frictional retention.
    """
    def __init__(self,rows,model,closed,pad_links):
        from scipy.spatial import ConvexHull
        if len(pad_links)!=2 or len(set(pad_links))!=2:
            raise ValueError('Two distinct opposing pad links required')
        transforms=model.transforms(closed);self.pads={name:[] for name in pad_links}
        for row in rows:
            if row['link'] not in self.pads:continue
            tf=transforms[row['link']]
            vertices=np.asarray(row['V'],float)@tf[:3,:3].T+tf[:3,3]
            self.pads[row['link']].append(ConvexHull(vertices).equations)
        if not all(self.pads.values()):raise ValueError('Missing source pad collision geometry')

    def check(self,point,position,rotation,*,margin=.001):
        point=np.asarray(point,float);position=np.asarray(position,float);rotation=np.asarray(rotation,float)
        if (point.shape!=(3,) or position.shape!=(3,) or rotation.shape!=(3,3)
                or not all(np.isfinite(x).all() for x in [point,position,rotation])
                or not np.isfinite(margin) or margin<0
                or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7)
                or not np.isclose(np.linalg.det(rotation),1.)):
            raise ValueError('Finite target, proper hand pose and nonnegative margin required')
        local=(point-position)@rotation
        distances={name:float(min(np.max(planes[:,:3]@local+planes[:,3]) for planes in hulls))
                   for name,hulls in self.pads.items()}
        return {'pass':all(d<=margin for d in distances.values()),'proposal_only':True,
                'closed_pad_halfspace_distance_m':distances,'margin_m':float(margin),
                'scope':'Closed source convex pad envelopes at selected material point; actual native contact and frictional lift still required.'}


class ApproachSamples:
    """An owned, immutable material snapshot for repeated geometric proposals.

    Construct a new snapshot after any simulated motion. Mutating the caller's
    original array cannot silently change the geometry under its search tree.
    """
    def __init__(self, points):
        from scipy.spatial import cKDTree
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1:]!=(3,) or not len(points) or not np.isfinite(points).all():
            raise ValueError('Finite nonempty material samples required')
        self.points=points.copy();self.points.flags.writeable=False
        self.tree=cKDTree(self.points)


class OpenApproachClearance:
    """Reject open-hand paths that would push material before closing.

    Sample the actual source convex colliders against the current sheet surface.
    This is a proposal screen, not continuous collision detection or a prediction
    of subsequent cloth motion. Native contact validation remains mandatory.
    """
    def __init__(self,rows,model,opened):
        from scipy.spatial import ConvexHull
        transforms=model.transforms(opened);self.rows=[]
        for row in rows:
            tf=transforms[row['link']]
            vertices=np.asarray(row['V'],float)@tf[:3,:3].T+tf[:3,3]
            planes=ConvexHull(vertices).equations
            _,ids=np.unique(np.round(planes,10),axis=0,return_index=True)
            self.rows.append((row['link'],vertices.min(0),vertices.max(0),planes[ids]))

    def check(self,points,positions,rotations,*,margin=.0005):
        prepared=points if isinstance(points,ApproachSamples) else ApproachSamples(points)
        points=prepared.points;positions=np.asarray(positions,float);rotations=np.asarray(rotations,float)
        if (points.ndim!=2 or points.shape[1:]!=(3,) or not len(points)
                or positions.ndim!=2 or positions.shape[1:]!=(3,) or not 2<=len(positions)<=128
                or rotations.shape!=(len(positions),3,3)
                or not all(np.isfinite(x).all() for x in [points,positions,rotations])
                or not np.isfinite(margin) or margin<0
                or not np.allclose(rotations.transpose(0,2,1)@rotations,np.eye(3),atol=1e-7)
                or not np.allclose(np.linalg.det(rotations),1.,atol=1e-7)):
            raise ValueError('Finite current material samples, bounded proper hand poses and nonnegative margin required')
        result={'pass':True,'proposal_only':True,'margin_m':float(margin),'path_samples':len(positions),
            'material_samples':len(points),'first_collision':None,
            'scope':'Sampled open-hand convex clearance against current material; no native motion or task approval.'}
        tree=prepared.tree
        for index,(p,r) in enumerate(zip(positions,rotations)):
            for name,lo,hi,planes in self.rows:
                centre=(lo+hi)/2;radius=np.linalg.norm((hi-lo)/2+margin)
                ids=tree.query_ball_point(centre@r.T+p,radius+1e-10)
                if not ids:continue
                local=(points[ids]-p)@r
                selected=((local>=lo-margin)&(local<=hi+margin)).all(1)
                if not selected.any():continue
                distances=np.max(local[selected]@planes[:,:3].T+planes[:,3],axis=1)
                count=int((distances<margin).sum())
                if count:
                    result.update({'pass':False,'first_collision':{'path_sample':index,'link':name,
                        'material_samples':count,'minimum_halfspace_distance_m':float(distances.min())}})
                    return result
        return result
