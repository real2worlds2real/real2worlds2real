"""Bounded remeshing of cut shell boundaries, before material coordinates exist.

Clipping an inferred bonded root can create submicron triangles. Such a mesh
can be geometrically valid yet numerically unusable by a continuum solver.
No simulated or measured state is repaired by this construction operation.
"""
import numpy as np


def quality(points,faces):
    x=np.asarray(points,float)[np.asarray(faces,int)]
    lengths=np.stack([np.linalg.norm(x[:,i]-x[:,(i+1)%3],axis=1) for i in range(3)],axis=1)
    altitude=np.linalg.norm(np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]),axis=1)/lengths.max(1)
    return lengths,altitude


def condition_root(points,faces,roots,*,minimum_altitude_m,max_displacement_m,maximum_aspect_ratio=30.,include_free_boundary=False,
                   minimum_root_spacing_m=0.,maximum_relative_area_change=.01):
    """Collapse only tiny root-region edges under an explicit shape-error bound.

Retain manifold edge incidence, triangle orientation, root membership and all
free-surface components. Refuse unresolved slivers instead of deleting faces.
The caller must check actual root contact and geometry again afterward.
"""
    v=np.array(points,float,copy=True);f=np.array(faces,int,copy=True)
    original=v.copy();groups={i:[i] for i in range(len(v))};root=set(map(int,roots))
    if not 0<minimum_altitude_m<max_displacement_m or not 2<=maximum_aspect_ratio<=100:
        raise ValueError('Explicit positive resolution, bounded aspect ratio and displacement bounds required')
    if not 0<=minimum_root_spacing_m<=2*max_displacement_m or not 0<maximum_relative_area_change<=.1:
        raise ValueError('Bounded root sampling and explicit construction-area error required')
    def scores_for(q,tri):
        lengths,altitude=quality(q,tri)
        score=np.minimum(altitude/minimum_altitude_m,maximum_aspect_ratio*altitude/lengths.max(1))
        if minimum_root_spacing_m:
            is_root=np.isin(tri,list(root))
            pairs=is_root&np.roll(is_root,-1,axis=1)
            spacing=np.where(pairs,lengths,np.inf).min(1)
            score=np.minimum(score,spacing/minimum_root_spacing_m)
        return score
    area0=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1).sum()/2
    collapsed=0;relocations=0;flips=0;maximum_chord_error=0.
    for iteration in range(8*len(v)):
        lengths,altitude=quality(v,f);scores=scores_for(v,f);bad=np.flatnonzero(scores<1-1e-7)
        if not len(bad):break
        # Small in-plane vertex relocation often resolves a cut cell without
        # deleting material. Actual root vertices stay fixed. The global
        # construction displacement and material-area budgets still apply.
        relocated=False
        for index in bad:
            tri=f[index]
            if not root.intersection(tri) and not include_free_boundary:continue
            for vertex in tri:
                vertex=int(vertex)
                if vertex in root:continue
                gradient=[];epsilon=minimum_altitude_m*.001
                def local_score(q):
                    return float(scores_for(q,tri[None])[0])
                for axis in range(3):
                    plus=v.copy();minus=v.copy();plus[vertex,axis]+=epsilon;minus[vertex,axis]-=epsilon
                    gradient.append((local_score(plus)-local_score(minus))/(2*epsilon))
                gradient=np.asarray(gradient);norm=float(gradient@gradient)
                if norm<1e-20:continue
                delta=gradient*(1-scores[index])*1.05/norm
                for fraction in (1.,.5):
                    vv=v.copy();vv[vertex]+=delta*fraction
                    if np.linalg.norm(original[groups[vertex]]-vv[vertex],axis=1).max()>max_displacement_m:continue
                    changed=np.any(f==vertex,axis=1);ff=f[changed]
                    oldn=np.cross(v[ff[:,1]]-v[ff[:,0]],v[ff[:,2]]-v[ff[:,0]])
                    newn=np.cross(vv[ff[:,1]]-vv[ff[:,0]],vv[ff[:,2]]-vv[ff[:,0]])
                    if np.any(np.einsum('ij,ij->i',oldn,newn)<=0):continue
                    newscore=scores_for(vv,ff)
                    if np.maximum(0,1-newscore).sum()>=np.maximum(0,1-scores[changed]).sum()-1e-8:continue
                    area=np.linalg.norm(np.cross(vv[f[:,1]]-vv[f[:,0]],vv[f[:,2]]-vv[f[:,0]]),axis=1).sum()/2
                    if abs(area/area0-1)>maximum_relative_area_change:continue
                    v=vv;relocated=True;break
                if relocated:break
            if relocated:break
        if relocated:relocations+=1;continue
        # A different diagonal can resolve a poor root cell without moving
        # any source vertex. Restrict nonplanar chord error and total area.
        flipped=False
        for index in bad:
            tri=f[index]
            if not root.intersection(tri):continue
            for j in range(3):
                a,b,c=map(int,np.roll(tri,-j))
                adjacent=np.flatnonzero(np.isin(f,[a,b]).sum(1)==2)
                if len(adjacent)!=2:continue
                other=int(adjacent[0] if adjacent[1]==index else adjacent[1])
                d=int(next(x for x in f[other] if x not in (a,b)))
                if np.any(np.isin(f,[c,d]).sum(1)==2):continue
                proposed=np.array([[c,d,b],[d,c,a]])
                old=f[[index,other]];oldx=v[old];newx=v[proposed]
                oldn=np.cross(oldx[:,1]-oldx[:,0],oldx[:,2]-oldx[:,0]);newn=np.cross(newx[:,1]-newx[:,0],newx[:,2]-newx[:,0])
                if np.any(newn@(oldn.sum(0))<=0):continue
                planes=np.cross(v[b]-v[a],v[[c,d]]-v[a]);normal=planes/np.linalg.norm(planes,axis=1)[:,None]
                error=max(abs((v[d]-v[a])@normal[0]),abs((v[c]-v[a])@normal[1]))
                if error>max_displacement_m:continue
                before_l,before_h=quality(v,old);after_l,after_h=quality(v,proposed)
                if scores_for(v,proposed).min()<=scores_for(v,old).min()*1.05:continue
                area_change=(np.linalg.norm(newn,axis=1).sum()-np.linalg.norm(oldn,axis=1).sum())/2
                current_area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1).sum()/2
                if abs((current_area+area_change)/area0-1)>maximum_relative_area_change:continue
                f[[index,other]]=proposed;flipped=True;break
            if flipped:break
        if flipped:flips+=1;maximum_chord_error=max(maximum_chord_error,error);continue
        edges=set()
        all_edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
        unique,counts=np.unique(all_edges,axis=0,return_counts=True)
        boundary={tuple(e) for e in unique[counts==1]} if include_free_boundary else set()
        for tri in f[bad]:
            rooted=bool(root.intersection(tri))
            for i in range(3):
                a,b=sorted((int(tri[i]),int(tri[(i+1)%3])))
                # A needle touching the root can have its shortest edge at
                # the opposite, free side. Bounded construction may collapse
                # that edge too; source shape and topology remain checked.
                if rooted or (a,b) in boundary:edges.add((a,b))
        accepted=False
        for a,b in sorted(edges,key=lambda ab:np.linalg.norm(v[ab[0]]-v[ab[1]])):
            target=v[a] if a in root and b not in root else (v[b] if b in root and a not in root else (v[a]+v[b])/2)
            members=groups[a]+groups[b]
            if np.linalg.norm(original[members]-target,axis=1).max()>max_displacement_m:continue
            ff=f.copy();ff[ff==b]=a;keep=np.all(np.diff(np.sort(ff,axis=1),axis=1)!=0,axis=1);ff=ff[keep]
            if not len(ff):continue
            vv=v.copy();vv[a]=target
            old=f[keep];oldn=np.cross(v[old[:,1]]-v[old[:,0]],v[old[:,2]]-v[old[:,0]])
            newn=np.cross(vv[ff[:,1]]-vv[ff[:,0]],vv[ff[:,2]]-vv[ff[:,0]])
            if np.any(np.einsum('ij,ij->i',oldn,newn)<=0):continue
            ee=np.sort(np.concatenate([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]]),axis=1)
            if np.unique(ee,axis=0,return_counts=True)[1].max()>2:continue
            if len(np.unique(np.sort(ff,axis=1),axis=0))!=len(ff):continue
            candidate_area=np.linalg.norm(newn,axis=1).sum()/2
            if abs(candidate_area/area0-1)>maximum_relative_area_change:continue
            if (a in root or b in root):root.add(a)
            root.discard(b);groups[a]=members;del groups[b];v=vv;f=ff;collapsed+=1;accepted=True;break
        if not accepted:
            free_bad=sum(not root.intersection(tri) for tri in f[bad])
            raise ValueError('Root remeshing cannot resolve thin triangles within declared geometry bound; '
                             f'bad_triangles={len(bad)}, free_bad_triangles={free_bad}, '
                             f'maximum_aspect_ratio={float((lengths[bad].max(1)/altitude[bad]).max()):.6g}, '
                             f'minimum_edge_m={float(lengths[bad].min()):.6g}')
    used=np.unique(f);mapping=np.full(len(v),-1,int);mapping[used]=np.arange(len(used));vnew=v[used];fnew=mapping[f]
    lengths,altitude=quality(vnew,fnew)
    if np.any(scores_for(v,f)<1-1e-7):
        raise ValueError('Unresolved thin/needle material triangles')
    area1=np.linalg.norm(np.cross(vnew[fnew[:,1]]-vnew[fnew[:,0]],vnew[fnew[:,2]]-vnew[fnew[:,0]]),axis=1).sum()/2
    if abs(area1/area0-1)>maximum_relative_area_change:raise ValueError('Root conditioning exceeded the explicit construction-area error budget')
    return vnew,fnew,[int(mapping[x]) for x in sorted(root) if mapping[x]>=0],{
        'collapsed_edges':collapsed,'maximum_construction_displacement_m':float(max(np.linalg.norm(original[groups[x]]-v[x],axis=1).max() for x in used)),
        'vertex_relocations':relocations,'edge_flips':flips,'maximum_local_retriangulation_chord_error_m':maximum_chord_error,
        'minimum_altitude_m':float(altitude.min()),'relative_area_change':float(area1/area0-1),
        'maximum_aspect_ratio':float((lengths.max(1)/altitude).max()),
        'minimum_root_spacing_required_m':minimum_root_spacing_m,'construction_area_error_budget':maximum_relative_area_change,
        'scope':'Bounded construction remesh before natural material chart; not simulation-state repair'}
