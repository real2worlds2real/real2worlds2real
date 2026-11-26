"""Finite tetrahedral sheet construction, independent of any object identity."""
import numpy as np


def extrude(points,faces,thickness):
    import trimesh
    v=np.asarray(points,float);f=np.asarray(faces,int)
    if not np.isfinite(thickness) or thickness<=0:raise ValueError('Positive physical thickness required')
    normals=np.asarray(trimesh.Trimesh(v,f,process=False).vertex_normals)
    if not np.isfinite(normals).all() or np.any(np.linalg.norm(normals,axis=1)<.9):raise ValueError('Resolved sheet normals required')
    count=len(v);solid=np.r_[v-normals*thickness/2,v+normals*thickness/2];tet=[]
    # Global vertex ordering chooses identical side diagonals on shared faces.
    for a,b,c in np.sort(f,axis=1):
        tet.extend([[a,b,c,a+count],[b,c,a+count,b+count],[c,a+count,b+count,c+count]])
    tet=np.asarray(tet);q=solid[tet];det=np.linalg.det(np.stack([q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]],axis=2))
    if np.any(abs(det)<1e-20):raise ValueError('Unresolved zero-volume shell cells')
    ordered=np.sort(f,axis=1);tri=v[ordered]
    orientation=np.einsum('ij,ij->i',np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),normals[ordered].mean(1))
    if np.any(det*np.repeat(orientation,3)<=0):raise ValueError('Physical extrusion inverted a sheet cell; revise thickness or geometry')
    negative=det<0;tet[negative]=tet[negative][:,[1,0,2,3]]
    faces4=np.concatenate([tet[:,[0,2,1]],tet[:,[0,1,3]],tet[:,[1,2,3]],tet[:,[2,0,3]]])
    _,inverse,counts=np.unique(np.sort(faces4,axis=1),axis=0,return_inverse=True,return_counts=True)
    if counts.max()>2:raise ValueError('Nonmanifold tetrahedral construction')
    surface=faces4[counts[inverse]==1]
    mass_weights=np.zeros(len(solid));volume=abs(det)/6
    for i in range(4):np.add.at(mass_weights,tet[:,i],volume/4)
    if (mass_weights<=0).any():raise ValueError('Every volume node must own material')
    return {'V':solid,'F':surface,'tetrahedra':tet,'volume_m3':float(volume.sum()),'volume_weights':mass_weights}
