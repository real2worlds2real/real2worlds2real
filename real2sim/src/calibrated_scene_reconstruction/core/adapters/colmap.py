"""Read COLMAP observations without importing scene or simulator policy."""
import struct
import numpy as np

def image_tracks(path, names):
    """Read selected image observations, preserving COLMAP point identities.

    Coordinates are the measured source features, not model projections.
    Unselected images are skipped without loading their observation arrays.
    """
    wanted=set(names);result={}
    observation=np.dtype([('xy','<f8',(2,)),('point_id','<i8')])
    with path.open('rb') as stream:
        count=struct.unpack('<Q',stream.read(8))[0]
        for _ in range(count):
            header=stream.read(64)
            if len(header)!=64:raise ValueError('Truncated COLMAP image header')
            name=bytearray()
            while True:
                char=stream.read(1)
                if not char:raise ValueError('Truncated COLMAP image name')
                if char==b'\0':break
                name.extend(char)
            name=name.decode('utf-8');size=struct.unpack('<Q',stream.read(8))[0]
            if name not in wanted:
                stream.seek(size*24,1);continue
            data=stream.read(size*24)
            if len(data)!=size*24:raise ValueError('Truncated COLMAP image observations')
            values=np.frombuffer(data,dtype=observation)
            valid=values['point_id']>=0
            if name in result:raise ValueError('Duplicate COLMAP image name')
            result[name]=(values['xy'][valid].copy(),values['point_id'][valid].copy())
    if set(result)!=wanted:raise ValueError('Requested source images absent from COLMAP model')
    return result


def point_records(path, point_ids):
    """Read selected sparse point XYZ/error/track count without pycolmap."""
    wanted=set(map(int,point_ids));result={}
    with path.open('rb') as stream:
        count=struct.unpack('<Q',stream.read(8))[0]
        for _ in range(count):
            header=stream.read(51)
            if len(header)!=51:raise ValueError('Truncated COLMAP point record')
            identity,*values=struct.unpack('<QdddBBBdQ',header)
            xyz=np.asarray(values[:3]);error=values[6];length=values[7]
            if identity in wanted:result[identity]=(xyz,error,length)
            stream.seek(length*8,1)
    if set(result)!=wanted:raise ValueError('Source feature references a missing COLMAP point')
    return result


def read_points3d(path):
    """(xyz (n, 3), error (n,), track length (n,)) of a COLMAP points3D.bin."""
    data = path.read_bytes()
    n = int(np.frombuffer(data, np.uint64, 1, 0)[0])
    xyz, err, trk = np.empty((n, 3)), np.empty(n), np.empty(n, np.int64)
    off = 8
    for i in range(n):
        xyz[i] = np.frombuffer(data, np.float64, 3, off + 8)
        err[i] = np.frombuffer(data, np.float64, 1, off + 8 + 24 + 3)[0]
        trk[i] = int(np.frombuffer(data, np.uint64, 1, off + 8 + 24 + 3 + 8)[0])
        off += 8 + 24 + 3 + 8 + 8 + 8 * trk[i]
    return xyz, err, trk
