"""Optional cuDSS0.5 SPD factorization over existing CUDA tensor storage.

The separately installed NVIDIA library retains its own license under the
installation directory. No physical model, matrix regularization or tolerance
is changed here. Callers must check their actual residual and conservation.
"""
from calibrated_scene_reconstruction.core.paths import ROOT
import ctypes as c
import os
from pathlib import Path

import torch


from calibrated_scene_reconstruction.core.environment import setting

def library_path():
    return Path(setting('CUDSS_LIBRARY',
        str(ROOT/'envs/numerics/nvidia/cu12/lib/libcudss.so.0')))


def _library():
    path=library_path()
    if not path.is_file():raise RuntimeError('Optional cuDSS0.5 library is not installed: '+str(path))
    lib=c.CDLL(str(path));pointer=c.c_void_p;integer=c.c_int
    signatures={
        'cudssGetProperty':[integer,c.POINTER(integer)],
        'cudssCreate':[c.POINTER(pointer)],'cudssDestroy':[pointer],
        'cudssConfigCreate':[c.POINTER(pointer)],'cudssConfigDestroy':[pointer],
        'cudssDataCreate':[pointer,c.POINTER(pointer)],'cudssDataDestroy':[pointer,pointer],
        'cudssSetStream':[pointer,pointer],
        'cudssMatrixCreateCsr':[c.POINTER(pointer),c.c_int64,c.c_int64,c.c_int64,
            pointer,pointer,pointer,pointer,integer,integer,integer,integer,integer],
        'cudssMatrixCreateDn':[c.POINTER(pointer),c.c_int64,c.c_int64,c.c_int64,pointer,integer,integer],
        'cudssMatrixDestroy':[pointer],
        'cudssExecute':[pointer,integer,pointer,pointer,pointer,pointer,pointer],
        'cudssDataGet':[pointer,pointer,integer,pointer,c.c_size_t,c.POINTER(c.c_size_t)]}
    for name,arguments in signatures.items():
        fn=getattr(lib,name);fn.argtypes=arguments;fn.restype=integer
    version=[]
    for property_ in range(3):
        value=integer();_check(lib.cudssGetProperty(property_,c.byref(value)),'version');version.append(value.value)
    if version[:2]!=[0,5]:raise RuntimeError('cuDSS ABI adapter requires version0.5, found '+str(version))
    return lib,tuple(version)


def _check(status,operation):
    if status:raise RuntimeError(f'cuDSS {operation} failed with status {status}')


class Factor:
    """Explicitly owned GPU SPD factor; sparse matrix storage stays alive."""
    def __init__(self,matrix):
        if (matrix.layout!=torch.sparse_csr or matrix.device.type!='cuda' or matrix.dtype!=torch.float64
                or len(matrix.shape)!=2 or matrix.shape[0]!=matrix.shape[1] or not matrix.shape[0]
                or max(matrix.shape)>=2**31 or matrix.values().numel()>=2**31):
            raise ValueError('Nonempty square CUDA float64 CSR matrix within int32 indexing required')
        if not torch.isfinite(matrix.values()).all():raise ValueError('Finite sparse matrix values required')
        self.lib,self.version=_library();self.device=matrix.device;self.n=matrix.shape[0]
        self.handle=c.c_void_p();self.config=c.c_void_p();self.data=c.c_void_p();self.matrix=c.c_void_p()
        self.rows=matrix.crow_indices().to(dtype=torch.int32).contiguous()
        self.columns=matrix.col_indices().to(dtype=torch.int32).contiguous();self.values=matrix.values().contiguous()
        self.stream=torch.cuda.current_stream(self.device)
        try:
            _check(self.lib.cudssCreate(c.byref(self.handle)),'create')
            _check(self.lib.cudssSetStream(self.handle,self.stream.cuda_stream),'stream')
            _check(self.lib.cudssConfigCreate(c.byref(self.config)),'config')
            _check(self.lib.cudssDataCreate(self.handle,c.byref(self.data)),'data')
            # cudaDataType: CUDA_R_32I=10, CUDA_R_64F=1; cuDSS0.5:
            # SPD=3, full matrix view=0, zero index base=0.
            _check(self.lib.cudssMatrixCreateCsr(c.byref(self.matrix),self.n,self.n,self.values.numel(),
                self.rows.data_ptr(),None,self.columns.data_ptr(),self.values.data_ptr(),10,1,3,0,0),'CSR')
            with torch.cuda.stream(self.stream):
                self.stream.synchronize()
                for phase in [1,2]:
                    _check(self.lib.cudssExecute(self.handle,phase,self.config,self.data,self.matrix,None,None),'factor phase '+str(phase))
                self._info()
        except BaseException:
            self.close();raise

    def _info(self):
        self.stream.synchronize();info=c.c_int();written=c.c_size_t()
        _check(self.lib.cudssDataGet(self.handle,self.data,0,c.byref(info),c.sizeof(info),c.byref(written)),'info')
        if info.value:raise RuntimeError('cuDSS numerical factorization/solve info '+str(info.value))

    def solve(self,rhs):
        if rhs.ndim not in [1,2] or rhs.shape[0]!=self.n or rhs.device!=self.device or rhs.dtype!=torch.float64:
            raise ValueError('Matching CUDA float64 RHS required')
        if not torch.isfinite(rhs).all():raise ValueError('Finite right hand side required')
        one=rhs.ndim==1;b=rhs[:,None] if one else rhs
        b=b.T.contiguous().T;x=torch.empty((b.shape[1],self.n),dtype=b.dtype,device=b.device).T
        bm=c.c_void_p();xm=c.c_void_p()
        try:
            for handle,value in [(bm,b),(xm,x)]:
                _check(self.lib.cudssMatrixCreateDn(c.byref(handle),self.n,b.shape[1],self.n,value.data_ptr(),1,0),'dense')
            with torch.cuda.stream(self.stream):
                _check(self.lib.cudssExecute(self.handle,8,self.config,self.data,self.matrix,xm,bm),'solve')
                self._info()
            if not torch.isfinite(x).all():raise RuntimeError('Nonfinite cuDSS solution')
            return x[:,0] if one else x
        finally:
            for handle in [xm,bm]:
                if handle.value:self.lib.cudssMatrixDestroy(handle)

    def close(self):
        if self.handle.value:
            self.stream.synchronize()
            if self.matrix.value:self.lib.cudssMatrixDestroy(self.matrix);self.matrix=c.c_void_p()
            if self.data.value:self.lib.cudssDataDestroy(self.handle,self.data);self.data=c.c_void_p()
            if self.config.value:self.lib.cudssConfigDestroy(self.config);self.config=c.c_void_p()
            self.lib.cudssDestroy(self.handle);self.handle=c.c_void_p()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
