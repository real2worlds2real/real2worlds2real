"""DROID column/pose adapter shared by data generation and physical calibration."""
import numpy as np
from scipy.spatial.transform import Rotation

def pose6_to_T(x):
    """DROID xyz + Euler (scipy "xyz", extrinsic) -> 4x4; works on (..., 6)."""
    x = np.asarray(x, float)
    T = np.zeros(x.shape[:-1] + (4, 4))
    T[..., :3, :3] = Rotation.from_euler("xyz", x[..., 3:].reshape(-1, 3)).as_matrix().reshape(x.shape[:-1] + (3, 3))
    T[..., :3, 3] = x[..., :3]
    T[..., 3, 3] = 1.0
    return T

def _col(df, name):
    return np.stack([np.asarray(v, float).ravel() for v in df[name].values])


def read_dataframe(path):
    import pandas as pd
    return pd.read_parquet(path)
