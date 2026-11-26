"""Preserve authored passive task assets while an export settles its robot."""
import hashlib

from pxr import Sdf, Usd


class AuthoredTaskState:
    def __init__(self, stage, path='/World/Objects'):
        self.path = Sdf.Path(path)
        if not self.path.IsAbsolutePath() or not stage.GetPrimAtPath(self.path):
            raise ValueError('Existing absolute task subtree required')
        self.source = self._copy(stage.GetRootLayer())
        self.sha256 = hashlib.sha256(self.source.ExportToString().encode()).hexdigest()

    def _copy(self, layer):
        saved = Sdf.Layer.CreateAnonymous('authored_task_state.usda')
        Sdf.CreatePrimInLayer(saved, self.path.GetParentPath())
        if not Sdf.CopySpec(layer, self.path, saved, self.path):
            raise ValueError('Cannot snapshot the authored task subtree')
        return saved

    def restore(self, stage):
        """Call after stopping simulation; keep the robot's separately saved pose."""
        with Usd.EditContext(stage, stage.GetRootLayer()):
            stage.RemovePrim(self.path)
            if not Sdf.CopySpec(self.source, self.path, stage.GetRootLayer(), self.path):
                raise ValueError('Cannot restore authored passive task state')
        actual = self._copy(stage.GetRootLayer()).ExportToString()
        if hashlib.sha256(actual.encode()).hexdigest() != self.sha256:
            raise ValueError('Restored task subtree differs from the authored input')
        return {'pass': True, 'path': str(self.path), 'authored_sha256': self.sha256,
                'scope': 'Original task geometry, transforms and physics attributes; robot pose remains separate.'}
