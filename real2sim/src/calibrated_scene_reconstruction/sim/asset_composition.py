"""Reject broken USD references before flattening can erase their errors."""


def legacy_errors(stage):
    """USD24.05 in Isaac5.1 predates Stage.GetCompositionErrors."""
    from pxr import Usd
    errors = list(stage._GetPcpCache().layerStack.localErrors)
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        index = prim.GetPrimIndex()
        errors.extend(index.localErrors)
        nodes = [index.rootNode]
        while nodes:
            node = nodes.pop()
            if not node:
                continue
            if node.layerStack:
                errors.extend(node.layerStack.localErrors)
            nodes.extend(node.children)
    return errors


def audit(stage):
    source = stage.GetCompositionErrors() if hasattr(stage, 'GetCompositionErrors') else legacy_errors(stage)
    unique = {(type(error).__name__, str(error)) for error in source}
    errors = [{'kind': kind, 'message': message} for kind, message in sorted(unique)]
    return {'pass': not errors, 'errors': errors,
            'scope': 'Composed source USD references, including missing prim targets; not task or appearance approval.'}
