"""Fail early on unsupported direct task-context calls before costly simulation.

Dynamic Python remains permitted; this checks only statically visible direct
ctx.method(...) calls, not the physical validity or success of the experiment.
"""
import ast
import inspect


def validate(path, context):
    tree=ast.parse(path.read_text())
    assigned={n.attr for node in ast.walk(tree) if isinstance(node,(ast.Assign,ast.AnnAssign))
              for n in ast.walk(node) if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store)
              and isinstance(n.value,ast.Name) and n.value.id=='ctx'}
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):continue
        call=node.func
        if not (isinstance(call,ast.Attribute) and isinstance(call.value,ast.Name) and call.value.id=='ctx'):continue
        if call.attr in assigned:continue
        if not hasattr(context,call.attr):raise ValueError(f'Task check line{node.lineno}: unsupported ctx.{call.attr}; inspect the task-context API')
        method=getattr(context,call.attr)
        if not callable(method):raise ValueError(f'ctx.{call.attr} is not callable')
        if any(isinstance(a,ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):continue
        try:signature=inspect.signature(method)
        except (TypeError,ValueError):continue
        try:signature.bind(*[None]*len(node.args),**{k.arg:None for k in node.keywords})
        except TypeError as exc:raise ValueError(f'Task check line{node.lineno}: ctx.{call.attr}: {exc}') from exc
