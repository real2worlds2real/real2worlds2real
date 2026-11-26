# Input object coverage

Set `object_inventory_contract: 1` in `task.json` for a named object inventory.
Every input object must have a reconstruction route before S2 can pass. S6
requires the current S2 inventory proof as well as the existing object, support,
appearance and task checks. Coverage alone does not approve any geometry.

Objects normally map to the same name in `scene_graph.dynamic_objects`.
Declare alternative routes in `task.json.object_routes`:

```json
{
  "tool": {"kind": "task_reset_only", "object": "tool"}
}
```

An explicitly requested reset tool needs declared physical links,
`initial_presence: task_reset_only` and a provenance explanation. It is not
a missing capture object that can be omitted without evidence.

`scene.twin.inventory` only reads declarations. It never imports or runs task
code. Legacy inputs without this contract keep their previous behavior.
