# Run configurations

| What you are configuring | Start here |
|---|---|
| End-to-end run | Copy the [chain request](workflow/chain.request.example.json) and set paths to your capture, station site, and task factory. Run `astra-e2e schemas` to inspect the request contract. |
| Oversight and resource bounds | Start with the [intake example](workflow/intake.example.json). |
| Policy station | The [station configurations](../policy_development/configs/station/) define the site, splits and available tools. |
| Calibrated scene reconstruction | [Scene configurations](../real2sim/configs/) provide camera, object and simulation inputs. Run reconstruction commands from `real2sim/` so session-relative paths resolve. |
| Harness runtime adapters | The [runner templates](../real2sim/examples/) configure simulation, fitting and video adapters for a particular runtime. |
