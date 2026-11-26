# The Astra work library

The work library connects the task, reward, and tool registries so Astra can use the results of earlier runs
when choosing what to try next. At the start of a task, Astra retrieves relevant experience and decides what to
reuse, adapt, or explore. After a development evaluation or a training failure, the runner records the outcome
and its source. A model-written summary starts as a draft and requires independent evidence to become a verified record.

## What the library stores

- **Task and stage experience.** Initialization ranges, observation and action contracts, the method used, and
  the number of successful development episodes out of the full number evaluated.
- **Rewards.** Formulas or native parameters, applicable features, parent versions, success protocols,
  validation levels, and results.
- **Tools.** Registered implementations, reusable combinations of tools, qualification records, the experience
  they draw on, version history, and rollback relationships.
- **Failures and lessons.** Failure categories, suspected causes, conditions for reproduction, and unresolved
  capability gaps.

Each record is immutable and identified by its content hash. The library appends evidence separately, distinguishes
verified observations from agent hypotheses, and includes failures in search results. Final evaluation data stays
outside the library used for tuning. Raw trajectories and model weights stay in the experiment directory.

## Finding and reusing experience

`task_inspect`, the planning catalog, and reward decisions all return relevant experience. This version ranks
matches deterministically by task family, contact type, stage skill, and contract. The ranking indicates relevance.
Astra decides whether to use a result.

When registering a task, you can supply `semantic_tags` such as `family:insertion`, `contact:rigid`, and
`skill:alignment`. The tags become part of the immutable task contract, so tasks beyond the five built-in names
can retrieve experience from the same family. For a task without tags, use `library_search` with a task description.

- **`exact_contract`** means the task, stage, backend, observation and action contracts, and success protocol all
  match. The current runtime must still verify the source and the capabilities needed to use it.
- **`adaptation_required`** means the task or skill is related, but the contracts differ. You can reuse the
  approach or reward structure after binding it to the target task and validating it there.
- **`inspiration_only`** marks a result as a design reference only.

Peg insertion and gear insertion share alignment, contact, and insertion skills. Using a source policy on the
other task still requires checking its action dimensions, geometry, and success criteria. A new reward experiment
can import a previous reward if it has an independent development evaluation, its contract matches exactly, and
its source is still valid. The imported version retains its provenance.

## Storage and access

Configure the directory for policy memory at the site. Use `library_inspect` for its status and `library_search` for retrieval. The storage interface in `work_library.py` supports draft records, verified evidence receipts, quarantine and invalidation. The station uses only records applicable to its current task.
