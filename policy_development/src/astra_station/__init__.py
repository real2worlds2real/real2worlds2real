"""Station-world path of the Astra harness.

A *station* is a simulation twin that serves one task through a fixed robot-native interface (observations in, joint
targets out), hides its success judge, and seals its evaluation starts.  The same station serves every method (our
harness and the baselines), so results are comparable.  This package binds a task to such a station, keeps a campaign's
budgets, candidates and hash-linked ledger, exposes registered method tools, and lets any reasoning agent (Claude, GPT,
an operator) choose among them through an agent-neutral decision interface.

Nothing here operates the real robot.  A frozen candidate is a station deliverable (a policy module or a program) that the
real-robot station worker can run unchanged; real validation stays a separate, operator-run step.
"""
