# Contributing

Keep operations eager, deterministic, and explicit about unsupported shapes. Every differentiable operation must include analytical gradient tests; use finite differences for nontrivial derivatives. Public API changes require documentation and migration notes.

Before submitting a change, run a Debug and Release build, `ctest --output-on-failure`, and clang-tidy when available. Do not add a required dependency without documenting why it is necessary.
