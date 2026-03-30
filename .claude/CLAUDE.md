# CLAUDE.md

## Code Style
- Use **Google-style docstrings** for all functions, classes, and modules.

## Environment
- **Environment:** Assume the necessary python environment is active
- **Environment Variables:** `TNG_API_KEY` must be set to download data from the TNG API.

## Common Commands
All scripts must be run from within the `msd-flow/` directory.
- **Run tests:** `pytest -m tests/`
- **Download data:** `python -m msdflow.data.download_tng` (Override params via Hydra: `... data.download.max_workers=10`)
- **Assign splits:** `python -m msdflow.data.split`

## Graph Navigation (code-review-graph)
When querying the codebase graph, be aware of how Hydra configuration files (`.yaml`) are modeled to avoid context-window bloat and resolve dependencies correctly:
1. **Definitions (`config::`):** Top-level keys defined within YAML files are extracted as `Config` nodes prefixed with `config::` (e.g., `config::seed`, `config::batch_size`). This prevents collisions with Python method names.
2. **Usages/Interpolations (`hydra::`):** When a file references a Hydra variable (e.g., `${data.dataset.split}`), the graph creates a `DEPENDS_ON` edge pointing to a virtual node containing the full path: `hydra::data.dataset.split`.
3. **Tracing Logic:** To trace a variable from usage to definition, do not search for the full `hydra::` path as a node. Instead:
   - Extract the base key (e.g., `split` from `data.dataset.split`).
   - Query the graph for `Config` nodes named `config::split`.
   - If multiple exist, use the original hierarchical path (like `dataset.yaml`) to logically deduce the correct source file.
4. **Instantiations:** Hydra `_target_` instantiations automatically resolve to `DEPENDS_ON` edges pointing to the fully qualified Python class/function in the graph. 