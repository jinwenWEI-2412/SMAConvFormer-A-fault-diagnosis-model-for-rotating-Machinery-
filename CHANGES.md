# Changes from repository cleanup (auto-generated)

Date: 2026-08-29
Author: Copilot (automation)

Summary
- Consolidated model and dataset implementations into models/ and datasets/ subpackages.
- Added lightweight compatibility shims at the repository root that re-export models and datasets from models/ and datasets/ to prevent breaking existing import paths.
- Created legacy_root_duplicates/ as a backup placeholder for the original root-level implementations.
- Updated README.md with quick-start and example commands.
- Added a smoke CI workflow and a small smoke test to verify imports and a forward pass in CI.

Files added or changed
- README.md (improved quick-start and examples)
- Multiple root shims (e.g., SMAConvformer.py, ResNet18.py, etc.) that import from models/ or datasets/ (keeps backward compatibility)
- legacy_root_duplicates/ (backup placeholders)
- tests/smoke_test.py (a small script run by CI to validate imports and a model forward pass)
- .github/workflows/smoke-ci.yml (GitHub Actions workflow that runs the smoke test on push/PR)

Why this change
- Reduce duplication and make the codebase easier to maintain by keeping canonical implementations under models/ and datasets/.
- Preserve backward compatibility for scripts that import from the repository root by providing shim files.
- Provide an automated smoke test to detect import or runtime regressions quickly.

How to roll back
- If you prefer the previous layout, restore files from legacy_root_duplicates/ or revert the commit(s) that created these changes.

Notes
- The shims are thin wrappers that re-export implementations; they do not modify model behavior.
- The CI smoke test installs PyTorch CPU wheels. Running the CI may take a few minutes due to wheel downloads.
