# Gotchas

## Common Failure Modes

- **Semantic drift hidden as cleanup**: A refactor changes behavior but the diff is described as “just cleanup”. Fix by separating hats and adding explicit behavior tests.
- **Mega-diff cleanup**: Formatting, renaming, moving, logic changes, and feature work are mixed. Fix by splitting commits and validating after each structural move.
- **Comment deodorant**: Long comments explain confusing code. Fix by trying Rename or Extract Function before adding prose.
- **Speculative abstraction**: Interfaces, subclasses, hooks, or config switches exist only for imagined future use. Fix by inlining or removing until real variation appears.
- **Test lock-in**: Tests assert implementation details so every refactor breaks them. Fix by moving tests toward behavior and public contract.
- **Dirty tests**: Test setup is harder to read than production behavior. Fix by extracting domain-specific test helpers.
- **Over-refactoring stable code**: A low-risk, rarely changed area is cleaned for style alone. Fix by requiring a change-cost reason.
- **Performance panic**: A readability refactor is rejected without measurement. Fix by benchmarking or profiling the relevant path.

## Smell Severity Calibration

| Severity | Use When | Example |
|---|---|---|
| High | The smell makes likely changes unsafe or already caused bugs | scattered condition branches across payment logic |
| Medium | The smell slows understanding or local changes but has a clear bounded fix | long function with extractable phases |
| Low | The smell is mostly local readability | unclear variable name or redundant wrapper |

## Stop Conditions

- No reliable way to verify behavior.
- Public API or data migration contract is unclear.
- Refactoring exposes a real product decision, not merely a structural issue.
- More than three sequential refactor steps fail tests in different ways.
- The diff cannot be explained as named, behavior-preserving transformations.
