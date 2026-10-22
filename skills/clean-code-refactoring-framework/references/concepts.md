# Concepts

| Concept | Definition | Related Concepts | Evidence |
|---|---|---|---|
| Behavior-preserving refactoring | A structural change that preserves observable behavior while improving understandability or lowering change cost. | tests, two hats, small steps | K0003, E0088 |
| Two hats | Explicitly alternate between adding behavior and changing structure; do not blur them silently. | feature work, refactoring work | K0004, E0088 |
| Boy scout rule | Leave the codebase cleaner than you found it through small bounded improvements. | local cleanup, continuous maintenance | K0002, E0003, E0038 |
| Intention-revealing name | A name that communicates existence, responsibility, usage, and domain meaning. | naming, extract function, API design | K0007, E0003, E0005 |
| One abstraction level | A function should not mix business intention, mid-level operations, and low-level mechanics in one body. | extract function, compose method | K0008, E0004 |
| Single reason to change | A function/class/module should be organized around one responsibility or change pressure. | SRP, divergent change, extract class | K0011, K0012 |
| Self-testing code | Code with a fast, reliable test suite that detects behavior regressions during change. | characterization tests, CI, refactoring safety | K0005, K0006 |
| Code smell | A symptom that suggests a possible design problem and points toward candidate refactorings. | smell catalog, refactoring recipes | K0012-K0017 |
| Preparatory refactoring | A small refactoring done before feature work to make the feature easier and safer to add. | two hats, YAGNI | K0004, E0089 |
| Speculative generality | Abstraction or hooks added for imagined future use that currently increase understanding and maintenance cost. | YAGNI, remove dead code, inline class | K0017 |
| Feature envy | Behavior that talks more to another module's data than to its own module. | move function, extract function | K0014 |
| Data clump | A repeated group of fields or parameters that wants a named structure. | introduce parameter object, extract class | K0015 |
| Shotgun surgery | One conceptual change requires many scattered edits. | move function, combine functions into class | K0013 |
| Divergent change | One module changes for unrelated reasons. | extract class, split phase | K0012 |
| Guard clause | A condition placed up front to handle exceptional cases and leave the main path flat. | decompose conditional, nested conditions | K0020 |
| Reversible refactoring pair | A pair such as extract/inline used to find the right boundary by moving both directions. | small steps, naming | K0018 |
