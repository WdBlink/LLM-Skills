# Source Map

Evidence IDs come from `.generated/book-to-skill/clean-code-refactoring-framework/source/chunks.jsonl`.
This file paraphrases book-derived knowledge and intentionally avoids long copied passages.

| Item ID | Extracted Knowledge | Type | Chapter/Page/Location | Evidence IDs | Confidence |
|---|---|---|---|---|---|
| K0001 | Code quality is largely a reading and changeability problem, not merely a correctness problem. | mental-model | Clean Code ch.1; Refactoring ch.2 | E0003, E0088, E0089 | high |
| K0002 | Leave code cleaner than you found it, but keep improvements local and bounded. | decision-rule | Clean Code ch.1, ch.15 | E0003, E0037, E0038 | high |
| K0003 | A refactoring is a behavior-preserving structural change; if behavior changes, it is not pure refactoring. | definition | Refactoring ch.2.1 | E0088, E0089 | high |
| K0004 | Separate feature work from refactoring work. Switch hats deliberately and keep progress measurable. | workflow | Refactoring ch.2.2 | E0088, E0089 | high |
| K0005 | The first step before refactoring is a reliable test safety net. | precondition | Refactoring ch.1.3, ch.4 | E0077, E0090, E0091 | high |
| K0006 | Tests must be clean and readable; dirty tests reduce the ability to improve production code. | decision-rule | Clean Code ch.9 | E0015, E0016 | high |
| K0007 | Names should reveal intent, unit, role, and domain; hard-to-name code often reveals unclear design. | heuristic | Clean Code ch.2; Refactoring ch.6 | E0003, E0005, E0092 | high |
| K0008 | Functions should be small, do one thing, and operate at one abstraction level. | heuristic | Clean Code ch.3 | E0004, E0005 | high |
| K0009 | Comments are useful for intent and warnings, but often indicate code that should be renamed or restructured. | smell | Clean Code ch.4; Refactoring ch.3.24 | E0006, E0090 | high |
| K0010 | Error handling should not obscure the main path; isolate try/catch and keep error handling coherent. | heuristic | Clean Code ch.7 | E0005, E0013 | medium |
| K0011 | Classes should be organized around responsibilities and reasons to change, not raw line counts alone. | mental-model | Clean Code ch.10 | E0016 | high |
| K0012 | Divergent change means one module changes for multiple reasons; split responsibilities or phases. | smell-to-refactor | Refactoring ch.3 | E0089 | high |
| K0013 | Shotgun surgery means one change requires scattered edits; move behavior/data together. | smell-to-refactor | Refactoring ch.3.8 | E0089, E0148 | high |
| K0014 | Feature envy means behavior is closer to another module's data; move or extract it. | smell-to-refactor | Refactoring ch.3.9 | E0089, E0129, E0130 | high |
| K0015 | Data clumps should usually become a named object or parameter object. | smell-to-refactor | Refactoring ch.3.10, ch.6.8 | E0089, E0096 | high |
| K0016 | Repeated switches are a stronger signal than a single conditional; consider polymorphism when type logic repeats. | smell-to-refactor | Refactoring ch.3.12, ch.10.4 | E0089, E0148 | high |
| K0017 | Speculative generality and dead code should be removed, not admired as future-proofing. | anti-pattern | Refactoring ch.3.15, ch.8.9 | E0089, E0090 | high |
| K0018 | Extract Function and Inline Function are paired tools; the right boundary emerges by moving both directions. | refactoring-pair | Refactoring ch.6.1-6.2 | E0092, E0093 | high |
| K0019 | Split Phase is a key move when a process mixes separate conceptual stages. | refactoring-tool | Refactoring ch.6.11 | E0076, E0088 | medium |
| K0020 | Guard clauses, extracted conditions, and polymorphism are progressively stronger tools for condition complexity. | refactoring-tool | Refactoring ch.10 | E0116, E0117 | high |
| K0021 | Structure first, then optimize with measurements; a clearer structure usually makes later performance work easier. | tradeoff | Refactoring ch.2.8 | E0079, E0080 | medium |
| K0022 | Architecture can evolve through refactoring, but this only works when tests and change discipline are present. | mental-model | Refactoring ch.2.6 | E0089 | medium |
