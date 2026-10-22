# Clean Code + Refactoring Checklist

## Before

- [ ] Behavior boundary is explicit.
- [ ] Existing tests/build commands are identified.
- [ ] If tests are missing, characterization or smoke tests are planned.
- [ ] Feature work and refactoring work are separated.

## Diagnosis

- [ ] Names reveal intent, units, and domain meaning.
- [ ] Functions do one thing at one abstraction level.
- [ ] Comments are not covering for unclear code.
- [ ] Error handling does not obscure the main path.
- [ ] Classes/modules have one main reason to change.
- [ ] Change topology was checked for divergent change or shotgun surgery.
- [ ] Data/behavior placement was checked for feature envy and data clumps.

## Execution

- [ ] Each step is a named refactoring.
- [ ] Tests run after each meaningful step.
- [ ] Public API changes are isolated or migrated with expand/contract.
- [ ] Performance claims are measured.

## After

- [ ] Full relevant test suite passed.
- [ ] Diff separates formatting, refactoring, and behavior change.
- [ ] Residual risk and follow-up refactors are documented.
