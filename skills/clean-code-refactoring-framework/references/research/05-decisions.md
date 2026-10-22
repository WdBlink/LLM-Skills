# 05 Decisions

## Framework Decisions Derived From the Books

1. Preserve behavior first.
   - Use when: any actual refactoring request.
   - Evidence: E0088, E0089.

2. Split feature work from refactoring work.
   - Use when: the user asks to add behavior and clean structure in one pass.
   - Evidence: E0088.

3. Require tests before non-trivial structural change.
   - Use when: existing behavior must be preserved.
   - Evidence: E0077, E0090, E0091.

4. Treat names as design probes.
   - Use when: concepts are unclear or abstractions feel forced.
   - Evidence: E0003, E0005, E0092.

5. Diagnose change topology before choosing a refactoring.
   - Use when: technical debt is described vaguely.
   - Evidence: E0089, E0148.

6. Remove unused abstraction.
   - Use when: code exists only for imagined future variation.
   - Evidence: E0089, E0090.

## Operational Policy

The generated Skill should not provide only principles. Every answer should include at least one concrete move, verification method, or stop condition unless the user explicitly asks for theory.
