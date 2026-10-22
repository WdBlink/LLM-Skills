# Derived Skill Recommendations

The combined framework is broad. If this becomes heavily used, split it into narrower task skills:

1. `code-smell-reviewer`
   - Focus: review findings, severity calibration, smell-to-refactoring mapping.
   - Best for: PR review and codebase audits.

2. `legacy-refactoring-planner`
   - Focus: characterization tests, safe sequencing, behavior-preserving commits.
   - Best for: old code with weak tests.

3. `naming-and-api-clarity`
   - Focus: intention-revealing names, function declarations, public API clarity.
   - Best for: library and SDK design.

4. `test-cleanliness-refactor`
   - Focus: readable tests, FIRST, test DSLs, removing implementation coupling.
   - Best for: brittle test suites.
