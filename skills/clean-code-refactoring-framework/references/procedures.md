# Procedures

## Workflow 1: Review a Code Slice

1. Identify the user-facing behavior or developer workflow this slice supports.
2. Locate tests and build commands. Mark missing tests as the first risk, not as an afterthought.
3. Read from caller to callee. Capture the intended narrative in one sentence.
4. Check names:
   - Does each name reveal why it exists?
   - Does it encode units, domain language, and distinction from neighbors?
   - Are synonyms used for one concept or one word used for multiple concepts?
5. Check functions:
   - Does the function do one thing at one abstraction level?
   - Are side effects visible from the name and call site?
   - Are commands and queries mixed?
6. Check change topology:
   - One change scattered across many files means shotgun surgery.
   - One file responding to many unrelated changes means divergent change.
   - Behavior living near the wrong data means feature envy.
7. Output findings with severity, refactoring move, behavior risk, and test evidence.

## Workflow 2: Make a Refactoring Plan

1. State behavior boundary: what must remain observably unchanged.
2. State guardrail: exact tests/commands to run after each step.
3. Split work into hats:
   - Refactoring hat: rename, extract, move, inline, encapsulate, split.
   - Feature hat: add or alter behavior.
4. Sequence low-risk moves first:
   - Rename for clarity.
   - Extract repeated or mixed intent.
   - Inline pointless indirection.
   - Move behavior toward data or caller intent.
   - Split phases after boundaries are visible.
5. Add rollback checkpoints:
   - after compile/typecheck;
   - after local tests;
   - after broader integration tests.
6. Stop if tests fail in a way unrelated to the current step, public API compatibility is unclear, or a “small refactor” starts requiring semantic decisions.

## Workflow 3: Refactor Long Function

1. Verify current behavior with existing or characterization tests.
2. Read the function and mark intention blocks.
3. Extract one block into a named function.
4. Run the narrowest test.
5. Rename the extracted function until the call site expresses intent.
6. Repeat until remaining body reads as a short story at one abstraction level.
7. Inline any extraction that adds indirection without improving the story.

## Workflow 4: Refactor Conditional Complexity

1. Separate condition calculation from action with Extract Function.
2. Flatten exceptional branches with guard clauses.
3. Consolidate duplicate conditional fragments.
4. If the same type code or mode branch appears in multiple places, consider polymorphism or strategy.
5. If the special case repeats, introduce a Special Case/Null Object only after behavior is well-covered.

## Workflow 5: Work With Legacy Code Without Tests

1. Do not start with broad cleanup.
2. Identify the narrow behavior you must preserve.
3. Add characterization tests around current behavior, even if current behavior looks odd.
4. Prefer automated safe refactorings: rename, extract with IDE support, move with compiler help.
5. Keep each commit behavior-neutral until tests and understanding improve.
6. Only after the harness is trustworthy should you reorganize deeper structure.

## Decision Rules

- If a comment explains what code does, try renaming or extraction first.
- If parameters travel together, introduce a named object.
- If a function asks many questions of another object, move the function or extract the envious fragment.
- If adding a feature requires copy-paste, do preparatory refactoring before adding the feature.
- If a proposed abstraction has no current caller, inline or remove it.
- If performance concern is speculative, keep clarity and add measurement before optimizing.
