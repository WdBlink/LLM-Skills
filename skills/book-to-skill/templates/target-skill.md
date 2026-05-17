---
name: <skill-slug>
description: <Specific activation description. Include task, domain, and key book-derived concepts.>
argument-hint: [task, topic, or artifact]
---

# <Skill Title>

Use this skill when <specific task scenario>. It applies the methods from `<book title>` to produce <specific output>.

## Core Workflow

1. Clarify the task and success criteria.
2. Select the relevant book-derived framework.
3. Read references when needed:
   - `references/source-map.md` for evidence and provenance.
   - `references/procedures.md` for detailed workflows.
   - `references/gotchas.md` for failure modes.
   - `templates/` for reusable output formats.
4. Apply the checklist.
5. Validate the answer against source limits and user context.

## Decision Rules

- Use `<framework>` when `<condition>`.
- Prefer `<approach>` over `<alternative>` because `<book-derived reason>`.
- If `<risk>`, check `references/gotchas.md` before finalizing.

## Output Format

```markdown
## Recommendation

## Framework Used

## Steps

## Risks / Gotchas

## Evidence / References
```

## Validation

Before finalizing:

- Did I use a book-derived rule rather than generic advice?
- Did I consult source-map or a reference file for specific claims?
- Did I distinguish book-supported content from my own general knowledge?
- Did I avoid unsupported claims?
