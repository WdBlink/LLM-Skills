# Book-to-Skill Conversion Schema

Use this schema while filling `work/*.md` and the final `references/*.md` files.

## Knowledge Item Types

- `concept`: term, definition, distinction, mental model.
- `procedure`: repeatable workflow or step-by-step method.
- `decision-rule`: if/then rule or decision criterion.
- `checklist`: ordered or unordered verification list.
- `anti-pattern`: behavior the book warns against.
- `example`: case, story, code sample, or worked example.
- `template`: reusable output/input structure.
- `quote`: short quoted phrase only when exact wording matters.

## Evidence Fields

Every important item should record:

```yaml
id: K0001
type: decision-rule
name: Exact framework name if present
rule: Use X when Y; avoid Z because W.
evidence: [E0012, E0013]
source_location: Chapter 3 / page 57 / EPUB location if available
confidence: high | medium | low
notes: Any uncertainty or interpretation boundary
```

## Target Skill Decision

Ask:

1. What recurring task should this skill improve?
2. What does the book teach that the base model might not reliably know?
3. What should the agent do differently after loading the skill?
4. Which details are always needed in `SKILL.md` and which can be referenced on demand?
5. Which evals would prove the skill is useful?
