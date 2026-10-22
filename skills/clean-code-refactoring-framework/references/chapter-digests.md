# Chapter Digests

## Extraction Summary

- Source A: `clean-code.epub`, extracted by zipfile fallback, 41,045 words, evidence E0001-E0074.
- Source B: `refactoring-2e.md`, converted from MOBI HTML with pandoc, 42,293 words, evidence E0074-E0149.
- Combined extraction: 83,346 words, 149 chunks.

## Clean Code Digest

### Ch.1 整洁代码

Core idea: Code is read and modified far more than it is initially written. Clean code is therefore an economic discipline, not ornament. The practical norm is the boy scout rule: leave the code slightly better than found.

Key evidence: E0002, E0003, E0038.

Operational takeaway: Prefer local, continuous improvement over periodic rewrite campaigns.

### Ch.2 有意义的命名

Core idea: Names should carry intent, distinction, consistency, searchability, domain vocabulary, and context. Naming is a design activity because bad names hide confused concepts.

Key evidence: E0003, E0004, E0005.

Operational takeaway: If you cannot name it clearly, do not extract it yet; first find the concept boundary.

### Ch.3 函数

Core idea: Functions should be small, do one thing, stay at one abstraction level, avoid side effects, separate command from query, and keep error handling from dominating the happy path.

Key evidence: E0004, E0005.

Operational takeaway: Extract functions around intention-revealing steps, then rename until the call site reads like a narrative.

### Ch.4 注释

Core idea: Comments cannot repair unclear code. Good comments explain intent, warnings, legal context, or public API use; bad comments duplicate, mislead, or deodorize bad structure.

Key evidence: E0006, E0090.

Operational takeaway: Treat explanatory comments as refactoring leads before treating them as documentation.

### Ch.7-8 错误处理与边界

Core idea: Error handling and third-party boundaries should be isolated so they do not pollute core logic. Unknown or external APIs need learning tests and wrappers.

Key evidence: E0005, E0013, E0015.

Operational takeaway: Keep policy, integration detail, and domain logic apart.

### Ch.9 测试

Core idea: Clean tests protect production code from decay. Readability is the central quality of tests, supported by speed, independence, repeatability, self-validation, and timeliness.

Key evidence: E0015, E0016.

Operational takeaway: If tests become painful to read or run, refactor them before they stop protecting the code.

### Ch.10 类

Core idea: Class size is measured by responsibility. Cohesive classes have fewer reasons to change and reduce modification risk.

Key evidence: E0016.

Operational takeaway: Split classes by responsibility and change pressure, not merely by line count.

## Refactoring 2e Digest

### Ch.1 第一个示例

Core idea: Refactoring starts with tests, then proceeds through small transformations: extract, rename, move, split phase, and introduce polymorphism when needed.

Key evidence: E0077, E0084, E0087, E0088.

Operational takeaway: Make understanding executable by moving it from your head into the code structure.

### Ch.2 重构的原则

Core idea: Refactoring means behavior-preserving structural change. Work alternates between feature hats and refactoring hats. Refactoring supports YAGNI, architecture evolution, and later performance tuning when guarded by tests.

Key evidence: E0088, E0089, E0079.

Operational takeaway: Do not mix semantic change and structure change in one unreviewable step.

### Ch.3 代码的坏味道

Core idea: Smells are decision triggers, not rules. They point from a maintenance symptom to a candidate refactoring.

Key evidence: E0089, E0148.

Operational takeaway: Diagnose the change pattern before choosing the refactoring.

### Ch.4 构筑测试体系

Core idea: Self-testing code reduces debugging time and gives rapid feedback during refactoring. Tests must be frequent, local, and trustworthy.

Key evidence: E0090, E0091.

Operational takeaway: For legacy code, first create a small behavioral harness around the slice you intend to change.

### Ch.5-12 重构名录

Core idea: The catalog is a set of safe transformation recipes. Many moves are reversible pairs: extract/inline, move/remove middleman, command/function, subclass/delegate.

Key evidence: E0091, E0092, E0093, E0116.

Operational takeaway: Prefer named, reversible moves over improvised edits.
