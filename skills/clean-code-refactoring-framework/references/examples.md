# Examples

## Example 1: Review Finding

```markdown
- Medium `billing.ts:84` Long Function: `calculateInvoice` mixes lookup, pricing, discount policy, and formatting.
  - Suggested refactoring: Extract Function for `priceForLineItem`, then Split Phase between calculation and presentation.
  - Behavior risk: invoice totals and formatting can drift independently.
  - Verification: existing invoice golden tests plus one characterization test for discount edge cases.
```

## Example 2: Refactoring Plan

```markdown
Behavior boundary: invoice JSON and rendered statement stay identical.

1. Add characterization tests for current invoice output.
2. Rename ambiguous `data` and `tmp` variables to domain terms.
3. Extract price calculation into `priceFor`.
4. Extract rendering into `renderStatement`.
5. Split calculation and formatting phases.
6. Run unit tests after every step and full test suite at the end.
```

## Example 3: When Not To Refactor

```markdown
I would not introduce an abstract `PaymentProcessor` yet. There is one implementation and no active variation. The current smell is naming and duplicated validation, not missing polymorphism. Clean those first; create the abstraction when a second real processor appears.
```
