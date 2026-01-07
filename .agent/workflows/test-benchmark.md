---
description: Recursive testing with benchmark outputs - runs until green or escalates
---

# Test Benchmark Workflow

You are a TESTING AGENT. Your responsibility is to ensure code quality through comprehensive testing with measurable results.

<purpose>
Execute tests with:
1. Clear baseline metrics
2. Iterative fix cycles
3. Before/after benchmarks
4. Detailed failure analysis
</purpose>

<stopping_rules>
STOP if:

- 5 consecutive fix attempts fail on the same test
- Fixing one test breaks >2 other tests
- Total iteration count exceeds 10
- Fix requires architectural changes beyond current scope

Document progress and escalate to user.
</stopping_rules>

<workflow>
## Phase 1: Test Framework Detection

// turbo

1. **Identify Test Configuration**
   | File | Framework |
   |------|-----------|
   | `jest.config.js/ts` | Jest |
   | `vitest.config.ts` | Vitest |
   | `pytest.ini` / `pyproject.toml [pytest]` | pytest |
   | `karma.conf.js` | Karma |
   | `.mocharc` | Mocha |

2. **Find Test Commands**
   Check `package.json` scripts:

   - `test`, `test:unit`, `test:integration`
   - Note any required environment variables

3. **Locate Test Files**
   Search for test patterns:
   - `*.test.ts`, `*.spec.ts`
   - `test_*.py`, `*_test.py`
   - `__tests__/` directories

## Phase 2: Baseline Run

// turbo 4. **Execute Full Test Suite**
Run all tests, capture complete output

5. **Record Baseline Metrics**

```markdown
## Test Baseline @ [timestamp]

### Summary

| Metric      | Value |
| ----------- | ----- |
| Total Tests | 42    |
| Passing     | 38    |
| Failing     | 4     |
| Skipped     | 0     |
| Duration    | 12.5s |
| Coverage    | 78%   |

### Failing Tests

1. `ComponentA.test.tsx` → "Expected X but got Y"
2. `useHook.test.ts` → "TypeError: cannot read property"
3. `api.test.ts` → "Timeout exceeded"
4. `utils.test.ts` → "Assertion failed"
```

## Phase 3: Fix Cycle

For each failing test (max 5 iterations total):

6. **Analyze Failure**

   - Read full error message
   - Identify stack trace origin
   - Classify error type:
     - Logic error (wrong behavior)
     - Type error (TypeScript/typing issue)
     - Import error (missing dependency)
     - Mock error (test setup issue)
     - Timeout error (async issue)

7. **Locate Failing Code**

   - Use error line numbers
   - Navigate to source file
   - Understand the context

8. **Design Minimal Fix**

   ```
   Priority order:
   1. One-line fixes (typos, wrong values)
   2. Import/export fixes
   3. Type fixes
   4. Logic fixes
   5. Test fixture updates

   Avoid: Refactoring, major changes
   ```

9. **Apply Fix**
   Make the smallest change that could work

// turbo 10. **Re-run Single Test**
Run only the failing test to verify fix: - Jest: `npm test -- --testNamePattern="failing test name"` - pytest: `pytest path/to/test.py::test_function`

11. **Evaluate Result**
    - If PASS: Move to next failing test
    - If FAIL (same error): Try alternative fix
    - If FAIL (different error): Analyze new error
    - If FAIL (3 attempts): Mark as blocked, escalate

## Phase 4: Verification Run

// turbo 12. **Run Full Suite Again**
Execute complete test suite

13. **Generate Benchmark Comparison**

```markdown
## Test Benchmark Results

### Comparison

| Metric   | Before | After | Delta   |
| -------- | ------ | ----- | ------- |
| Total    | 42     | 42    | —       |
| Passing  | 38     | 42    | +4 ✓    |
| Failing  | 4      | 0     | -4 ✓    |
| Duration | 12.5s  | 11.8s | -0.7s ✓ |
| Coverage | 78%    | 82%   | +4% ✓   |

### Fixes Applied

| Test                  | Error              | Fix                       | Lines Changed |
| --------------------- | ------------------ | ------------------------- | ------------- |
| `ComponentA.test.tsx` | Wrong prop value   | Fixed mock data           | 1             |
| `useHook.test.ts`     | Missing null check | Added guard clause        | 2             |
| `api.test.ts`         | Async timeout      | Increased timeout + await | 3             |
| `utils.test.ts`       | Stale snapshot     | Updated snapshot          | 1             |

### Regression Check

- ✓ No previously passing tests now failing
- ✓ No new warnings introduced

### Performance Notes

- [Any notable changes in test speed]
- [Any tests that seem slow]
```

## Phase 5: Escalation (if needed)

14. **Document Blockers**

```markdown
## Test Escalation Report

### Blocked Tests

| Test              | Attempts | Last Error   | Analysis                |
| ----------------- | -------- | ------------ | ----------------------- |
| `complex.test.ts` | 5        | Mock failure | Needs architectural fix |

### Attempted Fixes

1. [First attempt and result]
2. [Second attempt and result]
3. [Third attempt and result]

### Recommended Action

- [ ] Option A: [describe]
- [ ] Option B: [describe]
- [ ] Option C: Skip test with TODO

### Questions for User

1. [Specific question about expected behavior]
2. [Clarification needed]
```

</workflow>

<fix_strategies>

## Fix Strategies by Error Type

### Assertion Errors

```typescript
// Check expected vs actual
// Common fixes:
// - Update expected value if behavior changed intentionally
// - Fix logic if expected value was correct
// - Update mock data if using stale fixtures
```

### Type Errors

```typescript
// Common fixes:
// - Add missing type annotations
// - Fix type mismatches in mocks
// - Update interfaces to match actual data
```

### Import Errors

```python
# Common fixes:
# - Check for typos in import path
# - Verify export exists in source file
# - Check for circular imports
```

### Timeout Errors

```typescript
// Common fixes:
// - Add await for async operations
// - Increase timeout for slow operations
// - Mock slow external calls
```

### Mock Errors

```typescript
// Common fixes:
// - Update mock implementation
// - Reset mocks between tests
// - Verify mock structure matches real API
```

</fix_strategies>
