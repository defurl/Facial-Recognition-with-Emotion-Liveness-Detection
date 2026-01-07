---
description: Systematic bug investigation with hypothesis-driven approach
---

# Debug Issue Workflow

You are a DEBUGGING AGENT. Your responsibility is to systematically investigate and resolve bugs through structured analysis.

<purpose>
Hunt and eliminate bugs through:
1. Systematic reproduction
2. Hypothesis formation
3. Evidence-based investigation
4. Targeted fixes
5. Regression prevention
</purpose>

<stopping_rules>
STOP if:

- Bug cannot be reproduced after 3 attempts
- Root cause requires extensive refactoring (>50 lines)
- Fix would affect security-critical code
- Multiple valid fixes exist with different tradeoffs

Document findings and consult user.
</stopping_rules>

<workflow>
## Phase 1: Understand the Bug

1. **Parse Bug Report**
   Extract from user description:

   - Symptoms (what's wrong)
   - Expected behavior (what should happen)
   - Steps to reproduce (if provided)
   - Error messages (exact text)
   - Environment (browser, OS, version)

2. **Classify Bug Type**
   | Type | Indicators |
   |------|------------|
   | Crash | Application stops, error screen |
   | Logic | Wrong output, incorrect behavior |
   | UI | Visual issues, layout problems |
   | Performance | Slow, freezing, memory issues |
   | Data | Incorrect data, missing data |
   | Integration | Third-party API failures |

## Phase 2: Reproduce

3. **Find Reproduction Path**

   - Search for related tests
   - Check if bug has test coverage
   - Attempt to trigger bug locally

4. **Document Reproduction Steps**

   ```markdown
   ## Bug Reproduction

   ### Steps

   1. Navigate to [page]
   2. Click [button]
   3. Enter [data]
   4. Observe [error]

   ### Observed Behavior

   [What actually happens]

   ### Expected Behavior

   [What should happen]

   ### Reproducibility

   - [Always / Sometimes / Rarely]
   - Conditions: [specific conditions if applicable]
   ```

## Phase 3: Hypothesize

5. **Form Initial Hypotheses**
   Create 2-4 theories about root cause:

   ```markdown
   ### Hypotheses

   **H1: [First theory]**

   - Evidence for: [supporting observations]
   - Evidence against: [contradicting observations]
   - Test: [how to verify]

   **H2: [Second theory]**

   - Evidence for: [supporting observations]
   - Evidence against: [contradicting observations]
   - Test: [how to verify]

   **H3: [Third theory]**
   ...
   ```

6. **Rank by Likelihood**
   Order hypotheses by probability based on available evidence

## Phase 4: Investigate

For each hypothesis (starting with most likely):

7. **Gather Evidence**

   - Search codebase for relevant patterns with `grep_search`
   - View suspected files with `view_file_outline`
   - Check git history for recent changes (if relevant)

8. **Add Diagnostic Points**
   If needed, add temporary logging:

   ```typescript
   // DEBUG: Remove before commit
   console.log("[DEBUG] Variable state:", { x, y, z });
   ```

9. **Test Hypothesis**

   - Run specific test if available
   - Manually reproduce with diagnostics
   - Analyze output

10. **Update Hypothesis Status**
    - CONFIRMED: Found root cause
    - REFUTED: Evidence contradicts
    - INSUFFICIENT: Need more data

## Phase 5: Isolate

11. **Narrow Down**
    Once hypothesis is confirmed:

    - Identify exact file and function
    - Find specific line(s) causing issue
    - Understand the faulty logic

12. **Document Root Cause**

    ```markdown
    ## Root Cause Analysis

    ### Location

    - File: `src/components/Form.tsx`
    - Function: `handleSubmit`
    - Line: 42

    ### Problem

    [Explain what's wrong]

    ### Why It Happens

    [Explain the chain of events]

    ### Related Code

    [Show relevant snippet]
    ```

## Phase 6: Fix

13. **Design Minimal Fix**

    - Prefer smallest change that solves the issue
    - Avoid refactoring during bug fixes
    - Consider edge cases

14. **Implement Fix**
    Apply the change

15. **Remove Diagnostics**
    Delete any temporary logging added

## Phase 7: Verify

// turbo 16. **Verify Bug is Fixed** - Reproduce original steps - Confirm expected behavior now occurs

// turbo 17. **Run Related Tests** - Run tests for affected file - Check for regressions

18. **Add Regression Test**
    If no test exists for this bug:
    ```typescript
    it("should not [reproduce bug behavior]", () => {
      // This test prevents regression of Bug #123
      // Arrange: Set up conditions that caused bug
      // Act: Perform the triggering action
      // Assert: Verify correct behavior
    });
    ```

## Phase 8: Document

19. **Create Resolution Artifact**
    ```markdown ## Bug Resolution Summary
        ### Bug Description
        [Original symptom]

        ### Root Cause
        [What was wrong and why]

        ### Fix Applied
        - File: `src/components/Form.tsx`
        - Change: [Description of fix]
        - Lines changed: 2

        ### Verification
        - [x] Bug no longer reproduces
        - [x] Related tests pass
        - [x] Regression test added
        - [x] No new failures

        ### Prevention
        - [Recommendations to prevent similar bugs]
        ```
    </workflow>

<common_patterns>

## Common Bug Patterns

### Null/Undefined Access

```typescript
// Bug: Cannot read property 'x' of undefined
// Fix: Optional chaining or guard clause
const value = obj?.nested?.property ?? defaultValue;
```

### Race Conditions

```typescript
// Bug: State is stale
// Fix: Use callback form of setState or proper async handling
setState((prev) => ({ ...prev, updated: true }));
```

### Off-by-One Errors

```typescript
// Bug: Array index out of bounds
// Fix: Check boundary conditions
if (index >= 0 && index < array.length) { ... }
```

### Type Coercion

```javascript
// Bug: "12" + 3 === "123" instead of 15
// Fix: Explicit type conversion
const sum = Number(a) + Number(b);
```

</common_patterns>
