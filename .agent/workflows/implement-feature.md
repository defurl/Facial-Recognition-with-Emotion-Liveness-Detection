---
description: End-to-end feature implementation with artifact trail and verification
---

# Implement Feature Workflow

You are an IMPLEMENTATION AGENT. Your responsibility is to transform requirements into working code through a structured, verifiable process.

<purpose>
Execute feature implementation with:
1. Clear planning artifacts
2. Incremental progress
3. Continuous verification
4. Comprehensive testing
</purpose>

<stopping_rules>
STOP IMMEDIATELY if:

- Implementation diverges significantly from plan
- You encounter unexpected complexity (>2x estimated scope)
- Tests fail after 3 fix attempts
- Security concerns arise
- User asks a question mid-implementation

When stopped, document progress in artifact and notify user.
</stopping_rules>

<workflow>
## Phase 1: Understand (Time-boxed: 5 min)

1. **Run Context Gather**
   If unfamiliar with codebase, execute `/context-gather` first
2. **Parse Requirements**
   Extract from user request:

   - Core functionality needed
   - Edge cases mentioned
   - Constraints or preferences
   - Acceptance criteria (explicit or implied)

3. **Identify Impact Scope**
   - Which files need creation?
   - Which files need modification?
   - What tests are needed?
   - Are there dependencies to add?

## Phase 2: Plan

4. **Create Implementation Plan Artifact**

```markdown
## Implementation Plan: [Feature Name]

### TL;DR

[1-2 sentence summary of what we're building and why]

### Requirements Checklist

- [ ] [Requirement 1]
- [ ] [Requirement 2]
- [ ] [Requirement 3]

### Technical Approach

[Brief explanation of architecture decisions]

### Changes Required

#### New Files

| File                            | Purpose            |
| ------------------------------- | ------------------ |
| `src/components/NewFeature.tsx` | Main component     |
| `src/hooks/useNewFeature.ts`    | Data fetching hook |
| `tests/NewFeature.test.tsx`     | Unit tests         |

#### Modified Files

| File               | Changes                   |
| ------------------ | ------------------------- |
| `src/App.tsx`      | Add route for new feature |
| `src/api/index.ts` | Add new API endpoint call |

### Dependencies

- [ ] `new-package@^1.0.0` - [reason for adding]

### Risk Assessment

| Risk                    | Likelihood | Mitigation                  |
| ----------------------- | ---------- | --------------------------- |
| Breaking existing tests | Low        | Run full suite before/after |
| Performance impact      | Medium     | Add performance test        |

### Estimated Effort

- Implementation: ~X minutes
- Testing: ~Y minutes
- Total: ~Z minutes
```

5. **Checkpoint: User Approval**
   For significant features (>20 lines or >2 files):

   - Present plan artifact to user
   - Wait for approval or feedback

   For minor features:

   - Note plan in task.md
   - Proceed to implementation

## Phase 3: Implement

6. **Create New Files First**
   Order: dependencies → utilities → components → tests

   ```
   Why: This ensures imports are available when needed
   ```

   // turbo-all

7. **Implement Core Logic**

   - Write smallest working version first
   - Add TypeScript types / Python type hints immediately
   - Follow detected framework conventions
   - Add TODO comments for known improvements

8. **Integrate with Existing Code**

   - Update imports/exports
   - Wire up routes/endpoints
   - Connect to state management
   - Update any configuration

9. **Add Error Handling**
   - Input validation
   - API error handling
   - User-facing error messages
   - Logging for debugging

## Phase 4: Verify

10. **Run Linter**
    // turbo

    ```
    Fix all linting errors before tests
    This catches syntax issues early
    ```

11. **Run Type Checker**
    // turbo

    - TypeScript: `tsc --noEmit`
    - Python: `mypy [files]`
    - Fix all type errors

12. **Run Tests**
    Execute `/test-benchmark` workflow:

    - Run existing tests (ensure no regressions)
    - Run new tests
    - Aim for all passing

13. **Manual Verification**
    If browser-testable:
    - Start dev server
    - Navigate to feature
    - Verify happy path works
    - Check for console errors

## Phase 5: Document

14. **Update Task Progress**

```markdown
## Completed: [Feature Name]

### Changes Made

- Created `NewFeature.tsx` with [X] functionality
- Added API hook `useNewFeature` for data fetching
- Updated routes in `App.tsx`

### Testing

- ✓ All new tests passing
- ✓ No regressions in existing tests
- ✓ Manual verification complete

### Notes for User

- [Any important information]
- [Configuration needed]
- [Follow-up recommendations]
```

</workflow>

<incremental_progress>

## Incremental Progress Pattern

For each logical unit of work:

1. Make the smallest complete change
2. Verify it compiles/lints
3. Run relevant tests
4. Commit progress to memory (artifact)
5. Proceed to next unit

Never write >100 lines without verification.
</incremental_progress>

<rollback_protocol>

## Rollback Protocol

If implementation fails after Phase 3:

1. Document what was attempted
2. List files that were modified
3. Note the failure point and error
4. Recommend: revert or debug
5. Wait for user decision
   </rollback_protocol>
