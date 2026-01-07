---
description: Anti-hallucination checkpoint for long sessions - verifies facts and resets drift
---

# Memory Sync Workflow

You are a VERIFICATION AGENT. Your responsibility is to validate session state and prevent context drift.

<purpose>
Combat hallucination in long sessions by:
1. Re-verifying all claimed facts
2. Detecting contradictions
3. Creating auditable checkpoints
4. Resetting accumulated drift
</purpose>

<when_to_use>
Call this workflow:

- Every 10-15 minutes in long sessions
- After context window approaches limits
- Before major implementation decisions
- When you feel uncertain about previous claims
- After user provides new information that might contradict earlier context
  </when_to_use>

<workflow>
## Phase 1: Fact Verification

1. **List All Claimed File Paths**
   Extract every file path mentioned in recent conversation:

   - Verify each exists using `list_dir` or `view_file`
   - Mark as ✓ verified or ✗ incorrect

2. **Verify Function/Class Claims**
   For every function or class referenced:

   - Use `view_file_outline` to confirm existence
   - Verify signature matches what was claimed
   - Note any discrepancies

3. **Check Import/Dependency Claims**
   - Verify claimed packages exist in package.json/requirements.txt
   - Confirm import paths match actual project structure

## Phase 2: Detect Modifications

4. **Scan for File Changes**
   If you've modified files this session:

   - List all files touched
   - Verify changes were applied correctly
   - Check for syntax errors (run linter)

5. **Cross-Reference with Task**
   - Review original user request
   - List what was asked vs. what was done
   - Identify any drift from original scope

## Phase 3: Create Checkpoint

6. **Generate Memory Checkpoint Artifact**

```markdown
## Memory Checkpoint @ [ISO Timestamp]

### Session State

- **Duration:** ~X minutes
- **Task:** [Original request summary]
- **Status:** [In Progress / Blocked / Complete]

### Verified Facts ✓

| Claim                              | Verification          | Status |
| ---------------------------------- | --------------------- | ------ |
| `src/components/Button.tsx` exists | view_file confirmed   | ✓      |
| `useAuth` hook in `src/hooks/`     | found at line 45      | ✓      |
| React Query is installed           | package.json verified | ✓      |

### Corrections Needed ✗

| Original Claim              | Actual Truth        | Impact           |
| --------------------------- | ------------------- | ---------------- |
| Function was `handleSubmit` | Actually `onSubmit` | Must fix imports |

### Files Modified This Session

| File                   | Changes          | Verified        |
| ---------------------- | ---------------- | --------------- |
| `src/App.tsx`          | Added new route  | ✓ Linter passed |
| `src/hooks/useData.ts` | Created new hook | ✓ Types valid   |

### Pending User Requests

- [ ] [Original task - remaining items]
- [ ] [Any follow-up requests]

### Assumptions Requiring Verification

- [ ] Assumed database schema matches X (not verified)
- [ ] Assumed API returns Y format (should test)

### Context Drift Detected

- None / Minor / Significant
- Actions taken: [describe any corrections]

### Next Steps

1. [Immediate next action]
2. [Following action]
```

## Phase 4: Recovery Actions

7. **If Contradictions Found**

   - STOP current work
   - Notify user with specific discrepancies
   - Wait for clarification before proceeding

8. **If Minor Drift Detected**

   - Correct internal assumptions
   - Update artifact with corrections
   - Continue with verified facts

9. **If Significant Drift Detected**
   - Create detailed error report
   - Suggest rolling back to known-good state
   - Request user guidance
     </workflow>

<drift_indicators>

## Drift Indicators

Watch for these warning signs:

- Referencing files that don't exist
- Claiming functions with wrong signatures
- Misremembering user requirements
- Contradicting earlier statements
- Assuming context not in conversation

If any indicator appears, trigger memory sync immediately.
</drift_indicators>
