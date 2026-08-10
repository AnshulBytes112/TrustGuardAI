# AI Agent Workflow

## Initial Prompt

```text
Read /docs/00_PROJECT_OVERVIEW.md, /docs/01_IMPLEMENTATION_PLAN.md,
/docs/03_SRS.md, /docs/04_SYSTEM_ARCHITECTURE.md,
and /docs/21_AI_CODING_RULES.md.

Do not write code yet.

Summarize:
1. architecture,
2. current phase,
3. constraints,
4. relevant interfaces,
5. acceptance criteria.

Identify contradictions before implementation.
```

## Task Prompt

```text
Implement TASK-XXX from /docs/22_TASK_BREAKDOWN.md.

First read the relevant specification files.

Constraints:
- do not redesign architecture,
- do not implement future tasks,
- do not fabricate research results,
- preserve sample_id,
- do not access poison_ground_truth in detection,
- add tests,
- report files changed,
- report tests executed.

If a requirement is ambiguous, stop and identify the ambiguity instead of inventing a research decision.
```

## Review Prompt

```text
Review the implementation of TASK-XXX against:
- SRS
- architecture
- task acceptance criteria
- coding rules

Do not rewrite unrelated code.

Report:
1. deviations,
2. bugs,
3. missing tests,
4. reproducibility issues,
5. documentation updates required.
```

## Experiment Prompt

```text
Run experiment EXXX using only the committed configuration.

Do not alter parameters silently.

Save:
- configuration,
- seed,
- metrics,
- predictions,
- artifacts,
- logs.

Do not write conclusions until the experiment completes.
```
