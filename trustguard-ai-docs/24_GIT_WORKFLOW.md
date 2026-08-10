# Git Workflow

## Branches

```text
main
develop
feature/TASK-xxx-short-name
```

## Commit Format

```text
feat: implement TASK-025 isolation forest
fix: preserve sample ids during feature extraction
test: add poisoning determinism tests
docs: update risk scoring specification
```

## Rules

- One logical task per branch.
- Keep commits small.
- Do not commit datasets or large model artifacts.
- Use `.gitignore` for generated artifacts.
- Pull/rebase before final merge where appropriate.
- Do not force-push shared branches.

## Pull Request Checklist

- Task ID
- Summary
- Tests
- Docs updated
- No fabricated results
- No unrelated changes
