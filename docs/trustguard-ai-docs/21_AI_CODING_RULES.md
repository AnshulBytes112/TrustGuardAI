# AI Coding Rules — Mandatory

## Before Coding

1. Read the relevant `/docs` files.
2. Identify the task ID.
3. Check dependencies and existing interfaces.
4. Do not redesign architecture unless explicitly requested.

## During Coding

5. Implement only the requested task.
6. Do not invent research results.
7. Do not hard-code experiment results.
8. Do not silently change dataset preprocessing.
9. Do not silently change train/test splits.
10. Do not use poison ground truth inside detection.
11. Keep ML logic separate from API/UI.
12. Keep research configuration outside source code.
13. Preserve sample_id throughout every pipeline stage.
14. Add tests for new logic.
15. Use deterministic seeds where applicable.
16. Record versions and configurations.

## Research Integrity

17. Never claim exact FLARE reproduction without verification.
18. Clearly label FLARE-inspired adaptations.
19. Never claim an algorithm is universally effective without experiments.
20. Never fabricate metrics, charts, screenshots, or conclusions.

## Data Safety

21. Original datasets are immutable.
22. Quarantine is reversible.
23. Purification creates a new dataset version.
24. Never expose hidden poison ground truth through normal detection APIs.

## Completion

A task is complete only when:
- code exists
- tests pass
- docs remain consistent
- configuration is documented
- no unrelated files were changed
