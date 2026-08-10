# Explainability Specification

## Goal

Answer:

> Why was this sample flagged?

## Sample Explanation

Show:
- text
- class label if known
- risk score
- risk level
- detector scores
- layer-wise evidence
- cluster evidence
- nearest/similar samples where available

## Optional Token Evidence

Token-level highlighting may be added later using an appropriate attribution method.

It must be clearly labelled as model attribution/evidence, not proof that a token is malicious.

## Unlabelled Samples

Do not invent a class label.

Display:
`Class: Unknown`

## Explainability Principle

The UI should explain evidence, not make an absolute claim of maliciousness.
