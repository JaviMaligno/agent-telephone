# Delivery profile

## vcs
- host: local-only; no origin
- integration-branch: master
- production-branch: none
Source: Git metadata · checked 2026-10-23

## tracker
- kind: none
Source: local security follow-up · checked 2026-10-23

## environments
- kind: none
Source: no deployed environment in this source-maintenance task · checked 2026-10-23

## deploy
- kind: none
Rules:
- [invariant no-experiments] No model calls, experiment runs, image builds, publishing, or existing virtualenv mutation in this task.
Source: security-only local source changes · checked 2026-10-23
