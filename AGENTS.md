# Project instructions

This repository is a research candidate for Apex Text Clustering, not a deployed
miner or a promise of income. Keep the submission self-contained in solution.py.

- Runtime: Python 3.12; install requirements-dev.txt for tests and benchmarks.
- Run `python -m pytest -q` after behavior changes.
- Run `python benchmark.py --suite smoke` for API/algorithm changes; run the
  relevant public-corpus and 5,000-text suites when performance changes.
- Preserve the exact upstream baseline and its MIT attribution.
- Never use labels/hidden answers to choose clusters at inference time.
- Do not add network calls, model downloads, eval/exec, subprocess execution,
  wallet access or payments to solution.py.
- Track failures and benchmark regressions. Label synthetic versus public versus
  live competition results accurately. Never claim an unmeasured improvement.
- Paid submissions and registrations are separate from development. Do not
  perform them without explicit authorization for the concrete fee.
