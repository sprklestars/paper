# Decision Integrity in Federated Learning

Minimal reproducibility package for **Decision Integrity in Federated Learning:
Trust-State Manipulation and Functional-Audit Gaps**.

Start with [README.txt](README.txt) for setup, data acquisition, execution,
corrected interpretation of historical labels, and limitations.

- 56 Python files supporting 50 registered experimental and figure tasks.
- Selected historical results, including negative results and simple-cap comparisons.
- [Evidence index](evidence/index.json) connecting each task to source hashes,
  historical logs, environments, and result files.
- [Baseline attribution](BASELINES.txt) and [references](paper/refs.bib).
- [Package validation](evidence/package-validation.json): syntax checks and
  30 quick regression tests passed; research experiments were not rerun for release.

Raw datasets, manuscript files, correspondence, signed documents, virtual
environments, and the original working repository's history are not included.

## Quick regression checks

Run from this repository's root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r exp/requirements.txt
.venv/bin/python -B exp/test_scorers.py
.venv/bin/python -B exp/test_fiva.py
```

Read the full instructions before running research experiments: some scripts
execute at import time, and reruns can overwrite bundled result files.

`SHA256SUMS.txt` covers the distributed package files, excluding itself and
this repository landing page. Historical absolute-path metadata is redacted
in six copies; experiment source and numerical results are unchanged.

This is a single-process research harness, not a deployed federated system.
Public access does not imply an open-source license; no new reuse license
has been applied on the rights holders' behalf. See README.txt for details.
