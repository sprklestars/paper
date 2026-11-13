Decision Integrity in Federated Learning
Trust-State Manipulation and Functional-Audit Gaps
Minimal reproducibility package

SCOPE

This package contains the experimental source, selected historical results,
configuration and data-provenance records supporting the manuscript. It does
not contain the manuscript, correspondence, signed documents, thesis, raw public
datasets, model checkpoints, virtual environments, or the original Git history.

The experimental Python source is byte-identical to the inspected working copy.
Packaging did not rerun the research experiments or change their numerical
results. Some historical comments and printed labels predate manuscript
corrections: read INTERPRETATION below before using the results.

The 50 registered tasks cover the main article and supplement, including null
and negative results. All are small source files; retaining their dependencies
avoids publishing only favourable examples. This is a single-process research
harness, not a deployed federated system or an implementation of GFSecAgg.

CONTENTS

exp/                         Experiment scripts and regression tests
exp/tools/                   Data conversion, provenance and rank utilities
exp/results/                 Selected structured results and CIFAR partition
exp/provenance/              Dataset fingerprints and historical environments
paper/fig/                   Five existing result figures
paper/refs.bib               Bibliographic attribution, not manuscript source
BASELINES.txt                Local adaptations and stable manuscript labels
evidence/logs/               One selected historical log per registered task
evidence/index.json          Task-to-source, environment and result mapping
evidence/file-origins.json   Original/distributed hashes and redaction flags
SHA256SUMS.txt               Distributed file hashes (excluding itself)

ENVIRONMENT AND QUICK CHECKS

Run setup and driver commands from the package root. Install into isolated
virtual environments; no administrator privileges are needed.

    python3 -m venv .venv
    .venv/bin/python -m pip install -r exp/requirements.txt
    .venv/bin/python -B exp/test_scorers.py
    .venv/bin/python -B exp/test_fiva.py

These are 19 scoring/rank tests and 11 FIVA regression tests, not a full research
replication. Some tests use tiny synthetic arrays and short simulations.

The historical main environment used Python 3.14.6, NumPy 2.5.2, SciPy 1.18.1
and scikit-learn 1.9.0 on macOS x86_64. Requirements files are portable dependency
ranges, not a promise of cross-platform bitwise equality. Historical pip-freeze
records are under exp/provenance/requirements-*.lock.txt. Matching these may
require the original Python/platform. Exact environment and package versions
for each selected historical task are also retained in evidence/index.json.

CIFAR used a separate environment with Python 3.12.14, torch 2.2.2 and NumPy
1.26.4. Create it only if running the CNN experiment:

    python3.12 -m venv .venv-torch
    .venv-torch/bin/python -m pip install -r exp/requirements-torch.txt

RESEARCH RUNS (OPTIONAL, NOT AUTOMATIC)

Work in a separate extracted copy when rerunning: scripts can overwrite the
bundled exp/results/*.json, exp/e2_results.json and paper/fig/*.pdf. Keep this
release or its ZIP as the untouched evidence baseline. New driver logs use a
new exp/results/runs/<timestamp>/ directory, but current result files still
change. Never import arbitrary experiment modules just to inspect them:
several execute experiments at import time or call sys.exit.

Run a selected registered task, for example:

    .venv/bin/python exp/verify_all.py --only a14_auditgap_exact.py

Run the full registry after installing both environments and obtaining data:

    .venv/bin/python exp/verify_all.py --figures

The registry contains 44 main-environment tasks, 1 torch task and 5 figure
tasks. Missing torch causes the CNN task to be skipped; omitting --figures
skips figures unless explicitly selected with --only. Inspect PASSED/FAILED/
SKIPPED in the generated summary, not just the driver's process exit status.
A successful run is not a mathematical proof or a deployment guarantee.

The driver sets each child process's directory to exp/. If running an older
script directly, first enter exp/ and use ../.venv/bin/python SCRIPT.py
(or ../.venv-torch/bin/python for CIFAR). Figure output goes to paper/fig/.
Large training tasks can take substantial time; packaging made no runtime
or full-rerun claim.

PUBLIC DATA

No third-party dataset is redistributed. Users obtain datasets from their
sources and comply with their terms. California housing is fetched through
scikit-learn's fetch_california_housing, which may download on first use and
requires a writable cache. Digits is supplied by scikit-learn's load_digits.

For CIFAR, the historical source was HuggingFace uoft-cs/cifar10, downloaded
through the mirror below. From the package root:

    mkdir -p exp/data
    curl -L --fail -o exp/data/cifar10_train.parquet https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet
    curl -L --fail -o exp/data/cifar10_test.parquet https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet
    .venv-torch/bin/python exp/tools/fetch_cifar.py

The conversion utility decodes downloaded parquet; it does not download data.
The original upstream URLs and observed file/array hashes are recorded in
exp/provenance/cifar10.json. An immutable upstream revision was not recorded:
main may change. Compare downloaded parquet SHA-256 and decoded X/y shapes,
dtypes and C-order byte hashes against the shipped record before claiming
identical inputs. Regenerating a provenance file is not itself a comparison.
NPZ container hashes can change on recompression even when arrays are equal.

The actual input consists of the first 20000 training rows and first 2000 test
rows in parquet order, without initial shuffling. exp/results/e10_partition.json
contains the actual ten-client indices (Dirichlet alpha=0.5, partition seed=0).
Training seeds 0..5 share this partition. Attack injection is at aggregation 21,
after 20 warm-up rounds and one further local epoch.

INTERPRETATION AND KNOWN LIMITATIONS

A13/A26: California and digits losses are training-data probe MSE, not independent
held-out test error. A13 uses two 24-unit ReLU hidden layers, 3000 local SGD steps,
learning rate 0.004 and batch size 64. Preprocessing uses pooled data; digits
client samples may overlap. A26's across-run dispersion changes initialization
as well as minibatch randomness. Its capture test directly changes scoring
variance with fixed model vectors; it is not an end-to-end local-data attack.
Historical labels such as 'test' and 'same start' must not override these facts.

A15: a separate four-monitor configuration is used, not E2's common detector
suite. Norm/distance MAD reference samples exclude the known target, whereas
the centre includes it; the Krum check uses argmax and cosine uses the peer
mean. Acceptance requires all Boolean decisions to match the no-attack case.
This is not independent common-false-positive-rate calibration or an unknown-
attacker deployment. One fixed direction per seed and a finite [0,2] bisection
are searched; approximately 1.0001x is a search result, not a global harm bound.

A24/E9: the historical name M2w (and some M2 labels) denotes final-weight uniform
mixing, w_mix=(1-lambda)/N+lambda*w, not mixing independent reference information.
Uniform mass is allocated to every client, including malicious clients. For a
coalition of K among N clients, the bound is (1-lambda)*K/N+lambda; a per-client
cap c/N gives min(1,K*c/N). There is no K-independent trusted-anchor guarantee.
A24 shares 0.046 (K=1) and 0.686 (K=15) belong to cap c=1.4, not to mixing.

E2: offline krum_adv means all evaluated coalition clients are excluded by the
local Multi-Krum variant; partial exclusion is not that event. E2b forwards
these rules. E2's added clean control is seed 0; broader matched controls are
in E2b-E2d. These rules do not replace A15's separate configuration.

M4: theoretical conditions include nonzero carriers, equivariant matching costs,
a unique optimum and exact arithmetic. The CIFAR greedy matcher is a numerical
adaptation, not the exact unique assignment in the theorem. Dead-unit and other
out-of-group cases are not universally covered. Function equality, floating-
point tolerance and byte-hash equality are distinct properties.

k/N is a uniform reference, not deserved reward or true contribution. Increased
share does not by itself establish unfairness, task damage or financial harm.
Local adaptations of published methods are described in BASELINES.txt; they
are not reproductions of the authors' complete systems. Undefined constant-
vector rank correlations remain null, not zero. Null and adverse outcomes
are retained, including favourable simple-cap comparisons.

EVIDENCE AND PRIVACY

evidence/index.json selects the applicable historical run separately for each
task. Source dependency hashes were checked against those runs while packaging.
The old full run and later targeted repairs are not described as a new full
50-task execution. Raw run summaries and unrelated archives are omitted.
The A24 snapshot of a concurrently written scorer-comparison report was not
produced by A24 and is deliberately excluded from its result evidence.

Six distributed copies redact local absolute-path metadata. No experiment
source or numerical result value is changed. environment_id fields retained
inside historical records identify the original, unredacted environment; they
are not claimed to hash the redacted public record. file-origins.json provides
both original and distributed file hashes, including redacted logs/JSON files.

To verify distributed bytes on macOS:

    shasum -a 256 -c SHA256SUMS.txt

On Linux use sha256sum -c SHA256SUMS.txt. This checks package integrity, not
research conclusions. Packaging validation is recorded separately from the
historical research evidence in evidence/package-validation.json.

ATTRIBUTION AND LICENSING

Bibliographic references and adaptation details are retained. No third-party
papers, datasets or upstream repository snapshots are bundled. Dependencies
and downloaded data retain their own terms. The package does not apply a new
software license on the authors' behalf: public access is not an automatic
open-source license or a grant of unrestricted reuse rights. A reuse license
requires a separate decision by the rights holders.
