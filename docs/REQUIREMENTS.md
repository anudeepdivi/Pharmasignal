# Requirements and milestone acceptance

Priority: correctness > understandability > reproducibility > production realism > complexity.

## Runtime and dependencies

Target Python 3.12; `pyproject.toml` currently allows >=3.12,<3.14. Python 3.14.7 was observed on this workstation; no broad claim is made that every ML library lacks 3.14 support. A conservative common runtime avoids unnecessary compatibility work.

`pyproject.toml` is the dependency declaration. Its version ranges are proposed, not verified compatible or security-audited. No `uv.lock`, installed project environment or passing dependency import check exists at M0. Before M1 execution, resolve/install under Python 3.12, review the result, check imports, save `uv.lock`, and use frozen installs thereafter. Install only inside the project environment. Dependency upgrades require rerunning tests. Do not call a resolver success a vulnerability audit.

| Tool | Intended role | Status / rationale |
|---|---|---|
| NumPy / pandas | Seeded generation and tabular records | Declared |
| scikit-learn | TF-IDF, structured preprocessing, logistic regression, metrics | Declared |
| spaCy | Blank English tokenizer and explicit symptom/negation rules | Declared; no downloaded trained model required |
| SciPy | Sparse operations and transparent drift statistics | Declared |
| joblib | Serialize trusted fitted pipeline | Declared |
| MLflow | Local experiment metadata, metrics, artifacts | Declared; implementation at M4 |
| FastAPI / Pydantic / Uvicorn | Validated inference service | Declared; implementation at M5 |
| pytest / pytest-cov / httpx / Ruff | Tests, API test client, quality checks | Development group |
| Docker | Containerized local inference | Planned M5; availability/build not verified |
| PyTorch | None in linear baseline | Deliberately not included; do not claim usage |
| Evidently | Optional alternative to direct statistical reports | Not included in chosen baseline; do not claim usage |
| AWS | Deployment design | No account/resource/deployment verification or infrastructure creation |

No GPU, transformer, LLM, Kubernetes, distributed training, data warehouse, feature store or separate serving microservices.

## Milestone gates

Each milestone must update `INTERVIEW_NOTES.md`, document commands actually run and results, and STOP for human review. A future task in this table is not an implemented capability.

| Milestone | Deliverable | Acceptance evidence |
|---|---|---|
| M0 | Scaffold, architecture, dataset strategy/schema, requirements, README, interview notes | Files and syntax checks; no model/data claims |
| M1 | Generator, validators, deterministic group-disjoint split and manifest | Data validation/reproducibility/leakage tests; actual row and class counts |
| M2 | spaCy rules, TF-IDF, numeric/categorical preprocessing and fusion | Shape/type tests, negation cases, missing/unknown handling, proof learned transforms fit training only |
| M3 | Train, validation selection, frozen test evaluation, error analysis, artifact metadata | Real metrics with split labels, ablations, save/load agreement; serialization test |
| M4 | MLflow experiment tracking | Real run ID with configuration/data identity, parameters, metrics and retrievable model artifact; offline model use remains possible |
| M5 | API, versioned outputs, Dockerfile, reproducible dependency export | Endpoint/validation/startup-failure tests; actual image build and HTTP smoke test or explicit blocked status |
| M6 | Training reference and simulated production batch drift report | Unchanged/shifted distributions and constant/missing/unseen-category tests; no inferred performance claim |
| M7 | AWS, canary, rollback and monitoring design | Implemented vs designed separation; no resources without separate explicit approval |
| M8 | Final reproducibility/security/claims audit | `FINAL_PROJECT_AUDIT.md` with actual commands, metrics, supported claims and unresolved issues |

## Safety and operational requirements

- Synthetic only, no PHI, no real drug safety conclusions; do not use for clinical decisions.
- Version schema, model, preprocessing and threshold together. Return model/version with prediction and score.
- Reject malformed inputs predictably. Fail startup clearly on missing/incompatible model artifacts. Expose readiness without leaking paths or secrets.
- Load only trusted model artifacts. Never accept user-supplied pickle/joblib files.
- Local API binds loopback by default; external deployment requires authentication, TLS, network controls, rate/body limits and reviewed logging.
- No raw narrative logging by default, hard-coded secrets, public uploads, billable infrastructure or unrequested commits/pushes.
- No fabricated metrics, deployment history, clinical validation or use of technologies absent from implementation.
- Reproducibility means source/configuration/data/dependency identity and repeatable commands, not merely a random seed. Bitwise equality across operating systems or library versions is not promised.

## Resume claim reconciliation

The resume predates this implementation. New work cannot substantiate past deployment history. Today only project design exists. After later milestones pass, describe precisely what was built and tested, using synthetic-data and local-only qualifiers. Revise the technology list rather than importing unused tools to match it. Containerization is not AWS deployment; a diagram is not a canary rollout; feature drift is not proof of model failure.
