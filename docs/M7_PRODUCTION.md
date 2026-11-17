# M7 — Production architecture design (designed, not provisioned)

Status: **design milestone**. Everything in this document beyond §1 is a design; no AWS
resource, container build, router or deployment exists or was created. No credentials were
inspected, no AWS CLI was run, no costs were incurred, and no billable infrastructure is
proposed for creation by this document. The Docker image has since been built
and verified locally (2026-11-16); nothing has been pushed to a registry or deployed. A design is not
evidence of a safe rollout.

## 1. What exists today (implemented, with real identifiers)

| Component | Status | Evidence in repo |
|---|---|---|
| Versioned model bundle | Implemented | `artifacts/models/m4/`, version `baseline-a28dc00666102c16`, `metadata.json` with model/schema versions, dataset/config hashes, selected threshold 0.13818775588643972 |
| MLflow tracking (local) | Implemented | Run `39bcca5c03124296a602ec896d0418ee`, SQLite + filesystem artifacts, FINISHED, retrievable |
| Inference API | Implemented locally | FastAPI `/predict` returning `model_version` + `schema_version` + threshold-aware score; strict schema; trusted-artifact lifespan load |
| Container definition | Written, build **verified locally (2026-11-16)** | `Dockerfile` (Python 3.12.14, non-root UID 10001, HEALTHCHECK), `.dockerignore`, hash-pinned `docker/requirements.txt`; actual `docker build` failed: `/var/run/docker.sock` absent |
| Drift monitoring | Implemented locally | `docs/M6_DRIFT.md`; reports on simulated batches only |
| Registry pointers (stable/candidate) | **Design only** | Not implemented anywhere |
| AWS infrastructure | **Design only** | Nothing exists |
| Canary/rollback mechanics | **Design only** | Nothing routed, nothing deployed |
| Router (local canary) | **Not implemented** | Explicitly optional and unapproved; omitted |

## 2. Model versioning design

**Immutable versions.** Each trained model is a version directory — never mutated after
creation — containing exactly the M4 bundle (`pipeline.joblib`, `report.json`,
`metadata.json`) plus per-file SHA-256 checksums. The existing convention extends naturally
to storage: `s3://<artifacts-bucket>/models/<model_version>/` where `model_version` remains
`baseline-<12-hex>`, uniquely bound (as recorded in `metadata.json` today) to the dataset id,
configuration hash, source hashes and lockfile hash. S3 bucket versioning protects against
accidental overwrite; Object Lock (governance mode) is optional hardening for tamper
resistance. The M5 loader's exact-version compatibility check stays: a task refusing to
start on an incompatible artifact is the first rollback safety net.

**Promotion state as an auditable pointer, not a file edit.** Two small JSON pointer
objects — `stable` and `candidate` — live in a separate versioned prefix:

```json
{
  "version": "baseline-a28dc00666102c16",
  "promoted_at": "<utc timestamp>",
  "promoted_by": "<human or pipeline identity>",
  "gates_passed": ["offline_test", "schema_compatibility", "image_scan"],
  "drift_report_sha256": "<m6 report hash, informational>"
}
```

Promotion writes a new pointer; it never modifies a version directory. Rollback repoints
`stable` to the previous version — a single reversible write, with the full pointer history
retained by S3 versioning. This mirrors the project's existing manifest/pointer patterns
(dataset manifest, batch manifest, MLflow run) rather than introducing a new mechanism.

**Pre-registered promotion gates** (all decided before a candidate exists, mirroring M3/M6
discipline):

| Gate | Needs labels? | Automatable? | Notes |
|---|---|---|---|
| Offline test-set quality | Yes (fixed synthetic holdout) | Yes | Thresholds fixed in advance; the same guard as M3's one-test-pass rule |
| Schema/dependency compatibility | No | Yes | Already implemented: the M5 loader fails startup on mismatch |
| Image vulnerability scan (ECR scan + dependency audit) | No | Yes | Gate on critical findings |
| Feature drift | No | Report only | **Informational, never a gate** — M6's own caveat: feature drift proves neither concept drift nor performance degradation |
| Live quality comparison | Yes (delayed, adjudicated) | Partially | Only meaningful with matched scoring on identical requests and enough labeled canary traffic |

No automated "safe promotion" claim is made from feature drift or unlabeled traffic. The
canary's operational layer can run in minutes; the quality gate waits for labels.

## 3. AWS architecture (design only)

```text
client
  └─> Route 53 (optional) → ACM TLS cert → ALB (HTTPS listener only; HTTP redirected)
        └─> target group: stable | weighted forward: candidate (canary)
              └─> ECS Fargate service (private subnets; NAT gateway for pulls)
                    └─> task startup: download trusted model bundle from S3 (lifespan, fail-fast)
  ECR            immutable-tag image
  S3             versioned model artifacts + access logs
  CloudWatch     logs + operational metrics + alarms → SNS
  IAM            task role (read artifact prefix, put metrics) ≠ deploy role
```

- **Compute**: ECS Fargate, one service, task = 0.25 vCPU / 0.5 GB, container from ECR with
  immutable tags (digest-pinned task definition). Health checks reuse the M5 `/ready`
  semantics; unhealthy targets leave rotation.
- **Networking**: ALB in public subnets, tasks in private subnets; NAT gateway for image
  pulls and dependency downloads; security groups allow only ALB→task on the container
  port. HTTPS terminated at the ALB with an ACM certificate; the HTTP listener redirects.
- **Artifact supply**: task role scoped to `s3:GetObject` on one artifact prefix only; the
  M5 lifespan pattern (load trusted artifact at startup, fail fast, never accept runtime
  uploads) carries over unchanged — only the source moves from local disk to S3.
- **Encryption**: TLS in transit; SSE-S3 or KMS at rest; secrets (none today) would use
  Secrets Manager, never image environment variables.
- **Deliberate simplifications**: single region, single service, no custom domain design,
  autoscaling limited to target-tracking on ALB requests per target. Each is acceptable for
  a demonstration service and would be revisited with real traffic.
- **Simpler alternative, honestly compared**: one EC2 instance + systemd + Caddy/nginx TLS
  terminating the same container. Roughly a third of the fixed cost, far fewer moving
  parts, but you own patching, uptime and scale. Right call when traffic is light and
  operational attention exists; Fargate is right when the recurring cost buys back that
  work. This project does not provision either.

**Cost sketch (design aid, not a quote; public pricing pages, may be stale; no account was
consulted)**: NAT gateway ≈ $32/mo + data (the dominant fixed cost), ALB ≈ $17/mo + LCU,
Fargate task ≈ $9/mo, ECR/S3/CloudWatch negligible at this scale. Free-tier eligibility
varies. **Nothing is provisioned and nothing will be without separate explicit approval.**

## 4. Monitoring design (two layers, named honestly)

**Layer 1 — operational, real-time, no labels required.** ALB 5xx rate, target response
latency (p50/p95/p99), task CPU/memory, readiness-check failure count, error-log alarm.
CloudWatch alarms → SNS. These work on unlabeled traffic and are the first thing a canary
watches.

**Layer 2 — quality, requires labels.** The label here is the adjudicated review-priority
outcome — in production, delayed by definition. Canary quality comparison requires matched
scoring (same requests scored by both versions) and enough labeled canary volume; a 5%
fraction for a day is unlikely to distinguish versions statistically. That is stated as a
power limitation, not engineered around. The M6 drift monitor feeds this layer as an
**input-change investigation signal only** — it is never a quality verdict and never a
promotion gate.

Nothing in this project currently ships metrics anywhere: the design names the metric
names, alarm semantics and label requirements so the boundary between "can monitor now"
and "needs labels" is explicit.

## 5. Canary deployment design

Stable and candidate run as separate ECS services, each behind its own target group, under
one ALB listener using **weighted forwarding** (start 95/5). Alternative: CodeDeploy
blue/green with linear traffic shifting and `deployment_circuit_breaker` (auto-rollback on
CloudWatch alarm) — more automation, less control per step; the manual weighted design is
chosen here because every step maps to a reversible action a reviewer can audit.

Order of operations:

1. Pre-registered offline gates pass (test-set gate, compatibility, image scan).
2. Candidate image pushed to ECR (immutable tag), deployed to the candidate target group.
3. Readiness checks pass on the candidate service.
4. Shift 5% traffic. Layer-1 alarms watched immediately (minutes).
5. Layer-2 (labeled) comparison only after delayed labels accumulate; low power is assumed.
6. **Promote** = repoint `stable` pointer to candidate version + shift weights 100/0.
   **Rollback** = weights back to stable (+ pointer repoint if promotion had occurred).
   Both are reversible pointer/weight writes; nothing destructive. The stateless container
   means no schema/data rollback is involved.
7. Rollback drills are **untested here** — a design with no rehearsed rollback is not a
   proven rollback.

What the design does **not** wave away: the M5 API has no authentication — production
requires it (at the ALB, e.g. OIDC/mTLS, or in front of it); rate limiting and body-size
limits are likewise unimplemented; the no-narrative-logging pattern (M5) must be preserved
in the container logging config; retention policies for logs and artifacts must be set
explicitly, not defaulted.

## 6. If implemented, what would change in this repo (pointer list, not code)

- `Dockerfile` build verified locally (2026-11-16); still to do: digest pushed to ECR.
- Task definition JSON (task role, health check, immutable image tag).
- ALB with two target groups + weighted forward rule; ACM cert; private subnets + NAT.
- Lifespan model loading changes from local path to S3 download with checksum verification
  (the `PHARMASIGNAL_MODEL_DIR` contract stays).
- CloudWatch log/metric configuration; SNS alarm topics.
- IaC choice to make then: Terraform or CDK (reviewable, repeatable) vs console (not
  repeatable) — none written, because no account exists.

All of the above are **not done**. The honest reasons: no Docker daemon, no AWS account
access, and no explicit approval to create billable resources.

## 7. Implemented-vs-designed audit

| Claim | Evidence | Status |
|---|---|---|
| Model version strings end-to-end (artifact, API response, docs) | `metadata.json`, `/predict` response | Implemented |
| Local MLflow tracking | Run `39bcca5c…`, SQLite store | Implemented (local) |
| Serving API with versioned responses | 200 tests incl. live HTTP | Implemented (local) |
| Drift reports on simulated batches | `docs/M6_DRIFT.md`, recorded reports | Implemented (local, simulated) |
| Container definition | `Dockerfile` + pinned deps | Written; **build verified locally** |
| Versioned artifact registry + promotion pointers | `docs/M7_PRODUCTION.md` §2 | **Design only** |
| AWS architecture (ALB/Fargate/ECR/S3/CloudWatch/IAM) | `docs/M7_PRODUCTION.md` §3 | **Design only** |
| Canary rollout + rollback mechanics | `docs/M7_PRODUCTION.md` §5 | **Design only** |
| Operational monitoring | §4 Layer 1 | **Design only** |
| Quality monitoring with delayed labels | §4 Layer 2 | **Design only** |
| Authentication, rate limits, body limits | §5 closing paragraph | **Design only; explicitly missing** |

## 8. Commands

Nothing to run for M7 beyond the existing verification — this milestone is a document:

```bash
# Full suite must remain green with no code changed (M6 recorded result):
.venv/bin/python -m pytest -W error --cov=pharmasignal --cov-report=term-missing
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
```

There is no AWS command to run, by design. No deployment command exists in this repo, and
that absence is the honest M7 result.
