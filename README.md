# RAINY · Apex Text Cluster

A single-file, offline **research candidate** for Apex SN1 Text Clustering.
Монгол зааврыг доороос үзнэ үү.

**Implemented and locally tested. Not submitted to Apex, not a proven winner,
and not a passive cryptocurrency miner. No wallet or payment code is included.**

## What it does

`solution.py` accepts raw texts and returns integer topic cluster IDs. It uses:

1. URL/mention cleanup, Unicode-aware normalization, and exact-text deduplication.
2. Word/bigram and character TF-IDF features, followed by 96-dimensional LSA.
3. Average-linkage, spherical-space MiniBatchKMeans, and density candidates.
4. Unsupervised silhouette selection with noise and giant-cluster penalties.
5. Bounded fitting for large batches, then centroid assignment to remaining texts.

The candidate uses **no external model, runtime internet, labels, dynamic code
execution, subprocesses, wallet keys or blockchain calls**. Two numerical-library
threads are requested during computation; this is not a 5% CPU cap.

## Run locally

Use Python **3.12**. Native Windows is supported for local tests, benchmarks,
and the HTTP server. The Apex CLI workflow uses WSL2/Ubuntu.

### Native Windows (PowerShell)

```powershell
git clone https://github.com/ssboroo/Apex-Text-Cluster.git
cd Apex-Text-Cluster
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python benchmark.py --suite smoke --output results/smoke-windows.json
python solution.py --port 8001
```

If PowerShell blocks the activation script, run this in that PowerShell window,
then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The server stays open in that window. In a second PowerShell window, check it:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
$body = @{texts=@("bitcoin blockchain wallet", "football stadium match")} | ConvertTo-Json -Compress
Invoke-RestMethod -Uri http://127.0.0.1:8001/cluster -Method Post -ContentType "application/json" -Body $body
```

### WSL2 Ubuntu (Apex CLI workflow)

```bash
git clone https://github.com/ssboroo/Apex-Text-Cluster.git
cd Apex-Text-Cluster
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python benchmark.py --suite smoke --output results/smoke-wsl.json
```

```bash
curl http://localhost:8001/health
curl -X POST http://localhost:8001/cluster \
  -H 'Content-Type: application/json' \
  -d '{"texts":["bitcoin blockchain wallet","football stadium match","bitcoin wallet transaction"]}'
```

Response shape: `{"cluster_ids":[0,1,0]}`. This is an illustrative shape, not a
guaranteed assignment for those three texts. Labels are batch-local; `-1` is noise.

Docker (Docker execution was not available in the development environment):

```bash
docker build -t rainy-apex .
docker run --rm --cpus=2 --memory=1536m -p 8001:8001 rainy-apex
```

## Verify and reproduce

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python benchmark.py --suite smoke --output results/smoke.json
python benchmark.py --suite full --output results/synthetic-final.json
python benchmark.py --suite stress --output results/stress-final.json
```

Run the same commands in PowerShell after activating the Windows virtual
environment. CI runs the tests and smoke benchmark on both Ubuntu and Windows.
The benchmark compares the unmodified upstream starter and the candidate in
fresh spawned processes, both limited to two numerical-library threads. Scores
use `(max(0, ARI) + NMI) / 2`. Timing excludes process startup/imports and includes
clustering. Linux and macOS report the process high-water RSS; Windows samples
RSS every 50 ms using psutil, so very brief memory spikes may be missed. Results
include an `rss_method` field identifying the measurement method.

### Public corpus (optional download, never during a submission)

Download 20 Newsgroups explicitly on your own machine first:

```bash
python -c "from sklearn.datasets import fetch_20newsgroups; fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'), data_home='datasets')"
python benchmark.py --suite newsgroups --data-home datasets --output results/newsgroups-final.json
python benchmark.py --suite newsgroups --samples 5000 --seeds 404 \
  --data-home datasets --output results/newsgroups-5000.json
```

The benchmark itself sets `download_if_missing=False`. Corpus contents are not
included in this repository. Respect the dataset's original rights and terms.

## Measured results and limits

| Public corpus test | Upstream combined | Candidate combined | Candidate clustering time |
|---|---:|---:|---:|
| 1,500 texts, seed 101 | 0.14141 | 0.26167 | 6.59 s |
| 1,500 texts, seed 202 | 0.14131 | 0.26231 | 3.93 s |
| 1,500 texts, seed 303 | 0.15225 | 0.24243 | 3.97 s |
| 5,000 texts, seed 404 | 0.16707 | 0.27045 | 17.00 s |

The 5,000-text candidate process peaked at **539.1 MiB RSS** in this runtime.
These are individual local runs, not hardware-independent guarantees or rigorous
performance distributions. The first three splits guided development; seed 404
was run subsequently without another algorithm change. They are random subsets
of the same corpus, **not mutually disjoint held-out datasets**.

**Important:** 20 Newsgroups category labels differ from Apex's hidden
embedding-derived clusters. Its texts also differ from short social posts and
paper titles. Synthetic fixtures are intentionally controlled tests, not proof
of real-world semantic accuracy. The first candidate failed badly on the public
corpus; its raw result is retained in `results/experiments/v1-newsgroups.json`.

Current limitations:

- Lexical/LSA features miss paraphrases that share little vocabulary.
- Quality selection is a heuristic and can choose the wrong number of topics.
- Full hierarchical clustering is quadratic on at most 5,500 unique texts.
- Inputs above 5,500 unique texts use sampled fitting/centroid assignment.
  Their maximum-size resource behavior has not been validated.
- Up to 50,000 input strings / 12 million total characters are accepted. Each
  text is truncated to 6,000 characters **for clustering**. Total size validation
  occurs after JSON parsing; this server is not an internet-facing SaaS gateway.
- The 65-second budget is cooperative between candidate evaluations, not a hard
  interruption of SVD/linkage/HDBSCAN. Target sandbox CPU limits can change runtime.
- Docker, the official Apex screener, live leaderboards, hidden scoring and actual
  paid submission have **not** been verified here.
- A concurrent `/cluster` request receives 503; `/health` remains available.
  Invalid/empty input receives 422; excessive total text characters receive 413.

## Apex submission: prepare, then decide

The submission artifact is **only `solution.py`** (under 50,000 characters).
Dependencies match the upstream listed ML/API stack, with explicit transitive
pins for local reproducibility. Confirm the current live package allowlist and
screener before paying. CI never registers a wallet or submits anything.

According to the official documentation inspected on 2026-10-16:

- The Text Clustering competition documents a **$20 USD submission fee**.
- Mainnet hotkey registration has an additional dynamic TAO cost.
- Only the leading eligible solution receives the competition's emissions;
  a submission can earn **zero**. Apex rewards are subnet Alpha, not a fixed wage.
- The current competition page describes four subsets, while the checked-out
  upstream README still describes three. Verify live competition details.

No ROI, revenue or winning-score claim is made. Do not paste seed phrases or
private keys into issues, this repo, chat, configuration examples or CI secrets.

## Монгол заавар

Энэ нь компьютерийн хүчээр Bitcoin олборлогч биш. Apex-ийн текст бүлэглэх
даалгаварт алгоритмаар өрсөлдөх зориулалттай Python шийдэл юм.

1. Дээрх заавраар локал орчинд суулгана.
2. `python -m pytest -q` командаар тест хийнэ.
3. Benchmark-ийн оноо, хугацаа, санах ойн хэмжилтийг харна.
4. Тэмцээний одоогийн нөхцөл, тэргүүлэх оноо, бүртгэл болон илгээлтийн зардлыг
   шалгаж байж илгээх эсэхээ шийднэ.

**Одоогоор mainnet бүртгэл хийгээгүй, төлбөр төлөөгүй, тэмцээнд илгээгээгүй.**
Бодит тэмцээний нууц өгөгдөл дээр ялах эсэхийг локал тестээр батлах боломжгүй.

## Sources and attribution

- [Apex source](https://github.com/macrocosm-os/apex), inspected commit
  `da4a308ea7e097a1e357937e5a322a54bba9cc41`.
- `baseline_upstream.py` is copied unchanged from
  `shared/competition/src/competition/text_clustering/baseline.py` at that commit.
- [Current competition specification](https://docs.macrocosmos.ai/subnets/subnet-1-apex/subnet-1-current-competitions/text-clustering-competition).
- [Apex solver guide](https://github.com/macrocosm-os/apex/blob/main/SOLVERS.md).
- [20 Newsgroups loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html).

MIT license; upstream copyright and license are retained in `LICENSE`.
