"""Offline, reproducible benchmark; never pays fees or contacts a blockchain.

Synthetic scores measure controlled topic separation, not Apex hidden-ground-truth
performance. Optional 20 Newsgroups input must already be downloaded locally.
"""
import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import platform
import random
import threading
import time
from pathlib import Path

TOPICS = [
    'bitcoin blockchain wallet mining transaction crypto ledger consensus',
    'soccer football stadium league player goalkeeper goal tournament',
    'galaxy astronomy telescope planet orbit star cosmic universe',
    'recipe kitchen cooking ingredient garlic pasta soup restaurant',
    'guitar music melody concert singer album rhythm musician',
    'software python compiler programming debugger code function developer',
    'medicine hospital doctor patient clinical treatment disease healthcare',
    'election parliament government voter campaign senator legislation policy',
    'battery electric vehicle automobile charging motor driving dealership',
    'ocean marine coral fish reef underwater coastal conservation',
    'mathematics theorem algebra topology proof equation geometry manifold',
    'quantum photon electron particle physics entanglement atomic experiment',
]
FILLER = 'today discuss interesting research latest report community people question details update analysis'.split()


def synthetic(seed, size, n_topics, style):
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(size):
        label = rng.randrange(n_topics)
        words = rng.sample(TOPICS[label].split(), 4 if style == 'titles' else 6)
        words += rng.sample(FILLER, 2 if style == 'titles' else 5)
        # Shared vocabulary and unrelated words make the easy fixture less trivial.
        if rng.random() < 0.25:
            words += rng.sample(TOPICS[(label + 1) % n_topics].split(), 2)
        rng.shuffle(words)
        text = ' '.join(words)
        if style == 'social':
            text = f'@person{i % 19} {text} #discussion https://example.test/{i}'
        texts.append(text)
        labels.append(label)
    return texts, labels


def _start_peak_rss_reader():
    """Return a peak-RSS reader and method label on Linux, macOS, and Windows."""
    try:
        import resource
    except ImportError:
        # Windows has no resource module. Sample RSS periodically; very brief
        # memory spikes may be missed, so the result is an estimate.
        import psutil

        process = psutil.Process()
        stop = threading.Event()
        peak = [process.memory_info().rss]

        def sample():
            while not stop.wait(0.05):
                try:
                    peak[0] = max(peak[0], process.memory_info().rss)
                except psutil.Error:
                    break

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()

        def read_peak_mib():
            stop.set()
            sampler.join()
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
            except psutil.Error:
                pass
            return round(peak[0] / (1024 * 1024), 1)

        return read_peak_mib, 'psutil_sampled_50ms'

    def read_peak_mib():
        # Linux reports KiB; macOS reports bytes.
        divisor = 1024 * 1024 if platform.system() == 'Darwin' else 1024
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor, 1)

    return read_peak_mib, 'resource_maxrss'


def worker(method, texts, truth, queue):
    # Each method gets a fresh process, so RSS and timing are comparable.
    read_peak_mib, rss_method = _start_peak_rss_reader()
    import numpy as np
    import sklearn
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from threadpoolctl import threadpool_limits
    module = importlib.import_module(method)
    before = time.process_time()
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=2):
            predicted = module.cluster_texts(texts)
        if len(predicted) != len(texts):
            raise ValueError('output length mismatch')
        ari = adjusted_rand_score(truth, predicted)
        nmi = normalized_mutual_info_score(truth, predicted)
        queue.put({
            'ok': True, 'seconds': round(time.perf_counter() - started, 4),
            'cpu_seconds': round(time.process_time() - before, 4),
            'peak_rss_mib': read_peak_mib(), 'rss_method': rss_method,
            'ari': round(float(ari), 5), 'nmi': round(float(nmi), 5),
            'combined': round((max(0, float(ari)) + float(nmi)) / 2, 5),
            'clusters': len(set(predicted) - {-1}),
            'noise_fraction': round(float(np.mean(np.array(predicted) == -1)), 4),
            'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__},
        })
    except Exception as error:
        queue.put({'ok': False, 'error': f'{type(error).__name__}: {error}'})


def evaluate(method, texts, truth, timeout):
    context = mp.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=worker, args=(method, texts, truth, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return {'ok': False, 'error': f'timeout after {timeout}s'}
    try:
        return queue.get(timeout=2)
    except Exception:
        return {'ok': False, 'error': f'worker exited {process.exitcode} without result'}
    finally:
        queue.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['smoke', 'full', 'stress', 'newsgroups'], default='full')
    parser.add_argument('--data-home', default='datasets')
    parser.add_argument('--samples', type=int, default=1500)
    parser.add_argument('--seeds', default='101,202,303')
    parser.add_argument('--output', default='results/benchmark.json')
    args = parser.parse_args()
    cases = []
    if args.suite == 'newsgroups':
        from sklearn.datasets import fetch_20newsgroups
        import numpy as np
        corpus = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'),
                                   data_home=args.data_home, download_if_missing=False)
        for seed in map(int, args.seeds.split(',')):
            rng = np.random.default_rng(seed)
            ids = rng.choice(len(corpus.data), args.samples, replace=False)
            cases.append((f'20newsgroups-{seed}', [corpus.data[i] for i in ids], corpus.target[ids].tolist()))
    else:
        sizes = [5000] if args.suite == 'stress' else ([240] if args.suite == 'smoke' else [300, 1200])
        seeds = (101,) if args.suite in ('smoke', 'stress') else (101, 202, 303)
        for size in sizes:
            for seed in seeds:
                for style in ('social', 'titles'):
                    texts, truth = synthetic(seed, size, 8 if style == 'social' else 12, style)
                    cases.append((f'synthetic-{style}-{size}-{seed}', texts, truth))
    report = {
        'suite': args.suite, 'platform': platform.platform(),
        'solution_sha256': hashlib.sha256(Path('solution.py').read_bytes()).hexdigest(),
        'warning': 'Synthetic/public-category labels are NOT Apex embedding-based hidden ground truth. No live reward claim.',
        'cases': [],
    }
    for name, texts, truth in cases:
        result = {'case': name, 'texts': len(texts)}
        for method in ('baseline_upstream', 'solution'):
            result[method] = evaluate(method, texts, truth, 90)
        report['cases'].append(result)
        print(json.dumps(result), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    failed = [r for r in report['cases'] if not r['solution']['ok']]
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
