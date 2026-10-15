import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import solution


@pytest.fixture
def client():
    return TestClient(solution.app)


def test_health_and_contract(client):
    assert client.get('/health').json() == {'status': 'healthy'}
    texts = ['Bitcoin uses hash functions', 'A galaxy contains stars', 'Bitcoin uses hash functions']
    response = client.post('/cluster', json={'texts': texts})
    assert response.status_code == 200
    labels = response.json()['cluster_ids']
    assert len(labels) == len(texts)
    assert all(type(label) is int for label in labels)
    assert labels[0] == labels[2]


@pytest.mark.parametrize('texts', [[], [None], [1], 'not a list'])
def test_invalid_payload(client, texts):
    assert client.post('/cluster', json={'texts': texts}).status_code == 422


@pytest.mark.parametrize('texts', [
    ['the and of', 'and the of'], ['a', 'b'],
    ['!!!', '   ', 'https://example.org'], ['Монгол хэл', 'Өнөөдөр сайхан өдөр'],
    ['identical words'] * 100, ['\x00', '😀', '日本語テキスト'],
])
def test_degenerate_batches(texts):
    labels = solution.cluster_texts(texts)
    assert len(labels) == len(texts)
    assert all(type(x) is int for x in labels)


def test_noise_and_repeated_calls():
    texts = ['!!!', 'shared text', 'shared text', 'different subject']
    first = solution.cluster_texts(texts)
    assert first[0] == -1 and first[1] == first[2]
    assert solution.cluster_texts(texts) == first


def test_character_budget(client, monkeypatch):
    monkeypatch.setattr(solution, 'MAX_CHARACTERS', 10)
    assert client.post('/cluster', json={'texts': ['01234567890']}).status_code == 413


def test_busy_worker(client):
    solution._LOCK.acquire()
    try:
        assert client.post('/cluster', json={'texts': ['test']}).status_code == 503
        assert client.get('/health').status_code == 200
    finally:
        solution._LOCK.release()


def test_large_batch_projection(monkeypatch):
    monkeypatch.setattr(solution, 'FIT_LIMIT', 40)
    texts = [f'astronomy galaxy telescope observation {i}' for i in range(40)]
    texts += [f'soccer stadium football player {i}' for i in range(40)]
    labels = solution.cluster_texts(texts)
    assert len(labels) == 80
    assert len(set(labels)) > 1


def test_submission_size_and_no_dynamic_execution():
    path = Path(solution.__file__)
    assert len(path.read_text()) <= 50_000
    tree = ast.parse(path.read_text())
    forbidden = {'eval', 'exec', '__import__', 'compile'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden


def test_topic_separation_fixture():
    from benchmark import synthetic
    from sklearn.metrics import adjusted_rand_score
    texts, truth = synthetic(seed=991, size=320, n_topics=8, style='social')
    assert adjusted_rand_score(truth, solution.cluster_texts(texts)) > 0.9


def test_clustering_needs_no_network(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('network connection attempted')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    assert len(solution.cluster_texts(['galaxy telescope star', 'football match stadium'])) == 2
