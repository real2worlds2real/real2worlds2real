"""Robust file download: parallel HTTP range requests with retries (slow / shared links)."""
import concurrent.futures as cf
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "calibrated_scene_reconstruction-datagen/0.2"}


def _open(url, headers=None, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers={**UA, **(headers or {})}), timeout=timeout)


def _retry(fn, retries=8):
    for k in range(retries):
        try:
            return fn()
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(min(2 ** k, 30))


def _size(url):
    """Total size via a 1-byte range request; 0 when the server ignores ranges."""
    def get():
        with _open(url, {"Range": "bytes=0-0"}) as r:
            cr = r.headers.get("Content-Range")
            return int(cr.split("/")[-1]) if cr else 0
    return _retry(get)


def _chunk(url, a, b, retries=8):
    for k in range(retries):
        try:
            with _open(url, {"Range": f"bytes={a}-{b}"}, timeout=90) as r:
                data = r.read()
            if len(data) == b - a + 1:
                return data
        except Exception:
            pass
        time.sleep(min(2 ** k, 30))
    raise IOError(f"range {a}-{b} of {url} failed")


def fetch(url, dst, workers=16, chunk=2 << 20):
    """Download url -> dst (skipped when dst exists). Uses range requests when the server supports them."""
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    n = _size(url)
    if n <= chunk:
        def whole():
            with _open(url, timeout=300) as r:
                tmp.write_bytes(r.read())
        _retry(whole)
    else:
        spans = [(a, min(a + chunk, n) - 1) for a in range(0, n, chunk)]
        with open(tmp, "wb") as f, cf.ThreadPoolExecutor(workers) as ex:
            f.truncate(n)
            for (a, _), data in zip(spans, ex.map(lambda s: _chunk(url, *s), spans)):
                f.seek(a)
                f.write(data)
    tmp.rename(dst)
    return dst


def fetch_many(items, workers=8):
    """[(url, dst)] -> list of dst (None on failure); small files, one connection each."""
    def one(it):
        try:
            return fetch(*it, workers=4)
        except Exception as e:
            print("download failed", it[0], e)
            return None
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, items))
