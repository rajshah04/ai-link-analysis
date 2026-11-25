"""Call a RUNNING instance of the API, save real evidence for the README/submission.

    python -m scripts.capture_samples --urls https://site1.com https://site2.com https://site3.com
    python -m scripts.capture_samples --base http://localhost:4000 --urls ... [--api-key KEY]

Writes (into docs/):
  sample_response_<host>.json   full real responses
  error_cases.txt               request/response for invalid, SSRF-blocked, unresolvable URLs
  measured_rows.md              Markdown table rows to paste into the README "Measured results"
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ERROR_CASES = [
    ("Invalid URL", "not a url"),
    ("Private IP (SSRF)", "http://127.0.0.1/admin"),
    ("Cloud metadata IP (SSRF)", "http://169.254.169.254/latest/meta-data"),
    ("Localhost name (SSRF)", "http://localhost:4000/docs"),
    ("Non-existent domain", "https://this-domain-does-not-exist-12345.com"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:4000")
    ap.add_argument("--urls", nargs="*", default=[])
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args()
    headers = {"X-API-Key": a.api_key} if a.api_key else {}
    docs = Path("docs"); docs.mkdir(exist_ok=True)
    endpoint = a.base.rstrip("/") + "/link-analysis"
    rows = []

    with httpx.Client(timeout=180) as c:
        for url in a.urls:
            print(f"-> {url} ...", flush=True)
            t0 = time.perf_counter()
            r = c.post(endpoint, json={"url": url}, headers=headers)
            wall = int((time.perf_counter() - t0) * 1000)
            host = re.sub(r"[^a-z0-9]+", "_", (urlsplit(url).hostname or "site").lower()).strip("_")
            (docs / f"sample_response_{host}.json").write_text(json.dumps(r.json(), indent=2, ensure_ascii=False), encoding="utf-8")
            if r.status_code == 200:
                m = r.json()["meta"]
                q = r.json()["analysis"]["target_search_queries"]
                plats = {p for x in q for p in x["platforms"]}
                rows.append(f"| {url} | {m['latency_ms']} | {m['input_tokens']} | {m['output_tokens']} | {m['estimated_cost_usd']} | "
                            f"{'no (repair/fallback used)' if m['used_fallback_model'] else 'TODO'} | {len(plats)}/5 platforms covered; TODO manual notes |")
                print(f"   OK  model={m['model']} latency={m['latency_ms']}ms tokens={m['input_tokens']}/{m['output_tokens']} cached={m['cached']}")
            else:
                rows.append(f"| {url} | FAILED HTTP {r.status_code} | | | | | {r.json().get('error', {}).get('code')} |")
                print(f"   FAILED {r.status_code}: {r.text[:200]}")

        lines = []
        for name, url in ERROR_CASES:
            r = c.post(endpoint, json={"url": url}, headers=headers)
            lines.append(f"### {name}\nPOST {endpoint}\n{json.dumps({'url': url})}\n-> HTTP {r.status_code}\n{r.text}\n")
        (docs / "error_cases.txt").write_text("\n".join(lines), encoding="utf-8")
        print("Saved docs/error_cases.txt")

    if rows:
        (docs / "measured_rows.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print("\nPaste into README 'Measured results' table:\n" + "\n".join(rows))
        print("\nNote: 'Valid JSON first try' = TODO; if meta.used_fallback_model is false and the call succeeded, it passed validation (possibly after one internal repair retry).")


if __name__ == "__main__":
    main()