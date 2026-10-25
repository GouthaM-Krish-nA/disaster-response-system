"""
load_test.py

Fires a burst of synthetic reports at the running API concurrently, so you can
demo the queue absorbing a spike (the whole point of the SQS-style design).

Usage:
    python load_test.py 200      # fire 200 reports as fast as possible
"""

import asyncio
import sys
import time

import httpx

from data_generator import generate_reports

API_URL = "http://localhost:8000/report"


async def send_one(client: httpx.AsyncClient, report: dict):
    try:
        resp = await client.post(API_URL, json=report, timeout=10)
        return resp.status_code
    except Exception as e:
        return f"error: {e}"


async def main(n: int):
    reports = generate_reports(n)
    async with httpx.AsyncClient() as client:
        start = time.time()
        results = await asyncio.gather(*(send_one(client, r) for r in reports))
        elapsed = time.time() - start

    ok = sum(1 for r in results if r == 200)
    print(f"Sent {n} reports in {elapsed:.2f}s ({n/elapsed:.1f} req/s)")
    print(f"Accepted: {ok}/{n}")
    if ok < n:
        failures = [r for r in results if r != 200]
        print(f"Failures (sample): {failures[:5]}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    asyncio.run(main(n))
