"""Daily keepalive for the Xueqiu cookie (Lambda target of an EventBridge rule)."""
from __future__ import annotations

import boto3
import os

SSM_COOKIE = "/tactis/xueqiu_cookie"
HEALTH_URL = "https://xueqiu.com/cubes/rebalancing/current.json"


def handler(event, context):
    """Authenticate a tiny read against Xueqiu to refresh/keep session alive.

    Never logs the cookie. Raises loudly on rejection so CloudWatch alarms see it.
    """
    import requests

    client = boto3.client("ssm")
    cookie = client.get_parameter(Name=SSM_COOKIE, WithDecryption=True)["Parameter"]["Value"]
    headers = {
        "Cookie": cookie.lstrip("Cookie: "),
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126",
        "Referer": "https://xueqiu.com/",
        "X-Requested-With": "XMLHttpRequest",
    }
    resp = requests.get(HEALTH_URL, headers=headers, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"xueqiu cookie rejected: HTTP {resp.status_code}")
    return {"statusCode": 200, "body": "keepalive ok"}
