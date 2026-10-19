"""Sign in to Modal from a phone while a GitHub Actions run waits, so no token is ever stored.

Starts Modal's own browser sign-in (the flow `modal token new` uses), shows its link as a notice on the
Actions run page, waits for you to approve it with whichever Modal account should run the renders, then
hands the short-lived token to the following steps through $GITHUB_ENV, masked in the logs. The token
expires by itself (`--expires-in`, default 2 hours), so switching accounts is just running it again.

    python scripts/modal_login.py --wait 900 --expires-in 7200
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time


def notice(title: str, message: str) -> None:
    # Annotations show at the top of the run page on the phone while the job is still running.
    print(f"::notice title={title}::{message}", flush=True)


async def login(wait: float, expires_in: int) -> tuple[str, str, str]:
    from modal.client import _Client
    from modal.config import config
    from modal.token_flow import _TokenFlow

    from modal.exception import PermissionDeniedError

    deadline = time.time() + wait
    async with _Client.anonymous(config.get("server_url")) as client:
        while time.time() < deadline:
            flow = _TokenFlow(client)
            async with flow.start("namma-github-actions", None, expires_in) as (_, web_url, code):
                notice("Sign in to Modal", f"Open {web_url} on your phone and approve" + (f" (code {code})" if code else ""))
                print(f"\nSign in to Modal here:\n\n    {web_url}\n", flush=True)
                if code:
                    print(f"Code: {code}\n", flush=True)
                try:
                    while time.time() < deadline:
                        result = await flow.finish(timeout=min(40.0, max(1.0, deadline - time.time())))
                        if result is not None:
                            return result.token_id, result.token_secret, result.workspace_username
                        print("Still waiting for the sign-in…", flush=True)
                except PermissionDeniedError:  # Modal ends a link after about 15 minutes: start a fresh one
                    print("That sign-in link expired; here is a new one (open the newest link).", flush=True)
    raise TimeoutError(f"Nobody approved the Modal sign-in within {wait:.0f} s; run the workflow again.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wait", type=float, default=900, help="seconds to wait for the sign-in")
    parser.add_argument("--expires-in", type=int, default=7200, help="token lifetime in seconds")
    args = parser.parse_args()
    try:
        token_id, token_secret, workspace = asyncio.run(login(args.wait, args.expires_in))
    except TimeoutError as exc:
        print(exc)
        return 1
    print(f"::add-mask::{token_secret}", flush=True)
    print(f"::add-mask::{token_id}", flush=True)
    env = os.environ.get("GITHUB_ENV")
    if env:
        with open(env, "a") as handle:
            handle.write(f"MODAL_TOKEN_ID={token_id}\nMODAL_TOKEN_SECRET={token_secret}\nMODAL_WORKSPACE={workspace}\n")
    notice("Signed in to Modal", f"Deploying to the {workspace} workspace")
    return 0


if __name__ == "__main__":
    sys.exit(main())
