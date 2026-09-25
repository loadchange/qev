"""Refresh Colab runtime-proxy tokens before they expire (colab CLI <= 0.7.2 never does).

The proxy token issued with an assignment expires after one hour; the CLI then
gets 404s from the runtime, concludes the kernel is gone and prunes the
session, orphaning a VM that is still running. The assignments listing returns
a fresh token for every live assignment, so this rewrites the local session
entries with it. ``--attach NAME=ENDPOINT:KERNEL_ID`` re-registers an orphan
(whose kernel, globals and background jobs are still alive) and restarts its
keep-alive daemon.

Run with the colab CLI's own interpreter, e.g.::

    ~/.local/share/uv/tools/google-colab-cli/bin/python experiments/diffusion/colab_refresh.py
"""
import argparse
import datetime
import json
import shutil

from colab_cli.client import Client, Prod
from colab_cli.commands.session import spawn_keep_alive
from colab_cli.common import State
from colab_cli.state import SessionState


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attach", action="append", default=[], help="NAME=ENDPOINT:KERNEL_ID")
    args = ap.parse_args(argv)
    state = State()
    shutil.copy(state.store.path, f"{state.store.path}.bak")
    from colab_cli.auth import get_credentials

    client = Client(Prod(), get_credentials(state.client_oauth_config, provider=state.auth_provider))
    assignments = {a.endpoint: a for a in client.list_assignments()}
    now = datetime.datetime.now().astimezone()
    for name, session in state.store.list().items():
        live = assignments.get(session.endpoint)
        if live is None:
            print(json.dumps({"session": name, "status": "no live assignment"}))
            continue
        session.token, session.url = live.runtime_proxy_info.token, live.runtime_proxy_info.url
        state.store.add(session)
        expires = now + datetime.timedelta(seconds=live.runtime_proxy_info.token_expires_in_seconds)
        print(json.dumps({"session": name, "status": "refreshed", "expires": expires.isoformat(timespec="seconds")}))
    for spec in args.attach:
        name, rest = spec.split("=", 1)
        endpoint, kernel_id = rest.split(":", 1)
        live = assignments[endpoint]
        session = SessionState(name=name, token=live.runtime_proxy_info.token, url=live.runtime_proxy_info.url,
                               endpoint=endpoint, variant=live.variant.name, accelerator=getattr(live.accelerator, "value", str(live.accelerator)),
                               kernel_id=kernel_id)
        state.store.add(session)
        session.keep_alive_pid = spawn_keep_alive(endpoint, name, auth_provider=state.auth_provider,
                                                  config_path=state.config_path)
        state.store.add(session)
        print(json.dumps({"session": name, "status": "attached", "endpoint": endpoint,
                          "keep_alive_pid": session.keep_alive_pid}))


if __name__ == "__main__":
    main()
