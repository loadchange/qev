"""Qev command-line tools.

Heavy runtimes are imported only by the commands that load a model, so
`qev decide` against a warm `qev serve` starts in a fraction of a second.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

EXAMPLES = """examples:
  qev pull                                   download and verify the default model
  qev decide "Card declined twice at checkout" -i "Which team?" --choice billing technical account
  qev decide --image photo.jpg -i "Is there a dog?" --noul -q
  cat ticket.txt | qev decide - -i "How urgent?" --score low medium high --json
  qev decide --request request.json          full /v1/systemone request, several questions
  qev serve                                  HTTP API and web lab on http://127.0.0.1:8008
  brew services start qev                    keep the model warm in the background

`qev decide` and `qev chat` use a running server when one answers on
$QEV_SERVER (default http://127.0.0.1:8008); otherwise they load the model.
"""


def _stderr(message):
    print(message, file=sys.stderr, flush=True)


WEIGHTS_HELP = ("MLX decision weights: adapter (float32 + separate LoRA, validated), merged (float32), "
                "merged-bf16, or bf16 (whole model in bfloat16: least memory, fastest). "
                "Default: $QEV_DECISION_WEIGHTS or adapter")


def _add_model(parser, help_text="Registry name (default: $QEV_MODEL or qev-450m) or checkpoint directory"):
    parser.add_argument("--model", help=help_text)
    parser.add_argument("--no-pull", action="store_true", help="Fail instead of downloading a missing registry model")
    parser.add_argument("--decision-weights", choices=["adapter", "merged", "merged-bf16", "bf16"], help=WEIGHTS_HELP)


def _add_target(parser):
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--server", metavar="URL", help="Use this qev server (default: auto-detect $QEV_SERVER)")
    target.add_argument("--local", action="store_true", help="Load the model in this process; never use a server")
    parser.add_argument("--timeout", type=float, default=300, help="Server request timeout in seconds")


def build_parser():
    from . import __version__
    from .models import DEFAULT_MODEL

    formatter = argparse.RawDescriptionHelpFormatter
    ap = argparse.ArgumentParser(prog="qev", description="Qev: typed decisions on a local Qwen3.5 multimodal model.",
                                 epilog=EXAMPLES, formatter_class=formatter)
    ap.add_argument("--version", action="version", version=f"qev {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    decide = sub.add_parser("decide", help="Ask for a choice, yes/no or score decision", epilog=EXAMPLES,
                            formatter_class=formatter)
    decide.add_argument("state", nargs="?", help="State text to judge; '-' reads standard input")
    decide.add_argument("--state-file", type=Path, help="Read the state text from a file")
    decide.add_argument("--state-json", type=Path, help="Read a JSON state (object or array) from a file")
    decide.add_argument("--image", action="append", default=[], metavar="PATH", help="Attach an image (repeatable)")
    decide.add_argument("--frame", action="append", default=[], metavar="PATH",
                        help="Attach one video frame; repeat in order")
    decide.add_argument("--fps", type=float, default=1.0, help="Frame rate of --frame images (default 1)")
    kind = decide.add_mutually_exclusive_group()
    kind.add_argument("--choice", nargs="+", metavar="KEY[=DESCRIPTION]", help="Choose one option")
    kind.add_argument("--noul", action="store_true", help="Yes/no question")
    kind.add_argument("--score", nargs="+", metavar="LEVEL", help="Ordered levels, lowest first")
    kind.add_argument("--request", metavar="PATH", help="Full /v1/systemone JSON request ('-' reads stdin)")
    decide.add_argument("-i", "--instructions", help="The question the model answers")
    decide.add_argument("--id", default="answer", help="Question id in the output (default: answer)")
    decide.add_argument("--true", dest="true_description", metavar="TEXT", help="Meaning of yes for --noul")
    decide.add_argument("--false", dest="false_description", metavar="TEXT", help="Meaning of no for --noul")
    output = decide.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="Print the complete API response")
    output.add_argument("-q", "--quiet", action="store_true", help="Print only the answer(s)")
    decide.add_argument("--exit-status", action="store_true",
                        help="With one yes/no question, exit 0 for yes and 1 for no")
    _add_model(decide)
    _add_target(decide)

    chat = sub.add_parser("chat", help="Native generation with the original multimodal foundation")
    chat.add_argument("prompt", nargs="?", help="Message text; '-' reads standard input")
    chat.add_argument("--image", action="append", default=[], metavar="PATH")
    chat.add_argument("--system", help="System message")
    chat.add_argument("--max-tokens", type=int, default=256)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.add_argument("--think", action="store_true", help="Enable the model's thinking mode")
    chat.add_argument("--json", action="store_true", help="Print the complete API response")
    _add_model(chat)
    _add_target(chat)

    serve = sub.add_parser("serve", help="HTTP API and web lab")
    _add_model(serve)
    serve.add_argument("--backend", default="auto", choices=["auto", "torch", "mlx"])
    serve.add_argument("--device")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8008)
    serve.add_argument("--no-warmup", action="store_true", help="Skip startup text/image inference warmup")
    serve.add_argument("--no-wired-memory", action="store_true", help="Disable automatic MLX GPU residency and free-cache management")
    journal = serve.add_mutually_exclusive_group()
    journal.add_argument("--request-log-dir", type=Path,
                         help="Save API requests, images, responses and timing here (default: $QEV_HOME/request-logs)")
    journal.add_argument("--no-request-log", action="store_true", help="Disable local API request recording")

    predict = sub.add_parser("predict", help="Run one JSON request file and print the response")
    predict.add_argument("--model", required=True)
    predict.add_argument("--request", required=True, help="Path to a JSON state/questions request")
    predict.add_argument("--backend", default="auto", choices=["auto", "torch", "mlx"])
    predict.add_argument("--device")

    pull = sub.add_parser("pull", help="Download (or import) and verify a model")
    pull.add_argument("name", nargs="?", default=DEFAULT_MODEL)
    pull.add_argument("--from", dest="source", metavar="DIR", help="Import an existing download after verifying it")
    pull.add_argument("--link", action="store_true", help="With --from, symlink files instead of copying them")
    pull.add_argument("--force", action="store_true", help="Replace an installed copy")
    listing = sub.add_parser("list", help="Show registry models and what is installed")
    listing.add_argument("--json", action="store_true")
    remove = sub.add_parser("rm", help="Delete an installed model")
    remove.add_argument("name")
    doctor = sub.add_parser("doctor", help="Check this machine, the models and a local server")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--verify", action="store_true", help="Also hash every installed model file")
    doctor.add_argument("--server", metavar="URL")
    from .snake_cli import add_arguments

    add_arguments(sub.add_parser("snake", help="Run real Qev decisions in a terminal Snake game"))
    return ap


def resolve_model(spec, *, pull=True):
    """Registry names are resolved (and pulled if needed); paths pass through."""
    from .models import ModelNotInstalled, canonical, resolve
    from .models import pull as pull_model

    if spec is not None and canonical(spec) is None:
        return Path(spec).expanduser()
    try:
        return resolve(spec)
    except ModelNotInstalled as missing:
        if not pull:
            raise
        pull_model(missing.name, log=_stderr)
        return resolve(spec)


def load_agent(args):
    path = resolve_model(args.model, pull=not args.no_pull)
    if not (path / "qev_config.json").is_file():
        raise FileNotFoundError(f"No Qev checkpoint at {path} (qev_config.json is missing)")
    started = time.perf_counter()
    _stderr(f"Loading {path} …")
    from .inference import Agent

    agent = Agent(path, backend=getattr(args, "backend", "auto"), device=getattr(args, "device", None),
                  decision_weights=args.decision_weights)
    _stderr(f"Loaded on {agent.backend} in {time.perf_counter() - started:.1f} s")
    return agent


def pick_server(args):
    """URL of a usable running server, or None to run locally."""
    if args.local:
        return None
    from .client import ServerError, health, server_url, serves
    from .models import canonical, installed_path

    url = server_url(args.server)
    document = health(url, timeout=3 if args.server else 0.5)
    if document is None:
        if args.server:
            raise ServerError(f"No qev server answers at {url}")
        return None
    wanted = None
    if args.model:
        name = canonical(args.model)
        wanted = installed_path(name) if name else Path(args.model).expanduser()
    if serves(document, wanted):
        return url
    if args.server:
        raise ServerError(f"{url} serves {document.get('checkpoint')}, not {wanted}")
    return None


def _read_text(value, path):
    if path is not None:
        return path.read_text()
    if value == "-":
        return sys.stdin.read()
    return value or ""


def decide_request(args):
    from .api import SystemOneRequest
    from .decide import build_question, build_state

    notes = []
    if args.request:
        source = sys.stdin.read() if args.request == "-" else Path(args.request).read_text()
        request = json.loads(source)
        if args.state or args.state_file or args.state_json or args.image or args.frame:
            raise ValueError("--request already contains the state; drop the other state arguments")
    else:
        kind = "choice" if args.choice else "noul" if args.noul else "score" if args.score else None
        state_json = json.loads(args.state_json.read_text()) if args.state_json else None
        text = _read_text(args.state, args.state_file)
        request = {"state": build_state(text, images=args.image, frames=args.frame, fps=args.fps,
                                        state_json=state_json, notes=notes),
                   "questions": {args.id: build_question(kind, args.choice or args.score or (),
                                                         instructions=args.instructions,
                                                         true=args.true_description, false=args.false_description)}}
    SystemOneRequest.model_validate(request)
    for note in notes:
        _stderr(note)
    return request


def cmd_decide(args):
    from .decide import render_json, render_quiet, render_text

    request = decide_request(args)
    url = pick_server(args)
    started = time.perf_counter()
    if url:
        from .client import post

        response, via = post(url, "/v1/systemone", request, timeout=args.timeout), f"server {url}"
    else:
        agent = load_agent(args)
        started = time.perf_counter()
        response = agent.predict(request["state"], request["questions"], model=request.get("model"))
        via = "in-process"
    elapsed = (time.perf_counter() - started) * 1000
    if args.json:
        print(render_json(response))
    elif args.quiet:
        print(render_quiet(response))
    else:
        backend = response.get("qev", {}).get("backend", "?")
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        print(render_text(response, color=color, footer=f"{response.get('model')} · {backend} · {elapsed:.0f} ms · {via}"))
    answers = list(response["answers"].values())
    if args.exit_status:
        if len(answers) != 1 or answers[0]["type"] != "noul":
            raise ValueError("--exit-status needs exactly one --noul question")
        return 0 if answers[0]["noul"] >= 0.5 else 1
    return 0


def cmd_chat(args):
    from .decide import media_urls

    notes = []
    text = _read_text(args.prompt, None)
    content = [{"type": "text", "text": text}] if text else []
    content += [{"type": "image_url", "image_url": {"url": url}} for url in media_urls(args.image, notes=notes)]
    if not content:
        raise ValueError("Give a prompt or at least one --image")
    messages = ([{"role": "system", "content": args.system}] if args.system else []) + [
        {"role": "user", "content": content if args.image else text}]
    request = {"model": "qev-native", "messages": messages, "max_tokens": args.max_tokens,
               "temperature": args.temperature, "enable_thinking": args.think}
    for note in notes:
        _stderr(note)
    url = pick_server(args)
    if url:
        from .client import post

        response = post(url, "/v1/chat/completions", request, timeout=args.timeout)
    else:
        response = load_agent(args).chat_completions(request)
    print(json.dumps(response, ensure_ascii=False, indent=2) if args.json
          else response["choices"][0]["message"]["content"].rstrip("\n"))
    return 0


def cmd_serve(args):
    import uvicorn

    from .inference import Agent
    from .models import canonical, home
    from .serve import create_app
    from .service_runtime import service_runtime

    registry = args.model is None or canonical(args.model) is not None
    model = resolve_model(args.model, pull=not args.no_pull) if registry else Path(args.model)
    agent = Agent(model, backend=args.backend, device=args.device, decision_weights=args.decision_weights)
    request_log = None
    if not args.no_request_log:
        from . import __version__
        from .request_log import RequestLog

        directory = args.request_log_dir or home() / "request-logs"
        try:
            request_log = RequestLog(directory, model_context={
                "checkpoint": str(Path(model).resolve()), "backend": agent.backend,
                "config": agent.config, "qev_version": __version__,
            })
        except (OSError, ValueError) as exc:
            raise OSError(f"cannot initialize request log: {exc}") from exc
        _stderr(f"Qev service: local request logs: {Path(directory).resolve()}")
    with service_runtime(agent, warmup=not args.no_warmup, wired_memory=not args.no_wired_memory) as service_agent:
        _stderr(f"Qev service: http://{args.host}:{args.port} (web lab /, API /v1/systemone)")
        uvicorn.run(create_app(service_agent, request_log=request_log), host=args.host, port=args.port)
    return 0


def cmd_predict(args):
    from .inference import Agent

    agent = Agent(args.model, backend=args.backend, device=args.device)
    req = json.loads(Path(args.request).read_text())
    print(json.dumps(agent.predict(req["state"], req["questions"], model=req.get("model")), ensure_ascii=False, indent=2))
    return 0


def cmd_pull(args):
    from .models import pull

    print(pull(args.name, source=args.source, link=args.link, force=args.force, log=_stderr))
    return 0


def cmd_list(args):
    from .models import list_models

    rows = list_models()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'NAME':<12} {'SIZE':>10}  {'STATUS':<13} SOURCE")
    for row in rows:
        status = "installed" if row["installed"] else "not installed"
        name = row["name"] + (" *" if row["default"] else "")
        print(f"{name:<12} {row['size_bytes'] / 2**30:>6.2f} GiB  {status:<13} {row['source'] or row['repo_id']}")
    return 0


def cmd_rm(args):
    from .models import remove

    print(f"Removed {remove(args.name)}")
    return 0


def cmd_doctor(args):
    from .doctor import main as doctor

    return doctor(args)


def cmd_snake(args):
    from .snake_cli import run

    return run(args)


COMMANDS = {"decide": cmd_decide, "chat": cmd_chat, "serve": cmd_serve, "predict": cmd_predict,
            "pull": cmd_pull, "list": cmd_list, "rm": cmd_rm, "doctor": cmd_doctor, "snake": cmd_snake}


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    from .client import ServerError
    from .models import ModelNotInstalled

    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        return 130
    except ModelNotInstalled as exc:
        _stderr(f"qev {args.command}: {exc}")
        return 3
    except ServerError as exc:
        _stderr(f"qev {args.command}: {exc}")
        return 4
    except (ValueError, OSError, RuntimeError, NotImplementedError) as exc:
        if args.command == "snake":
            ap.exit(1, f"qev snake: {exc}\n")
        _stderr(f"qev {args.command}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
