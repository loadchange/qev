"""Qev command-line tools."""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Qev: Qwen3.5 typed decisions")
    sub = ap.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--model", required=True)
    serve.add_argument("--backend", default="auto", choices=["auto", "torch", "mlx"])
    serve.add_argument("--device")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8008)
    serve.add_argument("--no-warmup", action="store_true", help="Skip startup text/image inference warmup")
    serve.add_argument("--no-wired-memory", action="store_true", help="Disable automatic MLX GPU residency and free-cache management")
    journal = serve.add_mutually_exclusive_group()
    journal.add_argument("--request-log-dir", type=Path, default=Path("runs/request-logs"),
                         help="Save local API requests, images, responses and timing here (default: runs/request-logs)")
    journal.add_argument("--no-request-log", action="store_true", help="Disable local API request recording")
    predict = sub.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--request", required=True, help="Path to a JSON state/questions request")
    predict.add_argument("--backend", default="auto", choices=["auto", "torch", "mlx"])
    predict.add_argument("--device")
    from .snake_cli import add_arguments

    add_arguments(sub.add_parser("snake", help="Run real Qev decisions in a terminal Snake game"))
    args = ap.parse_args(argv)
    if args.command == "snake":
        from .snake_cli import run

        try:
            return run(args)
        except (ValueError, OSError, RuntimeError) as exc:
            ap.exit(1, f"qev snake: {exc}\n")
        except KeyboardInterrupt:
            return 130
    from .inference import Agent
    agent = Agent(args.model, backend=args.backend, device=args.device)
    if args.command == "serve":
        import uvicorn

        from .serve import create_app
        from .service_runtime import service_runtime

        request_log = None
        if not args.no_request_log:
            from . import __version__
            from .request_log import RequestLog

            try:
                request_log = RequestLog(args.request_log_dir, model_context={
                    "checkpoint": str(Path(args.model).resolve()), "backend": agent.backend,
                    "config": agent.config, "qev_version": __version__,
                })
            except (OSError, ValueError) as exc:
                ap.exit(1, f"qev serve: cannot initialize request log: {exc}\n")
            print(f"Qev service: local request logs: {args.request_log_dir.resolve()}", file=sys.stderr, flush=True)
        with service_runtime(agent, warmup=not args.no_warmup, wired_memory=not args.no_wired_memory) as service_agent:
            uvicorn.run(create_app(service_agent, request_log=request_log), host=args.host, port=args.port)
    else:
        req = json.loads(Path(args.request).read_text())
        print(json.dumps(agent.predict(req["state"], req["questions"], model=req.get("model")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
