"""Qev command-line tools."""
import argparse
import json
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
        uvicorn.run(create_app(agent), host=args.host, port=args.port)
    else:
        req = json.loads(Path(args.request).read_text())
        print(json.dumps(agent.predict(req["state"], req["questions"], model=req.get("model")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
