"""Read owned process status and bounded logs in the persistent Colab kernel."""
import json
from pathlib import Path

logs = {"qev_smoke": "/content/qev-smoke.log", "qev_job": "/content/qev-train.log",
        "qev_validation": "/content/qev-native.log", "qev_service": "/content/qev-service.log"}
for name, log in logs.items():
    job = globals().get(name)
    if job is None:
        continue
    status = job.poll()
    print(name, "RETURN_CODE", status)
    if name == "qev_job" and status == 0:
        result = json.loads(Path("/content/qev/models/qev-0.8b/evaluation.json").read_text())
        print(json.dumps({"steps": result["optimizer_steps"], "seconds": result["training_seconds"],
                          "development": result["development"]["calibrated"]}))
        continue
    if name == "qev_validation" and status == 0:
        result = json.loads(Path("/content/qev/runs/native_validation.json").read_text())
        print(json.dumps({"passed": result["passed"], "base_unchanged": result["base_integrity"]["unchanged"],
                          "image": result["image_native"]["text"], "video": result["video_native"]["text"]}))
        continue
    if name == "qev_service" and status is not None:
        path = Path("/content/qev/runs/service_torch.json")
        if path.exists():
            result = json.loads(path.read_text())
            print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]),
                              "failed": {key: value for key, value in result["checks"].items() if not value["passed"]}}))
            continue
    lines = Path(log).read_text().splitlines()
    print("\n".join(lines[-8:])[-5000:])
