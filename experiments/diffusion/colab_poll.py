"""Show queue progress, finished-arm summaries and the running arm's log tail."""
import json
from pathlib import Path

out_root = Path(globals().get("QEVD_OUT", "/content/qevd-runs"))
job = globals().get("qevd_job")
print("queue RETURN_CODE", None if job is None else job.poll())
queue = Path("/content/qevd-queue.log")
events = queue.read_text().splitlines() if queue.exists() else []
print("\n".join(events))
running = None
for line in events:
    parts = line.split()
    if parts[0] == "QEVD_START":
        running = parts[1]
    elif parts[0] == "QEVD_END" and parts[1] == running:
        running = None
for path in sorted(out_root.glob("*/evaluation.json")):
    result = json.loads(path.read_text())
    dev = result["development"]
    print(json.dumps({"arm": result["label"], "dev_acc": round(dev["calibrated"]["accuracy"], 4),
                      "snake_acc": round(dev["snake"]["accuracy"], 4), "general_acc": round(dev["general"]["accuracy"], 4),
                      "dev_nll": round(dev["calibrated"]["nll"], 4), "train_s": round(result["training_seconds"]),
                      "snake_food": result.get("snake_closed_loop", {}).get("food", {}).get("mean"),
                      "collisions": result.get("snake_closed_loop", {}).get("collisions")}))
if running:
    log = out_root / f"{running}.log"
    if log.exists():
        tail = [line for line in log.read_text().splitlines() if "Warning" not in line][-6:]
        print(f"--- {running} (running)\n" + "\n".join(line[:400] for line in tail))
