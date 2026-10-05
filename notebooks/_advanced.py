"""Save only measured advanced-notebook evidence, with an explicit run state."""
from datetime import datetime, timezone
import sys

from app.evaluation import save_result


class Evidence:
    def __init__(self, filename, **metadata):
        self.filename = filename
        self.data = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.executable, "status": "running", **metadata,
        }
        # A rerun that fails halfway must not leave a previous PASS report active.
        self.path = save_result(filename, self.data)

    def finish(self, *, checks, **measurements):
        checks = {key: bool(value) for key, value in checks.items()}
        self.data.update(measurements)
        self.data.update(
            checks=checks,
            status="passed" if all(checks.values()) else "needs_review",
        )
        self.path = save_result(self.filename, self.data)
        for name, passed in checks.items():
            print(f"{'PASS' if passed else 'REVIEW'}: {name}")
        print(f"Saved: {self.path} (status={self.data['status']})")
