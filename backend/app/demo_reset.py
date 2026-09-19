"""CLI: wipe the live demo database without planting scores.

Usage, from backend/:
    .venv\\Scripts\\python.exe -m app.demo_reset
"""

from __future__ import annotations

from app.behavioral.ml.store import delete_all_models
from app.db.database import get_connection, init_db
from app.db import repository


def main() -> None:
    init_db()
    with get_connection() as conn:
        deleted = repository.reset_operational_state(conn)
    # Trained models live on disk, not in the database, so wiping tables alone
    # would leave a previous user's model behind for the next enrollment.
    models = delete_all_models()
    print(f"Demo database reset. Trained models removed: {models}")
    print("Rows removed:")
    for table, count in deleted.items():
        print(f"  {table}: {count}")
    print("No scores were inserted. Re-enroll with real typing before the next demo.")


if __name__ == "__main__":
    main()
