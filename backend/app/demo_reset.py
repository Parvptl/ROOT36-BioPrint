"""CLI: wipe the live demo database without planting scores.

Usage, from backend/:
    .venv\\Scripts\\python.exe -m app.demo_reset
"""

from __future__ import annotations

from app.db.database import get_connection, init_db
from app.db import repository


def main() -> None:
    init_db()
    with get_connection() as conn:
        deleted = repository.reset_operational_state(conn)
    print("Demo database reset. Rows removed:")
    for table, count in deleted.items():
        print(f"  {table}: {count}")
    print("No scores were inserted. Re-enroll with real typing before the next demo.")


if __name__ == "__main__":
    main()
