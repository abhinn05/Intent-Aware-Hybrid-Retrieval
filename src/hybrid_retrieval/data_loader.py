import csv
from pathlib import Path

from .profiles import Profile, profile_from_row


def load_profiles_from_csv(csv_path: str | Path) -> list[Profile]:
    path = Path(csv_path)
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return [profile_from_row(row) for row in reader]
