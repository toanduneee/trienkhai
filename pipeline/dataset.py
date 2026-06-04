import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_behaviors(n: int = None, category: str = None) -> list[dict]:
    """
    Load harmful_behaviors.csv làm seed dataset.
    Cột: goal (seed prompt), target (expected harmful response)

    Args:
        n:        Giới hạn số lượng seed (None = lấy tất cả).
        category: Không dùng (dataset không có cột category).

    Returns:
        List of dict với keys: id, behavior, target.
    """
    path = DATA_DIR / "harmful_behaviors.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. "
            "Hãy đặt file harmful_behaviors.csv vào thư mục data/."
        )

    df = pd.read_csv(path)

    if n:
        df = df.head(n)

    df.insert(0, "id", [f"beh_{i:04d}" for i in range(len(df))])

    # Đổi tên cột về chuẩn pipeline
    df = df.rename(columns={"goal": "behavior"})

    print(f"[dataset] Loaded {len(df)} behaviors.")
    return df[["id", "behavior", "target"]].to_dict(orient="records")


def load_harmful_strings() -> list[str]:
    """
    Load harmful_strings.csv dùng cho Evaluator keyword matching.
    Cột: target (các chuỗi đặc trưng cho response có hại)

    Returns:
        List of lowercase harmful string patterns.
    """
    path = DATA_DIR / "harmful_strings.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. "
            "Hãy đặt file harmful_strings.csv vào thư mục data/."
        )

    df = pd.read_csv(path)
    strings = df["target"].dropna().str.lower().str.strip().tolist()

    print(f"[dataset] Loaded {len(strings)} harmful strings for evaluation.")
    return strings


# Xem truoc:
"""
python3 -c "
from pipeline.dataset import load_behaviors, load_harmful_strings
seeds = load_behaviors(n=3)
for s in seeds: print(s)
print('---')
strings = load_harmful_strings()
print(strings[:5])
"
"""