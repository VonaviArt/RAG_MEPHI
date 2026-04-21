# пересборка chroma

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    # чтобы находился mephi_rag, если запуск не из корня
    sys.path.insert(0, str(ROOT))

from mephi_rag.pipeline import rebuild_index

if __name__ == "__main__":
    path = rebuild_index()
    print("Индекс:", path)
