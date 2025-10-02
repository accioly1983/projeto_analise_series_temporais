# notebook_setup.py
import sys
from pathlib import Path

current = Path.cwd()

# Sobe até encontrar a pasta src
while not (current / "src").exists() and current != current.parent:
    current = current.parent

src_path = current / "src"

if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print(f"[INFO] src adicionado ao sys.path: {src_path}")
else:
    print(f"[ERRO] Pasta 'src' não encontrada a partir de {Path.cwd()}")
