"""
eval/exemplos.py
────────────────
Gera os PDFs de exemplo (data/exemplos/pdf/) a partir dos .txt, para a
avaliação do versionamento e para a demonstração na interface.

    python src/eval/exemplos.py
"""

import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent.parent
EXEMPLOS = ROOT / "data" / "exemplos"
PDF_DIR = EXEMPLOS / "pdf"


def texto_para_pdf(text: str, out_path: Path, width: int = 95) -> Path:
    """PDF com quebra de linha (o make_pdf dos testes corta a linha em 90 caracteres)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    c.setFont("Helvetica", 10)
    top, bottom, step = 800, 50, 13
    y = top
    for para in text.split("\n"):
        lines = textwrap.wrap(para, width=width) or [""]
        for line in lines:
            if y < bottom:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = top
            c.drawString(50, y, line)
            y -= step
    c.save()
    return out_path


def gerar_pdfs() -> dict:
    """Converte todos os .txt de data/exemplos em PDF. Retorna {nome: caminho}."""
    out = {}
    for txt in sorted(EXEMPLOS.glob("*.txt")):
        pdf = texto_para_pdf(txt.read_text(encoding="utf-8"), PDF_DIR / f"{txt.stem}.pdf")
        out[txt.stem] = pdf
    return out


if __name__ == "__main__":
    for nome, caminho in gerar_pdfs().items():
        print(f"{nome}: {caminho}")
