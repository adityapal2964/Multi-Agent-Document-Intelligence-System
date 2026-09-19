"""Generate a tiny sample PDF so you can test ingestion without hunting for a file.

Usage:  python scripts/create_sample_pdf.py
Output: data/documents/sample_report.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

TEXT_PAGES = [
    [
        "Acme Corp - Annual Report 2024",
        "",
        "Revenue for the fiscal year 2024 reached 42.5 million USD,",
        "an increase from 34.0 million USD in 2023.",
        "Gross margin improved to 61 percent.",
        "The company ended the year with 512 employees.",
    ],
    [
        "Section 2: Risks and Outlook",
        "",
        "The main risks identified are customer concentration,",
        "with the top three customers representing 38 percent of revenue,",
        "and rising cloud infrastructure costs.",
        "Management expects revenue growth of 20 to 25 percent in 2025.",
    ],
]


def build_pdf(path: Path) -> Path:
    """Write a minimal, valid, text-extractable PDF without extra dependencies."""
    objects: list[bytes] = []
    font_id = 3 + 2 * len(TEXT_PAGES)

    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(TEXT_PAGES)))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(TEXT_PAGES)} >>".encode()
    )

    for i, lines in enumerate(TEXT_PAGES):
        content_id = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        )
        body = "BT /F1 12 Tf 72 720 Td 16 TL\n"
        for line in lines:
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            body += f"({escaped}) Tj T*\n"
        body += "ET"
        stream = body.encode("latin-1")
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
                       + stream + b"\nendstream")

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n").encode()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


if __name__ == "__main__":
    target = build_pdf(config.DOCUMENTS_DIR / "sample_report.pdf")
    print(f"Sample PDF written to: {target}")
