from __future__ import annotations

import tempfile
from pathlib import Path

from _helpers import show_blocks, trace_block
from agent_team.tools.pdf_tool import extract_pdf


def build_sample_pdf(path: Path, text: str = "Hello PDF extractor") -> None:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 18 Tf 24 100 Td ({escaped}) Tj ET".encode("ascii")

    objects: list[bytes] = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        b"4 0 obj\n<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]

    parts = [b"%PDF-1.4\n"]
    offsets = [0]
    for obj in objects:
        offsets.append(sum(len(part) for part in parts))
        parts.append(obj)

    xref_start = sum(len(part) for part in parts)
    xref = [b"xref\n0 6\n", b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        + b"startxref\n"
        + str(xref_start).encode("ascii")
        + b"\n%%EOF\n"
    )

    path.write_bytes(b"".join(parts + xref + [trailer]))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "sample.pdf"
        build_sample_pdf(pdf_path)
        try:
            blocks = extract_pdf(pdf_path, on_block=trace_block)
        except ModuleNotFoundError as exc:
            print(f"pdf extractor skipped: {exc}")
            print("Install the optional pypdf dependency to run this script.")
            return
        show_blocks("pdf extractor", blocks)


if __name__ == "__main__":
    main()