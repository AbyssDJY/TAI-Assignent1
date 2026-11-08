"""Fill in the group's student IDs, rebuild the PDF and rename it as required.

The assignment requires the submitted file to be named
``<studentID_1>_<studentID_2>.pdf`` and to show both students' IDs on the report.

Usage
-----
    python src/set_student_ids.py --a G2501234A --b G2505678B

This rewrites the two \\newcommand lines in ``report/main.tex``, recompiles the
report with pdflatex (twice, for the table of contents) and copies the result to
``report/<id1>_<id2>.pdf``.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "report", "main.tex")


def patch(tex_path: str, id_a: str, id_b: str) -> None:
    with open(tex_path, encoding="utf-8") as f:
        s = f.read()
    s, n1 = re.subn(r"\\newcommand\{\\StudentIDA\}\{[^}]*\}",
                    r"\\newcommand{\\StudentIDA}{" + id_a + "}", s)
    s, n2 = re.subn(r"\\newcommand\{\\StudentIDB\}\{[^}]*\}",
                    r"\\newcommand{\\StudentIDB}{" + id_b + "}", s)
    if n1 != 1 or n2 != 1:
        raise SystemExit("could not find the \\StudentIDA / \\StudentIDB lines")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(s)
    print(f"[ids] set StudentIDA={id_a}  StudentIDB={id_b}")


def compile_pdf() -> str:
    rep = os.path.join(ROOT, "report")
    for i in (1, 2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                           cwd=rep, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            print(f"[ids] pdflatex failed on pass {i}; see report/main.log")
            sys.exit(1)
    pdf = os.path.join(rep, "main.pdf")
    if not os.path.exists(pdf):
        raise SystemExit("no PDF produced")
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="student ID of YaoChong CHEN")
    ap.add_argument("--b", required=True, help="student ID of Junyuan DENG")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    patch(TEX, args.a, args.b)
    if args.no_compile:
        return
    pdf = compile_pdf()
    dst = os.path.join(ROOT, "report", f"{args.a}_{args.b}.pdf")
    shutil.copyfile(pdf, dst)
    print(f"[ids] wrote {dst}")


if __name__ == "__main__":
    main()
