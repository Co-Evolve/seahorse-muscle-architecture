"""Package the Tendon Designer as a self-contained zip for a Windows user.

The zip holds the static web app, a double-click launcher and a small web server
(Python if available, otherwise the one built into Windows PowerShell). No
conda/repo needed on the receiving side.

    python -m seahorse_muscle_architecture.silico.tendon_designer.package_windows [--out dist/]

Layout of the zip:

    TendonDesigner/
      Start Tendon Designer.bat
      START HERE.txt
      app/        (contents of web/, without the dev pages, plus README.md)
      tools/      (serve.py, serve.ps1)
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
WEB_DIR = PACKAGE_DIR / "web"
WINDOWS_DIR = PACKAGE_DIR / "windows"
REPO_ROOT = PACKAGE_DIR.parents[2]

EXCLUDED_WEB_FILES = {"dev_sim.html", "dev_editor.html", ".DS_Store"}
ZIP_NOTE = ("> **Received the Tendon Designer as a zip?** Then you do not need the commands below:\n"
            "> double-click `Start Tendon Designer.bat` (see `START HERE.txt`).\n\n")


def version_string() -> str:
    today = datetime.date.today().isoformat()
    try:
        commit = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=True
                ).stdout.strip()
        return f"{today} (repo {commit} + local changes)"
    except (OSError, subprocess.CalledProcessError):
        return today


def to_crlf(
        text: str
        ) -> bytes:
    return text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")


def build(
        out_dir: Path
        ) -> Path:
    if not (WEB_DIR / "model" / "catalog.json").is_file():
        raise SystemExit("web/model/ is missing: run export_assets.py first.")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "TendonDesigner"
        app = root / "app"
        tools = root / "tools"

        shutil.copytree(
                WEB_DIR, app,
                ignore=lambda directory, names: [n for n in names if n in EXCLUDED_WEB_FILES]
                )
        (app / "README.md").write_text(ZIP_NOTE + (PACKAGE_DIR / "README.md").read_text(encoding="utf-8"),
                                       encoding="utf-8")

        tools.mkdir()
        shutil.copy2(PACKAGE_DIR / "serve.py", tools / "serve.py")
        # Windows-native line endings for the files a Windows user opens or runs.
        (tools / "serve.ps1").write_bytes(to_crlf((WINDOWS_DIR / "serve.ps1").read_text(encoding="utf-8")))
        (root / "Start Tendon Designer.bat").write_bytes(
                to_crlf((WINDOWS_DIR / "Start Tendon Designer.bat").read_text(encoding="utf-8"))
                )
        start_here = (WINDOWS_DIR / "START HERE.txt").read_text(encoding="utf-8")
        (root / "START HERE.txt").write_bytes(to_crlf(start_here.replace("{VERSION}", version_string())))

        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / f"TendonDesigner-{datetime.date.today().isoformat()}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root.parent).as_posix())
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the Tendon Designer for Windows.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "dist", help="output folder for the zip")
    args = parser.parse_args()
    zip_path = build(args.out)
    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
