"""First-run environment setup, not a setuptools packaging entry point."""
import argparse
import shutil
import subprocess
import sys

from config.ml_config import DATA_OUTPUT_DIR, MODEL_ARTEFACT_DIR, REPO_ROOT, TOOLS


def prepare_local_files():
    """Create workflow folders and a credential template without replacing files."""
    for directory in (DATA_OUTPUT_DIR, *(MODEL_ARTEFACT_DIR / tool for tool in TOOLS)):
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Ready: {directory}")

    env_path = REPO_ROOT / ".env"
    try:
        with env_path.open("x", encoding="utf-8") as env_file:
            env_file.write(
                "SUPABASE_URL=https://YOUR-PROJECT.supabase.co\n"
                "SUPBASE_SECRET_KEY=YOUR_SERVER_SIDE_SECRET_KEY\n"
            )
    except FileExistsError:
        print("Existing .env kept unchanged.")
    else:
        print("Created .env; replace its placeholders with your Supabase credentials.")


def install_dependencies():
    """Use the committed lockfile and keep the notebook kernel inside .venv."""
    executable = shutil.which("uv")
    if executable:
        uv = [executable]
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "uv"],
            cwd=REPO_ROOT, check=True,
        )
        uv = [sys.executable, "-m", "uv"]

    subprocess.run(
        [*uv, "sync", "--frozen", "--no-install-project"],
        cwd=REPO_ROOT, check=True,
    )
    subprocess.run(
        [*uv, "run", "--no-sync", "python", "-m", "ipykernel", "install",
         "--sys-prefix", "--name", "amozon_backend",
         "--display-name", "Amozon Backend (.venv)"],
        cwd=REPO_ROOT, check=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-install", action="store_true",
        help="Only create local folders and the .env template; skip dependencies and kernel setup.",
    )
    args = parser.parse_args(argv)
    if sys.version_info < (3, 13):
        print("Python 3.13 or newer is required; Python 3.13 is recommended.", file=sys.stderr)
        return 1
    try:
        prepare_local_files()
        if not args.skip_install:
            install_dependencies()
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        return 1
    print("Local files ready." if args.skip_install else "Setup complete. Run: uv run jupyter lab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
