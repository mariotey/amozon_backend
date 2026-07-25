"""
Creates the required directory structure and artefacts
"""
import subprocess
import sys
from config import (
    REPO_ROOT,
    DATA_DIR, DATA_INPUT_DIR, DATA_OUTPUT_DIR,
    MODEL_ARTEFACT_DIR
)

# Directories required for the forecasting pipeline.
# Each directory is created if it does not already exist.
DIRECTORIES = (
    DATA_DIR,
    DATA_INPUT_DIR,
    DATA_OUTPUT_DIR,
    MODEL_ARTEFACT_DIR
)

print("\nCreating project directories...")

# Create the required directory structure.
for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {directory}")

print("\nPulling data from Supabase Tables...")

# Pull data from Supabase into local environment
subprocess.run(
    [sys.executable, "-m", "preprocessing.extract_from_supabase"],
    cwd=REPO_ROOT,
    check=True,
)

print("\nSetup complete!\n")