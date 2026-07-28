"""
Creates the required directory structure and artefacts
"""
from data_loader import DataLoader
from models_loader import ModelsLoader
from config import (
    REPO_ROOT,
    DATA_DIR, DATA_INPUT_DIR, DATA_OUTPUT_DIR, MODEL_ARTEFACT_DIR,
    TOOLS
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

# Pull data from Supabase into local environment
print("\nPulling tabular data from Supabase Tables...\n")
data_obj = DataLoader()

# Pull model artefacts from Supabase into local environment
print("\nPulling model artefacts from Supabase Tables...\n")
models_obj = ModelsLoader(tools=TOOLS)

print("\nSetup complete!\n")