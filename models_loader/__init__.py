"""
Model artefact management utilities for recommender system pipelines.

This package provides utilities for managing recommender system model artefacts, including loading
cached artefacts, retrieving missing artefacts from supabase, and uploading generated artefacts
into supabase storage.

The package exposes model loading functionality through the package namespace.
"""
from .models_loader import *