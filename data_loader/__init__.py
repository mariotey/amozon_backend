"""
Data processing utilities for recommender system pipelines.

This package provides utilities for downloading, loading, transforming, and uploading recommender
system datasets. It includes functionality for retrieving raw datasets, performing feature
engineering, managing local dataset caches, and persisting processed datasets into supabase.

The package exposes dataset loading utilities through the package namespace.
"""
from .data_loader import *