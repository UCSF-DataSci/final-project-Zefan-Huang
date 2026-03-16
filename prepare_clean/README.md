# `prepare_clean`

This directory contains data preparation, alignment, and feature-cleaning
scripts used before or alongside the staged modeling pipeline.

## Contents

- `imaging_preprocessing.py`: CT preprocessing, tumor ROI token extraction,
  and semantic token parsing
- `imaging_preprocessing_normal.py`: compatibility entrypoint for IDE runs
- `label_construction_time_zero.py`: builds OS and recurrence labels
- `total_table.py`: builds the patient modality manifest
- `rna_alignment.py`: Stage 7.1 GEO-to-patient RNA alignment
- `clinical_feature_engineering.py`: Stage 8.1 clinical feature construction

## Compatibility

The original top-level numbered scripts `7.1_rna_alignment.py` and
`8.1_clinical_feature_engineering.py` are kept as thin wrappers so existing
commands and dynamic module loading keep working.
