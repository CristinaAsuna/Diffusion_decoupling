# Repository contracts

- Keep this repository portable: no host paths, patient identifiers, cohort statistics,
  datasets, pretrained weights, experiment outputs, or credentials in source control.
- Anatomical patient left is DICOM LPS +X, segmentation label 8, output channel 0.
  Patient right is LPS -X, label 9, output channel 1. Display orientation does not
  determine branch identity. Do not automatically swap branches.
- Models predict D = (left - right) / 2 with one train-only fixed scale s_D.
  Recover raw D from D/s_D and decode left = full/2 + D, right = full/2 - D.
  Preserve this hard additive decoder; do not center D or scale each image separately.
- Patient splits and scaling come from `decouple prepare-data`. Never estimate them
  from validation data. Checkpoints must retain their resolved config, scale and family.
- Keep package recipes self-contained under `consistent_cdm/recipes/`. Logging is
  optional and disabled by default. Store each training run below `runs/<family>/`.
- Optional CBCT segmentation requires the separately installed segmentation extra and
  user-provided nnU-Net weights. Core help, training and inference must import without it.
- Run checks appropriate to changes. Do not submit jobs, download weights, or publish
  without explicit authorization. Keep the Slurm template free of site-specific settings.
