# Repository contracts

## Required reading and documentation ownership

Read `AGENTS.md`, then `PROJECT_CONTEXT.md` (background, data provenance, preprocessing,
model design and bounded historical conclusions), then the README sections relevant to
execution. Follow only the code paths relevant to the task; do not import the source
project's exploratory experiment backlog as this repository's current plan.

Keep `PROJECT_CONTEXT.md` as the concise permanent handoff and `README.md` as the
operating guide. Update the appropriate document after verified interface/contract
changes. Label observations, open questions and verified results distinctly. Preserve
historical failure lessons without copying experiment diaries or exhaustive score tables.
Local source paths live in ignored `artifacts/migration/`, not in tracked recipes.
GitHub updates should contain concise conclusions and reusable code/tests. Keep
raw predictions, figures, HTML reports, logs and one-off experiment tools local;
do not publish the experiment workspace wholesale.

## Implementation and data contracts

- Keep this repository portable: no host paths, patient identifiers, patient-level statistics,
  datasets, pretrained weights, experiment outputs, or credentials in source control.
- Aggregate cohort provenance belongs only in `PROJECT_CONTEXT.md`; datasets and
  exact host paths remain external/local. Never treat a historical stored channel order
  as verified anatomy or retune using the locked external test cohort.
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
