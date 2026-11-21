# AI53 – MLOps Project Phase 1 – Lab Report

## Milestone 0 – Starter audit

I read `params.yaml`, `pyproject.toml`, `.gitignore` and the `reproducibility`, `validate_data`, `provenance`, `train`, `data`, `prepare` and `evaluate` modules in `src/` before changing anything.

| # | Area | File / setting | Problem | Suggested fix | How to check the fix |
|---|------|----------------|---------|---------------|----------------------|
| 1 | Random seeds | `src/reproducibility.py` – `set_reproducibility()` | Only Python's `random` is seeded. NumPy, PyTorch (CPU and CUDA) and the cuDNN flags are TODOs. Weight initialization, dropout and augmentation use the PyTorch RNG, so `train.random_state: 42` does not make two runs give the same results. | Also call `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)` when CUDA is available, and set `torch.backends.cudnn.deterministic = True` and `benchmark = False`. | `tests/test_reproducibility.py` passes. Two trainings with the same params give identical per-epoch losses. |
| 2 | Random seeds / data order | `src/reproducibility.py` – `make_generator()`, `seed_worker()`, used in `src/data.py` | `make_generator()` returns an **unseeded** `torch.Generator()` and `seed_worker()` does nothing. The train `DataLoader` shuffles from this generator, so the batch order changes every run even though a seed is configured. | Return `torch.Generator().manual_seed(seed)`. In `seed_worker`, derive `torch.initial_seed() % 2**32` and seed NumPy and `random` with it. | Taking the first batch of two loaders built with the same seed gives the same labels and image order. |
| 3 | Data checks | `src/validate_data.py` – `validate_dataset()` raises `NotImplementedError` | There is no data-quality gate. Missing splits, wrong or extra classes, wrong image counts (expected 1034 / 133 / 128), empty classes and corrupt images reach PREPARE and training unnoticed. `prepare.py` would also crash on a truncated image or skip it silently. | Implement the 5 Core checks and write a deterministic JSON report without timestamps to `reports/data_quality.json`. Fail (non-zero exit) when a check fails so the pipeline stops. | `tests/test_data_validation.py` / `test_validation_contract.py` pass. A copy of the data with one corrupt JPEG or a missing class gives FAIL. |
| 4 | Provenance / Run history | `src/provenance.py` – `build_run_manifest()` raises `NotImplementedError` | Nothing records the Git commit, whether the working tree was dirty, the prepared-data identity, library versions (Python, torch, torchvision, DVC, MLflow), OS, device or seed. A result cannot be linked back to the code, data and environment that produced it. | Collect these with `subprocess` (`git rev-parse HEAD`, `git status --porcelain`), `platform`, `importlib.metadata`, `torch.version.cuda` and `training_data_identity()`. Write the manifest **before** training starts. | `tests/test_provenance.py` passes. `reports/run_manifest.json` holds the current `git rev-parse HEAD` and `dirty: true` when a file is modified. |
| 5 | Run history | `src/train.py` – TODO 1–7 | MLflow Runs are created but empty: no params, per-epoch metrics, training time, dataset input, artifacts or logged model. Each run also overwrites `models/model.pt` and `reports/*.json` on disk, so earlier candidates are lost and cannot be compared or selected later. | Log params, metrics with `step=epoch`, `training_seconds`, the training dataset (`mlflow.log_input`), the reports as artifacts and the model `bean-cnn` with signature, input example and metadata (`class_names`, `image_size`). | In the MLflow UI each Run shows params, curves, artifacts and a model. `runs:/<id>/bean-cnn` loads. |
| 6 | Data versioning | `.gitignore` (`data/raw/`, `data/processed/`, `models/*.pt`) + no `.dvc/` | The images and checkpoints are excluded from Git, but nothing else tracks them: DVC is not initialized and there is no `.dvc` pointer or remote. Another person cannot get the exact same raw images, and a change to the data leaves no trace. | `dvc init`, `dvc add data/raw`, commit the `.dvc` pointer and push the data to a DVC remote. | `git clone` + `dvc pull` on a clean folder restores `data/raw` with the same hashes (`dvc status` is clean). |
| 7 | Pipeline | No `dvc.yaml` | VALIDATE → PREPARE → TRAIN → EVALUATE are separate scripts run by hand in any order. `prepare.py` even says it *must* depend on the quality report, but nothing enforces that. Dependencies (code, params, data) are not declared, so after a change nobody knows which steps must run again. | Write `dvc.yaml` with `deps`, `params`, `outs` and `metrics` for each stage. PREPARE depends on `reports/data_quality.json`. Commit `dvc.lock`. | `dvc dag` shows the expected graph. `dvc status` / `dvc repro` reruns only the stages affected by a change. |
| 8 | Test-set use | `src/evaluate.py`, `params.yaml` → `selection.run_id: ""` | The test set must not influence model choice, but nothing stops running `evaluate.py` on several Runs and keeping the best test score. That would turn the test set into a second validation set and make the reported score optimistic. | Choose the metric and tie-break rule **before** training. Select a Run only on validation metrics, freeze its Run ID in `selection.run_id`, and run EVALUATE once as the last DVC stage. | Git history shows `selection.run_id` set once before `reports/test_metrics.json` appears. TRAIN never loads `test/`. |
| 9 | Environment | `pyproject.toml` ranges vs `uv.lock`, `.python-version` | `pyproject.toml` only gives version ranges (e.g. `torch>=2.10,<2.12`). Installing with `pip install .` or `uv sync` without `--locked` can pick different versions on another day or machine. | Always install with `uv sync --locked` and check with `uv lock --check`. Record the installed versions in the run manifest (issue 4). | `uv lock --check` passes and the manifest versions match `uv.lock`. |
| 10 | Settings | `params.yaml` → `compute.device: "auto"` | The same params give CPU on one machine and CUDA on another. GPU kernels are not bit-identical to CPU, so the results differ, and the device actually used is not recorded. | Set `device: "cpu"` for this course (CPU-only wheels), or keep `auto` but log the resolved device as a param and in the manifest. | The Run shows `device=cpu`. Two machines give the same metrics. |
| 11 | Data / code | `src/prepare.py` line 64 – `dst = out_class / f"{src.stem}.jpg"` | Output names only keep the file stem. Files with the same stem (`leaf.png` and `leaf.jpg`, or in different subfolders found by `rglob`) overwrite each other silently. The manifest then lists the same `relative_path` twice and the processed count is lower than the raw count. | Detect duplicate destination names and fail, or keep the relative subpath/extension in the output name. The validator can also check that the processed counts match the raw counts. | Unit test with two files sharing a stem → PREPARE fails or keeps both. The `data_summary.json` counts equal 1034 / 133 / 128. |

### Engineering checkpoint 1

**Why is a working `model.pt` insufficient evidence that an experiment is reproducible?**

A `model.pt` only shows that *some* training run finished once. It holds the weights, class names, image size and dropout, but not what produced them:

- **Code:** no Git commit SHA and no sign of uncommitted changes, so we cannot tell which version of `train.py`, `model.py` or `data.py` was used.
- **Data:** no identity of the raw or prepared images. The file is identical whether the data was validated, partly corrupt or a different version.
- **Settings:** learning rate, epochs, batch size, augmentation and seed are missing, and `params.yaml` may have changed since.
- **Environment:** no Python/PyTorch/torchvision versions, OS or device. Different library versions or CPU vs GPU can change the results.
- **Randomness:** with the seeds not fully set (issues 1–2), training again with the *same* inputs still gives a different model. So the file cannot be regenerated, only copied.
- **Evidence:** it holds no validation metrics or training curves and is overwritten by every run. We cannot show it is the selected candidate or that the test set was not used to choose it.

Reproducibility means someone else can rebuild the result from recorded code, data, environment and parameters and get the same metrics. A working checkpoint is only an artifact; without its provenance it cannot be checked.

## Milestone 1 - Make training repeatable

### Environment files

- **`pyproject.toml`**: describes the project and the packages it needs (torch, mlflow, dvc...), with allowed version ranges. It says *what* we need, not the exact versions.
- **`uv.lock`**: records the exact version and hash of every package, including dependencies of dependencies. With `uv sync --locked` everyone installs exactly the same environment, and `uv lock --check` fails if `pyproject.toml` and the lock file don't match.
- **`.venv/`**: the local virtual environment that uv builds from `uv.lock`. It holds the installed packages for this project only. It is not committed (it's in `.gitignore`) because it can be rebuilt from the lock file at any time.

### Engineering checkpoint 2

**Does `random_state=42` guarantee bit-for-bit identical results on every CPU/GPU/software stack?**

No. The seed only fixes the random numbers (weight init, shuffle order, dropout, augmentation). The calculations done with them can still differ:

- CPUs and GPUs don't add floating-point numbers in the same order, and float addition is not associative, so results differ slightly.
- Some GPU operations are non-deterministic by default, even with a fixed seed.
- Other versions of PyTorch, CUDA, the math libraries, Pillow or the OS can use different algorithms.

These small differences add up over training. The seed gives identical results on the **same machine, device and locked environment**. That's why we also lock the environment with `uv.lock` and will record the device and library versions for each run. On a different setup, we can only expect very close metrics, not identical bits.


## Milestone 2 - Extract and track the images

### Engineering checkpoint 3

- **Git remote** (GitHub, `origin`): stores the code, configuration and history, including the small `data/raw.dvc` pointer file with the hash of the images, but not the images themselves.
- **DVC remote** (`lab-data`, the folder `../ai53-phase1-dvc-storage`): stores the actual image files, saved under their content hash, so the pointer in Git can be turned back into the real data with `dvc pull`.
- **Does `git push` transfer the raw images?** No, `git push` only sends the `.dvc` pointer because `data/raw/` is ignored by Git, so the images have to be uploaded separately with `dvc push`.

## Milestone 4 - Design the DVC pipeline

- **Should PREPARE depend on `reports/data_quality.json`?** Yes. This creates the link `validate -> prepare`, so DVC always runs the quality gate first. If validation fails, the report isn't written and PREPARE never runs on bad data.
- **Should PREPARE also depend directly on `data/raw`?** Yes, because PREPARE reads the images itself. The quality report only has counts, so if an image changes but the counts stay the same, the report doesn't change. Without `data/raw` as a dependency, PREPARE would wrongly be skipped.
- **Which `params.yaml` changes should rerun TRAIN?** Everything TRAIN reads: `train.*` (batch size, epochs, learning rate, dropout, seed), `augmentation.*`, `data.image_size`, `compute.device` and `tracking.*` (where the Run is logged). Changing `evaluation.batch_size` or `selection.run_id` must not retrain the model, so they are not listed. That's why I listed single keys instead of whole sections.
- **Why should EVALUATE track `selection.run_id`?** The test results depend on which model is tested. If I select another Run, DVC has to see that the test metrics are out of date and run EVALUATE again. It also records in Git which Run the reported test score belongs to.
- **Which file has scores DVC should track as metrics?** `reports/test_metrics.json` (final test accuracy and macro-F1). I also marked `reports/train_summary.json` as metrics because it holds the validation scores, so `dvc metrics show` can compare runs.

**Note:** EVALUATE also depends on the train/validation manifests, because `evaluate.py` recomputes `training_data_identity()` to check that the selected Run was trained on the current data. It doesn't depend on `model.pt`, since the model comes from MLflow, which keeps the test set out of the TRAIN branch.

## Milestone 5 - Record what each MLflow Run used

### Engineering checkpoint 4

- **Git** records the exact code and configuration of the Run: the commit SHA (and dirty flag) stored in the manifest points to the version of `src/`, `params.yaml`, `dvc.yaml` and `uv.lock` that was used.
- **DVC** records the data and pipeline state: `data/raw.dvc` and `dvc.lock` store the hashes of the raw images, prepared data, params and outputs, so the same inputs can be restored with `dvc pull` / `dvc checkout`.
- **MLflow** records the result of the Run itself: parameters, per-epoch metrics, the training Dataset, the provenance manifest and the logged model that we can later select and load by Run ID.

## Milestone 6 - Predict which stages need to run again

### 6.1 Predictions

| Change | VALIDATE | PREPARE | TRAIN | EVALUATE |
|--------|----------|---------|-------|----------|
| Change `train.learning_rate` | REUSED | REUSED | RERUN | REUSED |
| Change `data.image_size` | REUSED | RERUN | RERUN | NEEDS UPDATE |
| Edit `src/validate_data.py`; the quality report stays the same | RERUN | REUSED | REUSED | REUSED |
| Edit only `README.md` | REUSED | REUSED | REUSED | REUSED |

### Engineering checkpoint 5

**Setting change: `data.image_size`**
- VALIDATE: REUSED. It has no `params` in `dvc.yaml`, and its deps (`data/raw`, `validate_data.py`, `uv.lock`, `pyproject.toml`) don't change.
- PREPARE: RERUN. `data.image_size` is in its `params`, so its inputs changed.
- TRAIN: RERUN. `data.image_size` is also in its `params`, and its deps `data/processed/train`, `validation` and the manifests are rewritten with the new size.
- EVALUATE: NEEDS UPDATE. `data.image_size` is in its `params` and `data/processed/test` changes. The selected Run was also trained at the old size, so the saved test results no longer match (`evaluate.py` would even refuse the old Run).

**File edit: `src/validate_data.py` (report unchanged)**
- VALIDATE: RERUN. `src/validate_data.py` is one of its deps, so its hash changed.
- PREPARE: REUSED. Its dep is the *output* `reports/data_quality.json`, not the script. The report content is the same, so its hash is the same and none of PREPARE's inputs changed.
- TRAIN and EVALUATE: REUSED. Their inputs come from PREPARE, which didn't change.

### 6.3 Decisions before training

- **Main validation metric:** `val_macro_f1` at the final epoch. **Tie-break:** if two Runs have the same macro-F1 (to 3 decimals), I choose the lower `val_loss`. Macro-F1 gives the same weight to the 3 classes, so it's a better choice than accuracy if one disease class is predicted badly.
- **Expected effect of a lower learning rate (0.001 -> 0.0003):** with only 4 epochs, I expect the model to learn more slowly: a higher train and validation loss at each epoch and probably a slightly lower validation macro-F1, but smoother, more stable curves.
- **Run A (baseline, `learning_rate=0.001`):** `d7f31458868947a8817363e5f98676dc`
- **Run B (`learning_rate=0.0003`):** `5867735848e64a468cc3e29066036840`

### 6.4 Observed results

| Check | VALIDATE | PREPARE | TRAIN | EVALUATE |
|-------|----------|---------|-------|----------|
| Run A: first `dvc repro train` (no `dvc.lock` yet) | RERUN | RERUN | RERUN | NOT RUN |
| 1. Run again, nothing changed | REUSED | REUSED | REUSED | NOT RUN |
| 2. Comment in `src/validate_data.py` | RERUN | REUSED | REUSED | NOT RUN |
| Run B: `train.learning_rate` 0.001 -> 0.0003 | REUSED | REUSED | RERUN | NOT RUN |

Terminal lines:

```text
# Run A
Running stage 'validate':
Running stage 'prepare':
Running stage 'train':
Training complete. MLflow Run ID: d7f31458868947a8817363e5f98676dc

# Check 1
Stage 'validate' didn't change, skipping
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
Data and pipelines are up to date.

# Check 2
Running stage 'validate':
> uv run --locked python -m src.validate_data
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping

# Run B
Stage 'validate' didn't change, skipping
Stage 'prepare' didn't change, skipping
Running stage 'train':
Training complete. MLflow Run ID: 5867735848e64a468cc3e29066036840
```

**Comparison with the predictions:**
- **Check 2** matches my prediction for the `validate_data.py` row. VALIDATE reran because its code dep changed, but `reports/data_quality.json` had the same content (same MD5 before and after: `139ec61e...`), so PREPARE's deps didn't change and DVC skipped it, and TRAIN with it.
- **Run B** matches the `learning_rate` row. Only TRAIN has `train.learning_rate` in its `params`, so VALIDATE and PREPARE were skipped. EVALUATE was not run because `dvc repro train` only runs `train` and the stages before it.
- **Run A** reran everything because there was no `dvc.lock` yet, so DVC had nothing to compare with. This isn't one of the predicted rows. Check 1 shows that once `dvc.lock` exists, nothing reruns without a change.
- **Difference I didn't expect:** Run B's manifest says `dirty: true` although I committed before training. Before running a stage, DVC deletes that stage's outputs. `reports/run_manifest.json`, `train_summary.json` and `training_curves.png` are tracked by Git (`cache: false`), so they were missing when the manifest ran `git status`. The code at commit `5d5a3e4` was clean; only TRAIN's own outputs show as changed.

## Milestone 7 - Compare three controlled runs

### 7.1 Run C

**Change:** starting from Run A (`learning_rate` back to 0.001), I only set `augmentation.enabled: true` (random horizontal flip with p=0.5 and rotation up to 10 degrees on the training images). Data, seed, epochs and the other settings stay the same.

**Prediction:** augmentation shows the model a slightly different version of each image at every epoch, so the training task is harder. I expect a **higher training loss** than Run A and a **smaller gap between train and validation loss**. With only 4 epochs, the model doesn't have time to benefit much from the extra variety, so I expect the validation macro-F1 to stay **close to Run A** (maybe slightly lower). The benefit of augmentation usually shows up with more epochs, when the baseline starts to overfit.

### 7.2 Validation results (final epoch)

All three Runs have the same `training_data_identity` (`8fa943ff...`), seed (42), epochs (4), batch size (32) and dropout (0.3), as I checked in MLflow. Only the controlled change differs.

| Run | Controlled change | Val. Accuracy | Val. Macro-F1 | Training time | MLflow Run ID |
|-----|-------------------|---------------|---------------|---------------|---------------|
| A | Baseline (`lr=0.001`, no augmentation) | 0.6391 | 0.6254 | 9.3 s | `d7f31458868947a8817363e5f98676dc` |
| B | `learning_rate` 0.001 -> 0.0003 | 0.6165 | 0.6091 | 10.3 s | `5867735848e64a468cc3e29066036840` |
| C | `augmentation.enabled` false -> true | 0.6541 | 0.6416 | 10.0 s | `f46773905ce54c34bc2e714d3ff90041` |

**B vs A:** the results **support** my prediction. With the lower learning rate the model learned more slowly: the losses are higher at every epoch (final val loss 0.8927 vs 0.8328, train loss 0.9198 vs 0.8672) and the final macro-F1 is lower (-0.016). The B curves are still going down at epoch 4, so with more epochs B could catch up. We can't tell from 4 epochs whether it would end up better.

**C vs A:** the results **partly support** my prediction. The training loss is higher with augmentation (0.8770 vs 0.8672), as expected, and the validation loss is almost the same (0.8354 vs 0.8328). But the gap between train and val loss isn't smaller, and macro-F1 is slightly *higher* (+0.016) instead of equal or lower.

**Limits:** these differences are small. On 133 validation images, accuracy 0.6391 -> 0.6541 means 85 -> 87 correct images, only 2 images. With one seed and a small validation set, this isn't enough to say augmentation is better in general. We would need several seeds or more epochs to confirm it.

### Engineering checkpoint 6

**Primary metric:** I chose validation **macro-F1**. It computes the F1 score of each of the 3 classes (`angular_leaf_spot`, `bean_rust`, `healthy`) and averages them with the same weight. A model can have good accuracy while doing badly on one class, for example confusing the two diseases, and accuracy would hide it. Macro-F1 drops if any class is predicted badly, and for a farmer, missing one disease matters as much as missing the other. Our classes are nearly balanced, so accuracy and macro-F1 are close here, but macro-F1 is still the safer choice. The tie-break on lower `val_loss` prefers the model whose predicted probabilities are more confident on the right class.

**Test set only after choosing:** the test set must estimate how the chosen model performs on new, unseen data. If I used it to compare A, B and C and kept the best, I would be selecting on the test set: the score would be optimistic because the choice was fitted to those 128 images, and I would have no unbiased data left to check the final model. So I choose with validation only, freeze the Run ID, and evaluate on the test set once at the end.
