# Publication protocol and provenance audit (2026-11-16)

## Scope and conclusion

This audit compares the implementation with the manuscript *Leakage-Safe Virtual Sensing under Extreme Target Sparsity*. It establishes that the full-project CDAE code supports split-first, train-only preprocessing, and a coverage-based chronological split. It does **not** certify the exact historical experiment executions, model checkpoints or reported point estimates, because the raw 2021 process data, Phase 1/2 split files and fitted neural checkpoints are not committed here.

The top-level README describes legacy exploratory entrypoints, so **do not treat those examples as the publication execution record**. The relevant implementation is under `CDAE_validacao_melhorado`.

## Verified implementation

1. `CDAE.py` implements `blocked_split_indices_with_skip` and `temporal_split_indices_with_skip` using 51 candidate skip fractions, train-only coverage and observed target count. The complete runner supplies `--blocked_by_index --auto_skip_until_coverage --min_target_coverage 0.2 --min_target_count 10000 --max_skip_head_frac 0.95 --time_test_size 0.05 --time_val_size 0.1`.
2. The algorithm calculates `n_test=ceil(N_retained*0.05)`, `n_val=ceil((N_retained-n_test)*0.1)` and `n_train=N_retained-n_test-n_val`. Thus the actual nominal fractions **of retained chronology** are approximately **85.5% / 9.5% / 5.0%** (apart from integer rounding), not 85% / 10% / 5%. With N=105,121 and skip fraction 0.361, the deterministic nominal counts are N_retained=67,173, train=57,432, validation=6,382 and test=3,359 *before target-availability filtering*. These are mathematical consequences of the documented algorithm, not measurements of the historical run.
3. Feature screening uses `df.loc[train_idx]`; target-observed subsets are intersected with chronological train/val/test folds; auxiliary missing values are imputed using means of `train_idx_model`, and feature/target scalers are fitted on that set only.
4. `teste_controlado_cdae_v2.py` supports the same split functions; MCAR/MAR/MNAR scoring uses target-observed test references and always forces `f_theta(X,0,1)` at inference. The fact that floor(0.8*3,058)=2,446 agrees with the reported masked-index count is a useful *numerical consistency check* but does not prove which indices were selected.
5. **Crucial:** the older `run_complete_pipeline.sh` and `run_pipeline_any_target.sh` invoked the controlled-validation script without passing the chronological flags. That executable's original defaults were a random 70%/15%/15% split. This is a real reproducibility trap in the example runners, *not* proof that published results came from a random split. This branch adds explicit chronological and strict-validation flags for all three Phase 2 invocations. Phase 1 is also run with strict temporal feasibility validation.
6. Main Phase 1 (100 epochs, batch 256) and three Phase 2 commands (50 epochs, batch 128 each) instantiate and fit different network checkpoints. **They reuse the feature configuration and split policy, not Phase 1 model weights.** The three Phase 2 commands also each retrain their own neural network, so equality of architecture, seed and split policy does not guarantee identical model weights.
7. Baseline protocol has a separate shared artifact pipeline, with split comparison against an optional reference NPZ and train-only preprocessing. The existing configuration has `strict_validation=False` by default; in strict mode, it now fails rather than falling back to random splitting on empty target-observed folds. Actual baseline/index identity remains to be established using the archived files.

## Files visible in the public repository

- The auxiliary-only MLP `run_manifest.json` records `split_mode=blocked_by_index`, skip fraction `0.361`, 10 auxiliaries and training settings; its MCAR metrics have n=2,446 and MAE about 2.928, matching the paper's rounded MLP result.
- Aggregated baseline and model-comparison JSON files reproduce rounded values printed in manuscript baseline tables.
- Crucially, the repository tree does **not** contain the 2021 HDF5 dataset, the reference Phase 1/2 split NPZ files or the corresponding fitted neural checkpoints. An archived relative/absolute path in a JSON manifest is a *pointer*, not the data.

## How to verify exact original runs

On the machine holding the **original, trusted, unmodified** run directories, execute:

```bash
python CDAE_validacao_melhorado/audit_publication_protocol.py \
  --phase1 /PATH/ORIGINAL_PHASE1_OUTPUT \
  --phase2 /PATH/ORIGINAL_PHASE2_MCAR /PATH/ORIGINAL_PHASE2_MAR /PATH/ORIGINAL_PHASE2_MNAR \
  --baseline /PATH/BASELINES_SHARED \
  --output /PATH/LOCAL_REPORT/protocol_audit.json
```

The audit produces SHA-256 file hashes, disjointness/uniqueness checks and exact split-index comparisons. It reports `PASS_INDICES_ONLY` only when the indices match and no internal overlap is found; it **does not** prove training weights, mask identity, data hashes or experiment execution. The historic outputs may require trusted `np.load(..., allow_pickle=True)` because pandas indices sometimes serialize as object arrays; never use untrusted files.

Additional mandatory provenance items: raw dataset SHA-256; row order/date-column evidence; actual commands/environment; train/val/test index NPZ; feature selection configuration; training and inference checkpoints; mechanism-specific mask index files and prediction arrays; and per-run JSON metrics. These allow independent numerical comparison against tables in the paper.

## Future-run safeguards added on this branch

- Full and secondary-target shell runners now explicitly pass chronology flags and enable `--strict_temporal` in Phase 1 and Phase 2.
- `CDAE.py` and `teste_controlado_cdae_v2.py` stop with an error if strict chronological splitting leaves an empty observed-target fold, instead of silently randomly resplitting.
- `baselines/protocol.py` has an equivalent guard when strict validation is requested.
- Both complete-run scripts now write to timestamp-specific directories instead of removing existing experimental artifacts (the old full runner called `rm -rf testes`; the secondary-target runner deleted earlier mechanism results).
- Both consolidation blocks now correctly read numeric fields inside the `metrics` object of `controlled_test_results.json`; the former top-level `get` pattern could silently write zero-valued summary columns.
- `audit_publication_protocol.py` also reports SHA-256 hashes of available Phase 2 checkpoints. Identical file hashes establish byte identity; different hashes do not automatically establish different tensors. Checkpoint identity is **unknown** when files are missing.
- `plot_publication_comparison.py` can regenerate large-label vector PDF and 300-dpi PNG comparison figures from the committed aggregate JSON, without accessing confidential measurements. Multi-panel parity and temporal figures still require original prediction arrays or raw data.
- No historic metrics or model artifacts were changed. A fresh run under the corrected scripts is a **new experiment** until its actual indices and metrics have been checked.

## Reporting changes for manuscript

- Describe the exact algorithmic split as **5% test followed by 10% of the remainder for validation**, yielding about **85.5%/9.5%/5%** of the retained chronology.
- Specify explicitly that the controlled Phase 2 networks are independently fitted on the corresponding chronological folds, rather than implicitly suggesting the Phase 1 checkpoint is reused. The contiguous-outage replay uses its own independently trained model.
- Maintain transparency that artifacts needed for exact reproduction of the historic industrial results are constrained by confidentiality. Availability of publication-grade source code is not equivalent to publication of the training and evaluation populations.

## Generate larger comparison figures from committed statistics

```bash
python CDAE_validacao_melhorado/plot_publication_comparison.py \
  --summary CDAE_validacao_melhorado/testes/baselines_leakage_safe/primary_target_v1/consolidated/comparison_summary.json \
  --output /PATH/OUTPUT_FIGURES
```

This produces vector PDF and 300-dpi PNG equivalents for the MAE and R-squared comparisons. It is **not** a substitute for checking the complete paper layout after compilation.

## Scientific interpretation of controlled mechanisms

Each of the three controlled commands fits its own neural model. Fixed seeds and equal hyperparameters make comparable runs possible, but independent checkpoint weights must be verified before concluding that all metric changes arise only from mechanism-selected scoring populations. Comparing `best_model.pth` hashes is a first check; inspecting tensors and fixed predictions is more conclusive if hashes differ.

