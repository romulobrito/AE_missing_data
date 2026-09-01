# Auxiliary-only MLP ablation

Capacity-matched neural soft sensor `g(x) -> t` used to test whether the
TCDR input channels `[x, t_tilde, m_t]` add value beyond a plain MLP.

## Layout

```text
ablations/                 # code package
run_aux_mlp_ablation.py    # CLI

testes/ablation_aux_mlp/primary_target_v1/
  shared/                  # copied from baselines_leakage_safe/.../shared
  model/best_model.pth
  masks/{MCAR,MAR,MNAR}_rate080_seed42.npz
  eval/{MCAR,MAR,MNAR}/metrics.json
  consolidated/macros_aux_mlp.tex
```

## Commands

From `CDAE_validacao_melhorado/`:

```bash
python run_aux_mlp_ablation.py all --copy_to_tex
```

Or step-wise:

```bash
python run_aux_mlp_ablation.py prepare
python run_aux_mlp_ablation.py train
python run_aux_mlp_ablation.py eval
python run_aux_mlp_ablation.py export --copy_to_tex
```

Widths `h1,h2,latent` match TCDR sizing with `in_dim = n_features + 2`,
while the network input is only `x` (`n_features`).
