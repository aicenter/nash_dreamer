# Data naming conventions

Top-level folders in `data/` are algorithm variants; each contains one subfolder per game
(`config.json`, `nash_conv.txt`, ...). Folder names are listed exactly as they exist on disk.

| Folder | Meaning | Used in |
|---|---|---|
| `nash_dreamer_rnad` | NashDreamer[RNaD], default configuration for deterministic games | Goofspiel, Phantom TTT |
| `nash_dreamer_mmd` | NashDreamer[MMD], default configuration | Goofspiel, Phantom TTT |
| `nash_dreamer_rnad_factored` | NashDreamer[RNaD], **vanilla** hyperparameters, factored latent space | Leduc, the "vanilla" variant |
| `nash_dreamer_rnad_factored_tuned` | NashDreamer[RNaD], factored latent space, **tuned** hyperparameters | Leduc (this is "tuned" / the paper's "factored") |
| `nash_dreamer_mmd_factored_tuned` | NashDreamer[MMD], factored latent space, **tuned** hyperparameters | Leduc |
| `nash_dreamer_mmd_factored_vanilla` | NashDreamer[MMD], factored latent space, **vanilla** hyperparameters | Leduc, main-body NashConv figure |
| `nash_dreamer_rnad_flat` | NashDreamer[RNaD], flat latent space, tuned hyperparameters | Leduc, appendices |
| `nash_dreamer_rnad_one_way` | NashDreamer[RNaD], factored latent space, tuned hyperparameters except the representation loss is turned off (`beta_representation = 0`) | Leduc, appendices |
| `rnad`, `mmd`, `replayed_rnad` | Model-free baselines (RNaD, MMD, RNaD with replay) | various |
| `ablation_data`| Data for the warm-up sweep ablation, 7 checkpoints (6000 minibatches) for each warm up both for NashDreamerRNaD and NashDreamerMMD| appendices
| `le_trees` | The latent error trees mentioned, but not displayed in the paper due to size | appendices

Notes:
- "Tuned" always implies the factored latent unless the name says `flat`.
- The `_tuned` suffix is written `_factored_tuned` in the folder names.
