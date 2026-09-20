# Graph parameterization diagnostic

Validated 420 selected units for `posthoc_graph_parameterization_v1`.

This is a validation-only transfer diagnostic. Contrasts are paired descriptive summaries across repeated seeds; the validation-selected graph comparison is selection-optimistic and does not make a test or population claim.

## Validation accuracy by dataset and architecture

| Dataset | Model | Raw mean (SD) | NormalizeFeatures mean (SD) | Centered-scaled mean (SD) |
|---|---|---:|---:|---:|
| Roman-empire | MLP | 0.664426 (0.0045785) | 0.167544 (0.0420314) | 0.66447 (0.00442822) |
| Roman-empire | GCN | 0.509308 (0.00376823) | 0.204532 (0.0183355) | 0.507119 (0.00304228) |
| Roman-empire | GAT | 0.461464 (0.00973074) | 0.151072 (0.0134477) | 0.425293 (0.00312011) |
| Roman-empire | GraphSAGE | 0.783551 (0.00534904) | 0.21342 (0.0426214) | 0.785143 (0.00301494) |
| Roman-empire | H2GCN | 0.812072 (0.00530007) | 0.283418 (0.00478006) | 0.810679 (0.00432127) |
| Roman-empire | LINKX | 0.605395 (0.00720864) | 0.433407 (0.0403929) | 0.603958 (0.00490743) |
| Roman-empire | GPR-GNN | 0.706124 (0.00430459) | 0.13973 (0) | 0.704223 (0.0036356) |
| Amazon-ratings | MLP | 0.41844 (0.00502831) | 0.36798 (0) | 0.426486 (0.00688556) |
| Amazon-ratings | GCN | 0.437615 (0.00448222) | 0.36798 (0) | 0.44515 (0.00356885) |
| Amazon-ratings | GAT | 0.443557 (0.00520399) | 0.36798 (0) | 0.442067 (0.00413912) |
| Amazon-ratings | GraphSAGE | 0.470901 (0.00552659) | 0.36798 (0) | 0.489279 (0.00300893) |
| Amazon-ratings | H2GCN | 0.466878 (0.00465668) | 0.369144 (0.000805698) | 0.475638 (0.00537339) |
| Amazon-ratings | LINKX | 0.541005 (0.00283129) | 0.536145 (0.00298613) | 0.542148 (0.0036204) |
| Amazon-ratings | GPR-GNN | 0.458709 (0.00486304) | 0.36798 (0) | 0.463467 (0.00613919) |

## Main paired contrasts

| Dataset | Model | Contrast | Mean validation-accuracy difference | SD | Range | Positive / zero / negative |
|---|---|---|---:|---:|---:|---:|
| Roman-empire | MLP | centered_scaled_minus_normalized | 0.496927 | 0.0428305 | 0.411231–0.530179 | 10 / 0 / 0 |
| Roman-empire | MLP | centered_scaled_minus_raw | 4.42207e-05 | 0.00186674 | -0.00243199–0.00375855 | 4 / 2 / 4 |
| Roman-empire | MLP | normalized_minus_raw | -0.496883 | 0.0423849 | -0.530621–-0.412779 | 0 / 0 / 10 |
| Roman-empire | GCN | centered_scaled_minus_normalized | 0.302587 | 0.0192225 | 0.284767–0.333407 | 10 / 0 / 0 |
| Roman-empire | GCN | centered_scaled_minus_raw | -0.00218881 | 0.00286842 | -0.00596946–0.00198978 | 3 / 0 / 7 |
| Roman-empire | GCN | normalized_minus_raw | -0.304776 | 0.0196454 | -0.336723–-0.284767 | 0 / 0 / 10 |
| Roman-empire | GAT | centered_scaled_minus_normalized | 0.274221 | 0.0148079 | 0.250719–0.287862 | 10 / 0 / 0 |
| Roman-empire | GAT | centered_scaled_minus_raw | -0.0361707 | 0.0080921 | -0.0517356–-0.0249834 | 0 / 0 / 10 |
| Roman-empire | GAT | normalized_minus_raw | -0.310391 | 0.0169092 | -0.337608–-0.283219 | 0 / 0 / 10 |
| Roman-empire | GraphSAGE | centered_scaled_minus_normalized | 0.571722 | 0.0427407 | 0.526863–0.64581 | 10 / 0 / 0 |
| Roman-empire | GraphSAGE | centered_scaled_minus_raw | 0.00159187 | 0.00321882 | -0.00353748–0.00707495 | 6 / 1 / 3 |
| Roman-empire | GraphSAGE | normalized_minus_raw | -0.57013 | 0.0428526 | -0.644042–-0.521335 | 0 / 0 / 10 |
| Roman-empire | H2GCN | centered_scaled_minus_normalized | 0.527261 | 0.00612514 | 0.513155–0.535264 | 10 / 0 / 0 |
| Roman-empire | H2GCN | centered_scaled_minus_raw | -0.00139287 | 0.00258267 | -0.00641167–0.00154763 | 4 / 1 / 5 |
| Roman-empire | H2GCN | normalized_minus_raw | -0.528654 | 0.00731237 | -0.539244–-0.512713 | 0 / 0 / 10 |
| Roman-empire | LINKX | centered_scaled_minus_normalized | 0.170551 | 0.0413399 | 0.11143–0.253151 | 10 / 0 / 0 |
| Roman-empire | LINKX | centered_scaled_minus_raw | -0.0014371 | 0.0089446 | -0.0187928–0.0148132 | 3 / 0 / 7 |
| Roman-empire | LINKX | normalized_minus_raw | -0.171988 | 0.0440514 | -0.256467–-0.109883 | 0 / 0 / 10 |
| Roman-empire | GPR-GNN | centered_scaled_minus_normalized | 0.564493 | 0.0036356 | 0.56069–0.571966 | 10 / 0 / 0 |
| Roman-empire | GPR-GNN | centered_scaled_minus_raw | -0.0019014 | 0.00244026 | -0.00596946–0.00154763 | 2 / 0 / 8 |
| Roman-empire | GPR-GNN | normalized_minus_raw | -0.566394 | 0.00430459 | -0.574176–-0.56069 | 0 / 0 / 10 |
| Amazon-ratings | MLP | centered_scaled_minus_normalized | 0.0585052 | 0.00688556 | 0.049418–0.0692261 | 10 / 0 / 0 |
| Amazon-ratings | MLP | centered_scaled_minus_raw | 0.00804574 | 0.00494163 | -0.000816852–0.0136819 | 9 / 0 / 1 |
| Amazon-ratings | MLP | normalized_minus_raw | -0.0504595 | 0.00502831 | -0.0571779–-0.0418624 | 0 / 0 / 10 |
| Amazon-ratings | GCN | centered_scaled_minus_normalized | 0.0771697 | 0.00356885 | 0.0737186–0.0845416 | 10 / 0 / 0 |
| Amazon-ratings | GCN | centered_scaled_minus_raw | 0.00753523 | 0.00343117 | 0.00306311–0.0149071 | 10 / 0 / 0 |
| Amazon-ratings | GCN | normalized_minus_raw | -0.0696345 | 0.00448222 | -0.0745354–-0.0596283 | 0 / 0 / 10 |
| Amazon-ratings | GAT | centered_scaled_minus_normalized | 0.0740862 | 0.00413912 | 0.067184–0.0804574 | 10 / 0 / 0 |
| Amazon-ratings | GAT | centered_scaled_minus_raw | -0.00149071 | 0.00451112 | -0.0122524–0.0026547 | 4 / 0 / 6 |
| Amazon-ratings | GAT | normalized_minus_raw | -0.0755769 | 0.00520399 | -0.0867878–-0.0675924 | 0 / 0 / 10 |
| Amazon-ratings | GraphSAGE | centered_scaled_minus_normalized | 0.121299 | 0.00300893 | 0.11701–0.124974 | 10 / 0 / 0 |
| Amazon-ratings | GraphSAGE | centered_scaled_minus_raw | 0.0183786 | 0.00525851 | 0.00939351–0.0261385 | 10 / 0 / 0 |
| Amazon-ratings | GraphSAGE | normalized_minus_raw | -0.10292 | 0.00552659 | -0.110884–-0.0912804 | 0 / 0 / 10 |
| Amazon-ratings | H2GCN | centered_scaled_minus_normalized | 0.106494 | 0.00592571 | 0.0957729–0.114152 | 10 / 0 / 0 |
| Amazon-ratings | H2GCN | centered_scaled_minus_raw | 0.00876046 | 0.00437599 | -0.00122523–0.0134777 | 9 / 0 / 1 |
| Amazon-ratings | H2GCN | normalized_minus_raw | -0.0977333 | 0.00513567 | -0.103737–-0.0888299 | 0 / 0 / 10 |
| Amazon-ratings | LINKX | centered_scaled_minus_normalized | 0.00600368 | 0.00304444 | 0.00204206–0.0104145 | 10 / 0 / 0 |
| Amazon-ratings | LINKX | centered_scaled_minus_raw | 0.00114354 | 0.00470237 | -0.00714725–0.0075556 | 5 / 0 / 5 |
| Amazon-ratings | LINKX | normalized_minus_raw | -0.00486014 | 0.00386353 | -0.0120482–0 | 0 / 1 / 9 |
| Amazon-ratings | GPR-GNN | centered_scaled_minus_normalized | 0.095487 | 0.00613919 | 0.0833163–0.102103 | 10 / 0 / 0 |
| Amazon-ratings | GPR-GNN | centered_scaled_minus_raw | 0.00475802 | 0.00304877 | -0.00245047–0.00878087 | 9 / 0 / 1 |
| Amazon-ratings | GPR-GNN | normalized_minus_raw | -0.090729 | 0.00486304 | -0.0974066–-0.079028 | 0 / 0 / 10 |

## Controls

Control mismatch counts and magnitudes are disclosed for review; small metric drift is not silently treated as agreement.

| Control group | Dataset | Condition | Accuracy mismatches | Max | Loss mismatches | Max | Trial mismatches | Feature-hash mismatches |
|---|---|---|---:|---:|---:|---:|---:|---:|
| preprocessing_280 | Roman-empire | raw | 46 | 0.00972807 | 60 | 0.938138 | 8 | 0 |
| preprocessing_280 | Roman-empire | normalize_features | 15 | 0.24696 | 45 | 4.7462 | 8 | 0 |
| preprocessing_280 | Amazon-ratings | raw | 62 | 0.00673884 | 62 | 1.3052 | 22 | 0 |
| preprocessing_280 | Amazon-ratings | normalize_features | 67 | 0.00633043 | 48 | 1.22752 | 4 | 0 |
| mlp_60 | Roman-empire | raw | 0 | 0 | 6 | 1.19209e-07 | 0 | 0 |
| mlp_60 | Roman-empire | normalize_features | 0 | 0 | 7 | 7.15256e-07 | 0 | 0 |
| mlp_60 | Roman-empire | normalize_centered_scaled | 0 | 0 | 4 | 1.19209e-07 | 0 | 0 |
| mlp_60 | Amazon-ratings | raw | 4 | 2.98023e-08 | 2 | 1.19209e-07 | 0 | 0 |
| mlp_60 | Amazon-ratings | normalize_features | 10 | 2.98023e-08 | 8 | 1.19209e-07 | 0 | 0 |
| mlp_60 | Amazon-ratings | normalize_centered_scaled | 5 | 2.98023e-08 | 5 | 2.38419e-07 | 0 | 0 |

## Integrity

- Config digest: `ed0432fad313d371b3b5fbb6863fa024d05eb6688493c1023cdce653e8834692`
- Source commit: `c48c34ee160f7f6ab62bfcaf10b5e90f40568466`
- Split bindings: 20
- Raw NPZ data, train-fitted transforms, transformed hashes, and train/validation class counts were reconstructed.
- Public source visibility was not verified.
- New test evaluations and test metrics are absent.
- The prior MLP control was verified against its historical source commit and fingerprint.
