# MLP optimization diagnostic

Validated 200 records for `posthoc_mlp_optimization_v1`.

This is a validation-only post hoc diagnostic. It reports paired descriptive contrasts; it makes no population significance claim and does not use test predictions.

## Selected validation summaries

| Dataset | Condition | Decay | Val accuracy mean (sample SD) | Val loss mean (sample SD) | Balanced accuracy mean | Dominant prediction fraction range | All validation predictions training-majority |
|---|---|---|---:|---:|---:|---:|---:|
| Roman-empire | raw | wd_5e-4 | 0.664426 (0.0045785) | 1.02455 (0.014202) | 0.514717 | 0.273049–0.282556 | 0/10 |
| Roman-empire | raw | wd_0 | 0.666107 (0.00447774) | 1.04036 (0.0274575) | 0.524011 | 0.214681–0.292947 | 0/10 |
| Roman-empire | normalize_features | wd_5e-4 | 0.167544 (0.0420314) | 2.61333 (0.000245249) | 0.0667223 | 0.711033–1 | 5/10 |
| Roman-empire | normalize_features | wd_0 | 0.216648 (0.0441295) | 2.60162 (0.0267072) | 0.086492 | 0.682954–0.996463 | 0/10 |
| Roman-empire | normalize_scaled | wd_5e-4 | 0.412514 (0.0161482) | 2.00191 (0.0701915) | 0.208871 | 0.375636–0.629228 | 0/10 |
| Roman-empire | normalize_scaled | wd_0 | 0.352089 (0.0315486) | 2.2093 (0.138062) | 0.156761 | 0.58302–0.763874 | 0/10 |
| Roman-empire | normalize_centered | wd_5e-4 | 0.206014 (0.0457079) | 2.61743 (0.00861045) | 0.0833627 | 0.710811–1 | 1/10 |
| Roman-empire | normalize_centered | wd_0 | 0.306522 (0.0816481) | 2.34816 (0.358783) | 0.137367 | 0.504532–0.680522 | 0/10 |
| Roman-empire | normalize_centered_scaled | wd_5e-4 | 0.66447 (0.00442822) | 1.0244 (0.0111579) | 0.516562 | 0.265311–0.286093 | 0/10 |
| Roman-empire | normalize_centered_scaled | wd_0 | 0.665598 (0.00493525) | 1.03858 (0.0207563) | 0.519888 | 0.254035–0.304444 | 0/10 |
| Amazon-ratings | raw | wd_5e-4 | 0.41844 (0.00502831) | 1.35076 (0.00453278) | 0.261342 | 0.636104–0.677149 | 0/10 |
| Amazon-ratings | raw | wd_0 | 0.445354 (0.00652216) | 1.48614 (0.0322081) | 0.309566 | 0.532571–0.556259 | 0/10 |
| Amazon-ratings | normalize_features | wd_5e-4 | 0.36798 (0) | 1.41088 (4.85047e-06) | 0.2 | 1–1 | 10/10 |
| Amazon-ratings | normalize_features | wd_0 | 0.36798 (0) | 1.41089 (1.23636e-05) | 0.2 | 1–1 | 10/10 |
| Amazon-ratings | normalize_scaled | wd_5e-4 | 0.368001 (6.45755e-05) | 1.41187 (0.00510054) | 0.200015 | 0.999592–1 | 9/10 |
| Amazon-ratings | normalize_scaled | wd_0 | 0.36798 (0) | 1.41047 (0.000254683) | 0.2 | 1–1 | 10/10 |
| Amazon-ratings | normalize_centered | wd_5e-4 | 0.36798 (0) | 1.41088 (2.23731e-06) | 0.2 | 1–1 | 10/10 |
| Amazon-ratings | normalize_centered | wd_0 | 0.369594 (0.00235297) | 1.39992 (0.0048849) | 0.202062 | 0.945885–1 | 4/10 |
| Amazon-ratings | normalize_centered_scaled | wd_5e-4 | 0.426486 (0.00688556) | 1.3429 (0.00485985) | 0.269629 | 0.627323–0.658771 | 0/10 |
| Amazon-ratings | normalize_centered_scaled | wd_0 | 0.459261 (0.00667075) | 1.46133 (0.0224596) | 0.331057 | 0.514805–0.544415 | 0/10 |

## Validation contrasts

| Dataset | Decay | Contrast | Accuracy mean difference | Positive / zero / negative |
|---|---|---|---:|---:|
| Roman-empire | wd_5e-4 | normalized_minus_raw | -0.496883 | 0 / 0 / 10 |
| Roman-empire | wd_5e-4 | scaled_minus_normalized | 0.24497 | 10 / 0 / 0 |
| Roman-empire | wd_5e-4 | centered_minus_normalized | 0.03847 | 7 / 2 / 1 |
| Roman-empire | wd_5e-4 | centered_scaled_minus_scaled | 0.251957 | 10 / 0 / 0 |
| Roman-empire | wd_5e-4 | centered_scaled_minus_centered | 0.458457 | 10 / 0 / 0 |
| Roman-empire | wd_0 | normalized_minus_raw | -0.449458 | 0 / 0 / 10 |
| Roman-empire | wd_0 | scaled_minus_normalized | 0.135441 | 10 / 0 / 0 |
| Roman-empire | wd_0 | centered_minus_normalized | 0.089874 | 7 / 0 / 3 |
| Roman-empire | wd_0 | centered_scaled_minus_scaled | 0.313509 | 10 / 0 / 0 |
| Roman-empire | wd_0 | centered_scaled_minus_centered | 0.359076 | 10 / 0 / 0 |
| Roman-empire | weight decay | raw (zero minus 5e-4) | 0.0016803 | 7 / 0 / 3 |
| Roman-empire | weight decay | normalize_features (zero minus 5e-4) | 0.0491046 | 9 / 0 / 1 |
| Roman-empire | weight decay | normalize_scaled (zero minus 5e-4) | -0.0604245 | 1 / 0 / 9 |
| Roman-empire | weight decay | normalize_centered (zero minus 5e-4) | 0.100509 | 10 / 0 / 0 |
| Roman-empire | weight decay | normalize_centered_scaled (zero minus 5e-4) | 0.00112756 | 8 / 0 / 2 |
| Roman-empire | interaction | uncentered_scale_effect_zero_minus_nonzero_validation_accuracy | -0.109529 | 1 / 0 / 9 |
| Roman-empire | interaction | centered_scale_effect_zero_minus_nonzero_validation_accuracy | -0.099381 | 0 / 0 / 10 |
| Amazon-ratings | wd_5e-4 | normalized_minus_raw | -0.0504595 | 0 / 0 / 10 |
| Amazon-ratings | wd_5e-4 | scaled_minus_normalized | 2.04206e-05 | 1 / 9 / 0 |
| Amazon-ratings | wd_5e-4 | centered_minus_normalized | 0 | 0 / 10 / 0 |
| Amazon-ratings | wd_5e-4 | centered_scaled_minus_scaled | 0.0584848 | 10 / 0 / 0 |
| Amazon-ratings | wd_5e-4 | centered_scaled_minus_centered | 0.0585052 | 10 / 0 / 0 |
| Amazon-ratings | wd_0 | normalized_minus_raw | -0.0773739 | 0 / 0 / 10 |
| Amazon-ratings | wd_0 | scaled_minus_normalized | 0 | 0 / 10 / 0 |
| Amazon-ratings | wd_0 | centered_minus_normalized | 0.00161323 | 6 / 4 / 0 |
| Amazon-ratings | wd_0 | centered_scaled_minus_scaled | 0.0912804 | 10 / 0 / 0 |
| Amazon-ratings | wd_0 | centered_scaled_minus_centered | 0.0896671 | 10 / 0 / 0 |
| Amazon-ratings | weight decay | raw (zero minus 5e-4) | 0.0269144 | 10 / 0 / 0 |
| Amazon-ratings | weight decay | normalize_features (zero minus 5e-4) | 0 | 0 / 10 / 0 |
| Amazon-ratings | weight decay | normalize_scaled (zero minus 5e-4) | -2.04206e-05 | 0 / 9 / 1 |
| Amazon-ratings | weight decay | normalize_centered (zero minus 5e-4) | 0.00161323 | 6 / 4 / 0 |
| Amazon-ratings | weight decay | normalize_centered_scaled (zero minus 5e-4) | 0.0327752 | 10 / 0 / 0 |
| Amazon-ratings | interaction | uncentered_scale_effect_zero_minus_nonzero_validation_accuracy | -2.04206e-05 | 0 / 9 / 1 |
| Amazon-ratings | interaction | centered_scale_effect_zero_minus_nonzero_validation_accuracy | 0.0311619 | 10 / 0 / 0 |

## Original preprocessing controls

| Dataset | Condition | Accuracy exact mismatches | Accuracy max absolute difference | Loss exact mismatches | Loss max absolute difference | Trial mismatches | Feature hash mismatches |
|---|---|---:|---:|---:|---:|---:|---:|
| Roman-empire | raw | 0 | 0 | 0 | 0 | 0 | 0 |
| Roman-empire | normalize_features | 0 | 0 | 0 | 0 | 0 | 0 |
| Amazon-ratings | raw | 0 | 0 | 0 | 0 | 0 | 0 |
| Amazon-ratings | normalize_features | 0 | 0 | 0 | 0 | 0 | 0 |

## Integrity

- Config digest: `7bacfb01db8f50439d01e0106b2df40acaba0a15a09831a07039a1f7e4c90218`
- Source commit: `3163a5dcb04044c944635fccfb86e7bdd2666c3c`
- Original split bindings: 20
- Raw NPZ checksums and all transformed feature hashes were reconstructed.
- Current executable source snapshot matched the manifest: True.
- Public source visibility was not verified.
- Test evaluations: 0; test metrics are absent.
