# Evidence integration, 4 October 2026

This revision preserves the frozen benchmark and title. It integrates completed post-hoc input retraining, the MLP search extension, published-statistic calibration, and existing-record sensitivity analyses. No model is trained or tested by the manuscript revision or reader reconstruction.

## Distinct evidence scopes

| Record set | Scope | Execution source |
|---|---|---|
| Frozen benchmark | 770 model records, 110 decision units | Original release, unchanged |
| Input retraining | 1,540 records, 6,160 trials, two inputs | eb43adfe7c366154170526f30d9ffe2477a70436 |
| MLP extension | 220 units, 24 trials each, 4,400 new trials plus 880 inherited | 51f168217e5b55fc190dd3fd885878408df43fe5 |
| Input sensitivity | 336 condition/portfolio/exclusion/fallback cases, four-trial records only | 51f168217e5b55fc190dd3fd885878408df43fe5 |

The manuscript/source commit is separate from these execution revisions. Documentation does not imply experiments were rerun on the documentation commit. Resource and reliability evidence from aef8b87a6eb7bf6ea4c3fd1482b0a6882dc176b0 remains historical evidence for its tested scope.

## Material changes

| Location | Before | After and reason |
|---|---|---|
| Abstract and conclusion | Mainly the fixed-rule comparison and an aggregate extension result | Two inputs and 24-trial MLP results are identified as separate, post-hoc records. The default-fallback comparison is limited to all 11 datasets. |
| Calibration interpretation | Adjusted homophily 0.15 versus always-graph 0.53 without its gain distribution | Eleven saved CS/24 folds show all gain comes from Cornell and Wisconsin; nine datasets tie. This narrows the positive finding. |
| Calibration protocol | Shared LODO label could imply identical threshold search | Ordinary homophily keeps its 0.01 grid; published statistics use calibration-fold midpoints. These are complete policy comparisons, not isolated effects of the statistic. |
| New Appendix E material | No integrated reader account of completed new records | All-policy, input/budget, interval, fallback/exclusion, and calibration-fold displays with explicit source scopes. |
| Appendix C | “the evaluated graph statistics are inadequate selection rules” | Limited to fixed threshold policies in the original full portfolio, consistent with the calibrated-statistic result. |
| Budget explanation | Regret change attributed to resulting targets | Regret changes with selected test accuracy and oracle values; target changes affect selection accuracy. |
| Coverage curve | Confidence ordering without tied-score details | Dataset name then seed breaks confidence ties. Intermediate tied-score points are ranking points, not pure threshold policies. |
| Applied use | General advice to audit diagnostics | Concrete inputs, references, headroom, action/dataset loss localization, and a worked frozen-record example. |
| Data/code availability | An assembling archive described through earlier paths | Actual delivered layout, separate record sets, checksums and metadata transformations; score reconstruction distinguished from checkpoint replay. |

The four-trial sensitivity includes a local reversal. With N, graph fallback and Chameleon/Squirrel exclusion, Combined regret is 0.308 against always-graph 0.494. The analogous CS result remains worse. These results cannot support an “every configuration” claim and do not evaluate the 24-trial joint sensitivity.

## Review boundary and unresolved limits

Three isolated mock reviewers examined the same immutable 49-page candidate before the final sensitivity and fold-readout presentation. Their reports are frozen; the final 51-page manuscript is a subsequent author revision, not three new accept recommendations. Reports and the separate editorial consistency audit remain local review evidence.

The main remaining scientific and editorial risks are incremental originality, the small fixed dataset collection, positive calibrated gains concentrated in two small graphs, and unequal architectural breadth despite equal trial counts. Filtered graph versions and 24-trial joint fallback/exclusion effects remain untested. No universal selector advantage, computational savings, or general failure of graph diagnostics is claimed.

## Verification and delivery

The full local suite is run on an exact, hash-indexed export of tracked and intended new source files. This avoids Windows path limits caused by ignored historical output snapshots without dropping any assertion or test. The first in-place failure and both export logs are retained.

The current manuscript is compiled with the existing Tectonic runtime and checked for unresolved citations/references, duplicate labels, and overfull boxes. The unchanged TMLR manuscript remains part of CI. CI additionally builds this Applied Intelligence source within the existing manuscript job; the three required job names remain unchanged.

The delivered record reader uses the executed analysis functions to reconstruct both budgets, both inputs, all 63 portfolios, and the published-statistic policies. Its local acceptance is run through the existing monitored control-phase controller using the latest closed budget handoff. Its execution receipt, package digest, same-SHA CI, and resulting closed budget receipt are archived separately after completion.

Full checkpoint tensors and private cumulative ledgers remain in author storage. Their indexes and provenance are supplied, but the journal archive promises reconstruction from saved scores, not independent checkpoint replay. No journal submission is performed by this revision.

## Exact-replay platform qualification

The first Ubuntu CI run on 528baa4 failed two strict decision-dictionary assertions and the same existing guard in the fallback summarizer. A separate Ubuntu replay located 30 confidence-only differences across 20 of 110 units (maximum absolute difference 1.11e-16); all 990 actions, rank orders and tied-score groups matched exactly. Test-field perturbations left complete recomputed dictionaries exactly unchanged in that runtime. The scorer is byte-identical to the accepted execution source.

No assertion, tolerance, scorer or saved result was changed. The two archival-record CI jobs now run on Windows, the source platform of those records; the manuscript build remains on Ubuntu. Linux bitwise replay remains unsupported, rather than being declared repaired. The failed CI and its diagnosis are retained. This change establishes the platform scope of exact reconstruction and does not justify changing numerical or scientific conclusions.
