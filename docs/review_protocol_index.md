# Guide to the archived protocol documents

This guide was prepared for the pre-review revision. It is explanatory review
documentation, not a new protocol, a new registration, or a modification of the
original decision rule.

The anonymous supplement includes the following current archived versions:

- `preregistration_diagnostic_benchmark.md`: model selection, information
  boundaries, the nine policies, dataset-level inference, and the paper-level
  decision rule.
- `protocol_amendment_prospective_v2.md`: the dataset eligibility amendment
  following the Texas split-feasibility failure; the earlier records are
  excluded from the replacement run.
- `protocol_amendment_prospective_v2_1.md`: the Squirrel execution-time amendment,
  its blinding boundary, and immutable-resume conditions.

These are the versions retained in the current source archive. They include
later provenance and label-scope clarifications; they are not presented as
byte-identical pre-outcome registration snapshots. The source archive retains
their revision history. The supplement permits inspection of the recorded
rules and amendments, but does not independently authenticate their historical
commit times or establish whole-project preregistration. Historical terms such
as "confirmatory" and "reproducibility" remain in these documents as written;
the manuscript states the present interpretation and limitations.

## Comparisons and interval interpretation

The statistical specification compares each of the eight baseline policies
separately with Combined. It specifies 95% percentile-bootstrap intervals of
dataset-mean regret differences and Holm correction of the eight sign-flip
p-values. The reported bootstrap intervals are not multiplicity-adjusted.
A positive baseline-minus-Combined difference favors Combined.

The paper-level wording requires reduced full-set regret, no improvement
obtained only by reduced coverage, and an interval excluding zero. It does not
identify always-graph as a uniquely predeclared comparator, nor specify whether
all eight baselines must meet the criterion. The revised manuscript retains
this ambiguity and uses always-graph as its focal reference, rather than
introducing a retrospective success rule. Its observed comparison with
always-graph fails the regret-reduction condition irrespective of the Holm
resolution limitation.

## Review-copy provenance

Only the declared anonymization substitutions are applied to these documents
when packaging review copies. Source revisions and artifact names use stable
aliases, with their unredacted correspondence retained in the author archive.
The aliases are not Git revisions. Delivered-file hashes verify the review
copies, not the original bytes or the historical timing of the protocol.
No thresholds, numerical settings, outcomes, or historical decision-rule
wording are rewritten for this supplement.
