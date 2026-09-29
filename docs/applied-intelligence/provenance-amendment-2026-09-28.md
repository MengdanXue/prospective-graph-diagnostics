# Appendix-summary provenance amendment — 2026-09-28

Review issue L5 concerned the source digest in
`results/diagnostic/route_a_prospective_v2/analysis/reviewer_appendix_summary.json`.
The historical file declared audit SHA-256
`4bf33130bad67d2a6330abb1e3efb2ebf51b6fc6e50984b8c67d73610df0c131`,
while the delivered `diagnostic_audit.json` has SHA-256
`3c5aa160ac420536fd3f0be58b880fc99b3fe5912215a54d149e4cb62905b3bd`.
The former is not a checksum of the delivered audit.

## Evidence and unresolved origin

The audit was introduced at commit
`79c2fc86b47e34eb5ef6720826618e25d1d2bcd3`. Git-object bytes at that commit,
at summary introduction `9de586ccbdf15c3e26d044d9d2cc6f49acb10017`, and at the
Applied Intelligence baseline `ef48699cdfdce8137b3fc560ff838977751e6e52`
are identical: 357570 bytes, the `3c5aa160…` digest above, and no CRLF pairs.
The summary already declared `4bf33130…` when first added at `9de586c`.
Thus the discrepancy is historical and is not explained by checkout line endings.

A bounded read-only search examined the 36 local files named
`diagnostic_audit.json` under the sibling work directory and the audit/summary
members of the retained September 8 and September 9 `source.zip` archives.
No audit with the `4bf33130…` digest was found. The later anonymous package has
an explicitly aliased audit digest, `2a488368…`, and one retained reconstruction
has `23b8373f…`; neither establishes the missing input. A private source audit,
redaction, aliasing or runtime metadata difference is a possible explanation,
not a verified mapping. This amendment does not claim to identify the original
bytes or retrospectively authenticate the earlier source declaration.

## Preserved original and regenerated derivative

The historical summary is preserved byte for byte at
[`provenance/reviewer_appendix_summary.pre-2026-09-28.json`](provenance/reviewer_appendix_summary.pre-2026-09-28.json).
It is identical to its introducing Git blob: 8013 bytes, SHA-256
`d826f13959469402862fcd96c4e39b8255bc9a4b75a60192dd1395793d28e7d2`.

The revised `summarize_reviewer_appendix.py --metadata-summary` path invokes
the existing `summarize_audit` calculation on the delivered audit. It reads
only four retained per-dataset descriptive counts (nodes, canonical undirected
edges, classes and features) from the archived summary. Every other scientific
field is recomputed. The program rejects any difference between the resulting
complete scientific object and the historical summary with `provenance` removed.
The output was written to a fresh path, compared, and then installed as the
current derivative; the source audit and historical copy were not rewritten.

The regenerated summary is 8483 bytes, SHA-256
`22c66d32b8ecd73067a6c915fc1a20369ddd004eea1df25256066a7678957eb3`.
All fields except `provenance` are exactly equal, including all 11 dataset rows,
action counts, regrets, overall and fallback statistics, schema, status and
record counts. Its provenance now records the actual input audit digest, the
retained metadata file and digest, the previous unverified digest, and the
exact-comparison rule. Dataset sizes were retained, not reloaded from data.

The regression tests check the delivered summary's actual byte binding,
historical-copy digest, exact regeneration, rejection of the stale original
binding, rejection of a whitespace-only audit byte change, rejection of an
altered scientific input, and output overwrite refusal. Run:

```text
python -m unittest tests.test_reviewer_appendix_summary -v
```

All five tests passed during this repair. This is a reconstruction from the
retained 110-unit audit, not independent reconstruction of all 770 model
records, raw-data verification, training, or new test-set evaluation. The
[reconstruction index](reconstruction-index.md) separates those requirements.
