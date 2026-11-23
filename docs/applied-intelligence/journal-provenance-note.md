# Appendix-summary provenance

The retained historical appendix summary declared source audit SHA-256
`4bf33130bad67d2a6330abb1e3efb2ebf51b6fc6e50984b8c67d73610df0c131`.
The delivered `diagnostic_audit.json` has SHA-256
`3c5aa160ac420536fd3f0be58b880fc99b3fe5912215a54d149e4cb62905b3bd`.
The earlier declared digest does not authenticate the delivered audit, and its
origin remains unresolved.

The historical summary is retained unchanged at
`provenance/reviewer_appendix_summary.pre-2026-09-28.json` (8,013 bytes;
SHA-256 `d826f13959469402862fcd96c4e39b8255bc9a4b75a60192dd1395793d28e7d2`).
The current derivative binds the actual delivered audit and retains the prior
unverified digest in its provenance. The audit itself has not been rewritten.

`scripts/summarize_reviewer_appendix.py --metadata-summary` recomputes all
scientific fields from the delivered audit, retaining only four descriptive
counts per dataset (nodes, edges, classes and features) from the historical
summary. It rejects any difference in the complete scientific object. The
current and historical scientific values are exactly equal; only provenance
changes. This reconstructs the appendix from its 110-unit audit, without
independently reloading dataset counts or reaggregating all 770 model records.

See [the reconstruction index](journal-reconstruction-index.md) for the input
hierarchy, commands and requirements for deeper reconstruction.
