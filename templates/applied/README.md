# Official template provenance

- Target: Applied Intelligence, Springer Nature.
- Guide: https://link.springer.com/journal/10489/submission-guidelines
- Template hub: https://www.springernature.com/gp/authors/campaigns/latex-author-support
- Official archive: https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12
- Retrieved: 2026-09-28; December 2024 v3.1 package.
- Root `sn-jnl.cls` and `sn-basic.bst` are unmodified copies from that archive. Their copyright and license notices are retained.

The journal guide recommends the Springer Nature template and sn-basic bibliography style with numbered bracket citations. The same page retains a legacy smallcondensed instruction whose linked archive is unavailable. The current supported package is therefore used with `[pdflatex,sn-basic,Numbered]`.

The working source remains modular. `scripts/build_applied_intelligence_package.py` expands it into a single .tex with all figures, class, bibliography and style files at the same directory level, following the upload instructions. Changes to table spacing and PDF link destinations are in manuscript sources, never in publisher style files.

The guide requires a title page with author, affiliation and contact details. Its generic conditional discussion of double-anonymous review is not evidence of this journal's specific review model; verify any portal-specific anonymization requirement before upload.
