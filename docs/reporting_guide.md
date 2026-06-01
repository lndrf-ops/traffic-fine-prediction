# Reporting Guide

The deliverable is a **20–35 page academic report**. Claude writes report
content directly (in conversation or in a `.md`/`.tex` file) — scripts do
**not** need to emit `tX_report.json` unless metrics are consumed by another
script (e.g. evaluation results for a comparison table).

## When a script SHOULD write a JSON artifact

Only when a downstream script reads it: e.g. `t6_evaluate.py` writing
`outputs/reports/evaluation_results.json` so a comparison script can load it.
For everything else, report text is written by Claude directly.

## Writing Style for AI-Generated Text

- **Academic, third-person**: "the model achieves..." not "we got..."
- **Hedged claims**: "results suggest", "the data indicates" — not "we proved"
- **Justify every choice**: what was done, why, what alternatives were considered
- **Cite when possible**: prefer references from `docs/references.md`
- **No marketing language**: avoid "powerful", "cutting-edge", "state-of-the-art"

## Figures

- 300 DPI PNG, descriptive filenames (`outputs/plots/t4_dotted_chart_by_year.png`)
- Caption drafted as a docstring in the generating function
- Color-blind friendly palettes (matplotlib `viridis`, `cividis`)

## Reproducibility Disclosure

The report must disclose:

- Seeds used (numpy, torch, random)
- Train/val/test split sizes and date boundaries
- MPS non-determinism caveat (<1% metric variance expected)
- Package versions (`pip freeze` output as appendix)