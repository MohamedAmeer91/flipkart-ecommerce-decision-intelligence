# Decisions

- Extra tracked folders: `docs/` and `tests/` stay in the repository. The brief's layout lists required files and says shared helper files are fine; it does not forbid additional files. `common.py` is the shared helper.
- Line endings: the generator writes CRLF and the brief's MD5 is over those bytes, so `data/records.csv` and `generate_data.py` are `-text` in `.gitattributes`. Output files are written with LF.
- `elbow_k`: the brief defines it for K in {3,4,5}; if no K qualified the code would write 0 (not triggered: result is 3).
- Hierarchical cut: gap ties go to the first (lowest) gap; no ties occur.
- `ensemble_accuracy`: sign 0 would count as wrong; not triggered.
- Memo cost: only the logistic model's Task 2.8 cost is computed; the memo says so explicitly for other models.
- README derived figures (gap as percent of plan, 1.96 x SE margin, extra records needed) are arithmetic on output values, not new graded numbers.
- K-means pass counts (21/15/27) are printed to stdout only as a self-check; they matched.
- Part 4 recomputes the Part 2 split, logistic model and best K by calling the shared code (deterministic) rather than reading files.
