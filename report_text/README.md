# Report-text feature extraction

`report_text_extract.py` pre-fills, for each of the nine features, whether the original report text describes it as
present, explicitly absent, not mentioned, or needs review, using clause-level keywords and negation words; it reads only
two columns (`申请号`, `描述` = the report's description section) and nothing else (no images, pathology, labels, model output
or expert readings). In the study, one author then verified every entry against the full report under the written rules in
`RULES_zh.md` (Chinese; Methods, "Report-text feature extraction"). For the analyses, "absent" and "not mentioned" are both
negative.

    python report_text_extract.py <input.xlsx with columns 申请号, 描述> <output.xlsx>

The report text itself is not shared (it may contain identifying information); the program is provided so that the procedure
can be applied to other report collections. The rules and keywords are written for Chinese ultrasound reports.
