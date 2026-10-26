# Reader study tool and analysis (pilot; Supplementary Table 25)

* `reader_study_tool.html` — offline reading tool (open in Chrome or Edge; no network). It shows the stored images of each
  case, with or without the FMO draft according to the crossover block, records the eight-feature structured report, the
  decision on the draft (adopt unchanged, edit, or rewrite) and the time from display to submission, and exports one CSV per
  reader. The interface is in Chinese.
* `manifest.js` — the 60 cases (order, App_No, block, image file names, eight-feature FMO draft). Place the images as
  `images/<App_No>/<file>` next to the tool; they are the stored images of these patients in the figshare archive
  `FMS/TestImage_all.rar` (center-cropped there; the study used the uncropped frames).
* `analyze_reader_study.py` — reproduces Supplementary Table 25 from the reader CSVs and the case reference table:

      python analyze_reader_study.py <folder with reader CSVs> --key cases_reference.csv

  The reader results (`reader1_results.csv`, `reader2_results.csv`) and `cases_reference.csv` (final diagnosis, panel A
  expert majority and draft per case) are in the figshare record, `07_reader_study`.

Design: two senior physicians (more than 20 years' experience, not authors, no role in annotation) read 60 panel A cases in a
crossover design (reader 1 with drafts in block A and without in block B, reader 2 the reverse); case sampling seed 20261024.
