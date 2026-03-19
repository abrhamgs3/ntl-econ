# Revision Notes

## Major changes and why

1. Reframed the paper around validated findings only.
The previous draft's headline claims about "robust evidence" of leader birth-region favoritism, placebo significance, and strong causal interpretation are not supported by the saved fixed-effects outputs in the repository. The revised manuscript now treats favoritism as an open empirical question rather than a demonstrated result.

2. Separated results from discussion.
The revised manuscript reports numerical findings in the Results section and moves interpretation, implications, and limits of inference to the Discussion and Limitations sections.

3. Updated the quantitative claims to match the active files.
The manuscript now reports the values that are actually supported by the repository outputs:
- National NTL-log GDP correlation: 0.914
- Log-log elasticity: 1.887
- National regression R^2: 0.933
- Durbin-Watson: 0.713
- Regional Gini: 0.715 (1992), 0.513 (2018), 0.526 (2024)
- Fixed-effects leader birth-region coefficient: 0.005 with p-values about 0.975 to 0.995

4. Removed unsupported causal and placebo language.
The current repository does not contain a saved placebo result that matches the draft's stated p-values, and the event-study implementation is not adequate for strong pre-trend or dynamic-treatment claims. Those assertions were removed.

5. Tightened the literature review.
The revised manuscript keeps foundational references and adds recent work on NTL measurement limits and recent favoritism research so the paper is grounded in the current literature without overstating what the Ethiopia application can show.

6. Clarified the contribution.
The revised paper's contribution is now a careful account of regional inequality and the limits of the current favoritism evidence, which is much more defensible for journal review than the prior over-claimed version.

## Assumptions made

1. The active manuscript should be aligned to the repository's current processed files rather than to older external files referenced in comments or hard-coded legacy paths.

2. The effective study period is treated as 1992-2024 because both the active national workbook and the active regional panel contain observations through 2024.

3. The fixed-effects birth-region summaries saved in `outputs/results/regional/` and `data/tabular/` are treated as the most reliable archived evidence for the favoritism claim because they are explicit outputs, whereas several alternative scripts contain unresolved inconsistencies.

4. The active regional panel is treated as covering 13 regional units because that is what the processed CSV currently contains.

## Inconsistencies between code/results and manuscript claims

1. The draft claims that NTL-GDP validation yields `R^2 = 0.98`, but the saved national summary reports `R^2 = 0.933` and correlation `0.914`.

2. The draft claims that the Gini coefficient fell from about `0.4` to below `0.1`, but the active regional panel yields values around `0.715` in 1992 and `0.526` in 2024.

3. The draft claims leaders' birth regions received `32%` higher NTL and that this is strongly significant, but the saved fixed-effects panel summaries report a coefficient of about `0.005` with p-values near `0.975` to `0.995`.

4. The draft claims placebo significance (`p = 0.001` or `p < 0.01`), but no matching saved placebo output was found in the repository.

5. The manuscript's national validation equation includes a time trend, but `scripts/analyze_ntl_gdp.py` estimates a simple log-log OLS regression without a time trend.

6. Treatment coding is not consistent across files:
- `extract_ntl_panel.py` creates `Leader_Birth_Region`
- `data/tabular/regional_ntl_panel_data.csv` contains `Leader_region`
- `scripts/analyze_event_study.py` uses a different coding rule and treatment window

7. The current event-study implementation does not provide a credible basis for pre-trend testing because treated observations are constructed during leadership periods rather than from a clean pre-treatment setup for treated units.

## Files revised

- `Manuscript_NTL_revised.tex`
