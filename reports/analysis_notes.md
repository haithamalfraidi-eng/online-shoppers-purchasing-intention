# Analysis notes

Reuses the original visitor, month, May/November, and IQR questions.

Raw: 12,330 sessions; cleaned: 12,205; removed 125 exact duplicates. Clean conversion: 15.63%.

Product duration IQR upper bound: 3403.387 seconds. Outliers are retained: long sessions can be valid. This full-data bound is descriptive only and never used in model preprocessing.

The original file contains no active missing-value or duplicate removal step. Its duplicate count is confirmed. Figures are now separate and saved to prevent overlapping axes. Raw and deduplicated populations are distinguished.

Conversion differences are associations, not causal effects. Compare both counts and rates; small segments are unstable. May/November engagement and traffic motivate further funnel investigation, not claims about sales uplift. No monetary revenue, costs, or intervention data are available. EDA used the full dataset before modeling; the holdout is therefore not an independently blinded research validation.
