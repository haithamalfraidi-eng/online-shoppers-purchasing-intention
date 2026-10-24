"""Organizes the questions in the preserved original analysis into saved outputs."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import ROOT, MONTHS, load_data, clean_data, conversion_table


def run():
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    raw = load_data()
    df = clean_data(raw)
    audit = {'raw_rows': len(raw), 'clean_rows': len(df), 'duplicates_removed': int(raw.duplicated().sum()),
             'missing_cells': int(raw.isna().sum().sum()), 'raw_conversion': float(raw.Revenue.mean()),
             'clean_conversion': float(df.Revenue.mean())}
    (out / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    for col in ['VisitorType', 'Month', 'Weekend', 'TrafficType']:
        table = conversion_table(df, col)
        if col == 'Month':
            table = table.reindex([m for m in MONTHS if m in table.index])
        table.to_csv(out / f'conversion_by_{col}.csv')
        if col in ['VisitorType', 'Month']:
            fig, ax = plt.subplots(figsize=(8, 4))
            (table.conversion_rate * 100).plot.bar(ax=ax, color='#287d8e', rot=25)
            ax.set(title=f'Conversion by {col} (deduplicated)', ylabel='Purchasing sessions (%)')
            fig.tight_layout()
            fig.savefig(out / f'conversion_by_{col}.png', dpi=150)
            plt.close(fig)
    monthly = conversion_table(df, 'Month').reindex([m for m in MONTHS if m in df.Month.values])
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly.sessions.plot.bar(ax=ax, color='#287d8e', rot=0)
    ax.set(title='Monthly website traffic (deduplicated)', ylabel='Sessions')
    fig.tight_layout()
    fig.savefig(out / 'monthly_traffic.png', dpi=150)
    plt.close(fig)
    subset = df[df.Month.isin(['May', 'Nov'])]
    subset.groupby(['Month', 'VisitorType']).agg(sessions=('Revenue', 'size'),
        conversion_rate=('Revenue', 'mean'), mean_product_pages=('ProductRelated', 'mean')).to_csv(out / 'may_nov_comparison.csv')
    q1, q3 = df.ProductRelated_Duration.quantile([.25, .75])
    upper = float(q3 + 1.5 * (q3-q1))
    df.assign(long_duration=df.ProductRelated_Duration > upper).groupby('long_duration').Revenue.agg(
        sessions='size', purchases='sum', conversion_rate='mean').to_csv(out / 'duration_outliers.csv')
    (out / 'analysis_notes.md').write_text(
        f'# Analysis notes\n\nReuses the original visitor, month, May/November, and IQR questions.\n\n'
        f'Raw: {len(raw):,} sessions; cleaned: {len(df):,}; removed {raw.duplicated().sum()} exact duplicates. '
        f'Clean conversion: {df.Revenue.mean():.2%}.\n\n'
        f'Product duration IQR upper bound: {upper:.3f} seconds. Outliers are retained: long sessions can be valid. '
        'This full-data bound is descriptive only and never used in model preprocessing.\n\n'
        'The original file contains no active missing-value or duplicate removal step. Its duplicate count is confirmed. '
        'Figures are now separate and saved to prevent overlapping axes. Raw and deduplicated populations are distinguished.\n\n'
        'Conversion differences are associations, not causal effects. Compare both counts and rates; small segments are unstable. '
        'May/November engagement and traffic motivate further funnel investigation, not claims about sales uplift. '
        'No monetary revenue, costs, or intervention data are available. EDA used the full dataset before modeling; '
        'the holdout is therefore not an independently blinded research validation.\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    run()
