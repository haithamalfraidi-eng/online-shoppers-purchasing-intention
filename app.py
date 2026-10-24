"""Streamlit entry point: python -m streamlit run app.py."""
import hashlib
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from shopper.data import DATA, ROOT, FEATURES, NUMERIC, CATEGORICAL, MONTHS, load_data, clean_data, conversion_table

st.set_page_config(page_title='Shopper Signals', page_icon='🛒', layout='wide')


@st.cache_data
def dataset():
    return clean_data(load_data())


@st.cache_resource
def assets():
    # Load only the trusted artifact shipped in this repository. No uploads.
    bundle = joblib.load(ROOT / 'artifacts' / 'model.joblib')
    if bundle['dataset_sha256'] != hashlib.sha256(DATA.read_bytes()).hexdigest():
        raise ValueError('Dataset changed. Re-run python -m shopper.train.')
    return bundle, json.loads((ROOT / 'artifacts' / 'metrics.json').read_text())


st.title('Shopper Signals')
st.caption('Online shoppers purchasing intention · Portfolio MVP')
st.info('Historical session analysis. Predictions are experimental model scores, not a validated real-time purchase service.')
try:
    df = dataset()
    bundle, report = assets()
except (OSError, ValueError) as exc:
    st.error(f'Project assets unavailable: {exc}. Run python -m shopper.analyze and python -m shopper.train.')
    st.stop()

insights, predict, evaluation, limitations = st.tabs(['Conversion insights', 'Purchase prediction', 'Model evaluation', 'Data & limitations'])

with insights:
    st.subheader('Where sessions become purchases')
    left, right = st.columns(2)
    months = left.multiselect('Month filter', [m for m in MONTHS if m in df.Month.values], default=[m for m in MONTHS if m in df.Month.values])
    visitors = right.multiselect('Visitor filter', sorted(df.VisitorType.unique()), default=sorted(df.VisitorType.unique()))
    view = df[df.Month.isin(months) & df.VisitorType.isin(visitors)]
    if view.empty:
        st.warning('No sessions match these filters. Select at least one month and visitor type.')
    else:
        a, b, c = st.columns(3)
        a.metric('Sessions', f'{len(view):,}')
        b.metric('Purchasing sessions', f'{int(view.Revenue.sum()):,}')
        c.metric('Conversion rate', f'{view.Revenue.mean():.2%}')
        a, b = st.columns(2)
        with a:
            st.markdown('**Conversion by visitor type**')
            table = conversion_table(view, 'VisitorType')
            st.bar_chart(table.conversion_rate.rename('Conversion rate'))
            st.dataframe(table.style.format({'conversion_rate': '{:.2%}'}), width='stretch')
        with b:
            st.markdown('**Monthly traffic**')
            monthly = conversion_table(view, 'Month').reindex([m for m in MONTHS if m in view.Month.values])
            st.bar_chart(monthly.sessions, sort=False)
            st.dataframe(monthly.style.format({'conversion_rate': '{:.2%}'}), width='stretch')
        st.caption('Exact duplicate rows removed. Charts reflect the selected historical cohort; counts are sessions, not monetary sales.')
        st.markdown('**May / November engagement comparison**')
        comparison = view[view.Month.isin(['May', 'Nov'])].groupby(['Month', 'VisitorType']).agg(
            sessions=('Revenue', 'size'), conversion_rate=('Revenue', 'mean'), mean_product_pages=('ProductRelated', 'mean'))
        st.dataframe(comparison, width='stretch')
        st.download_button('Download selected sessions', view.to_csv(index=False), 'selected_sessions.csv', 'text/csv')
        st.markdown('Use these associations to prioritize funnel research. Validate any proposed intervention with an experiment; this dataset cannot estimate conversion uplift.')

with predict:
    st.subheader('Explore a session scenario')
    st.caption('PageValues is excluded. Defaults are training-set medians/modes and are illustrative, not a real customer.')
    training = df.loc[report['split']['train']['indices']]
    with st.form('prediction'):
        inputs = {}
        columns = st.columns(3)
        for i, name in enumerate(FEATURES):
            with columns[i % 3]:
                if name in NUMERIC:
                    count = name in ['Administrative', 'Informational', 'ProductRelated']
                    value = training[name].median()
                    if count:
                        inputs[name] = st.number_input(name, min_value=0, value=int(value), step=1)
                    else:
                        inputs[name] = st.number_input(name, min_value=0.0, max_value=1.0 if name in ['BounceRates', 'ExitRates', 'SpecialDay'] else None,
                                                       value=float(value), format='%.4f')
                else:
                    options = sorted(training[name].unique().tolist())
                    default = training[name].mode().iloc[0]
                    inputs[name] = st.selectbox(name, options, index=options.index(default))
        submitted = st.form_submit_button('Predict purchase intention', type='primary')
    if submitted:
        row = pd.DataFrame([inputs])[FEATURES]
        score = float(bundle['pipeline'].predict_proba(row)[0, 1])
        decision = score >= bundle['threshold']
        st.metric('Estimated purchase score', f'{score:.1%}')
        st.success('Predicted class: purchase' if decision else 'Predicted class: no purchase')
        st.caption(f"Model: {bundle['model_name']} · validation-selected threshold: {bundle['threshold']:.3f}. Scores are not independently calibrated probabilities.")
        outside = [name for name in NUMERIC if inputs[name] > training[name].max() or inputs[name] < training[name].min()]
        if outside:
            st.warning('Outside training range: ' + ', '.join(outside) + '. This estimate is especially unreliable.')

with evaluation:
    st.subheader('Untouched test partition')
    st.caption('60% train / 20% validation / 20% test, stratified, seed 42. Model selected by validation average precision; thresholds selected by validation F1.')
    rows = [{'model': n, **{k: v for k, v in item['test'].items() if k != 'confusion_matrix'}} for n, item in report['models'].items()]
    st.dataframe(pd.DataFrame(rows).set_index('model').style.format('{:.3f}'), width='stretch')
    st.markdown(f"Selected model: **{report['selected_model']}**. PR-AUC is reported as **average precision (AP)**, not trapezoidal area.")
    st.caption(f"Test sessions: {report['split']['test']['rows']:,}; purchase prevalence / random-ranking AP reference: {report['split']['test']['prevalence']:.2%}.")
    curves = json.loads((ROOT / 'artifacts' / 'curves.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, curve in curves.items():
        axes[0].plot(curve['fpr'], curve['tpr'], label=name)
        axes[1].plot(curve['recall'], curve['precision'], label=name)
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].axhline(report['split']['test']['prevalence'], linestyle='--', color='gray')
    axes[0].set(xlabel='False positive rate', ylabel='Recall', title='ROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall')
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('**Selected model confusion matrix** (rows actual, columns predicted)')
    st.dataframe(pd.DataFrame(report['models'][report['selected_model']]['test']['confusion_matrix'], index=['No purchase', 'Purchase'], columns=['No purchase', 'Purchase']))
    with st.expander('PageValues sensitivity — potential leakage, excluded from deployment'):
        st.dataframe(pd.DataFrame({n: {k: v for k, v in result.items() if k != 'confusion_matrix'} for n, result in report['page_values_sensitivity'].items()}).T)
        st.write('A higher score with transaction-derived PageValues does not establish valid prospective predictive power.')

with limitations:
    st.subheader('What this evidence supports')
    st.write('The original CSV and analysis are preserved byte for byte. The modeling copy removes 125 exact duplicate records; legitimate long sessions are retained.')
    st.write('PageValues can encode transaction outcomes or information unavailable at prediction time. Its construction window cannot be audited here, so it is excluded from the deployed pipeline.')
    st.write('Counts, durations, bounce and exit rates may summarize a completed session. Excluding PageValues alone does not establish point-in-time safety. A live system needs timestamped features computed before the decision and an independent future-period evaluation.')
    st.write('The data represents historical sessions over one year, with no exact timestamps, verified user IDs, cart events, costs, or monetary sales. Random splits do not prove future-period or cross-store performance. Prior full-data EDA also limits independence of the holdout.')
    st.write('The target is imbalanced; accuracy alone is misleading. Reported results use one fixed split and have sampling uncertainty. Predictions and segment associations do not establish causation or business impact.')
    st.markdown('[Dataset: Sakar & Kastro (2018), UCI, CC BY 4.0](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)')
