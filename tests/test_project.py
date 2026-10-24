import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.validation import check_is_fitted
from shopper.data import DATA, ROOT, FEATURES, NUMERIC, clean_data, load_data, conversion_table
from shopper.train import build_pipeline, split_data, choose_threshold


@pytest.fixture(scope='module')
def df():
    return clean_data(load_data())


def test_original_dataset_integrity():
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == 'b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9'
    raw = load_data()
    assert raw.shape == (12330, 18)
    assert raw.duplicated().sum() == 125
    assert not raw.isna().any().any()


def test_cleaning_idempotent(df):
    pd.testing.assert_frame_equal(df, clean_data(df))
    assert len(df) == 12205


def test_split_no_overlap_or_duplicate_features(df):
    parts = split_data(df)
    assert sum(map(len, parts)) == len(df)
    for i, a in enumerate(parts):
        for b in parts[i+1:]:
            assert set(a.index).isdisjoint(b.index)
            assert set(pd.util.hash_pandas_object(a[FEATURES], index=False)).isdisjoint(
                set(pd.util.hash_pandas_object(b[FEATURES], index=False)))
        assert abs(a.Revenue.mean() - df.Revenue.mean()) < .002


def test_preprocessing_fits_only_training_statistics(df):
    train, valid, _ = split_data(df)
    pipe = build_pipeline('logistic_regression')
    pipe.fit(train[FEATURES], train.Revenue)
    imputer = pipe.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
    np.testing.assert_allclose(imputer.statistics_, train[NUMERIC].median().to_numpy())
    probe = valid[FEATURES].iloc[:2].copy()
    probe['VisitorType'] = 'unseen_visitor'
    probe.loc[probe.index[0], 'ProductRelated_Duration'] = np.nan
    assert np.isfinite(pipe.predict_proba(probe)).all()


def test_pagevalues_and_target_are_excluded(df):
    bundle = joblib.load(ROOT / 'artifacts/model.joblib')
    assert 'Revenue' not in bundle['features']
    assert 'PageValues' not in bundle['features']
    names = bundle['pipeline'].named_steps['preprocess'].get_feature_names_out()
    assert not any('PageValues' in n or 'Revenue' in n for n in names)
    a = df.iloc[:10].copy()
    before = bundle['pipeline'].predict_proba(a)
    a['Revenue'] = ~a.Revenue
    a['PageValues'] = 999999
    np.testing.assert_allclose(before, bundle['pipeline'].predict_proba(a))


def test_saved_metrics_recompute_from_test_predictions():
    report = json.loads((ROOT / 'artifacts/metrics.json').read_text())
    p = pd.read_csv(ROOT / 'artifacts/test_predictions.csv')
    assert p.row_id.tolist() == report['split']['test']['indices']
    for name, result in report['models'].items():
        m = result['test']
        label = p[name] >= m['threshold']
        for key, function in [('precision', precision_score), ('recall', recall_score), ('f1', f1_score)]:
            assert m[key] == pytest.approx(function(p.actual, label, zero_division=0))
        assert m['roc_auc'] == pytest.approx(roc_auc_score(p.actual, p[name]))
        assert m['pr_auc_ap'] == pytest.approx(average_precision_score(p.actual, p[name]))
        assert m['confusion_matrix'] == confusion_matrix(p.actual, label).tolist()


def test_selection_and_threshold_use_validation(df):
    report = json.loads((ROOT / 'artifacts/metrics.json').read_text())
    bundle = joblib.load(ROOT / 'artifacts/model.joblib')
    chosen = max(['logistic_regression', 'random_forest'], key=lambda n: report['models'][n]['validation']['pr_auc_ap'])
    assert bundle['model_name'] == report['selected_model'] == chosen
    _, valid, _ = split_data(df)
    assert bundle['threshold'] == choose_threshold(valid.Revenue, bundle['pipeline'].predict_proba(valid[FEATURES])[:, 1])
    check_is_fitted(bundle['pipeline'])


@pytest.mark.parametrize('column,value', [('Revenue', 'maybe'), ('Weekend', None), ('BounceRates', 2), ('ProductRelated_Duration', -1), ('Month', 'invalid'), ('ProductRelated', 1.5)])
def test_invalid_data_rejected(tmp_path, column, value):
    raw = pd.read_csv(DATA).head(5).astype(object)
    raw.loc[0, column] = value
    path = tmp_path / 'bad.csv'
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)


def test_conversion_counts(df):
    table = conversion_table(df, 'VisitorType')
    assert table.sessions.sum() == len(df)
    assert table.purchases.sum() == df.Revenue.sum()


def test_dashboard_prediction_and_empty_filters():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    assert not app.exception
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert any(m.label == 'Estimated purchase score' for m in app.metric)
    app.multiselect[0].set_value([]).run(timeout=60)
    assert not app.exception
    assert any('No sessions match' in w.value for w in app.warning)
