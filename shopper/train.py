"""Run with python -m shopper.train. Never tune using the test partition."""
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import DATA, ROOT, FEATURES, NUMERIC, CATEGORICAL, clean_data, load_data

SEED = 42


def split_data(df):
    train, test = train_test_split(df, test_size=.2, stratify=df.Revenue, random_state=SEED)
    train, valid = train_test_split(train, test_size=.25, stratify=train.Revenue, random_state=SEED)
    return train, valid, test


def build_pipeline(name, page_values=False):
    numeric = NUMERIC + (['PageValues'] if page_values else [])
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
    ], remainder='drop')
    models = {
        'logistic_regression': LogisticRegression(max_iter=3000, random_state=SEED),
        'random_forest': RandomForestClassifier(n_estimators=250, min_samples_leaf=4,
                                               max_features='sqrt', n_jobs=2, random_state=SEED),
        'dummy': DummyClassifier(strategy='prior'),
    }
    return Pipeline([('preprocess', preprocess), ('model', models[name])])


def choose_threshold(y, probabilities):
    # Validation F1 chooses an operating point; all test labels remain untouched.
    thresholds = np.linspace(.05, .8, 151)
    scores = [f1_score(y, probabilities >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate(y, probabilities, threshold):
    predicted = probabilities >= threshold
    return {'precision': float(precision_score(y, predicted, zero_division=0)),
            'recall': float(recall_score(y, predicted, zero_division=0)),
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, probabilities)),
            'pr_auc_ap': float(average_precision_score(y, probabilities)),
            'threshold': threshold,
            'confusion_matrix': confusion_matrix(y, predicted, labels=[False, True]).tolist()}


def run(output=ROOT / 'artifacts'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = load_data()
    df = clean_data(raw)
    train, valid, test = split_data(df)
    report = {'seed': SEED, 'raw_rows': len(raw), 'duplicate_rows_removed': len(raw)-len(df),
              'clean_rows': len(df), 'dataset_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
              'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                           'pandas': pd.__version__, 'numpy': np.__version__},
              'split': {name: {'rows': len(part), 'purchases': int(part.Revenue.sum()),
                               'prevalence': float(part.Revenue.mean()), 'indices': part.index.tolist()}
                        for name, part in [('train', train), ('validation', valid), ('test', test)]},
              'models': {}}
    fitted = {}
    for name in ['dummy', 'logistic_regression', 'random_forest']:
        model = build_pipeline(name)
        model.fit(train[FEATURES], train.Revenue)
        threshold = .5 if name == 'dummy' else choose_threshold(valid.Revenue, model.predict_proba(valid[FEATURES])[:, 1])
        report['models'][name] = {'validation': evaluate(valid.Revenue, model.predict_proba(valid[FEATURES])[:, 1], threshold)}
        fitted[name] = model
    selected = max(['logistic_regression', 'random_forest'],
                   key=lambda n: report['models'][n]['validation']['pr_auc_ap'])
    report['selected_model'] = selected
    report['selection_rule'] = 'Maximum validation average precision; threshold maximizes validation F1; fixed models, no test tuning.'
    predictions = pd.DataFrame({'row_id': test.index, 'actual': test.Revenue.astype(int).to_numpy()})
    curves = {}
    for name, model in fitted.items():
        probabilities = model.predict_proba(test[FEATURES])[:, 1]
        threshold = report['models'][name]['validation']['threshold']
        report['models'][name]['test'] = evaluate(test.Revenue, probabilities, threshold)
        predictions[name] = probabilities
        fpr, tpr, _ = roc_curve(test.Revenue, probabilities)
        precision, recall, _ = precision_recall_curve(test.Revenue, probabilities)
        curves[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'precision': precision.tolist(), 'recall': recall.tolist()}
    # Explicit leakage sensitivity: these models cannot be selected for the app.
    report['page_values_sensitivity'] = {}
    for name in ['logistic_regression', 'random_forest']:
        model = build_pipeline(name, page_values=True)
        cols = FEATURES + ['PageValues']
        model.fit(train[cols], train.Revenue)
        threshold = choose_threshold(valid.Revenue, model.predict_proba(valid[cols])[:, 1])
        report['page_values_sensitivity'][name] = evaluate(test.Revenue, model.predict_proba(test[cols])[:, 1], threshold)
    bundle = {'pipeline': fitted[selected], 'threshold': report['models'][selected]['validation']['threshold'],
              'model_name': selected, 'features': FEATURES, 'dataset_sha256': report['dataset_sha256']}
    joblib.dump(bundle, output / 'model.joblib', compress=3)
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'curves.json').write_text(json.dumps(curves), encoding='utf-8')
    rows = [{'model': name, **{k: v for k, v in item['test'].items() if k != 'confusion_matrix'}} for name, item in report['models'].items()]
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print('Selected using validation AP:', selected)
    return report


if __name__ == '__main__':
    run()
