"""
코드 ② — baseline + 전체 모델 학습 / 튜닝 / 평가
- 모든 모델은 Pipeline(StandardScaler, clf)로 감싸 train에만 fit (누수 방지)
- 튜닝: PurgedBlockedCV + GridSearchCV, scoring=macro-F1
- 평가: test set에서 macro-F1, accuracy, macro-AUC(one-vs-rest)
- 결과/예측/확률을 results_models.joblib로 저장 (코드 ④에서 ROC·혼동행렬에 사용)
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from data_prep import load_and_clean, purged_temporal_split, PurgedBlockedCV

_OUT_DIR = Path(__file__).parent

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')
CLASS_NAMES = {1: '급가속', 2: '급우회전', 3: '급좌회전', 4: '급정거'}

# ── 데이터 + 분할 ─────────────────────────────────────────
X, y, dropped = load_and_clean()
X = X.values
y = y.values
tr, te = purged_temporal_split(y, train_ratio=0.7)
X_tr, X_te = X[tr], X[te]
y_tr, y_te = y[tr], y[te]
classes = np.unique(y)
print(f"train {len(tr)} / test {len(te)} / 특징 {X.shape[1]}")

cv = PurgedBlockedCV(n_splits=5)

def pipe(clf):
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])

# ── 모델 + 하이퍼파라미터 그리드 ──────────────────────────
models = {
    'LogReg':      (LogisticRegression(max_iter=2000),
                    {'clf__C': [0.01, 0.1, 1, 10]}),
    'LinearSVM':   (SVC(kernel='linear', probability=True),
                    {'clf__C': [0.1, 1, 10]}),
    'RBF_SVM':     (SVC(kernel='rbf', probability=True),
                    {'clf__C': [1, 10, 100], 'clf__gamma': ['scale', 0.01, 0.1]}),
    'GaussianNB':  (GaussianNB(),
                    {'clf__var_smoothing': [1e-9, 1e-7, 1e-5]}),
    'LDA':         (LinearDiscriminantAnalysis(solver='lsqr'),
                    {'clf__shrinkage': [None, 'auto', 0.5]}),
    'QDA':         (QuadraticDiscriminantAnalysis(),
                    {'clf__reg_param': [0.0, 0.01, 0.1, 0.5]}),
    'KNN':         (KNeighborsClassifier(),
                    {'clf__n_neighbors': [3, 5, 7, 11],
                     'clf__weights': ['uniform', 'distance']}),
    'DecisionTree':(DecisionTreeClassifier(random_state=0),
                    {'clf__max_depth': [3, 5, 10, None],
                     'clf__min_samples_leaf': [1, 5, 10]}),
    'RandomForest':(RandomForestClassifier(random_state=0),
                    {'clf__n_estimators': [200, 400],
                     'clf__max_depth': [None, 10, 20]}),
    'GradBoost':   (GradientBoostingClassifier(random_state=0),
                    {'clf__n_estimators': [100, 200],
                     'clf__learning_rate': [0.05, 0.1],
                     'clf__max_depth': [2, 3]}),
}

y_te_bin = label_binarize(y_te, classes=classes)
results = {}
rows = []

# ── Baseline (최빈 클래스) ────────────────────────────────
dummy = DummyClassifier(strategy='most_frequent').fit(X_tr, y_tr)
yp = dummy.predict(X_te)
proba = dummy.predict_proba(X_te)
try:
    auc = roc_auc_score(y_te_bin, proba, average='macro')
except Exception:
    auc = np.nan
results['Baseline(최빈)'] = {'y_pred': yp, 'y_proba': proba,
                             'best_params': {}, 'cv_f1': np.nan}
rows.append(['Baseline(최빈)', np.nan,
             f1_score(y_te, yp, average='macro'),
             accuracy_score(y_te, yp), auc])
print("Baseline 완료")

# ── 각 모델 튜닝 + 평가 ───────────────────────────────────
for name, (clf, grid) in models.items():
    gs = GridSearchCV(pipe(clf), grid, scoring='f1_macro',
                      cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_
    yp = best.predict(X_te)
    proba = best.predict_proba(X_te)
    try:
        auc = roc_auc_score(y_te_bin, proba, average='macro')
    except Exception:
        auc = np.nan
    results[name] = {'y_pred': yp, 'y_proba': proba,
                     'best_params': gs.best_params_, 'cv_f1': gs.best_score_}
    rows.append([name, gs.best_score_,
                 f1_score(y_te, yp, average='macro'),
                 accuracy_score(y_te, yp), auc])
    print(f"{name} 완료  (best: {gs.best_params_})")

# ── 결과 표 ───────────────────────────────────────────────
res_df = pd.DataFrame(rows, columns=['Model', 'CV_macroF1',
                                     'Test_macroF1', 'Test_Acc', 'Test_macroAUC'])
res_df = res_df.sort_values('Test_macroF1', ascending=False).reset_index(drop=True)
pd.set_option('display.float_format', lambda v: f'{v:.4f}')
print("\n" + "=" * 70)
print("모델 성능 비교 (Test macro-F1 내림차순)")
print("=" * 70)
print(res_df.to_string(index=False))

# ── 저장 (코드 ④에서 사용) ────────────────────────────────
joblib.dump({'results': results, 'y_test': y_te, 'classes': classes,
             'summary': res_df}, _OUT_DIR / 'results_models.joblib')
res_df.to_csv(_OUT_DIR / 'results_summary.csv', index=False)
print(f"\n저장: {_OUT_DIR}/results_models.joblib, {_OUT_DIR}/results_summary.csv")