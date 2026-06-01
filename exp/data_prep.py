"""
코드 ① — 데이터 준비 모듈 (누수 방지 기반)
이후 모델링/실험 스크립트가 import하여 사용.

제공:
  load_and_clean(path)              : 로드 + 완전중복 특징 제거
  purged_temporal_split(y, ...)     : 클래스별 purged temporal split
  PurgedBlockedCV                   : 하이퍼파라미터 튜닝용 블록 CV (purge 포함)

핵심 원칙:
  - 윈도우는 stride=1 겹침 → 인접 윈도우가 raw 샘플을 공유
  - 분할 경계에서 purge개(=윈도우-1=13) 윈도우를 제거해
    train·test가 같은 raw 샘플을 절대 공유하지 않게 함
  - 표준화는 모델링 스크립트에서 Pipeline(StandardScaler, model)로
    처리하여 train에만 fit (이 모듈은 표준화 안 함)
"""

import pandas as pd
import numpy as np
from pathlib import Path

_DATA_PATH = Path(__file__).parent.parent / 'data' / 'features_14.csv'

WINDOW = 14
PURGE = WINDOW - 1   # 13: 인접 윈도우가 공유하는 최대 샘플 수


def load_and_clean(path=_DATA_PATH):
    """로드 후 상관 1.0 완전중복 특징 제거."""
    df = pd.read_csv(path)
    X = df.drop(columns=['Target'])
    y = df['Target']

    corr = X.corr().abs().values.copy()
    np.fill_diagonal(corr, 0)
    cdf = pd.DataFrame(corr, index=X.columns, columns=X.columns)
    drop = set()
    for i, ci in enumerate(X.columns):
        if ci in drop:
            continue
        for cj in X.columns[i + 1:]:
            if cj not in drop and cdf.loc[ci, cj] > 0.999:
                drop.add(cj)
    X_clean = X.drop(columns=list(drop))
    return X_clean, y, sorted(drop)


def purged_temporal_split(y, train_ratio=0.7, purge=PURGE):
    """클래스별 블록에서 앞 train_ratio는 train, 뒤는 test.
    경계의 purge개 윈도우는 양쪽 모두에서 제외하여 누수 차단."""
    y = np.asarray(y)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        pos = np.where(y == c)[0]          # df상 위치 (클래스 내 시간순)
        m = len(pos)
        n_tr = int(m * train_ratio)
        train_idx.extend(pos[:n_tr].tolist())
        test_idx.extend(pos[n_tr + purge:].tolist())   # 가운데 purge개 폐기
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


class PurgedBlockedCV:
    """클래스별 블록 구조와 시간순을 유지하는 K-fold.
    각 클래스 블록을 K개 연속 구간으로 나누고, fold j의 검증 구간은
    각 클래스의 j번째 구간. 검증 구간에 인접한 purge개 윈도우는
    (같은 클래스 내에서) 학습에서 제외하여 누수 차단.
    sklearn의 cv 인자로 직접 사용 가능."""

    def __init__(self, n_splits=5, purge=PURGE):
        self.n_splits = n_splits
        self.purge = purge

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, groups=None):
        y = np.asarray(y)
        classes = np.unique(y)
        class_pos = {c: np.where(y == c)[0] for c in classes}
        n = len(y)
        for j in range(self.n_splits):
            val_idx, purged = [], set()
            for c in classes:
                pos = class_pos[c]
                m = len(pos)
                s = int(m * j / self.n_splits)
                e = int(m * (j + 1) / self.n_splits)
                val_idx.extend(pos[s:e].tolist())
                lo = max(0, s - self.purge)
                hi = min(m, e + self.purge)
                purged.update(pos[lo:s].tolist())
                purged.update(pos[e:hi].tolist())
            val_set = set(val_idx)
            train_idx = [i for i in range(n)
                         if i not in val_set and i not in purged]
            yield np.array(train_idx), np.array(sorted(val_idx))


# ── 데모 / 검증 ───────────────────────────────────────────
if __name__ == '__main__':
    CLASS_NAMES = {1: '급가속', 2: '급우회전', 3: '급좌회전', 4: '급정거'}

    X, y, dropped = load_and_clean(_DATA_PATH)
    print("=" * 60)
    print("데이터 준비 모듈 검증")
    print("=" * 60)
    print(f"표본 {X.shape[0]}개, 특징 {X.shape[1]}개 (중복 {len(dropped)}개 제거)")

    tr, te = purged_temporal_split(y, train_ratio=0.7)
    print(f"\nPurged temporal split (train_ratio=0.7, purge={PURGE})")
    print(f"  train: {len(tr)}개, test: {len(te)}개, "
          f"폐기(purge): {len(y) - len(tr) - len(te)}개")
    yv = np.asarray(y)
    print("  클래스별 train/test:")
    for c in np.unique(yv):
        n_tr = int((yv[tr] == c).sum())
        n_te = int((yv[te] == c).sum())
        print(f"    {c} ({CLASS_NAMES[c]}): train {n_tr}, test {n_te}")

    # 누수 검증: 각 클래스에서 train 마지막 위치와 test 첫 위치 간격 확인
    print("\n  누수 검증 (클래스 내 train끝~test시작 간격, ≥13이어야 안전):")
    for c in np.unique(yv):
        pos = np.where(yv == c)[0]
        tr_pos = [i for i in tr if yv[i] == c]
        te_pos = [i for i in te if yv[i] == c]
        gap = (min(te_pos) - max(tr_pos)) if tr_pos and te_pos else None
        print(f"    {c} ({CLASS_NAMES[c]}): 간격 {gap} "
              f"{'OK' if gap and gap > PURGE else '⚠'}")

    print(f"\n  블록 CV fold 크기 (n_splits=5):")
    cv = PurgedBlockedCV(n_splits=5)
    for k, (cv_tr, cv_val) in enumerate(cv.split(X.iloc[tr], yv[tr])):
        print(f"    fold {k}: train {len(cv_tr)}, val {len(cv_val)}")
    print("=" * 60)