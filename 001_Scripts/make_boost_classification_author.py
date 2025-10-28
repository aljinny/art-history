# -*- coding: utf-8 -*-
import os, random, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# 모델/스플릿/스케일러 등
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

import xgboost as xgb

# =========================
# 0) 입력 데이터 로드
# =========================
base_path = '/home/jinny/projects/Art-history/Art-history/datas/'
file_info = pd.read_csv(base_path + 'file_info.csv')

# AVEC 로드 후 병합
df_a = pd.DataFrame(np.load(base_path+'vectors/avec_latents.npy', allow_pickle=True),
                    columns=['avec','Path'])
file_info_latents = pd.merge(file_info, df_a, how='left', on='Path')
file_info_latents = file_info_latents[~file_info_latents['avec'].isnull()]

# CVEC 로드 후 병합
df_c = pd.DataFrame(np.load(base_path+'vectors/cvec_latents.npy', allow_pickle=True),
                    columns=['cvec','Path'])
file_info_latents = pd.merge(file_info_latents, df_c, how='left', on='Path')
file_info_latents = file_info_latents[~file_info_latents['cvec'].isnull()].copy()

# =========================
# 1) 학습 대상 스타일 정규화
#    (좌: 원문 라벨, 우: 정규화 라벨)
# =========================
target_authors = [['Albrecht Durer (1528)','albrecht durer'],['Rembrandt (1669)','rembrandt'],['Johannes Vermeer (1675)','johannes vermeer'],['Jean-Francois Millet (1875)','jean-francois millet'],['Edouard Manet (1883)','edouard manet'],['Paul Cezanne (1906)','paul cezanne'],['Pierre-Auguste Renoir (1919)','pierre auguste renoir'],['Vincent van Gogh (1890)','vincent van gogh'],['Gustav Klimt (1918)','gustav klimt'],['Egon Schiele (1918)','egon schiele']]

alias = {long: short for long, short in target_authors}
valid_labels = set(alias.keys()) | set(alias.values())

# Author을 두 집합(any)에 포함되는 것만 필터 및 정규화
temp = file_info_latents[file_info_latents['author_name'].isin(valid_labels)].copy()
temp['author_norm'] = temp['author_name'].map(lambda s: alias.get(s, s))

# =========================
# 2) X(특징)와 y(라벨) 준비
# =========================
# temp['avec'], temp['cvec'] 각각은 객체 배열일 수 있으니 안전하게 vstack
avec = np.vstack([np.asarray(v).reshape(-1) for v in temp['avec'].tolist()])
cvec = np.vstack([np.asarray(v).reshape(-1) for v in temp['cvec'].tolist()])

# 인덱스를 temp.index로 강제 부여하여 X–y 정렬 보장
X_raw_a = pd.DataFrame(avec, index=temp.index)
X_raw_c = pd.DataFrame(cvec, index=temp.index)

enc = LabelEncoder()
y_all_author = temp['author_norm'].tolist()
y_all_bin = enc.fit_transform(y_all_author)            # 정수 라벨
n_classes = len(enc.classes_)

# =========================
# 3) 실험 설정/컨테이너
# =========================
with open("datas/models/seeds.pkl", "rb") as f:
     seeds = pickle.load(f)
out_dir = "datas/models/xgboosts_author"
os.makedirs(out_dir, exist_ok=True)

predict_wide = {}     # 시드별 테스트 인덱스/정답/예측 (author 라벨)
trains_wide  = {}     # 시드별 학습 인덱스
metrics_rows = []     # 시드별 메트릭(롱 포맷)

# =========================
# 4) 헬퍼: 모델 빌더 / 평가자
# =========================
def build_xgb_classifier(seed: int) -> xgb.XGBClassifier:
    # 멀티클래스에 맞게 설정
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.75,
        colsample_bytree=1.0,
        objective='multi:softprob',  # 확률 출력
        num_class=n_classes,
        random_state=seed,
        tree_method='auto',
        eval_metric='mlogloss'
    )

def eval_classification(y_true, y_pred, y_proba):
    # macro/weighted 모두 저장 (클래스 불균형 대비)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
    }
    # log_loss는 확률 필요. 테스트셋에 모든 클래스가 없으면 labels 지정
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=list(range(n_classes))))
    except Exception:
        metrics["log_loss"] = np.nan
    return metrics

# =========================
# 5) 시드 루프
# =========================
for seed in tqdm(seeds, desc="seeds"):
    # --- stratified split (클래스 수 부족 시 폴백) ---
    try:
        X_tr_idx, X_te_idx, y_tr, y_te = train_test_split(
            temp.index.tolist(), y_all_bin,
            test_size=0.30, random_state=seed, stratify=y_all_bin
        )
    except ValueError:
        print(f"*** seed {seed}: stratify 불가 → 무작위 분할로 폴백 ***")
        X_tr_idx, X_te_idx, y_tr, y_te = train_test_split(
            temp.index.tolist(), y_all_bin,
            test_size=0.30, random_state=seed, stratify=None
        )

    # --- 특징 분할 (순서 보장) ---
    A_tr, A_te = X_raw_a.loc[X_tr_idx], X_raw_a.loc[X_te_idx]
    C_tr, C_te = X_raw_c.loc[X_tr_idx], X_raw_c.loc[X_te_idx]

    # --- A-vector ---
    a_model = build_xgb_classifier(seed)
    a_model.fit(A_tr, y_tr)
    y_pred_a = a_model.predict(A_te)
    y_proba_a = a_model.predict_proba(A_te)

    # --- C-vector ---
    c_model = build_xgb_classifier(seed)
    c_model.fit(C_tr, y_tr)
    y_pred_c = c_model.predict(C_te)
    y_proba_c = c_model.predict_proba(C_te)

    # --- 메트릭 계산 ---
    m_a = eval_classification(y_te, y_pred_a, y_proba_a)
    m_c = eval_classification(y_te, y_pred_c, y_proba_c)

    # --- 저장(모델) ---
    a_model.save_model(os.path.join(out_dir, f"avec_xgb_cls_{seed}.json"))
    c_model.save_model(os.path.join(out_dir, f"cvec_xgb_cls_{seed}.json"))

    # --- 저장(예측/정답: 사람이 읽기 쉬운 author 라벨로) ---
    seed_key = f"seed_{seed}"
    predict_wide[f"{seed_key}_index"] = X_te_idx
    predict_wide[f"{seed_key}_y_true"] = enc.inverse_transform(y_te)
    predict_wide[f"{seed_key}_A_pred"] = enc.inverse_transform(y_pred_a)
    predict_wide[f"{seed_key}_C_pred"] = enc.inverse_transform(y_pred_c)

    # 학습 인덱스
    trains_wide[f"{seed_key}_train_index"] = X_tr_idx

    # --- 메트릭(롱 포맷) ---
    metrics_rows.append({"seed": seed, "vector": "A", **m_a})
    metrics_rows.append({"seed": seed, "vector": "C", **m_c})

# =========================
# 6) 산출물 저장
# =========================
df_predicts = pd.DataFrame(predict_wide)
df_trains   = pd.DataFrame(trains_wide)
df_metrics  = pd.DataFrame(metrics_rows)

df_predicts.to_csv(os.path.join(out_dir, "predicts_author.csv"), index=False)
df_trains.to_csv(os.path.join(out_dir, "trains_author.csv"), index=False)
df_metrics.to_csv(os.path.join(out_dir, "metrics_author.csv"), index=False)

print("\n=== Saved ===")
print(f"- {os.path.join(out_dir, 'predicts_author.csv')}")
print(f"- {os.path.join(out_dir, 'trains_author.csv')}")
print(f"- {os.path.join(out_dir, 'metrics_author.csv')}")
print(f"- {os.path.join(out_dir, 'seeds_author.pkl')}")