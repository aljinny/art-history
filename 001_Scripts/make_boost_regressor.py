import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.image import imread
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split

import cv2
import pickle

base_path = '/home/jinny/projects/Art-history/Art-history/datas/'
file_info = pd.read_csv(base_path+'file_info.csv')

# file_info_latent
df = pd.DataFrame(( np.load(base_path+'vectors/avec_latents.npy', allow_pickle=True)),columns=['avec','Path'])
file_info_latents = pd.merge(file_info, df, how = 'left', on = 'Path')
file_info_latents = file_info_latents[~file_info_latents.avec.isnull()]

avec = np.array([i.reshape(-1) for i in file_info_latents['avec']])

df = pd.DataFrame(np.load(base_path+'vectors/cvec_latents.npy', allow_pickle=True),columns=['cvec','Path'])
file_info_latents = pd.merge(file_info_latents, df, how = 'left', on = 'Path')

cvec = np.array([i.reshape(-1) for i in file_info_latents['cvec']])

import sklearn
from xgboost import XGBClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from scipy.stats import kendalltau

import numpy as np
from xgboost import plot_importance
import xgboost as xgb

import random
from tqdm import tqdm
import os
seeds = [ random.randrange(1, 10000) for _ in range(100) ]
out_dir = "datas/models/xgboosts"

# 결과 컨테이너
predict_wide = {}      # 시드별 테스트셋 예측/정답(연/버킷) - wide
trains_wide  = {}      # 시드별 학습 인덱스 - wide
metrics_rows = []      # 시드별 지표 - long(rows)

# 1) 학습에 사용할 대상 필터링
temp = file_info_latents[file_info_latents['new_date'] >= 1500].copy()

# 2) y(bins) 준비: 1500년 기준 10년 단위 버킷
y_all_year = temp['new_date'].tolist()
y_all_bin  = [int((y - 1500) / 10) for y in y_all_year]

# 3) X 준비: 인덱스를 temp.index로 강제 맞추고, 순서도 동일하게
#    (cvec/avec가 np.array/scipy.spmatrix여도 DataFrame으로 씌우며 index를 강제 지정)
X_raw_c = pd.DataFrame(cvec, index=file_info_latents.index).loc[temp.index]
X_raw_a = pd.DataFrame(avec, index=file_info_latents.index).loc[temp.index]

# 4) 시드 루프
for seed in tqdm(seeds, desc="seeds"):
    # --- train/test split ---
    # stratify는 버킷 기준. 부족 클래스가 있으면 폴백
    try:
        X_train_idx, X_test_idx, y_train_bin, y_test_bin = train_test_split(
            temp.index.tolist(), y_all_bin,
            test_size=0.30,
            random_state=seed,
            stratify=y_all_bin
        )
    except ValueError:
        # 버킷별 샘플 수 부족 → stratify 없이 분할
        print(f'***{seed}seed stratify 없이 분할***')
        X_train_idx, X_test_idx, y_train_bin, y_test_bin = train_test_split(
            temp.index.tolist(), y_all_bin,
            test_size=0.30,
            random_state=seed,
            stratify=None
        )

    # 평가/저장 편의를 위해 '연도' 버전도 준비
    y_test_year = [1500 + b * 10 for b in y_test_bin]

    # --- 특징행렬 분할 (순서 보장: loc로 정확한 순서 재배치) ---
    C_train = X_raw_c.loc[X_train_idx]
    C_test  = X_raw_c.loc[X_test_idx]
    A_train = X_raw_a.loc[X_train_idx]
    A_test  = X_raw_a.loc[X_test_idx]

    # --- 모델 정의(랜덤 고정) ---
    def build_xgb(seed):
        return xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.08,
            gamma=0,
            subsample=0.75,
            colsample_bytree=1,
            max_depth=7,
            random_state=1
        )

    # --- C-vector 학습/예측 ---
    c_model = build_xgb(seed)
    c_model.fit(C_train, y_train_bin)
    y_pred_c_bin  = c_model.predict(C_test)  # 실수(버킷 연속값)
    y_pred_c_year  = 1500 + y_pred_c_bin * 10               # 연도환산(정수)
    c_model.save_model(f"datas/models/xgboosts/cvec_xgb_model_{seed}.json") 

    # --- A-vector 학습/예측 ---
    a_model = build_xgb(seed)
    a_model.fit(A_train, y_train_bin)
    y_pred_a_bin  = a_model.predict(A_test)
    y_pred_a_year  = 1500 + y_pred_a_bin * 10
    a_model.save_model(f"datas/models/xgboosts/avec_xgb_model_{seed}.json") 

    # --- 지표 계산 (버킷 스페이스에서 계산) ---
    def eval_all(y_true_bin, y_pred_bin):
        pear = float(pearsonr(y_true_bin, y_pred_bin)[0])
        spear = float(spearmanr(y_true_bin, y_pred_bin).correlation)
        mae = float(mean_absolute_error(y_true_bin, y_pred_bin))
        rmse = float(mean_squared_error(y_true_bin, y_pred_bin, squared=False))
        r2 = float(r2_score(y_true_bin, y_pred_bin))
        return dict(pearson=pear, spearman=spear, mae=mae, rmse=rmse, r2=r2)

    m_c = eval_all(y_test_bin, y_pred_c_bin)
    m_a = eval_all(y_test_bin, y_pred_a_bin)

    # --- 콘솔 로그 ---
    print(f"\n------------ seed: {seed} ------------")
    print("------ C-vector ------")
    print(f"Pearson: {m_c['pearson']:.4f} | Spearman: {m_c['spearman']:.4f} | R2: {m_c['r2']:.4f} | RMSE: {m_c['rmse']:.4f} | MAE: {m_c['mae']:.4f}")
    print("------ A-vector ------")
    print(f"Pearson: {m_a['pearson']:.4f} | Spearman: {m_a['spearman']:.4f} | R2: {m_a['r2']:.4f} | RMSE: {m_a['rmse']:.4f} | MAE: {m_a['mae']:.4f}")

    # --- wide 저장: 시드별 컬럼 생성 (테스트셋 인덱스 기준) ---
    seed_key = f"seed_{seed}"
    # 테스트 인덱스는 시드별로 다르므로 인덱스를 키로 사용
    # 각 시드마다 별도 DataFrame 만들어서 이후 조인해도 되지만, 여기서는 컬럼 네이밍으로 분리
    predict_wide[f"{seed_key}_index"]        = X_test_idx
    predict_wide[f"{seed_key}_y_test_year"]  = y_test_year
    predict_wide[f"{seed_key}_C_pred_year"]  = y_pred_c_year
    predict_wide[f"{seed_key}_A_pred_year"]  = y_pred_a_year

    # 학습 인덱스도 저장
    trains_wide[f"{seed_key}_train_index"] = X_train_idx

    # 지표 저장 (long rows)
    metrics_rows.append({
        "seed": seed, "vector": "C",
        **m_c
    })
    metrics_rows.append({
        "seed": seed, "vector": "A",
        **m_a
    })

# --- 결과물 저장 ---
df_predicts = pd.DataFrame(predict_wide)
df_trains   = pd.DataFrame(trains_wide)
df_metrics  = pd.DataFrame(metrics_rows)

df_predicts.to_csv(os.path.join(out_dir, "predicts.csv"), index=False)
df_trains.to_csv(os.path.join(out_dir, "trains.csv"), index=False)
df_metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

with open(os.path.join(out_dir, "seeds.pkl"), "wb") as f:
    pickle.dump(seeds, f)

print("\n=== Saved ===")
print(f"- {os.path.join(out_dir, 'predicts.csv')}")
print(f"- {os.path.join(out_dir, 'trains.csv')}")
print(f"- {os.path.join(out_dir, 'metrics.csv')}")
print(f"- {os.path.join(out_dir, 'seeds.pkl')}")

