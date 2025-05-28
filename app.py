import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from math import ceil
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


@st.cache_data
def load_data():
    serials_df = pd.read_excel("data/業務員資料0325V01.xlsx", sheet_name="DATA", engine="openpyxl", usecols=[0])
    serials_df.columns = ["serial_no"]
    features_df = pd.read_excel("data/filled_data_label_numeric.xlsx", engine="openpyxl")
    df = pd.concat([serials_df, features_df], axis=1)
    return df


def predict_abnormal_probability(X_input):
    model = joblib.load("models/best_lgbm_model.pkl")
    scaler = joblib.load("models/final_scaler.pkl")
    X_input = X_input[scaler.feature_names_in_]
    X_scaled = scaler.transform(X_input)
    prob = model.predict_proba(X_scaled)[0][1]
    return prob


def plot_continuous_features_with_highlight(df, serial_no, save_dir="output_plots"):
    if "abnormal_target" not in df.columns or "serial_no" not in df.columns:
        raise ValueError("缺少必要欄位")

    highlight_row = df[df["serial_no"] == serial_no]
    if highlight_row.empty:
        return

    continuous_columns = [
        "fyp_month_avg", "persistence_prem_25", "salary_ded_ratio", "age", "register_until_now",
        "bill_invalid_ratio", "bill_invalid_cnt", "agent_own_plcy", "one_year_claim_plcy_cnt",
        "one_year_claim_plcy_ratio", "addr_not_vld_rate", "comm_amount_1y_rate", "pc_in_60_days_rate",
        "lapse_2y_rate"
    ] + [f"rule{i}_counts" for i in range(1, 40)]

    os.makedirs(save_dir, exist_ok=True)

    for col in continuous_columns:
        if col not in df.columns:
            continue

        temp_df = df[[col, "abnormal_target"]].dropna()
        if temp_df.empty or temp_df["abnormal_target"].nunique() < 2:
            continue

        value_to_highlight = highlight_row[col].values[0]

        plt.figure(figsize=(6, 4))
        sns.histplot(
            data=temp_df,
            x=col,
            hue="abnormal_target",
            element="step",
            stat="density",
            common_norm=False,
            palette="Set2",
            bins=30
        )
        plt.axvline(value_to_highlight, color='blue', linestyle='--', linewidth=1.5)
        plt.text(value_to_highlight, plt.ylim()[1]*0.8, f"{serial_no}", color='blue', fontsize=8, rotation=90)
        plt.title(col)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{col}.png")
        plt.close()


# Streamlit App
st.set_page_config(
    page_title="業務員異常風險預測系統",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("業務員異常風險預測系統")
st.caption("Made with Streamlit")

df = load_data()

# 載入模型與 scaler（全域一次）
model = joblib.load("models/best_lgbm_model.pkl")
scaler = joblib.load("models/final_scaler.pkl")

search_tab, edit_tab, plot_tab, rank_tab = st.tabs([
    "🔍 員工異常查詢",
    "🛠️ 特徵調整預測",
    "📊 特徵視覺化分析",
    "📉 風險業務員排行"
])

with search_tab:
    st.subheader("查詢單一員工異常風險")
    serial_suffix = st.text_input("請輸入 4 位數字編號，例如：3000")
    if st.button("查詢異常風險"):
        if not serial_suffix.isdigit() or len(serial_suffix) != 4:
            st.error("⚠️ 員工編號格式錯誤，請輸入 4 位數字")
        else:
            full_serial_no = f"agnt_{serial_suffix}"
            target_row = df[df["serial_no"] == full_serial_no]
            if target_row.empty:
                st.error(f"找不到員工編號：{full_serial_no}")
            else:
                X_input = target_row.drop(columns=["serial_no", "abnormal_target"])
                X_input = X_input[X_input.columns.intersection(df.columns)]
                prob = predict_abnormal_probability(X_input)
                st.metric(label="異常機率", value=f"{prob:.2%}")
                if prob > 0.5:
                    st.warning("判斷結果：有異常風險")
                else:
                    st.success("判斷結果：無異常")

                # 產生圖檔
                plot_continuous_features_with_highlight(df, full_serial_no)

with edit_tab:
    st.subheader("修改特徵後預測異常風險")
    serial_list = df["serial_no"].tolist()
    selected_serial = st.selectbox("選擇員工編號", serial_list, placeholder="輸入員工編號以搜尋")

    if selected_serial:
        employee_row = df[df["serial_no"] == selected_serial].iloc[0]

        input_columns = {
            "xagfd_flag": "是否為指定通路旗艦業務員",
            "salary_ded_ratio": "薪資扣繳比率",
            "fyp_month_avg": "月平均保費收入FYP",
            "one_year_claim_plcy_cnt": "理賠件數（近一年）",
            "rule18_counts": "違規規則18次數",
            "area_flg": "區域旗標",
            "dept_flg": "部門旗標",
            "addr_not_vld_rate": "地址無效比率",
            "agent_level": "業務員等級",
            "cntr_flg": "合約旗標"
        }

        st.markdown("請調整下列欄位值後進行預測：")
        user_input = []
        for col, label in input_columns.items():
            default_val = float(employee_row[col])
            if col.endswith("_flg") or col.endswith("_flag"):
                val = st.selectbox(label, options=["Y", "N"], index=0 if default_val == 1 else 1)
                user_input.append(1 if val == "Y" else 0)
            else:
                val = st.number_input(label, value=default_val, format="%.2f")
                user_input.append(val)

        if st.button("預測異常風險（使用修改後參數）"):
            input_df = pd.DataFrame([user_input], columns=input_columns.keys())
            full_features_df = df.drop(columns=["serial_no", "abnormal_target"]).iloc[0:1].copy()
            for col in input_columns:
                full_features_df[col] = input_df[col].values[0]
            prob = predict_abnormal_probability(full_features_df)

            st.subheader("預測結果")
            st.metric(label="異常機率", value=f"{prob:.2%}")
            if prob > 0.5:
                st.warning("判斷結果：有異常風險")
            else:
                st.success("判斷結果：無異常")

with plot_tab:
    st.subheader("所有連續特徵視覺化")
    plot_dir = "output_plots"
    if os.path.exists(plot_dir):
        img_files = sorted([f for f in os.listdir(plot_dir) if f.endswith(".png")])
        cols = st.columns(3)
        for i, img_file in enumerate(img_files):
            img_path = os.path.join(plot_dir, img_file)
            with cols[i % 3]:
                st.image(Image.open(img_path), caption=img_file, use_column_width=True)
    else:
        st.info("尚未產生圖檔，請先執行一次查詢。")

with rank_tab:
    st.subheader("異常風險最高的前 50 位業務員")

    try:
        X_all = df.drop(columns=["serial_no", "abnormal_target"])
        X_all = X_all[scaler.feature_names_in_]
        X_scaled = scaler.transform(X_all)
        all_probs = model.predict_proba(X_scaled)[:, 1]

        df["abnormal_prob"] = all_probs
        top50_df = df.sort_values("abnormal_prob", ascending=False).head(50)[["serial_no", "abnormal_prob"]]
        top50_df["abnormal_prob"] = top50_df["abnormal_prob"].apply(lambda p: f"{p:.2%}")

        st.table(top50_df.rename(columns={"serial_no": "員工編號", "abnormal_prob": "預測異常機率"}))
    except Exception as e:
        st.error(f"發生錯誤：{e}")