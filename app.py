import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import numpy as np

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

st.set_page_config(
    page_title="業務員異常風險預測系統",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("業務員異常風險預測系統")
st.caption("Made with Streamlit")

df = load_data()

search_tab, edit_tab = st.tabs(["🔍 員工異常查詢", "🛠️ 特徵調整預測"])

with search_tab:
    st.subheader("查詢單一員工異常風險")
    serial_suffix = st.text_input("請輸入 4 位數字編號，例如：3000")
    if st.button("查詢異常風險"):
        if not serial_suffix.isdigit() or len(serial_suffix) != 4:
            st.error("⚠️ 員工編號格式錯誤，請輸入 4 位數字，例如：3000")
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

with edit_tab:
    st.subheader("修改特徵後預測異常風險")
    serial_list = ["請選擇員工編號"] + df["serial_no"].tolist()
    selected_serial = st.selectbox("選擇員工編號", serial_list, index=0, placeholder="輸入員工編號以搜尋")

    if selected_serial != "請選擇員工編號":
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
