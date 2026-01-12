import os
import json
import datetime as dt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import requests
import streamlit as st
import pandas as pd
import altair as alt

# ==================================================
# Gemini 2.5 設定
# ==================================================
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@st.cache_resource
def get_client():
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        return genai.Client(api_key=GEMINI_API_KEY)
    return None

# ==================================================
# 設定 & 定数
# ==================================================
APP_TITLE = "Wellness Forecast Pro"
CITIES = {
    "東京": (35.6895, 139.6917),
    "大阪": (34.6937, 135.5023),
    "名古屋": (35.1815, 136.9066),
    "札幌": (43.0618, 141.3545),
    "福岡": (33.5904, 130.4017),
    "那覇": (26.2124, 127.6809),
    "手動入力": (None, None)
}

# ==================================================
# 天気データ取得 (キャッシュ付)
# ==================================================
@st.cache_data(ttl=3600)
def fetch_weather(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["pressure_msl", "temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation"],
        "timezone": "auto", "past_days": 1, "forecast_days": 3
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df

# ==================================================
# ロジック関数 (元の機能をブラッシュアップ)
# ==================================================
def calc_bmi(h, w):
    return w / ((h / 100)**2) if h and w else None

def get_risk_info(score):
    if score <= 3: return "おちついている", "#3CB371", "🟢"
    if score <= 6: return "少し注意したい", "#FFD54F", "🟡"
    return "今日はかなり慎重に", "#FF6B6B", "🔴"

# ==================================================
# UI: スタイル適用
# ==================================================
def inject_css():
    st.markdown("""
        <style>
        .wf-card {
            background: rgba(255,255,255,0.8);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #eee;
            margin-bottom: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .stMetric { background: #f8f9fa; padding: 10px; border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)

# ==================================================
# メイン画面
# ==================================================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")
    inject_css()
    client = get_client()

    st.title(f"🩺 {APP_TITLE}")
    st.caption("Gemini 2.5 Flash があなたの体調管理をサポートします")

    # サイドバー：プロフィール設定
    with st.sidebar:
        st.header("👤 プロフィール")
        age = st.number_input("年齢", 0, 120, 40)
        height = st.number_input("身長(cm)", 100.0, 250.0, 170.0)
        weight = st.number_input("体重(kg)", 30.0, 200.0, 60.0)
        
        st.header("📍 場所設定")
        city = st.selectbox("エリアを選択", list(CITIES.keys()))
        if city == "手動入力":
            lat = st.number_input("緯度", -90.0, 90.0, 35.68)
            lon = st.number_input("経度", -180.0, 180.0, 139.69)
        else:
            lat, lon = CITIES[city]

    # メインコンテンツ
    col_input, col_chart = st.columns([1, 1.5])

    with col_input:
        st.markdown('<div class="wf-card">', unsafe_allow_html=True)
        st.subheader("📝 今日の状況")
        sleep = st.slider("昨夜の睡眠時間", 0.0, 12.0, 7.0)
        alcohol = st.checkbox("昨日お酒を飲んだ")
        note = st.text_area("気になる症状", placeholder="例：少し肩が凝っている")
        st.markdown('</div>', unsafe_allow_html=True)

    # データ取得と計算
    df = fetch_weather(lat, lon)
    now = datetime.now()
    current_data = df.iloc[(df['time'] - now).abs().argsort()[:1]].iloc[0]
    
    # 直近3時間の気圧変化
    idx_now = (df['time'] - now).abs().argsort()[:1][0]
    p_now = df.iloc[idx_now]['pressure_msl']
    p_old = df.iloc[idx_now-3]['pressure_msl']
    p_drop = p_now - p_old

    # スコア計算 (簡易版)
    score = 0
    reasons = []
    if p_drop <= -2.0: score += 2; reasons.append(f"気圧が急低下しています({p_drop:.1f}hPa/3h)")
    if sleep < 6: score += 2; reasons.append("睡眠が不足しています")
    if alcohol: score += 1; reasons.append("飲酒の影響があるかもしれません")
    
    label, color, emoji = get_risk_info(score)

    with col_chart:
        # 気圧・気温チャート
        chart_df = df[(df['time'] >= now - timedelta(hours=6)) & (df['time'] <= now + timedelta(hours=24))]
        base = alt.Chart(chart_df).encode(x='time:T')
        
        line_p = base.mark_line(color='#8884d8').encode(y=alt.Y('pressure_msl:Q', scale=alt.Scale(zero=False), title='気圧(hPa)'))
        line_t = base.mark_line(color='#ff7300').encode(y=alt.Y('temperature_2m:Q', title='気温(℃)'))
        
        st.altair_chart(alt.layer(line_p, line_t).resolve_scale(y='independent'), use_container_width=True)

    st.divider()

    # 結果表示
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.markdown(f"""
            <div style="background:{color}22; border:2px solid {color}; border-radius:15px; padding:20px; text-align:center;">
                <h2 style="color:{color};">{emoji} {label}</h2>
                <p>トータルリスクスコア: <b>{score}</b></p>
            </div>
        """, unsafe_allow_html=True)
        for r in reasons:
            st.write(f"・{r}")

    with res_col2:
        if st.button("🤖 Gemini 2.5 に養生アドバイスをもらう"):
            if not client:
                st.error("APIキーが設定されていません。")
            else:
                with st.spinner("AIが分析中..."):
                    prompt = f"""
                    あなたはウェルネスアドバイザーです。以下のデータに基づき、今日の過ごし方を日本語でアドバイスしてください。
                    【データ】
                    年齢: {age}, BMI: {calc_bmi(height, weight):.1;f}
                    睡眠: {sleep}時間, 飲酒: {"あり" if alcohol else "なし"}
                    気圧変化: {p_drop:.1f} hPa/3h, 現在の気温: {current_data['temperature_2m']}℃
                    気になる症状: {note}
                    """
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.4)
                    )
                    st.info(response.text)

if __name__ == "__main__":
    main()
