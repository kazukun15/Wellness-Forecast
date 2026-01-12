import os
import json
import datetime as dt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import html
import requests
import streamlit as st
import pandas as pd
import altair as alt

# ==================================================
# Gemini 2.5 Flash 設定
# ==================================================
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@st.cache_resource
def get_gemini_client():
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            return genai.Client(api_key=GEMINI_API_KEY)
        except Exception:
            return None
    return None

client = get_gemini_client()

# ==================================================
# 設定 & 定数
# ==================================================
APP_TITLE = "Wellness Forecast Pro"
PROFILE_PATH = "profile.json"

# 地域プリセット（広島、愛媛を追加）
CITIES = {
    "広島（広島市）": (34.3853, 132.4553),
    "愛媛（松山市）": (33.8392, 132.7655),
    "愛媛（上島町）": (34.25, 133.20),
    "東京": (35.6895, 139.6917),
    "大阪": (34.6937, 135.5023),
    "福岡": (33.5904, 130.4017),
    "手動入力": (None, None)
}

# ==================================================
# プロフィール管理（元のロジックを忠実に再現）
# ==================================================
def default_profile() -> Dict[str, Any]:
    return {
        "age": None, "sex": "未設定", "height_cm": None, "weight_kg": None, "blood_type": "",
        "chronic": {
            "migraine": False, "tension_headache": False, "asthma": False,
            "copd": False, "hypertension": False, "diabetes": False,
            "cvd": False, "anxiety_depression": False,
        },
        "allergy": {"nsaids": False, "antibiotics": False, "food": "", "others": ""},
    }

def load_profile() -> Dict[str, Any]:
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            base = default_profile()
            base.update(data)
            return base
        except Exception:
            return default_profile()
    return default_profile()

def save_profile(profile: Dict[str, Any]):
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

def calc_bmi(h, w):
    if not h or not w or h <= 0: return None
    return w / ((h / 100.0) ** 2)

def calc_profile_base_risk(profile: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = 0
    reasons = []
    age = profile.get("age")
    if age:
        if age >= 60: score += 2; reasons.append("60歳以上で、体調が崩れやすい年齢帯です。")
        elif age >= 40: score += 1; reasons.append("40代以降で、回復に時間がかかりやすい時期です。")
    
    bmi = calc_bmi(profile.get("height_cm"), profile.get("weight_kg"))
    if bmi:
        if bmi < 18.5: score += 1; reasons.append("やせ気味で、冷え・疲れが出やすいことがあります。")
        elif 25 <= bmi < 30: score += 1; reasons.append("BMIがやや高めで、疲労が残りやすい場合があります。")
        elif bmi >= 30: score += 2; reasons.append("肥満（BMI≥30）で、体への負担が大きい状態です。")

    chronic = profile.get("chronic", {})
    if chronic.get("migraine"): score += 1; reasons.append("片頭痛があり、気圧の影響を受けやすいです。")
    if chronic.get("asthma") or chronic.get("copd"): score += 1; reasons.append("呼吸器の持病があります。")
    if chronic.get("hypertension") or chronic.get("cvd"): score += 1; reasons.append("血圧・心臓に注意が必要です。")
    
    return min(score, 3), reasons

# ==================================================
# 天気データ取得 & 予報ロジック（忠実再現）
# ==================================================
@st.cache_data(ttl=3600)
def fetch_weather_detailed(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "hourly": ["pressure_msl", "temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df

def build_daily_forecast(df: pd.DataFrame):
    # 元の「日ごとのリスク集計」ロジックを実装
    df['date'] = df['time'].dt.date
    daily_groups = df.groupby('date')
    results = []
    for date, group in daily_groups:
        pressures = group['pressure_msl'].tolist()
        min_p = min(pressures)
        
        # 3時間最大降圧の計算
        max_drop_3h = 0.0
        for i in range(3, len(pressures)):
            drop = pressures[i] - pressures[i-3]
            if drop < max_drop_3h: max_drop_3h = drop

        score = 0
        reasons = []
        if max_drop_3h <= -6.0: score += 2; reasons.append("急激な気圧低下があります。")
        elif max_drop_3h <= -3.0: score += 1; reasons.append("気圧低下の傾向があります。")
        if min_p < 1005: score += 1; reasons.append("低気圧圏内です。")

        results.append({
            "date": date, "score": score, "reasons": reasons,
            "min_p": min_p, "max_drop": max_drop_3h,
            "temp_range": (group['temperature_2m'].min(), group['temperature_2m'].max())
        })
    return results

# ==================================================
# UI: スタイル & 表示
# ==================================================
def inject_css():
    st.markdown("""
        <style>
        .wf-card { background: white; border-radius: 15px; padding: 20px; border: 1px solid #e0e0e0; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
        .wf-badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-weight: bold; font-size: 0.8em; margin-right: 5px; background: #f0f2f6; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 10px 10px 0 0; padding: 10px 20px; }
        </style>
    """, unsafe_allow_html=True)

# ==================================================
# メインアプリケーション
# ==================================================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")
    inject_css()

    if "profile" not in st.session_state:
        st.session_state.profile = load_profile()
    profile = st.session_state.profile

    st.title(f"🩺 {APP_TITLE}")
    st.caption("最新の Gemini 2.5 Flash と気象データによる精密体調予報")

    tab_today, tab_profile = st.tabs(["🌈 今日の予報", "🧑‍⚕️ プロフィール設定"])

    # --- プロフィールタブ ---
    with tab_profile:
        st.markdown('<div class="wf-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("年齢", 0, 120, int(profile["age"]) if profile["age"] else 40)
            height = st.number_input("身長(cm)", 50.0, 250.0, float(profile["height_cm"]) if profile["height_cm"] else 170.0)
        with col2:
            sex = st.selectbox("性別", ["未設定", "男性", "女性", "その他"], index=["未設定", "男性", "女性", "その他"].index(profile["sex"]))
            weight = st.number_input("体重(kg)", 10.0, 300.0, float(profile["weight_kg"]) if profile["weight_kg"] else 60.0)
        
        st.write("##### 慢性的な症状・持病")
        ch = profile["chronic"]
        c1, c2, c3 = st.columns(3)
        with c1:
            ch["migraine"] = st.checkbox("片頭痛", ch["migraine"])
            ch["anxiety_depression"] = st.checkbox("メンタル不調", ch["anxiety_depression"])
        with c2:
            ch["asthma"] = st.checkbox("喘息", ch["asthma"])
            ch["hypertension"] = st.checkbox("高血圧", ch["hypertension"])
        with c3:
            ch["diabetes"] = st.checkbox("糖尿病", ch["diabetes"])

        if st.button("💾 プロフィールを保存"):
            profile.update({"age": age, "sex": sex, "height_cm": height, "weight_kg": weight, "chronic": ch})
            save_profile(profile)
            st.success("情報を保存しました。")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 今日の予報タブ ---
    with tab_today:
        # 地域選択
        st.markdown('<div class="wf-card">', unsafe_allow_html=True)
        st.subheader("📍 場所と条件")
        col_city, col_lat, col_lon = st.columns([1, 1, 1])
        with col_city:
            city_choice = st.selectbox("地域プリセット", list(CITIES.keys()))
        
        # プリセット選択時の自動入力
        default_lat, default_lon = CITIES[city_choice]
        with col_lat:
            lat = st.number_input("緯度", -90.0, 90.0, default_lat if default_lat else 34.25)
        with col_lon:
            lon = st.number_input("経度", -180.0, 180.0, default_lon if default_lon else 133.20)
        
        col_sleep, col_memo = st.columns([1, 2])
        with col_sleep:
            sleep = st.slider("昨夜の睡眠時間", 0.0, 12.0, 7.0)
            alcohol = st.checkbox("昨日お酒を飲んだ")
        with col_memo:
            user_note = st.text_area("気になる症状メモ", placeholder="例：広島は今日少し冷え込みます...")
        st.markdown('</div>', unsafe_allow_html=True)

        # データ取得と解析
        try:
            df = fetch_weather_detailed(lat, lon)
            daily_forecasts = build_daily_forecast(df)
            
            # 現在のデータ
            now = datetime.now()
            current = df.iloc[(df['time'] - now).abs().argsort()[:1]].iloc[0]
            
            # リスク計算
            base_score, base_reasons = calc_profile_base_risk(profile)
            w_score = daily_forecasts[0]["score"]
            life_score = 2 if sleep < 6 else (0 if sleep > 7 else 1)
            if alcohol: life_score += 1
            
            total_score = base_score + w_score + life_score
            
            # 表示
            c_res, c_chart = st.columns([1, 1.5])
            with c_res:
                color = "#3CB371" if total_score <= 3 else ("#FFD54F" if total_score <= 6 else "#FF6B6B")
                st.markdown(f"""
                    <div style="background:{color}22; border:2px solid {color}; border-radius:15px; padding:20px;">
                        <h2 style="margin:0; color:{color};">リスクスコア: {total_score}</h2>
                        <p style="margin:5px 0;">{"おちついている" if total_score <= 3 else "警戒が必要です"}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("**主な要因:**")
                for r in base_reasons + daily_forecasts[0]["reasons"]:
                    st.write(f"・{r}")
                
                if st.button("🤖 Gemini 2.5 Flash に養生法を聞く"):
                    if not client: st.error("APIキーが必要です。")
                    else:
                        with st.spinner("AI分析中..."):
                            prompt = f"ウェルネスアドバイザーとして助言を。年齢:{age}、持病:{[k for k,v in ch.items() if v]}、睡眠:{sleep}h、気圧変化:{daily_forecasts[0]['max_drop']:.1f}hPa、現在気温:{current['temperature_2m']}度、メモ:{user_note}。簡潔な箇条書きで。"
                            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                            st.info(response.text)

            with c_chart:
                # 気圧グラフの追加
                chart_df = df.head(48) # 48時間分
                base = alt.Chart(chart_df).encode(x=alt.X('time:T', title='時間'))
                line = base.mark_line(color='#42a5f5').encode(y=alt.Y('pressure_msl:Q', scale=alt.Scale(zero=False), title='気圧 (hPa)'))
                st.altair_chart(line.properties(height=250), use_container_width=True)

            st.divider()
            # 週間カード表示
            st.subheader("🗓️ 週間リスク予報")
            cols = st.columns(7)
            for i, f in enumerate(daily_forecasts[:7]):
                with cols[i]:
                    st.markdown(f"""
                        <div class="wf-card" style="text-align:center; padding:10px;">
                            <div style="font-size:0.8em;">{f['date'].strftime('%m/%d')}</div>
                            <div style="font-weight:bold; font-size:1.2em;">Score: {f['score']}</div>
                            <div style="font-size:0.7em;">{f['temp_range'][0]:.0f}〜{f['temp_range'][1]:.0f}℃</div>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"データの取得に失敗しました: {e}")

if __name__ == "__main__":
    main()
