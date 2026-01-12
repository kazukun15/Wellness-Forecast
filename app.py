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
# Optional: Gemini
# ==================================================
GEMINI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ==================================================
# Constants & Settings
# ==================================================
APP_TITLE = "Wellness Forecast"
PROFILE_PATH = "profile.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 日本の主要都市プリセット (緯度, 経度)
CITIES = {
    "東京": (35.6895, 139.6917),
    "大阪": (34.6937, 135.5023),
    "名古屋": (35.1815, 136.9066),
    "札幌": (43.0618, 141.3545),
    "福岡": (33.5904, 130.4017),
    "仙台": (38.2682, 140.8694),
    "広島": (34.3853, 132.4553),
    "那覇": (26.2124, 127.6809),
    "金沢": (36.5613, 136.6562),
    "高松": (34.3428, 134.0466),
    "手動入力": (None, None)
}

# ==================================================
# Gemini Client
# ==================================================
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
# CSS Styling
# ==================================================
def inject_css():
    css = """
    <style>
    :root{
      --wf-text: #2f2f2f;
      --wf-text-sub: #555555;
      --wf-bg-card: rgba(255, 255, 255, 0.85);
      --wf-border: rgba(0,0,0,0.08);
    }
    
    .stApp {
        background: 
          radial-gradient(circle at 15% 10%, rgba(255, 214, 102, 0.25), transparent 40%),
          radial-gradient(circle at 85% 15%, rgba(186, 104, 200, 0.20), transparent 42%),
          radial-gradient(circle at 20% 90%, rgba(129, 199, 132, 0.20), transparent 45%),
          radial-gradient(circle at 90% 85%, rgba(79, 195, 247, 0.20), transparent 45%),
          #fbfbff;
        color: var(--wf-text);
    }

    /* General Text Color Force */
    html, body, [class*="css"], .stMarkdown, div, p, li, label, h1, h2, h3 {
        color: var(--wf-text) !important;
        font-family: "Helvetica Neue", Arial, "Hiragino Kaku Gothic ProN", "Hiragino Sans", Meiryo, sans-serif;
    }
    
    .stMarkdown small {
        color: var(--wf-text-sub) !important;
    }

    /* Cards */
    .wf-card {
        background: var(--wf-bg-card);
        border: 1px solid var(--wf-border);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    .wf-section-title {
        font-size: 1.1rem;
        font-weight: 800;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Streamlit Components adjustment */
    .stButton>button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--wf-text) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--wf-text-sub) !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ==================================================
# Profile Logic
# ==================================================
def default_profile() -> Dict[str, Any]:
    return {
        "age": 40, "sex": "未設定", "height_cm": 170.0, "weight_kg": 60.0,
        "blood_type": "",
        "chronic": {
            "migraine": False, "tension_headache": False, "asthma": False,
            "copd": False, "hypertension": False, "diabetes": False,
            "cvd": False, "anxiety_depression": False,
        },
        "allergy": {
            "nsaids": False, "antibiotics": False, "food": "", "others": "",
        },
    }

def load_profile() -> Dict[str, Any]:
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            base = default_profile()
            # Deep update for nested dicts
            base.update({k: v for k, v in data.items() if k not in ["chronic", "allergy"]})
            base["chronic"].update(data.get("chronic", {}))
            base["allergy"].update(data.get("allergy", {}))
            return base
        except Exception:
            return default_profile()
    return default_profile()

def save_profile(profile: Dict[str, Any]) -> None:
    try:
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def calc_bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    if not height_cm or not weight_kg or height_cm <= 0:
        return None
    return weight_kg / ((height_cm / 100.0) ** 2)

def calc_profile_base_risk(profile: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    age = profile.get("age", 40)
    if age >= 60:
        score += 2; reasons.append("60歳以上 (回復力低下のリスク)")
    elif age >= 40:
        score += 1; reasons.append("40代以降 (体調変化のリスク)")

    bmi = calc_bmi(profile.get("height_cm"), profile.get("weight_kg"))
    if bmi:
        if bmi < 18.5:
            score += 1; reasons.append("低体重 (冷え・スタミナ不足)")
        elif bmi >= 30:
            score += 2; reasons.append("肥満 (循環器・関節への負担)")
        elif bmi >= 25:
            score += 1; reasons.append("過体重 (疲労蓄積)")

    c = profile.get("chronic", {})
    if c.get("migraine"): score += 1; reasons.append("片頭痛 (気圧変化に敏感)")
    if c.get("asthma") or c.get("copd"): score += 1; reasons.append("呼吸器疾患 (気温差・乾燥に敏感)")
    if c.get("hypertension") or c.get("cvd"): score += 1; reasons.append("循環器リスク")
    if c.get("diabetes"): score += 1; reasons.append("血糖・代謝リスク")
    if c.get("anxiety_depression"): score += 1; reasons.append("メンタル不調 (自律神経)")

    return min(score, 3), reasons

def get_profile_summary_text(profile: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"{profile.get('age', '?')}歳")
    parts.append(profile.get('sex', '性別不明'))
    c = profile.get("chronic", {})
    conditions = [k for k, v in c.items() if v]
    if conditions:
        parts.append(f"持病: {', '.join(conditions)}")
    return " / ".join(parts)

# ==================================================
# Weather Logic (Cached)
# ==================================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_weather_data(lat: float, lon: float) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", 
                   "precipitation", "pressure_msl", "wind_speed_10m"],
        "timezone": "auto",
        "past_days": 1,  # Get yesterday's data to calculate pressure trend smoothly
        "forecast_days": 7
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        hourly = data.get("hourly", {})
        if not hourly.get("time"):
            return None, "No data available."
            
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        return df, None
    except Exception as e:
        return None, str(e)

def analyze_weather_risk(df: pd.DataFrame) -> Tuple[int, List[str], Dict[str, float]]:
    """Analyzes the *current* weather risk based on the latest available data point."""
    now = datetime.now()
    # Find the row closest to current time
    current_df = df.iloc[(df['time'] - now).abs().argsort()[:1]]
    
    if current_df.empty:
        return 0, ["データがありません"], {}
        
    row = current_df.iloc[0]
    
    # Calculate pressure drop (compare with 3 hours ago)
    # We need to find the row 3 hours before this row
    idx_3h_ago = (df['time'] - (row['time'] - timedelta(hours=3))).abs().argsort()[:1]
    p_now = row["pressure_msl"]
    p_prev = df.iloc[idx_3h_ago].iloc[0]["pressure_msl"]
    p_drop = p_now - p_prev

    score = 0
    reasons = []
    
    # Pressure
    if p_drop <= -4:
        score += 2; reasons.append(f"急激な気圧低下 ({p_drop:+.1f} hPa/3h)")
    elif p_drop <= -2:
        score += 1; reasons.append(f"気圧低下の傾向 ({p_drop:+.1f} hPa/3h)")
        
    # Temperature/Feels Like
    feels = row["apparent_temperature"]
    if feels <= 5:
        score += 1; reasons.append("強い冷え込み")
    elif feels >= 30:
        score += 1; reasons.append("暑さによる消耗")
        
    # Humidity
    rh = row["relative_humidity_2m"]
    if rh <= 25: score += 1; reasons.append("極度の乾燥")
    elif rh >= 80: score += 1; reasons.append("高湿度・蒸れ")
    
    # Wind/Rain
    if row["precipitation"] >= 1.0: score += 1; reasons.append("降雨")
    if row["wind_speed_10m"] >= 8.0: score += 1; reasons.append("強風")

    snapshot = {
        "temp": row["temperature_2m"],
        "pressure": p_now,
        "pressure_drop": p_drop,
        "humidity": rh,
        "wind": row["wind_speed_10m"],
        "precip": row["precipitation"]
    }
    
    return min(score, 3), reasons, snapshot

def create_weather_chart(df: pd.DataFrame):
    """Creates an Altair chart for Temp & Pressure forecast."""
    now = datetime.now()
    # Filter for next 48 hours
    chart_df = df[(df['time'] >= now - timedelta(hours=3)) & (df['time'] <= now + timedelta(hours=48))].copy()
    
    if chart_df.empty:
        return st.write("No data for chart.")

    base = alt.Chart(chart_df).encode(x=alt.X('time:T', axis=alt.Axis(format='%d日 %H時', title='日時')))

    # Pressure (Area)
    pressure = base.mark_line(color='#A8A4CE', strokeWidth=3).encode(
        y=alt.Y('pressure_msl:Q', scale=alt.Scale(zero=False), axis=alt.Axis(title='気圧 (hPa)', titleColor='#A8A4CE'))
    )
    
    # Temperature (Line)
    temp = base.mark_line(color='#FFD666', strokeWidth=3).encode(
        y=alt.Y('temperature_2m:Q', scale=alt.Scale(zero=False), axis=alt.Axis(title='気温 (℃)', titleColor='#E6B422'))
    )

    chart = alt.layer(pressure, temp).resolve_scale(y='independent').properties(
        height=250, title="今後48時間の気圧(紫)と気温(黄)の推移"
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)


# ==================================================
# Daily & Total Risk
# ==================================================
def calc_daily_lifestyle_risk(
    sleep: float, alcohol: bool, steps: Optional[int], rhr_diff: float
) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    if sleep < 5.5: score += 2; reasons.append("睡眠不足 (5.5h未満)")
    elif sleep < 6.5: score += 1; reasons.append("睡眠不足気味")
    
    if alcohol: score += 1; reasons.append("アルコール摂取翌日")
    
    if rhr_diff >= 8: score += 2; reasons.append("心拍高値 (疲労/ストレス)")
    elif rhr_diff >= 4: score += 1; reasons.append("心拍やや高め")
    
    if steps is not None:
        if steps < 2000: score += 1; reasons.append("活動量不足 (血行不良)")
        elif steps > 15000: score += 1; reasons.append("活動過多 (疲労蓄積)")

    return score, reasons

def get_risk_level_info(total_score: int) -> Tuple[str, str, str]:
    if total_score <= 3: return "良好〜安定", "#4CAF50", "🟢"
    elif total_score <= 6: return "少し注意", "#FFC107", "🟡"
    else: return "要警戒", "#FF5252", "🔴"

# ==================================================
# AI Advice
# ==================================================
def generate_gemini_advice(
    profile_summary: str, risk_label: str, total_score: int, 
    all_reasons: List[str], user_note: str
) -> str:
    if not client:
        return "Gemini APIキーが設定されていません。"

    prompt = f"""
    あなたは親しみやすい「専属ウェルネス・コンシェルジュ」です。
    ユーザーの今日の体調リスクスコアと要因に基づき、今日一日を快適に過ごすための具体的なアドバイスをください。

    【ユーザー情報】
    - プロフィール: {profile_summary}
    - 今日のリスクレベル: {risk_label} (スコア: {total_score}/10)
    - リスク要因: {', '.join(all_reasons)}
    - ユーザーメモ: {user_note}

    【指示】
    1. ユーザーを労う一言から始めてください。
    2. リスク要因に対する具体的な対策（食事、運動、休息、環境調整など）を3つ提案してください。
    3. 全体的に「優しく、前向きな」トーンで。
    4. 医療行為や断定的な診断は避けてください。
    5. 400文字以内でまとめてください。
    """
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash", # Use latest fast model
            contents=prompt,
        )
        return resp.text
    except Exception as e:
        return f"AIアドバイスの生成中にエラーが発生しました: {e}"

# ==================================================
# UI Components
# ==================================================
def render_sidebar_profile(profile):
    st.sidebar.markdown("### 👤 プロフィール設定")
    with st.sidebar.expander("基本情報の編集", expanded=False):
        profile["age"] = st.number_input("年齢", 0, 100, profile["age"])
        profile["sex"] = st.selectbox("性別", ["未設定", "男性", "女性", "その他"], index=["未設定", "男性", "女性", "その他"].index(profile["sex"]))
        profile["height_cm"] = st.number_input("身長(cm)", 0.0, 250.0, profile["height_cm"])
        profile["weight_kg"] = st.number_input("体重(kg)", 0.0, 300.0, profile["weight_kg"])
        
        st.caption("持病・体質")
        c = profile["chronic"]
        c["migraine"] = st.checkbox("片頭痛", c["migraine"])
        c["asthma"] = st.checkbox("喘息・気管支", c["asthma"])
        c["hypertension"] = st.checkbox("高血圧", c["hypertension"])
        c["anxiety_depression"] = st.checkbox("メンタル不調", c["anxiety_depression"])
        
        if st.button("プロフィールを保存"):
            save_profile(profile)
            st.success("保存しました")

def render_sidebar_location():
    st.sidebar.markdown("### 📍 場所設定")
    city_name = st.sidebar.selectbox("エリアを選択", list(CITIES.keys()))
    
    lat, lon = CITIES[city_name]
    
    if city_name == "手動入力":
        lat = st.sidebar.number_input("緯度", -90.0, 90.0, 35.69)
        lon = st.sidebar.number_input("経度", -180.0, 180.0, 139.69)
    
    return lat, lon, city_name

def render_dashboard(profile, lat, lon):
    # 1. Fetch Data
    with st.spinner("気象データを分析中..."):
        df, err = fetch_weather_data(lat, lon)
    
    if err:
        st.error(f"天気データの取得に失敗しました: {err}")
        return

    # 2. Daily Inputs
    st.markdown('<div class="wf-section-title">📝 今日のコンディション</div>', unsafe_allow_html=True)
    with st.container():
        # Using columns for input feels cleaner
        c1, c2, c3 = st.columns(3)
        with c1:
            sleep = st.number_input("睡眠時間 (h)", 0.0, 15.0, 6.5, 0.5)
        with c2:
            rhr_diff = st.number_input("安静時心拍のズレ", -20, 20, 0, help="普段より高いと+、低いと-")
        with c3:
            steps = st.number_input("昨日の歩数", 0, 50000, 6000, 1000)
        
        c4, c5 = st.columns([1, 2])
        with c4:
            alcohol = st.checkbox("昨日飲酒した")
        with c5:
            note = st.text_input("気になる症状・メモ", placeholder="例: 頭が重い、少し風邪気味...")

    # 3. Calculate Risks
    # Profile Risk
    base_score, base_reasons = calc_profile_base_risk(profile)
    # Lifestyle Risk
    life_score, life_reasons = calc_daily_lifestyle_risk(sleep, alcohol, steps, rhr_diff)
    # Weather Risk
    w_score, w_reasons, w_snapshot = analyze_weather_risk(df)
    
    total_score = base_score + life_score + w_score
    risk_label, risk_color, risk_emoji = get_risk_level_info(total_score)
    all_reasons = base_reasons + life_reasons + w_reasons

    st.markdown("---")

    # 4. Main Score Display
    c_main, c_weather = st.columns([1.2, 1])
    
    with c_main:
        st.markdown(f"""
        <div class="wf-card" style="border-left: 6px solid {risk_color};">
            <div style="font-size:0.9rem; color:#666;">今日のウェルネス・スコア</div>
            <div style="font-size:2.4rem; font-weight:900; color:{risk_color}; display:flex; align-items:center; gap:10px;">
                {risk_emoji} {risk_label} <span style="font-size:1.2rem; color:#888;">({total_score} pts)</span>
            </div>
            <div style="margin-top:0.5rem; font-size:0.95rem;">
                <b>要注意ポイント:</b><br>
                {'<br>'.join([f"・{r}" for r in all_reasons]) if all_reasons else "・特になし。素晴らしいコンディションです！"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Advice
        if st.button("🤖 AIアドバイスを生成", type="primary", use_container_width=True):
            with st.spinner("AIがカルテを作成中..."):
                advice = generate_gemini_advice(
                    get_profile_summary_text(profile), 
                    risk_label, total_score, all_reasons, note
                )
                st.info(advice)

    with c_weather:
        st.markdown('<div class="wf-section-title" style="margin-top:0;">🌤 現在の気象状況</div>', unsafe_allow_html=True)
        w_cols = st.columns(2)
        w_cols[0].metric("気圧", f"{w_snapshot['pressure']:.0f} hPa", f"{w_snapshot['pressure_drop']:+.1f} (3h)")
        w_cols[1].metric("気温", f"{w_snapshot['temp']:.1f} ℃")
        w_cols[0].metric("湿度", f"{w_snapshot['humidity']:.0f} %")
        w_cols[1].metric("風速", f"{w_snapshot['wind']:.1f} m/s")
        
        if w_snapshot['pressure_drop'] <= -2.0:
            st.warning("⚠️ 気圧が低下傾向です。頭痛等に注意してください。")

    # 5. Charts
    st.markdown('<div class="wf-section-title">📉 今後の気圧・気温予報</div>', unsafe_allow_html=True)
    create_weather_chart(df)
    
    # 6. Weekly Table
    with st.expander("週間予報の詳細データを見る"):
        # Create a simple summary table
        df['date'] = df['time'].dt.date
        daily = df.groupby('date').agg({
            'temperature_2m': ['min', 'max'],
            'pressure_msl': 'min',
            'precipitation': 'sum'
        }).reset_index()
        daily.columns = ['日付', '最低気温', '最高気温', '最低気圧', '降水量']
        st.dataframe(daily, hide_index=True, use_container_width=True)


# ==================================================
# Main App
# ==================================================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🌿", layout="wide")
    inject_css()
    
    # Session State Init
    if "profile" not in st.session_state:
        st.session_state.profile = load_profile()
    
    # Header
    st.markdown(f'<h1 style="margin-bottom:0;">🌿 {APP_TITLE}</h1>', unsafe_allow_html=True)
    st.caption("気象データとあなたの体調プロフィールから、今日のリスクを予測します。")

    # Sidebar
    lat, lon, city_name = render_sidebar_location()
    st.sidebar.markdown("---")
    render_sidebar_profile(st.session_state.profile)
    
    if lat is None:
        st.warning("👈 サイドバーから場所を設定してください。")
        return

    # Dashboard
    st.markdown(f"### 📍 {city_name} の予報")
    render_dashboard(st.session_state.profile, lat, lon)

if __name__ == "__main__":
    main()
