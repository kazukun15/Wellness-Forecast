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
# ロジック
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
        if age >= 60: score += 2; reasons.append("60歳以上 (体調変化に注意)")
        elif age >= 40: score += 1; reasons.append("40代以降 (回復力の低下)")
    
    bmi = calc_bmi(profile.get("height_cm"), profile.get("weight_kg"))
    if bmi:
        if bmi < 18.5: score += 1; reasons.append("低体重 (冷え・スタミナ不足)")
        elif 25 <= bmi < 30: score += 1; reasons.append("BMI高め (疲労蓄積)")
        elif bmi >= 30: score += 2; reasons.append("肥満傾向 (身体的負荷)")

    chronic = profile.get("chronic", {})
    if chronic.get("migraine"): score += 1; reasons.append("片頭痛持ち (気圧敏感)")
    if chronic.get("asthma") or chronic.get("copd"): score += 1; reasons.append("呼吸器疾患")
    if chronic.get("hypertension") or chronic.get("cvd"): score += 1; reasons.append("循環器リスク")
    
    return min(score, 3), reasons

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
    df['date'] = df['time'].dt.date
    daily_groups = df.groupby('date')
    results = []
    for date, group in daily_groups:
        pressures = group['pressure_msl'].tolist()
        min_p = min(pressures)
        
        max_drop_3h = 0.0
        for i in range(3, len(pressures)):
            drop = pressures[i] - pressures[i-3]
            if drop < max_drop_3h: max_drop_3h = drop

        score = 0
        reasons = []
        if max_drop_3h <= -6.0: score += 2; reasons.append("急激な気圧低下")
        elif max_drop_3h <= -3.0: score += 1; reasons.append("気圧低下傾向")
        if min_p < 1005: score += 1; reasons.append("低気圧圏")

        results.append({
            "date": date, "score": score, "reasons": reasons,
            "min_p": min_p, "max_drop": max_drop_3h,
            "temp_range": (group['temperature_2m'].min(), group['temperature_2m'].max())
        })
    return results

# ==================================================
# UI Styling & Components
# ==================================================
def inject_custom_css():
    st.markdown("""
        <style>
        /* 全体のフォント設定 */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');
        html, body, [class*="css"] { font-family: 'Noto Sans JP', sans-serif; }
        
        /* カードスタイル */
        .wf-card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            border: 1px solid rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }
        .wf-card:hover { transform: translateY(-2px); }

        /* リスクスコア表示 */
        .score-container {
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            color: white;
            margin-bottom: 15px;
        }
        .score-val { font-size: 3rem; font-weight: 800; line-height: 1; }
        .score-label { font-size: 1.1rem; font-weight: bold; opacity: 0.9; margin-top:5px; }
        
        /* 週間予報のスクロールコンテナ */
        .forecast-scroll {
            display: flex;
            overflow-x: auto;
            gap: 15px;
            padding: 10px 5px 20px 5px;
            scrollbar-width: thin;
        }
        .forecast-item {
            flex: 0 0 auto;
            width: 130px;
            background: #fff;
            border-radius: 12px;
            padding: 15px 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid #f0f0f0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: space-between;
        }
        .f-date { font-size: 0.85rem; color: #666; font-weight: bold; margin-bottom: 5px; }
        .f-icon { font-size: 2rem; margin: 5px 0; }
        .f-temp { font-size: 0.8rem; color: #555; margin-top: 5px; background: #f5f5f5; padding: 2px 8px; border-radius: 10px; }
        .f-badge { font-size: 0.7rem; color: white; padding: 2px 8px; border-radius: 4px; margin-top: 5px; width: 100%; }
        
        /* Streamlitのデフォルト調整 */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: none; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            background-color: transparent;
            border: 1px solid transparent;
            padding: 8px 16px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #f0f7ff;
            color: #1976d2;
            border: 1px solid #e3f2fd;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

def get_risk_design(score: int):
    # スコアに応じた色とアイコン定義
    if score <= 3:
        return {"color": "#4CAF50", "bg": "linear-gradient(135deg, #66BB6A 0%, #43A047 100%)", "text": "良好", "icon": "😊", "sub": "安定しています"}
    elif score <= 5:
        return {"color": "#FFA726", "bg": "linear-gradient(135deg, #FFB74D 0%, #F57C00 100%)", "text": "注意", "icon": "😐", "sub": "無理は禁物"}
    else:
        return {"color": "#EF5350", "bg": "linear-gradient(135deg, #EF5350 0%, #D32F2F 100%)", "text": "警戒", "icon": "😫", "sub": "休息を優先"}

# ==================================================
# メインアプリケーション
# ==================================================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")
    inject_custom_css()

    if "profile" not in st.session_state:
        st.session_state.profile = load_profile()
    profile = st.session_state.profile

    # ヘッダーエリア
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title(f"{APP_TITLE}")
        st.markdown("<span style='color:#666;'>AIと気象データであなたの体調をプレビューする</span>", unsafe_allow_html=True)
    with col_h2:
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
             st.success("✅ Gemini AI Active")
        else:
             st.warning("⚠️ Gemini AI Inactive")

    st.markdown("---")

    tab_today, tab_profile = st.tabs(["🌈 今日の予報・週間天気", "👤 プロフィール設定"])

    # --- プロフィールタブ ---
    with tab_profile:
        st.markdown("### あなたの基本情報")
        st.info("ここで設定した情報は、リスク計算の基礎データとして使用されます。")
        
        with st.container():
            st.markdown('<div class="wf-card">', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                age = st.number_input("年齢", 0, 120, int(profile["age"]) if profile["age"] else 40)
            with c2:
                sex = st.selectbox("性別", ["未設定", "男性", "女性", "その他"], index=["未設定", "男性", "女性", "その他"].index(profile["sex"]))
            with c3:
                height = st.number_input("身長 (cm)", 50.0, 250.0, float(profile["height_cm"]) if profile["height_cm"] else 170.0)
            with c4:
                weight = st.number_input("体重 (kg)", 10.0, 300.0, float(profile["weight_kg"]) if profile["weight_kg"] else 60.0)
            
            st.markdown("#### 🏥 持病・慢性症状")
            ch = profile["chronic"]
            
            # チェックボックスを綺麗に並べる
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                ch["migraine"] = st.checkbox("⚡ 片頭痛", ch["migraine"])
                ch["tension_headache"] = st.checkbox("🤕 緊張型頭痛", ch["tension_headache"])
            with cc2:
                ch["asthma"] = st.checkbox("🌬️ 喘息", ch["asthma"])
                ch["copd"] = st.checkbox("🫁 COPD", ch["copd"])
            with cc3:
                ch["hypertension"] = st.checkbox("🩸 高血圧", ch["hypertension"])
                ch["diabetes"] = st.checkbox("🍬 糖尿病", ch["diabetes"])
            with cc4:
                ch["anxiety_depression"] = st.checkbox("☁️ メンタル不調", ch["anxiety_depression"])
                ch["cvd"] = st.checkbox("❤️ 心疾患", ch["cvd"])

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 プロフィールを保存", type="primary", use_container_width=True):
                profile.update({"age": age, "sex": sex, "height_cm": height, "weight_kg": weight, "chronic": ch})
                save_profile(profile)
                st.toast("プロフィールを更新しました", icon="✅")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- 今日の予報タブ ---
    with tab_today:
        # 設定エリア
        with st.expander("📍 地域・体調入力の設定", expanded=True):
            ec1, ec2, ec3 = st.columns([1.2, 1, 1.5])
            with ec1:
                city_choice = st.selectbox("地域を選択", list(CITIES.keys()))
                default_lat, default_lon = CITIES[city_choice]
            with ec2:
                lat = st.number_input("緯度", -90.0, 90.0, default_lat if default_lat else 34.25, label_visibility="collapsed")
                lon = st.number_input("経度", -180.0, 180.0, default_lon if default_lon else 133.20, label_visibility="collapsed")
                st.caption(f"Lat: {lat}, Lon: {lon}")
            with ec3:
                sleep = st.slider("💤 昨夜の睡眠時間", 0.0, 12.0, 7.0, 0.5)
                alcohol = st.checkbox("🍺 昨日お酒を飲んだ")
                user_note = st.text_input("📝 気になる症状メモ", placeholder="例：少し頭が重い...")

        # データ処理
        try:
            df = fetch_weather_detailed(lat, lon)
            daily_forecasts = build_daily_forecast(df)
            
            # リスク計算
            now = datetime.now()
            current = df.iloc[(df['time'] - now).abs().argsort()[:1]].iloc[0]
            
            base_score, base_reasons = calc_profile_base_risk(profile)
            w_score = daily_forecasts[0]["score"]
            
            life_score = 0
            life_reasons = []
            if sleep < 6: 
                life_score += 2
                life_reasons.append("睡眠不足")
            elif sleep > 9: 
                life_score += 1
            if alcohol: 
                life_score += 1
                life_reasons.append("アルコール摂取")
            
            total_score = base_score + w_score + life_score
            design = get_risk_design(total_score)

            # --- メインダッシュボード ---
            st.markdown("### 📅 本日の体調予報")
            
            col_main_L, col_main_R = st.columns([1, 2])
            
            # 左カラム：スコア表示
            with col_main_L:
                st.markdown(f"""
                    <div class="wf-card" style="padding:0; overflow:hidden;">
                        <div class="score-container" style="background: {design['bg']};">
                            <div style="font-size:4rem; margin-bottom:10px;">{design['icon']}</div>
                            <div class="score-val">Lv.{total_score}</div>
                            <div class="score-label">{design['text']}</div>
                            <div style="font-size:0.8rem; margin-top:5px; opacity:0.8;">{design['sub']}</div>
                        </div>
                        <div style="padding: 15px;">
                            <b style="color:#555;">⚠️ リスク要因:</b>
                            <ul style="font-size:0.9rem; color:#666; padding-left:20px; margin-top:5px;">
                                {''.join([f'<li>{r}</li>' for r in base_reasons + daily_forecasts[0]["reasons"] + life_reasons])}
                            </ul>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if st.button("✨ Geminiにアドバイスをもらう", type="primary", use_container_width=True):
                    if not client: 
                        st.error("APIキーが設定されていません")
                    else:
                        with st.spinner("AIが分析中..."):
                            prompt = f"""
                            ウェルネスアドバイザーになりきってください。以下のユーザー情報に基づき、今日一日を快適に過ごすための具体的なアドバイスを3点、簡潔に教えてください。
                            ユーザー属性: {age}歳 {sex}
                            リスクスコア: {total_score} ({design['text']})
                            要因: {base_reasons + daily_forecasts[0]["reasons"] + life_reasons}
                            気象: 気圧変化 {daily_forecasts[0]['max_drop']:.1f}hPa, 気温 {current['temperature_2m']}℃
                            メモ: {user_note}
                            """
                            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                            st.info(response.text)

            # 右カラム：グラフ
            with col_main_R:
                st.markdown('<div class="wf-card">', unsafe_allow_html=True)
                st.markdown("###### 📉 気圧とリスクの推移 (48時間)")
                
                chart_df = df.head(48).copy()
                
                base = alt.Chart(chart_df).encode(x=alt.X('time:T', title=None, axis=alt.Axis(format='%H:%M')))
                
                area = base.mark_area(
                    line={'color':'#42a5f5'},
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[alt.GradientStop(color='#42a5f5', offset=0),
                               alt.GradientStop(color='rgba(255,255,255,0)', offset=1)],
                        x1=1, x2=1, y1=1, y2=0
                    )
                ).encode(
                    y=alt.Y('pressure_msl:Q', scale=alt.Scale(zero=False, padding=1), title='気圧 (hPa)'),
                    tooltip=['time', 'pressure_msl', 'temperature_2m']
                )
                
                points = base.mark_circle(size=60, color='#1976D2').encode(
                    y='pressure_msl:Q',
                    tooltip=['time', 'pressure_msl']
                )

                st.altair_chart((area + points).properties(height=300), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # --- 週間予報（カレンダー）---
            st.subheader("🗓️ 週間リスクカレンダー")
            
            # HTML構築
            # 注意: ここでf-string内のインデントをなくし、Markdownのコードブロック誤認を防止しています
            forecast_html = '<div class="forecast-scroll">'
            
            week_days = ["月", "火", "水", "木", "金", "土", "日"]
            
            for f in daily_forecasts[:10]:
                d_score = f['score'] + base_score
                d_design = get_risk_design(d_score)
                wd = week_days[f['date'].weekday()]
                date_str = f['date'].strftime('%m/%d')
                
                # HTMLを一行、またはインデントなしで結合
                forecast_html += f"""
<div class="forecast-item">
<div class="f-date">{date_str} ({wd})</div>
<div class="f-icon">{d_design['icon']}</div>
<div class="f-badge" style="background:{d_design['color']};">Lv.{d_score}</div>
<div class="f-temp">🌡️ {f['temp_range'][0]:.0f}-{f['temp_range'][1]:.0f}℃</div>
</div>"""
            
            forecast_html += '</div>'
            
            st.markdown(forecast_html, unsafe_allow_html=True)
            st.caption("※ 横にスクロールして予報を確認できます")

        except Exception as e:
            st.error(f"データ取得エラー: {e}")
            st.code(str(e))

if __name__ == "__main__":
    main()
