import os
import json
import datetime as dt
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import calendar as pycal
import html  # ✅ 追加：Gemini出力のHTML崩れ対策

import requests
import streamlit as st

# ==================================================
# Optional: Gemini
# ==================================================
GEMINI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# ==================================================
# Optional: FullCalendar for Streamlit
# ==================================================
CALENDAR_AVAILABLE = False
try:
    from streamlit_calendar import calendar as st_calendar
    CALENDAR_AVAILABLE = True
except Exception:
    CALENDAR_AVAILABLE = False


# ==================================================
# Settings
# ==================================================
APP_TITLE = "Wellness Forecast"
PROFILE_PATH = "profile.json"

DEFAULT_LAT = 34.25
DEFAULT_LON = 133.20

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None


# ==================================================
# Utilities
# ==================================================
def safe_html_text(text: Optional[str]) -> str:
    """✅ Gemini出力などをHTML崩れさせず表示するためのエスケープ"""
    if not text:
        return ""
    return html.escape(text).replace("\n", "<br>")


# ==================================================
# Profile
# ==================================================
def default_profile() -> Dict[str, Any]:
    return {
        "age": None,
        "sex": "未設定",
        "height_cm": None,
        "weight_kg": None,
        "blood_type": "",
        "chronic": {
            "migraine": False,
            "tension_headache": False,
            "asthma": False,
            "copd": False,
            "hypertension": False,
            "diabetes": False,
            "cvd": False,
            "anxiety_depression": False,
        },
        "allergy": {
            "nsaids": False,
            "antibiotics": False,
            "food": "",
            "others": "",
        },
    }


def load_profile() -> Dict[str, Any]:
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            base = default_profile()
            base.update(data)
            for k, v in default_profile()["chronic"].items():
                base["chronic"].setdefault(k, v)
            for k, v in default_profile()["allergy"].items():
                base["allergy"].setdefault(k, v)
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
    h_m = height_cm / 100.0
    return weight_kg / (h_m * h_m)


def calc_profile_base_risk(profile: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []

    age = profile.get("age")
    if age is not None:
        if age >= 60:
            score += 2
            reasons.append("60歳以上で、体調が崩れやすい年齢帯です。")
        elif age >= 40:
            score += 1
            reasons.append("40代以降で、回復に時間がかかりやすい時期です。")

    bmi = calc_bmi(profile.get("height_cm"), profile.get("weight_kg"))
    if bmi is not None:
        if bmi < 18.5:
            score += 1
            reasons.append("やせ気味（BMI<18.5）で、冷え・疲れが出やすいことがあります。")
        elif 25 <= bmi < 30:
            score += 1
            reasons.append("BMIがやや高めで、疲労が残りやすい場合があります。")
        elif bmi >= 30:
            score += 2
            reasons.append("肥満（BMI≥30）で、体への負担が大きい状態です。")

    chronic = profile.get("chronic", {})
    if chronic.get("migraine"):
        score += 1
        reasons.append("片頭痛があり、気圧や睡眠の影響を受けやすいです。")
    if chronic.get("asthma") or chronic.get("copd"):
        score += 1
        reasons.append("呼吸器の持病があり、寒さや風邪の影響を受けやすいです。")
    if chronic.get("hypertension") or chronic.get("cvd"):
        score += 1
        reasons.append("血圧・心臓に注意が必要な背景があります。")
    if chronic.get("diabetes"):
        score += 1
        reasons.append("血糖の影響で体調変化が出やすい場合があります。")
    if chronic.get("anxiety_depression"):
        score += 1
        reasons.append("ストレス・睡眠の影響を受けやすい背景があります。")

    return min(score, 3), reasons


def summarize_profile_for_gemini(profile: Dict[str, Any]) -> str:
    parts = []
    age = profile.get("age")

    if age is not None:
        if age < 30:
            parts.append("20〜30代前半")
        elif age < 40:
            parts.append("30代後半")
        elif age < 50:
            parts.append("40代")
        elif age < 60:
            parts.append("50代")
        else:
            parts.append("60代以上")
    else:
        parts.append("年齢不明")

    bmi = calc_bmi(profile.get("height_cm"), profile.get("weight_kg"))
    if bmi is not None:
        if bmi < 18.5:
            parts.append("やせ気味")
        elif bmi >= 30:
            parts.append("肥満傾向")
        elif bmi >= 25:
            parts.append("少しぽっちゃり")
        else:
            parts.append("ほぼ標準体型")

    chronic = profile.get("chronic", {})
    tags = []
    if chronic.get("migraine"):
        tags.append("片頭痛")
    if chronic.get("asthma") or chronic.get("copd"):
        tags.append("呼吸器")
    if chronic.get("hypertension") or chronic.get("cvd"):
        tags.append("血圧・心臓")
    if chronic.get("diabetes"):
        tags.append("糖代謝")
    if chronic.get("anxiety_depression"):
        tags.append("メンタル")
    if tags:
        parts.append("注意点: " + "・".join(tags))

    allergy = profile.get("allergy", {})
    if allergy.get("nsaids"):
        parts.append("NSAIDsに注意")

    return " / ".join(parts)


# ==================================================
# Open-Meteo fetch
# ==================================================
def fetch_weather_from_open_meteo(latitude: float, longitude: float) -> Tuple[Optional[float], Optional[float], str, Optional[Dict[str, Any]]]:
    url = "https://api.open-meteo.com/v1/forecast"
    hourly_fields = [
        "pressure_msl",
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "wind_speed_10m",
        "wind_gusts_10m",
        "cloud_cover",
    ]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "auto",
        "hourly": ",".join(hourly_fields),
    }

    try:
        resp = requests.get(url, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return None, None, "天気データの時刻情報が取得できませんでした。", None

        def get_series(name: str):
            s = hourly.get(name)
            return s if isinstance(s, list) and len(s) == len(times) else None

        series = {"time": times}
        for f in hourly_fields:
            series[f] = get_series(f)

        if series.get("pressure_msl") is None:
            return None, None, "気圧データが取得できませんでした。", None

        pressures = [float(x) for x in series["pressure_msl"]]
        latest_p = float(pressures[-1])

        pressure_drop_3h = None
        if len(pressures) >= 4:
            pressure_drop_3h = latest_p - float(pressures[-4])

        def latest_float(key: str) -> Optional[float]:
            s = series.get(key)
            if not s:
                return None
            try:
                return float(s[-1])
            except Exception:
                return None

        temp = latest_float("temperature_2m")
        feels = latest_float("apparent_temperature")
        rh = latest_float("relative_humidity_2m")
        wind = latest_float("wind_speed_10m")
        prec = latest_float("precipitation")

        msg_lines = ["天気データを取得しました。"]
        msg_lines.append(f"・現在の気圧: {latest_p:.1f} hPa")
        if pressure_drop_3h is not None:
            msg_lines.append(f"・直近3時間の気圧変化: {pressure_drop_3h:+.1f} hPa")
        if temp is not None:
            msg_lines.append(f"・気温: {temp:.1f} ℃")
        if feels is not None:
            msg_lines.append(f"・体感温度: {feels:.1f} ℃")
        if rh is not None:
            msg_lines.append(f"・湿度: {rh:.0f} %")
        if wind is not None:
            msg_lines.append(f"・風速: {wind:.1f} m/s")
        if prec is not None:
            msg_lines.append(f"・降水（1時間）: {prec:.1f} mm")

        bundle = {"hourly": series, "timezone": data.get("timezone", "")}
        return pressure_drop_3h, latest_p, "\n".join(msg_lines), bundle

    except Exception as e:
        return None, None, f"天気データの取得に失敗しました: {e}", None


# ==================================================
# Risk calculation
# ==================================================
def calc_daily_risk(
    sleep_hours: float,
    alcohol: bool,
    pressure_drop_3h: Optional[float],
    resting_hr_diff: float,
    steps: Optional[int],
) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []

    if pressure_drop_3h is not None:
        if pressure_drop_3h <= -4:
            score += 2
            reasons.append("直近3時間で4hPa以上の気圧低下がありそうです。")
        elif pressure_drop_3h <= -2:
            score += 1
            reasons.append("直近3時間で2〜4hPa程度の気圧低下がありそうです。")

    if sleep_hours < 5.5:
        score += 2
        reasons.append("睡眠がかなり少なめ（5.5時間未満）です。")
    elif sleep_hours < 6.5:
        score += 1
        reasons.append("睡眠が少し少なめ（6.5時間未満）です。")

    if alcohol:
        score += 1
        reasons.append("前日にお酒があり、体の負担が残ることがあります。")

    if resting_hr_diff >= 8:
        score += 2
        reasons.append("安静時心拍がいつもより8bpm以上高めです。")
    elif resting_hr_diff >= 4:
        score += 1
        reasons.append("安静時心拍が少し高めです。")

    if steps is not None:
        if steps < 2000:
            score += 1
            reasons.append("歩数が少なく、だるさが出やすいことがあります。")
        elif steps > 15000:
            score += 1
            reasons.append("活動量が多く、疲れが残ることがあります。")

    return score, reasons


def add_weather_risk_from_latest(hourly: Dict[str, Any]) -> Tuple[int, List[str], Dict[str, Optional[float]]]:
    score = 0
    reasons: List[str] = []

    def lastf(key: str) -> Optional[float]:
        s = hourly.get(key)
        if not s:
            return None
        try:
            return float(s[-1])
        except Exception:
            return None

    temp = lastf("temperature_2m")
    feels = lastf("apparent_temperature")
    rh = lastf("relative_humidity_2m")
    prec = lastf("precipitation")
    wind = lastf("wind_speed_10m")

    ref = feels if feels is not None else temp

    if ref is not None:
        if ref <= 0:
            score += 2
            reasons.append("体感がかなり寒い（0℃以下）ため、冷えの負担が増えます。")
        elif ref <= 5:
            score += 1
            reasons.append("体感が寒め（5℃以下）で、冷えに注意です。")
        elif ref >= 33:
            score += 2
            reasons.append("体感がかなり暑い（33℃以上）ため、だるさが出やすいです。")
        elif ref >= 30:
            score += 1
            reasons.append("体感が暑め（30℃以上）で、負担が増えやすいです。")

    if rh is not None:
        if rh <= 25:
            score += 1
            reasons.append("湿度がかなり低め（25%以下）で、乾燥の負担が出やすいです。")
        elif rh >= 80:
            score += 1
            reasons.append("湿度が高め（80%以上）で、だるさが出やすいです。")

    if prec is not None:
        if prec >= 5:
            score += 2
            reasons.append("降水が強め（1時間5mm以上）で、冷えや負担が増えやすいです。")
        elif prec >= 1:
            score += 1
            reasons.append("雨（降水）があり、負担が増えやすいです。")

    if wind is not None:
        if wind >= 10:
            score += 2
            reasons.append("風がかなり強め（10m/s以上）で、体感が下がりやすいです。")
        elif wind >= 8:
            score += 1
            reasons.append("風が強めで、体感が下がりやすいです。")

    snapshot = {
        "temperature_2m": temp,
        "apparent_temperature": feels,
        "relative_humidity_2m": rh,
        "precipitation": prec,
        "wind_speed_10m": wind,
    }
    return score, reasons, snapshot


def classify_total_risk(total_score: int) -> Tuple[str, str, str]:
    if total_score <= 2:
        return "おちついている", "#3CB371", "🟢"
    elif total_score <= 6:
        return "少し注意したい", "#FFD54F", "🟡"
    else:
        return "今日はかなり慎重に", "#FF6B6B", "🔴"


# ==================================================
# Forecast
# ==================================================
def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def build_daily_forecast_from_hourly(series: Dict[str, Any], days_ahead: int = 14) -> List[Dict[str, Any]]:
    times = series.get("time", [])
    if not times:
        return []

    idx_by_date: Dict[dt.date, List[int]] = {}
    for i, t in enumerate(times):
        dtt = _parse_iso(t)
        if not dtt:
            continue
        idx_by_date.setdefault(dtt.date(), []).append(i)

    today = dt.date.today()
    dates = sorted([d for d in idx_by_date.keys() if d >= today])[:max(1, days_ahead)]
    if not dates:
        return []

    def get_f(key: str, i: int) -> Optional[float]:
        arr = series.get(key)
        if not arr:
            return None
        try:
            return float(arr[i])
        except Exception:
            return None

    out: List[Dict[str, Any]] = []
    for d in dates:
        idxs = idx_by_date[d]
        if not idxs:
            continue

        pressures = [get_f("pressure_msl", i) for i in idxs]
        pressures = [p for p in pressures if p is not None]
        if not pressures:
            continue

        min_pressure = float(min(pressures))

        max_drop_3h = 0.0
        for j in range(3, len(idxs)):
            p_now = get_f("pressure_msl", idxs[j])
            p_prev = get_f("pressure_msl", idxs[j - 3])
            if p_now is None or p_prev is None:
                continue
            drop = float(p_now - p_prev)
            if drop < max_drop_3h:
                max_drop_3h = drop

        temps = [get_f("temperature_2m", i) for i in idxs if get_f("temperature_2m", i) is not None]
        feels = [get_f("apparent_temperature", i) for i in idxs if get_f("apparent_temperature", i) is not None]
        rh = [get_f("relative_humidity_2m", i) for i in idxs if get_f("relative_humidity_2m", i) is not None]
        prec = [get_f("precipitation", i) for i in idxs if get_f("precipitation", i) is not None]
        wind = [get_f("wind_speed_10m", i) for i in idxs if get_f("wind_speed_10m", i) is not None]

        min_temp = float(min(temps)) if temps else None
        max_temp = float(max(temps)) if temps else None
        min_feels = float(min(feels)) if feels else None
        max_feels = float(max(feels)) if feels else None
        min_rh = float(min(rh)) if rh else None
        max_rh = float(max(rh)) if rh else None
        prec_sum = float(sum(prec)) if prec else 0.0
        wind_max = float(max(wind)) if wind else None

        score = 0
        reasons: List[str] = []

        if max_drop_3h <= -6.0:
            score += 2
            reasons.append("3時間で6hPa以上の急な気圧低下がありそうです。")
        elif max_drop_3h <= -3.0:
            score += 1
            reasons.append("3時間で3〜6hPa程度の気圧低下がありそうです。")

        if min_pressure < 1000.0:
            score += 2
            reasons.append("気圧が1000hPa未満の時間帯がありそうです。")
        elif min_pressure < 1005.0:
            score += 1
            reasons.append("気圧が1005hPa未満の時間帯がありそうです。")

        ref_min = min_feels if min_feels is not None else min_temp
        ref_max = max_feels if max_feels is not None else max_temp

        if ref_min is not None:
            if ref_min <= 0:
                score += 2
                reasons.append("体感がかなり寒い（0℃以下）の時間帯がありそうです。")
            elif ref_min <= 5:
                score += 1
                reasons.append("体感が寒め（5℃以下）の時間帯がありそうです。")

        if ref_max is not None:
            if ref_max >= 33:
                score += 2
                reasons.append("体感がかなり暑い（33℃以上）の時間帯がありそうです。")
            elif ref_max >= 30:
                score += 1
                reasons.append("体感が暑め（30℃以上）の時間帯がありそうです。")

        if min_temp is not None and max_temp is not None:
            swing = max_temp - min_temp
            if swing >= 12:
                score += 2
                reasons.append("日内の気温差が大きめ（12℃以上）です。")
            elif swing >= 8:
                score += 1
                reasons.append("日内の気温差がやや大きめ（8℃以上）です。")

        if min_rh is not None and min_rh <= 25:
            score += 1
            reasons.append("湿度がかなり低い（25%以下）時間帯がありそうです。")
        if max_rh is not None and max_rh >= 80:
            score += 1
            reasons.append("湿度が高い（80%以上）時間帯がありそうです。")

        if prec_sum >= 20:
            score += 2
            reasons.append("降水量が多め（合計20mm以上）になりそうです。")
        elif prec_sum >= 5:
            score += 1
            reasons.append("雨が降りそう（合計5mm以上）です。")

        if wind_max is not None:
            if wind_max >= 10:
                score += 2
                reasons.append("風がかなり強め（最大10m/s以上）になりそうです。")
            elif wind_max >= 8:
                score += 1
                reasons.append("風が強め（最大8m/s以上）になりそうです。")

        label = "低め" if score <= 2 else ("やや高め" if score <= 5 else "高め")

        out.append(
            {
                "date": d,
                "label": label,
                "score": score,
                "min_pressure": min_pressure,
                "max_drop_3h": float(max_drop_3h),
                "min_temp": min_temp,
                "max_temp": max_temp,
                "min_feels": min_feels,
                "max_feels": max_feels,
                "min_rh": min_rh,
                "max_rh": max_rh,
                "prec_sum": float(prec_sum),
                "wind_max": wind_max,
                "reasons": reasons,
            }
        )

    return out


# ==================================================
# Gemini advice
# ==================================================
def call_gemini_for_advice(
    profile_summary: str,
    risk_label: str,
    total_score: int,
    base_score: int,
    daily_score: int,
    base_reasons: List[str],
    daily_reasons: List[str],
    user_note: str,
) -> Optional[str]:
    if client is None:
        return None

    base_text = "\n".join(f"- {r}" for r in base_reasons) if base_reasons else "特になし"
    daily_text = "\n".join(f"- {r}" for r in daily_reasons) if daily_reasons else "特になし"

    prompt = f"""
あなたは日本人の成人に対して、医学的常識に沿った「ふだんの養生アドバイス」を行う専門家です。
診断や治療の指示は行わず、日常生活の工夫と、必要な場合の受診の目安だけを伝えてください。

【背景（プロフィール要約）】
{profile_summary}

【今日の総合リスク】
- レベル: {risk_label}
- トータルスコア: {total_score}（ベース {base_score} + 今日の条件 {daily_score}）

【ベース要因】
{base_text}

【今日の要因】
{daily_text}

【本人メモ】
{user_note if user_note else "特になし"}

【出力条件】
- 日本語・ですます調
- 800字以内
- 構成：
  1) 今日の状態イメージ（3〜5行）
  2) 今日のおすすめ（箇条書き3〜5）
  3) 受診の目安（2〜4）
- 薬の具体名は出さない
- 危険サインが疑われる場合は「早めに医療機関の受診を検討してください」を入れる
""".strip()

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4),
        )
        return resp.text
    except Exception as e:
        return f"Geminiの呼び出しに失敗しました: {e}"


# ==================================================
# UI: CSS（上部が切れない）
# ==================================================
def inject_css():
    css = """
    <style>
    .stApp {
        background:
          radial-gradient(circle at 15% 10%, rgba(255, 214, 102, 0.35), transparent 40%),
          radial-gradient(circle at 85% 15%, rgba(186, 104, 200, 0.28), transparent 42%),
          radial-gradient(circle at 20% 90%, rgba(129, 199, 132, 0.28), transparent 45%),
          radial-gradient(circle at 90% 85%, rgba(79, 195, 247, 0.25), transparent 45%),
          #fbfbff;
        overflow-x: hidden;
    }
    body { overflow-x: hidden; }
    html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }

    .block-container {
        max-width: 980px;
        padding-top: calc(env(safe-area-inset-top) + 4.6rem);
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    @media (max-width: 640px) {
        .block-container {
            padding-top: calc(env(safe-area-inset-top) + 5.6rem);
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }
    }

    .wf-title {
        font-size: 1.7rem; font-weight: 900; letter-spacing: .2px;
        background: linear-gradient(90deg, #42a5f5, #ab47bc, #66bb6a);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        display: flex; gap: .4rem; align-items: center; margin-bottom: .15rem;
    }
    .wf-sub { font-size: .95rem; opacity: .86; margin-bottom: .55rem; }

    .wf-card {
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 18px;
        padding: 12px 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        margin-top: .6rem;
    }
    .wf-section {
        font-size: 1.05rem;
        font-weight: 850;
        margin-top: 1rem;
        margin-bottom: .45rem;
        display:flex;
        align-items:center;
        gap:.35rem;
    }

    .stButton>button {
        border-radius: 14px !important;
        padding: 0.58rem 0.82rem !important;
        font-weight: 800 !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.05) !important;
    }

    .fc { font-size: 1.05rem; }
    .fc .fc-toolbar-title { font-size: 1.25rem; font-weight: 900; }
    .fc .fc-daygrid-day-number { font-weight: 900; }
    .fc .fc-daygrid-day-frame { min-height: 92px; }
    @media (max-width: 640px) {
        .fc { font-size: 0.95rem; }
        .fc .fc-daygrid-day-frame { min-height: 78px; }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def risk_card(label: str, color: str, emoji: str, total_score: int, base_score: int, daily_score: int):
    bg = f"{color}22"
    st.markdown(
        f"""
        <div class="wf-card" style="border-color:{color}44;background:{bg};">
          <div style="font-size:1.1rem;font-weight:900;display:flex;gap:.45rem;align-items:center;">
            <span style="font-size:1.25rem;">{emoji}</span>
            <span>きょうの体調リスク：{label}</span>
          </div>
          <div style="opacity:.92;margin-top:.25rem;">
            スコア合計 <b>{total_score}</b>（ベース {base_score} ＋ 今日 {daily_score}）
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# Calendar helpers
# ==================================================
def forecast_to_events(forecast_days: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    events: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Any]] = {}

    for d in forecast_days:
        date_obj: dt.date = d["date"]
        date_str = date_obj.isoformat()
        label = d["label"]

        if label == "低め":
            title = "🟢 低め"
            bg = "#B7F0C1"
            border = "#57C46A"
        elif label == "やや高め":
            title = "🟡 やや高め"
            bg = "#FFF2B2"
            border = "#F4C44E"
        else:
            title = "🔴 高め"
            bg = "#FFD1D9"
            border = "#FF6B6B"

        events.append(
            {
                "title": title,
                "start": date_str,
                "end": date_str,
                "allDay": True,
                "backgroundColor": bg,
                "borderColor": border,
                "textColor": "#1f1f1f",
                "extendedProps": d,
            }
        )
        index[date_str] = d

    return events, index


def build_simple_calendar_html(forecast_days: List[Dict[str, Any]]) -> str:
    if not forecast_days:
        return "<div class='wf-card'>予報データがありません。</div>"

    by_date = {d["date"]: d for d in forecast_days}
    first_date = forecast_days[0]["date"]
    year, month = first_date.year, first_date.month

    cal = pycal.Calendar(firstweekday=6)
    weeks = cal.monthdayscalendar(year, month)
    week_labels = ["日", "月", "火", "水", "木", "金", "土"]

    html_s = f"<div class='wf-card'><div style='font-weight:900;margin-bottom:.4rem'>{year}年{month}月（簡易表示）</div>"
    html_s += "<table style='width:100%;border-collapse:collapse;table-layout:fixed;border-radius:14px;overflow:hidden;'>"
    html_s += "<tr>"
    for w in week_labels:
        html_s += f"<th style='background:#E1BEE7;padding:.45rem;font-size:.92rem'>{w}</th>"
    html_s += "</tr>"

    for week in weeks:
        html_s += "<tr>"
        for day in week:
            if day == 0:
                html_s += "<td style='background:rgba(0,0,0,0.03);height:80px'></td>"
                continue
            cur = dt.date(year, month, day)
            info = by_date.get(cur)
            if not info:
                html_s += f"<td style='background:rgba(0,0,0,0.04);height:80px;padding:.25rem;vertical-align:top'><b>{day}</b><div style='opacity:.6'>—</div></td>"
            else:
                label = info["label"]
                if label == "低め":
                    bg, em = "#B7F0C1", "🟢"
                elif label == "やや高め":
                    bg, em = "#FFF2B2", "🟡"
                else:
                    bg, em = "#FFD1D9", "🔴"
                html_s += f"<td style='background:{bg};height:80px;padding:.25rem;vertical-align:top'><b>{day}</b><div style='font-weight:800'>{em} {label}</div></td>"
        html_s += "</tr>"

    html_s += "</table></div>"
    return html_s


# ==================================================
# Profile tab UI
# ==================================================
def profile_tab_ui(profile: Dict[str, Any]) -> Dict[str, Any]:
    st.markdown('<div class="wf-section">🧑‍⚕️ プロフィール</div>', unsafe_allow_html=True)
    st.markdown('<div class="wf-card">体調の「崩れやすさ」の土台に使います。任意の項目は空でOKです。</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年齢", min_value=0, max_value=120, value=int(profile["age"]) if profile["age"] is not None else 40)
        height_cm = st.number_input("身長（cm）", min_value=0.0, max_value=250.0,
                                    value=float(profile["height_cm"]) if profile["height_cm"] is not None else 170.0, step=0.5)
    with col2:
        weight_kg = st.number_input("体重（kg）", min_value=0.0, max_value=300.0,
                                    value=float(profile["weight_kg"]) if profile["weight_kg"] is not None else 60.0, step=0.5)
        blood_type = st.text_input("血液型（任意）", value=profile.get("blood_type", ""))

    sex = st.selectbox("性別（任意）", ["未設定", "男性", "女性", "その他"],
                       index=["未設定", "男性", "女性", "その他"].index(profile.get("sex", "未設定")))

    st.markdown("##### 慢性的なもの（当てはまる場合だけ）")
    ch = profile["chronic"]
    c1, c2, c3 = st.columns(3)
    with c1:
        ch["migraine"] = st.checkbox("片頭痛", value=ch.get("migraine", False))
        ch["tension_headache"] = st.checkbox("緊張型頭痛", value=ch.get("tension_headache", False))
        ch["anxiety_depression"] = st.checkbox("不安・うつなど", value=ch.get("anxiety_depression", False))
    with c2:
        ch["asthma"] = st.checkbox("喘息", value=ch.get("asthma", False))
        ch["copd"] = st.checkbox("COPD / 肺気腫", value=ch.get("copd", False))
    with c3:
        ch["hypertension"] = st.checkbox("高血圧", value=ch.get("hypertension", False))
        ch["diabetes"] = st.checkbox("糖尿病", value=ch.get("diabetes", False))
        ch["cvd"] = st.checkbox("心臓の病気", value=ch.get("cvd", False))

    st.markdown("##### アレルギー")
    al = profile["allergy"]
    al["nsaids"] = st.checkbox("痛み止め（NSAIDs）で強い副反応/アレルギーが出たことがある", value=al.get("nsaids", False))
    al["antibiotics"] = st.checkbox("抗生物質でアレルギーが出たことがある", value=al.get("antibiotics", False))
    al["food"] = st.text_input("食べ物（あれば）", value=al.get("food", ""))
    al["others"] = st.text_input("その他（あれば）", value=al.get("others", ""))

    if st.button("💾 保存する", use_container_width=True):
        profile["age"] = int(age)
        profile["sex"] = sex
        profile["height_cm"] = float(height_cm) if height_cm > 0 else None
        profile["weight_kg"] = float(weight_kg) if weight_kg > 0 else None
        profile["blood_type"] = blood_type
        profile["chronic"] = ch
        profile["allergy"] = al
        save_profile(profile)
        st.success("保存しました！次回以降もこの情報を使います。")

    bmi = calc_bmi(profile.get("height_cm"), profile.get("weight_kg"))
    if bmi is not None:
        st.info(f"BMI（目安）: {bmi:.1f}")

    base_score, base_reasons = calc_profile_base_risk(profile)
    st.markdown('<div class="wf-section">🧩 ベースの崩れやすさ</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='wf-card'>ベーススコア：<b>{base_score}</b>（0〜3）</div>", unsafe_allow_html=True)
    if base_reasons:
        for r in base_reasons:
            st.write(f"- {r}")
    else:
        st.write("今の登録内容では、目立つベース要因は少なめです。")

    return profile


# ==================================================
# Main
# ==================================================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")
    inject_css()

    if "profile" not in st.session_state:
        st.session_state.profile = load_profile()
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "today"
    if "wf_result" not in st.session_state:
        st.session_state.wf_result = None

    profile = st.session_state.profile

    st.markdown(f'<div class="wf-title">🩺 {APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="wf-sub">気圧だけじゃなく、気温・湿度・雨・風も使って「崩れやすさ」を見える化します。</div>', unsafe_allow_html=True)
    st.markdown("<div class='wf-card'>※このアプリは体調管理の目安です。強い症状があるときはスコアに関係なく医療機関の受診を検討してください。</div>", unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        if st.button("🌈 きょうのようす", use_container_width=True):
            st.session_state.active_tab = "today"
    with t2:
        if st.button("🧑‍⚕️ プロフィール", use_container_width=True):
            st.session_state.active_tab = "profile"

    if st.session_state.active_tab == "profile":
        st.session_state.profile = profile_tab_ui(profile)
        return

    # ✅ ここから入力はフォーム（再実行のチラつきを抑える）
    st.markdown('<div class="wf-section">🌤️ きょうの入力</div>', unsafe_allow_html=True)

    with st.form("wf_form", clear_on_submit=False):
        st.markdown("<div class='wf-card'>📍 場所（天気を取る地点です）</div>", unsafe_allow_html=True)
        colA, colB, colC = st.columns([1.2, 1.2, 1])
        with colA:
            latitude = st.number_input("緯度", -90.0, 90.0, DEFAULT_LAT, 0.01, key="lat")
        with colB:
            longitude = st.number_input("経度", -180.0, 180.0, DEFAULT_LON, 0.01, key="lon")
        with colC:
            use_auto_weather = st.checkbox("APIで自動取得", value=True, key="auto_weather")

        st.markdown("<div class='wf-card'>🧸 きょうの体調メモ（だいたいでOK）</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            sleep_hours = st.number_input("昨夜の睡眠時間（時間）", 0.0, 15.0, 6.0, 0.5, key="sleep")
            alcohol = st.checkbox("きのうお酒を飲んだ", value=False, key="alcohol")
            steps_raw = st.number_input("きのうの歩数（0なら不明）", 0, 50000, 6000, 500, key="steps")
            steps = steps_raw if steps_raw > 0 else None
        with col2:
            manual_pressure_drop = st.number_input("直近3時間の気圧変化[hPa]（API不調のとき）", -20.0, 20.0, 0.0, 0.1, key="pdrop")
            resting_hr_diff = st.number_input("安静時心拍（ふだんとの差）[bpm]", -30.0, 30.0, 0.0, 1.0, key="rhr")

        user_note = st.text_area("気になる症状・予定（任意）", placeholder="例）片側の頭がズキズキ／鼻づまり／今日は冷えた… など", key="note")

        st.markdown("<div class='wf-card'>🗓️ 予報の表示日数</div>", unsafe_allow_html=True)
        days_ahead = st.slider("何日先まで表示する？", min_value=3, max_value=14, value=7, step=1, key="days")

        run_clicked = st.form_submit_button("✨ きょうのリスク＋予報カレンダーを見る", use_container_width=True)

    # クリアはフォーム外（いつでも）
    if st.button("🧹 結果をクリア", use_container_width=True):
        st.session_state.wf_result = None

    if run_clicked:
        pressure_drop_3h = manual_pressure_drop
        latest_pressure = None
        bundle = None

        if use_auto_weather:
            with st.spinner("天気データを取得しています…"):
                p_drop, latest_p, msg, bundle = fetch_weather_from_open_meteo(latitude, longitude)
            api_msg = msg
            if p_drop is not None:
                pressure_drop_3h = p_drop
            latest_pressure = latest_p
        else:
            api_msg = "API自動取得はOFFです（手動の気圧変化を使用）。"

        base_score, base_reasons = calc_profile_base_risk(profile)
        daily_score, daily_reasons = calc_daily_risk(
            sleep_hours=sleep_hours,
            alcohol=alcohol,
            pressure_drop_3h=pressure_drop_3h,
            resting_hr_diff=resting_hr_diff,
            steps=steps,
        )

        weather_snapshot = {}
        if bundle and bundle.get("hourly"):
            ws, wr, snap = add_weather_risk_from_latest(bundle["hourly"])
            daily_score += ws
            daily_reasons.extend(wr)
            weather_snapshot = snap

        total_score = base_score + daily_score
        label, color, emoji = classify_total_risk(total_score)

        forecast_days = []
        if bundle and bundle.get("hourly"):
            forecast_days = build_daily_forecast_from_hourly(bundle["hourly"], days_ahead=days_ahead)

        gemini_text = None
        if client is not None:
            with st.spinner("AIのやさしいアドバイスを作成中…"):
                profile_summary = summarize_profile_for_gemini(profile)
                gemini_text = call_gemini_for_advice(
                    profile_summary=profile_summary,
                    risk_label=label,
                    total_score=total_score,
                    base_score=base_score,
                    daily_score=daily_score,
                    base_reasons=base_reasons,
                    daily_reasons=daily_reasons,
                    user_note=user_note,
                )

        st.session_state.wf_result = {
            "api_msg": api_msg,
            "latest_pressure": latest_pressure,
            "pressure_drop_3h": pressure_drop_3h,
            "base_score": base_score,
            "base_reasons": base_reasons,
            "daily_score": daily_score,
            "daily_reasons": daily_reasons,
            "total_score": total_score,
            "label": label,
            "color": color,
            "emoji": emoji,
            "weather_snapshot": weather_snapshot,
            "forecast_days": forecast_days,
            "gemini_text": gemini_text,
        }

    # ==================================================
    # 結果表示（常にsession_stateから描画）
    # ==================================================
    res = st.session_state.wf_result
    if res:
        st.info(res["api_msg"])

        st.markdown('<div class="wf-section">🧡 きょうの結果</div>', unsafe_allow_html=True)
        risk_card(res["label"], res["color"], res["emoji"], res["total_score"], res["base_score"], res["daily_score"])

        st.markdown("<div class='wf-card'>📌 いまの天気（参考）</div>", unsafe_allow_html=True)
        cW1, cW2, cW3, cW4, cW5 = st.columns(5)
        with cW1:
            lp = res["latest_pressure"]
            st.metric("気圧(hPa)", f"{lp:.1f}" if lp is not None else "—")
        with cW2:
            t = res["weather_snapshot"].get("temperature_2m")
            st.metric("気温(℃)", f"{t:.1f}" if t is not None else "—")
        with cW3:
            a = res["weather_snapshot"].get("apparent_temperature")
            st.metric("体感(℃)", f"{a:.1f}" if a is not None else "—")
        with cW4:
            h = res["weather_snapshot"].get("relative_humidity_2m")
            st.metric("湿度(%)", f"{h:.0f}" if h is not None else "—")
        with cW5:
            w = res["weather_snapshot"].get("wind_speed_10m")
            st.metric("風速(m/s)", f"{w:.1f}" if w is not None else "—")

        st.write(f"直近3時間の気圧変化（判定に使用）: {res['pressure_drop_3h']:+.1f} hPa")

        st.markdown('<div class="wf-section">🧩 理由（ざっくり）</div>', unsafe_allow_html=True)
        st.markdown("<div class='wf-card'>", unsafe_allow_html=True)
        st.write("ベース（プロフィール）:")
        if res["base_reasons"]:
            for r in res["base_reasons"]:
                st.write(f"- {r}")
        else:
            st.write("- 目立つベース要因は少なめです。")

        st.write("\nきょう（睡眠・気圧・天気など）:")
        if res["daily_reasons"]:
            for r in res["daily_reasons"]:
                st.write(f"- {r}")
        else:
            st.write("- 目立つ追加要因は少なめです。")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="wf-section">🤖 AIのやさしいアドバイス</div>', unsafe_allow_html=True)
        if client is None:
            st.markdown("<div class='wf-card'>Geminiは未設定です（環境変数 GEMINI_API_KEY を設定すると有効になります）。</div>", unsafe_allow_html=True)
        else:
            # ✅ ここが本命修正：HTMLエスケープして崩れないようにする
            st.markdown(f"<div class='wf-card'>{safe_html_text(res.get('gemini_text'))}</div>", unsafe_allow_html=True)

        st.markdown('<div class="wf-section">🗓️ 予報カレンダー（気圧＋気温＋湿度＋雨＋風）</div>', unsafe_allow_html=True)

        forecast_days = res["forecast_days"]
        if not forecast_days:
            st.markdown("<div class='wf-card'>予報を計算できませんでした（API自動取得OFFの場合もここになります）。</div>", unsafe_allow_html=True)
        else:
            events, index = forecast_to_events(forecast_days)

            if CALENDAR_AVAILABLE:
                st.markdown("<div class='wf-card'>📌 日付（色つき）をクリックすると、その日の根拠が下に出ます。</div>", unsafe_allow_html=True)
                options = {
                    "initialView": "dayGridMonth",
                    "locale": "ja",
                    "height": 780,
                    "headerToolbar": {"left": "prev,next today", "center": "title", "right": "dayGridMonth,listWeek"},
                    "dayMaxEventRows": True,
                }
                cal_state = st_calendar(events=events, options=options, key="wf_calendar")
                selected = None
                if isinstance(cal_state, dict):
                    selected = cal_state.get("eventClick") or cal_state.get("event")

                if selected and isinstance(selected, dict):
                    start = selected.get("start", "")
                    date_str = start[:10] if start else ""
                    info = index.get(date_str)
                    if info:
                        st.markdown("<div class='wf-card'>", unsafe_allow_html=True)
                        st.write(f"📅 {date_str} の予報：**{info['label']}**（スコア: {info['score']}）")
                        if info.get("reasons"):
                            st.write("理由：")
                            for r in info["reasons"]:
                                st.write(f"- {r}")
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='wf-card'>本物のカレンダーを使うには <code>pip install streamlit-calendar</code> を実行してください。いまは簡易表示です。</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(build_simple_calendar_html(forecast_days), unsafe_allow_html=True)

        st.markdown(
            "<div class='wf-card'>"
            "🆘 強い頭痛、胸の痛み、息苦しさ、ろれつが回らない、片側の手足が動きにくい、意識がもうろう… "
            "などがある場合は、スコアに関係なく早めに医療機関の受診を検討してください。"
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
