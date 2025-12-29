import os
import json
import datetime as dt
from datetime import datetime
from typing import Dict, Any, List, Optional
import calendar as pycal

import requests
import streamlit as st

# Gemini（任意）
from google import genai
from google.genai import types

# 本物のカレンダーUI（任意：入っていれば使う）
CALENDAR_AVAILABLE = False
try:
    from streamlit_calendar import calendar as st_calendar
    CALENDAR_AVAILABLE = True
except Exception:
    CALENDAR_AVAILABLE = False


# ==================================================
# 設定
# ==================================================
PROFILE_PATH = "profile.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None

DEFAULT_LAT = 34.25
DEFAULT_LON = 133.20


# ==================================================
# プロフィール
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


def calc_profile_base_risk(profile: Dict[str, Any]) -> (int, List[str]):
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
            reasons.append("やせ気味（BMI<18.5）で、疲れや冷えが出やすい体質です。")
        elif 25 <= bmi < 30:
            score += 1
            reasons.append("少しぽっちゃり（BMI≥25）で、関節や心臓への負担がやや高い状態です。")
        elif bmi >= 30:
            score += 2
            reasons.append("肥満（BMI≥30）で、心臓や関節への負担が高い状態です。")

    chronic = profile.get("chronic", {})
    if chronic.get("migraine"):
        score += 1
        reasons.append("片頭痛があり、気圧の変化や睡眠不足の影響を受けやすいです。")
    if chronic.get("asthma") or chronic.get("copd"):
        score += 1
        reasons.append("呼吸器の持病があり、寒さや風邪の影響を受けやすいです。")
    if chronic.get("hypertension") or chronic.get("cvd"):
        score += 1
        reasons.append("血圧や心臓に負担がかかりやすい背景があります。")
    if chronic.get("diabetes"):
        score += 1
        reasons.append("血糖のコントロールが必要な状態で、体調変化の影響を受けやすいです。")
    if chronic.get("anxiety_depression"):
        score += 1
        reasons.append("こころの負担があり、睡眠やストレスの影響を受けやすい状態です。")

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
        parts.append("年齢は不明")

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
    chronic_tags = []
    if chronic.get("migraine"):
        chronic_tags.append("片頭痛がある")
    if chronic.get("asthma") or chronic.get("copd"):
        chronic_tags.append("呼吸器の持病がある")
    if chronic.get("hypertension") or chronic.get("cvd"):
        chronic_tags.append("血圧・心臓に注意が必要")
    if chronic.get("diabetes"):
        chronic_tags.append("糖尿病がある")
    if chronic.get("anxiety_depression"):
        chronic_tags.append("こころの不調がある")

    if chronic_tags:
        parts.append("慢性疾患として " + "・".join(chronic_tags) + " がある")
    else:
        parts.append("大きな慢性疾患は登録されていない")

    allergy = profile.get("allergy", {})
    if allergy.get("nsaids"):
        parts.append("一部の痛み止め（NSAIDs）にアレルギーの可能性がある")

    return " / ".join(parts)


# ==================================================
# 気圧取得（Open-Meteo）
# ==================================================
def fetch_pressure_from_open_meteo(latitude: float, longitude: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": latitude, "longitude": longitude, "hourly": "pressure_msl", "timezone": "auto"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        pressures = hourly.get("pressure_msl", [])
        if not times or not pressures:
            return None, None, "気圧データが取得できませんでした。", None, None

        latest = float(pressures[-1])
        if len(pressures) < 4:
            msg = f"気圧データは取得できましたが、過去3時間分が足りません。現在の気圧: {latest:.1f} hPa"
            return None, latest, msg, times, pressures

        prev3 = float(pressures[-4])
        pressure_drop = latest - prev3
        msg = (
            "気圧データを取得しました。\n"
            f"・現在の気圧: {latest:.1f} hPa\n"
            f"・約3時間前との差: {pressure_drop:+.1f} hPa"
        )
        return pressure_drop, latest, msg, times, pressures
    except Exception as e:
        return None, None, f"気圧データの取得に失敗しました: {e}", None, None


def classify_pressure_risk(max_drop_3h: float, min_pressure: float):
    score = 0
    reasons: List[str] = []

    if max_drop_3h <= -6.0:
        score += 2
        reasons.append("3時間で6hPa以上の急な気圧低下が予想されます。")
    elif max_drop_3h <= -3.0:
        score += 1
        reasons.append("3時間で3〜6hPa程度の気圧低下が予想されます。")

    if min_pressure < 1000.0:
        score += 2
        reasons.append("一日の中で気圧が1000hPaを下回る時間帯があります。")
    elif min_pressure < 1005.0:
        score += 1
        reasons.append("一日の中で気圧が1005hPaを下回る時間帯があります。")

    if score <= 1:
        label = "低め"
    elif score <= 3:
        label = "やや高め"
    else:
        label = "高め"

    return label, score, reasons


def make_pressure_forecast(times, pressures, days_ahead: int = 14):
    if not times or not pressures:
        return []

    by_date: Dict[dt.date, List[float]] = {}
    for t_str, p in zip(times, pressures):
        try:
            dt_obj = datetime.fromisoformat(t_str)
        except Exception:
            continue
        d = dt_obj.date()
        by_date.setdefault(d, []).append(float(p))

    today = dt.date.today()
    target_dates = sorted(d for d in by_date.keys() if d >= today)[:days_ahead]

    results = []
    for d in target_dates:
        day_pressures = by_date[d]
        min_p = min(day_pressures)

        max_drop_3h = 0.0
        if len(day_pressures) >= 4:
            for i in range(3, len(day_pressures)):
                drop = day_pressures[i] - day_pressures[i - 3]
                if drop < max_drop_3h:
                    max_drop_3h = drop

        label, score, reasons = classify_pressure_risk(max_drop_3h, min_p)
        results.append(
            {
                "date": d,
                "label": label,
                "score": score,
                "max_drop_3h": max_drop_3h,
                "min_pressure": min_p,
                "reasons": reasons,
            }
        )
    return results


# ==================================================
# 今日のリスク
# ==================================================
def calc_daily_risk(
    sleep_hours: float,
    alcohol: bool,
    pressure_drop: Optional[float],
    resting_hr_diff: float,
    steps: Optional[int],
) -> (int, List[str]):
    score = 0
    reasons: List[str] = []

    if pressure_drop is not None:
        if pressure_drop <= -4:
            score += 2
            reasons.append("直近3時間で4hPa以上の急な気圧低下があります。")
        elif pressure_drop <= -2:
            score += 1
            reasons.append("直近3時間で2〜4hPa程度の気圧低下があります。")

    if sleep_hours < 5.5:
        score += 2
        reasons.append("睡眠時間が5.5時間未満で、かなり睡眠不足ぎみです。")
    elif sleep_hours < 6.5:
        score += 1
        reasons.append("睡眠時間が6.5時間未満で、少し寝不足ぎみです。")

    if alcohol:
        score += 1
        reasons.append("前日にお酒を飲んでいて、身体に負担が残っているかもしれません。")

    if resting_hr_diff >= 8:
        score += 2
        reasons.append("安静時心拍がいつもより8bpm以上高く、疲れや体調の負荷が強そうです。")
    elif resting_hr_diff >= 4:
        score += 1
        reasons.append("安静時心拍が少し高めで、疲れやストレスが溜まっているかもしれません。")

    if steps is not None:
        if steps < 2000:
            score += 1
            reasons.append("前日の歩数が少なく、だるさが出やすい状態です。")
        elif steps > 15000:
            score += 1
            reasons.append("前日の活動量がかなり多く、疲れが残っているかもしれません。")

    return score, reasons


def classify_total_risk(total_score: int) -> (str, str, str):
    if total_score <= 2:
        return "おちついている", "#3CB371", "🟢"
    elif total_score <= 5:
        return "少し注意したい", "#FFD54F", "🟡"
    else:
        return "今日はかなり慎重に", "#FF6B6B", "🔴"


# ==================================================
# Gemini
# ==================================================
def call_gemini_for_advice(
    profile_summary: str,
    risk_label: str,
    total_score: int,
    base_score: int,
    daily_score: int,
    base_reasons: List[str],
    daily_reasons: List[str],
    sleep_hours: float,
    alcohol: bool,
    pressure_drop: Optional[float],
    resting_hr_diff: float,
    steps: Optional[int],
    user_note: str,
) -> Optional[str]:
    if client is None:
        return None

    base_text = "\n  ".join(f"- {r}" for r in base_reasons) if base_reasons else "特になし"
    daily_text = "\n  ".join(f"- {r}" for r in daily_reasons) if daily_reasons else "特になし"

    prompt = f"""
あなたは日本人の成人に対して、医学的常識に沿った「ふだんの養生アドバイス」を行う専門家です。
診断や治療の指示は行わず、日常生活の工夫と、必要な場合の受診の目安だけを伝えてください。

【この人の背景（プロフィール）】
{profile_summary}

【今日の総合リスク】
- レベル: {risk_label}
- トータルスコア: {total_score}（ベース {base_score} + 今日の条件 {daily_score}）

【ベースリスクの理由】
  {base_text}

【今日の条件によるリスク要因】
  {daily_text}

【今日の入力データ】
- 睡眠時間: {sleep_hours} 時間
- 前日のアルコール: {"あり" if alcohol else "なし"}
- 直近3時間の気圧変化: {pressure_drop if pressure_drop is not None else "不明"} hPa
- 安静時心拍の平常値との差: {resting_hr_diff} bpm
- 前日の歩数: {steps if steps is not None else "不明"}
- 本人メモ:
  {user_note if user_note else "特になし"}

【出力条件】
- 日本語・ですます調、やさしい言葉で。
- 800字以内。
- 構成：
  1) 今日の状態イメージ（3〜5行）
  2) 今日のおすすめ（箇条書き3〜5）
  3) 受診の目安（2〜4）
- 薬の具体名は出さない。
- 危険サインが疑われる場合は「早めに医療機関を受診することを検討してください」を入れる。
""".strip()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4),
        )
        return response.text
    except Exception as e:
        return f"Geminiからのアドバイス取得に失敗しました。\nエラー概要: {e}"


# ==================================================
# UI（カラフル＆親しみ）
# ==================================================
def inject_colorful_css():
    css = """
    <style>
    .stApp {
        background:
          radial-gradient(circle at 15% 10%, rgba(255, 214, 102, 0.35), transparent 40%),
          radial-gradient(circle at 85% 15%, rgba(186, 104, 200, 0.28), transparent 42%),
          radial-gradient(circle at 20% 90%, rgba(129, 199, 132, 0.28), transparent 45%),
          radial-gradient(circle at 90% 85%, rgba(79, 195, 247, 0.25), transparent 45%),
          #fbfbff;
    }
    html, body, [class*="css"]  { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
    .block-container { max-width: 980px; padding-top: 1rem; padding-bottom: 2rem; }
    @media (max-width: 640px) { .block-container { padding-left: 0.7rem; padding-right: 0.7rem; } }

    .wf-title {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: 0.2px;
        background: linear-gradient(90deg, #42a5f5, #ab47bc, #66bb6a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex; gap: .4rem; align-items: center;
        margin-bottom: .2rem;
    }
    .wf-sub {
        font-size: 0.9rem;
        opacity: 0.85;
        margin-bottom: .5rem;
    }

    .wf-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 18px;
        padding: 12px 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        margin-top: .6rem;
    }

    .wf-section {
        font-size: 1.02rem;
        font-weight: 750;
        margin-top: 1rem;
        margin-bottom: .4rem;
        display:flex; align-items:center; gap:.35rem;
    }

    /* ボタンを少しポップに */
    .stButton>button {
        border-radius: 14px !important;
        padding: 0.55rem 0.8rem !important;
        font-weight: 700 !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.05) !important;
    }

    /* FullCalendarを大きく */
    .fc { font-size: 1.05rem; }
    .fc .fc-toolbar-title { font-size: 1.25rem; font-weight: 800; }
    .fc .fc-daygrid-day-number { font-weight: 800; }
    .fc .fc-daygrid-day-frame { min-height: 92px; }  /* ここが「大きめ」の肝 */
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
          <div style="font-size:1.05rem;font-weight:800;display:flex;gap:.4rem;align-items:center;">
            <span style="font-size:1.2rem;">{emoji}</span>
            <span>きょうの体調リスク：{label}</span>
          </div>
          <div style="opacity:.9;margin-top:.2rem;">
            スコア合計 <b>{total_score}</b>（ベース {base_score} ＋ 今日の条件 {daily_score}）
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# 予報 → “本物カレンダー”用イベント生成
# ==================================================
def forecast_to_events(forecast_days: List[Dict[str, Any]]) -> (List[Dict[str, Any]], Dict[str, Dict[str, Any]]):
    events = []
    index = {}
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
                "extendedProps": {
                    "label": label,
                    "min_pressure": float(d["min_pressure"]),
                    "max_drop_3h": float(d["max_drop_3h"]),
                    "reasons": d["reasons"],
                },
            }
        )
        index[date_str] = d
    return events, index


# ==================================================
# フォールバック（簡易カレンダーHTML）
# ==================================================
def build_simple_calendar_html(forecast_days: List[Dict[str, Any]]) -> str:
    if not forecast_days:
        return "<div class='wf-card'>予報データがありません。</div>"

    by_date = {d["date"]: d for d in forecast_days}
    first_date = forecast_days[0]["date"]
    year, month = first_date.year, first_date.month

    cal = pycal.Calendar(firstweekday=6)  # 日曜はじまり
    weeks = cal.monthdayscalendar(year, month)
    week_labels = ["日", "月", "火", "水", "木", "金", "土"]

    html = f"<div class='wf-card'><div style='font-weight:800;margin-bottom:.4rem'>{year}年{month}月（簡易表示）</div>"
    html += "<table style='width:100%;border-collapse:collapse;table-layout:fixed;border-radius:14px;overflow:hidden;'>"
    html += "<tr>"
    for w in week_labels:
        html += f"<th style='background:#E1BEE7;padding:.4rem;font-size:.9rem'>{w}</th>"
    html += "</tr>"

    for week in weeks:
        html += "<tr>"
        for day in week:
            if day == 0:
                html += "<td style='background:rgba(0,0,0,0.03);height:76px'></td>"
                continue
            cur = dt.date(year, month, day)
            info = by_date.get(cur)
            if not info:
                html += f"<td style='background:rgba(0,0,0,0.04);height:76px;padding:.2rem;vertical-align:top'><b>{day}</b><div style='opacity:.6'>—</div></td>"
            else:
                label = info["label"]
                if label == "低め":
                    bg, em = "#B7F0C1", "🟢"
                elif label == "やや高め":
                    bg, em = "#FFF2B2", "🟡"
                else:
                    bg, em = "#FFD1D9", "🔴"
                html += f"<td style='background:{bg};height:76px;padding:.2rem;vertical-align:top'><b>{day}</b><div style='font-weight:700'>{em} {label}</div></td>"
        html += "</tr>"

    html += "</table></div>"
    return html


# ==================================================
# プロフィールUI
# ==================================================
def profile_tab_ui(profile: Dict[str, Any]) -> Dict[str, Any]:
    st.markdown('<div class="wf-section">🧑‍⚕️ プロフィール</div>', unsafe_allow_html=True)
    st.markdown('<div class="wf-card">体調の「崩れやすさ」の土台を作るための情報です。任意のものは空でもOKです。</div>', unsafe_allow_html=True)

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
    al["food"] = st.text_input("食べ物のアレルギー（あれば）", value=al.get("food", ""))
    al["others"] = st.text_input("その他（あれば）", value=al.get("others", ""))

    if st.button("💾 プロフィールを保存する", use_container_width=True):
        profile["age"] = int(age)
        profile["sex"] = sex
        profile["height_cm"] = float(height_cm) if height_cm > 0 else None
        profile["weight_kg"] = float(weight_kg) if weight_kg > 0 else None
        profile["blood_type"] = blood_type
        profile["chronic"] = ch
        profile["allergy"] = al
        save_profile(profile)
        st.success("保存しました！次回以降もこの情報を使って予測します。")

    bmi = calc_bmi(profile.get("height_cm"), profile.get("weight_kg"))
    if bmi is not None:
        st.info(f"BMI（目安）: {bmi:.1f}")

    base_score, base_reasons = calc_profile_base_risk(profile)
    st.markdown('<div class="wf-section">🧩 ベースの崩れやすさ</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='wf-card'>ベーススコア：<b>{base_score}</b>（0〜3）</div>", unsafe_allow_html=True)
    if base_reasons:
        st.write("理由：")
        for r in base_reasons:
            st.write(f"- {r}")
    else:
        st.write("今の登録内容では、強いベース要因は見つかりません。")

    return profile


# ==================================================
# メイン
# ==================================================
def main():
    st.set_page_config(page_title="Wellness Forecast", page_icon="🩺", layout="wide")
    inject_colorful_css()

    if "profile" not in st.session_state:
        st.session_state.profile = load_profile()
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "today"
    profile = st.session_state.profile

    # ヘッダー
    st.markdown('<div class="wf-title">🩺 Wellness Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="wf-sub">気圧×生活リズムで、体調の「崩れやすさ」をカレンダーで見える化します。</div>', unsafe_allow_html=True)
    st.markdown("<div class='wf-card'>※体調管理の目安です。強い症状があるときはスコアに関係なく医療機関の受診を検討してください。</div>", unsafe_allow_html=True)

    # タブ
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🌈 きょうのようす", use_container_width=True):
            st.session_state.active_tab = "today"
    with c2:
        if st.button("🧑‍⚕️ プロフィール", use_container_width=True):
            st.session_state.active_tab = "profile"

    if st.session_state.active_tab == "profile":
        st.session_state.profile = profile_tab_ui(profile)
        return

    # --- 今日 ---
    st.markdown('<div class="wf-section">🌤️ きょうの入力</div>', unsafe_allow_html=True)

    # 気圧
    with st.container():
        st.markdown("<div class='wf-card'>📍 場所（気圧を取る場所です）</div>", unsafe_allow_html=True)
        colA, colB, colC = st.columns([1.2, 1.2, 1])
        with colA:
            latitude = st.number_input("緯度", -90.0, 90.0, DEFAULT_LAT, 0.01)
        with colB:
            longitude = st.number_input("経度", -180.0, 180.0, DEFAULT_LON, 0.01)
        with colC:
            use_auto_pressure = st.checkbox("APIで自動取得", value=True)

    # 今日の状態
    st.markdown("<div class='wf-card'>🧸 きょうの体調メモ（だいたいでOK）</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        sleep_hours = st.number_input("昨夜の睡眠時間（時間）", 0.0, 15.0, 6.0, 0.5)
        alcohol = st.checkbox("きのうお酒を飲んだ", value=False)
        steps_raw = st.number_input("きのうの歩数（0なら不明）", 0, 50000, 6000, 500)
        steps = steps_raw if steps_raw > 0 else None
    with col2:
        manual_pressure_drop = st.number_input("直近3時間の気圧変化[hPa]（API不調のとき）", -20.0, 20.0, 0.0, 0.1)
        resting_hr_diff = st.number_input("安静時心拍（ふだんとの差）[bpm]", -30.0, 30.0, 0.0, 1.0)

    user_note = st.text_area("気になる症状・予定（任意）", placeholder="例）片側の頭がズキズキ／鼻づまり／今日は冷えた… など")

    st.markdown("---")

    # 実行
    if st.button("✨ きょうのリスク＋カレンダー予報を見る", use_container_width=True):
        pressure_drop = manual_pressure_drop
        latest_pressure = None
        times = None
        pressures = None

        if use_auto_pressure:
            with st.spinner("気圧データを取得しています…"):
                p_drop, latest, msg, times, pressures = fetch_pressure_from_open_meteo(latitude, longitude)
            st.info(msg)
            if p_drop is not None:
                pressure_drop = p_drop
            if latest is not None:
                latest_pressure = latest

        # 今日のリスク
        base_score, base_reasons = calc_profile_base_risk(profile)
        daily_score, daily_reasons = calc_daily_risk(sleep_hours, alcohol, pressure_drop, resting_hr_diff, steps)
        total_score = base_score + daily_score
        label, color, emoji = classify_total_risk(total_score)

        st.markdown('<div class="wf-section">🧡 きょうの結果</div>', unsafe_allow_html=True)
        risk_card(label, color, emoji, total_score, base_score, daily_score)

        st.markdown("<div class='wf-card'>", unsafe_allow_html=True)
        if latest_pressure is not None:
            st.write(f"現在の気圧（参考）: {latest_pressure:.1f} hPa")
        st.write(f"直近3時間の気圧変化（判定に使用）: {pressure_drop:+.1f} hPa")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="wf-section">🧩 理由（ざっくり）</div>', unsafe_allow_html=True)
        st.markdown("<div class='wf-card'>", unsafe_allow_html=True)
        st.write("ベース（プロフィール）:")
        if base_reasons:
            for r in base_reasons:
                st.write(f"- {r}")
        else:
            st.write("- 目立つベース要因は少なめです。")

        st.write("\nきょう（睡眠・気圧など）:")
        if daily_reasons:
            for r in daily_reasons:
                st.write(f"- {r}")
        else:
            st.write("- 大きな追加要因は少なめです。")
        st.markdown("</div>", unsafe_allow_html=True)

        # Gemini
        st.markdown('<div class="wf-section">🤖 AIのやさしいアドバイス</div>', unsafe_allow_html=True)
        if client is None:
            st.markdown("<div class='wf-card'>GeminiのAPIキーが未設定のため、AIコメントはオフです（GEMINI_API_KEYを設定すると有効になります）。</div>", unsafe_allow_html=True)
        else:
            profile_summary = summarize_profile_for_gemini(profile)
            with st.spinner("アドバイスを作成中…"):
                txt = call_gemini_for_advice(
                    profile_summary, label, total_score, base_score, daily_score,
                    base_reasons, daily_reasons, sleep_hours, alcohol,
                    pressure_drop, resting_hr_diff, steps, user_note
                )
            st.markdown(f"<div class='wf-card'>{txt}</div>", unsafe_allow_html=True)

        # 予報（本物カレンダー）
        st.markdown('<div class="wf-section">🗓️ 予報カレンダー（気圧ベース）</div>', unsafe_allow_html=True)

        if times is None or pressures is None:
            st.markdown("<div class='wf-card'>気圧データがないため、カレンダー予報は表示できません。</div>", unsafe_allow_html=True)
            return

        forecast_days = make_pressure_forecast(times, pressures, days_ahead=14)  # 2週間くらい
        if not forecast_days:
            st.markdown("<div class='wf-card'>予報を計算できませんでした。</div>", unsafe_allow_html=True)
            return

        events, index = forecast_to_events(forecast_days)

        if CALENDAR_AVAILABLE:
            st.markdown("<div class='wf-card'>📌 日付をタップすると、その日の理由が下に出ます。</div>", unsafe_allow_html=True)

            options = {
                "initialView": "dayGridMonth",
                "locale": "ja",
                "height": 780,  # 大きめ
                "headerToolbar": {"left": "prev,next today", "center": "title", "right": "dayGridMonth,listWeek"},
                "dayMaxEventRows": True,
            }

            cal_state = st_calendar(events=events, options=options, key="wf_fullcalendar")

            # クリックされたイベントの詳細表示（選択イベントが返ってくる）
            selected = None
            if isinstance(cal_state, dict):
                selected = cal_state.get("eventClick") or cal_state.get("event")

            if selected and isinstance(selected, dict):
                start = selected.get("start", "")
                date_str = start[:10] if start else ""
                info = index.get(date_str)
                if info:
                    st.markdown("<div class='wf-card'>", unsafe_allow_html=True)
                    st.write(f"📅 {date_str} の予報：**{info['label']}**")
                    st.write(f"・最低気圧（目安）: {info['min_pressure']:.1f} hPa")
                    st.write(f"・3時間あたり最大変化: {info['max_drop_3h']:+.1f} hPa")
                    if info["reasons"]:
                        st.write("理由：")
                        for r in info["reasons"]:
                            st.write(f"- {r}")
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='wf-card'>"
                "本物のカレンダー表示を使うには、次を実行してください：<br>"
                "<code>pip install streamlit-calendar</code><br>"
                "いまは簡易カレンダーで表示します。</div>",
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
