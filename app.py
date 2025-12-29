import os
import json
import datetime as dt
from datetime import datetime
from typing import Dict, Any, List, Optional

import requests
import streamlit as st

from google import genai
from google.genai import types

# --------------------------------------------------
# 設定
# --------------------------------------------------

PROFILE_PATH = "profile.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None

# デフォルト座標（愛媛・瀬戸内あたりのイメージ）
DEFAULT_LAT = 34.25
DEFAULT_LON = 133.20


# --------------------------------------------------
# プロファイル保存まわり
# --------------------------------------------------

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
        # 権限などで保存できない場合はスルー
        pass


# --------------------------------------------------
# ベースリスク計算
# --------------------------------------------------

def calc_bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    if not height_cm or not weight_kg or height_cm <= 0:
        return None
    h_m = height_cm / 100.0
    return weight_kg / (h_m * h_m)


def calc_profile_base_risk(profile: Dict[str, Any]) -> (int, List[str]):
    """
    プロファイルからベースリスク（体調の崩れやすさの土台）を計算
    ※ 病気の診断ではありません
    """
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

    # ベーススコアの上限
    if score > 3:
        score = 3
    return score, reasons


def summarize_profile_for_gemini(profile: Dict[str, Any]) -> str:
    """
    Gemini に渡す用に、プロフィールを少しぼかして要約
    """
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


# --------------------------------------------------
# Open-Meteo から気圧取得
# --------------------------------------------------

def fetch_pressure_from_open_meteo(latitude: float, longitude: float):
    """
    Open-Meteo から気圧（hourly）を取得
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pressure_msl",
        "timezone": "auto",
    }

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


# --------------------------------------------------
# 日別の気圧リスク解析
# --------------------------------------------------

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


def make_pressure_forecast(times, pressures, days_ahead: int = 5):
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
        if len(day_pressures) < 4:
            min_p = min(day_pressures)
            max_drop_3h = 0.0
        else:
            min_p = min(day_pressures)
            max_drop_3h = 0.0
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


# --------------------------------------------------
# 今日のリスク計算（状態＋気圧）
# --------------------------------------------------

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
            reasons.append("前日の歩数が少なく、血行不良やだるさが出やすい状態です。")
        elif steps > 15000:
            score += 1
            reasons.append("前日の活動量がかなり多く、疲れが残っているかもしれません。")

    return score, reasons


def classify_total_risk(total_score: int) -> (str, str, str):
    if total_score <= 2:
        return "おちついている", "#2e7d32", "🟢"
    elif total_score <= 5:
        return "少し注意したい", "#f9a825", "🟡"
    else:
        return "今日はかなり慎重に", "#c62828", "🔴"


# --------------------------------------------------
# Gemini アドバイス
# --------------------------------------------------

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

【ベースリスクの理由（もともとの体質・持病など）】
  {base_text}

【今日の条件によるリスク要因】
  {daily_text}

【今日の入力データ】
- 睡眠時間: {sleep_hours} 時間
- 前日のアルコール: {"あり" if alcohol else "なし"}
- 直近3時間の気圧変化: {pressure_drop if pressure_drop is not None else "不明"} hPa
- 安静時心拍の平常値との差: {resting_hr_diff} bpm
- 前日の歩数（おおよそ）: {steps if steps is not None else "不明"}
- 本人メモ・症状・予定:
  {user_note if user_note else "特になし"}

【出力条件】
- 日本語・ですます調で、やさしい言葉で書いてください。
- 800字以内。
- 構成：
  1. 今日のからだの状態のイメージ（3〜5行）
  2. 今日のおすすめの過ごし方（箇条書き3〜5個）
  3. こんなサインが出たら受診を考えてほしい（2〜4個）
- 市販薬や具体的な薬の名前は出さないでください。
- 命にかかわる可能性がある症状が疑われる場合は
  「早めに医療機関を受診することを検討してください」と必ず書いてください。
""".strip()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
            ),
        )
        return response.text
    except Exception as e:
        return f"Geminiからのアドバイス取得に失敗しました。\nエラー概要: {e}"


# --------------------------------------------------
# UI・スタイル
# --------------------------------------------------

def inject_mobile_css():
    css = """
    <style>
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 900px;
        margin: auto;
    }
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.6rem;
            padding-right: 0.6rem;
        }
    }
    .wf-header-title {
        font-size: 1.4rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .wf-header-sub {
        font-size: 0.82rem;
        opacity: 0.85;
    }
    .wf-pill-tabs {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .wf-pill {
        flex: 1;
        text-align: center;
        padding: 0.5rem 0.6rem;
        border-radius: 999px;
        font-size: 0.85rem;
        border: 1px solid #e0e0e0;
        background: #ffffffee;
    }
    .wf-pill-active {
        background: linear-gradient(120deg, #4caf50, #81c784);
        color: #fff;
        border-color: transparent;
        box-shadow: 0 4px 10px rgba(76,175,80,0.3);
    }
    .wf-risk-card {
        border-radius: 18px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .wf-risk-main {
        font-size: 1.05rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .wf-risk-sub {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .wf-section-title {
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 1.0rem;
        margin-bottom: 0.3rem;
    }
    .wf-forecast-item {
        padding: 0.6rem 0.2rem;
        border-bottom: 1px solid rgba(0,0,0,0.06);
        font-size: 0.86rem;
    }
    .wf-forecast-item:last-child {
        border-bottom: none;
    }
    .wf-forecast-date {
        font-weight: 600;
        margin-right: 0.3rem;
    }
    .wf-forecast-reasons {
        font-size: 0.78rem;
        opacity: 0.9;
        margin-left: 1.4rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def risk_card(label: str, color: str, emoji: str, total_score: int, base_score: int, daily_score: int):
    bg = f"{color}20"
    html = f"""
    <div class="wf-risk-card" style="background-color:{bg}; border: 1px solid {color}33;">
      <div class="wf-risk-main">
        <span>{emoji}</span>
        <span>きょうの体調リスク：{label}</span>
      </div>
      <div class="wf-risk-sub">
        スコア合計 <b>{total_score}</b>（ベース {base_score} ＋ 今日の条件 {daily_score}）
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------
# プロファイルタブ UI
# --------------------------------------------------

def profile_tab_ui(profile: Dict[str, Any]) -> Dict[str, Any]:
    st.markdown("#### プロフィール（からだの基本情報）")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年齢", min_value=0, max_value=120,
                              value=int(profile["age"]) if profile["age"] is not None else 40)
        height_cm = st.number_input(
            "身長（cm）", min_value=0.0, max_value=250.0,
            value=float(profile["height_cm"]) if profile["height_cm"] is not None else 170.0,
            step=0.5,
        )
    with col2:
        weight_kg = st.number_input(
            "体重（kg）", min_value=0.0, max_value=300.0,
            value=float(profile["weight_kg"]) if profile["weight_kg"] is not None else 60.0,
            step=0.5,
        )
        blood_type = st.text_input("血液型（任意）", value=profile.get("blood_type", ""))

    sex = st.selectbox(
        "性別（任意）",
        ["未設定", "男性", "女性", "その他"],
        index=["未設定", "男性", "女性", "その他"].index(profile.get("sex", "未設定")),
    )

    st.markdown("##### 慢性的な病気（あてはまるものにチェック）")
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
    al["nsaids"] = st.checkbox("ロキソニンなどの痛み止めでアレルギーが出たことがある", value=al.get("nsaids", False))
    al["antibiotics"] = st.checkbox("抗生物質でアレルギーが出たことがある", value=al.get("antibiotics", False))
    al["food"] = st.text_input("食べ物のアレルギー（あれば）", value=al.get("food", ""))
    al["others"] = st.text_input("その他のアレルギー（あれば）", value=al.get("others", ""))

    save_col, _ = st.columns([1, 1])
    with save_col:
        if st.button("プロフィールを保存する", use_container_width=True):
            profile["age"] = int(age)
            profile["sex"] = sex
            profile["height_cm"] = float(height_cm) if height_cm > 0 else None
            profile["weight_kg"] = float(weight_kg) if weight_kg > 0 else None
            profile["blood_type"] = blood_type
            profile["chronic"] = ch
            profile["allergy"] = al
            save_profile(profile)
            st.success("プロフィールを保存しました。次回以降もこの情報が使われます。")

    bmi = calc_bmi(profile.get("height_cm"), profile.get("weight_kg"))
    if bmi is not None:
        st.info(f"BMI（目安）: {bmi:.1f}")

    base_score, base_reasons = calc_profile_base_risk(profile)
    st.markdown('<div class="wf-section-title">プロフィールから見た「崩れやすさ」の傾向</div>', unsafe_allow_html=True)
    st.write(f"ベースリスクスコア: {base_score}（0〜3）")
    if base_reasons:
        for r in base_reasons:
            st.write(f"- {r}")
    else:
        st.write("今の登録内容からは、特に大きなベースリスクは見つかりません。")

    return profile


# --------------------------------------------------
# メインアプリ
# --------------------------------------------------

def main():
    st.set_page_config(page_title="Wellness Forecast", page_icon="🩺", layout="wide")
    inject_mobile_css()

    if "profile" not in st.session_state:
        st.session_state.profile = load_profile()
    profile = st.session_state.profile

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "today"

    # ヘッダー
    header_col1, header_col2 = st.columns([3, 2])
    with header_col1:
        st.markdown(
            '<div class="wf-header-title">🩺 Wellness Forecast</div>'
            '<div class="wf-header-sub">気圧と生活リズムから、きょうの「崩れやすさ」をそっとお知らせするアプリです。</div>',
            unsafe_allow_html=True,
        )
    with header_col2:
        st.write("")

    st.caption(
        "※このアプリは、体調管理の目安として使うためのものです。"
        " 病気の診断や治療に代わるものではありません。"
    )

    # タブ切り替え
    pill_col1, pill_col2 = st.columns(2)
    with pill_col1:
        if st.button("きょうのようす", key="tab_today_btn", use_container_width=True):
            st.session_state.active_tab = "today"
    with pill_col2:
        if st.button("プロフィール", key="tab_profile_btn", use_container_width=True):
            st.session_state.active_tab = "profile"

    if st.session_state.active_tab == "today":
        pill_html = """
        <div class="wf-pill-tabs">
          <div class="wf-pill wf-pill-active">きょうのようす</div>
          <div class="wf-pill">プロフィール</div>
        </div>
        """
    else:
        pill_html = """
        <div class="wf-pill-tabs">
          <div class="wf-pill">きょうのようす</div>
          <div class="wf-pill wf-pill-active">プロフィール</div>
        </div>
        """
    st.markdown(pill_html, unsafe_allow_html=True)

    # プロファイルタブ
    if st.session_state.active_tab == "profile":
        profile = profile_tab_ui(profile)
        st.session_state.profile = profile
        return

    # きょうのようすタブ
    today = dt.date.today()
    st.write(f"本日の日付：{today}")

    # 1. 気圧
    st.markdown('<div class="wf-section-title">1. 気圧の情報（Open-Meteo）</div>', unsafe_allow_html=True)
    col_loc1, col_loc2, col_loc3 = st.columns([1.3, 1.3, 1])
    with col_loc1:
        latitude = st.number_input("緯度（latitude）", -90.0, 90.0, DEFAULT_LAT, 0.01)
    with col_loc2:
        longitude = st.number_input("経度（longitude）", -180.0, 180.0, DEFAULT_LON, 0.01)
    with col_loc3:
        use_auto_pressure = st.checkbox("APIから自動取得する", value=True)

    # 2. 今日の状態
    st.markdown('<div class="wf-section-title">2. きょうのからだの状態</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        sleep_hours = st.number_input("昨夜の睡眠時間（時間）", 0.0, 15.0, 6.0, 0.5)
        alcohol = st.checkbox("きのうお酒を飲んだ", value=False)
        steps = st.number_input("きのうの歩数（だいたいでOK・0なら不明扱い）", 0, 50000, 6000, 500)
        steps = steps if steps > 0 else None

    with col2:
        manual_pressure_drop = st.number_input(
            "直近3時間の気圧変化 [hPa]（マイナスで低下。APIが使えないときの予備）",
            -20.0, 20.0, 0.0, 0.1
        )
        resting_hr_diff = st.number_input(
            "安静時心拍（ふだんとの差）[bpm]",
            -30.0, 30.0, 0.0, 1.0
        )

    user_note = st.text_area(
        "きょう気になっている症状・予定など（任意）",
        placeholder="例）左側の頭がズキズキする／午後から外出／鼻づまりがつらい、など"
    )

    st.markdown("---")

    if st.button("きょうのリスクと数日予報を見る", use_container_width=True):
        # 気圧
        pressure_drop = manual_pressure_drop
        latest_pressure = None
        times = None
        pressures = None

        if use_auto_pressure:
            with st.spinner("気圧データを取得しています…"):
                p_drop, latest, msg, times, pressures = fetch_pressure_from_open_meteo(
                    latitude, longitude
                )
            st.info(msg)
            if p_drop is not None:
                pressure_drop = p_drop
            if latest is not None:
                latest_pressure = latest

        # ベース＋今日のスコア
        base_score, base_reasons = calc_profile_base_risk(profile)
        daily_score, daily_reasons = calc_daily_risk(
            sleep_hours,
            alcohol,
            pressure_drop,
            resting_hr_diff,
            steps,
        )
        total_score = base_score + daily_score
        label, color, emoji = classify_total_risk(total_score)

        # 3. 今日の総合リスク
        st.markdown('<div class="wf-section-title">3. きょうの総合リスク</div>', unsafe_allow_html=True)
        risk_card(label, color, emoji, total_score, base_score, daily_score)

        if latest_pressure is not None:
            st.write(f"現在の気圧（参考値）: {latest_pressure:.1f} hPa")
        st.write(f"直近3時間の気圧変化（判定に使用）: {pressure_drop:+.1f} hPa")

        st.markdown('<div class="wf-section-title">ベースリスク（もともとの体質・持病など）</div>', unsafe_allow_html=True)
        if base_reasons:
            for r in base_reasons:
                st.write(f"- {r}")
        else:
            st.write("登録されている情報からは、大きなベースリスクは見つかりません。")

        st.markdown('<div class="wf-section-title">きょうの追加リスク（睡眠・気圧など）</div>', unsafe_allow_html=True)
        if daily_reasons:
            for r in daily_reasons:
                st.write(f"- {r}")
        else:
            st.write("きょうの条件からは、大きな追加リスクは検出されませんでした。")

        # 4. Gemini アドバイス
        st.markdown("---")
        st.markdown('<div class="wf-section-title">4. AIからのやさしいアドバイス（Gemini）</div>', unsafe_allow_html=True)
        profile_summary = summarize_profile_for_gemini(profile)

        if client is None:
            st.info(
                "Gemini の API キーが設定されていないため、AIからのアドバイスは表示できません。"
                " 環境変数 GEMINI_API_KEY を設定すると、ここにコメントが表示されます。"
            )
        else:
            with st.spinner("きょうの過ごし方について考えています…"):
                gemini_text = call_gemini_for_advice(
                    profile_summary,
                    label,
                    total_score,
                    base_score,
                    daily_score,
                    base_reasons,
                    daily_reasons,
                    sleep_hours,
                    alcohol,
                    pressure_drop,
                    resting_hr_diff,
                    steps,
                    user_note,
                )
            st.write(gemini_text)

        # 5. 数日予報
        st.markdown("---")
        st.markdown('<div class="wf-section-title">5. 気圧から見た数日先のリスク予報</div>', unsafe_allow_html=True)

        if times is None or pressures is None:
            st.info("気圧データがないため、数日予報は表示できません。")
        else:
            forecast_days = make_pressure_forecast(times, pressures, days_ahead=7)
            if not forecast_days:
                st.write("数日分の気圧予報を計算できませんでした。")
            else:
                for day_info in forecast_days:
                    d = day_info["date"]
                    d_label = day_info["label"]
                    max_drop = day_info["max_drop_3h"]
                    min_p = day_info["min_pressure"]
                    reasons_f = day_info["reasons"]

                    if d_label == "低め":
                        icon = "🟢"
                    elif d_label == "やや高め":
                        icon = "🟡"
                    else:
                        icon = "🔴"

                    st.markdown(
                        f'<div class="wf-forecast-item">'
                        f'<span class="wf-forecast-date">{d}</span>'
                        f'{icon} {d_label} '
                        f'(3時間あたり最大変化: {max_drop:+.1f} hPa, 最低気圧: {min_p:.1f} hPa)'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if reasons_f:
                        for r in reasons_f:
                            st.markdown(
                                f'<div class="wf-forecast-reasons">- {r}</div>',
                                unsafe_allow_html=True,
                            )

        st.caption(
            "このアプリは、体調のセルフチェックをお手伝いするためのものです。"
            " 強い頭痛や胸の痛み、息苦しさ、ろれつが回らない、片側の手足が動きにくい、"
            " 意識がもうろうとする、といった症状があるときは、スコアに関係なく"
            " 早めに医療機関を受診してください。"
        )


if __name__ == "__main__":
    main()
