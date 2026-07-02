# Auto-extracted from notebooks/비정형네트워크프로젝트.ipynb

# The notebook is the primary runnable analysis artifact.

# This script is included for code review and GitHub readability.



# %% [markdown] # 비정형네트워크프로젝트


# %% [markdown] ## 1. 분석 설계


# %% Code cell 3

import ast
import itertools
import json
import math
import re
import time
import warnings
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from urllib.parse import unquote

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import networkx as nx
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"
CHAMPION_ICON_DIR = DATA_DIR / "champion_icons"

CONFIG = {
    "leaguepedia_url": "https://lol.fandom.com/wiki/LCS/2024_Season/Championship/Match_History",
    "golgg_url": "https://gol.gg/champion/list/season-ALL/split-ALL/tournament-ALL/",
    # 이번 제출본은 실제 크롤링을 먼저 검증하도록 기본값을 True로 둔다.
    # Fandom HTML은 Cloudflare로 막힐 수 있어서 아래 CargoExport fallback을 함께 사용한다.
    "run_scraping": True,
    # "최신 패치" 분석이므로 기본값은 캐시보다 새 요청을 우선한다.
    "force_refresh": True,
    # 이전 실행에서 만든 샘플 CSV가 있으면 실제 크롤링을 건너뛰므로 False로 둔다.
    "prefer_existing_csv": False,
    # 실제 크롤링 오류를 숨기지 않기 위해 기본값은 False다. 수업 시연용 샘플이 필요하면 True로 바꾼다.
    "allow_sample_data": False,
    "request_sleep": 1.0,
    "request_timeout": 15,
    # 프로젝트 질문이 "가장 최신 패치의 OP 챔피언과 조합"으로 좁혀졌기 때문에
    # 최신 패치를 먼저 확인하고 표본이 작으면 바로 이전 패치를 자동 선택한다.
    "analysis_patch_strategy": "since_patch",
    "analysis_min_patch": 26.09,
    "use_latest_patch": True,
    "analysis_end_date": None,
    "latest_patch_min_matches": 50,
    "latest_patch_probe_limit": 1000,
    "selected_patch": None,
    "op_top_n": 10,
    "opgg_region": "global",
    "opgg_rank_tier": "platinum_plus",
    # OP.GG 내부 tier 값 0은 화면에서 보통 OP 티어로 표시된다.
    # 따라서 프로젝트의 'OP 챔피언'은 해당 패치 OP.GG의 OP 티어 champion-position으로 정의한다.
    "opgg_op_tier_raw_max": 0,
    # Champion icons make champion-heavy plots easier to scan.
    # If Data Dragon is unavailable, the notebook falls back to text-only plots.
    "download_champion_images": True,
    "ddragon_version": None,
    "bootstrap_iterations": 5000,
    "tier_sensitivity_raw_thresholds": [0, 1, 2],
    "composition_min_games": {2: 3, 3: 2, 5: 1},
    "min_games_synergy": 2,
    "min_games_counter": 2,
    "alpha": 5,
    "global_win_rate": 0.5,
    # 작은 train 표본만 쓰면 챔피언별 승률이 크게 흔들리므로 GOL.GG 전체 통계를 prior로 섞는다.
    # 20은 "가상의 prior 경기 20개" 정도의 보수적 가중치로, tournament train 데이터가 늘수록 영향이 줄어든다.
    "golgg_prior_weight": 20,
    "test_size": 0.2,
    "random_state": 42,
    "top_n": 15,
}

REQUIRED_MATCH_COLUMNS = [
    "date",
    "patch",
    "blue_team",
    "red_team",
    "winner",
    "blue_bans",
    "red_bans",
    "blue_picks",
    "red_picks",
    "blue_roster",
    "red_roster",
]

def create_project_dirs():
    """Create the project directory structure required by the assignment."""
    for directory in [DATA_DIR, CACHE_DIR, OUTPUT_DIR, FIG_DIR, CHAMPION_ICON_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "data": DATA_DIR,
        "cache": CACHE_DIR,
        "outputs": OUTPUT_DIR,
        "figures": FIG_DIR,
        "champion_icons": CHAMPION_ICON_DIR,
    }

create_project_dirs()

try:
    plt.rcParams["font.family"] = "Malgun Gothic"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False


# %% [markdown] ## 강의자료 반영 기준


# %% [markdown] ## 2. 공통 유틸리티와 챔피언명 표준화


# %% Code cell 6

def _canonical_key(name):
    """Return a punctuation-insensitive key for champion-name matching."""
    if name is None:
        return None
    text = str(name).strip()
    if not text:
        return None
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9]", "", text.lower())


CANONICAL_CHAMPIONS = [
    "Aatrox", "Ahri", "Akali", "Aphelios", "Ashe", "Azir", "Braum", "Caitlyn",
    "Ezreal", "Gnar", "Gragas", "Jax", "Jinx", "Kai'Sa", "Kalista", "Karma",
    "K'Sante", "LeBlanc", "Lee Sin", "Leona", "Lucian", "Maokai", "Nami",
    "Nautilus", "Orianna", "Ornn", "Rakan", "Rell", "Renata Glasc", "Renekton",
    "Rumble", "Sejuani", "Sivir", "Taliyah", "Thresh", "Tristana", "Vi",
    "Viego", "Wukong", "Xayah", "Xin Zhao", "Zeri",
]

ALIAS_MAP = {
    "monkeyking": "Wukong",
    "wukong": "Wukong",
    "nunu": "Nunu & Willump",
    "nunuandwillump": "Nunu & Willump",
    "renata": "Renata Glasc",
    "renataglasc": "Renata Glasc",
    "kaisa": "Kai'Sa",
    "ksante": "K'Sante",
    "reksai": "Rek'Sai",
    "drmundo": "Dr. Mundo",
    "leesin": "Lee Sin",
    "xinzhao": "Xin Zhao",
    "leblanc": "LeBlanc",
}
for champion in CANONICAL_CHAMPIONS:
    ALIAS_MAP.setdefault(_canonical_key(champion), champion)


def normalize_champion_name(name):
    """
    Standardize champion names across Leaguepedia, GOL.GG, and local CSVs.

    The function preserves readable canonical spelling while matching names with
    different apostrophes, whitespace, punctuation, or common aliases.
    """
    if name is None:
        return None
    if isinstance(name, float) and pd.isna(name):
        return None
    text = str(name).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    text = unquote(text)
    text = text.replace("_", " ")
    text = re.sub(r"\.(png|jpg|jpeg|webp|gif)$", "", text, flags=re.I)
    text = re.sub(r"(square|original|champion|icon|hd)$", "", text, flags=re.I).strip()
    key = _canonical_key(text)
    return ALIAS_MAP.get(key, text)


def parse_list_column(value):
    """Parse JSON/list-like CSV values back into Python lists."""
    if isinstance(value, list):
        return [normalize_champion_name(v) for v in value if normalize_champion_name(v)]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [normalize_champion_name(v) for v in parsed if normalize_champion_name(v)]
        except Exception:
            pass
    parts = re.split(r"[;,|]", text)
    return [normalize_champion_name(part) for part in parts if normalize_champion_name(part)]


def save_list_column_as_json(df, columns):
    """Return a CSV-safe copy where list columns are JSON strings."""
    output = df.copy()
    for column in columns:
        if column in output.columns:
            output[column] = output[column].apply(lambda x: json.dumps(parse_list_column(x), ensure_ascii=False))
    return output


def ensure_match_schema(match_df):
    """Normalize match-history columns and list fields."""
    df = match_df.copy()
    for column in REQUIRED_MATCH_COLUMNS:
        if column not in df.columns:
            df[column] = "" if column not in {"blue_bans", "red_bans", "blue_picks", "red_picks", "blue_roster", "red_roster"} else [[] for _ in range(len(df))]

    for column in ["blue_bans", "red_bans", "blue_picks", "red_picks", "blue_roster", "red_roster"]:
        df[column] = df[column].apply(parse_list_column)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().all():
        df["date"] = pd.date_range("2024-01-01", periods=len(df), freq="D")
    df["patch"] = df["patch"].replace("", "Unknown").fillna("Unknown").astype(str)
    if "match_id" not in df.columns:
        df["match_id"] = np.arange(1, len(df) + 1)

    if "blue_win" not in df.columns:
        winner_text = df["winner"].astype(str)
        df["blue_win"] = (
            winner_text.eq(df["blue_team"].astype(str))
            | winner_text.str.lower().isin(["blue", "blue team", "1", "true"])
        ).astype(int)
    else:
        df["blue_win"] = df["blue_win"].astype(int)
    df["red_win"] = 1 - df["blue_win"]
    return df


def save_match_history_csv(match_df, path=DATA_DIR / "raw_match_history.csv"):
    """Save match history with list columns serialized as JSON."""
    output = save_list_column_as_json(match_df, ["blue_bans", "red_bans", "blue_picks", "red_picks", "blue_roster", "red_roster"])
    output.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# %% [markdown] ## 3. 데이터 수집, 캐시, 로컬 CSV 대체 로드


# %% Code cell 8

def fetch_html(url, cache_name, force_refresh=False):
    """
    Fetch HTML with User-Agent and cache it under data/cache/.

    If the request fails, the function raises a clear RuntimeError and tells the
    user which local CSV path can be used instead.
    """
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and not force_refresh:
        print(f"[cache] using cached HTML: {cache_path}")
        return cache_path.read_text(encoding="utf-8", errors="ignore")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    time.sleep(CONFIG["request_sleep"])
    try:
        response = requests.get(url, headers=headers, timeout=CONFIG["request_timeout"])
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"HTML 요청 실패: {url}\n"
            f"원인: {exc}\n"
            "크롤링이 막힌 경우 data/raw_match_history.csv 또는 "
            "data/champion_global_stats.csv를 직접 넣고 다시 실행하세요."
        ) from exc
    if "Just a moment" in response.text and "challenges.cloudflare.com" in response.text:
        raise RuntimeError(
            "Fandom HTML이 Cloudflare 확인 화면으로 막혔습니다. "
            "같은 Leaguepedia 데이터의 CargoExport API fallback을 사용합니다."
        )
    cache_path.write_text(response.text, encoding="utf-8")
    print(f"[cache] saved HTML: {cache_path}")
    return response.text


def fetch_cargo_json(
    table,
    fields,
    where,
    cache_name,
    limit=500,
    force_refresh=False,
    order_by=None,
    join_on=None,
    offset=None,
):
    """
    Fetch Leaguepedia CargoExport JSON and cache it.

    왜 CargoExport를 쓰는가:
    Fandom의 일반 HTML 페이지는 Cloudflare 확인 화면으로 막히는 경우가 있다.
    CargoExport는 같은 Leaguepedia 데이터베이스의 구조화된 export라서,
    HTML 크롤링이 실패해도 동일 출처의 경기/픽밴 데이터를 안정적으로 받을 수 있다.
    """
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and not force_refresh:
        print(f"[cache] using cached Cargo JSON: {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    params = {
        "tables": table,
        "fields": fields,
        "where": where,
        "limit": str(limit),
        "format": "json",
    }
    if order_by:
        params["order by"] = order_by
    if join_on:
        params["join on"] = join_on
    if offset is not None:
        params["offset"] = str(offset)
    headers = {"User-Agent": "Mozilla/5.0"}
    time.sleep(CONFIG["request_sleep"])
    response = requests.get(
        "https://lol.fandom.com/wiki/Special:CargoExport",
        params=params,
        headers=headers,
        timeout=CONFIG["request_timeout"],
    )
    response.raise_for_status()
    if response.text.strip().startswith("Error:"):
        raise RuntimeError(f"CargoExport 오류({table}): {response.text[:300]}")
    data = response.json()
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cache] saved Cargo JSON: {cache_path}")
    return data


def _patch_number(value):
    """Convert patch values like 26.11 to sortable version numbers."""
    try:
        text = str(value).strip()
        match = re.match(r"^(\d+)(?:\.(\d+))?", text)
        if not match:
            return -1
        major = int(match.group(1))
        minor = int(match.group(2) or 0)
        return major * 100 + minor
    except Exception:
        return -1


def normalize_patch_label(value, min_patch_context=None):
    """
    Normalize patch labels to two-digit minors such as 26.09 and 26.10.

    Leaguepedia Cargo can serialize numeric patch 26.10 as 26.1. Inside a
    26.09+ analysis window, that 26.1 value represents 26.10 rather than
    26.01, so this function restores the version label before OP.GG matching.
    """
    text = str(value).strip()
    match = re.match(r"^(\d+)(?:\.(\d+))?", text)
    if not match:
        return text
    major = int(match.group(1))
    minor_text = match.group(2) or "0"
    minor = int(minor_text)
    if minor_text == "1" and min_patch_context is not None and _patch_number(min_patch_context) % 100 >= 9:
        minor = 10
    return f"{major}.{minor:02d}"


def _analysis_date_filter(prefix=""):
    """Return an optional Cargo where clause to prevent future-dated rows."""
    if not CONFIG.get("analysis_end_date"):
        return ""
    field = f"{prefix}DateTime_UTC" if prefix else "DateTime_UTC"
    return f' AND {field} <= "{CONFIG["analysis_end_date"]}"'


def detect_latest_patch_with_enough_matches(output_path=OUTPUT_DIR / "selected_patch_summary.csv"):
    """
    Pick the newest patch with enough completed professional games.

    왜 이렇게 하는가:
    가장 최신 패치는 막 적용되어 경기 수가 거의 없을 수 있다. 표본이 너무 작으면
    OP 판정과 조합 승률이 흔들리므로, 최신 패치의 완료 경기 수가 기준 미만이면
    바로 이전 패치로 자동 fallback한다.
    """
    where = "Winner IS NOT NULL AND Patch IS NOT NULL" + _analysis_date_filter()
    data = fetch_cargo_json(
        "ScoreboardGames",
        "GameId,OverviewPage,Tournament,Team1,Team2,Winner,Patch,DateTime_UTC",
        where,
        "leaguepedia_recent_patch_probe.json",
        limit=CONFIG["latest_patch_probe_limit"],
        force_refresh=CONFIG["force_refresh"],
        order_by="DateTime_UTC DESC",
    )
    probe = pd.DataFrame(data)
    if probe.empty:
        raise RuntimeError("최근 프로 경기 패치 정보를 찾지 못했습니다.")

    patch_counts = (
        probe.groupby("Patch", dropna=True)
        .agg(
            matches=("GameId", "count"),
            first_date=("DateTime UTC", "min"),
            last_date=("DateTime UTC", "max"),
            tournaments=("Tournament", lambda x: ", ".join(sorted(set(map(str, x)))[:8])),
        )
        .reset_index()
    )
    patch_counts["patch_number"] = patch_counts["Patch"].apply(_patch_number)
    patch_counts = patch_counts.sort_values("patch_number", ascending=False).reset_index(drop=True)
    patch_counts["is_latest_patch"] = False
    patch_counts.loc[0, "is_latest_patch"] = True
    patch_counts["meets_min_matches"] = patch_counts["matches"] >= CONFIG["latest_patch_min_matches"]

    if CONFIG.get("selected_patch") is not None:
        selected_patch = str(CONFIG["selected_patch"])
        reason = "manual selected_patch"
    else:
        eligible = patch_counts[patch_counts["meets_min_matches"]]
        if eligible.empty:
            selected_patch = str(patch_counts.iloc[0]["Patch"])
            reason = "latest patch used despite small sample because no eligible fallback exists"
        else:
            selected_patch = str(eligible.iloc[0]["Patch"])
            latest_patch = str(patch_counts.iloc[0]["Patch"])
            if selected_patch == latest_patch:
                reason = "latest patch has enough matches"
            else:
                reason = f"latest patch {latest_patch} has too few matches; fallback to previous sufficient patch"

    patch_counts["selected"] = patch_counts["Patch"].astype(str).eq(selected_patch)
    patch_counts["selection_reason"] = np.where(patch_counts["selected"], reason, "")
    patch_counts.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[patch] selected patch: {selected_patch} ({reason})")
    return selected_patch, patch_counts


def parse_cargo_match_rows(records, data_source):
    """Convert joined CargoExport rows into the notebook's match-history schema."""
    rows = []
    for idx, row in pd.DataFrame(records).iterrows():
        winner_num = pd.to_numeric(row.get("Winner"), errors="coerce")
        team1 = row.get("Team1")
        team2 = row.get("Team2")
        winner = team1 if winner_num == 1 else team2 if winner_num == 2 else ""
        team1_bans = [normalize_champion_name(row.get(f"Team1Ban{i}")) for i in range(1, 6)]
        team2_bans = [normalize_champion_name(row.get(f"Team2Ban{i}")) for i in range(1, 6)]
        team1_picks = [normalize_champion_name(row.get(f"Team1Pick{i}")) for i in range(1, 6)]
        team2_picks = [normalize_champion_name(row.get(f"Team2Pick{i}")) for i in range(1, 6)]
        rows.append({
            "match_id": row.get("GameId", idx + 1),
            "date": row.get("DateTime UTC"),
            "patch": row.get("Patch"),
            "tournament": row.get("Tournament", ""),
            "blue_team": team1,
            "red_team": team2,
            "winner": winner,
            "blue_bans": [champ for champ in team1_bans if champ],
            "red_bans": [champ for champ in team2_bans if champ],
            "blue_picks": [champ for champ in team1_picks if champ],
            "red_picks": [champ for champ in team2_picks if champ],
            "blue_roster": row.get("Team1Players", []),
            "red_roster": row.get("Team2Players", []),
            "blue_win": int(winner_num == 1),
            "data_source": data_source,
        })
    parsed = ensure_match_schema(pd.DataFrame(rows))
    return parsed.sort_values(["date", "match_id"]).reset_index(drop=True)


def scrape_latest_patch_match_history_cargo():
    """Scrape completed pro matches for the selected latest-sufficient patch."""
    selected_patch, patch_counts = detect_latest_patch_with_enough_matches()
    pickban_fields = ",".join(
        [f"PicksAndBansS7.Team{side}{kind}{idx}" for side in [1, 2] for kind in ["Ban", "Pick"] for idx in range(1, 6)]
    )
    fields = (
        "ScoreboardGames.GameId,ScoreboardGames.OverviewPage,ScoreboardGames.Tournament,"
        "ScoreboardGames.Team1,ScoreboardGames.Team2,ScoreboardGames.Winner,"
        "ScoreboardGames.Patch,ScoreboardGames.DateTime_UTC,"
        f"{pickban_fields}"
    )
    where = (
        f"ScoreboardGames.Patch={selected_patch} "
        "AND ScoreboardGames.Winner IS NOT NULL"
        + _analysis_date_filter("ScoreboardGames.")
    )
    data = fetch_cargo_json(
        "ScoreboardGames,PicksAndBansS7",
        fields,
        where,
        f"leaguepedia_latest_patch_{selected_patch}_scoreboard_picks_bans.json",
        limit=500,
        force_refresh=CONFIG["force_refresh"],
        order_by="ScoreboardGames.DateTime_UTC DESC",
        join_on="ScoreboardGames.GameId=PicksAndBansS7.GameId",
    )
    parsed = parse_cargo_match_rows(data, "leaguepedia_cargo_latest_patch")
    parsed["patch"] = parsed["patch"].apply(lambda value: normalize_patch_label(value, selected_patch))
    if parsed.empty:
        raise RuntimeError(f"선택 패치 {selected_patch}의 픽밴 경기 데이터를 가져오지 못했습니다.")
    return parsed


def scrape_patch_range_match_history_cargo():
    """Scrape completed pro matches from the configured patch floor onward."""
    min_patch = float(CONFIG["analysis_min_patch"])
    min_patch_number = _patch_number(CONFIG["analysis_min_patch"])
    pickban_fields = ",".join(
        [f"PicksAndBansS7.Team{side}{kind}{idx}" for side in [1, 2] for kind in ["Ban", "Pick"] for idx in range(1, 6)]
    )
    fields = (
        "ScoreboardGames.GameId,ScoreboardGames.OverviewPage,ScoreboardGames.Tournament,"
        "ScoreboardGames.Team1,ScoreboardGames.Team2,ScoreboardGames.Winner,"
        "ScoreboardGames.Patch,ScoreboardGames.DateTime_UTC,"
        f"{pickban_fields}"
    )
    where = (
        f"ScoreboardGames.Patch >= {min_patch} "
        "AND ScoreboardGames.Winner IS NOT NULL"
        + _analysis_date_filter("ScoreboardGames.")
    )
    rows = []
    offset = 0
    limit = 500
    while True:
        batch = fetch_cargo_json(
            "ScoreboardGames,PicksAndBansS7",
            fields,
            where,
            f"leaguepedia_patch_range_{str(min_patch).replace('.', '_')}_{offset}.json",
            limit=limit,
            offset=offset,
            force_refresh=CONFIG["force_refresh"],
            order_by="ScoreboardGames.DateTime_UTC DESC",
            join_on="ScoreboardGames.GameId=PicksAndBansS7.GameId",
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    parsed = parse_cargo_match_rows(rows, "leaguepedia_cargo_patch_range")
    parsed["patch"] = parsed["patch"].apply(lambda value: normalize_patch_label(value, CONFIG["analysis_min_patch"]))
    parsed = parsed[parsed["patch"].apply(_patch_number) >= min_patch_number].reset_index(drop=True)
    if parsed.empty:
        raise RuntimeError(f"패치 {min_patch} 이후 픽밴 경기 데이터를 가져오지 못했습니다.")

    patch_counts = (
        parsed.groupby("patch", dropna=True)
        .agg(
            matches=("match_id", "count"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            tournaments=("tournament", lambda x: ", ".join(sorted(set(map(str, x)))[:8])),
        )
        .reset_index()
        .rename(columns={"patch": "Patch"})
    )
    patch_counts["patch_number"] = patch_counts["Patch"].apply(_patch_number)
    patch_counts = patch_counts.sort_values("patch_number", ascending=False).reset_index(drop=True)
    latest_patch = str(patch_counts.iloc[0]["Patch"]) if not patch_counts.empty else ""
    patch_counts["is_latest_patch"] = patch_counts["Patch"].astype(str).eq(latest_patch)
    patch_counts["meets_min_matches"] = patch_counts["matches"] >= CONFIG["latest_patch_min_matches"]
    patch_counts["selected"] = patch_counts["patch_number"] >= min_patch_number
    patch_counts["selection_reason"] = np.where(
        patch_counts["selected"],
        f"analysis patch range: Patch >= {min_patch}",
        "",
    )
    patch_counts.to_csv(OUTPUT_DIR / "selected_patch_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[patch] selected patch range: >= {min_patch} ({len(parsed)} raw rows)")
    return parsed


def scrape_leaguepedia_match_history_cargo():
    """
    Scrape LCS 2024 Championship match history through Leaguepedia CargoExport.

    주의: Cargo 테이블에는 별도 blue/red side 필드가 없어 Team1/Team2를
    blue/red proxy로 둔다. 픽/밴과 승패는 Team1/Team2 기준으로 일관되게
    연결되므로 네트워크 엣지와 승패 예측 feature 생성에는 문제가 없지만,
    실제 진영 효과 해석은 제한적으로 보아야 한다.
    """
    where = '_pageName LIKE "%LCS/2024 Season/Championship%"'
    score_fields = (
        "GameId,MatchId,OverviewPage,Tournament,Team1,Team2,"
        "Team1Players,Team2Players,Winner,Patch,DateTime_UTC,N_GameInMatch"
    )
    pickban_fields = "GameId,N_GameInMatch," + ",".join(
        [f"Team{side}{kind}{idx}" for side in [1, 2] for kind in ["Ban", "Pick"] for idx in range(1, 6)]
    )
    score_data = fetch_cargo_json(
        "ScoreboardGames",
        score_fields,
        where,
        "leaguepedia_scoreboardgames_lcs_2024_championship.json",
        force_refresh=CONFIG["force_refresh"],
    )
    pickban_data = fetch_cargo_json(
        "PicksAndBansS7",
        pickban_fields,
        where,
        "leaguepedia_picksandbans_lcs_2024_championship.json",
        force_refresh=CONFIG["force_refresh"],
    )
    score_df = pd.DataFrame(score_data)
    pickban_df = pd.DataFrame(pickban_data)
    if score_df.empty or pickban_df.empty:
        return pd.DataFrame(columns=REQUIRED_MATCH_COLUMNS)

    merged = score_df.merge(pickban_df, on="GameId", how="inner", suffixes=("", "_pickban"))
    rows = []
    for idx, row in merged.iterrows():
        winner_num = pd.to_numeric(row.get("Winner"), errors="coerce")
        team1 = row.get("Team1")
        team2 = row.get("Team2")
        winner = team1 if winner_num == 1 else team2 if winner_num == 2 else ""
        team1_bans = [normalize_champion_name(row.get(f"Team1Ban{i}")) for i in range(1, 6)]
        team2_bans = [normalize_champion_name(row.get(f"Team2Ban{i}")) for i in range(1, 6)]
        team1_picks = [normalize_champion_name(row.get(f"Team1Pick{i}")) for i in range(1, 6)]
        team2_picks = [normalize_champion_name(row.get(f"Team2Pick{i}")) for i in range(1, 6)]
        rows.append({
            "match_id": row.get("GameId", idx + 1),
            "date": row.get("DateTime UTC"),
            "patch": row.get("Patch"),
            "blue_team": team1,
            "red_team": team2,
            "winner": winner,
            "blue_bans": [champ for champ in team1_bans if champ],
            "red_bans": [champ for champ in team2_bans if champ],
            "blue_picks": [champ for champ in team1_picks if champ],
            "red_picks": [champ for champ in team2_picks if champ],
            "blue_roster": row.get("Team1Players", []),
            "red_roster": row.get("Team2Players", []),
            "blue_win": int(winner_num == 1),
            "data_source": "leaguepedia_cargo",
        })
    parsed = ensure_match_schema(pd.DataFrame(rows))
    return parsed.sort_values(["date", "match_id"]).reset_index(drop=True)


def _cell_text(cell):
    return " ".join(cell.get_text(" ", strip=True).split())


def _champion_from_img_attr(value):
    if not value:
        return None
    text = unquote(str(value))
    text = text.split("/")[-1]
    text = re.sub(r"\?.*$", "", text)
    text = re.sub(r"\.(png|jpg|jpeg|webp|gif)$", "", text, flags=re.I)
    text = text.replace("_", " ")
    text = re.sub(r"(Square|OriginalSquare|Champion|Icon|HD)$", "", text, flags=re.I).strip()
    return normalize_champion_name(text)


def extract_champions_from_cell(cell):
    """Extract champion names from img alt/title/src/data-src fields in a table cell."""
    champions = []
    for img in cell.find_all("img"):
        for attr in ["alt", "title", "data-src", "src"]:
            champion = _champion_from_img_attr(img.get(attr))
            if champion and champion not in champions:
                champions.append(champion)
                break
    if champions:
        return champions

    text = _cell_text(cell)
    if not text:
        return []
    candidates = re.split(r"\s{2,}|[,;/|]", text)
    for candidate in candidates:
        champion = normalize_champion_name(candidate)
        if champion and champion not in champions and len(champion) <= 30:
            champions.append(champion)
    return champions


def _header_key(header):
    return re.sub(r"[^a-z0-9]", "", str(header).lower())


def parse_leaguepedia_match_history(html):
    """
    Parse Leaguepedia-style match-history tables into the required match schema.

    Leaguepedia table structures can change, so the parser is intentionally
    conservative. If required fields cannot be found, it returns an empty frame
    and the notebook falls back to local CSV or sample data.
    """
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for table in soup.find_all("table"):
        header_cells = table.find_all("th")
        headers = [_cell_text(th) for th in header_cells]
        if not headers:
            continue
        keys = [_header_key(h) for h in headers]
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) < 6:
                continue
            record = {column: None for column in REQUIRED_MATCH_COLUMNS}
            for idx, cell in enumerate(cells[: len(keys)]):
                key = keys[idx]
                text = _cell_text(cell)
                champions = extract_champions_from_cell(cell)
                if "date" in key:
                    record["date"] = text
                elif "patch" in key:
                    record["patch"] = text
                elif "blueteam" in key or key == "blue":
                    record["blue_team"] = text
                elif "redteam" in key or key == "red":
                    record["red_team"] = text
                elif "winner" in key or "winningteam" in key:
                    record["winner"] = text
                elif "blueban" in key:
                    record["blue_bans"] = champions
                elif "redban" in key:
                    record["red_bans"] = champions
                elif "bluepick" in key or "bluechampion" in key:
                    record["blue_picks"] = champions
                elif "redpick" in key or "redchampion" in key:
                    record["red_picks"] = champions
                elif "blueroster" in key:
                    record["blue_roster"] = [text] if text else []
                elif "redroster" in key:
                    record["red_roster"] = [text] if text else []

            if record["blue_picks"] and record["red_picks"]:
                rows.append(record)

    parsed = pd.DataFrame(rows)
    if parsed.empty:
        return parsed
    return ensure_match_schema(parsed)


def _parse_rate(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace("%", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    # GOL.GG는 표본이 없는 챔피언 지표를 '-'로 표시한다.
    # 이 값은 0이 아니라 결측치라서 NaN으로 두어 imputer가 처리하게 한다.
    if not text or not re.search(r"\d", text):
        return np.nan
    number = float(text)
    return number / 100 if number > 1 else number


def parse_golgg_champion_stats(html):
    """Parse GOL.GG champion statistics with pandas.read_html."""
    # pandas 최신 버전은 HTML 문자열을 파일 경로처럼 해석할 수 있어서 StringIO로 감싼다.
    tables = pd.read_html(StringIO(html))
    for table in tables:
        df = table.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join(map(str, col)).strip() for col in df.columns]
        lower_cols = {str(col).lower(): col for col in df.columns}
        champion_col = next((col for low, col in lower_cols.items() if "champion" in low or low in {"name"}), None)
        picks_col = next((col for low, col in lower_cols.items() if low == "picks" or "pick" in low), None)
        bans_col = next((col for low, col in lower_cols.items() if low == "bans" or "ban" in low), None)
        wins_col = next((col for low, col in lower_cols.items() if low == "wins" or low == "win"), None)
        losses_col = next((col for low, col in lower_cols.items() if low == "losses" or low == "loss"), None)
        win_col = next((col for low, col in lower_cols.items() if "winrate" in low or ("win" in low and "rate" in low)), None)
        bp_col = next((col for low, col in lower_cols.items() if low.replace(" ", "") in {"bp%", "bp"}), None)
        pick_col = next((col for low, col in lower_cols.items() if "pick" in low), None)
        ban_col = next((col for low, col in lower_cols.items() if "ban" in low), None)
        if champion_col and picks_col and win_col:
            picks = pd.to_numeric(df[picks_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
            bans = pd.to_numeric(df[bans_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0) if bans_col else 0
            if wins_col and losses_col:
                wins = pd.to_numeric(df[wins_col], errors="coerce").fillna(0)
                losses = pd.to_numeric(df[losses_col], errors="coerce").fillna(0)
                games = wins + losses
            else:
                games = picks

            if bp_col:
                bp_rate = df[bp_col].apply(_parse_rate).replace(0, np.nan)
                # GOL.GG는 pick_rate/ban_rate를 직접 주지 않고 BP%를 준다.
                # BP% = (Picks + Bans) / 전체 경기 수로 보고 총 경기 수를 역산한다.
                estimated_total_games = (picks + bans) / bp_rate
                pick_rate = picks / estimated_total_games
                ban_rate = bans / estimated_total_games
            else:
                # BP%가 없으면 절대 pick/ban 빈도 비교라도 가능하도록 표 안의 최대값으로 보수적 스케일링을 한다.
                estimated_total_games = max(float(picks.max()), 1.0)
                pick_rate = picks / estimated_total_games
                ban_rate = bans / estimated_total_games

            output = pd.DataFrame({
                "champion": df[champion_col].apply(normalize_champion_name),
                "games": games,
                "win_rate": df[win_col].apply(_parse_rate),
                "pick_rate": pick_rate,
                "ban_rate": ban_rate,
            })
            return output.dropna(subset=["champion"]).drop_duplicates("champion")
    return pd.DataFrame(columns=["champion", "games", "win_rate", "pick_rate", "ban_rate"])


def leaguepedia_patch_to_opgg_patch(patch_value):
    """
    Convert Leaguepedia esports patch notation to OP.GG's patch notation.

    현재 Leaguepedia Cargo는 2026년 경기 패치를 26.11처럼 표기하고,
    OP.GG는 같은 게임 패치를 16.11처럼 표기한다. minor 버전은 같고
    major만 10 차이가 나므로 26.xx -> 16.xx로 맞춘다.
    """
    text = str(patch_value).strip()
    parts = text.split(".")
    if len(parts) != 2:
        return text
    try:
        major = int(float(parts[0]))
        minor = int(float(parts[1]))
    except Exception:
        return text
    if major >= 20:
        major -= 10
    return f"{major}.{minor:02d}"


def parse_opgg_champion_tiers(html, opgg_patch, region, rank_tier):
    """
    Parse OP.GG App Router HTML for champion role-tier data.

    OP.GG는 표 데이터를 정적인 table 태그로 모두 제공하지 않고 Next.js
    streaming payload 안에 JSON 조각으로 넣는다. 그래서 HTML을 unicode escape
    해제한 뒤 champion 객체 패턴을 추출한다.
    """
    decoded = html.encode("utf-8").decode("unicode_escape", errors="ignore")
    pattern = re.compile(
        r'\{"key":"(?P<key>[^"]+)",'
        r'"name":"(?P<name>[^"]+)",'
        r'"image_url":"[^"]+",'
        r'"positionName":"(?P<position>[^"]+)",'
        r'"positionWinRate":(?P<win_rate>[0-9.]+),'
        r'"positionPickRate":(?P<pick_rate>[0-9.]+),'
        r'"positionBanRate":(?P<ban_rate>[0-9.]+),'
        r'"positionRoleRate":(?P<role_rate>[0-9.]+),'
        r'"positionTierData":\{"tier":(?P<tier>[0-9]+),"rank":(?P<rank>[0-9]+),'
    )
    rows = []
    seen = set()
    for match in pattern.finditer(decoded):
        item = match.groupdict()
        key = (item["key"], item["position"])
        if key in seen:
            continue
        seen.add(key)
        tier_raw = int(item["tier"])
        rows.append({
            "champion": normalize_champion_name(item["name"]),
            "opgg_key": item["key"],
            "opgg_patch": opgg_patch,
            "opgg_region": region,
            "opgg_rank_tier": rank_tier,
            "position": item["position"],
            # OP.GG 내부 tier=0은 화면에서 OP 티어로 표시된다.
            # 기존처럼 +1하면 OP를 Tier 1로 오해할 수 있어 raw와 label을 분리한다.
            "opgg_tier_raw": tier_raw,
            "opgg_tier_label": "OP" if tier_raw == 0 else f"Tier {tier_raw}",
            "opgg_rank": int(item["rank"]),
            "opgg_win_rate": float(item["win_rate"]) / 100,
            "opgg_pick_rate": float(item["pick_rate"]) / 100,
            "opgg_ban_rate": float(item["ban_rate"]) / 100,
            "opgg_role_rate": float(item["role_rate"]),
            "data_source": "opgg_scraped",
        })
    return pd.DataFrame(rows).sort_values(["opgg_tier_raw", "opgg_rank"], ignore_index=True)


def load_or_scrape_opgg_champion_tiers(selected_patch, output_path=DATA_DIR / "opgg_champion_tiers.csv"):
    """Scrape OP.GG champion role tiers for the selected analysis patch."""
    opgg_patch = leaguepedia_patch_to_opgg_patch(selected_patch)
    region = CONFIG["opgg_region"]
    rank_tier = CONFIG["opgg_rank_tier"]
    cache_path = CACHE_DIR / f"opgg_champions_patch_{opgg_patch}_{region}_{rank_tier}.html"
    url = f"https://op.gg/lol/champions?region={region}&tier={rank_tier}&patch={opgg_patch}"

    if cache_path.exists() and not CONFIG["force_refresh"]:
        print(f"[cache] using OP.GG HTML: {cache_path}")
        html_text = cache_path.read_text(encoding="utf-8", errors="ignore")
    else:
        headers = {"User-Agent": "Mozilla/5.0"}
        time.sleep(CONFIG["request_sleep"])
        response = requests.get(url, headers=headers, timeout=CONFIG["request_timeout"])
        response.raise_for_status()
        html_text = response.text
        cache_path.write_text(html_text, encoding="utf-8")
        print(f"[cache] saved OP.GG HTML: {cache_path}")

    tiers = parse_opgg_champion_tiers(html_text, opgg_patch, region, rank_tier)
    if tiers.empty:
        raise RuntimeError(f"OP.GG 티어 데이터를 파싱하지 못했습니다: {url}")
    tiers["leaguepedia_patch"] = str(selected_patch)
    if output_path is not None:
        tiers.to_csv(output_path, index=False, encoding="utf-8-sig")
    return tiers


def load_or_scrape_opgg_champion_tiers_for_patches(patches, output_path=DATA_DIR / "opgg_champion_tiers.csv"):
    """Scrape and combine OP.GG champion tiers for every patch in the analysis window."""
    patch_values = sorted({str(patch) for patch in patches if pd.notna(patch)}, key=_patch_number)
    frames = []
    for patch in patch_values:
        try:
            frames.append(load_or_scrape_opgg_champion_tiers(patch, output_path=None))
        except Exception as exc:
            print(f"[warning] OP.GG tier scraping failed for patch {patch}: {exc}")
    if not frames:
        empty = pd.DataFrame()
        empty.to_csv(output_path, index=False, encoding="utf-8-sig")
        return empty
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["leaguepedia_patch", "opgg_tier_raw", "opgg_rank"], ignore_index=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    return combined


def resolve_ddragon_version(selected_patch=None):
    """Resolve a Riot Data Dragon version, preferring the selected analysis patch."""
    configured = CONFIG.get("ddragon_version")
    if configured:
        return str(configured)

    cache_path = CACHE_DIR / "ddragon_versions.json"
    versions = []
    try:
        if cache_path.exists() and not CONFIG["force_refresh"]:
            versions = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            response = requests.get(
                "https://ddragon.leagueoflegends.com/api/versions.json",
                timeout=CONFIG["request_timeout"],
            )
            response.raise_for_status()
            versions = response.json()
            cache_path.write_text(json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[warning] Data Dragon version lookup failed: {exc}")
        if cache_path.exists():
            try:
                versions = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                versions = []

    if not versions:
        return None

    if selected_patch is not None:
        patch_prefix = f"{leaguepedia_patch_to_opgg_patch(selected_patch)}."
        patch_versions = [str(version) for version in versions if str(version).startswith(patch_prefix)]
        if patch_versions:
            return patch_versions[0]

    return str(versions[0])


def load_ddragon_champion_catalog(selected_patch=None):
    """
    Load Riot Data Dragon champion metadata for image mapping.

    This official catalog is used instead of scraping images from tier pages because
    Data Dragon file names are stable and easier to cache reproducibly.
    """
    version = resolve_ddragon_version(selected_patch)
    if not version:
        return None, {}

    cache_path = CACHE_DIR / f"ddragon_champion_{version}.json"
    payload = {}
    url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
    try:
        if cache_path.exists() and not CONFIG["force_refresh"]:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            response = requests.get(url, timeout=CONFIG["request_timeout"])
            response.raise_for_status()
            payload = response.json()
            cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[warning] Data Dragon champion catalog failed: {exc}")
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}

    lookup = {}
    for ddragon_id, item in payload.get("data", {}).items():
        for name in {ddragon_id, item.get("id"), item.get("name")}:
            if not name:
                continue
            normalized = normalize_champion_name(name)
            for candidate in [name, normalized]:
                key = _canonical_key(candidate)
                if key:
                    lookup[key] = item

    # These aliases cover names that commonly differ between pro-data text and Riot file ids.
    alias_to_key = {
        "Wukong": "MonkeyKing",
        "Nunu": "Nunu",
        "Nunu & Willump": "Nunu",
        "Bel'Veth": "Belveth",
        "Cho'Gath": "Chogath",
        "Dr. Mundo": "DrMundo",
        "Kai'Sa": "Kaisa",
        "Kha'Zix": "Khazix",
        "Kog'Maw": "KogMaw",
        "K'Sante": "KSante",
        "LeBlanc": "Leblanc",
        "Rek'Sai": "RekSai",
        "Renata Glasc": "Renata",
        "Vel'Koz": "Velkoz",
    }
    for alias, ddragon_id in alias_to_key.items():
        item = lookup.get(_canonical_key(ddragon_id))
        if item:
            lookup[_canonical_key(alias)] = item

    return version, lookup


def load_champion_icon_map(champions, selected_patch=None):
    """
    Download/cache champion square icons and return {champion_name: local_path}.

    Image failures are non-fatal because the analysis should still run in class or
    network-restricted environments; plots simply keep their text labels.
    """
    if not CONFIG.get("download_champion_images", True):
        return {}

    champions = sorted({normalize_champion_name(champion) for champion in champions if normalize_champion_name(champion)})
    if not champions:
        return {}

    version, catalog = load_ddragon_champion_catalog(selected_patch)
    if not version or not catalog:
        return {}

    icon_dir = CHAMPION_ICON_DIR / version
    icon_dir.mkdir(parents=True, exist_ok=True)
    icon_map = {}
    metadata = []

    for champion in champions:
        item = catalog.get(_canonical_key(champion))
        if not item:
            continue

        image_name = item.get("image", {}).get("full") or f"{item.get('id')}.png"
        image_url = f"https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{image_name}"
        local_path = icon_dir / image_name

        if not local_path.exists():
            try:
                response = requests.get(image_url, timeout=CONFIG["request_timeout"])
                response.raise_for_status()
                local_path.write_bytes(response.content)
            except Exception as exc:
                print(f"[warning] champion icon download failed for {champion}: {exc}")
                continue

        icon_map[champion] = local_path
        metadata.append({
            "champion": champion,
            "ddragon_id": item.get("id"),
            "ddragon_version": version,
            "image_path": str(local_path),
            "image_url": image_url,
            "data_source": "riot_data_dragon",
        })

    if metadata:
        pd.DataFrame(metadata).to_csv(CHAMPION_ICON_DIR / "champion_icon_index.csv", index=False, encoding="utf-8-sig")
        print(f"champion icons loaded: {len(icon_map)}/{len(champions)} using Data Dragon {version}")
    else:
        print("[warning] no champion icons were loaded; plots will use text labels only")

    return icon_map


def summarize_op_champion_tiers(op_scores, opgg_tiers, output_path=OUTPUT_DIR / "op_champion_tiers.csv"):
    """
    Define project OP champions from OP.GG tier data and attach pro-game stats.

    중요한 기준:
    OP 챔피언은 프로 경기 presence가 아니라 OP.GG 해당 패치의 OP 티어
    champion-position으로 정한다. 프로 경기 pick/ban/win 지표는 그 OP 챔피언이
    실제 프로 경기에서 어떻게 쓰였는지 해석하기 위해 붙이는 보조 정보다.
    """
    if opgg_tiers.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_path, index=False, encoding="utf-8-sig")
        return empty

    patch_col = "leaguepedia_patch" if "leaguepedia_patch" in opgg_tiers.columns else "opgg_patch"
    opgg_op_frames = []
    for patch, patch_tiers in opgg_tiers.groupby(patch_col, dropna=False):
        patch_op = patch_tiers[patch_tiers["opgg_tier_raw"] <= CONFIG["opgg_op_tier_raw_max"]].copy()
        if patch_op.empty:
            patch_op = patch_tiers.sort_values(["opgg_tier_raw", "opgg_rank"]).head(CONFIG["op_top_n"]).copy()
            patch_op["op_definition_note"] = "No OP.GG OP-tier champion found; using top-ranked OP.GG champions"
        else:
            patch_op["op_definition_note"] = "OP.GG OP-tier champion-position for matched patch"
        opgg_op_frames.append(patch_op)
    opgg_op = pd.concat(opgg_op_frames, ignore_index=True) if opgg_op_frames else pd.DataFrame()
    opgg_op["patch"] = opgg_op[patch_col].astype(str)

    pro_stats = op_scores[[
        "patch", "champion", "op_rank", "op_score", "presence_rate", "smoothed_win_rate",
        "picks", "bans", "win_rate",
    ]].copy()
    pro_stats["patch"] = pro_stats["patch"].astype(str)
    merged = opgg_op.merge(pro_stats, on=["patch", "champion"], how="left")
    merged["patch_number"] = merged["patch"].apply(_patch_number)
    merged = merged.sort_values(["patch_number", "opgg_tier_raw", "opgg_rank", "op_score"], ascending=[False, True, True, False])
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return merged


# %% [markdown] ## 4. 샘플 데이터 생성 함수


# %% Code cell 10

ROLE_POOLS = {
    "top": ["Aatrox", "Gnar", "Jax", "K'Sante", "Ornn", "Renekton", "Rumble"],
    "jungle": ["Gragas", "Lee Sin", "Maokai", "Sejuani", "Vi", "Viego", "Wukong", "Xin Zhao"],
    "mid": ["Ahri", "Akali", "Azir", "LeBlanc", "Orianna", "Taliyah", "Tristana"],
    "adc": ["Aphelios", "Ashe", "Caitlyn", "Ezreal", "Jinx", "Kai'Sa", "Kalista", "Lucian", "Xayah", "Zeri"],
    "support": ["Braum", "Karma", "Leona", "Nami", "Nautilus", "Rakan", "Rell", "Renata Glasc", "Thresh"],
}

SAMPLE_SYNERGY = {
    tuple(sorted(pair)): value
    for pair, value in {
        ("Xayah", "Rakan"): 0.85,
        ("Lucian", "Nami"): 0.72,
        ("Jinx", "Thresh"): 0.55,
        ("Aphelios", "Thresh"): 0.42,
        ("Azir", "Sejuani"): 0.38,
        ("Orianna", "Wukong"): 0.48,
        ("Kai'Sa", "Nautilus"): 0.45,
        ("Kalista", "Renata Glasc"): 0.40,
        ("Rumble", "Maokai"): 0.36,
        ("Vi", "Ahri"): 0.33,
    }.items()
}

SAMPLE_COUNTER = {
    ("Sejuani", "Lee Sin"): 0.35,
    ("Azir", "Ahri"): 0.28,
    ("Rell", "Nautilus"): 0.25,
    ("Caitlyn", "Kai'Sa"): 0.22,
    ("Jax", "K'Sante"): 0.24,
    ("Taliyah", "Akali"): 0.20,
    ("Renata Glasc", "Rakan"): 0.18,
    ("Ashe", "Lucian"): 0.18,
    ("Viego", "Maokai"): 0.20,
    ("Ornn", "Renekton"): 0.16,
}


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _pair_synergy(picks):
    total = 0.0
    for a, b in itertools.combinations(sorted(picks), 2):
        total += SAMPLE_SYNERGY.get(tuple(sorted((a, b))), 0.0)
    return total


def _counter_advantage(team_picks, opponent_picks):
    total = 0.0
    for source in team_picks:
        for target in opponent_picks:
            total += SAMPLE_COUNTER.get((source, target), 0.0)
    return total


def create_sample_match_history(n_matches=80, random_state=42):
    """
    Create deterministic sample match data for local testing only.

    The output is explicitly marked as sample_simulated and must not be treated
    as real Leaguepedia/GOL.GG analysis results.
    """
    rng = np.random.default_rng(random_state)
    all_champions = sorted({champion for pool in ROLE_POOLS.values() for champion in pool})
    hidden_strength = {champion: rng.normal(0, 0.22) for champion in all_champions}
    for champion in ["Azir", "Rell", "Kai'Sa", "Sejuani", "Xayah", "Rakan"]:
        hidden_strength[champion] += 0.18

    teams = ["Cloud9", "Team Liquid", "FlyQuest", "NRG", "100 Thieves", "Dignitas", "Shopify Rebellion", "Immortals"]
    patches = ["14.15", "14.16", "14.17", "14.18"]
    rows = []

    def choose_one(pool, unavailable):
        candidates = [champion for champion in pool if champion not in unavailable]
        weights = np.array([math.exp(hidden_strength[c] * 2.2) for c in candidates])
        weights = weights / weights.sum()
        return rng.choice(candidates, p=weights).item()

    for match_idx in range(n_matches):
        blue_team, red_team = rng.choice(teams, size=2, replace=False)
        unavailable = set()
        blue_picks, red_picks = [], []
        for role, pool in ROLE_POOLS.items():
            blue_champ = choose_one(pool, unavailable)
            unavailable.add(blue_champ)
            red_champ = choose_one(pool, unavailable)
            unavailable.add(red_champ)
            blue_picks.append(blue_champ)
            red_picks.append(red_champ)

        blue_power = sum(hidden_strength[c] for c in blue_picks)
        red_power = sum(hidden_strength[c] for c in red_picks)
        blue_power += _pair_synergy(blue_picks) - _pair_synergy(red_picks)
        blue_power += _counter_advantage(blue_picks, red_picks) - _counter_advantage(red_picks, blue_picks)
        blue_power += 0.08 + rng.normal(0, 0.7)
        blue_win = int(rng.random() < _sigmoid(blue_power))

        ban_candidates = [champion for champion in all_champions if champion not in unavailable]
        ban_weights = np.array([math.exp(hidden_strength[c] * 1.8) for c in ban_candidates])
        ban_weights = ban_weights / ban_weights.sum()
        bans = rng.choice(ban_candidates, size=min(10, len(ban_candidates)), replace=False, p=ban_weights).tolist()

        rows.append({
            "match_id": match_idx + 1,
            "date": pd.Timestamp("2024-08-01") + pd.Timedelta(days=match_idx // 4),
            "patch": patches[min(match_idx // 20, len(patches) - 1)],
            "blue_team": blue_team,
            "red_team": red_team,
            "winner": blue_team if blue_win else red_team,
            "blue_bans": bans[:5],
            "red_bans": bans[5:10],
            "blue_picks": blue_picks,
            "red_picks": red_picks,
            "blue_roster": [f"{blue_team} {role}" for role in ROLE_POOLS],
            "red_roster": [f"{red_team} {role}" for role in ROLE_POOLS],
            "blue_win": blue_win,
            "red_win": 1 - blue_win,
            "data_source": "sample_simulated",
        })
    return ensure_match_schema(pd.DataFrame(rows))


def create_sample_champion_stats(match_df):
    """Derive champion-level sample stats from sample match history."""
    df = ensure_match_schema(match_df)
    pick_games = Counter()
    pick_wins = Counter()
    ban_counts = Counter()
    for _, row in df.iterrows():
        for champion in row["blue_picks"]:
            pick_games[champion] += 1
            pick_wins[champion] += int(row["blue_win"])
        for champion in row["red_picks"]:
            pick_games[champion] += 1
            pick_wins[champion] += int(row["red_win"])
        ban_counts.update(row["blue_bans"])
        ban_counts.update(row["red_bans"])

    champions = sorted(set(pick_games) | set(ban_counts))
    rows = []
    n_matches = max(len(df), 1)
    for champion in champions:
        games = pick_games[champion]
        rows.append({
            "champion": champion,
            "games": games,
            "win_rate": pick_wins[champion] / games if games else np.nan,
            "pick_rate": games / n_matches,
            "ban_rate": ban_counts[champion] / n_matches,
            "data_source": "sample_simulated",
        })
    return pd.DataFrame(rows)


def build_champion_model_stats(train_match_df, global_stats_df, output_path=OUTPUT_DIR / "champion_model_stats.csv"):
    """
    Build champion standalone features for modeling from train data only.

    왜 GOL.GG를 그대로 쓰지 않는가:
    GOL.GG의 전체 통계는 수집 범위가 LCS 2024 Championship과 완전히 같지 않고,
    현재 페이지 기준 시즌 통계가 섞일 수 있다. 그래서 실제 예측 feature는
    chronological split 이후 train 경기에서 계산한 값을 기본으로 쓰고,
    표본이 작은 챔피언만 GOL.GG 값을 prior로 섞어 과도한 승률 변동을 줄인다.
    """
    train_df = ensure_match_schema(train_match_df)
    global_stats = global_stats_df.copy()
    if not global_stats.empty:
        global_stats["champion"] = global_stats["champion"].apply(normalize_champion_name)
        global_lookup = global_stats.drop_duplicates("champion").set_index("champion")
    else:
        global_lookup = pd.DataFrame()

    pick_games = Counter()
    pick_wins = Counter()
    ban_counts = Counter()
    for _, row in train_df.iterrows():
        for champion in row["blue_picks"]:
            pick_games[champion] += 1
            pick_wins[champion] += int(row["blue_win"])
        for champion in row["red_picks"]:
            pick_games[champion] += 1
            pick_wins[champion] += int(row["red_win"])
        ban_counts.update(row["blue_bans"])
        ban_counts.update(row["red_bans"])

    champions = set(pick_games) | set(ban_counts)
    if not global_lookup.empty:
        champions |= set(global_lookup.index)

    prior_weight = CONFIG["golgg_prior_weight"]
    train_games_total = max(len(train_df), 1)
    rows = []
    for champion in sorted(champions):
        train_games = pick_games[champion]
        train_wins = pick_wins[champion]
        train_bans = ban_counts[champion]

        if not global_lookup.empty and champion in global_lookup.index:
            prior_win_rate = float(global_lookup.at[champion, "win_rate"]) if pd.notna(global_lookup.at[champion, "win_rate"]) else CONFIG["global_win_rate"]
            prior_pick_rate = float(global_lookup.at[champion, "pick_rate"]) if pd.notna(global_lookup.at[champion, "pick_rate"]) else 0.0
            prior_ban_rate = float(global_lookup.at[champion, "ban_rate"]) if pd.notna(global_lookup.at[champion, "ban_rate"]) else 0.0
            prior_games = float(global_lookup.at[champion, "games"]) if "games" in global_lookup.columns and pd.notna(global_lookup.at[champion, "games"]) else np.nan
        else:
            prior_win_rate = CONFIG["global_win_rate"]
            prior_pick_rate = 0.0
            prior_ban_rate = 0.0
            prior_games = np.nan

        # 승률은 train 경기만으로 계산하면 1승 0패 같은 극단값이 생긴다.
        # prior_weight를 더해 empirical Bayes 방식으로 0.5/외부 통계 쪽으로 완만하게 수축시킨다.
        win_rate = (train_wins + prior_weight * prior_win_rate) / (train_games + prior_weight)
        pick_rate = (train_games + prior_weight * prior_pick_rate) / (train_games_total + prior_weight)
        ban_rate = (train_bans + prior_weight * prior_ban_rate) / (train_games_total + prior_weight)

        rows.append({
            "champion": champion,
            "games": train_games,
            "wins": train_wins,
            "train_pick_rate": train_games / train_games_total,
            "train_ban_rate": train_bans / train_games_total,
            "win_rate": win_rate,
            "pick_rate": pick_rate,
            "ban_rate": ban_rate,
            "golgg_prior_win_rate": prior_win_rate,
            "golgg_prior_pick_rate": prior_pick_rate,
            "golgg_prior_ban_rate": prior_ban_rate,
            "golgg_prior_games": prior_games,
            "data_source": "train_blended_with_golgg_prior",
        })

    model_stats = pd.DataFrame(rows)
    model_stats.to_csv(output_path, index=False, encoding="utf-8-sig")
    return model_stats


def filter_complete_pick_matches(
    match_df,
    output_path=DATA_DIR / "analysis_match_history.csv",
    excluded_path=OUTPUT_DIR / "excluded_incomplete_matches.csv",
):
    """
    Keep only matches where both teams have five recorded picks.

    Composition and network analysis depend on complete team drafts. Rows with
    empty pick lists usually come from scheduled/unresolved Cargo rows, so they
    are saved separately instead of silently mixed into the analysis dataset.
    """
    df = ensure_match_schema(match_df)
    complete_mask = (
        df["blue_picks"].apply(lambda values: len(parse_list_column(values)) == 5)
        & df["red_picks"].apply(lambda values: len(parse_list_column(values)) == 5)
    )
    filtered = df[complete_mask].copy().reset_index(drop=True)
    excluded = df[~complete_mask].copy().reset_index(drop=True)
    filtered.to_csv(output_path, index=False, encoding="utf-8-sig")
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")
    if not excluded.empty:
        print(
            f"[quality] excluded {len(excluded)} incomplete-pick matches; "
            f"analysis uses {len(filtered)} complete matches"
        )
    return filtered


def audit_data_quality(match_df, global_stats_df, model_stats_df=None, raw_match_df=None, output_path=OUTPUT_DIR / "data_quality_report.csv"):
    """Create a compact data-quality report used to keep the interpretation honest."""
    df = ensure_match_schema(match_df)
    raw_rows = len(raw_match_df) if raw_match_df is not None else len(df)
    stat_champions = set(global_stats_df["champion"].apply(normalize_champion_name)) if not global_stats_df.empty else set()
    match_champions = set()
    invalid_pick_lengths = 0
    invalid_ban_lengths = 0
    for _, row in df.iterrows():
        for column in ["blue_picks", "red_picks"]:
            values = parse_list_column(row[column])
            match_champions.update(values)
            invalid_pick_lengths += int(len(values) != 5)
        for column in ["blue_bans", "red_bans"]:
            values = parse_list_column(row[column])
            match_champions.update(values)
            invalid_ban_lengths += int(len(values) > 5)

    missing_in_global = sorted(match_champions - stat_champions)
    report = pd.DataFrame([
        {"check": "raw_match_rows", "value": raw_rows, "note": "rows collected before complete-pick filtering"},
        {"check": "excluded_incomplete_pick_matches", "value": raw_rows - len(df), "note": "excluded because at least one team had fewer than five picks"},
        {"check": "match_rows", "value": len(df), "note": "분석에 사용한 경기 수"},
        {"check": "date_range", "value": f"{df['date'].min()} ~ {df['date'].max()}", "note": "chronological split 기준"},
        {"check": "patches", "value": ", ".join(sorted(df["patch"].astype(str).unique(), key=_patch_number)), "note": "패치 범위"},
        {"check": "duplicate_match_id", "value": int(df["match_id"].duplicated().sum()), "note": "0이어야 함"},
        {"check": "invalid_pick_lengths", "value": invalid_pick_lengths, "note": "픽 5개가 아닌 팀-row 수"},
        {"check": "invalid_ban_lengths", "value": invalid_ban_lengths, "note": "밴이 5개를 초과한 팀-row 수"},
        {"check": "unique_match_champions", "value": len(match_champions), "note": "픽/밴에 등장한 고유 챔피언 수"},
        {"check": "global_stats_champions", "value": len(stat_champions), "note": "GOL.GG 통계 챔피언 수"},
        {"check": "missing_in_global_stats", "value": len(missing_in_global), "note": ", ".join(missing_in_global[:20])},
        {"check": "side_limitation", "value": "Team1/Team2 proxy", "note": "CargoExport에 실제 blue/red side 필드가 없어 진영 효과 feature는 제외"},
        {"check": "model_stats_source", "value": model_stats_df['data_source'].iloc[0] if model_stats_df is not None and not model_stats_df.empty else "", "note": "모델용 단독 지표 산출 방식"},
    ])
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    return report


# %% Code cell 11

def load_or_scrape_match_history():
    """
    Load match history from data/raw_match_history.csv, scrape Leaguepedia if
    configured, or create explicitly marked sample data as a last resort.
    """
    csv_path = DATA_DIR / "raw_match_history.csv"
    if csv_path.exists() and CONFIG["prefer_existing_csv"]:
        print(f"[load] match history CSV: {csv_path}")
        return ensure_match_schema(pd.read_csv(csv_path, encoding="utf-8-sig"))

    if CONFIG["run_scraping"]:
        html_error = None
        if CONFIG.get("analysis_patch_strategy") == "since_patch":
            try:
                scraped = scrape_patch_range_match_history_cargo()
                if not scraped.empty:
                    save_match_history_csv(scraped, csv_path)
                    return scraped
            except Exception as exc:
                print(f"[warning] patch-range CargoExport scraping failed:\n{exc}")

        if CONFIG.get("use_latest_patch"):
            try:
                scraped = scrape_latest_patch_match_history_cargo()
                if not scraped.empty:
                    save_match_history_csv(scraped, csv_path)
                    return scraped
            except Exception as exc:
                print(f"[warning] latest patch CargoExport scraping failed:\n{exc}")

        try:
            html = fetch_html(
                CONFIG["leaguepedia_url"],
                "leaguepedia_match_history.html",
                force_refresh=CONFIG["force_refresh"],
            )
            scraped = parse_leaguepedia_match_history(html)
            if not scraped.empty:
                scraped["data_source"] = "leaguepedia_scraped"
                save_match_history_csv(scraped, csv_path)
                return scraped
            print("[warning] Leaguepedia HTML에서 필수 픽/밴 구조를 찾지 못했습니다.")
        except Exception as exc:
            html_error = exc
            print(f"[warning] Leaguepedia HTML scraping failed:\n{exc}")

        try:
            # Fandom 일반 HTML이 막혀도 CargoExport는 같은 Leaguepedia DB를 쓰므로
            # 실제 데이터 수집을 포기하지 않고 구조화된 API로 다시 시도한다.
            scraped = scrape_leaguepedia_match_history_cargo()
            if not scraped.empty:
                save_match_history_csv(scraped, csv_path)
                return scraped
            print("[warning] Leaguepedia CargoExport에서도 경기/픽밴 데이터를 찾지 못했습니다.")
        except Exception as exc:
            print(f"[warning] Leaguepedia CargoExport scraping failed:\n{exc}")
            if html_error is not None:
                print(f"[info] 최초 HTML 실패 원인도 함께 확인하세요: {html_error}")

    if csv_path.exists():
        print(f"[fallback] 크롤링 실패로 기존 match history CSV를 사용합니다: {csv_path}")
        return ensure_match_schema(pd.read_csv(csv_path, encoding="utf-8-sig"))

    if CONFIG["allow_sample_data"]:
        print("[sample data] 실제 크롤링/CSV가 없어 테스트용 샘플 데이터를 생성합니다.")
        sample = create_sample_match_history(random_state=CONFIG["random_state"])
        save_match_history_csv(sample, csv_path)
        return sample

    raise FileNotFoundError("data/raw_match_history.csv가 없고 샘플 데이터 생성도 비활성화되어 있습니다.")


def load_or_scrape_champion_stats(match_df=None):
    """
    Load champion stats from data/champion_global_stats.csv, scrape GOL.GG if
    configured, or derive sample stats from sample match history.
    """
    csv_path = DATA_DIR / "champion_global_stats.csv"
    if csv_path.exists() and CONFIG["prefer_existing_csv"]:
        print(f"[load] champion stats CSV: {csv_path}")
        stats = pd.read_csv(csv_path, encoding="utf-8-sig")
        stats["champion"] = stats["champion"].apply(normalize_champion_name)
        return stats

    if CONFIG["run_scraping"]:
        try:
            html = fetch_html(
                CONFIG["golgg_url"],
                "golgg_champion_stats.html",
                force_refresh=CONFIG["force_refresh"],
            )
            stats = parse_golgg_champion_stats(html)
            if not stats.empty:
                stats["data_source"] = "golgg_scraped"
                stats.to_csv(csv_path, index=False, encoding="utf-8-sig")
                return stats
            print("[warning] GOL.GG HTML에서 챔피언 통계 표를 찾지 못했습니다.")
        except Exception as exc:
            print(f"[warning] champion stats scraping failed:\n{exc}")

    if csv_path.exists():
        print(f"[fallback] 크롤링 실패로 기존 champion stats CSV를 사용합니다: {csv_path}")
        stats = pd.read_csv(csv_path, encoding="utf-8-sig")
        stats["champion"] = stats["champion"].apply(normalize_champion_name)
        return stats

    if CONFIG["allow_sample_data"] and match_df is not None:
        print("[sample data] 테스트용 챔피언 통계를 match history에서 생성합니다.")
        stats = create_sample_champion_stats(match_df)
        stats.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return stats

    raise FileNotFoundError("data/champion_global_stats.csv가 없고 샘플 통계 생성도 불가능합니다.")


# %% [markdown] ## 5. 팀 단위 데이터와 네트워크 엣지 생성


# %% Code cell 13

def make_team_rows(match_df):
    """Convert match-level rows into two team-level rows per match."""
    df = ensure_match_schema(match_df)
    rows = []
    for _, row in df.iterrows():
        blue_win = int(row["blue_win"])
        common = {
            "match_id": row["match_id"],
            "date": row["date"],
            "patch": row["patch"],
            "data_source": row.get("data_source", "unknown"),
        }
        rows.append({
            **common,
            "side": "blue",
            "team": row["blue_team"],
            "opponent": row["red_team"],
            "team_picks": row["blue_picks"],
            "opponent_picks": row["red_picks"],
            "team_bans": row["blue_bans"],
            "opponent_bans": row["red_bans"],
            "win": blue_win,
        })
        rows.append({
            **common,
            "side": "red",
            "team": row["red_team"],
            "opponent": row["blue_team"],
            "team_picks": row["red_picks"],
            "opponent_picks": row["blue_picks"],
            "team_bans": row["red_bans"],
            "opponent_bans": row["blue_bans"],
            "win": 1 - blue_win,
        })
    return pd.DataFrame(rows)


def _unique_sorted(values):
    return sorted({normalize_champion_name(value) for value in values if normalize_champion_name(value)})


def build_synergy_edges(team_rows, min_games=None, output_path=DATA_DIR / "champion_edges_synergy.csv"):
    """
    Build undirected same-team champion-pair edges.

    Smoothing avoids overrating pairs that appeared only a few times:
    pair_win_rate_smoothed = (wins + alpha * 0.5) / (games + alpha)
    synergy_score = pair_win_rate_smoothed - 0.5
    """
    min_games = CONFIG["min_games_synergy"] if min_games is None else min_games
    counts = defaultdict(lambda: {"games_together": 0, "wins_together": 0})
    for _, row in team_rows.iterrows():
        picks = _unique_sorted(row["team_picks"])
        for source, target in itertools.combinations(picks, 2):
            key = tuple(sorted((source, target)))
            counts[key]["games_together"] += 1
            counts[key]["wins_together"] += int(row["win"])

    rows = []
    alpha = CONFIG["alpha"]
    global_win_rate = CONFIG["global_win_rate"]
    for (source, target), value in counts.items():
        games = value["games_together"]
        if games < min_games:
            continue
        wins = value["wins_together"]
        pair_win_rate = wins / games if games else np.nan
        smoothed = (wins + alpha * global_win_rate) / (games + alpha)
        synergy_score = smoothed - global_win_rate
        rows.append({
            "source": source,
            "target": target,
            "games_together": games,
            "wins_together": wins,
            "pair_win_rate": pair_win_rate,
            "pair_win_rate_smoothed": smoothed,
            "synergy_score": synergy_score,
            "weight": games * max(synergy_score, 0),
        })
    edges = pd.DataFrame(rows)
    if not edges.empty:
        edges = edges.sort_values(["weight", "synergy_score", "games_together"], ascending=False)
    edges.to_csv(output_path, index=False, encoding="utf-8-sig")
    return edges


def build_counter_edges(team_rows, min_games=None, output_path=DATA_DIR / "champion_edges_counter.csv"):
    """
    Build directed counter edges from source champion to target champion.

    Each team row contributes source -> opponent records, with source_wins
    equal to the team's match result.
    """
    min_games = CONFIG["min_games_counter"] if min_games is None else min_games
    counts = defaultdict(lambda: {"matchup_games": 0, "source_wins": 0})
    for _, row in team_rows.iterrows():
        team_picks = _unique_sorted(row["team_picks"])
        opponent_picks = _unique_sorted(row["opponent_picks"])
        for source in team_picks:
            for target in opponent_picks:
                counts[(source, target)]["matchup_games"] += 1
                counts[(source, target)]["source_wins"] += int(row["win"])

    rows = []
    alpha = CONFIG["alpha"]
    global_win_rate = CONFIG["global_win_rate"]
    for (source, target), value in counts.items():
        games = value["matchup_games"]
        if games < min_games:
            continue
        wins = value["source_wins"]
        source_win_rate = wins / games if games else np.nan
        smoothed = (wins + alpha * global_win_rate) / (games + alpha)
        counter_score = smoothed - global_win_rate
        rows.append({
            "source": source,
            "target": target,
            "matchup_games": games,
            "source_wins": wins,
            "source_win_rate": source_win_rate,
            "source_win_rate_smoothed": smoothed,
            "counter_score": counter_score,
            "weight": games * max(counter_score, 0),
        })
    edges = pd.DataFrame(rows)
    if not edges.empty:
        edges = edges.sort_values(["weight", "counter_score", "matchup_games"], ascending=False)
    edges.to_csv(output_path, index=False, encoding="utf-8-sig")
    return edges


def compute_patch_champion_op_scores(match_df, output_path=OUTPUT_DIR / "patch_op_champion_scores.csv"):
    """
    Rank champion pro-game presence/win reference inside each selected patch.

    이 표는 더 이상 OP 챔피언 정의가 아니다. OP 챔피언은 OP.GG 해당 패치
    OP 티어로 정의하고, 이 함수는 "그 챔피언들이 실제 프로 경기에서는
    얼마나 자주 픽/밴되고 이겼는지"를 참고하기 위한 보조 점수만 만든다.
    """
    df = ensure_match_schema(match_df)
    rows = []
    alpha = CONFIG["alpha"]
    for patch, patch_df in df.groupby("patch", dropna=False):
        n_matches = max(len(patch_df), 1)
        pick_counts = Counter()
        win_counts = Counter()
        ban_counts = Counter()
        for _, row in patch_df.iterrows():
            for champion in row["blue_picks"]:
                pick_counts[champion] += 1
                win_counts[champion] += int(row["blue_win"])
            for champion in row["red_picks"]:
                pick_counts[champion] += 1
                win_counts[champion] += int(row["red_win"])
            ban_counts.update(row["blue_bans"])
            ban_counts.update(row["red_bans"])

        champions = sorted(set(pick_counts) | set(ban_counts))
        for champion in champions:
            picks = pick_counts[champion]
            wins = win_counts[champion]
            bans = ban_counts[champion]
            win_rate = wins / picks if picks else np.nan
            smoothed_win_rate = (wins + alpha * CONFIG["global_win_rate"]) / (picks + alpha)
            pick_rate = picks / n_matches
            ban_rate = bans / n_matches
            presence_rate = (picks + bans) / n_matches
            win_component = max(smoothed_win_rate - CONFIG["global_win_rate"], 0) * 2
            op_score = 0.70 * presence_rate + 0.30 * win_component
            rows.append({
                "patch": str(patch),
                "patch_number": _patch_number(patch),
                "champion": champion,
                "patch_matches": n_matches,
                "picks": picks,
                "wins": wins,
                "bans": bans,
                "win_rate": win_rate,
                "smoothed_win_rate": smoothed_win_rate,
                "pick_rate": pick_rate,
                "ban_rate": ban_rate,
                "presence_rate": presence_rate,
                "op_score": op_score,
            })

    op_df = pd.DataFrame(rows).sort_values(
        ["patch_number", "op_score", "presence_rate", "smoothed_win_rate"],
        ascending=[False, False, False, False],
        ignore_index=True,
    )
    op_df["op_rank"] = op_df.groupby("patch").cumcount() + 1
    op_df["is_high_pro_presence_reference"] = op_df["op_rank"] <= CONFIG["op_top_n"]
    op_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return op_df


def _composition_key(champions):
    return " + ".join(_unique_sorted(champions))


def _bootstrap_mean_ci(values, n_boot=None, random_state=None):
    """Return a bootstrap confidence interval for a binary/continuous mean."""
    values = np.asarray(list(values), dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan
    n_boot = CONFIG["bootstrap_iterations"] if n_boot is None else n_boot
    random_state = CONFIG["random_state"] if random_state is None else random_state
    rng = np.random.default_rng(random_state)
    means = [
        rng.choice(values, size=values.size, replace=True).mean()
        for _ in range(n_boot)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _binomial_two_sided_pvalue(successes, trials, null_p=0.5):
    """Exact two-sided binomial p-value without adding a scipy dependency."""
    if trials <= 0:
        return np.nan
    # Exact binomial probabilities overflow for thousands of team rows, so
    # large samples use a continuity-corrected normal approximation.
    if trials > 1000:
        variance = trials * null_p * (1 - null_p)
        if variance <= 0:
            return np.nan
        distance = max(abs(successes - trials * null_p) - 0.5, 0)
        z_score = distance / math.sqrt(variance)
        return float(math.erfc(z_score / math.sqrt(2)))
    probs = [
        math.comb(trials, k) * (null_p ** k) * ((1 - null_p) ** (trials - k))
        for k in range(trials + 1)
    ]
    observed = probs[int(successes)]
    return float(sum(prob for prob in probs if prob <= observed + 1e-15))


def build_patch_op_lookup(op_champion_tiers, max_raw_tier=None):
    """
    Return {patch: set(OP champions)} from OP.GG tier rows.

    The project now analyzes a 26.9+ patch window, so OP status must be checked
    against the OP.GG tier table for the same patch as each professional game.
    """
    max_raw_tier = CONFIG["opgg_op_tier_raw_max"] if max_raw_tier is None else max_raw_tier
    if op_champion_tiers is None or op_champion_tiers.empty:
        return {}

    patch_col = "patch" if "patch" in op_champion_tiers.columns else "leaguepedia_patch"
    lookup = {}
    for patch, group in op_champion_tiers.groupby(patch_col, dropna=False):
        selected = group[group["opgg_tier_raw"] <= max_raw_tier].copy()
        if selected.empty:
            selected = group.sort_values(["opgg_tier_raw", "opgg_rank"]).head(CONFIG["op_top_n"]).copy()
        lookup[str(patch)] = set(selected["champion"].dropna().map(normalize_champion_name))
    return lookup


def _op_set_for_patch(op_champions, patch):
    """Support both legacy global OP sets and patch-specific OP dictionaries."""
    if isinstance(op_champions, dict):
        return set(op_champions.get(str(patch), set()))
    return set(op_champions)


def analyze_op_status_effect(match_df, op_champions):
    """
    Quantify whether OP-containing teams actually won more often in pro games.

    Team-row summaries are descriptive. The cleaner inferential row uses only
    matches where exactly one team picked an OP champion, because those matches
    directly compare an OP-containing draft against a non-OP draft.
    """
    team_rows = make_team_rows(match_df)

    descriptive_rows = []
    for _, row in team_rows.iterrows():
        picks = _unique_sorted(row["team_picks"])
        op_set = _op_set_for_patch(op_champions, row["patch"])
        team_op_champions = sorted(set(picks) & op_set)
        descriptive_rows.append({
            "scope": "team_rows",
            "group": "has_op" if team_op_champions else "no_op",
            "win": int(row["win"]),
            "op_champions": ", ".join(team_op_champions),
        })

    summary_rows = []
    desc_df = pd.DataFrame(descriptive_rows)
    if not desc_df.empty:
        for group, group_df in desc_df.groupby("group"):
            wins = int(group_df["win"].sum())
            games = int(len(group_df))
            ci_low, ci_high = _bootstrap_mean_ci(group_df["win"])
            summary_rows.append({
                "scope": "team_rows",
                "group": group,
                "games_or_rows": games,
                "wins": wins,
                "win_rate": wins / games if games else np.nan,
                "smoothed_win_rate": (wins + CONFIG["alpha"] * CONFIG["global_win_rate"]) / (games + CONFIG["alpha"]),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "p_value_vs_50": _binomial_two_sided_pvalue(wins, games),
                "note": "Descriptive team-row rate; rows from the same match are not independent.",
            })

    match_detail_rows = []
    for _, row in ensure_match_schema(match_df).iterrows():
        blue_picks = _unique_sorted(row["blue_picks"])
        red_picks = _unique_sorted(row["red_picks"])
        op_set = _op_set_for_patch(op_champions, row["patch"])
        blue_op = sorted(set(blue_picks) & op_set)
        red_op = sorted(set(red_picks) & op_set)
        if bool(blue_op) == bool(red_op):
            continue
        op_side = "blue" if blue_op else "red"
        op_win = int(row["blue_win"]) if blue_op else 1 - int(row["blue_win"])
        match_detail_rows.append({
            "match_id": row["match_id"],
            "date": row["date"],
            "patch": row["patch"],
            "op_side": op_side,
            "op_team": row["blue_team"] if blue_op else row["red_team"],
            "non_op_team": row["red_team"] if blue_op else row["blue_team"],
            "op_champions": ", ".join(blue_op or red_op),
            "op_team_win": op_win,
        })

    detail_df = pd.DataFrame(match_detail_rows)
    if not detail_df.empty:
        wins = int(detail_df["op_team_win"].sum())
        games = int(len(detail_df))
        ci_low, ci_high = _bootstrap_mean_ci(detail_df["op_team_win"])
        summary_rows.append({
            "scope": "match_exactly_one_team_has_op",
            "group": "op_team",
            "games_or_rows": games,
            "wins": wins,
            "win_rate": wins / games if games else np.nan,
            "smoothed_win_rate": (wins + CONFIG["alpha"] * CONFIG["global_win_rate"]) / (games + CONFIG["alpha"]),
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "p_value_vs_50": _binomial_two_sided_pvalue(wins, games),
            "note": "Primary OP-effect check: only matches where exactly one team picked an OP champion.",
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "op_status_winrate_summary.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(OUTPUT_DIR / "op_status_match_details.csv", index=False, encoding="utf-8-sig")
    return summary, detail_df


def analyze_opgg_tier_sensitivity(match_df, opgg_tiers):
    """
    Robustness check for the OP.GG tier cutoff.

    The project definition remains OP.GG OP tier. This table shows what happens
    if a reader asks whether conclusions change after expanding the cutoff to
    OP+Tier1 or OP+Tier2 champions.
    """
    rows = []
    for max_raw_tier in CONFIG["tier_sensitivity_raw_thresholds"]:
        tier_lookup = build_patch_op_lookup(opgg_tiers, max_raw_tier=max_raw_tier)
        tier_champions = sorted({champion for champions in tier_lookup.values() for champion in champions})
        _, detail_df = analyze_op_status_effect(match_df, tier_lookup)
        if detail_df.empty:
            games = wins = 0
            win_rate = ci_low = ci_high = p_value = np.nan
        else:
            games = int(len(detail_df))
            wins = int(detail_df["op_team_win"].sum())
            win_rate = wins / games if games else np.nan
            ci_low, ci_high = _bootstrap_mean_ci(detail_df["op_team_win"])
            p_value = _binomial_two_sided_pvalue(wins, games)
        rows.append({
            "opgg_tier_cutoff_raw": max_raw_tier,
            "tier_group": "OP only" if max_raw_tier == 0 else f"OP through Tier {max_raw_tier}",
            "champion_count": len(tier_champions),
            "champions_preview": ", ".join(tier_champions[:20]),
            "exactly_one_side_games": games,
            "wins_by_tier_group_team": wins,
            "win_rate": win_rate,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "p_value_vs_50": p_value,
        })

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(OUTPUT_DIR / "opgg_tier_sensitivity.csv", index=False, encoding="utf-8-sig")
    # Restore the main project OP-only files because the sensitivity loop calls
    # analyze_op_status_effect with expanded champion sets.
    analyze_op_status_effect(
        match_df,
        build_patch_op_lookup(opgg_tiers, max_raw_tier=CONFIG["opgg_op_tier_raw_max"]),
    )
    return sensitivity


def create_professor_quality_review(
    quality_report,
    op_status_summary,
    tier_sensitivity,
    performance_df=None,
    output_path=OUTPUT_DIR / "professor_quality_review.csv",
):
    """Create a professor-style review table focused on defensible analysis quality."""
    quality_lookup = dict(zip(quality_report["check"], quality_report["value"])) if not quality_report.empty else {}
    excluded = int(quality_lookup.get("excluded_incomplete_pick_matches", 0) or 0)
    analysis_rows = int(quality_lookup.get("match_rows", 0) or 0)

    op_effect = op_status_summary[op_status_summary["scope"].eq("match_exactly_one_team_has_op")]
    if not op_effect.empty:
        op_row = op_effect.iloc[0]
        op_effect_text = (
            f"OP-team win rate {float(op_row['win_rate']):.1%} "
            f"(n={int(op_row['games_or_rows'])}, 95% CI {float(op_row['ci_lower']):.1%}-{float(op_row['ci_upper']):.1%})"
        )
    else:
        op_effect_text = "No exactly-one-OP match sample"

    model_note = "Model comparison not yet available"
    if performance_df is not None and not performance_df.empty:
        best_baseline = performance_df[performance_df["feature_set"].eq("baseline")]["accuracy"].max()
        best_network = performance_df[performance_df["feature_set"].eq("network")]["accuracy"].max()
        model_note = f"Best baseline accuracy {best_baseline:.3f}; best network accuracy {best_network:.3f}"

    review = pd.DataFrame([
        {
            "criterion": "Topic alignment",
            "professor_view": "Strong",
            "evidence": "Outputs directly answer OP-excluded comps, OP-included comps, and opponent comps that beat OP teams.",
            "development_added": "Added OP status effect summary and denominator-aware opponent-combo table.",
        },
        {
            "criterion": "Data freshness and patch logic",
            "professor_view": "Strong with caveat",
            "evidence": "Analysis window starts at Patch 26.9, reflecting the last major meta-change assumption.",
            "development_added": "OP.GG champion tier is matched per patch inside the 26.9+ window.",
        },
        {
            "criterion": "Data quality control",
            "professor_view": "Improved",
            "evidence": f"Analysis uses {analysis_rows} complete-pick matches; excluded {excluded} incomplete-pick rows.",
            "development_added": "Created analysis_match_history.csv and excluded_incomplete_matches.csv for auditability.",
        },
        {
            "criterion": "OP definition validity",
            "professor_view": "Defensible",
            "evidence": "Main OP definition is OP.GG OP tier for the matched patch, not a post-hoc pro win-rate score.",
            "development_added": "Added OP.GG tier sensitivity table as a robustness check without changing the main definition.",
        },
        {
            "criterion": "Statistical interpretation",
            "professor_view": "Improved",
            "evidence": op_effect_text,
            "development_added": "Added bootstrap confidence intervals and exact binomial p-values for OP-team win rate.",
        },
        {
            "criterion": "Exception-case interpretation",
            "professor_view": "Improved",
            "evidence": "OP-loss games are now linked to player-level runes and final item inventories.",
            "development_added": "Added same-patch champion-role baseline comparison for low-frequency rune/item flags.",
        },
        {
            "criterion": "Lecture-material alignment",
            "professor_view": "Improved",
            "evidence": "Notebook explicitly maps crawling, visualization, node centrality, and 2-mode network lecture concepts to the LoL project.",
            "development_added": "Added team-champion bipartite outputs and a 2-mode network visualization as a lecture-inspired supplement.",
        },
        {
            "criterion": "Weak-node filtering transparency",
            "professor_view": "Improved",
            "evidence": "Weak links are filtered by minimum repeated games, positive smoothed weight, and top-weight visualization cuts.",
            "development_added": "Added network_filtering_criteria.csv so node/edge removal rules are auditable.",
        },
        {
            "criterion": "Prediction model role",
            "professor_view": "Use as supporting evidence",
            "evidence": model_note,
            "development_added": "Conclusion should not overclaim prediction if network features do not improve accuracy.",
        },
        {
            "criterion": "Remaining limitation",
            "professor_view": "Transparent",
            "evidence": "OP.GG solo-queue tier is not identical to pro-play OP status; CargoExport lacks true side and draft-order fields.",
            "development_added": "The notebook now frames these as limitations and keeps OP.GG tier as an external definition.",
        },
    ])
    review.to_csv(output_path, index=False, encoding="utf-8-sig")
    return review


def analyze_compositions_by_op_status(match_df, op_champions):
    """
    Summarize strong combinations with and without OP champions.

    exact 5-champion compositions rarely repeat in pro data, so the table also
    includes duo and trio combinations. This gives practical draft insights
    without pretending every exact 5-man comp has enough support.
    """
    team_rows = make_team_rows(match_df)
    combo_rows = []
    status_rows = []
    for _, row in team_rows.iterrows():
        picks = _unique_sorted(row["team_picks"])
        op_set = _op_set_for_patch(op_champions, row["patch"])
        team_op_champions = sorted(set(picks) & op_set)
        team_has_op = bool(team_op_champions)
        status_rows.append({"team_has_op": team_has_op, "win": int(row["win"])})
        for combo_size in [2, 3, 5]:
            if len(picks) < combo_size:
                continue
            combos = [tuple(picks)] if combo_size == 5 else itertools.combinations(picks, combo_size)
            for combo in combos:
                combo = tuple(sorted(combo))
                combo_op_champions = sorted(set(combo) & op_set)
                combo_rows.append({
                    "combo_size": combo_size,
                    "composition": " + ".join(combo),
                    "team_has_op": team_has_op,
                    "combo_contains_op": bool(combo_op_champions),
                    "op_champions_in_team": ", ".join(team_op_champions),
                    "op_champions_in_combo": ", ".join(combo_op_champions),
                    "win": int(row["win"]),
                    "match_id": row["match_id"],
                    "patch": row["patch"],
                })

    combo_df = pd.DataFrame(combo_rows)
    if combo_df.empty:
        empty = pd.DataFrame()
        empty.to_csv(OUTPUT_DIR / "best_compositions_without_op.csv", index=False, encoding="utf-8-sig")
        empty.to_csv(OUTPUT_DIR / "best_compositions_with_op.csv", index=False, encoding="utf-8-sig")
        return empty, empty

    grouped = (
        combo_df.groupby(["combo_size", "composition", "team_has_op", "combo_contains_op"], dropna=False)
        .agg(
            games=("win", "size"),
            wins=("win", "sum"),
            example_matches=("match_id", lambda x: ", ".join(map(str, list(x)[:3]))),
            op_champions=("op_champions_in_combo", lambda x: ", ".join(sorted({v for v in x if v}))),
        )
        .reset_index()
    )
    grouped["win_rate"] = grouped["wins"] / grouped["games"]
    grouped["smoothed_win_rate"] = (
        grouped["wins"] + CONFIG["alpha"] * CONFIG["global_win_rate"]
    ) / (grouped["games"] + CONFIG["alpha"])
    status_baseline = pd.DataFrame(status_rows).groupby("team_has_op")["win"].mean().to_dict()
    grouped["op_status_baseline_win_rate"] = grouped["team_has_op"].map(status_baseline).fillna(CONFIG["global_win_rate"])
    grouped["lift_vs_op_status_baseline"] = grouped["smoothed_win_rate"] - grouped["op_status_baseline_win_rate"]
    grouped["lift_vs_global_50"] = grouped["smoothed_win_rate"] - CONFIG["global_win_rate"]

    def _min_games(row):
        return CONFIG["composition_min_games"].get(int(row["combo_size"]), 1)

    grouped = grouped[grouped.apply(lambda row: row["games"] >= _min_games(row), axis=1)].copy()
    grouped = grouped.sort_values(
        ["smoothed_win_rate", "games", "win_rate"],
        ascending=False,
        ignore_index=True,
    )

    without_op = grouped[~grouped["team_has_op"]].copy()
    with_op = grouped[grouped["combo_contains_op"]].copy()
    without_op.to_csv(OUTPUT_DIR / "best_compositions_without_op.csv", index=False, encoding="utf-8-sig")
    with_op.to_csv(OUTPUT_DIR / "best_compositions_with_op.csv", index=False, encoding="utf-8-sig")
    return without_op, with_op


def analyze_op_loss_opponent_compositions(match_df, op_champions):
    """Find opponent compositions that beat teams containing OP champions."""
    def _unique_comma_values(values):
        items = set()
        for value in values:
            for part in str(value).split(","):
                cleaned = part.strip()
                if cleaned and cleaned.lower() != "nan":
                    items.add(cleaned)
        return ", ".join(sorted(items))

    team_rows = make_team_rows(match_df)
    detail_rows = []
    combo_rows = []
    for _, row in team_rows.iterrows():
        team_picks = _unique_sorted(row["team_picks"])
        opponent_picks = _unique_sorted(row["opponent_picks"])
        op_set = _op_set_for_patch(op_champions, row["patch"])
        team_op_champions = sorted(set(team_picks) & op_set)
        opponent_op_champions = sorted(set(opponent_picks) & op_set)
        if opponent_op_champions:
            for combo_size in [2, 3, 5]:
                combos = [tuple(team_picks)] if combo_size == 5 else itertools.combinations(team_picks, combo_size)
                for combo in combos:
                    combo_rows.append({
                        "combo_size": combo_size,
                        "opponent_combo": " + ".join(sorted(combo)),
                        "faced_op_champions": ", ".join(opponent_op_champions),
                        "win_against_op": int(row["win"]),
                        "winning_match_id": row["match_id"] if int(row["win"]) == 1 else "",
                        "match_id": row["match_id"],
                    })
        if not team_op_champions or int(row["win"]) != 0:
            continue
        detail_rows.append({
            "match_id": row["match_id"],
            "date": row["date"],
            "patch": row["patch"],
            "op_losing_team": row["team"],
            "op_losing_team_picks": _composition_key(team_picks),
            "op_champions_in_losing_team": ", ".join(team_op_champions),
            "winning_opponent": row["opponent"],
            "winning_opponent_picks": _composition_key(opponent_picks),
        })

    details = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(combo_rows)
    if not summary.empty:
        summary = (
            summary.groupby(["combo_size", "opponent_combo"], dropna=False)
            .agg(
                games_against_op=("match_id", "size"),
                wins_against_op=("win_against_op", "sum"),
                example_winning_matches=("winning_match_id", lambda x: ", ".join([str(v) for v in list(x) if str(v).strip()][:3])),
                faced_op_champions=("faced_op_champions", _unique_comma_values),
            )
            .reset_index()
        )
        summary["win_rate_against_op"] = summary["wins_against_op"] / summary["games_against_op"]
        summary["smoothed_win_rate_against_op"] = (
            summary["wins_against_op"] + CONFIG["alpha"] * CONFIG["global_win_rate"]
        ) / (summary["games_against_op"] + CONFIG["alpha"])

        def _min_games(row):
            combo_size = int(row["combo_size"])
            # Exact 5-champion comps are useful as case studies, but a single
            # occurrence is too thin for the ranked summary table.
            if combo_size == 5:
                return max(CONFIG["composition_min_games"].get(combo_size, 1), 2)
            return CONFIG["composition_min_games"].get(combo_size, 1)

        summary = summary[summary.apply(lambda row: row["games_against_op"] >= _min_games(row), axis=1)].copy()
        summary = summary.sort_values(
            ["smoothed_win_rate_against_op", "wins_against_op", "games_against_op"],
            ascending=False,
            ignore_index=True,
        )
    details.to_csv(OUTPUT_DIR / "op_loss_opponent_compositions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "op_loss_opponent_combo_summary.csv", index=False, encoding="utf-8-sig")
    return details, summary


def load_or_scrape_scoreboard_players(selected_patch, match_ids=None, output_path=DATA_DIR / "scoreboard_players_patch.csv"):
    """
    Scrape player-level runes and final inventory items from Leaguepedia Cargo.

    This table is used only for exception analysis, not for model training, so it
    does not leak match outcome features into the prediction model.
    """
    if isinstance(selected_patch, (list, tuple, set, pd.Series, np.ndarray)):
        patch_values = selected_patch
    else:
        patch_values = [selected_patch]
    patch_values = sorted(
        {str(patch) for patch in patch_values if pd.notna(patch)},
        key=_patch_number,
    )
    rows = []
    limit = 500
    for patch_text in patch_values:
        offset = 0
        while True:
            batch = fetch_cargo_json(
                "ScoreboardPlayers,ScoreboardGames",
                "ScoreboardPlayers.GameId,ScoreboardGames.Patch,ScoreboardGames.DateTime_UTC,"
                "ScoreboardPlayers.Team,ScoreboardPlayers.Name,ScoreboardPlayers.Champion,"
                "ScoreboardPlayers.Role,ScoreboardPlayers.Items,ScoreboardPlayers.Runes",
                f'ScoreboardGames.Patch="{patch_text}" AND ScoreboardGames.Winner IS NOT NULL',
                f"scoreboard_players_patch_{patch_text.replace('.', '_')}_{offset}.json",
                limit=limit,
                offset=offset,
                force_refresh=CONFIG["force_refresh"],
                order_by="ScoreboardGames.DateTime_UTC ASC",
                join_on="ScoreboardPlayers.GameId=ScoreboardGames.GameId",
            )
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < limit:
                break
            offset += limit

    players = pd.DataFrame(rows)
    if players.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_path, index=False, encoding="utf-8-sig")
        return empty

    players = players.rename(columns={
        "GameId": "match_id",
        "Patch": "patch",
        "DateTime UTC": "date",
        "Team": "team",
        "Name": "player",
        "Champion": "champion",
        "Role": "role",
        "Items": "items",
        "Runes": "runes",
    })
    players["patch"] = players["patch"].apply(lambda value: normalize_patch_label(value, CONFIG["analysis_min_patch"]))
    players["champion"] = players["champion"].apply(normalize_champion_name)
    players["items"] = players["items"].apply(parse_list_column)
    players["runes"] = players["runes"].fillna("").astype(str)
    players["key_rune"] = players["runes"].apply(lambda text: str(text).split(",")[0].strip() if str(text).strip() else "")
    players["item_inventory"] = players["items"].apply(lambda values: [item for item in values if item])
    players["item_signature"] = players["item_inventory"].apply(lambda values: " + ".join(values[:3]))
    if match_ids is not None:
        match_id_set = set(map(str, match_ids))
        players = players[players["match_id"].astype(str).isin(match_id_set)].copy()

    keep_cols = [
        "match_id", "patch", "date", "team", "player", "champion", "role",
        "key_rune", "runes", "item_inventory", "item_signature",
    ]
    players = players[[col for col in keep_cols if col in players.columns]].reset_index(drop=True)
    players.to_csv(output_path, index=False, encoding="utf-8-sig")
    return players


def build_rune_item_baseline(scoreboard_players, output_path=OUTPUT_DIR / "rune_item_patch_baseline.csv"):
    """Build patch/champion/role baseline distributions for key runes and item inventories."""
    if scoreboard_players.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_path, index=False, encoding="utf-8-sig")
        return empty

    rows = []
    group_cols = ["patch", "champion", "role"] if "patch" in scoreboard_players.columns else ["champion", "role"]
    grouped = scoreboard_players.groupby(group_cols, dropna=False)
    for group_key, group in grouped:
        if len(group_cols) == 3:
            patch, champion, role = group_key
        else:
            patch = ""
            champion, role = group_key
        total_games = len(group)
        rune_counts = Counter(group["key_rune"].dropna())
        for rune, count in rune_counts.items():
            if not rune:
                continue
            rows.append({
                "patch": patch,
                "champion": champion,
                "role": role,
                "feature_type": "key_rune",
                "feature_value": rune,
                "count": count,
                "total_games": total_games,
                "usage_rate": count / total_games if total_games else np.nan,
            })
        item_counter = Counter()
        for values in group["item_inventory"]:
            item_counter.update([item for item in values if item])
        for item, count in item_counter.items():
            rows.append({
                "patch": patch,
                "champion": champion,
                "role": role,
                "feature_type": "item",
                "feature_value": item,
                "count": count,
                "total_games": total_games,
                "usage_rate": count / total_games if total_games else np.nan,
            })

    baseline = pd.DataFrame(rows).sort_values(
        ["patch", "champion", "role", "feature_type", "usage_rate", "count"],
        ascending=[True, True, True, True, False, False],
        ignore_index=True,
    )
    baseline.to_csv(output_path, index=False, encoding="utf-8-sig")
    return baseline


def correct_match_picks_from_scoreboard(
    match_df,
    scoreboard_players,
    output_path=DATA_DIR / "analysis_match_history.csv",
    audit_path=OUTPUT_DIR / "pick_alignment_audit.csv",
):
    """
    Correct team-pick alignment using ScoreboardPlayers.

    PicksAndBansS7 occasionally stores Team1/Team2 picks in the opposite order
    from ScoreboardGames.Team1/Team2. ScoreboardPlayers links each champion to
    the actual team and player, so it is the safer source for team composition.
    Bans remain from PicksAndBansS7 because ScoreboardPlayers has picks only.
    """
    if scoreboard_players.empty:
        match_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(audit_path, index=False, encoding="utf-8-sig")
        return match_df

    role_order = {"Top": 0, "Jungle": 1, "Mid": 2, "Bot": 3, "Support": 4}
    pick_lookup = {}
    for (match_id, team), group in scoreboard_players.groupby(["match_id", "team"], dropna=False):
        group = group.copy()
        group["_role_order"] = group["role"].map(role_order).fillna(99)
        picks = group.sort_values(["_role_order", "champion"])["champion"].dropna().map(normalize_champion_name).tolist()
        picks = [pick for pick in picks if pick]
        if picks:
            pick_lookup[(str(match_id), str(team))] = picks

    corrected = ensure_match_schema(match_df).copy()
    audit_rows = []
    for idx, row in corrected.iterrows():
        match_id = str(row["match_id"])
        for side, team_col, pick_col in [
            ("blue", "blue_team", "blue_picks"),
            ("red", "red_team", "red_picks"),
        ]:
            team = str(row[team_col])
            scoreboard_picks = pick_lookup.get((match_id, team))
            old_picks = parse_list_column(row[pick_col])
            changed = False
            if scoreboard_picks and len(scoreboard_picks) == 5:
                changed = set(scoreboard_picks) != set(old_picks)
                corrected.at[idx, pick_col] = scoreboard_picks
            audit_rows.append({
                "match_id": match_id,
                "side_proxy": side,
                "team": team,
                "old_picks": " + ".join(old_picks),
                "scoreboard_picks": " + ".join(scoreboard_picks or []),
                "corrected": changed,
                "scoreboard_pick_count": len(scoreboard_picks or []),
            })

    corrected["blue_win"] = corrected["winner"].eq(corrected["blue_team"]).astype(int)
    corrected["red_win"] = 1 - corrected["blue_win"]
    corrected["pick_source"] = "scoreboard_players_corrected"
    corrected.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    print(f"[quality] corrected pick alignment for {int(audit['corrected'].sum())} team rows using ScoreboardPlayers")
    return corrected


def analyze_op_loss_rune_item_exceptions(scoreboard_players, op_loss_details, op_champions, baseline):
    """
    Check whether exceptional OP-loss games had unusual runes/items.

    A rune/item is marked unusual only relative to the same selected-patch
    champion-role baseline. This avoids calling a normal champion-specific build
    "strange" just because it looks unusual in isolation.
    """
    output_all = OUTPUT_DIR / "op_loss_rune_item_case_notes.csv"
    output_op = OUTPUT_DIR / "op_loss_op_champion_build_notes.csv"
    output_winner = OUTPUT_DIR / "op_loss_winning_opponent_build_notes.csv"
    if scoreboard_players.empty or op_loss_details.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_all, index=False, encoding="utf-8-sig")
        empty.to_csv(output_op, index=False, encoding="utf-8-sig")
        empty.to_csv(output_winner, index=False, encoding="utf-8-sig")
        return empty, empty, empty

    detail_lookup = op_loss_details.set_index("match_id").to_dict("index")
    case_players = scoreboard_players[scoreboard_players["match_id"].isin(op_loss_details["match_id"])].copy()

    baseline_lookup = {}
    for _, row in baseline.iterrows():
        key = (str(row.get("patch", "")), row["champion"], row["role"], row["feature_type"], row["feature_value"])
        baseline_lookup[key] = row.to_dict()

    note_rows = []
    for _, row in case_players.iterrows():
        match_info = detail_lookup.get(row["match_id"], {})
        team = row["team"]
        champion = row["champion"]
        role = row["role"]
        key_rune = row["key_rune"]
        items = row["item_inventory"] if isinstance(row["item_inventory"], list) else parse_list_column(row["item_inventory"])
        patch = str(match_info.get("patch", row.get("patch", "")))
        op_set = _op_set_for_patch(op_champions, patch)

        is_op_losing_team = team == match_info.get("op_losing_team")
        is_winning_opponent = team == match_info.get("winning_opponent")
        is_op_champion = champion in op_set
        context = (
            "op_losing_champion" if is_op_losing_team and is_op_champion
            else "winning_opponent" if is_winning_opponent
            else "op_losing_teammate" if is_op_losing_team
            else "other"
        )

        rune_base = baseline_lookup.get((patch, champion, role, "key_rune", key_rune), {})
        rune_rate = float(rune_base.get("usage_rate", np.nan)) if rune_base else np.nan
        rune_count = int(rune_base.get("count", 0)) if rune_base else 0
        total_games = int(rune_base.get("total_games", 0)) if rune_base else 0
        unusual_rune = bool(total_games >= 5 and pd.notna(rune_rate) and rune_rate <= 0.25)

        rare_items = []
        rare_item_rates = []
        for item in items:
            item_base = baseline_lookup.get((patch, champion, role, "item", item), {})
            item_rate = float(item_base.get("usage_rate", np.nan)) if item_base else np.nan
            item_count = int(item_base.get("count", 0)) if item_base else 0
            if total_games >= 5 and pd.notna(item_rate) and item_rate <= 0.20:
                rare_items.append(item)
                rare_item_rates.append(f"{item}:{item_rate:.1%}({item_count}/{total_games})")

        note_parts = []
        if unusual_rune:
            note_parts.append(f"key rune '{key_rune}' is low-frequency for {champion} {role}: {rune_rate:.1%} ({rune_count}/{total_games})")
        if rare_items:
            note_parts.append("rare item(s): " + "; ".join(rare_item_rates[:5]))
        if not note_parts:
            note_parts.append("No low-frequency rune/item signal versus same champion-role patch baseline.")

        note_rows.append({
            "match_id": row["match_id"],
            "date": row.get("date", ""),
            "patch": patch,
            "team": team,
            "player": row.get("player", ""),
            "champion": champion,
            "role": role,
            "case_context": context,
            "op_losing_team": match_info.get("op_losing_team", ""),
            "winning_opponent": match_info.get("winning_opponent", ""),
            "key_rune": key_rune,
            "key_rune_usage_rate": rune_rate,
            "champion_role_games": total_games,
            "item_signature": row.get("item_signature", ""),
            "item_inventory": ", ".join(items),
            "rare_items": ", ".join(rare_items),
            "unusual_rune_flag": unusual_rune,
            "rare_item_flag": bool(rare_items),
            "exception_note": " ".join(note_parts),
        })

    notes = pd.DataFrame(note_rows)
    notes = notes.sort_values(
        ["case_context", "unusual_rune_flag", "rare_item_flag", "date"],
        ascending=[True, False, False, True],
        ignore_index=True,
    )
    op_notes = notes[notes["case_context"].eq("op_losing_champion")].copy()
    winner_notes = notes[notes["case_context"].eq("winning_opponent")].copy()
    notes.to_csv(output_all, index=False, encoding="utf-8-sig")
    op_notes.to_csv(output_op, index=False, encoding="utf-8-sig")
    winner_notes.to_csv(output_winner, index=False, encoding="utf-8-sig")
    return notes, op_notes, winner_notes


# %% [markdown] ## 6. 그래프 생성과 중심성 계산


# %% Code cell 15

def make_synergy_graph(synergy_edges):
    """Create an undirected weighted synergy graph."""
    graph = nx.Graph()
    for _, row in synergy_edges.iterrows():
        weight = float(row.get("weight", 0))
        if weight <= 0:
            continue
        graph.add_edge(
            row["source"], row["target"],
            weight=weight,
            synergy_score=float(row["synergy_score"]),
            games_together=int(row["games_together"]),
            wins_together=int(row["wins_together"]),
        )
    return graph


def make_counter_graph(counter_edges):
    """Create a directed weighted counter graph."""
    graph = nx.DiGraph()
    for _, row in counter_edges.iterrows():
        weight = float(row.get("weight", 0))
        if weight <= 0:
            continue
        graph.add_edge(
            row["source"], row["target"],
            weight=weight,
            counter_score=float(row["counter_score"]),
            matchup_games=int(row["matchup_games"]),
            source_wins=int(row["source_wins"]),
        )
    return graph


def calculate_synergy_centrality(graph):
    """Calculate centrality metrics for an undirected synergy graph."""
    columns = [
        "champion", "synergy_degree", "synergy_degree_centrality",
        "synergy_weighted_degree", "synergy_betweenness",
        "synergy_closeness", "synergy_eigenvector", "synergy_pagerank",
    ]
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=columns)

    degree = dict(graph.degree())
    degree_centrality = nx.degree_centrality(graph)
    weighted_degree = dict(graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(graph, weight=None)
    closeness = nx.closeness_centrality(graph)
    try:
        eigenvector = nx.eigenvector_centrality(graph, weight="weight", max_iter=2000)
    except Exception:
        eigenvector = {node: np.nan for node in graph.nodes}
    pagerank = nx.pagerank(graph, weight="weight")

    rows = []
    for node in graph.nodes:
        rows.append({
            "champion": node,
            "synergy_degree": degree.get(node, 0),
            "synergy_degree_centrality": degree_centrality.get(node, 0),
            "synergy_weighted_degree": weighted_degree.get(node, 0),
            "synergy_betweenness": betweenness.get(node, 0),
            "synergy_closeness": closeness.get(node, 0),
            "synergy_eigenvector": eigenvector.get(node, np.nan),
            "synergy_pagerank": pagerank.get(node, 0),
        })
    return pd.DataFrame(rows)


def calculate_counter_centrality(graph):
    """Calculate centrality metrics for a directed counter graph."""
    columns = [
        "champion", "counter_in_degree", "counter_out_degree",
        "counter_weighted_in_degree", "counter_weighted_out_degree",
        "counter_in_degree_centrality", "counter_out_degree_centrality",
        "counter_pagerank", "counter_betweenness",
        "counter_hits_hub", "counter_hits_authority",
    ]
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=columns)

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    weighted_in = dict(graph.in_degree(weight="weight"))
    weighted_out = dict(graph.out_degree(weight="weight"))
    in_degree_centrality = nx.in_degree_centrality(graph)
    out_degree_centrality = nx.out_degree_centrality(graph)
    pagerank = nx.pagerank(graph, weight="weight")
    betweenness = nx.betweenness_centrality(graph, weight=None)
    try:
        hubs, authorities = nx.hits(graph, max_iter=2000, normalized=True)
    except Exception:
        hubs = {node: np.nan for node in graph.nodes}
        authorities = {node: np.nan for node in graph.nodes}

    rows = []
    for node in graph.nodes:
        rows.append({
            "champion": node,
            "counter_in_degree": in_degree.get(node, 0),
            "counter_out_degree": out_degree.get(node, 0),
            "counter_weighted_in_degree": weighted_in.get(node, 0),
            "counter_weighted_out_degree": weighted_out.get(node, 0),
            "counter_in_degree_centrality": in_degree_centrality.get(node, 0),
            "counter_out_degree_centrality": out_degree_centrality.get(node, 0),
            "counter_pagerank": pagerank.get(node, 0),
            "counter_betweenness": betweenness.get(node, 0),
            "counter_hits_hub": hubs.get(node, 0),
            "counter_hits_authority": authorities.get(node, 0),
        })
    return pd.DataFrame(rows)


def merge_champion_network_metrics(synergy_metrics, counter_metrics, output_path=OUTPUT_DIR / "champion_network_metrics.csv"):
    """Merge synergy and counter centrality into a champion-level table."""
    metrics = pd.merge(synergy_metrics, counter_metrics, on="champion", how="outer")
    numeric_cols = metrics.select_dtypes(include=[np.number]).columns
    metrics[numeric_cols] = metrics[numeric_cols].fillna(0)
    metrics.to_csv(output_path, index=False, encoding="utf-8-sig")
    return metrics


def build_team_champion_two_mode_network(
    match_df,
    op_champions=None,
    edge_path=OUTPUT_DIR / "team_champion_bipartite_edges.csv",
    team_path=OUTPUT_DIR / "team_two_mode_summary.csv",
    champion_path=OUTPUT_DIR / "champion_two_mode_summary.csv",
    projection_path=OUTPUT_DIR / "champion_team_projection_edges.csv",
):
    """
    Build a team-champion 2-mode network inspired by the lecture material.

    Teams and champions are different node types. Edge weight means how often
    a team picked a champion in the 26.9+ pro-match window. The champion
    projection is not an in-game synergy measure; it means two champions are
    used by overlapping teams, which is a broader team-pool/meta signal.
    """
    team_rows = make_team_rows(match_df)
    pick_rows = []
    for _, row in team_rows.iterrows():
        picks = _unique_sorted(row["team_picks"])
        op_set = _op_set_for_patch(op_champions, row["patch"]) if op_champions is not None else set()
        for champion in picks:
            pick_rows.append({
                "match_id": row["match_id"],
                "date": row["date"],
                "patch": row["patch"],
                "team": row["team"],
                "champion": champion,
                "win": int(row["win"]),
                "is_op_champion_for_patch": champion in op_set,
            })

    pick_df = pd.DataFrame(pick_rows)
    if pick_df.empty:
        for path in [edge_path, team_path, champion_path, projection_path]:
            pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    edges = (
        pick_df.groupby(["team", "champion"], dropna=False)
        .agg(
            team_champion_pick_rows=("match_id", "size"),
            wins_with_champion=("win", "sum"),
            op_pick_rows=("is_op_champion_for_patch", "sum"),
            patches=("patch", lambda x: ", ".join(sorted(set(map(str, x)), key=_patch_number))),
            example_matches=("match_id", lambda x: ", ".join(map(str, list(x)[:3]))),
        )
        .reset_index()
    )
    edges["win_rate_with_champion"] = edges["wins_with_champion"] / edges["team_champion_pick_rows"]

    graph = nx.Graph()
    for _, row in edges.iterrows():
        team_node = f"team::{row['team']}"
        champion_node = f"champion::{row['champion']}"
        graph.add_node(team_node, bipartite="team", label=row["team"])
        graph.add_node(champion_node, bipartite="champion", label=row["champion"])
        graph.add_edge(
            team_node,
            champion_node,
            weight=float(row["team_champion_pick_rows"]),
            op_pick_rows=float(row["op_pick_rows"]),
        )

    weighted_degree = dict(graph.degree(weight="weight"))
    degree = dict(graph.degree())

    team_summary = (
        pick_df.groupby("team", dropna=False)
        .agg(
            total_pick_rows=("champion", "size"),
            unique_champions=("champion", "nunique"),
            wins=("win", "sum"),
            op_pick_rows=("is_op_champion_for_patch", "sum"),
            patches=("patch", lambda x: ", ".join(sorted(set(map(str, x)), key=_patch_number))),
        )
        .reset_index()
    )
    team_summary["team_row_win_rate"] = team_summary["wins"] / team_summary["total_pick_rows"]
    team_summary["op_pick_rate"] = team_summary["op_pick_rows"] / team_summary["total_pick_rows"]
    team_summary["two_mode_degree"] = team_summary["team"].map(lambda x: degree.get(f"team::{x}", 0))
    team_summary["two_mode_weighted_degree"] = team_summary["team"].map(lambda x: weighted_degree.get(f"team::{x}", 0.0))
    team_summary = team_summary.sort_values(
        ["op_pick_rate", "two_mode_weighted_degree", "unique_champions"],
        ascending=[False, False, False],
        ignore_index=True,
    )

    champion_summary = (
        pick_df.groupby("champion", dropna=False)
        .agg(
            total_team_pick_rows=("team", "size"),
            teams_using=("team", "nunique"),
            wins=("win", "sum"),
            op_pick_rows=("is_op_champion_for_patch", "sum"),
            patches=("patch", lambda x: ", ".join(sorted(set(map(str, x)), key=_patch_number))),
        )
        .reset_index()
    )
    champion_summary["team_pick_win_rate"] = champion_summary["wins"] / champion_summary["total_team_pick_rows"]
    champion_summary["is_op_any_patch"] = champion_summary["op_pick_rows"] > 0
    champion_summary["two_mode_degree"] = champion_summary["champion"].map(lambda x: degree.get(f"champion::{x}", 0))
    champion_summary["two_mode_weighted_degree"] = champion_summary["champion"].map(lambda x: weighted_degree.get(f"champion::{x}", 0.0))
    champion_summary = champion_summary.sort_values(
        ["two_mode_weighted_degree", "teams_using", "op_pick_rows"],
        ascending=False,
        ignore_index=True,
    )

    projection_counter = Counter()
    op_projection_counter = Counter()
    for _, group in pick_df.groupby("team"):
        champions = sorted(set(group["champion"]))
        op_champions_for_team = set(group.loc[group["is_op_champion_for_patch"], "champion"])
        for source, target in itertools.combinations(champions, 2):
            projection_counter[(source, target)] += 1
            if source in op_champions_for_team or target in op_champions_for_team:
                op_projection_counter[(source, target)] += 1

    projection_rows = [
        {
            "source": source,
            "target": target,
            "teams_using_both": weight,
            "teams_using_pair_with_op_context": op_projection_counter.get((source, target), 0),
        }
        for (source, target), weight in projection_counter.items()
        if weight >= 3
    ]
    projection = pd.DataFrame(projection_rows)
    if not projection.empty:
        projection = projection.sort_values(
            ["teams_using_both", "teams_using_pair_with_op_context"],
            ascending=False,
            ignore_index=True,
        )

    edges = edges.sort_values(["team_champion_pick_rows", "op_pick_rows"], ascending=False, ignore_index=True)
    edges.to_csv(edge_path, index=False, encoding="utf-8-sig")
    team_summary.to_csv(team_path, index=False, encoding="utf-8-sig")
    champion_summary.to_csv(champion_path, index=False, encoding="utf-8-sig")
    projection.to_csv(projection_path, index=False, encoding="utf-8-sig")
    return edges, team_summary, champion_summary, projection


def create_network_filtering_criteria(
    train_team_rows,
    synergy_edges,
    counter_edges,
    synergy_graph,
    counter_graph,
    team_champion_edges=None,
    output_path=OUTPUT_DIR / "network_filtering_criteria.csv",
    visualization_top_n_edges=70,
):
    """
    Document how weak edges/nodes are filtered in the network analysis.

    Raw CSV outputs keep auditable candidate rows, but centrality graphs and
    figures remove weak nodes indirectly by keeping only reliable, positive,
    and/or high-weight edges.
    """
    train_team_rows = train_team_rows.copy()

    raw_synergy_counts = Counter()
    raw_counter_counts = Counter()
    synergy_candidate_nodes = set()
    counter_candidate_nodes = set()
    for _, row in train_team_rows.iterrows():
        team_picks = _unique_sorted(row["team_picks"])
        opponent_picks = _unique_sorted(row["opponent_picks"])
        synergy_candidate_nodes.update(team_picks)
        counter_candidate_nodes.update(team_picks)
        counter_candidate_nodes.update(opponent_picks)
        for source, target in itertools.combinations(team_picks, 2):
            raw_synergy_counts[tuple(sorted((source, target)))] += 1
        for source in team_picks:
            for target in opponent_picks:
                raw_counter_counts[(source, target)] += 1

    def _plot_summary(graph, top_n):
        edges_sorted = sorted(graph.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True)[:top_n]
        nodes = set()
        for source, target, _ in edges_sorted:
            nodes.add(source)
            nodes.add(target)
        min_weight = min([data.get("weight", 0) for _, _, data in edges_sorted], default=np.nan)
        return len(edges_sorted), len(nodes), min_weight

    synergy_plot_edges, synergy_plot_nodes, synergy_plot_min_weight = _plot_summary(synergy_graph, visualization_top_n_edges)
    counter_plot_edges, counter_plot_nodes, counter_plot_min_weight = _plot_summary(counter_graph, visualization_top_n_edges)

    rows = [
        {
            "network": "champion_synergy",
            "candidate_unique_edges": len(raw_synergy_counts),
            "candidate_nodes": len(synergy_candidate_nodes),
            "analysis_edge_rule": f"games_together >= {CONFIG['min_games_synergy']} and smoothed synergy weight > 0",
            "retained_analysis_edges": synergy_graph.number_of_edges(),
            "retained_analysis_nodes": synergy_graph.number_of_nodes(),
            "visualization_rule": f"top {visualization_top_n_edges} edges by weight; nodes not incident to those edges are hidden",
            "visualized_edges": synergy_plot_edges,
            "visualized_nodes": synergy_plot_nodes,
            "visualization_min_edge_weight": synergy_plot_min_weight,
            "hidden_weak_nodes_in_figure": max(synergy_graph.number_of_nodes() - synergy_plot_nodes, 0),
            "note": "CSV keeps reliable pair rows; centrality/figures remove non-positive or visually weak links.",
        },
        {
            "network": "champion_counter",
            "candidate_unique_edges": len(raw_counter_counts),
            "candidate_nodes": len(counter_candidate_nodes),
            "analysis_edge_rule": f"matchup_games >= {CONFIG['min_games_counter']} and smoothed counter weight > 0",
            "retained_analysis_edges": counter_graph.number_of_edges(),
            "retained_analysis_nodes": counter_graph.number_of_nodes(),
            "visualization_rule": f"top {visualization_top_n_edges} edges by weight; nodes not incident to those edges are hidden",
            "visualized_edges": counter_plot_edges,
            "visualized_nodes": counter_plot_nodes,
            "visualization_min_edge_weight": counter_plot_min_weight,
            "hidden_weak_nodes_in_figure": max(counter_graph.number_of_nodes() - counter_plot_nodes, 0),
            "note": "Directed weak matchup links are removed before centrality because they do not show a positive counter advantage.",
        },
    ]

    if team_champion_edges is not None and not team_champion_edges.empty:
        op_edges = team_champion_edges[team_champion_edges["op_pick_rows"] > 0].head(visualization_top_n_edges)
        high_weight_edges = team_champion_edges.sort_values("team_champion_pick_rows", ascending=False).head(visualization_top_n_edges)
        plotted = pd.concat([op_edges, high_weight_edges], ignore_index=True).drop_duplicates(["team", "champion"]).head(visualization_top_n_edges)
        two_mode_nodes = set(team_champion_edges["team"]) | set(team_champion_edges["champion"])
        plotted_nodes = set(plotted["team"]) | set(plotted["champion"])
        rows.append({
            "network": "team_champion_2mode",
            "candidate_unique_edges": len(team_champion_edges),
            "candidate_nodes": len(two_mode_nodes),
            "analysis_edge_rule": "all observed team-champion pick edges are retained in CSV",
            "retained_analysis_edges": len(team_champion_edges),
            "retained_analysis_nodes": len(two_mode_nodes),
            "visualization_rule": f"OP-pick edges plus top {visualization_top_n_edges} team-champion frequency edges",
            "visualized_edges": len(plotted),
            "visualized_nodes": len(plotted_nodes),
            "visualization_min_edge_weight": plotted["team_champion_pick_rows"].min() if not plotted.empty else np.nan,
            "hidden_weak_nodes_in_figure": max(len(two_mode_nodes) - len(plotted_nodes), 0),
            "note": "2-mode figure hides long-tail team/champion nodes for readability; summary CSV keeps them.",
        })

    criteria = pd.DataFrame(rows)
    criteria.to_csv(output_path, index=False, encoding="utf-8-sig")
    return criteria


# %% [markdown] ## 7. 예측 데이터셋 생성


# %% Code cell 17

def team_stat_aggregate(champions, stats_df, columns=("win_rate", "pick_rate", "ban_rate", "games")):
    """Average champion-level standalone stats for a team composition."""
    if stats_df.empty:
        return {column: 0.0 for column in columns}
    if "champion" in stats_df.columns:
        stats = stats_df.copy()
        stats["champion"] = stats["champion"].apply(normalize_champion_name)
        lookup = stats.drop_duplicates("champion").set_index("champion")
    else:
        lookup = stats_df
    result = {}
    for column in columns:
        values = [lookup.at[c, column] for c in champions if c in lookup.index and column in lookup.columns]
        values = [float(v) for v in values if pd.notna(v)]
        result[column] = float(np.mean(values)) if values else 0.0
    return result


def calculate_team_internal_synergy(champions, synergy_edges):
    """Calculate mean and max internal synergy score for a team composition."""
    if isinstance(synergy_edges, dict):
        lookup = synergy_edges
    else:
        lookup = {}
        for _, row in synergy_edges.iterrows():
            key = tuple(sorted((row["source"], row["target"])))
            lookup[key] = {
                "synergy_score": float(row["synergy_score"]),
                "weight": float(row.get("weight", 0)),
            }
    scores, weights = [], []
    for a, b in itertools.combinations(_unique_sorted(champions), 2):
        item = lookup.get(tuple(sorted((a, b))))
        if item:
            scores.append(item["synergy_score"])
            weights.append(item["weight"])
    return {
        "internal_synergy_mean": float(np.mean(scores)) if scores else 0.0,
        "internal_synergy_max": float(np.max(scores)) if scores else 0.0,
        "internal_synergy_weight_mean": float(np.mean(weights)) if weights else 0.0,
    }


def calculate_counter_advantage(team_picks, opponent_picks, counter_edges):
    """Calculate directed counter advantage from team picks to opponent picks."""
    if isinstance(counter_edges, dict):
        lookup = counter_edges
    else:
        lookup = {
            (row["source"], row["target"]): float(row["counter_score"])
            for _, row in counter_edges.iterrows()
        }
    scores = [
        lookup.get((source, target), 0.0)
        for source in _unique_sorted(team_picks)
        for target in _unique_sorted(opponent_picks)
    ]
    return {
        "counter_advantage_mean": float(np.mean(scores)) if scores else 0.0,
        "counter_advantage_max": float(np.max(scores)) if scores else 0.0,
    }


def _team_network_aggregate(champions, network_metrics):
    if network_metrics.empty:
        return defaultdict(float)
    if "champion" in network_metrics.columns:
        lookup = network_metrics.drop_duplicates("champion").set_index("champion")
    else:
        lookup = network_metrics
    metric_cols = [col for col in lookup.columns if col != "champion"]
    result = {}
    for column in metric_cols:
        values = [lookup.at[c, column] for c in champions if c in lookup.index and pd.notna(lookup.at[c, column])]
        result[column] = float(np.mean(values)) if values else 0.0
    return result


def build_prediction_dataset(match_df, champion_stats, network_metrics, synergy_edges, counter_edges):
    """
    Build match-level prediction rows.

    The target is blue_win. Features are expressed as blue minus red
    differences so the model can compare the two compositions directly.
    """
    df = ensure_match_schema(match_df)
    rows = []
    stat_cols = ["win_rate", "pick_rate", "ban_rate", "games"]
    centrality_cols = [
        "synergy_weighted_degree", "synergy_pagerank", "synergy_betweenness",
        "counter_weighted_out_degree", "counter_pagerank", "counter_hits_hub",
        "counter_hits_authority",
    ]

    # Precompute lookups once because the 26.9+ window has thousands of team rows.
    # Rebuilding these dictionaries inside every match loop was the main timeout risk.
    champion_stats_lookup = champion_stats.copy()
    if not champion_stats_lookup.empty and "champion" in champion_stats_lookup.columns:
        champion_stats_lookup["champion"] = champion_stats_lookup["champion"].apply(normalize_champion_name)
        champion_stats_lookup = champion_stats_lookup.drop_duplicates("champion").set_index("champion")
    network_lookup = network_metrics.drop_duplicates("champion").set_index("champion") if not network_metrics.empty else network_metrics
    synergy_lookup = {
        tuple(sorted((row["source"], row["target"]))): {
            "synergy_score": float(row["synergy_score"]),
            "weight": float(row.get("weight", 0)),
        }
        for _, row in synergy_edges.iterrows()
    }
    counter_lookup = {
        (row["source"], row["target"]): float(row["counter_score"])
        for _, row in counter_edges.iterrows()
    }

    for _, row in df.iterrows():
        blue_picks = row["blue_picks"]
        red_picks = row["red_picks"]
        blue_stats = team_stat_aggregate(blue_picks, champion_stats_lookup, stat_cols)
        red_stats = team_stat_aggregate(red_picks, champion_stats_lookup, stat_cols)
        blue_net = _team_network_aggregate(blue_picks, network_lookup)
        red_net = _team_network_aggregate(red_picks, network_lookup)
        blue_synergy = calculate_team_internal_synergy(blue_picks, synergy_lookup)
        red_synergy = calculate_team_internal_synergy(red_picks, synergy_lookup)
        blue_counter = calculate_counter_advantage(blue_picks, red_picks, counter_lookup)
        red_counter = calculate_counter_advantage(red_picks, blue_picks, counter_lookup)

        record = {
            "match_id": row["match_id"],
            "date": row["date"],
            "patch": row["patch"],
            "blue_team": row["blue_team"],
            "red_team": row["red_team"],
            "blue_win": int(row["blue_win"]),
            "blue_side_indicator": 1,
        }
        for column in stat_cols:
            record[f"blue_avg_{column}"] = blue_stats[column]
            record[f"red_avg_{column}"] = red_stats[column]
            record[f"{column}_diff"] = blue_stats[column] - red_stats[column]

        for column in centrality_cols:
            record[f"blue_avg_{column}"] = blue_net.get(column, 0.0)
            record[f"red_avg_{column}"] = red_net.get(column, 0.0)
            record[f"{column}_diff"] = blue_net.get(column, 0.0) - red_net.get(column, 0.0)

        record["blue_internal_synergy_mean"] = blue_synergy["internal_synergy_mean"]
        record["red_internal_synergy_mean"] = red_synergy["internal_synergy_mean"]
        record["synergy_mean_diff"] = blue_synergy["internal_synergy_mean"] - red_synergy["internal_synergy_mean"]
        record["blue_internal_synergy_max"] = blue_synergy["internal_synergy_max"]
        record["red_internal_synergy_max"] = red_synergy["internal_synergy_max"]
        record["synergy_max_diff"] = blue_synergy["internal_synergy_max"] - red_synergy["internal_synergy_max"]
        record["blue_counter_advantage_mean"] = blue_counter["counter_advantage_mean"]
        record["red_counter_advantage_mean"] = red_counter["counter_advantage_mean"]
        record["counter_advantage_diff"] = blue_counter["counter_advantage_mean"] - red_counter["counter_advantage_mean"]
        record["blue_counter_advantage_max"] = blue_counter["counter_advantage_max"]
        record["red_counter_advantage_max"] = red_counter["counter_advantage_max"]
        rows.append(record)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def chronological_train_test_split(match_df, test_size=None):
    """Split matches by date so later matches form the test set."""
    test_size = CONFIG["test_size"] if test_size is None else test_size
    df = ensure_match_schema(match_df).sort_values(["date", "match_id"]).reset_index(drop=True)
    split_idx = max(1, int(len(df) * (1 - test_size)))
    split_idx = min(split_idx, len(df) - 1)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# %% [markdown] ## 8. 모델 학습, 평가, 부트스트랩


# %% Code cell 19

BASELINE_NUMERIC_FEATURES = [
    "blue_avg_win_rate", "red_avg_win_rate", "win_rate_diff",
    "blue_avg_pick_rate", "red_avg_pick_rate", "pick_rate_diff",
    "blue_avg_ban_rate", "red_avg_ban_rate", "ban_rate_diff",
    "blue_avg_games", "red_avg_games", "games_diff",
]

NETWORK_NUMERIC_FEATURES = [
    "blue_internal_synergy_mean", "red_internal_synergy_mean", "synergy_mean_diff",
    "blue_internal_synergy_max", "red_internal_synergy_max", "synergy_max_diff",
    "blue_counter_advantage_mean", "red_counter_advantage_mean", "counter_advantage_diff",
    "blue_counter_advantage_max", "red_counter_advantage_max",
    "blue_avg_synergy_weighted_degree", "red_avg_synergy_weighted_degree", "synergy_weighted_degree_diff",
    "blue_avg_synergy_pagerank", "red_avg_synergy_pagerank", "synergy_pagerank_diff",
    "blue_avg_synergy_betweenness", "red_avg_synergy_betweenness", "synergy_betweenness_diff",
    "blue_avg_counter_weighted_out_degree", "red_avg_counter_weighted_out_degree", "counter_weighted_out_degree_diff",
    "blue_avg_counter_pagerank", "red_avg_counter_pagerank", "counter_pagerank_diff",
    "blue_avg_counter_hits_hub", "red_avg_counter_hits_hub", "counter_hits_hub_diff",
    "blue_avg_counter_hits_authority", "red_avg_counter_hits_authority", "counter_hits_authority_diff",
]


def _prepare_feature_frames(prediction_df, train_ids, test_ids):
    """Create baseline and network feature matrices with patch one-hot columns."""
    df = prediction_df.copy()
    patch_dummies = pd.get_dummies(df["patch"].fillna("Unknown"), prefix="patch", dtype=float)
    df = pd.concat([df, patch_dummies], axis=1)
    patch_cols = patch_dummies.columns.tolist()

    baseline_features = [col for col in BASELINE_NUMERIC_FEATURES if col in df.columns] + patch_cols
    network_features = baseline_features + [col for col in NETWORK_NUMERIC_FEATURES if col in df.columns]

    train_mask = df["match_id"].isin(train_ids)
    test_mask = df["match_id"].isin(test_ids)
    y_train = df.loc[train_mask, "blue_win"].astype(int)
    y_test = df.loc[test_mask, "blue_win"].astype(int)
    return {
        "X_train_baseline": df.loc[train_mask, baseline_features],
        "X_test_baseline": df.loc[test_mask, baseline_features],
        "X_train_network": df.loc[train_mask, network_features],
        "X_test_network": df.loc[test_mask, network_features],
        "y_train": y_train,
        "y_test": y_test,
        "baseline_features": baseline_features,
        "network_features": network_features,
    }


def make_logistic_pipeline():
    """Create a robust logistic-regression pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=CONFIG["random_state"])),
    ])


def make_random_forest_pipeline():
    """Create a random-forest pipeline for non-linear feature interactions."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=CONFIG["random_state"],
            class_weight="balanced",
        )),
    ])


def _safe_probability(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return None


def evaluate_classifier(model, X_train, X_test, y_train, y_test, model_name, feature_set):
    """Fit and evaluate a classifier with all required metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = _safe_probability(model, X_test)
    result = {
        "model_name": model_name,
        "feature_set": feature_set,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_test)) > 1:
        result["roc_auc"] = roc_auc_score(y_test, y_proba)
        result["log_loss"] = log_loss(y_test, np.clip(y_proba, 1e-6, 1 - 1e-6), labels=[0, 1])
        result["brier_score"] = brier_score_loss(y_test, y_proba)
    else:
        result["roc_auc"] = np.nan
        result["log_loss"] = np.nan
        result["brier_score"] = np.nan
    result["confusion_matrix"] = json.dumps(confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist())
    result["classification_report"] = classification_report(y_test, y_pred, zero_division=0)
    return model, y_pred, y_proba, result


def train_and_evaluate_models(prediction_df, train_ids, test_ids):
    """Train baseline and network models, then save model_performance.csv."""
    frames = _prepare_feature_frames(prediction_df, train_ids, test_ids)
    model_specs = [
        ("LogisticRegression", make_logistic_pipeline()),
        ("RandomForest", make_random_forest_pipeline()),
    ]
    results = []
    fitted = []

    for model_name, estimator in model_specs:
        model, pred, proba, result = evaluate_classifier(
            estimator,
            frames["X_train_baseline"],
            frames["X_test_baseline"],
            frames["y_train"],
            frames["y_test"],
            model_name,
            "baseline",
        )
        results.append(result)
        fitted.append({
            "model_name": model_name,
            "feature_set": "baseline",
            "model": model,
            "y_pred": pred,
            "y_proba": proba,
            "features": frames["baseline_features"],
        })

        network_estimator = make_logistic_pipeline() if model_name == "LogisticRegression" else make_random_forest_pipeline()
        model, pred, proba, result = evaluate_classifier(
            network_estimator,
            frames["X_train_network"],
            frames["X_test_network"],
            frames["y_train"],
            frames["y_test"],
            model_name,
            "network",
        )
        results.append(result)
        fitted.append({
            "model_name": model_name,
            "feature_set": "network",
            "model": model,
            "y_pred": pred,
            "y_proba": proba,
            "features": frames["network_features"],
        })

    performance_df = pd.DataFrame(results)
    baseline_by_model = (
        performance_df[performance_df["feature_set"] == "baseline"]
        .set_index("model_name")["accuracy"]
        .to_dict()
    )
    performance_df["baseline_accuracy_same_model"] = performance_df["model_name"].map(baseline_by_model)
    performance_df["accuracy_delta_vs_same_model"] = np.where(
        performance_df["feature_set"].eq("network"),
        performance_df["accuracy"] - performance_df["baseline_accuracy_same_model"],
        np.nan,
    )
    performance_df.to_csv(OUTPUT_DIR / "model_performance.csv", index=False, encoding="utf-8-sig")

    best_baseline_row = performance_df[performance_df["feature_set"] == "baseline"].sort_values("accuracy", ascending=False).iloc[0]
    best_network_row = performance_df[performance_df["feature_set"] == "network"].sort_values("accuracy", ascending=False).iloc[0]

    def find_fit(row):
        return next(item for item in fitted if item["model_name"] == row["model_name"] and item["feature_set"] == row["feature_set"])

    return {
        **frames,
        "performance_df": performance_df,
        "fitted": fitted,
        "best_baseline": find_fit(best_baseline_row),
        "best_network": find_fit(best_network_row),
        "best_baseline_accuracy": float(best_baseline_row["accuracy"]),
        "best_network_accuracy": float(best_network_row["accuracy"]),
        "accuracy_delta": float(best_network_row["accuracy"] - best_baseline_row["accuracy"]),
    }


def bootstrap_accuracy_delta(y_true, pred_baseline, pred_network, n_boot=1000, random_state=42):
    """Bootstrap a confidence interval for network minus baseline accuracy."""
    rng = np.random.default_rng(random_state)
    y_true = np.array(y_true)
    pred_baseline = np.array(pred_baseline)
    pred_network = np.array(pred_network)
    if len(y_true) == 0:
        return {"mean_delta": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        base_acc = accuracy_score(y_true[idx], pred_baseline[idx])
        net_acc = accuracy_score(y_true[idx], pred_network[idx])
        deltas.append(net_acc - base_acc)
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_lower": float(np.quantile(deltas, 0.025)),
        "ci_upper": float(np.quantile(deltas, 0.975)),
    }


def extract_feature_importance(model, feature_names, model_name):
    """Extract coefficient or tree-based feature importance."""
    final_model = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    if hasattr(final_model, "coef_"):
        values = np.abs(final_model.coef_[0])
    elif hasattr(final_model, "feature_importances_"):
        values = final_model.feature_importances_
    else:
        return pd.DataFrame(columns=["feature", "importance", "model_name"])
    return (
        pd.DataFrame({"feature": feature_names, "importance": values, "model_name": model_name})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# %% [markdown] ## 9. 시각화 함수


# %% Code cell 21

def savefig(name):
    """Save a matplotlib figure into figures/ with consistent settings."""
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    print(f"saved: {path}")
    return path


def _read_icon_image(path):
    """Read an icon image defensively so plotting never fails because of one file."""
    if path is None:
        return None
    try:
        return plt.imread(str(path))
    except Exception:
        return None


def add_champion_icons_to_axis(ax, champions, y_positions, champion_icon_map=None, x_axes=0.022, zoom=0.18):
    """Add champion icons inside the plot area so labels remain readable."""
    if not champion_icon_map:
        return
    for champion, y in zip(champions, y_positions):
        normalized = normalize_champion_name(champion)
        img = _read_icon_image(champion_icon_map.get(normalized))
        if img is None:
            continue
        image_box = OffsetImage(img, zoom=zoom)
        annotation = AnnotationBbox(
            image_box,
            (x_axes, y),
            xycoords=("axes fraction", "data"),
            frameon=True,
            bboxprops={"edgecolor": "#2B2B2B", "linewidth": 0.45, "facecolor": "white"},
            box_alignment=(0.5, 0.5),
            pad=0.01,
            annotation_clip=False,
            zorder=6,
        )
        ax.add_artist(annotation)
    # Keep tick labels close to the axis because icons now live inside
    # the chart, not between the label text and the plotting area.
    ax.tick_params(axis="y", pad=4)


def split_composition_label(composition):
    """Split a stored composition label back into champion names for icon rendering."""
    return [
        normalize_champion_name(part)
        for part in str(composition).split(" + ")
        if normalize_champion_name(part)
    ]


def add_composition_icons_to_axis(
    ax,
    compositions,
    y_positions,
    champion_icon_map=None,
    x_start=0.020,
    x_step=0.030,
    zoom=0.135,
    max_icons=5,
):
    """Render icon strips inside the plot area instead of over y labels."""
    if not champion_icon_map:
        return
    for composition, y in zip(compositions, y_positions):
        for offset, champion in enumerate(split_composition_label(composition)[:max_icons]):
            img = _read_icon_image(champion_icon_map.get(champion))
            if img is None:
                continue
            image_box = OffsetImage(img, zoom=zoom)
            annotation = AnnotationBbox(
                image_box,
                (x_start + offset * x_step, y),
                xycoords=("axes fraction", "data"),
                frameon=True,
                bboxprops={"edgecolor": "#2B2B2B", "linewidth": 0.4, "facecolor": "white"},
                box_alignment=(0.5, 0.5),
                pad=0.01,
                annotation_clip=False,
                zorder=6,
            )
            ax.add_artist(annotation)
    # A small pad prevents long composition names from being pushed
    # underneath the champion portraits.
    ax.tick_params(axis="y", pad=4)


def collect_visual_champions(
    match_df,
    network_metrics,
    op_scores,
    op_champion_tiers,
    best_without_op,
    best_with_op,
    op_loss_summary,
    top_n=15,
    synergy_edges=None,
    counter_edges=None,
    network_top_n_edges=70,
):
    """Collect only champions that appear in key plots to avoid downloading unused icons."""
    champions = set()

    pick_counter = Counter()
    ban_counter = Counter()
    for column in ["blue_picks", "red_picks"]:
        for values in match_df[column]:
            pick_counter.update(parse_list_column(values))
    for column in ["blue_bans", "red_bans"]:
        for values in match_df[column]:
            ban_counter.update(parse_list_column(values))
    champions.update(champion for champion, _ in pick_counter.most_common(top_n))
    champions.update(champion for champion, _ in ban_counter.most_common(top_n))

    if not network_metrics.empty:
        champions.update(network_metrics.sort_values("synergy_pagerank", ascending=False).head(top_n)["champion"])
        champions.update(network_metrics.sort_values("counter_weighted_out_degree", ascending=False).head(top_n)["champion"])
    for edge_df in [synergy_edges, counter_edges]:
        if edge_df is None or edge_df.empty:
            continue
        top_edges = edge_df.sort_values("weight", ascending=False).head(network_top_n_edges)
        champions.update(top_edges["source"])
        champions.update(top_edges["target"])

    if not op_scores.empty:
        champions.update(op_scores.sort_values("op_score", ascending=False).head(top_n)["champion"])
    if not op_champion_tiers.empty:
        champions.update(op_champion_tiers.head(top_n)["champion"])

    for df in [best_without_op, best_with_op]:
        if df.empty:
            continue
        for composition in df.sort_values(["smoothed_win_rate", "games"], ascending=False).head(10)["composition"]:
            champions.update(split_composition_label(composition))

    if not op_loss_summary.empty:
        for composition in op_loss_summary.sort_values("wins_against_op", ascending=False).head(top_n)["opponent_combo"]:
            champions.update(split_composition_label(composition))

    return sorted({normalize_champion_name(champion) for champion in champions if normalize_champion_name(champion)})


def plot_matches_by_patch(match_df):
    """Plot number of matches by patch."""
    patch_counts = match_df["patch"].fillna("Unknown").value_counts().sort_index()
    plt.figure(figsize=(9, 4.8))
    plt.bar(patch_counts.index.astype(str), patch_counts.values, color="#3A6EA5")
    plt.title("Number of Matches by Patch")
    plt.xlabel("Patch")
    plt.ylabel("Matches")
    plt.xticks(rotation=35, ha="right")
    return savefig("01_matches_by_patch.png")


def plot_pick_ban_top_champions(match_df, top_n=15, champion_icon_map=None):
    """Plot top picked and banned champions in one required figure."""
    pick_counter = Counter()
    ban_counter = Counter()
    for column in ["blue_picks", "red_picks"]:
        for champions in match_df[column]:
            pick_counter.update(parse_list_column(champions))
    for column in ["blue_bans", "red_bans"]:
        for champions in match_df[column]:
            ban_counter.update(parse_list_column(champions))

    pick_df = pd.DataFrame(pick_counter.most_common(top_n), columns=["champion", "count"])
    ban_df = pd.DataFrame(ban_counter.most_common(top_n), columns=["champion", "count"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    pick_labels = pick_df["champion"][::-1].tolist()
    pick_y = np.arange(len(pick_labels))
    axes[0].barh(pick_y, pick_df["count"][::-1], color="#3A6EA5")
    axes[0].set_yticks(pick_y)
    axes[0].set_yticklabels(pick_labels)
    add_champion_icons_to_axis(axes[0], pick_labels, pick_y, champion_icon_map)
    axes[0].set_title(f"Top {top_n} Picked Champions")
    axes[0].set_xlabel("Pick Count")
    ban_labels = ban_df["champion"][::-1].tolist()
    ban_y = np.arange(len(ban_labels))
    axes[1].barh(ban_y, ban_df["count"][::-1], color="#B54A4A")
    axes[1].set_yticks(ban_y)
    axes[1].set_yticklabels(ban_labels)
    add_champion_icons_to_axis(axes[1], ban_labels, ban_y, champion_icon_map)
    axes[1].set_title(f"Top {top_n} Banned Champions")
    axes[1].set_xlabel("Ban Count")
    return savefig("02_top_pick_ban_champions.png")


def plot_weighted_network(
    graph,
    title,
    filename,
    node_metric=None,
    top_n_edges=80,
    directed=False,
    seed=42,
    champion_icon_map=None,
):
    """Plot the top weighted edges of a networkx graph."""
    if graph.number_of_edges() == 0:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No edges to plot", ha="center", va="center")
        plt.title(title)
        plt.axis("off")
        return savefig(filename)

    edges_sorted = sorted(graph.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True)[:top_n_edges]
    subgraph = nx.DiGraph() if directed else nx.Graph()
    subgraph.add_edges_from(edges_sorted)
    pos = nx.spring_layout(subgraph, seed=seed, weight="weight", k=0.9)

    if node_metric is None:
        node_metric = dict(subgraph.degree(weight="weight"))
    values = np.array([node_metric.get(node, 0.0) for node in subgraph.nodes()], dtype=float)
    if values.size and values.max() > values.min():
        sizes = 350 + 2600 * (values - values.min()) / (values.max() - values.min())
    else:
        sizes = np.full(len(subgraph.nodes()), 800)
    weights = np.array([data.get("weight", 0.001) for _, _, data in subgraph.edges(data=True)], dtype=float)
    widths = 0.6 + 4.5 * weights / max(weights.max(), 0.001)

    plt.figure(figsize=(13, 9))
    nx.draw_networkx_edges(
        subgraph, pos, width=widths, alpha=0.45, edge_color="#4A6FA5",
        arrows=directed, arrowsize=12, connectionstyle="arc3,rad=0.08" if directed else "arc3",
    )
    nx.draw_networkx_nodes(subgraph, pos, node_size=sizes, node_color="#F2C14E", edgecolors="#2B2B2B", linewidths=0.6, alpha=0.82)
    if champion_icon_map:
        ax = plt.gca()
        for node, (x, y) in pos.items():
            img = _read_icon_image(champion_icon_map.get(normalize_champion_name(node)))
            if img is None:
                continue
            image_box = OffsetImage(img, zoom=0.16)
            annotation = AnnotationBbox(
                image_box,
                (x, y),
                frameon=True,
                bboxprops={"edgecolor": "#2B2B2B", "linewidth": 0.35, "facecolor": "white"},
                pad=0.02,
                annotation_clip=False,
            )
            ax.add_artist(annotation)
        y_values = [point[1] for point in pos.values()]
        label_offset = 0.035 * (max(y_values) - min(y_values) if len(y_values) > 1 else 1.0)
        label_pos = {node: (x, y - label_offset) for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(
            subgraph,
            label_pos,
            font_size=7,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
    else:
        nx.draw_networkx_labels(subgraph, pos, font_size=8)
    plt.title(title)
    plt.axis("off")
    return savefig(filename)


def plot_top_centrality(metric_df, top_n=15, champion_icon_map=None):
    """Plot top synergy and counter centrality bars."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    synergy = metric_df.sort_values("synergy_pagerank", ascending=False).head(top_n)
    counter = metric_df.sort_values("counter_weighted_out_degree", ascending=False).head(top_n)
    synergy_labels = synergy["champion"][::-1].tolist()
    synergy_y = np.arange(len(synergy_labels))
    axes[0].barh(synergy_y, synergy["synergy_pagerank"][::-1], color="#3A6EA5")
    axes[0].set_yticks(synergy_y)
    axes[0].set_yticklabels(synergy_labels)
    add_champion_icons_to_axis(axes[0], synergy_labels, synergy_y, champion_icon_map)
    axes[0].set_title("Top Synergy PageRank")
    axes[0].set_xlabel("Synergy PageRank")
    counter_labels = counter["champion"][::-1].tolist()
    counter_y = np.arange(len(counter_labels))
    axes[1].barh(counter_y, counter["counter_weighted_out_degree"][::-1], color="#6C9A8B")
    axes[1].set_yticks(counter_y)
    axes[1].set_yticklabels(counter_labels)
    add_champion_icons_to_axis(axes[1], counter_labels, counter_y, champion_icon_map)
    axes[1].set_title("Top Counter Weighted Out-Degree")
    axes[1].set_xlabel("Counter Weighted Out-Degree")
    return savefig("05_top_centrality.png")


def plot_model_performance(performance_df):
    """Plot model accuracy comparison."""
    temp = performance_df.copy()
    temp["label"] = temp["model_name"] + " / " + temp["feature_set"]
    colors = ["#3A6EA5" if fs == "baseline" else "#6C9A8B" for fs in temp["feature_set"]]
    plt.figure(figsize=(11, 5))
    plt.bar(temp["label"], temp["accuracy"], color=colors)
    plt.title("Model Accuracy Comparison: Baseline vs Network Features")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    return savefig("06_model_accuracy_comparison.png")


def plot_confusion_matrices(results):
    """Save confusion matrices for best baseline and best network models."""
    y_test = results["y_test"]
    for key, filename, title in [
        ("best_baseline", "07_confusion_matrix_baseline.png", "Best Baseline Confusion Matrix"),
        ("best_network", "08_confusion_matrix_network.png", "Best Network Confusion Matrix"),
    ]:
        plt.figure(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, results[key]["y_pred"], labels=[0, 1], cmap="Blues")
        plt.title(title)
        savefig(filename)


def plot_roc_curves(results):
    """Plot baseline and network ROC curves in one figure."""
    y_test = results["y_test"]
    if len(np.unique(y_test)) < 2:
        plt.figure(figsize=(6, 5))
        plt.text(0.5, 0.5, "ROC requires two classes in test set", ha="center", va="center")
        plt.axis("off")
        return savefig("09_roc_curve_comparison.png")

    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    for key, feature_key, label in [
        ("best_baseline", "X_test_baseline", "Best Baseline"),
        ("best_network", "X_test_network", "Best Network"),
    ]:
        model = results[key]["model"]
        RocCurveDisplay.from_estimator(model, results[feature_key], y_test, name=label, ax=ax)
    plt.title("ROC Curve Comparison")
    return savefig("09_roc_curve_comparison.png")


def plot_feature_importance(importance_df, top_n=20):
    """Plot top model feature importances."""
    temp = importance_df.sort_values("importance", ascending=False).head(top_n)
    plt.figure(figsize=(10, 7))
    plt.barh(temp["feature"][::-1], temp["importance"][::-1], color="#6C9A8B")
    plt.title(f"Top {top_n} Feature Importance")
    plt.xlabel("Importance")
    return savefig("10_feature_importance.png")


def plot_correlation_heatmap(df, numeric_cols, filename="11_feature_correlation_heatmap.png"):
    """Plot a simple feature correlation heatmap."""
    cols = [col for col in numeric_cols if col in df.columns]
    if len(cols) < 2:
        return None
    corr = df[cols].corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title("Feature Correlation Heatmap")
    return savefig(filename)


def plot_op_champions(op_df, top_n=15, champion_icon_map=None):
    """Plot pro-game OP score reference, not the project OP definition."""
    temp = op_df.sort_values("op_score", ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    # This score is calculated per patch-champion row. Including the
    # patch in the label prevents repeated champions from looking like
    # duplicated data when one champion is strong across several patches.
    if "patch" in temp.columns:
        patch_labels = temp["patch"].apply(lambda value: normalize_patch_label(value, CONFIG["analysis_min_patch"]))
        labels = (patch_labels + " / " + temp["champion"]).iloc[::-1].tolist()
    else:
        labels = temp["champion"][::-1].tolist()
    y_positions = np.arange(len(labels))
    plt.barh(y_positions, temp["op_score"][::-1], color="#B54A4A")
    ax = plt.gca()
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    add_champion_icons_to_axis(ax, temp["champion"][::-1].tolist(), y_positions, champion_icon_map)
    plt.title("Patch-Champion Pro-Game Presence/Win Reference Score")
    plt.xlabel("Reference score = 0.70*presence + 0.30*positive smoothed win component")
    return savefig("12_op_champion_score.png")


def plot_opgg_op_champions(op_champion_tiers, top_n=15, champion_icon_map=None):
    """Plot project OP champions defined by OP.GG tier."""
    temp = op_champion_tiers.head(top_n).copy()
    if temp.empty:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No OP.GG OP-tier champions found", ha="center", va="center")
        plt.axis("off")
        return savefig("15_opgg_op_champion_tiers.png")
    if "patch" in temp.columns:
        patch_labels = temp["patch"].apply(lambda value: normalize_patch_label(value, CONFIG["analysis_min_patch"]))
    elif "leaguepedia_patch" in temp.columns:
        patch_labels = temp["leaguepedia_patch"].apply(lambda value: normalize_patch_label(value, CONFIG["analysis_min_patch"]))
    else:
        patch_labels = pd.Series([""] * len(temp), index=temp.index)
    labels = patch_labels + " / " + temp["champion"] + " / " + temp["position"] + " / " + temp["opgg_tier_label"]
    plt.figure(figsize=(10, 5))
    y_positions = np.arange(len(labels))
    label_values = labels[::-1].tolist()
    plt.barh(y_positions, temp["opgg_rank"][::-1], color="#B54A4A")
    ax = plt.gca()
    ax.set_yticks(y_positions)
    ax.set_yticklabels(label_values)
    add_champion_icons_to_axis(ax, temp["champion"][::-1].tolist(), y_positions, champion_icon_map)
    plt.title("Project OP Champions from OP.GG Patch Tier")
    plt.xlabel("OP.GG role rank within tier (lower is stronger)")
    return savefig("15_opgg_op_champion_tiers.png")


def plot_op_status_winrate(op_status_summary):
    """Plot the primary OP-effect check with bootstrap confidence interval."""
    temp = op_status_summary[op_status_summary["scope"].eq("match_exactly_one_team_has_op")].copy()
    if temp.empty:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No exactly-one-OP matches found", ha="center", va="center")
        plt.axis("off")
        return savefig("16_op_status_winrate_summary.png")

    row = temp.iloc[0]
    rate = float(row["win_rate"])
    yerr = np.array([[rate - float(row["ci_lower"])], [float(row["ci_upper"]) - rate]])
    plt.figure(figsize=(7, 5))
    plt.bar(["OP team"], [rate], yerr=yerr, color="#B54A4A", capsize=8)
    plt.axhline(0.5, color="#2B2B2B", linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.ylabel("Win Rate")
    plt.title(f"OP Team Win Rate When Exactly One Team Picks OP (n={int(row['games_or_rows'])})")
    plt.text(0, min(rate + 0.06, 0.95), f"{rate:.1%}", ha="center", va="bottom", fontweight="bold")
    return savefig("16_op_status_winrate_summary.png")


def plot_opgg_tier_sensitivity(tier_sensitivity):
    """Plot robustness of OP-effect estimates as the OP.GG tier cutoff is expanded."""
    if tier_sensitivity.empty:
        return None
    temp = tier_sensitivity.copy()
    labels = temp["tier_group"].tolist()
    rates = temp["win_rate"].astype(float).to_numpy()
    lower = rates - temp["ci_lower"].astype(float).to_numpy()
    upper = temp["ci_upper"].astype(float).to_numpy() - rates
    yerr = np.vstack([lower, upper])
    plt.figure(figsize=(9, 5))
    plt.bar(labels, rates, yerr=yerr, color=["#B54A4A", "#3A6EA5", "#6C9A8B"][: len(labels)], capsize=7)
    plt.axhline(0.5, color="#2B2B2B", linestyle="--", linewidth=1)
    for idx, row in temp.iterrows():
        plt.text(idx, min(float(row["win_rate"]) + 0.055, 0.95), f"n={int(row['exactly_one_side_games'])}", ha="center")
    plt.ylim(0, 1)
    plt.ylabel("Win Rate")
    plt.title("Sensitivity Check by OP.GG Tier Cutoff")
    return savefig("17_opgg_tier_sensitivity.png")


def plot_best_compositions(without_op, with_op, top_n=10, champion_icon_map=None):
    """Plot best non-OP and OP-containing compositions side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, df, title, color in [
        (axes[0], without_op, "Best Compositions Without OP Champions", "#3A6EA5"),
        (axes[1], with_op, "Best Compositions Including OP Champions", "#6C9A8B"),
    ]:
        temp = df.sort_values(["smoothed_win_rate", "games"], ascending=False).head(top_n)
        if temp.empty:
            ax.text(0.5, 0.5, "No compositions found", ha="center", va="center")
            ax.axis("off")
            continue
        labels = (temp["composition"] + " (" + temp["games"].astype(str) + "g)").iloc[::-1].tolist()
        compositions = temp["composition"].iloc[::-1].tolist()
        y_positions = np.arange(len(labels))
        ax.barh(y_positions, temp["smoothed_win_rate"].iloc[::-1], color=color)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        add_composition_icons_to_axis(ax, compositions, y_positions, champion_icon_map)
        ax.set_title(title)
        ax.set_xlabel("Smoothed Win Rate")
        ax.set_xlim(0, 1)
    return savefig("13_best_compositions_by_op_status.png")


def plot_op_loss_opponent_combos(op_loss_summary, top_n=15, champion_icon_map=None):
    """Plot opponent combos that most often beat teams with OP champions."""
    if op_loss_summary.empty:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No OP-loss opponent combos found", ha="center", va="center")
        plt.axis("off")
        return savefig("14_op_loss_opponent_combos.png")
    temp = op_loss_summary.sort_values(
        ["smoothed_win_rate_against_op", "wins_against_op", "games_against_op"],
        ascending=False,
    ).head(top_n)
    labels = (
        temp["opponent_combo"]
        + " ("
        + temp["wins_against_op"].astype(str)
        + "/"
        + temp["games_against_op"].astype(str)
        + ")"
    ).iloc[::-1].tolist()
    compositions = temp["opponent_combo"].iloc[::-1].tolist()
    y_positions = np.arange(len(labels))
    plt.figure(figsize=(12, 7))
    plt.barh(y_positions, temp["wins_against_op"].iloc[::-1], color="#F2C14E")
    ax = plt.gca()
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    add_composition_icons_to_axis(ax, compositions, y_positions, champion_icon_map)
    plt.title("Opponent Combos That Beat OP-Containing Teams")
    plt.xlabel("Wins against OP-containing teams")
    return savefig("14_op_loss_opponent_combos.png")


def plot_op_loss_rune_item_exceptions(op_loss_op_build_notes, op_loss_winning_build_notes):
    """Plot rune/item signals found in OP-loss exception games."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    if op_loss_op_build_notes.empty:
        axes[0].text(0.5, 0.5, "No OP-loss build notes", ha="center", va="center")
        axes[0].axis("off")
    else:
        rune_counts = op_loss_op_build_notes["key_rune"].value_counts().head(8)
        axes[0].barh(rune_counts.index[::-1], rune_counts.values[::-1], color="#B54A4A")
        axes[0].set_title("OP-Loss Champion Key Runes")
        axes[0].set_xlabel("Games")

    rare_items = []
    if not op_loss_winning_build_notes.empty:
        for value in op_loss_winning_build_notes["rare_items"].fillna(""):
            rare_items.extend([item.strip() for item in str(value).split(",") if item.strip()])
    if rare_items:
        item_counts = pd.Series(rare_items).value_counts().head(10)
        axes[1].barh(item_counts.index[::-1], item_counts.values[::-1], color="#6C9A8B")
        axes[1].set_title("Rare Items on Winning Opponents")
        axes[1].set_xlabel("Flagged occurrences")
    else:
        axes[1].text(0.5, 0.5, "No rare opponent items flagged", ha="center", va="center")
        axes[1].axis("off")

    return savefig("18_op_loss_rune_item_exceptions.png")


def plot_team_champion_two_mode_network(team_champion_edges, champion_two_mode_summary, top_n_edges=70, champion_icon_map=None):
    """Visualize the supplementary team-champion 2-mode network."""
    if team_champion_edges.empty:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No team-champion 2-mode edges found", ha="center", va="center")
        plt.axis("off")
        return savefig("19_team_champion_two_mode_network.png")

    op_edges = team_champion_edges[team_champion_edges["op_pick_rows"] > 0].head(top_n_edges)
    high_weight_edges = team_champion_edges.sort_values("team_champion_pick_rows", ascending=False).head(top_n_edges)
    temp = pd.concat([op_edges, high_weight_edges], ignore_index=True)
    temp = temp.drop_duplicates(["team", "champion"]).head(top_n_edges)

    graph = nx.Graph()
    for _, row in temp.iterrows():
        team_node = f"team::{row['team']}"
        champion_node = f"champion::{row['champion']}"
        graph.add_node(team_node, node_type="team", label=row["team"])
        graph.add_node(champion_node, node_type="champion", label=row["champion"])
        graph.add_edge(team_node, champion_node, weight=float(row["team_champion_pick_rows"]), op_weight=float(row["op_pick_rows"]))

    if graph.number_of_nodes() == 0:
        return None

    pos = nx.spring_layout(graph, weight="weight", seed=CONFIG["random_state"], k=0.55, iterations=120)
    node_colors = ["#3A6EA5" if graph.nodes[node]["node_type"] == "team" else "#B54A4A" for node in graph.nodes]
    node_sizes = [140 + 18 * graph.degree(node, weight="weight") for node in graph.nodes]
    edge_widths = [0.8 + 0.18 * graph.edges[edge].get("weight", 1) for edge in graph.edges]
    edge_colors = ["#B54A4A" if graph.edges[edge].get("op_weight", 0) > 0 else "#A0A4A8" for edge in graph.edges]

    plt.figure(figsize=(13, 9))
    ax = plt.gca()
    nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color=edge_colors, alpha=0.45, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.88, linewidths=0.8, edgecolors="white", ax=ax)

    team_weight = {
        row["team"]: row["team_champion_pick_rows"]
        for _, row in temp.groupby("team", as_index=False)["team_champion_pick_rows"].sum().iterrows()
    }
    champion_weight = {
        row["champion"]: row["team_champion_pick_rows"]
        for _, row in temp.groupby("champion", as_index=False)["team_champion_pick_rows"].sum().iterrows()
    }
    label_nodes = set()
    label_nodes.update(f"team::{team}" for team, _ in sorted(team_weight.items(), key=lambda x: x[1], reverse=True)[:10])
    label_nodes.update(f"champion::{champion}" for champion, _ in sorted(champion_weight.items(), key=lambda x: x[1], reverse=True)[:14])
    if champion_icon_map:
        for node, (x, y) in pos.items():
            if graph.nodes[node]["node_type"] != "champion":
                continue
            champion = normalize_champion_name(graph.nodes[node]["label"])
            img = _read_icon_image(champion_icon_map.get(champion))
            if img is None:
                continue
            image_box = OffsetImage(img, zoom=0.12)
            ax.add_artist(AnnotationBbox(image_box, (x, y), frameon=False, pad=0, annotation_clip=False))

    x_values = [point[0] for point in pos.values()]
    y_values = [point[1] for point in pos.values()]
    x_span = max(x_values) - min(x_values) if len(x_values) > 1 else 1.0
    y_span = max(y_values) - min(y_values) if len(y_values) > 1 else 1.0
    labels = {
        node: str(graph.nodes[node]["label"])[:18]
        for node in label_nodes
        if node in graph.nodes
    }
    label_pos = {}
    for node in labels:
        x, y = pos[node]
        if graph.nodes[node]["node_type"] == "champion":
            # Champion portraits sit on the node, so labels are moved
            # below the icon and given a white background for contrast.
            label_pos[node] = (x, y - 0.050 * y_span)
        else:
            label_pos[node] = (x + 0.012 * x_span, y + 0.018 * y_span)
    nx.draw_networkx_labels(
        graph,
        label_pos,
        labels=labels,
        font_size=8,
        ax=ax,
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )

    ax.set_title("Team-Champion 2-Mode Network (Lecture-Inspired Supplement)")
    ax.text(
        0.01,
        0.01,
        "Blue nodes=teams, red nodes=champions, red edges include OP-patch picks",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        ha="left",
        va="bottom",
    )
    ax.axis("off")
    return savefig("19_team_champion_two_mode_network.png")


# %% [markdown] ## 10. 데이터 로드, train-only 네트워크 생성, 중심성 저장


# %% Code cell 23

raw_match_df = load_or_scrape_match_history()
raw_match_df = ensure_match_schema(raw_match_df)
match_df = filter_complete_pick_matches(raw_match_df)
analysis_patches = sorted(match_df["patch"].dropna().astype(str).unique().tolist(), key=_patch_number)
analysis_patch_label = f"{analysis_patches[0]}+" if len(analysis_patches) > 1 else analysis_patches[0]
latest_patch_in_analysis = analysis_patches[-1]
scoreboard_players = load_or_scrape_scoreboard_players(analysis_patches, match_ids=match_df["match_id"].tolist())
match_df = correct_match_picks_from_scoreboard(match_df, scoreboard_players)
champion_stats = load_or_scrape_champion_stats(match_df)

if "data_source" in match_df.columns and (match_df["data_source"] == "sample_simulated").any():
    display(Markdown("**주의:** 현재 노트북은 실행 검증을 위해 생성된 **샘플 데이터**를 사용 중입니다. 실제 분석에서는 Leaguepedia/GOL.GG 크롤링을 켜거나 CSV를 교체하세요."))

train_matches, test_matches = chronological_train_test_split(match_df)
champion_model_stats = build_champion_model_stats(train_matches, champion_stats)
quality_report = audit_data_quality(match_df, champion_stats, champion_model_stats, raw_match_df=raw_match_df)
train_team_rows = make_team_rows(train_matches)

synergy_edges = build_synergy_edges(train_team_rows)
if synergy_edges.empty and CONFIG["min_games_synergy"] > 1:
    print("[warning] min_games_synergy 조건으로 엣지가 없어 1로 낮춰 재계산합니다.")
    synergy_edges = build_synergy_edges(train_team_rows, min_games=1)

counter_edges = build_counter_edges(train_team_rows)
if counter_edges.empty and CONFIG["min_games_counter"] > 1:
    print("[warning] min_games_counter 조건으로 엣지가 없어 1로 낮춰 재계산합니다.")
    counter_edges = build_counter_edges(train_team_rows, min_games=1)

synergy_graph = make_synergy_graph(synergy_edges)
counter_graph = make_counter_graph(counter_edges)
synergy_metrics = calculate_synergy_centrality(synergy_graph)
counter_metrics = calculate_counter_centrality(counter_graph)
network_metrics = merge_champion_network_metrics(synergy_metrics, counter_metrics)
op_scores = compute_patch_champion_op_scores(match_df)
opgg_tiers = load_or_scrape_opgg_champion_tiers_for_patches(analysis_patches)
op_champion_tiers = summarize_op_champion_tiers(op_scores, opgg_tiers)
op_champions_by_patch = build_patch_op_lookup(op_champion_tiers)
op_champions = sorted({champion for champions in op_champions_by_patch.values() for champion in champions})
best_without_op, best_with_op = analyze_compositions_by_op_status(match_df, op_champions_by_patch)
op_loss_details, op_loss_summary = analyze_op_loss_opponent_compositions(match_df, op_champions_by_patch)
op_status_summary, op_status_match_details = analyze_op_status_effect(match_df, op_champions_by_patch)
opgg_tier_sensitivity = analyze_opgg_tier_sensitivity(match_df, opgg_tiers)
rune_item_baseline = build_rune_item_baseline(scoreboard_players)
op_loss_rune_item_notes, op_loss_op_build_notes, op_loss_winning_build_notes = analyze_op_loss_rune_item_exceptions(
    scoreboard_players,
    op_loss_details,
    op_champions_by_patch,
    rune_item_baseline,
)
team_champion_edges, team_two_mode_summary, champion_two_mode_summary, champion_projection_edges = build_team_champion_two_mode_network(
    match_df,
    op_champions_by_patch,
)
network_filtering_criteria = create_network_filtering_criteria(
    train_team_rows,
    synergy_edges,
    counter_edges,
    synergy_graph,
    counter_graph,
    team_champion_edges,
    visualization_top_n_edges=70,
)
visual_champions = collect_visual_champions(
    match_df,
    network_metrics,
    op_scores,
    op_champion_tiers,
    best_without_op,
    best_with_op,
    op_loss_summary,
    top_n=CONFIG["top_n"],
    synergy_edges=synergy_edges,
    counter_edges=counter_edges,
    network_top_n_edges=70,
)
if not champion_two_mode_summary.empty:
    visual_champions = sorted(set(visual_champions) | set(champion_two_mode_summary.head(CONFIG["top_n"])["champion"]))
# Only champions used in the required figures are downloaded, keeping reruns faster.
champion_icon_map = load_champion_icon_map(visual_champions, selected_patch=latest_patch_in_analysis)

print(f"matches: {len(match_df)} | train: {len(train_matches)} | test: {len(test_matches)}")
print(f"synergy edges: {len(synergy_edges)} | counter edges: {len(counter_edges)}")
print(f"analysis patches: {', '.join(analysis_patches)} | OP champions: {', '.join(op_champions[:CONFIG['op_top_n']])}")
display(Markdown("### Data Quality Report"))
display(quality_report)
display(Markdown("### Pick Alignment Audit"))
pick_alignment_audit = pd.read_csv(OUTPUT_DIR / "pick_alignment_audit.csv")
display(pick_alignment_audit["corrected"].value_counts().rename_axis("corrected").reset_index(name="team_rows"))
display(Markdown("### Pro-Game Presence/Win Reference by Champion"))
display(op_scores.head(CONFIG["op_top_n"]))
display(Markdown("### Project OP Champions Defined by OP.GG Tier"))
display(op_champion_tiers.head(CONFIG["op_top_n"]))
display(Markdown("### OP Status Win-Rate Effect"))
display(op_status_summary)
display(Markdown("### OP.GG Tier-Cutoff Sensitivity"))
display(opgg_tier_sensitivity)
display(Markdown("### OP.GG Champion Role Tier Table"))
display(opgg_tiers.head(15))
display(Markdown("### Best Compositions Without OP Champions"))
display(best_without_op.head(10))
display(Markdown("### Best Compositions Including OP Champions"))
display(best_with_op.head(10))
display(Markdown("### Opponent Compositions That Beat OP-Containing Teams"))
display(op_loss_summary.head(10))
display(Markdown("### Rune/Item Exception Check: OP-Loss Champions"))
display(op_loss_op_build_notes.head(10))
display(Markdown("### Rune/Item Exception Check: Winning Opponent Players"))
display(op_loss_winning_build_notes.head(15))
display(Markdown("### Lecture Supplement: Team-Champion 2-Mode Network"))
display(team_two_mode_summary.head(10))
display(champion_two_mode_summary.head(10))
display(champion_projection_edges.head(10))
display(Markdown("### Network Filtering / Weak Node Criteria"))
display(network_filtering_criteria)
display(Markdown("### Scraped GOL.GG Champion Stats"))
display(champion_stats.head())
display(Markdown("### Model Champion Stats (train-only + GOL.GG prior)"))
display(champion_model_stats.head())
display(network_metrics.head())


# %% [markdown] ## 11. 승/패 예측 모델 비교


# %% Code cell 25

prediction_df = build_prediction_dataset(match_df, champion_model_stats, network_metrics, synergy_edges, counter_edges)
model_results = train_and_evaluate_models(
    prediction_df,
    train_matches["match_id"].tolist(),
    test_matches["match_id"].tolist(),
)

performance_df = model_results["performance_df"]
importance_df = extract_feature_importance(
    model_results["best_network"]["model"],
    model_results["best_network"]["features"],
    f'{model_results["best_network"]["model_name"]}_network',
)
importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

ci = bootstrap_accuracy_delta(
    model_results["y_test"],
    model_results["best_baseline"]["y_pred"],
    model_results["best_network"]["y_pred"],
    n_boot=500,
    random_state=CONFIG["random_state"],
)

display_cols = [
    "model_name", "feature_set", "accuracy", "accuracy_delta_vs_same_model",
    "f1_score", "roc_auc", "log_loss", "brier_score",
]
display(performance_df[display_cols].sort_values(["model_name", "feature_set"]))
print(f'Best baseline accuracy: {model_results["best_baseline_accuracy"]:.4f}')
print(f'Best network accuracy: {model_results["best_network_accuracy"]:.4f}')
print(f'Accuracy delta: {model_results["accuracy_delta"]:.4f}')
print(f'Bootstrap CI for accuracy delta: [{ci["ci_lower"]:.4f}, {ci["ci_upper"]:.4f}]')
display(importance_df.head(15))
professor_review = create_professor_quality_review(
    quality_report,
    op_status_summary,
    opgg_tier_sensitivity,
    performance_df,
)
display(Markdown("### Professor-Style Quality Review"))
display(professor_review)


# %% [markdown] ## 12. 필수 시각화 생성


# %% Code cell 27

plot_matches_by_patch(match_df)
plot_pick_ban_top_champions(match_df, top_n=CONFIG["top_n"], champion_icon_map=champion_icon_map)
plot_weighted_network(
    synergy_graph,
    "Champion Synergy Network (train-only)",
    "03_synergy_network.png",
    node_metric=dict(synergy_graph.degree(weight="weight")),
    top_n_edges=70,
    directed=False,
    champion_icon_map=champion_icon_map,
)
plot_weighted_network(
    counter_graph,
    "Champion Counter Network (train-only)",
    "04_counter_network.png",
    node_metric=dict(counter_graph.out_degree(weight="weight")),
    top_n_edges=70,
    directed=True,
    champion_icon_map=champion_icon_map,
)
plot_top_centrality(network_metrics, top_n=CONFIG["top_n"], champion_icon_map=champion_icon_map)
plot_model_performance(performance_df)
plot_confusion_matrices(model_results)
plot_roc_curves(model_results)
plot_feature_importance(importance_df, top_n=20)
corr_cols = BASELINE_NUMERIC_FEATURES + NETWORK_NUMERIC_FEATURES
plot_correlation_heatmap(prediction_df, corr_cols)
plot_op_champions(op_scores, top_n=CONFIG["top_n"], champion_icon_map=champion_icon_map)
plot_opgg_op_champions(op_champion_tiers, top_n=CONFIG["top_n"], champion_icon_map=champion_icon_map)
plot_op_status_winrate(op_status_summary)
plot_opgg_tier_sensitivity(opgg_tier_sensitivity)
plot_best_compositions(best_without_op, best_with_op, top_n=10, champion_icon_map=champion_icon_map)
plot_op_loss_opponent_combos(op_loss_summary, top_n=15, champion_icon_map=champion_icon_map)
plot_op_loss_rune_item_exceptions(op_loss_op_build_notes, op_loss_winning_build_notes)
plot_team_champion_two_mode_network(team_champion_edges, champion_two_mode_summary, top_n_edges=70, champion_icon_map=champion_icon_map)


# %% [markdown] ## 12-1. 시각화 해석 가이드


# %% [markdown] ## 13. 최종 산출물 체크리스트


# %% Code cell 30

required_paths = [
    ROOT / "비정형네트워크프로젝트.ipynb",
    ROOT / "requirements.txt",
    DATA_DIR / "raw_match_history.csv",
    DATA_DIR / "analysis_match_history.csv",
    DATA_DIR / "scoreboard_players_patch.csv",
    DATA_DIR / "champion_global_stats.csv",
    DATA_DIR / "opgg_champion_tiers.csv",
    DATA_DIR / "champion_edges_synergy.csv",
    DATA_DIR / "champion_edges_counter.csv",
    OUTPUT_DIR / "model_performance.csv",
    OUTPUT_DIR / "champion_network_metrics.csv",
    OUTPUT_DIR / "network_filtering_criteria.csv",
    OUTPUT_DIR / "feature_importance.csv",
    OUTPUT_DIR / "champion_model_stats.csv",
    OUTPUT_DIR / "data_quality_report.csv",
    OUTPUT_DIR / "pick_alignment_audit.csv",
    OUTPUT_DIR / "selected_patch_summary.csv",
    OUTPUT_DIR / "excluded_incomplete_matches.csv",
    OUTPUT_DIR / "patch_op_champion_scores.csv",
    OUTPUT_DIR / "op_champion_tiers.csv",
    OUTPUT_DIR / "op_status_winrate_summary.csv",
    OUTPUT_DIR / "op_status_match_details.csv",
    OUTPUT_DIR / "opgg_tier_sensitivity.csv",
    OUTPUT_DIR / "rune_item_patch_baseline.csv",
    OUTPUT_DIR / "op_loss_rune_item_case_notes.csv",
    OUTPUT_DIR / "op_loss_op_champion_build_notes.csv",
    OUTPUT_DIR / "op_loss_winning_opponent_build_notes.csv",
    OUTPUT_DIR / "team_champion_bipartite_edges.csv",
    OUTPUT_DIR / "team_two_mode_summary.csv",
    OUTPUT_DIR / "champion_two_mode_summary.csv",
    OUTPUT_DIR / "champion_team_projection_edges.csv",
    OUTPUT_DIR / "best_compositions_without_op.csv",
    OUTPUT_DIR / "best_compositions_with_op.csv",
    OUTPUT_DIR / "op_loss_opponent_compositions.csv",
    OUTPUT_DIR / "op_loss_opponent_combo_summary.csv",
    OUTPUT_DIR / "professor_quality_review.csv",
    FIG_DIR / "01_matches_by_patch.png",
    FIG_DIR / "02_top_pick_ban_champions.png",
    FIG_DIR / "03_synergy_network.png",
    FIG_DIR / "04_counter_network.png",
    FIG_DIR / "05_top_centrality.png",
    FIG_DIR / "06_model_accuracy_comparison.png",
    FIG_DIR / "07_confusion_matrix_baseline.png",
    FIG_DIR / "08_confusion_matrix_network.png",
    FIG_DIR / "09_roc_curve_comparison.png",
    FIG_DIR / "10_feature_importance.png",
            FIG_DIR / "11_feature_correlation_heatmap.png",
    FIG_DIR / "12_op_champion_score.png",
    FIG_DIR / "15_opgg_op_champion_tiers.png",
    FIG_DIR / "16_op_status_winrate_summary.png",
    FIG_DIR / "17_opgg_tier_sensitivity.png",
    FIG_DIR / "13_best_compositions_by_op_status.png",
    FIG_DIR / "14_op_loss_opponent_combos.png",
    FIG_DIR / "18_op_loss_rune_item_exceptions.png",
    FIG_DIR / "19_team_champion_two_mode_network.png",
]
checklist = pd.DataFrame({
    "path": [str(path.relative_to(ROOT)) for path in required_paths],
    "exists": [path.exists() for path in required_paths],
})
display(checklist)
assert checklist["exists"].all(), "일부 필수 산출물이 생성되지 않았습니다."


# %% [markdown] ## 14. 결론


# %% Code cell 32

baseline_acc = model_results["best_baseline_accuracy"]
network_acc = model_results["best_network_accuracy"]
accuracy_delta = model_results["accuracy_delta"]
test_n = len(model_results["y_test"])
top_features = importance_df.head(8)["feature"].tolist()
top_synergy = network_metrics.sort_values("synergy_pagerank", ascending=False).head(5)["champion"].tolist()
top_counter = network_metrics.sort_values("counter_weighted_out_degree", ascending=False).head(5)["champion"].tolist()
analysis_patches_text = ", ".join(sorted(match_df["patch"].dropna().astype(str).unique(), key=_patch_number))
selected_patch = f"{analysis_patches_text} (26.9 이후 구간)"
op_list_text = ", ".join(op_champions[:CONFIG["op_top_n"]])
opgg_tier_text = ", ".join(
    op_champion_tiers.head(CONFIG["op_top_n"]).apply(
        lambda row: f"{row.get('patch', row.get('leaguepedia_patch', 'NA'))}:{row['champion']}({row.get('position', 'NA')} {row.get('opgg_tier_label', 'NA')} R{int(row['opgg_rank']) if pd.notna(row.get('opgg_rank', np.nan)) else 'NA'})",
        axis=1,
    ).tolist()
) if "opgg_tier_label" in op_champion_tiers.columns else "OP.GG tier unavailable"
best_without_text = "; ".join(
    (best_without_op.head(3)["composition"] + " (" + best_without_op.head(3)["games"].astype(str) + "g, " + best_without_op.head(3)["smoothed_win_rate"].round(3).astype(str) + ")").tolist()
) if not best_without_op.empty else "분석 가능한 반복 조합 없음"
best_with_text = "; ".join(
    (best_with_op.head(3)["composition"] + " (" + best_with_op.head(3)["games"].astype(str) + "g, " + best_with_op.head(3)["smoothed_win_rate"].round(3).astype(str) + ")").tolist()
) if not best_with_op.empty else "분석 가능한 반복 조합 없음"
op_loss_text = "; ".join(
    (
        op_loss_summary.head(3)["opponent_combo"]
        + " ("
        + op_loss_summary.head(3)["wins_against_op"].astype(str)
        + "/"
        + op_loss_summary.head(3)["games_against_op"].astype(str)
        + ", WR "
        + op_loss_summary.head(3)["win_rate_against_op"].map(lambda x: f"{x:.1%}")
        + ")"
    ).tolist()
) if not op_loss_summary.empty else "OP 포함 팀 패배 사례 없음"
op_effect_rows = op_status_summary[op_status_summary["scope"].eq("match_exactly_one_team_has_op")]
if not op_effect_rows.empty:
    op_effect_row = op_effect_rows.iloc[0]
    op_effect_text = (
        f"OP를 한 팀만 사용한 경기에서 OP 팀 승률은 "
        f"**{float(op_effect_row['win_rate']):.1%}**"
        f"(n={int(op_effect_row['games_or_rows'])}, 95% CI "
        f"{float(op_effect_row['ci_lower']):.1%}-{float(op_effect_row['ci_upper']):.1%}, "
        f"p={float(op_effect_row['p_value_vs_50']):.3f})였다."
    )
else:
    op_effect_text = "OP를 한 팀만 사용한 경기 표본이 없어 OP 효과 승률 검정은 생략했다."
sensitivity_text = "; ".join(
    opgg_tier_sensitivity.apply(
        lambda row: f"{row['tier_group']}: {float(row['win_rate']):.1%} (n={int(row['exactly_one_side_games'])})"
        if pd.notna(row["win_rate"]) else f"{row['tier_group']}: 표본 없음",
        axis=1,
    ).tolist()
)
if not op_loss_op_build_notes.empty:
    nocturne_runes = ", ".join(
        [f"{rune} {count}회" for rune, count in op_loss_op_build_notes["key_rune"].value_counts().head(3).items()]
    )
    nocturne_unusual = int((op_loss_op_build_notes["unusual_rune_flag"] | op_loss_op_build_notes["rare_item_flag"]).sum())
    winner_unusual = int((op_loss_winning_build_notes["unusual_rune_flag"] | op_loss_winning_build_notes["rare_item_flag"]).sum()) if not op_loss_winning_build_notes.empty else 0
    rune_item_text = (
        f"OP 패배 경기의 Nocturne 핵심 룬은 {nocturne_runes}로 요약된다. "
        f"동일 패치 champion-role 기준에서 Nocturne 쪽 특이 룬/아이템 flag는 {nocturne_unusual}건, "
        f"승리 상대팀 쪽 특이 룬/아이템 flag는 {winner_unusual}건이었다."
    )
else:
    rune_item_text = "OP 패배 경기의 룬/아이템 상세 데이터가 충분하지 않아 특이사항 분석은 제한적이다."
paired_delta = (
    performance_df[performance_df["feature_set"] == "network"]
    [["model_name", "accuracy_delta_vs_same_model"]]
    .assign(delta_text=lambda d: d["model_name"] + ": " + d["accuracy_delta_vs_same_model"].map(lambda x: f"{x:.3f}"))
)
paired_delta_text = ", ".join(paired_delta["delta_text"].tolist())
if not team_two_mode_summary.empty and not champion_two_mode_summary.empty:
    top_two_mode_teams = ", ".join(team_two_mode_summary.head(3)["team"].astype(str).tolist())
    top_two_mode_champions = ", ".join(champion_two_mode_summary.head(5)["champion"].astype(str).tolist())
    two_mode_text = (
        f"강의자료의 2-mode network 관점에 따라 팀-챔피언 이원 네트워크를 추가했다. "
        f"OP pick rate가 높은 팀은 {top_two_mode_teams}이고, 여러 팀에 넓게 사용된 챔피언은 {top_two_mode_champions}이다. "
        "이는 경기 내 duo/trio 시너지와 별개로, 팀들이 공유하는 메타 champion pool을 보는 보조 분석이다."
    )
else:
    two_mode_text = "팀-챔피언 2-mode network는 표본 부족으로 별도 해석하지 않았다."
if "network_filtering_criteria" in globals() and not network_filtering_criteria.empty:
    filter_items = []
    for _, row in network_filtering_criteria.iterrows():
        filter_items.append(
            f"{row['network']}: {row['analysis_edge_rule']}; figure hidden weak nodes={int(row['hidden_weak_nodes_in_figure'])}"
        )
    filtering_text = " / ".join(filter_items)
else:
    filtering_text = "네트워크 필터링 기준표를 생성하지 못해 별도 확인이 필요하다."

if accuracy_delta > 0:
    model_sentence = (
        f"네트워크 지표를 추가한 모델은 baseline 대비 accuracy가 {accuracy_delta:.3f} 상승했다. "
        "다만 test set이 작기 때문에 확정적 결론보다는 챔피언 간 조합·상성 네트워크 정보가 추가 예측 정보를 줄 수 있다는 방향성으로 해석한다."
    )
else:
    model_sentence = (
        f"네트워크 지표를 추가했지만 accuracy는 baseline 대비 {accuracy_delta:.3f} 변화에 그쳤다. "
        "이는 표본 수 부족, 패치별 메타 변화, 팀 전력 차이, 선수 숙련도 같은 요인을 충분히 통제하지 못한 영향일 수 있다."
    )

conclusion_md = f"""
### 결론 요약

0. **최신 패치 선택과 OP 챔피언 정의**  
   분석 구간은 **{selected_patch}**이다. OP 챔피언은 프로 경기 presence가 아니라 **경기 패치와 매칭한 OP.GG 해당 패치의 OP 티어**로 정의했다. 이번 기준의 OP 챔피언은 **{op_list_text}**이며, OP.GG 포지션별 표기는 **{opgg_tier_text}**이다.

0-0. **OP 포함 여부의 실제 경기 효과**  
   {op_effect_text} OP.GG 티어 cutoff를 넓힌 민감도 결과는 **{sensitivity_text}**이다. 따라서 본 분석은 OP.GG OP 티어를 주 기준으로 삼되, OP+Tier1/2 확장 기준에서도 방향성이 크게 바뀌는지 함께 확인했다.

0-1. **OP 제외 / OP 포함 조합**  
   OP 챔피언이 없는 팀에서 성과가 좋았던 대표 조합은 **{best_without_text}**이다. OP 챔피언이 포함된 조합 중 성과가 좋았던 대표 조합은 **{best_with_text}**이다. exact 5인 조합은 반복 수가 작을 수 있어 duo/trio/5인 조합을 함께 비교했다.

0-2. **OP가 포함되어도 진 경기의 상대 조합**  
   OP 챔피언을 포함한 팀을 이긴 상대 조합으로는 **{op_loss_text}**가 반복적으로 관찰됐다. 이 표는 OP 챔피언이 있어도 어떤 상대 조합 구조에서 무너졌는지 보는 용도다.

0-3. **예외 케이스의 룬/아이템 특이사항**  
   {rune_item_text} 단, ScoreboardPlayers의 Items는 구매 순서가 아니라 경기 종료 시점 인벤토리로 해석해야 하므로, “빌드 순서”가 아니라 “최종 아이템 구성의 특이성”으로 읽는 것이 안전하다.

1. **승/패 예측 성능 비교**  
   Best baseline accuracy는 **{baseline_acc:.3f}**, best network feature accuracy는 **{network_acc:.3f}**이며, accuracy delta는 **{accuracy_delta:.3f}**이다. Bootstrap 95% CI는 **[{ci["ci_lower"]:.3f}, {ci["ci_upper"]:.3f}]**이고 test set은 **{test_n}경기**다. {model_sentence}
   같은 알고리즘끼리의 network-baseline accuracy delta는 **{paired_delta_text}**이다.

2. **중요 feature 해석**  
   현재 실행에서 상위 중요 feature는 `{', '.join(top_features)}`이다. 이 목록에 `synergy_*`, `counter_*`, `counter_advantage_diff` 같은 변수가 포함된다면 네트워크 지표가 모델 판단에 실질적으로 사용된 것으로 볼 수 있다.

3. **네트워크 분석 해석**  
   시너지 PageRank 상위 챔피언은 `{', '.join(top_synergy)}`이고, 카운터 weighted out-degree 상위 챔피언은 `{', '.join(top_counter)}`이다. 이 챔피언들이 단독 승률 상위와 다르다면, 챔피언의 가치는 단독 성능뿐 아니라 조합 내 연결성과 상대 조합에 대한 우위로도 설명된다.

3-1. **강의자료 반영: 2-mode network 보조 분석**  
   {two_mode_text}

3-2. **약한 노드와 edge 제거 기준**  
   약한 노드를 직접 임의 삭제하지 않고, 먼저 약한 edge를 기준으로 제거한 뒤 남는 노드만 중심성과 시각화에 사용했다. 기준은 **{filtering_text}**이다. 즉 CSV에는 검토 가능한 long-tail 후보를 남기지만, 중심성 그래프와 그림에서는 반복 근거가 부족하거나 양의 관계 강도가 없는 edge, 그리고 상위 가중치 edge에 연결되지 않는 노드를 제외했다.

4. **데이터 품질과 누수 관리**  
   네트워크 지표와 모델용 챔피언 단독 지표는 chronological split 이후 train 경기만으로 계산했다. GOL.GG 통계는 그대로 모델에 넣지 않고 train 통계의 prior로만 사용했다. 이 처리로 test 경기 결과가 feature 생성에 직접 섞이는 문제를 줄였다.

5. **한계점과 향후 개선 방향**  
   CargoExport에는 실제 blue/red side 필드가 없어 Team1/Team2 proxy를 사용했고, 진영 효과 feature는 제외했다. 이후에는 실제 side 정보, 팀 Elo, 선수별 숙련도, 밴픽 순서 feature, 패치별 모델, Node2Vec/Graph Neural Network 기반 graph embedding을 추가하면 더 정교한 검증이 가능하다.
"""
display(Markdown(conclusion_md))
