import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="RB Draft Round Probability App",
    layout="wide"
)

# ============================================================
# HELPERS
# ============================================================

def find_round_col(df):
    for c in ["draft_round", "round", "draftRound", "DraftRound"]:
        if c in df.columns:
            return c
    return None


def prepare_data(df, feat):
    round_col = find_round_col(df)
    if round_col is None:
        raise ValueError("No draft round column found in yearly_player_stats_offense.csv.")

    # one draft round per player
    player_round = (
        df[["player_name", round_col]]
        .dropna()
        .drop_duplicates("player_name")
        .copy()
    )

    player_round[round_col] = pd.to_numeric(player_round[round_col], errors="coerce")
    player_round = player_round.dropna(subset=[round_col])
    player_round[round_col] = player_round[round_col].astype(int)

    # check feat columns
    needed = ["player_name", "season", "score"]
    missing = [c for c in needed if c not in feat.columns]
    if len(missing) > 0:
        raise ValueError(f"feat file is missing required columns: {missing}")

    feat_clean = feat[needed].dropna().copy()
    feat_clean["season"] = pd.to_numeric(feat_clean["season"], errors="coerce")
    feat_clean["score"] = pd.to_numeric(feat_clean["score"], errors="coerce")
    feat_clean = feat_clean.dropna(subset=["season", "score"])
    feat_clean["season"] = feat_clean["season"].astype(int)

    # average duplicates within player-season if they exist
    feat_clean = (
        feat_clean.groupby(["player_name", "season"], as_index=False)["score"]
        .mean()
    )

    # build best consecutive 3-year average
    feat3 = feat_clean.copy().sort_values(["player_name", "season"])

    feat3["roll3_avg"] = (
        feat3.groupby("player_name")["score"]
        .rolling(window=3, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    best_3yr = (
        feat3.dropna(subset=["roll3_avg"])
        .groupby("player_name", as_index=False)["roll3_avg"]
        .max()
        .rename(columns={"roll3_avg": "best_consecutive_3yr_avg"})
    )

    best_3yr = best_3yr.merge(player_round, on="player_name", how="inner")

    return feat_clean, best_3yr, round_col


def prob_reach_score_by_round_3yr_avg(best_3yr, round_col, input_score, ci=True):
    s = float(input_score)

    tmp = best_3yr.copy()
    tmp["hit"] = (tmp["best_consecutive_3yr_avg"] >= s).astype(int)

    out = (
        tmp.groupby(round_col)["hit"]
        .agg(count="count", hits="sum", prob="mean")
        .reset_index()
        .sort_values(round_col)
    )

    if ci:
        se = np.sqrt(out["prob"] * (1 - out["prob"]) / out["count"])
        out["ci_lower"] = (out["prob"] - 1.96 * se).clip(0, 1)
        out["ci_upper"] = (out["prob"] + 1.96 * se).clip(0, 1)

    return out


def get_player_score(feat_clean, player_name, season):
    temp = feat_clean.copy()
    temp["player_name_clean"] = temp["player_name"].astype(str).str.strip().str.lower()
    player_name_clean = str(player_name).strip().lower()

    match = temp[
        (temp["player_name_clean"] == player_name_clean) &
        (temp["season"] == int(season))
    ].copy()

    if len(match) == 0:
        return None

    row = match.iloc[0]
    return {
        "player_name": row["player_name"],
        "season": int(row["season"]),
        "score": float(row["score"])
    }


def make_plot(result, round_col, score):
    fig, ax = plt.subplots(figsize=(8, 5))

    x_vals = result[round_col].astype(int).tolist()
    y_vals = result["prob"].tolist()

    ax.bar(x_vals, y_vals)
    ax.set_title(f"Probability of Reaching Value ≥ {score:.3f}")
    ax.set_xlabel("Draft Round")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y")

    mean_prob = result["prob"].mean()
    ax.axhline(mean_prob, linestyle="--", label=f"Mean = {mean_prob:.3f}")
    ax.legend()

    return fig


def clear_results():
    st.session_state["result"] = None
    st.session_state["lookup_message"] = ""
    st.session_state["target_score"] = None


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state["result"] = None

if "lookup_message" not in st.session_state:
    st.session_state["lookup_message"] = ""

if "target_score" not in st.session_state:
    st.session_state["target_score"] = None


# ============================================================
# APP UI
# ============================================================

st.title("RB Draft Round Probability App")
st.write(
    "Use either a target score directly or a player name and year to look up that player's value."
)
st.caption(
    "Probabilities shown = chance that a draft pick reaches at least that value "
    "as their BEST 3-year consecutive average at any point in their career."
)

with st.sidebar:
    st.header("Upload Data")

    offense_file = st.file_uploader(
        "Upload yearly_player_stats_offense.csv",
        type=["csv"]
    )

    feat_file = st.file_uploader(
        "Upload feat.csv",
        type=["csv"]
    )

    st.header("Inputs")

    score_text = st.text_input("Target Score")
    player_text = st.text_input("Player Name")
    year_text = st.text_input("Year")
    show_ci = st.checkbox("Show 95% CI", value=True)

    calculate = st.button("Calculate", use_container_width=True)
    reset = st.button("Reset Results", use_container_width=True)

if reset:
    clear_results()

if offense_file is None or feat_file is None:
    st.info("Upload both CSV files in the sidebar to begin.")
    st.stop()

try:
    df = pd.read_csv(offense_file)
    feat = pd.read_csv(feat_file)
    feat_clean, best_3yr, round_col = prepare_data(df, feat)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if calculate:
    target_score = None
    lookup_message = ""

    if score_text.strip() != "":
        try:
            target_score = float(score_text)
        except ValueError:
            st.error("Target score must be a valid number.")
            st.stop()

        if target_score < 0 or target_score > 10:
            st.error("Target score must be between 0 and 10.")
            st.stop()

        lookup_message = f"Using manually entered target score: {target_score:.3f}"

    else:
        if player_text.strip() == "" or year_text.strip() == "":
            st.error("Enter either a target score, or a player name and year.")
            st.stop()

        try:
            year = int(year_text)
        except ValueError:
            st.error("Year must be an integer.")
            st.stop()

        player_info = get_player_score(feat_clean, player_text, year)

        if player_info is None:
            st.error("Could not find that player and year in the uploaded feat file.")
            st.stop()

        target_score = float(player_info["score"])
        lookup_message = (
            f"Looked up {player_info['player_name']} ({player_info['season']}) "
            f"and found score = {player_info['score']:.3f}"
        )

    result = prob_reach_score_by_round_3yr_avg(
        best_3yr,
        round_col,
        target_score,
        ci=show_ci
    )

    if len(result) == 0:
        st.warning("No results found.")
        st.stop()

    st.session_state["result"] = result
    st.session_state["lookup_message"] = lookup_message
    st.session_state["target_score"] = target_score

if st.session_state["result"] is not None:
    result = st.session_state["result"]
    target_score = st.session_state["target_score"]

    st.info(st.session_state["lookup_message"])

    mean_prob = result["prob"].mean()
    best_row = result.loc[result["prob"].idxmax()]
    best_round = int(best_row[round_col])
    best_prob = float(best_row["prob"])

    st.subheader(
        f"Target value: {target_score:.3f}   |   Mean probability: {mean_prob:.3f}   |   "
        f"Best draft round: {best_round} ({best_prob:.3f})"
    )

    col1, col2 = st.columns([1.05, 1])

    with col1:
        display_df = result.copy()
        display_df[round_col] = display_df[round_col].astype(int)
        display_df["count"] = display_df["count"].astype(int)
        display_df["hits"] = display_df["hits"].astype(int)
        display_df["prob"] = display_df["prob"].round(3)

        if "ci_lower" in display_df.columns and "ci_upper" in display_df.columns:
            display_df["ci_lower"] = display_df["ci_lower"].round(3)
            display_df["ci_upper"] = display_df["ci_upper"].round(3)

        rename_map = {
            round_col: "Round",
            "count": "Players",
            "hits": "Hits",
            "prob": "Probability",
            "ci_lower": "CI Lower",
            "ci_upper": "CI Upper"
        }
        display_df = display_df.rename(columns=rename_map)

        st.markdown("### Round-by-Round Results")
        st.dataframe(display_df, use_container_width=True)

    with col2:
        st.markdown("### Probability by Draft Round")
        fig = make_plot(result, round_col, target_score)
        st.pyplot(fig, use_container_width=True)

else:
    st.write("Upload both CSVs, enter a target score or a player name and year, then click **Calculate**.")