import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Data preparation
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("cleaned_primeira_liga.csv")  # Replace with your actual CSV path

    df["utc_date"] = pd.to_datetime(df["utc_date"])
    df["month"] = df["utc_date"].dt.to_period("M").apply(lambda r: r.start_time)

    df["conversion_rate"] = df["full_time_home"] * (df["home_or_away"] == "home") + df["full_time_away"] * (df["home_or_away"] == "away")
    df["conversion_rate"] = df["conversion_rate"] / df["shots"].replace([np.inf, -np.inf], np.nan)
    df["shots_on_goal_ratio"] = df["shots_on_goal"] / df["shots"]

    df["defensive_efficiency"] = df["saves"] / df["shots_on_goal"].replace(0, np.nan)
    df["offensive_control"] = (df["corner_kicks"] + df["ball_possession"]) / 2

    df["fair_play_index"] = 1 / (
        df["fouls"] *
        (1 / (df["yellow_cards"] + 1))**2 *
        (1 / (df["yellow_red_cards"] + 1))**3 *
        (1 / (df["red_cards"] + 1))**4
    )

    return df

df = load_and_process_data()




# Streamlit side bar 

team_list = sorted(df["team"].unique())
selected_team = st.sidebar.selectbox("Select Team", team_list)
metric_category = st.sidebar.radio("Metric Type", ["Offensive", "Defensive", "Possession", "Discipline"])


# Main page title
logo_path = f"logos/{selected_team.replace(' ', '_')}.png"

col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo_path, width=80)
with col2:
    st.title(f"{selected_team} - Primeira Liga 2024/2025 Season Analysis")


# Aggregations by month
def monthly_aggregates():
    agg_funcs = {
        "Offensive": {
            "df": df.groupby(["team", "month"]).agg(
                avg_shots=("shots", "mean"),
                avg_conversion_rate=("conversion_rate", lambda x: x.replace([np.inf, -np.inf], np.nan).mean()),
                avg_sog_ratio=("shots_on_goal_ratio", "mean")
            ).reset_index(),
            "metrics": ["avg_shots", "avg_conversion_rate", "avg_sog_ratio"]
        },
        "Defensive": {
            "df": df.groupby(["team", "month"]).agg(
                avg_saves=("saves", "mean"),
                avg_goal_kicks=("goal_kicks", "mean"),
                avg_def_efficiency=("defensive_efficiency", "mean")
            ).reset_index(),
            "metrics": ["avg_saves", "avg_goal_kicks", "avg_def_efficiency"]
        },
        "Possession": {
            "df": df.groupby(["team", "month"]).agg(
                avg_possession=("ball_possession", "mean"),
                avg_corner_kicks=("corner_kicks", "mean"),
                avg_off_control=("offensive_control", "mean")
            ).reset_index(),
            "metrics": ["avg_possession", "avg_corner_kicks", "avg_off_control"]
        },
        "Discipline": {
            "df": df.groupby(["team", "month"]).agg(
                avg_yellow_cards=("yellow_cards", "mean"),
                avg_red_cards=("red_cards", "mean"),
                avg_fair_play_index=("fair_play_index", "mean")
            ).reset_index(),
            "metrics": ["avg_yellow_cards", "avg_red_cards", "avg_fair_play_index"]
        }
    }

    return agg_funcs[metric_category]

# Time series Graph
def plot_time_series(data, metric, ylabel, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    santa_data = data[data["team"] == selected_team]
    others_data = data[data["team"] != selected_team].groupby("month")[metric].mean().reset_index()
    others_data["team"] = "Other Teams"

    plt.plot(others_data["month"], others_data[metric], "--", label="Other Teams", color="grey", linewidth=2)
    plt.plot(santa_data["month"], santa_data[metric], marker='o', label=selected_team, color="orange", linewidth=3)

    # Horizontal lines for averages
    plt.axhline(santa_data[metric].mean(), linestyle=":", color="orange", label=f"{selected_team} Avg: {santa_data[metric].mean():.2f}")
    plt.axhline(others_data[metric].mean(), linestyle=":", color="grey", label=f"Others Avg: {others_data[metric].mean():.2f}")

    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Radar plots
def radar_plot(df, features, title):
    team_df = df[df["team"] == selected_team]
    home_vals = team_df[team_df["home_or_away"] == "home"][features].mean().tolist()
    away_vals = team_df[team_df["home_or_away"] == "away"][features].mean().tolist()
    home_vals += [home_vals[0]]
    away_vals += [away_vals[0]]
    theta = features + [features[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=home_vals, theta=theta, fill='toself', name='Home'))
    fig.add_trace(go.Scatterpolar(r=away_vals, theta=theta, fill='toself', name='Away'))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, max(home_vals + away_vals)])),
        showlegend=True
    )
    st.plotly_chart(fig)

# Bubble plot
def bubble_plot(df):
    team_df = df[df["team"] == selected_team]
    fig = px.scatter(
        team_df,
        x='ball_possession',
        y='shots',
        size='goals_scored',
        color='conversion_rate',
        hover_data=['matchday', 'winner', 'home_or_away'],
        color_continuous_scale='Viridis',
        size_max=30,
        title=f"{selected_team}: Possession vs Shots (Size = Goals, Color = Conversion Rate)"
    )
    fig.update_layout(xaxis_title="Ball Possession (%)", yaxis_title="Shots")
    st.plotly_chart(fig)

# Scatter plots
def result_plot(df):
    st_df = df[df["team"] == selected_team].copy()
    st_df['result'] = st_df.apply(lambda row:
        'Win' if (row['winner'] == 'HOME_TEAM' and row['home_or_away'] == 'home') or
                 (row['winner'] == 'AWAY_TEAM' and row['home_or_away'] == 'away')
        else 'Draw' if row['winner'] == 'DRAW'
        else 'Loss', axis=1)

    fig = px.scatter(
        st_df, x="ball_possession", y="shots",
        facet_col="result", trendline="lowess",
        color="result",
        hover_data=["matchday", "goals_scored"],
        labels={"ball_possession": "Possession in %", "shots": "Shots"}
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(title="Possession % vs. Shots by Match Result", height=400)
    fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
    st.plotly_chart(fig)

# Main dashboard
agg = monthly_aggregates()
data = agg["df"]
metrics =  agg["metrics"]


st.title(f" {metric_category} Analysis")

for metric in metrics:
    plot_time_series(data, metric, ylabel=metric.replace("_", " ").title(), title=f"{metric_category} - {metric.replace('_', ' ').title()}")

if metric_category == "Offensive":
    radar_plot(df, ["shots", "shots_on_goal", "shots_off_goal", "goals_scored", "corner_kicks"], f"{selected_team}: Offensive Radar (Home vs Away)")
    bubble_plot(df)
    result_plot(df)

if metric_category == "Defensive":
    radar_plot(df, ["saves", "fouls", "goal_kicks", "offsides"], f"{selected_team}: Defensive Radar (Home vs Away)")

st_df = df[df["team"] == selected_team]

matchday_range = st.slider("Filter by Matchday", int(df["matchday"].min()), int(df["matchday"].max()), (1, 34))
st_df = st_df[st_df["matchday"].between(*matchday_range)]


if st.checkbox(f"Show match dataframe of  {selected_team}"):
    st.dataframe(st_df)

csv = st_df.to_csv(index=False).encode('utf-8')
st.download_button(f"📥 Download {selected_team} data as CSV", csv, f"{selected_team}_data.csv", "text/csv")


with st.expander("Click to know more about this App!"):
    st.markdown("""
    This dashboard allows users to analyze Primeira Liga 2024/2025 season team performance  across multiple categories:
    
    Such as offensive, defensive, ball possession and defensive stats and much more!

    Users can compare their favorite team performance over time, between home and away games, and see how results influence key metrics.
    """)

