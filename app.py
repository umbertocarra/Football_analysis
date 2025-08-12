import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# Carica il tuo dataset (adatta il path o il modo in cui carichi il df)
df = pd.read_csv("fifa_players_final_named_clusters.csv")  


cluster_cols = [
    "Defending",
    "Physical_Mobility",
    "Finishing_Attacking",
    "Static_Physical_Reactivity",
    "Technique_Creativity",
    "Aggression_Aerial"
]

st.title("Giocatori simili - filtro interattivo")


# Crea colonna birth_year se manca
if "birth_year" not in df.columns:
    if "birth_date" in df.columns:
        df["birth_year"] = pd.to_datetime(df["birth_date"], errors="coerce").dt.year
    else:
        df["birth_year"] = np.nan

# Pulizia colonne numeriche
def to_numeric_safe(series):
    s = series.astype(str).str.replace(r'[^\d\.-]', '', regex=True)
    return pd.to_numeric(s, errors='coerce')

df["value_euro"] = to_numeric_safe(df["value_euro"])
df["overall_rating"] = to_numeric_safe(df["overall_rating"])
for c in cluster_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Se cluster in 0-1, scala a 0-100
if df[cluster_cols].max().max() <= 1.0:
    df[cluster_cols] = df[cluster_cols] * 100

df[cluster_cols] = df[cluster_cols].fillna(df[cluster_cols].mean())

# Funzione per estrarre tutte le posizioni uniche nel dataset
def extract_positions_set(series):
    pos_set = set()
    for x in series.dropna().astype(str):
        parts = re.split(r'[,\;/\|]', x)
        for p in parts:
            p = p.strip()
            if p:
                pos_set.add(p)
    return sorted(pos_set)

# Funzione filtro posizione che verifica intersezione con lista scelta
def posizione_filter_multi(series, scelte):
    scelte_set = set(scelte)
    def match_pos(pos_str):
        if pd.isna(pos_str):
            return False
        pos_list = [p.strip() for p in re.split(r'[,\;/\|]', str(pos_str)) if p.strip()]
        return len(scelte_set.intersection(pos_list)) > 0
    return series.apply(match_pos)

# Setup filtri slider con valori di default robusti
def min_max_default(series, fallback_min, fallback_max):
    s = series.dropna()
    if s.empty:
        return fallback_min, fallback_max, (fallback_min, fallback_max)
    mn = int(s.min())
    mx = int(s.max())
    return mn, mx, (mn, mx)

birth_min_global, birth_max_global, birth_default = min_max_default(df["birth_year"], 1980, 2005)
rating_min_global, rating_max_global, rating_default = min_max_default(df["overall_rating"], 0, 100)
value_min_global, value_max_global, value_default = min_max_default(df["value_euro"], 0, 150_000_000)

# --- 1. Select giocatore su tutto il dataset ---
nomi_totali = sorted(df["name"].dropna().unique())
player_name = st.selectbox("Seleziona giocatore", options=nomi_totali)

if not player_name:
    st.warning("Seleziona un giocatore per procedere.")
    st.stop()

selected_player = df[df["name"] == player_name].iloc[0]

# --- 2. Mostro filtri posizione, anno, rating, valore per filtrare gli altri giocatori ---
posizioni_disponibili = extract_positions_set(df["positions"])
posizioni_scelte = st.multiselect(
    "Seleziona posizione",
    posizioni_disponibili,
    default=posizioni_disponibili
)

birth_min, birth_max = st.slider(
    "Range anno di nascita",
    birth_min_global, birth_max_global,
    value=(1980, 2005)
)

rating_min, rating_max = st.slider(
    "Range rating",
    rating_min_global, rating_max_global,
    value=(50, 99)
)

value_min, value_max = st.slider(
    "Range valore (€)",
    value_min_global, value_max_global,
    value=(0, 150_000_000), step=1_000_000
)

# --- 3. Filtra il dataframe (escludendo il giocatore selezionato) secondo i filtri ---
mask_pos = posizione_filter_multi(df["positions"], posizioni_scelte)
mask_birth = df["birth_year"].between(birth_min, birth_max, inclusive="both")
mask_rating = df["overall_rating"].between(rating_min, rating_max, inclusive="both")
mask_value = df["value_euro"].between(value_min, value_max, inclusive="both")

filtered_df = df[mask_pos & mask_birth & mask_rating & mask_value & (df["name"] != player_name)].copy()

st.info(f"Giocatori totali: {len(df)} — dopo filtro: {len(filtered_df)}")

# --- 4. Calcolo similarità e ordino risultati per valore decrescente ---
if filtered_df.empty:
    st.warning("Nessun altro giocatore disponibile con questi filtri.")
else:
    v_sel = selected_player[cluster_cols].values.reshape(1, -1)
    X_sim = filtered_df[cluster_cols].values
    similarities = cosine_similarity(v_sel, X_sim)[0]
    filtered_df["similarity"] = similarities
    top_sim = filtered_df.sort_values("similarity", ascending=False).head(3)
    top_sim = top_sim.sort_values("value_euro", ascending=False)

# Funzione per disegnare radar chart
def radar_chart(player_name, values, labels, color="blue"):
    import numpy as np
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    vals = values + values[:1]
    angs = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angs, vals, color=color, linewidth=2)
    ax.fill(angs, vals, color=color, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title(player_name, size=12, color=color, y=1.08)
    return fig

# --- 5. Visualizzo il profilo del giocatore selezionato ---
st.subheader(f"Profilo di {player_name}")
values_sel = [float(selected_player[c]) for c in cluster_cols]
fig_sel = radar_chart(player_name, values_sel, cluster_cols, color="blue")
st.pyplot(fig_sel)
st.markdown(
    f"**Posizione:** {selected_player['positions']}  \n"
    f"**Anno nascita:** {int(selected_player['birth_year']) if pd.notna(selected_player['birth_year']) else 'N/A'}  \n"
    f"**Rating:** {int(selected_player['overall_rating']) if pd.notna(selected_player['overall_rating']) else 'N/A'}  \n"
    f"**Valore:** €{int(selected_player['value_euro']):,}"
)

# --- 6. Visualizzo i top 3 simili ordinati per valore ---
if not filtered_df.empty and not top_sim.empty:
    st.subheader("Top 3 giocatori simili")
    cols = st.columns(3)
    colors = ["green", "orange", "red"]
    for i, (_, row) in enumerate(top_sim.iterrows()):
        with cols[i]:
            vals = [float(row[c]) for c in cluster_cols]
            fig = radar_chart(row["name"], vals, cluster_cols, color=colors[i])
            st.pyplot(fig)
            st.markdown(
                f"**Posizione:** {row['positions']}  \n"
                f"**Anno nascita:** {int(row['birth_year']) if pd.notna(row['birth_year']) else 'N/A'}  \n"
                f"**Rating:** {int(row['overall_rating']) if pd.notna(row['overall_rating']) else 'N/A'}  \n"
                f"**Valore:** €{int(row['value_euro']):,}  \n"
                f"**Similarità:** {row['similarity']:.2f}"
            )
else:
    st.info("Non ci sono giocatori simili da mostrare (dopo i filtri).")