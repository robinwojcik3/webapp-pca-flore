# Web‑app Streamlit : PCA + clustering botaniques
# Reconnaît les noms saisis au format « Genre épithète » même si la base
# contient les auteurs (« Acacia mearnsii » ↔ « Acacia mearnsii De Wild. »)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage
import core

# ----------------------------------------------------------------------------
# CONFIGURATION UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="PCA flore interactive", layout="wide")

# -- Style persiste sur toute la page ---------------------------------------
CUSTOM_CSS = """
<style>
/* couleur de fond et bordure pour le panneau cortège */
.cortege-box {
    background:#f6c7a5;
    border:2px solid #222;
    padding:0.4rem 0.3rem;
    font-size:0.85rem;
}
/* alignement tableau importance */
th, td {text-align:left !important;}

/* supprime la barre horizontale Streamlit entre widgets */
hr {visibility:hidden}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Analyse interactive de clusters botaniques")

# ----------------------------------------------------------------------------
# CHARGEMENT DE LA BASE
# ----------------------------------------------------------------------------
uploaded = st.file_uploader("Base de données CSV (optionnel)", type=["csv"])
if uploaded:
    ref = core.read_reference(uploaded)
    st.success(f"{uploaded.name} chargé ({ref.shape[0]} lignes).")
else:
    ref = core.read_reference("data_ref.csv")
    st.info("Fichier local *data_ref.csv* utilisé.")

# pré‑calcul : version « binomiale » (Genre + épithète) en minuscules
ref_binom = (
    ref["Espece"].str.split().str[:2].str.join(" ").str.lower()
)

# ----------------------------------------------------------------------------
# PARAMÈTRES UTILISATEUR (colonne cortège)
# ----------------------------------------------------------------------------
TOP_L, TOP_R = st.columns([1, 9])  # ≈ 10 % / 90 %

with TOP_L:
    st.markdown("<div class='cortege-box'><b>CORTÈGE</b></div>", unsafe_allow_html=True)
    species_txt = st.text_area(
        "", height=550,
        placeholder="Allium schoenoprasum\nAnacamptis palustris\n…",
    )
    species_raw = [s.strip() for s in species_txt.splitlines() if s.strip()]
    species_binom = [" ".join(s.split()[:2]).lower() for s in species_raw]
    n_clusters = st.slider("Nombre de clusters", 2, 8, 3, key="clusters_slider")
    run = st.button("Lancer l'analyse", type="primary")

# ----------------------------------------------------------------------------
# ANALYSE ET AFFICHAGE (colonne droite)
# ----------------------------------------------------------------------------
with TOP_R:

    if run:
        if not species_binom:
            st.error("Veuillez saisir au moins un nom d'espèce.")
            st.stop()

        mask = ref_binom.isin(species_binom)
        sub = ref[mask].copy()

        if sub.empty:
            st.error("Aucune des espèces saisies n'a été trouvée.")
            st.stop()

        # noms non trouvés
        not_found = [
            s for s in species_raw if " ".join(s.split()[:2]).lower() not in ref_binom.values
        ]
        if not_found:
            st.warning("Non trouvées : " + ", ".join(not_found), icon="⚠️")

        if sub.shape[0] < n_clusters:
            st.error(f"{sub.shape[0]} espèces < {n_clusters} clusters.")
            st.stop()

        labels, pca, coords, X = core.analyse(sub, n_clusters)

        # ------------------------------------------------------------------
        # FIGURE PCA
        # ------------------------------------------------------------------
        pdf = pd.DataFrame(coords, columns=[f"PC{{i+1}}" for i in range(coords.shape[1])])
        pdf["Cluster"] = labels.astype(str)
        pdf["Espece"] = sub["Espece"].values

        fig_pca = px.scatter(
            pdf,
            x="PC1",
            y="PC2" if "PC2" in pdf else None,
            color="Cluster",
            text="Espece",
            template="plotly_dark",
            height=600,
        )
        fig_pca.update_traces(textposition="top center")
        st.plotly_chart(fig_pca, use_container_width=True)

        # ------------------------------------------------------------------
        # TABLEAUX (2ᵉ rangée)
        # ------------------------------------------------------------------
        BOT_L, BOT_R = st.columns([3, 7])  # 30 % / 70 %

        # Importance des variables
        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings ** 2).sum(axis=1)
        vip = (
            pd.DataFrame({
                "Variable": sub.columns[1:],
                "Communauté (%)": (communal * 100).round(1),
            })
            .sort_values("Communauté (%)", ascending=False)
            .reset_index(drop=True)
        )
        with BOT_L:
            st.markdown("### Importance des Variables")
            st.dataframe(vip, use_container_width=True, height=450)

        # Composition des clusters
        with BOT_R:
            st.markdown("### Composition des Clusters")
            for c in sorted(set(labels)):
                esp = pdf.loc[pdf["Cluster"] == str(c), "Espece"].tolist()
                st.markdown(f"**Cluster {c}** — {len(esp)} espèces : {', '.join(esp)}")

        # ------------------------------------------------------------------
        # DENDROGRAMME (rangée 3 — pleine largeur)
        # ------------------------------------------------------------------
        st.markdown("### Dendrogramme")
        Z = linkage(X, method="ward")
        fig_dend = ff.create_dendrogram(
            X,
            orientation="left",
            labels=sub["Espece"].tolist(),
            linkagefun=lambda _: Z,
        )
        fig_dend.update_layout(template="plotly_dark", height=650)
        st.plotly_chart(fig_dend, use_container_width=True)
