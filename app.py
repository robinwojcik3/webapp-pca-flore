"""
Web-app Streamlit : PCA + clustering botaniques (v6)
---------------------------------------------------
• Reconnaît les noms saisis au format « Genre épithète » même si la base contient les auteurs.
• Déduplication stricte côté référence et côté saisie.
• **NOUVEAUTÉ v6 :** suppression interactive d’une espèce directement sur le nuage PCA :
  – clic sur un point ➡ ouverture d’un modal de confirmation ➡ relance complète de l’analyse sans l’espèce.
  – la liste des espèces exclues reste visible et peut être vidée.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import linkage
from scipy.spatial import ConvexHull
from streamlit_plotly_events import plotly_events  # ← capture des clics Plotly

import core  # Assurez-vous que core.py est accessible

# --------------------------------------------------------------------------- #
# Configuration UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="PCA flore interactive", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Analyse interactive des clusters botaniques</h1>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# États persistants Streamlit
# --------------------------------------------------------------------------- #
st.session_state.setdefault("removed_species", set())  # noms bruts exclus
st.session_state.setdefault("pending_delete", None)    # candidat à suppression

# --------------------------------------------------------------------------- #
# Chargement de la base
# --------------------------------------------------------------------------- #
@st.cache_data
def load_data(file_path: str = "data_ref.csv") -> pd.DataFrame:
    """Charge les données de référence à partir du chemin fourni."""
    try:
        return core.read_reference(file_path)
    except FileNotFoundError:
        st.error(
            f"ERREUR CRITIQUE : Fichier de données '{file_path}' non trouvé. L'application ne peut pas fonctionner."
        )
        return pd.DataFrame()
    except Exception as err:  # noqa: BLE001
        st.error(
            f"ERREUR CRITIQUE : Impossible de charger les données depuis '{file_path}' : {err}"
        )
        return pd.DataFrame()


ref = load_data()
ref_binom_series: pd.Series = pd.Series(dtype="str")
if not ref.empty:
    ref_binom_series = (
        ref["Espece"].str.split().str[:2].str.join(" ").str.lower()
    )

# --------------------------------------------------------------------------- #
# Layout principal
# --------------------------------------------------------------------------- #
col_input, col_pca_plot = st.columns([1, 3])

with col_input:
    st.subheader("CORTEGE")

    n_clusters_selected = st.slider(
        "Nombre de clusters", min_value=2, max_value=8, value=3, disabled=ref.empty
    )

    species_txt = st.text_area(
        "Liste d'espèces (une par ligne)",
        height=250,
        placeholder="Teucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus\n…",
        disabled=ref.empty,
    )

    # Parsing + exclusion interactive
    species_raw_unique = sorted({s.strip() for s in species_txt.splitlines() if s.strip()})
    species_raw_unique = [
        s for s in species_raw_unique if s not in st.session_state["removed_species"]
    ]
    species_binom_user_unique = [" ".join(s.split()[:2]).lower() for s in species_raw_unique]

    if st.session_state["removed_species"]:
        st.info(
            "Espèces exclues actuellement : "
            + ", ".join(sorted(st.session_state["removed_species"]))
        )
        if st.button("Ré-initialiser la liste exclue"):
            st.session_state["removed_species"].clear()
            st.experimental_rerun()

    run = st.button("Lancer l'analyse", type="primary", disabled=ref.empty)

# Variables globales d'affichage
fig_pca: go.Figure | None = None
fig_dend: go.Figure | None = None
vip_styled = None
cluster_compositions: list[dict[str, object]] = []

# --------------------------------------------------------------------------- #
# Analyse (déclenchée par le bouton)
# --------------------------------------------------------------------------- #
if run and not ref.empty:
    # -- Contrôles préalables -------------------------------------------------
    if not species_binom_user_unique:
        st.error("Veuillez saisir au moins un nom d'espèce (hors exclusions).")
        st.stop()

    # -- Déduplication côté référence ----------------------------------------
    indices_ref: list[int] = []
    if not ref_binom_series.empty:
        tmp = ref_binom_series.reset_index(names="idx").to_frame("binom")
        for b in species_binom_user_unique:
            matches = tmp[tmp["binom"] == b]
            if not matches.empty:
                indices_ref.append(matches["idx"].iloc[0])

    sub = ref.loc[indices_ref].copy() if indices_ref else pd.DataFrame(columns=ref.columns)

    # Espèces non trouvées ----------------------------------------------------
    found_binoms = (
        sub["Espece"].str.split().str[:2].str.join(" ").str.lower().tolist()
        if not sub.empty
        else []
    )
    not_found = [
        species_raw_unique[i]
        for i, b in enumerate(species_binom_user_unique)
        if b not in found_binoms
    ]
    if not_found:
        with col_input:
            st.warning("Non trouvées dans la base : " + ", ".join(not_found), icon="⚠️")

    # Vérifications ­---------------------------------------------------------
    if sub.empty:
        st.error("Aucune espèce utilisable après déduplication.")
        st.stop()
    if sub.shape[0] < n_clusters_selected:
        st.error(
            f"{sub.shape[0]} espèce(s) unique(s) trouvée(s) < nombre de clusters demandé ({n_clusters_selected})."
        )
        st.stop()
    if sub.shape[0] < 2:
        st.error("Au moins 2 espèces uniques sont nécessaires pour l'analyse PCA.")
        st.stop()

    # Mapping Raw ↔ Ref -------------------------------------------------------
    map_raw_by_binom = {
        " ".join(s.split()[:2]).lower(): s for s in species_raw_unique
    }

    # -- Analyse principale ---------------------------------------------------
    try:
        labels, pca, coords, X = core.analyse(sub, n_clusters_selected)

        pdf = pd.DataFrame(coords, columns=[f"PC{i + 1}" for i in range(coords.shape[1])])
        pdf["Cluster"] = labels.astype(str)
        pdf["Espece_Ref"] = sub["Espece"].values
        pdf["Espece_User"] = pdf["Espece_Ref"].apply(
            lambda full: map_raw_by_binom.get(" ".join(full.split()[:2]).lower(), full)
        )

        # -- Nuage PCA --------------------------------------------------------
        color_seq = px.colors.qualitative.Plotly
        fig_pca = px.scatter(
            pdf,
            x="PC1",
            y="PC2" if coords.shape[1] > 1 else None,
            color="Cluster",
            text="Espece_User",
            template="plotly_dark",
            height=600,
            color_discrete_sequence=color_seq,
        )
        fig_pca.update_traces(
            textposition="top center",
            marker=dict(opacity=0.75, size=10),
            hovertemplate="<b>%{text}</b><extra></extra>",
        )

        # Enveloppes convexes (si 2D) ---------------------------------------
        if coords.shape[1] > 1:
            color_map = {
                c: color_seq[i % len(color_seq)] for i, c in enumerate(sorted(pdf["Cluster"].unique()))
            }
            for c in sorted(pdf["Cluster"].unique()):
                pts = pdf[pdf["Cluster"] == c][["PC1", "PC2"]].drop_duplicates().values
                if pts.shape[0] >= 3:
                    hull = ConvexHull(pts)
                    hpts = pts[hull.vertices]
                    fig_pca.add_trace(
                        go.Scatter(
                            x=np.append(hpts[:, 0], hpts[0, 0]),
                            y=np.append(hpts[:, 1], hpts[0, 1]),
                            mode="lines",
                            fill="toself",
                            line=dict(color=color_map[c], width=1.2),
                            fillcolor=color_map[c],
                            opacity=0.2,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
        fig_pca.update_layout(title_text="Clusters d'espèces (PCA)", title_x=0.5)

        # -- Dendrogramme -----------------------------------------------------
        fig_dend = None
        if X.shape[0] > 1:
            Z = linkage(X, method="ward")
            thr = Z[-(n_clusters_selected - 1), 2] * 0.99 if n_clusters_selected > 1 else 0
            fig_dend = ff.create_dendrogram(
                X,
                orientation="left",
                labels=pdf["Espece_User"].tolist(),
                linkagefun=lambda _: Z,
                color_threshold=thr,
                colorscale=color_seq,
            )
            fig_dend.update_layout(
                template="plotly_dark",
                height=max(650, sub.shape[0] * 20),
                title_text="Dendrogramme",
                title_x=0.5,
            )

        # -- VIP --------------------------------------------------------------
        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings**2).sum(axis=1)
        vip_df = (
            pd.DataFrame(
                {
                    "Variable": sub.columns[1:],
                    "Communalité (%)": (communal * 100).round(0).astype(int),
                }
            )
            .sort_values("Communalité (%)", ascending=False)
            .reset_index(drop=True)
        )
        vip_styled = vip_df.style.set_properties(
            **{"text-align": "center"}
        ).format({"Communalité (%)": "{:}%"})

        # -- Composition des clusters ----------------------------------------
        cluster_compositions = [
            {
                "cluster_label": c,
                "count": len(pdf.loc[pdf["Cluster"] == c, "Espece_User"].unique()),
                "species_list": sorted(pdf.loc[pdf["Cluster"] == c, "Espece_User"].unique()),
            }
            for c in sorted(pdf["Cluster"].unique())
        ]

    except Exception as err:  # noqa: BLE001
        st.error(f"Une erreur est survenue lors de l'analyse : {err}")
        st.exception(err)
        st.stop()

# --------------------------------------------------------------------------- #
# Affichage interactif & capture des clics
# --------------------------------------------------------------------------- #
with col_pca_plot:
    if fig_pca:
        click_data = plotly_events(
            fig_pca,
            click_event=True,
            hover_event=False,
            override_height=600,
            override_width="100%",
        )
        # clic ⇒ ouverture du modal de confirmation --------------------------
        if click_data:
            sp_clicked = click_data[0].get("text")  # texte du point (Espece_User)
            if sp_clicked:
                st.session_state["pending_delete"] = sp_clicked
                st.experimental_rerun()
    elif run and ref.empty:
        st.warning("Veuillez charger des données pour afficher le graphique PCA.")
    elif run and not species_binom_user_unique:
        pass
    elif run:
        st.info("Le graphique PCA sera affiché ici après une analyse réussie.")

# -- Modal de confirmation ---------------------------------------------------
if st.session_state["pending_delete"]:
    sp = st.session_state["pending_delete"]
    with st.modal("Supprimer l'espèce ?"):
        st.write(f"Supprimer définitivement **{sp}** de l'analyse ?")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Oui, supprimer", key="confirm_del"):
                st.session_state["removed_species"].add(sp)
                st.session_state["pending_delete"] = None
                st.experimental_rerun()
        with col_no:
            if st.button("Annuler", key="cancel_del"):
                st.session_state["pending_delete"] = None

# --------------------------------------------------------------------------- #
# Reste des affichages (variables, clusters, dendro)
# --------------------------------------------------------------------------- #
col_vars, col_comp = st.columns([1, 2])

with col_vars:
    st.subheader("Importance des Variables")
    if vip_styled is not None:
        st.write(vip_styled.to_html(escape=False), unsafe_allow_html=True)
    elif run:
        st.info("Le tableau d'importance des variables sera affiché ici.")

with col_comp:
    st.subheader("Composition des Clusters")
    if cluster_compositions:
        cols = st.columns(len(cluster_compositions))
        for col, comp in zip(cols, cluster_compositions):
            with col:
                st.markdown(f"**Cluster {comp['cluster_label']}** ({comp['count']} espèces)")
                for sp in comp["species_list"]:
                    st.markdown(f"- {sp}")
    elif run:
        st.info("La composition des clusters sera affichée ici.")

if fig_dend:
    st.plotly_chart(fig_dend, use_container_width=True)
elif run and not ref.empty and species_binom_user_unique:
    st.info("Le dendrogramme n'a pas pu être généré (≥2 espèces uniques requises).")
elif run and ref.empty:
    st.warning("Veuillez charger des données pour afficher le dendrogramme.")

if not run and not ref.empty:
    with col_pca_plot:
        st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse'.")
