"""
Web‑app Streamlit : PCA + clustering botaniques
Reconnaît les noms saisis au format « Genre épithète » même si la base
contient les auteurs (« Acacia mearnsii » ↔ « Acacia mearnsii De Wild. »)

Modifications v5
---------------
- Correction du problème de duplication des espèces dans les analyses et affichages ;
  assure qu'une seule entrée de la base de référence est utilisée par espèce
  utilisateur unique.
- Maintien des améliorations précédentes (centrage, colonnes pour clusters,
  noms utilisateur).
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

import core  # Assurez‑vous que core.py est accessible

# --------------------------------------------------------------------------- #
# Configuration UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="PCA flore interactive", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Analyse interactive des clusters botaniques</h1>",
    unsafe_allow_html=True,
)

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
            f"ERREUR CRITIQUE : Fichier de données '{file_path}' non trouvé. L'application ne peut pas fonctionner."
        )
        return pd.DataFrame()
    except Exception as err:  # noqa: BLE001
        st.error(
            f"ERREUR CRITIQUE : Impossible de charger les données depuis '{file_path}' : {err}"
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
        "Nombre de clusters",
        min_value=2,
        max_value=8,
        value=3,
        key="n_clusters_slider",
        disabled=ref.empty,
    )

    species_txt = st.text_area(
        "Liste d'espèces (une par ligne)",
        height=250,
        placeholder="Teucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus\n…",
        disabled=ref.empty,
    )

    species_raw_unique = sorted({s.strip() for s in species_txt.splitlines() if s.strip()})
    species_binom_user_unique = [
        " ".join(s.split()[:2]).lower() for s in species_raw_unique
    ]

    run = st.button("Lancer l'analyse", type="primary", disabled=ref.empty)

fig_pca: go.Figure | None = None
fig_dend: go.Figure | None = None
vip_styled = None
cluster_compositions_data: list[dict[str, object]] = []

# --------------------------------------------------------------------------- #
# Analyse (déclenchée par le bouton)
# --------------------------------------------------------------------------- #
if run and not ref.empty:
    if not species_binom_user_unique:
        st.error("Veuillez saisir au moins un nom d'espèce.")
        st.stop()

    # Déduplication côté référence
    indices_to_keep_from_ref: list[int] = []
    if not ref_binom_series.empty:
        ref_indexed_binom = ref_binom_series.reset_index()
        ref_indexed_binom.columns = ["Original_Ref_Index", "ref_binom_val"]

        for user_binom in species_binom_user_unique:
            matches = ref_indexed_binom[ref_indexed_binom["ref_binom_val"] == user_binom]
            if not matches.empty:
                indices_to_keep_from_ref.append(matches["Original_Ref_Index"].iloc[0])

    sub = ref.loc[indices_to_keep_from_ref].copy() if indices_to_keep_from_ref else pd.DataFrame(columns=ref.columns)

    # Vérification des non‑trouvées
    found_binoms = (
        sub["Espece"].str.split().str[:2].str.join(" ").str.lower().tolist()
        if not sub.empty
        else []
    )
    not_found_user_raw_names = [
        species_raw_unique[i]
        for i, user_binom in enumerate(species_binom_user_unique)
        if user_binom not in found_binoms
    ]
    if not_found_user_raw_names:
        with col_input:
            st.warning(
                "Non trouvées dans la base : " + ", ".join(not_found_user_raw_names),
                icon="⚠️",
            )

    if sub.empty:
        st.error(
            "Aucune des espèces saisies n'a pu être utilisée pour l'analyse après déduplication."
        )
        st.stop()

    if sub.shape[0] < n_clusters_selected:
        st.error(
            f"Le nombre d'espèces uniques trouvées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters_selected})."
        )
        st.stop()

    if sub.shape[0] < 2:
        st.error("Au moins 2 espèces uniques sont nécessaires pour l'analyse PCA.")
        st.stop()

    # Mapping Raw↔Ref
    user_input_binom_to_raw = {
        " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_raw_unique
    }

    try:
        labels, pca, coords, X = core.analyse(sub, n_clusters_selected)

        pdf = pd.DataFrame(coords, columns=[f"PC{i + 1}" for i in range(coords.shape[1])])
        pdf["Cluster"] = labels.astype(str)
        pdf["Espece_Ref"] = sub["Espece"].values
        pdf["Espece_User"] = pdf["Espece_Ref"].apply(
            lambda full_name: user_input_binom_to_raw.get(
                " ".join(full_name.split()[:2]).lower(), full_name
            )
        )

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
            marker=dict(opacity=0.7),
            hovertemplate="<b>%{text}</b><extra></extra>",
        )

        # Enveloppes convexes
        if coords.shape[1] > 1:
            cluster_color_map = {
                cl: color_seq[i % len(color_seq)]
                for i, cl in enumerate(sorted(pdf["Cluster"].unique()))
            }
            for cl in sorted(pdf["Cluster"].unique()):
                pts = (
                    pdf[pdf["Cluster"] == cl][["PC1", "PC2"]]
                    .drop_duplicates()
                    .values
                )
                if len(pts) >= 3:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    path_x = np.append(hull_pts[:, 0], hull_pts[0, 0])
                    path_y = np.append(hull_pts[:, 1], hull_pts[0, 1])
                    fig_pca.add_trace(
                        go.Scatter(
                            x=path_x,
                            y=path_y,
                            fill="toself",
                            fillcolor=cluster_color_map[cl],
                            line=dict(color=cluster_color_map[cl], width=1.5),
                            mode="lines",
                            name=f"Cluster {cl} Hull",
                            opacity=0.2,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

        fig_pca.update_layout(
            title_text="Clusters d'espèces (PCA)", title_x=0.5, legend_title_text="Cluster"
        )

        # Dendrogramme
        if X.shape[0] > 1:
            Z = linkage(X, method="ward")
            threshold = (
                Z[-(n_clusters_selected - 1), 2] * 0.99 if n_clusters_selected > 1 else 0
            )
            fig_dend = ff.create_dendrogram(
                X,
                orientation="left",
                labels=pdf["Espece_User"].tolist(),
                linkagefun=lambda _: Z,
                color_threshold=threshold,
                colorscale=color_seq,
            )
            fig_dend.update_layout(
                template="plotly_dark",
                height=max(650, sub.shape[0] * 20),
                title_text="Dendrogramme",
                title_x=0.5,
            )
        else:
            fig_dend = None

        # VIP
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

        # Composition des clusters
        cluster_compositions_data = []
        for cl in sorted(pdf["Cluster"].unique()):
            sp_list = sorted(pdf.loc[pdf["Cluster"] == cl, "Espece_User"].unique())
            cluster_compositions_data.append(
                {
                    "cluster_label": cl,
                    "count": len(sp_list),
                    "species_list": sp_list,
                }
            )

    except Exception as err:  # noqa: BLE001
        st.error(f"Une erreur est survenue lors de l'analyse : {err}")
        st.exception(err)
        st.stop()

# --------------------------------------------------------------------------- #
# Affichage des résultats
# --------------------------------------------------------------------------- #
with col_pca_plot:
    if fig_pca:
        st.plotly_chart(fig_pca, use_container_width=True)
    elif run and ref.empty:
        st.warning("Veuillez d'abord charger des données pour afficher le graphique PCA.")
    elif run and not species_binom_user_unique:
        pass
    elif run:
        st.info("Le graphique PCA sera affiché ici après une analyse réussie.")

col_vars, col_cluster_comp = st.columns([1, 2])

with col_vars:
    st.subheader("Importance des Variables")
    if vip_styled is not None:
        st.write(vip_styled.to_html(escape=False), unsafe_allow_html=True)
    elif run:
        st.info("Le tableau d'importance des variables sera affiché ici.")

with col_cluster_comp:
    st.subheader("Composition des Clusters")
    if cluster_compositions_data:
        cols = st.columns(len(cluster_compositions_data))
        for col, comp in zip(cols, cluster_compositions_data):
            with col:
                st.markdown(f"**Cluster {comp['cluster_label']}** ({comp['count']} espèces)")
                for sp in comp["species_list"]:
                    st.markdown(f"- {sp}")
    elif run:
        st.info("La composition des clusters sera affichée ici.")

if fig_dend:
    st.plotly_chart(fig_dend, use_container_width=True)
elif run and not ref.empty and species_binom_user_unique:
    st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces uniques).")
elif run and ref.empty:
    st.warning("Veuillez d'abord charger des données pour afficher le dendrogramme.")

if not run and not ref.empty:
    with col_pca_plot:
        st.info(
            "Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse'."
        )
