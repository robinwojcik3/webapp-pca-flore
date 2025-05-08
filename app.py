"""
Web-app Streamlit : PCA + clustering botaniques
Reconnaît les noms saisis au format « Genre épithète » même si la base
contient les auteurs (« Acacia mearnsii » ↔ « Acacia mearnsii De Wild. »)
Modifications:
- Amélioration de la lisibilité des étiquettes sur le PCA.
- Infobulle personnalisée (nom de l'espèce en gras).
- Mise en évidence des clusters PCA par des enveloppes convexes colorées.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go # Ajouté pour les formes des clusters
from scipy.cluster.hierarchy import linkage
from scipy.spatial import ConvexHull # Ajouté pour les enveloppes convexes
import numpy as np # Ajouté pour les opérations sur les tableaux

# Assurez-vous que le fichier core.py est dans le même répertoire ou accessible
import core

# ---------------------------------------------------------------------------- #
# CONFIGURATION UI
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="PCA flore interactive", layout="wide")
# Titre principal centré
st.markdown("<h1 style='text-align: center;'>Analyse interactive des clusters botaniques</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------- #
# CHARGEMENT DE LA BASE
# ---------------------------------------------------------------------------- #
@st.cache_data
def load_data(file_path="data_ref.csv"):
    """Charge les données de référence à partir du chemin spécifié."""
    try:
        data = core.read_reference(file_path)
        return data
    except FileNotFoundError:
        st.error(f"ERREUR CRITIQUE: Fichier de données '{file_path}' non trouvé. L'application ne peut pas fonctionner.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"ERREUR CRITIQUE: Impossible de charger les données depuis '{file_path}': {e}")
        return pd.DataFrame()

# Initialisation des données de référence
ref = load_data()

ref_binom = pd.Series(dtype='str')
if not ref.empty:
    ref_binom = (
        ref["Espece"]
        .str.split()
        .str[:2]
        .str.join(" ")
        .str.lower()
    )

# ---------------------------------------------------------------------------- #
# LAYOUT DE LA PAGE
# ---------------------------------------------------------------------------- #
col_input, col_pca_plot = st.columns([1, 3]) # Ratio 1/4 (entrées) et 3/4 (PCA)

with col_input:
    st.subheader("CORTEGE")
    n_clusters = st.slider("Nombre de clusters", 2, 8, 3, disabled=ref.empty)
    species_txt = st.text_area(
        "Liste d'espèces (une par ligne)", height=250,
        placeholder="Teucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus\n…",
        disabled=ref.empty
    )
    species_raw = [s.strip() for s in species_txt.splitlines() if s.strip()]
    species_binom = [" ".join(s.split()[:2]).lower() for s in species_raw]
    run = st.button("Lancer l'analyse", type="primary", disabled=ref.empty)

fig_pca = None
fig_dend = None
vip = pd.DataFrame()
cluster_compositions = []

# ---------------------------------------------------------------------------- #
# ANALYSE (déclenchée par le bouton)
# ---------------------------------------------------------------------------- #
if run and not ref.empty:
    if not species_binom:
        st.error("Veuillez saisir au moins un nom d'espèce.")
        st.stop()

    mask = ref_binom.isin(species_binom)
    sub = ref[mask].copy()

    if sub.empty:
        st.error("Aucune des espèces saisies n'a été trouvée dans la base de données.")
        truly_not_found = [s_raw for s_raw, s_bin in zip(species_raw, species_binom) if s_bin not in ref_binom.values]
        if truly_not_found:
            with col_input:
                st.warning(
                    "Espèces non trouvées dans la base de référence : " + ", ".join(truly_not_found),
                    icon="⚠️"
                )
        st.stop()

    not_found = [
        s_raw for s_raw, s_bin in zip(species_raw, species_binom)
        if s_bin not in ref_binom.values
    ]
    if not_found:
        with col_input:
            st.warning(
                "Non trouvées dans la base : " + ", ".join(not_found),
                icon="⚠️"
            )

    if sub.shape[0] < n_clusters:
        st.error(f"Le nombre d'espèces trouvées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters}).")
        st.stop()
    
    if sub.shape[0] < 2:
        st.error(f"Au moins 2 espèces sont nécessaires pour l'analyse PCA. {sub.shape[0]} espèce(s) trouvée(s).")
        st.stop()
    
    # Pour les enveloppes convexes, il faut au moins 3 points par cluster.
    min_points_for_hull = 3

    try:
        labels, pca, coords, X = core.analyse(sub, n_clusters)

        pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
        pdf["Cluster"] = labels.astype(str) # Assurer que Cluster est une chaîne pour la coloration
        pdf["Espece"] = sub["Espece"].values

        # FIGURE PCA
        # Utilisation de la palette de couleurs par défaut de Plotly Express
        color_sequence = px.colors.qualitative.Plotly 
        
        fig_pca = px.scatter(
            pdf,
            x="PC1",
            y="PC2" if coords.shape[1] > 1 else None,
            color="Cluster",
            text="Espece",
            template="plotly_dark",
            height=600,
            color_discrete_sequence=color_sequence # Appliquer la séquence de couleurs
        )

        fig_pca.update_traces(
            textposition="top center",
            marker=dict(opacity=0.7), # Rend les marqueurs un peu transparents pour mieux voir le texte
            hovertemplate="<b>%{text}</b><extra></extra>" # Infobulle personnalisée
        )
        
        # Ajout des enveloppes convexes pour chaque cluster
        # S'assurer que les noms de clusters sont triés pour une attribution de couleur cohérente
        unique_clusters = sorted(pdf["Cluster"].unique())
        cluster_color_map = {cluster_label: color_sequence[i % len(color_sequence)] for i, cluster_label in enumerate(unique_clusters)}

        if coords.shape[1] > 1: # Les enveloppes n'ont de sens qu'en 2D ou plus
            for i, cluster_label in enumerate(unique_clusters):
                cluster_points = pdf[pdf["Cluster"] == cluster_label][["PC1", "PC2"]].values
                if len(cluster_points) >= min_points_for_hull:
                    try:
                        hull = ConvexHull(cluster_points)
                        hull_points = cluster_points[hull.vertices]
                        
                        # Fermer la forme en ajoutant le premier point à la fin
                        path_x = np.append(hull_points[:, 0], hull_points[0, 0])
                        path_y = np.append(hull_points[:, 1], hull_points[0, 1])

                        fig_pca.add_trace(go.Scatter(
                            x=path_x,
                            y=path_y,
                            fill="toself",
                            fillcolor=cluster_color_map[cluster_label],
                            line=dict(color=cluster_color_map[cluster_label], width=1.5), # Couleur et épaisseur du contour
                            mode='lines', # 'lines' pour le contour, pas de marqueurs pour la forme
                            name=f'Cluster {cluster_label} Hull',
                            opacity=0.2, # Opacité du remplissage
                            showlegend=False, # Ne pas montrer les formes dans la légende
                            hoverinfo='skip' # Pas d'infobulle pour la forme
                        ))
                    except Exception as e_hull: # Gérer le cas où ConvexHull échoue (ex: points colinéaires)
                        print(f"Could not generate convex hull for cluster {cluster_label}: {e_hull}")
                        # Optionnel: tracer les points du cluster sans enveloppe ou avec une autre forme
                        # Pour l'instant, on ne fait rien de plus si l'enveloppe échoue.
                elif len(cluster_points) > 0: # Si moins de 3 points, on ne peut pas faire d'enveloppe
                     print(f"Cluster {cluster_label} has less than {min_points_for_hull} points, skipping hull.")


        fig_pca.update_layout(
            title_text="Clusters d'espèces (PCA)",
            title_x=0.5,
            legend_title_text='Cluster'
        )
        # Remettre les traces de scatter au premier plan après avoir ajouté les formes
        # Les traces sont ajoutées dans l'ordre, donc les scatter plots originaux sont déjà "au-dessus"
        # des formes si les formes sont ajoutées en premier.
        # Si on ajoute les formes après, il faut s'assurer que les points scatter sont visibles.
        # Plotly gère cela par défaut (les dernières traces sont au-dessus),
        # mais on peut forcer l'ordre avec fig.data = tuple(nouvel_ordre_des_traces) si besoin.


        # DENDROGRAMME
        if X.shape[0] > 1:
            Z = linkage(X, method="ward")
            dynamic_color_threshold = 0
            if n_clusters > 1 and (n_clusters -1) <= Z.shape[0] :
                idx_threshold = -(n_clusters - 1)
                if idx_threshold == 0: 
                    dynamic_color_threshold = Z[0, 2] / 2 
                elif Z.shape[0] >= (n_clusters -1) and (n_clusters -1) > 0:
                     dynamic_color_threshold = Z[-(n_clusters-1), 2] * 0.99 

            fig_dend = ff.create_dendrogram(
                X,
                orientation="left",
                labels=sub["Espece"].tolist(),
                linkagefun=lambda _: Z,
                color_threshold=dynamic_color_threshold if n_clusters > 1 else 0,
                colorscale=color_sequence # Essayer d'utiliser la même palette
            )
            fig_dend.update_layout(
                template="plotly_dark",
                height=max(650, sub.shape[0] * 20),
                title_text="Dendrogramme",
                title_x=0.5
            )
        else:
            fig_dend = None

        # TABLEAUX DESCRIPTIFS - Importance des variables
        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings**2).sum(axis=1)
        vip_data = {
            "Variable": sub.columns[1:],
            "Communalité (%)": (communal * 100).round(0).astype(int),
        }
        min_len = min(len(vip_data["Variable"]), len(vip_data["Communalité (%)"]))
        vip = (
            pd.DataFrame({
                "Variable": vip_data["Variable"][:min_len],
                "Communalité (%)": vip_data["Communalité (%)"][:min_len]
            })
            .sort_values("Communalité (%)", ascending=False)
            .reset_index(drop=True)
        )
        
        # TABLEAUX DESCRIPTIFS - Composition des clusters
        cluster_compositions = []
        for c_label in sorted(pdf["Cluster"].unique()):
            esp = pdf.loc[pdf["Cluster"] == c_label, "Espece"].tolist()
            cluster_compositions.append({
                "cluster_label": c_label,
                "count": len(esp),
                "species_list": ", ".join(esp)
            })

    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'analyse : {e}")
        st.exception(e) 
        st.stop()

# Affichage des résultats
with col_pca_plot:
    if fig_pca:
        st.plotly_chart(fig_pca, use_container_width=True)
    elif run and ref.empty:
        st.warning("Veuillez d'abord charger des données pour afficher le graphique PCA.")
    elif run and not species_binom:
        pass 
    elif run:
        st.info("Le graphique PCA sera affiché ici après une analyse réussie.")

col_vars, col_cluster_comp = st.columns([1, 2])

with col_vars:
    st.subheader("Importance des Variables")
    if not vip.empty:
        st.dataframe(
            vip,
            use_container_width=True,
            column_config={
                "Communalité (%)": st.column_config.NumberColumn(format="%d%%")
            }
        )
    elif run:
        st.info("Le tableau d'importance des variables sera affiché ici.")

with col_cluster_comp:
    st.subheader("Composition des Clusters")
    if cluster_compositions:
        for comp in cluster_compositions:
            st.markdown(f"**Cluster {comp['cluster_label']}** — {comp['count']} espèces : {comp['species_list']}")
    elif run:
        st.info("La composition des clusters sera affichée ici.")

if fig_dend:
    st.plotly_chart(fig_dend, use_container_width=True)
elif run and not ref.empty and species_binom:
    st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces ou problème de seuil).")
elif run and ref.empty:
    st.warning("Veuillez d'abord charger des données pour afficher le dendrogramme.")

if not run and not ref.empty:
    with col_pca_plot:
        st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse'.")
elif not run and ref.empty:
    pass
