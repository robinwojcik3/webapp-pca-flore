"""
Web-app Streamlit : PCA + clustering botaniques
Reconnaît les noms saisis au format « Genre épithète » même si la base
contient les auteurs (« Acacia mearnsii » ↔ « Acacia mearnsii De Wild. »)
Modifications v4:
- Centrage de la colonne 'Communalité (%)' dans le tableau d'importance.
- Affichage de la composition des clusters en colonnes.
- Utilisation des noms d'espèces saisis par l'utilisateur dans la composition des clusters.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
from scipy.spatial import ConvexHull
import numpy as np

# Assurez-vous que le fichier core.py est dans le même répertoire ou accessible
import core

# ---------------------------------------------------------------------------- #
# CONFIGURATION UI
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="PCA flore interactive", layout="wide")
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
col_input, col_pca_plot = st.columns([1, 3])

with col_input:
    st.subheader("CORTEGE")
    n_clusters_selected = st.slider("Nombre de clusters", 2, 8, 3, key="n_clusters_slider", disabled=ref.empty) # Renamed to avoid conflict
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
vip_styled = None # Pour le DataFrame stylé
cluster_compositions_data = [] # Pour stocker les données de composition

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

    not_found_in_ref = [
        s_raw for s_raw, s_bin in zip(species_raw, species_binom)
        if s_bin not in ref_binom.values
    ]
    if not_found_in_ref: # Renamed variable for clarity
        with col_input:
            st.warning(
                "Non trouvées dans la base : " + ", ".join(not_found_in_ref),
                icon="⚠️"
            )

    if sub.shape[0] < n_clusters_selected:
        st.error(f"Le nombre d'espèces trouvées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters_selected}).")
        st.stop()
    
    if sub.shape[0] < 2:
        st.error(f"Au moins 2 espèces sont nécessaires pour l'analyse PCA. {sub.shape[0]} espèce(s) trouvée(s).")
        st.stop()
    
    min_points_for_hull = 3

    # Créer un mappage des noms binomiaux (issus des noms complets de la base) vers les noms bruts saisis par l'utilisateur
    # Ce mappage sera utilisé pour afficher les noms saisis par l'utilisateur dans la composition des clusters.
    # On ne peut le faire qu'avec les espèces effectivement trouvées et présentes dans `sub`.
    
    # D'abord, un mappage des noms binomiaux saisis par l'utilisateur vers les noms bruts saisis.
    user_input_binom_to_raw_map = {
        " ".join(s.split()[:2]).lower(): s for s in species_raw
    }

    try:
        labels, pca, coords, X = core.analyse(sub, n_clusters_selected)

        pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
        pdf["Cluster"] = labels.astype(str)
        pdf["Espece_Ref"] = sub["Espece"].values # Nom complet de la base de données

        # Mapper Espece_Ref vers le nom saisi par l'utilisateur
        def get_user_input_name(full_ref_name):
            binom_ref_name = " ".join(full_ref_name.split()[:2]).lower()
            return user_input_binom_to_raw_map.get(binom_ref_name, full_ref_name) # Fallback au nom ref si non trouvé

        pdf["Espece_User"] = pdf["Espece_Ref"].apply(get_user_input_name)


        color_sequence = px.colors.qualitative.Plotly 
        
        fig_pca = px.scatter(
            pdf,
            x="PC1",
            y="PC2" if coords.shape[1] > 1 else None,
            color="Cluster",
            text="Espece_User", # Utiliser le nom saisi par l'utilisateur pour le texte sur le graphique
            template="plotly_dark",
            height=600,
            color_discrete_sequence=color_sequence
        )

        fig_pca.update_traces(
            textposition="top center",
            marker=dict(opacity=0.7),
            hovertemplate="<b>%{text}</b><extra></extra>"
        )
        
        unique_clusters = sorted(pdf["Cluster"].unique())
        cluster_color_map = {cluster_label: color_sequence[i % len(color_sequence)] for i, cluster_label in enumerate(unique_clusters)}

        if coords.shape[1] > 1:
            for i, cluster_label in enumerate(unique_clusters):
                cluster_points = pdf[pdf["Cluster"] == cluster_label][["PC1", "PC2"]].values
                if len(cluster_points) >= min_points_for_hull:
                    try:
                        hull = ConvexHull(cluster_points)
                        hull_points = cluster_points[hull.vertices]
                        path_x = np.append(hull_points[:, 0], hull_points[0, 0])
                        path_y = np.append(hull_points[:, 1], hull_points[0, 1])

                        fig_pca.add_trace(go.Scatter(
                            x=path_x,
                            y=path_y,
                            fill="toself",
                            fillcolor=cluster_color_map[cluster_label],
                            line=dict(color=cluster_color_map[cluster_label], width=1.5),
                            mode='lines',
                            name=f'Cluster {cluster_label} Hull',
                            opacity=0.2,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    except Exception as e_hull:
                        print(f"Could not generate convex hull for cluster {cluster_label}: {e_hull}")
                elif len(cluster_points) > 0:
                     print(f"Cluster {cluster_label} has less than {min_points_for_hull} points, skipping hull.")

        fig_pca.update_layout(
            title_text="Clusters d'espèces (PCA)",
            title_x=0.5,
            legend_title_text='Cluster'
        )

        if X.shape[0] > 1:
            Z = linkage(X, method="ward")
            dynamic_color_threshold = 0
            if n_clusters_selected > 1 and (n_clusters_selected -1) <= Z.shape[0] :
                idx_threshold = -(n_clusters_selected - 1)
                if idx_threshold == 0: 
                    dynamic_color_threshold = Z[0, 2] / 2 
                elif Z.shape[0] >= (n_clusters_selected -1) and (n_clusters_selected -1) > 0:
                     dynamic_color_threshold = Z[-(n_clusters_selected-1), 2] * 0.99 

            fig_dend = ff.create_dendrogram(
                X,
                orientation="left",
                labels=pdf["Espece_User"].tolist(), # Utiliser les noms saisis par l'utilisateur pour les étiquettes du dendrogramme
                linkagefun=lambda _: Z,
                color_threshold=dynamic_color_threshold if n_clusters_selected > 1 else 0,
                colorscale=color_sequence
            )
            fig_dend.update_layout(
                template="plotly_dark",
                height=max(650, sub.shape[0] * 20),
                title_text="Dendrogramme",
                title_x=0.5
            )
        else:
            fig_dend = None

        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings**2).sum(axis=1)
        vip_data_df = pd.DataFrame({ # Renamed to avoid conflict with styled version
            "Variable": sub.columns[1:],
            "Communalité (%)": (communal * 100).round(0).astype(int),
        }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
        
        # Appliquer le style pour centrer la colonne Communalité
        vip_styled = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)'])\
                                   .format({"Communalité (%)": "{:}%"})


        cluster_compositions_data = []
        for c_label in sorted(pdf["Cluster"].unique()):
            # Utiliser Espece_User pour la liste des espèces
            esp_user_names = pdf.loc[pdf["Cluster"] == c_label, "Espece_User"].tolist()
            cluster_compositions_data.append({
                "cluster_label": c_label,
                "count": len(esp_user_names),
                "species_list": esp_user_names # Liste de noms, pas une chaîne jointe
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

col_vars, col_cluster_comp_container = st.columns([1, 2]) # col_cluster_comp renommé pour éviter confusion

with col_vars:
    st.subheader("Importance des Variables")
    if vip_styled is not None: # Vérifier si le DataFrame stylé existe
        st.write(vip_styled.to_html(escape=False), unsafe_allow_html=True)
    elif run:
        st.info("Le tableau d'importance des variables sera affiché ici.")

with col_cluster_comp_container:
    st.subheader("Composition des Clusters")
    if cluster_compositions_data:
        num_clusters_found = len(cluster_compositions_data)
        if num_clusters_found > 0:
            # Créer des colonnes pour chaque cluster
            cluster_cols = st.columns(num_clusters_found)
            for i, comp_data in enumerate(cluster_compositions_data):
                with cluster_cols[i]:
                    st.markdown(f"**Cluster {comp_data['cluster_label']}** ({comp_data['count']} espèces)")
                    for species_name in comp_data['species_list']:
                        st.markdown(f"- {species_name}")
        else:
            st.info("Aucun cluster à afficher.")
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
