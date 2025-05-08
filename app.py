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
# CHARGEMENT DE LA BASE DE TRAITS
# ---------------------------------------------------------------------------- #
@st.cache_data
def load_data(file_path="data_ref.csv"):
    """Charge les données de référence (traits) à partir du chemin spécifié."""
    try:
        data = core.read_reference(file_path)
        return data
    except FileNotFoundError:
        st.error(f"ERREUR CRITIQUE: Fichier de données de traits '{file_path}' non trouvé. L'application ne peut pas fonctionner.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"ERREUR CRITIQUE: Impossible de charger les données de traits depuis '{file_path}': {e}")
        return pd.DataFrame()

ref = load_data()

ref_binom_series = pd.Series(dtype='str')
if not ref.empty:
    ref_binom_series = (
        ref["Espece"]
        .str.split()
        .str[:2]
        .str.join(" ")
        .str.lower()
    )

# ---------------------------------------------------------------------------- #
# NOUVEAU: CHARGEMENT DE LA BASE ECOLOGIQUE
# ---------------------------------------------------------------------------- #
@st.cache_data
def load_ecology_data(file_path="data_ecologie_espece.csv"):
    """Charge les données écologiques à partir du chemin spécifié."""
    try:
        # Tente de lire les deux premières colonnes.
        # S'attend à ce que la première colonne soit le nom de l'espèce et la seconde la description.
        eco_data = pd.read_csv(file_path, usecols=[0, 1], encoding='utf-8') # Spécifiez l'encodage si nécessaire
        eco_data.columns = ['Espece', 'Description_Ecologie']  # Assigne des noms de colonnes standards
        
        # Normalise les noms d'espèces: deux premiers mots, minuscules
        eco_data['Espece_norm'] = (
            eco_data['Espece']
            .astype(str)
            .str.split()
            .str[:2]
            .str.join(" ")
            .str.lower()
        )
        # Définit le nom normalisé comme index pour une recherche facile
        eco_data = eco_data.set_index('Espece_norm')
        # Retourne uniquement la colonne de description, indexée par le nom normalisé
        return eco_data[["Description_Ecologie"]]
    except FileNotFoundError:
        st.warning(f"Fichier de données écologiques '{file_path}' non trouvé. Les descriptions écologiques ne seront pas disponibles.")
        return pd.DataFrame()
    except ValueError:
        st.warning(f"Le fichier '{file_path}' ne semble pas avoir deux colonnes ou n'est pas un CSV valide. Les descriptions écologiques ne seront pas disponibles.")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Impossible de charger les données écologiques depuis '{file_path}': {e}. Les descriptions écologiques ne seront pas disponibles.")
        return pd.DataFrame()

ecology_df = load_ecology_data() # Charger les données écologiques

# ---------------------------------------------------------------------------- #
# LAYOUT DE LA PAGE
# ---------------------------------------------------------------------------- #
col_input, col_pca_plot = st.columns([1, 3])

with col_input:
    st.subheader("CORTEGE")
    n_clusters_selected = st.slider("Nombre de clusters", 2, 8, 3, key="n_clusters_slider", disabled=ref.empty)
    species_txt = st.text_area(
        "Liste d'espèces (une par ligne)", height=250,
        placeholder="Teucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus\n…",
        disabled=ref.empty
    )
    species_raw_unique = sorted(list(set(s.strip() for s in species_txt.splitlines() if s.strip())))
    species_binom_user_unique = [" ".join(s.split()[:2]).lower() for s in species_raw_unique]

    run = st.button("Lancer l'analyse", type="primary", disabled=ref.empty)

fig_pca = None
fig_dend = None
vip_styled = None
cluster_compositions_data = []

# ---------------------------------------------------------------------------- #
# ANALYSE (déclenchée par le bouton)
# ---------------------------------------------------------------------------- #
if run and not ref.empty:
    if not species_binom_user_unique:
        st.error("Veuillez saisir au moins un nom d'espèce.")
        st.stop()

    indices_to_keep_from_ref = []
    if not ref_binom_series.empty:
        ref_indexed_binom = ref_binom_series.reset_index()
        ref_indexed_binom.columns = ['Original_Ref_Index', 'ref_binom_val']
        for user_binom_specie in species_binom_user_unique:
            matches_in_ref = ref_indexed_binom[ref_indexed_binom['ref_binom_val'] == user_binom_specie]
            if not matches_in_ref.empty:
                indices_to_keep_from_ref.append(matches_in_ref['Original_Ref_Index'].iloc[0])
    
    if indices_to_keep_from_ref:
        sub = ref.loc[indices_to_keep_from_ref].copy()
    else:
        sub = pd.DataFrame(columns=ref.columns)

    found_ref_binom_values_in_sub = []
    if not sub.empty:
        found_ref_binom_values_in_sub = (
            sub["Espece"]
            .str.split()
            .str[:2]
            .str.join(" ")
            .str.lower()
            .tolist()
        )

    not_found_user_raw_names = []
    for i, user_binom_name in enumerate(species_binom_user_unique):
        if user_binom_name not in found_ref_binom_values_in_sub:
            not_found_user_raw_names.append(species_raw_unique[i])
            
    if not_found_user_raw_names:
        with col_input:
            st.warning(
                "Non trouvées dans la base de traits : " + ", ".join(not_found_user_raw_names),
                icon="⚠️"
            )

    if sub.empty:
        st.error("Aucune des espèces saisies (après déduplication et recherche dans la base de traits) n'a pu être utilisée pour l'analyse.")
        st.stop()

    if sub.shape[0] < n_clusters_selected:
        st.error(f"Le nombre d'espèces uniques trouvées et utilisées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters_selected}).")
        st.stop()
    
    if sub.shape[0] < 2:
        st.error(f"Au moins 2 espèces uniques sont nécessaires pour l'analyse PCA. {sub.shape[0]} espèce(s) trouvée(s) et utilisée(s).")
        st.stop()
    
    min_points_for_hull = 3
    user_input_binom_to_raw_map = {
        " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_raw_unique
    }

    try:
        labels, pca, coords, X = core.analyse(sub, n_clusters_selected)

        pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
        pdf["Cluster"] = labels.astype(str)
        pdf["Espece_Ref"] = sub["Espece"].values # Nom de l'espèce tel qu'il est dans la base de référence (peut inclure auteur)

        def get_user_input_name(full_ref_name):
            binom_ref_name = " ".join(full_ref_name.split()[:2]).lower()
            return user_input_binom_to_raw_map.get(binom_ref_name, full_ref_name)
        pdf["Espece_User"] = pdf["Espece_Ref"].apply(get_user_input_name) # Nom tel que saisi par l'utilisateur (ou le plus proche)

        # NOUVEAU: Ajout des données écologiques à pdf
        if not ecology_df.empty:
            # Normaliser Espece_Ref de pdf pour correspondre à l'index de ecology_df
            pdf['Espece_Ref_norm_for_eco'] = (
                pdf['Espece_Ref'] # Utiliser Espece_Ref car c'est le nom complet de la base
                .astype(str)
                .str.split()
                .str[:2]
                .str.join(" ")
                .str.lower()
            )
            # Joindre (map) les descriptions écologiques
            pdf['Ecologie'] = pdf['Espece_Ref_norm_for_eco'].map(ecology_df['Description_Ecologie'])
            pdf['Ecologie'] = pdf['Ecologie'].fillna("Description écologique non disponible.")
            # Supprimer la colonne temporaire de normalisation si elle n'est plus nécessaire
            # pdf = pdf.drop(columns=['Espece_Ref_norm_for_eco']) 
        else:
            pdf['Ecologie'] = "Description écologique non disponible (fichier non chargé ou vide)."

        color_sequence = px.colors.qualitative.Plotly 
        
        # MODIFIÉ: Ajout de hover_name et custom_data pour l'affichage de l'écologie
        fig_pca = px.scatter(
            pdf,
            x="PC1",
            y="PC2" if coords.shape[1] > 1 else None,
            color="Cluster",
            text="Espece_User", # Texte affiché sur les points
            hover_name="Espece_User", # Nom principal affiché en gras dans le tooltip
            custom_data=["Espece_User", "Ecologie", "Cluster"], # Données supplémentaires pour le hovertemplate
            template="plotly_dark",
            height=600,
            color_discrete_sequence=color_sequence
        )

        # MODIFIÉ: Mise à jour de hovertemplate pour inclure l'écologie
        fig_pca.update_traces(
            textposition="top center",
            marker=dict(opacity=0.7),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br><br>" + # Espece_User
                "Cluster: %{customdata[2]}<br>" +   # Numéro de Cluster
                "PC1: %{x:.2f}<br>" +
                "PC2: %{y:.2f}<br>" +
                "--------------------<br>" +
                "<i>Écologie:</i><br>%{customdata[1]}" + # Description écologique
                "<extra></extra>" # Masque les informations de trace par défaut
            )
        )
        
        unique_clusters = sorted(pdf["Cluster"].unique())
        cluster_color_map = {cluster_label: color_sequence[i % len(color_sequence)] for i, cluster_label in enumerate(unique_clusters)}

        if coords.shape[1] > 1: # Pour les enveloppes convexes
            for i, cluster_label in enumerate(unique_clusters):
                cluster_points_df = pdf[pdf["Cluster"] == cluster_label]
                unique_cluster_points = cluster_points_df[["PC1", "PC2"]].drop_duplicates().values
                
                if len(unique_cluster_points) >= min_points_for_hull:
                    try:
                        hull = ConvexHull(unique_cluster_points)
                        hull_points = unique_cluster_points[hull.vertices]
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
                        st.sidebar.warning(f"Impossible de générer l'enveloppe convexe pour le cluster {cluster_label}: {e_hull}") # Affichage discret
                elif len(unique_cluster_points) > 0:
                     st.sidebar.info(f"Cluster {cluster_label}: pas assez de points uniques ({len(unique_cluster_points)}) pour l'enveloppe (min {min_points_for_hull}).")


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
                labels=pdf["Espece_User"].tolist(),
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
        vip_data_df = pd.DataFrame({
            "Variable": sub.columns[1:],
            "Communalité (%)": (communal * 100).round(0).astype(int),
        }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
        
        vip_styled = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)'])\
                                   .format({"Communalité (%)": "{:}%"})

        cluster_compositions_data = []
        for c_label in sorted(pdf["Cluster"].unique()):
            esp_user_names = sorted(list(pdf.loc[pdf["Cluster"] == c_label, "Espece_User"].unique()))
            cluster_compositions_data.append({
                "cluster_label": c_label,
                "count": len(esp_user_names),
                "species_list": esp_user_names
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
        st.warning("Veuillez d'abord charger des données de traits pour afficher le graphique PCA.")
    elif run and not species_binom_user_unique:
        pass 
    elif run:
        st.info("Le graphique PCA sera affiché ici après une analyse réussie.")

col_vars, col_cluster_comp_container = st.columns([1, 2])

with col_vars:
    st.subheader("Importance des Variables")
    if vip_styled is not None:
        st.write(vip_styled.to_html(escape=False), unsafe_allow_html=True)
    elif run:
        st.info("Le tableau d'importance des variables sera affiché ici.")

with col_cluster_comp_container:
    st.subheader("Composition des Clusters")
    if cluster_compositions_data:
        num_clusters_found = len(cluster_compositions_data)
        if num_clusters_found > 0:
            # S'assurer que le nombre de colonnes ne dépasse pas un maximum raisonnable (par ex. 4)
            # ou le nombre de clusters trouvés si inférieur.
            num_display_cols = min(num_clusters_found, 4) 
            cluster_cols = st.columns(num_display_cols)
            for i, comp_data in enumerate(cluster_compositions_data):
                # Distribuer les clusters dans les colonnes disponibles
                with cluster_cols[i % num_display_cols]: 
                    st.markdown(f"**Cluster {comp_data['cluster_label']}** ({comp_data['count']} espèces)")
                    for species_name in comp_data['species_list']:
                        st.markdown(f"- {species_name}")
                    # Ajouter un séparateur ou de l'espace si plusieurs clusters sont dans la même colonne streamlit
                    if i // num_display_cols < (num_clusters_found -1) // num_display_cols and (i+1) % num_display_cols == 0 :
                         st.markdown("---") # Séparateur horizontal
        else:
            st.info("Aucun cluster à afficher.")
    elif run:
        st.info("La composition des clusters sera affichée ici.")

if fig_dend:
    st.plotly_chart(fig_dend, use_container_width=True)
elif run and not ref.empty and species_binom_user_unique:
    st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces uniques ou problème de seuil).")
elif run and ref.empty:
    st.warning("Veuillez d'abord charger des données de traits pour afficher le dendrogramme.")

if not run and not ref.empty:
    with col_pca_plot:
        st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse'.")
elif not run and ref.empty:
    pass
