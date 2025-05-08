import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
from scipy.spatial import ConvexHull
import numpy as np
import textwrap # Importé pour la mise en forme du texte de survol

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
# FONCTION UTILITAIRE POUR FORMATER L'ÉCOLOGIE
# ---------------------------------------------------------------------------- #
def format_ecology_for_hover(text, line_width_chars=65):
    """Formate le texte pour l'affichage dans le survol Plotly avec des retours à la ligne."""
    if pd.isna(text) or text.strip() == "":
        return "Description écologique non disponible."
    wrapped_lines = textwrap.wrap(text, width=line_width_chars)
    return "<br>".join(wrapped_lines)

# ---------------------------------------------------------------------------- #
# CHARGEMENT DE LA BASE ECOLOGIQUE
# ---------------------------------------------------------------------------- #
@st.cache_data
def load_ecology_data(file_path="data_ecologie_espece.csv"):
    """Charge les données écologiques à partir du chemin spécifié."""
    try:
        eco_data = pd.read_csv(
            file_path, 
            sep=';',
            header=None, 
            usecols=[0, 1], 
            names=['Espece', 'Description_Ecologie'], 
            encoding='utf-8-sig'
        )
        
        eco_data['Espece_norm'] = (
            eco_data['Espece']
            .astype(str) 
            .str.strip()
            .str.split()
            .str[:2]
            .str.join(" ")
            .str.lower()
        )
        eco_data = eco_data.drop_duplicates(subset=['Espece_norm'], keep='first')
        eco_data = eco_data.set_index('Espece_norm')
        
        return eco_data[["Description_Ecologie"]]
    except FileNotFoundError:
        print(f"AVERTISSEMENT: Fichier de données écologiques '{file_path}' non trouvé.")
        st.toast(f"Fichier écologique '{file_path}' non trouvé.", icon="⚠️")
        return pd.DataFrame()
    except ValueError as ve:
        print(f"AVERTISSEMENT: Erreur de valeur lors de la lecture du fichier '{file_path}'. Détails: {ve}.")
        st.toast(f"Erreur format fichier écologique '{file_path}'.", icon="🔥")
        return pd.DataFrame()
    except Exception as e:
        print(f"AVERTISSEMENT: Impossible de charger les données écologiques depuis '{file_path}': {e}.")
        st.toast(f"Erreur chargement fichier écologique.", icon="🔥")
        return pd.DataFrame()

ecology_df = load_ecology_data()

# ---------------------------------------------------------------------------- #
# LAYOUT DE LA PAGE
# ---------------------------------------------------------------------------- #
col_input, col_pca_plot = st.columns([1, 3]) 

with col_input:
    st.subheader("CORTEGE FLORISTIQUE")
    n_clusters_selected = st.slider("Nombre de clusters (pour ACP)", 2, 8, 3, key="n_clusters_slider", disabled=ref.empty)
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
vip_data_df = pd.DataFrame() # Initialiser vip_data_df
cluster_compositions_data = []
sub = pd.DataFrame() 
pdf = pd.DataFrame() 

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

    if sub.shape[0] < n_clusters_selected and n_clusters_selected > 0 : # Ajout de condition pour n_clusters_selected > 0
        st.error(f"Le nombre d'espèces uniques trouvées et utilisées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters_selected}).")
        st.stop()
    
    if sub.shape[0] < 2: 
        st.error(f"Au moins 2 espèces uniques sont nécessaires pour l'analyse. {sub.shape[0]} espèce(s) trouvée(s) et utilisée(s).")
        st.stop()
    
    min_points_for_hull = 3
    user_input_binom_to_raw_map = {
        " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_raw_unique
    }

    try:
        labels, pca, coords, X = core.analyse(sub, n_clusters_selected)

        pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])]) 
        pdf["Cluster"] = labels.astype(str)
        pdf["Espece_Ref"] = sub["Espece"].values 

        def get_user_input_name(full_ref_name):
            binom_ref_name = " ".join(full_ref_name.split()[:2]).lower()
            return user_input_binom_to_raw_map.get(binom_ref_name, full_ref_name)
        pdf["Espece_User"] = pdf["Espece_Ref"].apply(get_user_input_name)

        if not ecology_df.empty:
            pdf['Espece_Ref_norm_for_eco'] = (
                pdf['Espece_Ref']
                .astype(str)
                .str.strip() 
                .str.split()
                .str[:2]
                .str.join(" ")
                .str.lower()
            )
            pdf['Ecologie_raw'] = pdf['Espece_Ref_norm_for_eco'].map(ecology_df['Description_Ecologie'])
            pdf['Ecologie'] = pdf['Ecologie_raw'].apply(lambda x: format_ecology_for_hover(x, line_width_chars=65))
            pdf['Ecologie'] = pdf['Ecologie'].fillna(format_ecology_for_hover("Description écologique non disponible."))
        else:
            pdf['Ecologie'] = format_ecology_for_hover("Description écologique non disponible (fichier non chargé ou vide).")

        color_sequence = px.colors.qualitative.Plotly 
        
        fig_pca = px.scatter(
            pdf, x="PC1", y="PC2" if coords.shape[1] > 1 else None,
            color="Cluster", text="Espece_User", hover_name="Espece_User", 
            custom_data=["Espece_User", "Ecologie"], template="plotly_dark",
            height=600, color_discrete_sequence=color_sequence
        )
        fig_pca.update_traces(
            textposition="top center", marker=dict(opacity=0.7),
            hovertemplate=("<b>%{customdata[0]}</b><br><br><i>Écologie:</i><br>%{customdata[1]}<extra></extra>")
        )
        
        unique_clusters = sorted(pdf["Cluster"].unique())
        cluster_color_map = {cluster_label: color_sequence[i % len(color_sequence)] for i, cluster_label in enumerate(unique_clusters)}

        if coords.shape[1] > 1: 
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
                            x=path_x, y=path_y, fill="toself", fillcolor=cluster_color_map[cluster_label],
                            line=dict(color=cluster_color_map[cluster_label], width=1.5), mode='lines',
                            name=f'Cluster {cluster_label} Hull', opacity=0.2, showlegend=False, hoverinfo='skip'
                        ))
                    except Exception as e_hull: print(f"Note: Impossible de générer l'enveloppe convexe pour le cluster {cluster_label}: {e_hull}")
                elif len(unique_cluster_points) > 0: print(f"Note: Cluster {cluster_label}: pas assez de points uniques ({len(unique_cluster_points)}) pour l'enveloppe (min {min_points_for_hull}).")
        fig_pca.update_layout(title_text="Clusters d'espèces (ACP)", title_x=0.5, legend_title_text='Cluster')

        # Calcul de vip_data_df ici pour qu'il soit disponible pour le graphique interactif
        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings**2).sum(axis=1)
        vip_data_df = pd.DataFrame({ # vip_data_df est défini ici
            "Variable": sub.columns[1:], "Communalité (%)": (communal * 100).round(0).astype(int),
        }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
        vip_styled = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)']).format({"Communalité (%)": "{:}%"})


        if X.shape[0] > 1: # Pour le dendrogramme
            Z = linkage(X, method="ward")
            dynamic_color_threshold = 0
            if n_clusters_selected > 1 and (n_clusters_selected -1) <= Z.shape[0] :
                idx_threshold = -(n_clusters_selected - 1)
                if idx_threshold == 0: dynamic_color_threshold = Z[0, 2] / 2 
                elif Z.shape[0] >= (n_clusters_selected -1) and (n_clusters_selected -1) > 0: dynamic_color_threshold = Z[-(n_clusters_selected-1), 2] * 0.99 
            fig_dend = ff.create_dendrogram(
                X, orientation="left", labels=pdf["Espece_User"].tolist(), linkagefun=lambda _: Z,
                color_threshold=dynamic_color_threshold if n_clusters_selected > 1 else 0, colorscale=color_sequence
            )
            fig_dend.update_layout(template="plotly_dark", height=max(650, sub.shape[0] * 20), title_text="Dendrogramme", title_x=0.5)
        else: fig_dend = None


        cluster_compositions_data = []
        for c_label in sorted(pdf["Cluster"].unique()):
            esp_user_names = sorted(list(pdf.loc[pdf["Cluster"] == c_label, "Espece_User"].unique()))
            cluster_compositions_data.append({"cluster_label": c_label, "count": len(esp_user_names), "species_list": esp_user_names})
    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'analyse ACP : {e}")
        st.exception(e) 
        st.stop()

# ---------------------------------------------------------------------------- #
# AFFICHAGE DES RESULTATS ACP ET ASSOCIES
# ---------------------------------------------------------------------------- #
with col_pca_plot: 
    if fig_pca:
        st.plotly_chart(fig_pca, use_container_width=True)
    elif run and ref.empty:
        st.warning("Veuillez d'abord charger des données de traits pour afficher le graphique ACP.")
    elif run and not species_binom_user_unique and not sub.empty : 
         pass 
    elif run: 
        st.info("Le graphique ACP sera affiché ici après une analyse réussie.")

col_vars_main, col_cluster_comp_main = st.columns([1, 2]) 
with col_vars_main:
    st.subheader("Importance des Variables (ACP)")
    if vip_styled is not None: # vip_styled est basé sur vip_data_df
        st.write(vip_styled.to_html(escape=False), unsafe_allow_html=True)
    elif run and not sub.empty: 
        st.info("Le tableau d'importance des variables (ACP) sera affiché ici.")
with col_cluster_comp_main:
    st.subheader("Composition des Clusters (ACP)")
    if cluster_compositions_data:
        num_clusters_found = len(cluster_compositions_data)
        if num_clusters_found > 0:
            num_display_cols = min(num_clusters_found, 3) 
            cluster_cols = st.columns(num_display_cols)
            for i, comp_data in enumerate(cluster_compositions_data):
                with cluster_cols[i % num_display_cols]: 
                    st.markdown(f"**Cluster {comp_data['cluster_label']}** ({comp_data['count']} espèces)")
                    for species_name in comp_data['species_list']: st.markdown(f"- {species_name}")
                    if i // num_display_cols < (num_clusters_found -1) // num_display_cols and (i+1) % num_display_cols == 0 : st.markdown("---") 
        else: st.info("Aucun cluster (ACP) à afficher.")
    elif run and not sub.empty:
        st.info("La composition des clusters (ACP) sera affichée ici.")

# ---------------------------------------------------------------------------- #
# NOUVEAU: EXPLORATION INTERACTIVE DES TRAITS BRUTS (Déplacé avant le dendrogramme)
# ---------------------------------------------------------------------------- #
if run and not sub.empty: 
    st.markdown("---")
    st.subheader("🔬 Exploration interactive des traits bruts")

    potential_traits = [col for col in sub.columns if col.lower() != "espece"] 
    numeric_trait_names = [
        col for col in potential_traits if pd.api.types.is_numeric_dtype(sub[col])
    ]

    if len(numeric_trait_names) >= 2: # Nécessite au moins deux traits pour un graphique 2D
        st.markdown("##### Sélectionnez les variables pour les axes du nuage de points :")
        
        # Déterminer les variables par défaut
        default_x_trait = numeric_trait_names[0]
        default_y_trait = numeric_trait_names[1]

        if not vip_data_df.empty and len(vip_data_df) >= 2:
            # S'assurer que les variables importantes sont bien dans numeric_trait_names
            top_vars_from_vip = [var for var in vip_data_df["Variable"].tolist() if var in numeric_trait_names]
            if len(top_vars_from_vip) >=1:
                default_x_trait = top_vars_from_vip[0]
            if len(top_vars_from_vip) >=2:
                default_y_trait = top_vars_from_vip[1]
                # S'assurer que x et y sont différents si possible
                if default_x_trait == default_y_trait and len(numeric_trait_names) > 1:
                    # Tenter de prendre le suivant dans numeric_trait_names si le premier de vip est identique au second
                    # et que numeric_trait_names a d'autres options
                    available_defaults_y = [t for t in numeric_trait_names if t != default_x_trait]
                    if available_defaults_y:
                        default_y_trait = available_defaults_y[0]


        col_scatter_select_x, col_scatter_select_y = st.columns(2)
        with col_scatter_select_x:
            # MODIFIÉ: Utilisation de st.radio
            x_axis_trait = st.radio(
                "Axe X:",
                options=numeric_trait_names,
                index=numeric_trait_names.index(default_x_trait) if default_x_trait in numeric_trait_names else 0,
                key="scatter_x_trait_radio"
            )
        with col_scatter_select_y:
            # MODIFIÉ: Utilisation de st.radio
            y_axis_trait = st.radio(
                "Axe Y:",
                options=numeric_trait_names,
                index=numeric_trait_names.index(default_y_trait) if default_y_trait in numeric_trait_names else (1 if len(numeric_trait_names) > 1 else 0),
                key="scatter_y_trait_radio"
            )

        if x_axis_trait and y_axis_trait:
            if not pdf.empty: # pdf doit contenir Espece_User, Ecologie, Cluster
                # 'sub' contient les valeurs des traits bruts. 'pdf' et 'sub' sont alignés par index.
                # Assurer que les données pour le graphique interactif utilisent l'index de 'sub'
                # pour garantir l'alignement avec les traits bruts.
                
                # Création de plot_data_interactive en s'assurant de l'alignement
                # sub.index devrait correspondre à l'ordre des espèces dans pdf après l'analyse
                
                # Les données pour le graphique interactif doivent provenir de `sub` pour les traits
                # et de `pdf` pour les métadonnées (Espece_User, Ecologie, Cluster)
                # Il est crucial que `sub` et `pdf` soient alignés par leur index implicite (ordre des lignes)
                # après la sélection des espèces et l'analyse ACP.
                
                # Vérifions la taille de sub et pdf
                if len(sub) == len(pdf):
                    plot_data_interactive = pd.DataFrame({
                        'Espece_User': pdf['Espece_User'].values, # Utiliser .values pour éviter les problèmes d'index
                        'Ecologie': pdf['Ecologie'].values,
                        x_axis_trait: sub[x_axis_trait].values,
                        y_axis_trait: sub[y_axis_trait].values,
                        'Cluster': pdf['Cluster'].values 
                    })

                    fig_interactive_scatter = px.scatter(
                        plot_data_interactive,
                        x=x_axis_trait,
                        y=y_axis_trait,
                        color="Cluster", 
                        text="Espece_User", # MODIFIÉ: Afficher le nom de l'espèce sur le point
                        hover_name="Espece_User",
                        custom_data=["Espece_User", "Ecologie", x_axis_trait, y_axis_trait], # Ajout des valeurs des traits pour le survol
                        template="plotly_dark",
                        height=600,
                        color_discrete_sequence=color_sequence 
                    )

                    fig_interactive_scatter.update_traces(
                        textposition="top center", # Position du nom de l'espèce
                        marker=dict(opacity=0.8, size=8),
                        hovertemplate=( # MODIFIÉ: hovertemplate pour inclure les valeurs des traits
                            "<b>%{customdata[0]}</b><br>" +
                            f"<br><i>{x_axis_trait}:</i> %{{customdata[2]}}<br>" + # Utiliser customdata pour les valeurs des traits
                            f"<i>{y_axis_trait}:</i> %{{customdata[3]}}<br>" +
                            "<br><i>Écologie:</i><br>%{customdata[1]}" + 
                            "<extra></extra>" 
                        )
                    )
                    
                    fig_interactive_scatter.update_layout(
                        title_text=f"Traits bruts: {y_axis_trait} en fonction de {x_axis_trait}",
                        title_x=0.5,
                        xaxis_title=x_axis_trait,
                        yaxis_title=y_axis_trait
                    )
                    st.plotly_chart(fig_interactive_scatter, use_container_width=True)
                else:
                    st.warning("Discordance dans la taille des données pour le graphique interactif des traits bruts. Le graphique ne peut être généré.")
            else:
                st.warning("Les données pour le graphique interactif des traits bruts n'ont pas pu être préparées (dépend des résultats de l'ACP).")
            
    elif len(numeric_trait_names) == 1:
        st.warning("Au moins deux traits numériques sont nécessaires dans les données pour créer un nuage de points à 2 dimensions pour l'exploration interactive.")
    else: 
        st.warning("Aucun trait numérique n'a été trouvé dans les données pour l'exploration interactive des traits bruts.")

# ---------------------------------------------------------------------------- #
# AFFICHAGE DU DENDROGRAMME (Déplacé après l'exploration interactive)
# ---------------------------------------------------------------------------- #
if fig_dend: 
    st.plotly_chart(fig_dend, use_container_width=True)
elif run and not sub.empty and species_binom_user_unique :
    st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces uniques ou problème de seuil).")


# Message final si l'analyse n'a pas été lancée
if not run and not ref.empty:
    with col_pca_plot: 
        st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse'.")
elif not run and ref.empty:
     with col_input: 
        st.warning("Les données de référence n'ont pas pu être chargées. Vérifiez le fichier 'data_ref.csv'.")

