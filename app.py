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
# INITIALISATION DES ETATS DE SESSION (pour l'exploration interactive)
# ---------------------------------------------------------------------------- #
if 'x_axis_trait_interactive' not in st.session_state:
    st.session_state.x_axis_trait_interactive = None
if 'y_axis_trait_interactive' not in st.session_state:
    st.session_state.y_axis_trait_interactive = None
if 'run_main_analysis_once' not in st.session_state: # Pour savoir si l'analyse principale a déjà tourné
    st.session_state.run_main_analysis_once = False


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

    run_main_analysis_button = st.button("Lancer l'analyse principale", type="primary", disabled=ref.empty, key="main_analysis_button")

# Initialisation des DataFrames en dehors des blocs conditionnels pour qu'ils aient une portée globale dans le script
fig_pca = None
fig_dend = None
vip_styled = None
vip_data_df = pd.DataFrame() 
cluster_compositions_data = []
sub = pd.DataFrame() 
pdf = pd.DataFrame() 
color_sequence = px.colors.qualitative.Plotly # Défini globalement pour être accessible partout

# ---------------------------------------------------------------------------- #
# ANALYSE PRINCIPALE (déclenchée par le bouton "Lancer l'analyse principale")
# ---------------------------------------------------------------------------- #
if run_main_analysis_button and not ref.empty:
    st.session_state.run_main_analysis_once = True # Marquer que l'analyse principale a tourné
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
        # Stocker sub dans st.session_state pour persistance
        st.session_state.sub = ref.loc[indices_to_keep_from_ref].copy() 
    else:
        st.session_state.sub = pd.DataFrame(columns=ref.columns) 

    sub = st.session_state.sub # Récupérer sub de session_state

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
        st.session_state.run_main_analysis_once = False # Réinitialiser si l'analyse échoue tôt
        st.stop()

    if sub.shape[0] < n_clusters_selected and n_clusters_selected > 0 :
        st.error(f"Le nombre d'espèces uniques trouvées et utilisées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters_selected}).")
        st.session_state.run_main_analysis_once = False
        st.stop()
    
    if sub.shape[0] < 2: 
        st.error(f"Au moins 2 espèces uniques sont nécessaires pour l'analyse. {sub.shape[0]} espèce(s) trouvée(s) et utilisée(s).")
        st.session_state.run_main_analysis_once = False
        st.stop()
    
    min_points_for_hull = 3 # Défini ici pour être utilisé plus tard
    user_input_binom_to_raw_map = {
        " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_raw_unique
    }

    try:
        labels, pca, coords, X = core.analyse(sub, n_clusters_selected)

        # Stocker pdf dans st.session_state
        current_pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])]) 
        current_pdf["Cluster"] = labels.astype(str)
        current_pdf["Espece_Ref"] = sub["Espece"].values 

        def get_user_input_name(full_ref_name):
            binom_ref_name = " ".join(full_ref_name.split()[:2]).lower()
            return user_input_binom_to_raw_map.get(binom_ref_name, full_ref_name)
        current_pdf["Espece_User"] = current_pdf["Espece_Ref"].apply(get_user_input_name)

        if not ecology_df.empty:
            current_pdf['Espece_Ref_norm_for_eco'] = (
                current_pdf['Espece_Ref']
                .astype(str)
                .str.strip() 
                .str.split()
                .str[:2]
                .str.join(" ")
                .str.lower()
            )
            current_pdf['Ecologie_raw'] = current_pdf['Espece_Ref_norm_for_eco'].map(ecology_df['Description_Ecologie'])
            current_pdf['Ecologie'] = current_pdf['Ecologie_raw'].apply(lambda x: format_ecology_for_hover(x, line_width_chars=65))
            current_pdf['Ecologie'] = current_pdf['Ecologie'].fillna(format_ecology_for_hover("Description écologique non disponible."))
        else:
            current_pdf['Ecologie'] = format_ecology_for_hover("Description écologique non disponible (fichier non chargé ou vide).")
        
        st.session_state.pdf = current_pdf # Stocker le pdf calculé
        
        # Calcul de vip_data_df et stockage
        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings**2).sum(axis=1)
        st.session_state.vip_data_df = pd.DataFrame({ 
            "Variable": sub.columns[1:], "Communalité (%)": (communal * 100).round(0).astype(int),
        }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
        
        # Stocker X pour le dendrogramme
        st.session_state.X_for_dendro = X

        # Initialiser les sélections d'axes pour l'exploration interactive après la première analyse principale
        numeric_trait_names_init = [col for col in sub.columns if col.lower() != "espece" and pd.api.types.is_numeric_dtype(sub[col])]
        if len(numeric_trait_names_init) >= 2:
            default_x_init = numeric_trait_names_init[0]
            default_y_init = numeric_trait_names_init[1]
            if not st.session_state.vip_data_df.empty and len(st.session_state.vip_data_df) >= 2:
                top_vars_init = [var for var in st.session_state.vip_data_df["Variable"].tolist() if var in numeric_trait_names_init]
                if len(top_vars_init) >= 1: default_x_init = top_vars_init[0]
                if len(top_vars_init) >= 2: default_y_init = top_vars_init[1]
                if default_x_init == default_y_init: # S'assurer qu'ils sont différents
                    available_defaults_y_init = [t for t in numeric_trait_names_init if t != default_x_init]
                    if available_defaults_y_init: default_y_init = available_defaults_y_init[0]
            st.session_state.x_axis_trait_interactive = default_x_init
            st.session_state.y_axis_trait_interactive = default_y_init
        elif len(numeric_trait_names_init) == 1:
            st.session_state.x_axis_trait_interactive = numeric_trait_names_init[0]
            st.session_state.y_axis_trait_interactive = numeric_trait_names_init[0]


    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'analyse ACP : {e}")
        st.exception(e) 
        st.session_state.run_main_analysis_once = False # Réinitialiser en cas d'erreur
        st.stop()

# Récupérer les DataFrames de st.session_state si l'analyse principale a déjà tourné
if st.session_state.run_main_analysis_once:
    sub = st.session_state.get('sub', pd.DataFrame())
    pdf = st.session_state.get('pdf', pd.DataFrame())
    vip_data_df = st.session_state.get('vip_data_df', pd.DataFrame())
    X_for_dendro = st.session_state.get('X_for_dendro', np.array([]))

    if not pdf.empty: # Générer les graphiques ACP etc. uniquement si pdf est disponible
        # --- Section ACP ---
        fig_pca = px.scatter(
            pdf, x="PC1", y="PC2" if pdf.shape[1] > 1 and "PC2" in pdf.columns else None, # Vérifier si PC2 existe
            color="Cluster", text="Espece_User", hover_name="Espece_User", 
            custom_data=["Espece_User", "Ecologie"], template="plotly_dark",
            height=600, color_discrete_sequence=color_sequence
        )
        fig_pca.update_traces(
            textposition="top center", marker=dict(opacity=0.7),
            hovertemplate=("<b>%{customdata[0]}</b><br><br><i>Écologie:</i><br>%{customdata[1]}<extra></extra>")
        )
        unique_clusters_pca = sorted(pdf["Cluster"].unique())
        cluster_color_map_pca = {cluster_label: color_sequence[i % len(color_sequence)] for i, cluster_label in enumerate(unique_clusters_pca)}
        if pdf.shape[1] > 1 and "PC2" in pdf.columns: 
            for i, cluster_label in enumerate(unique_clusters_pca):
                cluster_points_df_pca = pdf[pdf["Cluster"] == cluster_label]
                # S'assurer que PC1 et PC2 sont présents avant de tenter de créer l'enveloppe
                if "PC1" in cluster_points_df_pca.columns and "PC2" in cluster_points_df_pca.columns:
                    unique_cluster_points_pca = cluster_points_df_pca[["PC1", "PC2"]].drop_duplicates().values
                    if len(unique_cluster_points_pca) >= min_points_for_hull: # min_points_for_hull est défini plus haut
                        try:
                            hull_pca = ConvexHull(unique_cluster_points_pca)
                            hull_points_pca = unique_cluster_points_pca[hull_pca.vertices]
                            path_x_pca = np.append(hull_points_pca[:, 0], hull_points_pca[0, 0])
                            path_y_pca = np.append(hull_points_pca[:, 1], hull_points_pca[0, 1])
                            fig_pca.add_trace(go.Scatter(
                                x=path_x_pca, y=path_y_pca, fill="toself", fillcolor=cluster_color_map_pca[cluster_label],
                                line=dict(color=cluster_color_map_pca[cluster_label], width=1.5), mode='lines',
                                name=f'Cluster {cluster_label} Hull', opacity=0.2, showlegend=False, hoverinfo='skip'
                            ))
                        except Exception as e_hull_pca: print(f"Note: Impossible de générer l'enveloppe convexe ACP pour le cluster {cluster_label}: {e_hull_pca}")
                    elif len(unique_cluster_points_pca) > 0: print(f"Note: Cluster ACP {cluster_label}: pas assez de points uniques ({len(unique_cluster_points_pca)}) pour l'enveloppe (min {min_points_for_hull}).")
        fig_pca.update_layout(title_text="Clusters d'espèces (ACP)", title_x=0.5, legend_title_text='Cluster')
        
        # --- VIP Styled ---
        if not vip_data_df.empty:
            vip_styled = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)']).format({"Communalité (%)": "{:}%"})

        # --- Cluster Compositions ---
        cluster_compositions_data = []
        for c_label in sorted(pdf["Cluster"].unique()):
            esp_user_names = sorted(list(pdf.loc[pdf["Cluster"] == c_label, "Espece_User"].unique()))
            cluster_compositions_data.append({"cluster_label": c_label, "count": len(esp_user_names), "species_list": esp_user_names})

        # --- Dendrogram ---
        if X_for_dendro.shape[0] > 1:
            Z = linkage(X_for_dendro, method="ward")
            dynamic_color_threshold = 0
            if n_clusters_selected > 1 and (n_clusters_selected -1) <= Z.shape[0] :
                idx_threshold = -(n_clusters_selected - 1)
                if idx_threshold == 0: dynamic_color_threshold = Z[0, 2] / 2 
                elif Z.shape[0] >= (n_clusters_selected -1) and (n_clusters_selected -1) > 0: dynamic_color_threshold = Z[-(n_clusters_selected-1), 2] * 0.99 
            fig_dend = ff.create_dendrogram(
                X_for_dendro, orientation="left", labels=pdf["Espece_User"].tolist(), linkagefun=lambda _: Z,
                color_threshold=dynamic_color_threshold if n_clusters_selected > 1 else 0, colorscale=color_sequence
            )
            fig_dend.update_layout(template="plotly_dark", height=max(650, sub.shape[0] * 20), title_text="Dendrogramme", title_x=0.5)
        else: fig_dend = None


# ---------------------------------------------------------------------------- #
# AFFICHAGE DES RESULTATS ACP ET ASSOCIES
# ---------------------------------------------------------------------------- #
with col_pca_plot: 
    if fig_pca:
        st.plotly_chart(fig_pca, use_container_width=True)
    elif run_main_analysis_button and ref.empty: # Si le bouton principal a été cliqué mais ref est vide
        st.warning("Veuillez d'abord charger des données de traits pour afficher le graphique ACP.")
    elif run_main_analysis_button and sub.empty : # Si le bouton principal a été cliqué mais sub est vide après traitement
         st.warning("Aucune espèce valide pour l'analyse ACP.")
    elif st.session_state.run_main_analysis_once and not fig_pca: # Si l'analyse a tourné mais fig_pca n'est pas là
        st.info("Le graphique ACP sera affiché ici après une analyse principale réussie.")


col_vars_main, col_cluster_comp_main = st.columns([1, 2]) 
with col_vars_main:
    st.subheader("Importance des Variables (ACP)")
    if vip_styled is not None: 
        st.write(vip_styled.to_html(escape=False), unsafe_allow_html=True)
    elif st.session_state.run_main_analysis_once and not sub.empty: 
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
    elif st.session_state.run_main_analysis_once and not sub.empty:
        st.info("La composition des clusters (ACP) sera affichée ici.")

# ---------------------------------------------------------------------------- #
# EXPLORATION INTERACTIVE DES VARIABLES (anciennement traits bruts)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty: 
    st.markdown("---")
    # MODIFIÉ: Renommage de la section
    st.subheader("🔬 Exploration interactive des variables")

    potential_traits = [col for col in sub.columns if col.lower() != "espece"] 
    numeric_trait_names = [
        col for col in potential_traits if pd.api.types.is_numeric_dtype(sub[col])
    ]

    if len(numeric_trait_names) >= 2:
        st.markdown("##### Sélectionnez les variables pour les axes du nuage de points :")
        
        # Utiliser les valeurs de session_state pour les sélections d'axes
        # Les valeurs par défaut ont été initialisées après la première analyse principale
        current_x_trait = st.session_state.get('x_axis_trait_interactive', numeric_trait_names[0])
        current_y_trait = st.session_state.get('y_axis_trait_interactive', numeric_trait_names[1])


        col_scatter_select_x, col_scatter_select_y, col_button_update = st.columns([2,2,1]) # Ajout d'une colonne pour le bouton
        with col_scatter_select_x:
            x_axis_trait_selected = st.radio(
                "Axe X:",
                options=numeric_trait_names,
                index=numeric_trait_names.index(current_x_trait) if current_x_trait in numeric_trait_names else 0,
                key="interactive_x_radio" 
            )
        with col_scatter_select_y:
            y_axis_trait_selected = st.radio(
                "Axe Y:",
                options=numeric_trait_names,
                index=numeric_trait_names.index(current_y_trait) if current_y_trait in numeric_trait_names else (1 if len(numeric_trait_names) > 1 else 0),
                key="interactive_y_radio"
            )
        
        # Mettre à jour session_state si les sélections radio changent
        st.session_state.x_axis_trait_interactive = x_axis_trait_selected
        st.session_state.y_axis_trait_interactive = y_axis_trait_selected
        
        with col_button_update:
            st.markdown("<br>", unsafe_allow_html=True) # Pour aligner verticalement le bouton
            # NOUVEAU: Bouton d'actualisation pour cette section
            run_interactive_update = st.button("Actualiser l'exploration", key="update_interactive_plot_button")


        # Le graphique est généré si le bouton est cliqué OU si les radios changent (car Streamlit re-exécute)
        # La condition run_interactive_update est optionnelle ici car les radios provoquent déjà un re-run.
        # Cependant, le bouton donne un contrôle explicite à l'utilisateur.
        
        if x_axis_trait_selected and y_axis_trait_selected: # Toujours vrai avec st.radio
            if not pdf.empty: 
                if len(sub) == len(pdf):
                    plot_data_interactive = pd.DataFrame({
                        'Espece_User': pdf['Espece_User'].values,
                        'Ecologie': pdf['Ecologie'].values,
                        x_axis_trait_selected: sub[x_axis_trait_selected].values,
                        y_axis_trait_selected: sub[y_axis_trait_selected].values,
                        'Cluster': pdf['Cluster'].values 
                    })

                    fig_interactive_scatter = px.scatter(
                        plot_data_interactive,
                        x=x_axis_trait_selected,
                        y=y_axis_trait_selected,
                        color="Cluster", 
                        text="Espece_User", 
                        hover_name="Espece_User",
                        custom_data=["Espece_User", "Ecologie", x_axis_trait_selected, y_axis_trait_selected], 
                        template="plotly_dark",
                        height=600,
                        color_discrete_sequence=color_sequence 
                    )
                    fig_interactive_scatter.update_traces(
                        textposition="top center", 
                        marker=dict(opacity=0.8, size=8),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>" +
                            f"<br><i>{x_axis_trait_selected}:</i> %{{customdata[2]}}<br>" + 
                            f"<i>{y_axis_trait_selected}:</i> %{{customdata[3]}}<br>" +
                            "<br><i>Écologie:</i><br>%{customdata[1]}" + 
                            "<extra></extra>" 
                        )
                    )
                    
                    # NOUVEAU: Ajout des enveloppes convexes pour l'exploration interactive
                    unique_clusters_interactive = sorted(plot_data_interactive["Cluster"].unique())
                    cluster_color_map_interactive = {
                        cluster_label: color_sequence[i % len(color_sequence)] 
                        for i, cluster_label in enumerate(unique_clusters_interactive)
                    }

                    for cluster_label in unique_clusters_interactive:
                        cluster_points_df_interactive = plot_data_interactive[plot_data_interactive["Cluster"] == cluster_label]
                        # Utiliser les colonnes sélectionnées pour les axes
                        points_for_hull = cluster_points_df_interactive[[x_axis_trait_selected, y_axis_trait_selected]].drop_duplicates().values
                        
                        if len(points_for_hull) >= min_points_for_hull:
                            try:
                                hull_interactive = ConvexHull(points_for_hull)
                                hull_points_interactive = points_for_hull[hull_interactive.vertices]
                                path_x_interactive = np.append(hull_points_interactive[:, 0], hull_points_interactive[0, 0])
                                path_y_interactive = np.append(hull_points_interactive[:, 1], hull_points_interactive[0, 1])
                                fig_interactive_scatter.add_trace(go.Scatter(
                                    x=path_x_interactive, y=path_y_interactive, fill="toself", 
                                    fillcolor=cluster_color_map_interactive[cluster_label],
                                    line=dict(color=cluster_color_map_interactive[cluster_label], width=1.5), 
                                    mode='lines', name=f'Cluster {cluster_label} Hull', 
                                    opacity=0.2, showlegend=False, hoverinfo='skip'
                                ))
                            except Exception as e_hull_interactive:
                                print(f"Note: Impossible de générer l'enveloppe convexe interactive pour le cluster {cluster_label}: {e_hull_interactive}")
                        elif len(points_for_hull) > 0:
                            print(f"Note: Cluster interactif {cluster_label}: pas assez de points uniques ({len(points_for_hull)}) pour l'enveloppe (min {min_points_for_hull}).")

                    fig_interactive_scatter.update_layout(
                        title_text=f"Variables: {y_axis_trait_selected} en fonction de {x_axis_trait_selected}",
                        title_x=0.5,
                        xaxis_title=x_axis_trait_selected,
                        yaxis_title=y_axis_trait_selected
                    )
                    st.plotly_chart(fig_interactive_scatter, use_container_width=True)
                else:
                    st.warning("Discordance dans la taille des données pour le graphique interactif des variables. Le graphique ne peut être généré.")
            else:
                st.warning("Les données pour le graphique interactif des variables n'ont pas pu être préparées (dépend des résultats de l'analyse principale).")
            
    elif len(numeric_trait_names) == 1:
        st.warning("Au moins deux traits numériques sont nécessaires dans les données pour créer un nuage de points à 2 dimensions pour l'exploration interactive.")
    else: 
        st.warning("Aucun trait numérique n'a été trouvé dans les données pour l'exploration interactive des variables.")

# ---------------------------------------------------------------------------- #
# AFFICHAGE DU DENDROGRAMME 
# ---------------------------------------------------------------------------- #
if fig_dend: 
    st.plotly_chart(fig_dend, use_container_width=True)
elif st.session_state.run_main_analysis_once and not sub.empty and species_binom_user_unique :
    st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces uniques ou problème de seuil).")


# Message final si l'analyse n'a pas été lancée
if not st.session_state.run_main_analysis_once and not ref.empty:
    with col_pca_plot: 
        st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse principale'.")
elif not st.session_state.run_main_analysis_once and ref.empty:
     with col_input: 
        st.warning("Les données de référence n'ont pas pu être chargées. Vérifiez le fichier 'data_ref.csv'.")

