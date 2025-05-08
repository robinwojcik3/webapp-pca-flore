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
# CONSTANTES ET CHARGEMENT DE DONNÉES INITIALES
# ---------------------------------------------------------------------------- #
MIN_POINTS_FOR_HULL = 3
COLOR_SEQUENCE = px.colors.qualitative.Plotly

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
# INITIALISATION DES ETATS DE SESSION
# ---------------------------------------------------------------------------- #
if 'x_axis_trait_interactive' not in st.session_state:
    st.session_state.x_axis_trait_interactive = None
if 'y_axis_trait_interactive' not in st.session_state:
    st.session_state.y_axis_trait_interactive = None
if 'run_main_analysis_once' not in st.session_state:
    st.session_state.run_main_analysis_once = False

# ---------------------------------------------------------------------------- #
# LAYOUT DE LA PAGE (SECTION ENTRÉE UTILISATEUR)
# ---------------------------------------------------------------------------- #
col_input, col_pca_plot_container = st.columns([1, 3]) # Renommé pour clarté

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

# Initialisation des variables pour les figures et données qui seront utilisées plus tard
fig_pca = None
fig_dend = None
vip_styled = None # Pour le tableau stylisé VIP
vip_data_df = pd.DataFrame() # Pour les données brutes VIP
cluster_compositions_data = [] # Pour les données de composition des clusters
sub = pd.DataFrame() # Sous-ensemble des données de traits basé sur l'entrée utilisateur
pdf = pd.DataFrame() # DataFrame pour les coordonnées PCA et infos associées
X_for_dendro = np.array([]) # Données pour le dendrogramme

# ---------------------------------------------------------------------------- #
# ANALYSE PRINCIPALE (CALCULS)
# ---------------------------------------------------------------------------- #
if run_main_analysis_button and not ref.empty:
    st.session_state.run_main_analysis_once = True
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
        st.session_state.sub = ref.loc[indices_to_keep_from_ref].copy()
    else:
        st.session_state.sub = pd.DataFrame(columns=ref.columns)

    sub = st.session_state.sub

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
        with col_input: # Afficher l'avertissement dans la colonne des inputs
            st.warning(
                "Non trouvées dans la base de traits : " + ", ".join(not_found_user_raw_names),
                icon="⚠️"
            )

    if sub.empty:
        st.error("Aucune des espèces saisies (après déduplication et recherche dans la base de traits) n'a pu être utilisée pour l'analyse.")
        st.session_state.run_main_analysis_once = False
        st.stop()

    if sub.shape[0] < n_clusters_selected and n_clusters_selected > 0 :
        st.error(f"Le nombre d'espèces uniques trouvées et utilisées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters_selected}).")
        st.session_state.run_main_analysis_once = False
        st.stop()

    if sub.shape[0] < 2:
        st.error(f"Au moins 2 espèces uniques sont nécessaires pour l'analyse. {sub.shape[0]} espèce(s) trouvée(s) et utilisée(s).")
        st.session_state.run_main_analysis_once = False
        st.stop()

    user_input_binom_to_raw_map = {
        " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_raw_unique
    }

    try:
        labels, pca, coords, X = core.analyse(sub, n_clusters_selected)

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
                .astype(str).str.strip().str.split().str[:2].str.join(" ").str.lower()
            )
            current_pdf['Ecologie_raw'] = current_pdf['Espece_Ref_norm_for_eco'].map(ecology_df['Description_Ecologie'])
            current_pdf['Ecologie'] = current_pdf['Ecologie_raw'].apply(lambda x: format_ecology_for_hover(x, line_width_chars=65))
            current_pdf['Ecologie'] = current_pdf['Ecologie'].fillna(format_ecology_for_hover("Description écologique non disponible."))
        else:
            current_pdf['Ecologie'] = format_ecology_for_hover("Description écologique non disponible (fichier non chargé ou vide).")
        st.session_state.pdf = current_pdf

        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings**2).sum(axis=1)
        trait_columns = [col for col in sub.columns if col.lower() != "espece"] # Noms des traits
        st.session_state.vip_data_df = pd.DataFrame({
            "Variable": trait_columns,
            "Communalité (%)": (communal * 100).round(0).astype(int),
        }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
        st.session_state.X_for_dendro = X

        numeric_trait_names_init = [col for col in trait_columns if pd.api.types.is_numeric_dtype(sub[col])]
        default_x_init, default_y_init = None, None
        if not st.session_state.vip_data_df.empty and len(numeric_trait_names_init) >=1:
            top_vars_from_vip = [var for var in st.session_state.vip_data_df["Variable"].tolist() if var in numeric_trait_names_init]
            if len(top_vars_from_vip) >= 1: default_x_init = top_vars_from_vip[0]
            if len(top_vars_from_vip) >= 2: default_y_init = top_vars_from_vip[1]
            elif len(top_vars_from_vip) == 1:
                remaining_numeric_traits = [t for t in numeric_trait_names_init if t != default_x_init]
                if remaining_numeric_traits: default_y_init = remaining_numeric_traits[0]
                else: default_y_init = default_x_init
        if default_x_init is None and len(numeric_trait_names_init) >= 1: default_x_init = numeric_trait_names_init[0]
        if default_y_init is None:
            if len(numeric_trait_names_init) >= 2:
                default_y_init = numeric_trait_names_init[1] if numeric_trait_names_init[0] != numeric_trait_names_init[1] else (numeric_trait_names_init[0] if len(numeric_trait_names_init) == 1 else (numeric_trait_names_init[2] if len(numeric_trait_names_init) > 2 else numeric_trait_names_init[0]))
            elif default_x_init: default_y_init = default_x_init # Une seule var numérique
        st.session_state.x_axis_trait_interactive = default_x_init
        st.session_state.y_axis_trait_interactive = default_y_init

    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'analyse ACP : {e}")
        st.exception(e)
        st.session_state.run_main_analysis_once = False
        st.stop()

# Récupérer les DataFrames de st.session_state si l'analyse principale a déjà tourné
# Ces variables sont utilisées pour construire les figures même si le bouton n'est pas cliqué à nouveau
if st.session_state.run_main_analysis_once:
    sub = st.session_state.get('sub', pd.DataFrame())
    pdf = st.session_state.get('pdf', pd.DataFrame())
    vip_data_df = st.session_state.get('vip_data_df', pd.DataFrame())
    X_for_dendro = st.session_state.get('X_for_dendro', np.array([]))

    # Construction des figures si les données sont prêtes
    if not pdf.empty:
        if "Cluster" not in pdf.columns: pdf["Cluster"] = "0"
        fig_pca = px.scatter(
            pdf, x="PC1", y="PC2" if pdf.shape[1] > 1 and "PC2" in pdf.columns else None,
            color="Cluster", text="Espece_User", hover_name="Espece_User",
            custom_data=["Espece_User", "Ecologie"], template="plotly_dark",
            height=600, color_discrete_sequence=COLOR_SEQUENCE
        )
        fig_pca.update_traces(
            textposition="top center", marker=dict(opacity=0.7),
            hovertemplate=("<b>%{customdata[0]}</b><br><br><i>Écologie:</i><br>%{customdata[1]}<extra></extra>")
        )
        unique_clusters_pca = sorted(pdf["Cluster"].unique())
        cluster_color_map_pca = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_pca)}

        if "PC2" in pdf.columns and pdf.shape[1] > 1 :
            for i, cluster_label in enumerate(unique_clusters_pca):
                cluster_points_df_pca = pdf[pdf["Cluster"] == cluster_label]
                if "PC1" in cluster_points_df_pca.columns and "PC2" in cluster_points_df_pca.columns:
                    unique_cluster_points_pca = cluster_points_df_pca[["PC1", "PC2"]].drop_duplicates().values
                    if len(unique_cluster_points_pca) >= MIN_POINTS_FOR_HULL:
                        try:
                            hull_pca = ConvexHull(unique_cluster_points_pca)
                            hull_points_pca = unique_cluster_points_pca[hull_pca.vertices]
                            path_x = np.append(hull_points_pca[:, 0], hull_points_pca[0, 0])
                            path_y = np.append(hull_points_pca[:, 1], hull_points_pca[0, 1])
                            clr = cluster_color_map_pca.get(cluster_label, COLOR_SEQUENCE[0])
                            fig_pca.add_trace(go.Scatter(
                                x=path_x, y=path_y, fill="toself", fillcolor=clr,
                                line=dict(color=clr, width=1.5), mode='lines',
                                name=f'Cluster {cluster_label} Hull', opacity=0.2, showlegend=False, hoverinfo='skip'
                            ))
                        except Exception as e: print(f"Hull ACP {cluster_label}: {e}")
        fig_pca.update_layout(title_text="Clusters d'espèces (ACP)", title_x=0.5, legend_title_text='Cluster')

        if not vip_data_df.empty:
            vip_styled = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)']).format({"Communalité (%)": "{:}%"})

        cluster_compositions_data = []
        for c_label in sorted(pdf["Cluster"].unique()):
            s_list = sorted(list(pdf.loc[pdf["Cluster"] == c_label, "Espece_User"].unique()))
            cluster_compositions_data.append({"cluster_label": c_label, "count": len(s_list), "species_list": s_list})

        if X_for_dendro.shape[0] > 1:
            Z = linkage(X_for_dendro, method="ward")
            dyn_thresh = 0
            if n_clusters_selected > 1 and (n_clusters_selected -1) <= Z.shape[0] :
                idx = -(n_clusters_selected - 1)
                if idx == 0 and Z.shape[0]>0: dyn_thresh = Z[0, 2] / 2
                elif Z.shape[0] >= (n_clusters_selected -1) and (n_clusters_selected -1) > 0 and (-(n_clusters_selected-1) + Z.shape[0] >= 0):
                    dyn_thresh = Z[-(n_clusters_selected-1), 2] * 0.99
                elif Z.shape[0] > 0 : dyn_thresh = Z[0, 2] / 2 # Fallback
            fig_dend = ff.create_dendrogram(
                X_for_dendro, orientation="left", labels=pdf["Espece_User"].tolist(), linkagefun=lambda _: Z,
                color_threshold=dyn_thresh if n_clusters_selected > 1 else 0, colorscale=COLOR_SEQUENCE
            )
            fig_dend.update_layout(template="plotly_dark", height=max(650, sub.shape[0] * 20), title_text="Dendrogramme", title_x=0.5)
        else: fig_dend = None

# ---------------------------------------------------------------------------- #
# SECTION 1: AFFICHAGE DU GRAPHIQUE ACP (DANS LA COLONNE DROITE)
# ---------------------------------------------------------------------------- #
with col_pca_plot_container: # Utilise la colonne droite définie en haut
    if fig_pca:
        st.plotly_chart(fig_pca, use_container_width=True)
    elif run_main_analysis_button and ref.empty:
        st.warning("Veuillez d'abord charger des données de traits pour afficher le graphique ACP.")
    elif run_main_analysis_button and sub.empty :
        st.warning("Aucune espèce valide pour l'analyse ACP.")
    elif st.session_state.run_main_analysis_once and not fig_pca:
        st.info("Le graphique ACP sera affiché ici après une analyse principale réussie.")
    elif not st.session_state.run_main_analysis_once and not ref.empty:
         st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse principale'.")


# Message si données de référence non chargées (affiché dans la colonne des inputs)
if not st.session_state.run_main_analysis_once and ref.empty:
    with col_input:
        st.warning("Les données de référence n'ont pas pu être chargées. Vérifiez le fichier 'data_ref.csv'.")


# ---------------------------------------------------------------------------- #
# SECTION 2: EXPLORATION INTERACTIVE DES VARIABLES (PLEINE LARGEUR)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty: # S'affiche seulement si l'analyse a tourné et sub n'est pas vide
    st.markdown("---")
    st.subheader("🔬 Exploration interactive des variables")

    potential_traits = [col for col in sub.columns if col.lower() != "espece"]
    numeric_trait_names = [col for col in potential_traits if pd.api.types.is_numeric_dtype(sub[col])]

    if len(numeric_trait_names) >= 2:
        st.markdown("##### Sélectionnez les variables pour les axes du nuage de points :")
        current_x_trait = st.session_state.get('x_axis_trait_interactive', numeric_trait_names[0])
        current_y_trait = st.session_state.get('y_axis_trait_interactive', numeric_trait_names[1] if len(numeric_trait_names)>1 else numeric_trait_names[0])

        if current_x_trait not in numeric_trait_names: current_x_trait = numeric_trait_names[0] if numeric_trait_names else None
        if current_y_trait not in numeric_trait_names: current_y_trait = (numeric_trait_names[1] if len(numeric_trait_names) > 1 else (numeric_trait_names[0] if numeric_trait_names else None))

        col_scatter_select_x, col_scatter_select_y = st.columns([1,1])
        with col_scatter_select_x:
            x_axis_trait_selected = st.radio(
                "Axe X:", options=numeric_trait_names,
                index=numeric_trait_names.index(current_x_trait) if current_x_trait in numeric_trait_names else 0,
                key="interactive_x_radio"
            )
        with col_scatter_select_y:
            y_axis_trait_selected = st.radio(
                "Axe Y:", options=numeric_trait_names,
                index=numeric_trait_names.index(current_y_trait) if current_y_trait in numeric_trait_names else (1 if len(numeric_trait_names) > 1 else 0),
                key="interactive_y_radio"
            )
        st.session_state.x_axis_trait_interactive = x_axis_trait_selected
        st.session_state.y_axis_trait_interactive = y_axis_trait_selected

        if x_axis_trait_selected and y_axis_trait_selected and not pdf.empty and len(sub) == len(pdf):
            plot_data_interactive = pd.DataFrame({
                'Espece_User': pdf['Espece_User'].values, 'Ecologie': pdf['Ecologie'].values,
                x_axis_trait_selected: sub[x_axis_trait_selected].values,
                y_axis_trait_selected: sub[y_axis_trait_selected].values,
                'Cluster': pdf['Cluster'].values
            })
            fig_interactive_scatter = px.scatter(
                plot_data_interactive, x=x_axis_trait_selected, y=y_axis_trait_selected,
                color="Cluster", text="Espece_User", hover_name="Espece_User",
                custom_data=["Espece_User", "Ecologie", x_axis_trait_selected, y_axis_trait_selected],
                template="plotly_dark", height=600, color_discrete_sequence=COLOR_SEQUENCE
            )
            fig_interactive_scatter.update_traces(
                textposition="top center", marker=dict(opacity=0.8, size=8),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>" +
                    f"<br><i>{x_axis_trait_selected}:</i> %{{customdata[2]}}<br>" +
                    f"<i>{y_axis_trait_selected}:</i> %{{customdata[3]}}<br>" +
                    "<br><i>Écologie:</i><br>%{customdata[1]}<extra></extra>"
                )
            )
            unique_clusters_interactive = sorted(plot_data_interactive["Cluster"].unique())
            cluster_color_map_interactive = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_interactive)}

            for cluster_label in unique_clusters_interactive:
                cluster_points_df_interactive = plot_data_interactive[plot_data_interactive["Cluster"] == cluster_label]
                if x_axis_trait_selected in cluster_points_df_interactive and y_axis_trait_selected in cluster_points_df_interactive:
                    points_for_hull = cluster_points_df_interactive[[x_axis_trait_selected, y_axis_trait_selected]].drop_duplicates().values
                    if len(points_for_hull) >= MIN_POINTS_FOR_HULL:
                        try:
                            hull_interactive = ConvexHull(points_for_hull)
                            hull_points_interactive = points_for_hull[hull_interactive.vertices]
                            path_x = np.append(hull_points_interactive[:, 0], hull_points_interactive[0, 0])
                            path_y = np.append(hull_points_interactive[:, 1], hull_points_interactive[0, 1])
                            clr_int = cluster_color_map_interactive.get(cluster_label, COLOR_SEQUENCE[0])
                            fig_interactive_scatter.add_trace(go.Scatter(
                                x=path_x, y=path_y, fill="toself", fillcolor=clr_int,
                                line=dict(color=clr_int, width=1.5), mode='lines',
                                name=f'Cluster {cluster_label} Hull', opacity=0.2, showlegend=False, hoverinfo='skip'
                            ))
                        except Exception as e: print(f"Hull Inter.: {cluster_label} ({x_axis_trait_selected}, {y_axis_trait_selected}): {e}")
            fig_interactive_scatter.update_layout(
                title_text=f"Variables: {y_axis_trait_selected} en fonction de {x_axis_trait_selected}", title_x=0.5,
                xaxis_title=x_axis_trait_selected, yaxis_title=y_axis_trait_selected
            )
            st.plotly_chart(fig_interactive_scatter, use_container_width=True)
        elif not (x_axis_trait_selected and y_axis_trait_selected):
             st.warning("Veuillez sélectionner des variables pour les axes X et Y.")
        elif pdf.empty or len(sub) != len(pdf) :
             st.warning("Données pour le graphique interactif non prêtes ou incohérentes.")

    elif len(numeric_trait_names) == 1:
        st.warning("Au moins deux traits numériques sont nécessaires pour un nuage de points 2D interactif.")
    else:
        st.warning("Aucun trait numérique trouvé pour l'exploration interactive.")

# ---------------------------------------------------------------------------- #
# SECTION 3: IMPORTANCE DES VARIABLES ET COMPOSITION DES CLUSTERS (EN COLONNES)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty: # S'affiche seulement si l'analyse a tourné et sub n'est pas vide
    col_vars_main, col_cluster_comp_main = st.columns([1, 2])
    with col_vars_main:
        st.subheader("Importance des Variables (ACP)")
        if vip_styled is not None:
            st.write(vip_styled.to_html(escape=False), unsafe_allow_html=True)
        else: # Cas où vip_styled n'a pas pu être généré mais l'analyse a tourné
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
        else: # Cas où cluster_compositions_data est vide mais l'analyse a tourné
             st.info("La composition des clusters (ACP) sera affichée ici.")

# ---------------------------------------------------------------------------- #
# SECTION 4: AFFICHAGE DU DENDROGRAMME (PLEINE LARGEUR, EN DERNIER)
# ---------------------------------------------------------------------------- #
if fig_dend: # fig_dend est défini si l'analyse a réussi et les conditions sont remplies
    st.plotly_chart(fig_dend, use_container_width=True)
elif st.session_state.run_main_analysis_once and not sub.empty and species_binom_user_unique :
    # Ce message s'affiche si l'analyse a été lancée, des espèces ont été trouvées, mais le dendrogramme n'a pas pu être généré
    st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces uniques après traitement ou problème de données pour le linkage).")
