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
# LAYOUT DE LA PAGE (SECTION SUPERIEURE AVEC INPUTS ET IMPORTANCE VARIABLES)
# ---------------------------------------------------------------------------- #
col_input_main, col_vip_main, col_interactive_selection_main = st.columns([1, 1, 1])

with col_input_main:
    st.subheader("CORTEGE FLORISTIQUE")
    n_clusters_selected = st.slider("Nombre de clusters (pour ACP)", 2, 8, 3, key="n_clusters_slider", disabled=ref.empty)
    species_txt = st.text_area(
        "Liste d'espèces (une par ligne)", height=250, # Ajuster la hauteur si nécessaire
        placeholder="Teucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus\n…",
        disabled=ref.empty
    )
    species_raw_unique = sorted(list(set(s.strip() for s in species_txt.splitlines() if s.strip())))
    species_binom_user_unique = [" ".join(s.split()[:2]).lower() for s in species_raw_unique]

    run_main_analysis_button = st.button("Lancer l'analyse principale", type="primary", disabled=ref.empty, key="main_analysis_button")

fig_pca = None
fig_dend = None
vip_styled = None
vip_data_df = pd.DataFrame()
cluster_compositions_data = []
sub = pd.DataFrame()
pdf = pd.DataFrame()
X_for_dendro = np.array([])
fig_interactive_scatter = None # Initialisation

# ---------------------------------------------------------------------------- #
# ANALYSE PRINCIPALE (CALCULS)
# ---------------------------------------------------------------------------- #
if run_main_analysis_button and not ref.empty:
    st.session_state.run_main_analysis_once = True
    if not species_binom_user_unique:
        with col_input_main: # Afficher l'erreur dans la colonne d'input
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
        found_ref_binom_values_in_sub = ( sub["Espece"].str.split().str[:2].str.join(" ").str.lower().tolist() )

    not_found_user_raw_names = [species_raw_unique[i] for i, user_binom_name in enumerate(species_binom_user_unique) if user_binom_name not in found_ref_binom_values_in_sub]
    if not_found_user_raw_names:
        with col_input_main: # Afficher l'avertissement dans la colonne d'input
            st.warning("Non trouvées dans la base de traits : " + ", ".join(not_found_user_raw_names), icon="⚠️")

    if sub.empty:
        with col_input_main: # Afficher l'erreur dans la colonne d'input
            st.error("Aucune des espèces saisies (après déduplication et recherche dans la base de traits) n'a pu être utilisée pour l'analyse.")
        st.session_state.run_main_analysis_once = False; st.stop()
    if sub.shape[0] < n_clusters_selected and n_clusters_selected > 0 :
        with col_input_main: # Afficher l'erreur dans la colonne d'input
            st.error(f"Le nombre d'espèces uniques trouvées et utilisées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters_selected}).")
        st.session_state.run_main_analysis_once = False; st.stop()
    if sub.shape[0] < 2:
        with col_input_main: # Afficher l'erreur dans la colonne d'input
            st.error(f"Au moins 2 espèces uniques sont nécessaires pour l'analyse. {sub.shape[0]} espèce(s) trouvée(s) et utilisée(s).")
        st.session_state.run_main_analysis_once = False; st.stop()

    user_input_binom_to_raw_map = { " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_raw_unique }
    try:
        labels, pca, coords, X = core.analyse(sub, n_clusters_selected)
        current_pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
        current_pdf["Cluster"] = labels.astype(str)
        current_pdf["Espece_Ref"] = sub["Espece"].values
        current_pdf["Espece_User"] = current_pdf["Espece_Ref"].apply(lambda full_ref_name: user_input_binom_to_raw_map.get(" ".join(full_ref_name.split()[:2]).lower(), full_ref_name))

        if not ecology_df.empty:
            current_pdf['Espece_Ref_norm_for_eco'] = ( current_pdf['Espece_Ref'].astype(str).str.strip().str.split().str[:2].str.join(" ").str.lower() )
            current_pdf['Ecologie_raw'] = current_pdf['Espece_Ref_norm_for_eco'].map(ecology_df['Description_Ecologie'])
            current_pdf['Ecologie'] = current_pdf['Ecologie_raw'].apply(lambda x: format_ecology_for_hover(x))
            current_pdf['Ecologie'] = current_pdf['Ecologie'].fillna(format_ecology_for_hover("Description écologique non disponible."))
        else:
            current_pdf['Ecologie'] = format_ecology_for_hover("Description écologique non disponible (fichier non chargé ou vide).")
        st.session_state.pdf = current_pdf

        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings**2).sum(axis=1)
        trait_columns = [col for col in sub.columns if col.lower() != "espece"]
        st.session_state.vip_data_df = pd.DataFrame({ "Variable": trait_columns, "Communalité (%)": (communal * 100).round(0).astype(int), }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
        st.session_state.X_for_dendro = X

        numeric_trait_names_init = [col for col in trait_columns if pd.api.types.is_numeric_dtype(sub[col])]
        default_x_init, default_y_init = None, None
        if not st.session_state.vip_data_df.empty and len(numeric_trait_names_init) >=1:
            top_vars_from_vip = [var for var in st.session_state.vip_data_df["Variable"].tolist() if var in numeric_trait_names_init]
            if len(top_vars_from_vip) >= 1: default_x_init = top_vars_from_vip[0]
            if len(top_vars_from_vip) >= 2: default_y_init = top_vars_from_vip[1]
            elif len(top_vars_from_vip) == 1: default_y_init = ([t for t in numeric_trait_names_init if t != default_x_init] or [default_x_init])[0] # Fallback if only one top var
        if default_x_init is None and len(numeric_trait_names_init) >= 1: default_x_init = numeric_trait_names_init[0]
        if default_y_init is None: # Ensure y_init is set
            if len(numeric_trait_names_init) >= 2:
                default_y_init = numeric_trait_names_init[1] if numeric_trait_names_init[0] != numeric_trait_names_init[1] else numeric_trait_names_init[0]
            elif default_x_init: # If only one numeric trait, use it for both axes
                 default_y_init = default_x_init
        st.session_state.x_axis_trait_interactive, st.session_state.y_axis_trait_interactive = default_x_init, default_y_init

    except Exception as e:
        with col_input_main: # Afficher l'erreur dans la colonne d'input
             st.error(f"Une erreur est survenue lors de l'analyse ACP : {e}")
        st.exception(e) # Log complet de l'erreur dans la console/logs Streamlit
        st.session_state.run_main_analysis_once = False; st.stop()

# Récupération des données de session si l'analyse a déjà été lancée
if st.session_state.run_main_analysis_once:
    sub = st.session_state.get('sub', pd.DataFrame())
    pdf = st.session_state.get('pdf', pd.DataFrame())
    vip_data_df = st.session_state.get('vip_data_df', pd.DataFrame())
    X_for_dendro = st.session_state.get('X_for_dendro', np.array([]))
    # n_clusters_selected est déjà défini globalement, mais s'assurer qu'il est cohérent si besoin.

    if not vip_data_df.empty:
        vip_styled = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)']).format({"Communalité (%)": "{:}%"})

    if not pdf.empty:
        if "Cluster" not in pdf.columns: pdf["Cluster"] = "0" # Fallback
        # --- Préparation Figure PCA ---
        fig_pca = px.scatter(pdf, x="PC1", y="PC2" if pdf.shape[1] > 1 and "PC2" in pdf.columns else None,
                             color="Cluster", text="Espece_User", hover_name="Espece_User",
                             custom_data=["Espece_User", "Ecologie"], template="plotly_dark",
                             height=600, color_discrete_sequence=COLOR_SEQUENCE)
        fig_pca.update_traces(textposition="top center", marker=dict(opacity=0.7),
                               hovertemplate=("<b>%{customdata[0]}</b><br><br><i>Écologie:</i><br>%{customdata[1]}<extra></extra>"))
        unique_clusters_pca = sorted(pdf["Cluster"].unique())
        cluster_color_map_pca = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_pca)}
        if "PC2" in pdf.columns and pdf.shape[1] > 1: # Assure que PC2 existe pour les enveloppes
            for cluster_label in unique_clusters_pca:
                cluster_points_df_pca = pdf[pdf["Cluster"] == cluster_label]
                if "PC1" in cluster_points_df_pca.columns and "PC2" in cluster_points_df_pca.columns:
                    unique_cluster_points_pca = cluster_points_df_pca[["PC1", "PC2"]].drop_duplicates().values
                    if len(unique_cluster_points_pca) >= MIN_POINTS_FOR_HULL:
                        try:
                            hull_pca = ConvexHull(unique_cluster_points_pca)
                            hull_path = unique_cluster_points_pca[np.append(hull_pca.vertices, hull_pca.vertices[0])]
                            clr = cluster_color_map_pca.get(cluster_label, COLOR_SEQUENCE[0])
                            fig_pca.add_trace(go.Scatter(x=hull_path[:, 0], y=hull_path[:, 1], fill="toself",
                                                          fillcolor=clr, line=dict(color=clr, width=1.5),
                                                          mode='lines', name=f'Cluster {cluster_label} Hull',
                                                          opacity=0.2, showlegend=False, hoverinfo='skip'))
                        except Exception as e: print(f"Erreur calcul enveloppe convexe ACP pour cluster {cluster_label}: {e}")
        fig_pca.update_layout(title_text="Clusters d'espèces (ACP)", title_x=0.5, legend_title_text='Cluster', dragmode='pan')

    # --- Préparation Dendrogramme ---
    if X_for_dendro.shape[0] > 1 and not pdf.empty: # Vérifier aussi que pdf n'est pas vide pour les labels
        Z = linkage(X_for_dendro, method="ward")
        dyn_thresh = 0
        # Utiliser n_clusters_selected de la session si disponible et pertinent, sinon celui du slider.
        # Pour le dendrogramme, n_clusters_selected (du slider) est pertinent pour la coloration.
        if n_clusters_selected > 1 and (n_clusters_selected - 1) <= Z.shape[0]:
            if (-(n_clusters_selected - 1) + Z.shape[0] >= 0): # Vérifie que l'index n'est pas hors limites
                dyn_thresh = Z[-(n_clusters_selected - 1), 2] * 0.99 # Seuil dynamique pour la coloration
            elif Z.shape[0] > 0: dyn_thresh = Z[0, 2] / 2 # Fallback si n_clusters_selected est trop grand
        elif Z.shape[0] > 0: dyn_thresh = Z[0, 2] / 2 # Fallback si Z a des lignes mais n_clusters_selected <= 1

        fig_dend = ff.create_dendrogram(X_for_dendro, orientation="left",
                                        labels=pdf["Espece_User"].tolist(),
                                        linkagefun=lambda _: Z,
                                        color_threshold=dyn_thresh if n_clusters_selected > 1 else 0, # Pas de coloration si 1 cluster
                                        colorscale=COLOR_SEQUENCE)
        fig_dend.update_layout(template="plotly_dark", height=max(650, sub.shape[0] * 20), # Hauteur dynamique
                               title_text="Dendrogramme", title_x=0.5)
    else:
        fig_dend = None

    cluster_compositions_data = [{"cluster_label": c, "count": len(pdf.loc[pdf["Cluster"] == c, "Espece_User"].unique()), "species_list": sorted(list(pdf.loc[pdf["Cluster"] == c, "Espece_User"].unique()))} for c in sorted(pdf["Cluster"].unique())] if not pdf.empty else []


# ---------------------------------------------------------------------------- #
# SECTION SUPERIEURE - PARTIE 2: IMPORTANCE DES VARIABLES (ACP)
# ---------------------------------------------------------------------------- #
with col_vip_main:
    st.subheader("Importance des Variables (ACP)")
    if st.session_state.run_main_analysis_once and vip_styled is not None:
        st.write(vip_styled.to_html(escape=False), unsafe_allow_html=True)
    elif st.session_state.run_main_analysis_once and (vip_data_df.empty if 'vip_data_df' in st.session_state else True):
        st.info("Le tableau d'importance des variables (ACP) n'a pas pu être généré ou est vide.")
    elif not st.session_state.run_main_analysis_once and not ref.empty:
         st.info("Sera affiché après l'analyse.")
    elif ref.empty:
        st.info("Chargement des données de référence requis.")


# ---------------------------------------------------------------------------- #
# SECTION SUPERIEURE - PARTIE 3: SÉLECTION POUR EXPLORATION INTERACTIVE
# ---------------------------------------------------------------------------- #
with col_interactive_selection_main:
    st.subheader("🔬 Exploration interactive")
    if st.session_state.run_main_analysis_once and not sub.empty:
        potential_traits_interactive = [col for col in sub.columns if col.lower() != "espece"]
        numeric_trait_names_interactive = sorted([col for col in potential_traits_interactive if pd.api.types.is_numeric_dtype(sub[col])]) # Trié pour la cohérence

        if len(numeric_trait_names_interactive) >= 1:
            st.markdown("###### Sélectionnez les variables pour les axes du nuage de points :")
            # Utiliser les valeurs de session_state comme défauts, sinon les premiers traits
            default_x_idx = numeric_trait_names_interactive.index(st.session_state.x_axis_trait_interactive) if st.session_state.x_axis_trait_interactive in numeric_trait_names_interactive else 0
            default_y_idx = 0
            if len(numeric_trait_names_interactive) > 1:
                default_y_idx = numeric_trait_names_interactive.index(st.session_state.y_axis_trait_interactive) if st.session_state.y_axis_trait_interactive in numeric_trait_names_interactive else 1
            elif len(numeric_trait_names_interactive) == 1 : # Si un seul trait, l'index pour Y est aussi 0
                default_y_idx = 0


            x_axis_trait_selected_interactive = st.radio("Axe X:", numeric_trait_names_interactive, index=default_x_idx, key="interactive_x_radio_main")

            if len(numeric_trait_names_interactive) > 1:
                 y_axis_trait_selected_interactive = st.radio("Axe Y:", numeric_trait_names_interactive, index=default_y_idx, key="interactive_y_radio_main")
            else:
                y_axis_trait_selected_interactive = x_axis_trait_selected_interactive
                st.caption("Un seul trait numérique disponible. Utilisé pour X et Y.")

            # Mise à jour de l'état de session immédiatement après la sélection par l'utilisateur
            st.session_state.x_axis_trait_interactive = x_axis_trait_selected_interactive
            st.session_state.y_axis_trait_interactive = y_axis_trait_selected_interactive

        elif st.session_state.run_main_analysis_once: # run_main_analysis_once est vrai, mais pas de traits numériques
            st.info("Aucun trait numérique trouvé dans les données sélectionnées pour l'exploration interactive.")
        # Si l'analyse n'a pas tourné, le message global s'applique
    elif not st.session_state.run_main_analysis_once and not ref.empty:
        st.info("Sera disponible après l'analyse.")
    elif ref.empty:
        st.info("Chargement des données de référence requis.")


# Message d'attente global pour les graphiques si l'analyse n'a pas tourné
if not st.session_state.run_main_analysis_once and not ref.empty:
    st.info("Les graphiques (Exploration interactive, ACP, Dendrogramme) et la composition des clusters seront affichés ici après avoir lancé l'analyse principale.")
elif not st.session_state.run_main_analysis_once and ref.empty: # Cas où les données de référence ne sont pas chargées
     with col_input_main: st.warning("Les données de référence ('data_ref.csv') n'ont pas pu être chargées. L'application est limitée.", icon="🚫")


# ---------------------------------------------------------------------------- #
# SECTION 2: GRAPHIQUE D'EXPLORATION INTERACTIVE DES VARIABLES (PLEINE LARGEUR)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty and \
   st.session_state.x_axis_trait_interactive is not None and \
   st.session_state.y_axis_trait_interactive is not None:

    x_axis_trait_selected = st.session_state.x_axis_trait_interactive
    y_axis_trait_selected = st.session_state.y_axis_trait_interactive

    if not pdf.empty and len(sub) == len(pdf) and x_axis_trait_selected in sub.columns and y_axis_trait_selected in sub.columns:
        plot_data_interactive = pd.DataFrame({
            'Espece_User': pdf['Espece_User'].values,
            'Ecologie': pdf['Ecologie'].values,
            x_axis_trait_selected: sub[x_axis_trait_selected].values.copy(),
            y_axis_trait_selected: sub[y_axis_trait_selected].values.copy(),
            'Cluster': pdf['Cluster'].values
        })

        plot_data_to_use = plot_data_interactive.copy()
        # Jitter pour les points superposés
        temp_x_col_grp = "_temp_x_group_col_"
        temp_y_col_grp = "_temp_y_group_col_"
        plot_data_to_use[temp_x_col_grp] = plot_data_to_use[x_axis_trait_selected]
        plot_data_to_use[temp_y_col_grp] = plot_data_to_use[y_axis_trait_selected]
        duplicates_mask = plot_data_to_use.duplicated(subset=[temp_x_col_grp, temp_y_col_grp], keep=False)

        if duplicates_mask.any():
            # S'assurer que les colonnes sont de type float pour le jittering
            if not pd.api.types.is_float_dtype(plot_data_to_use[x_axis_trait_selected]):
                plot_data_to_use[x_axis_trait_selected] = plot_data_to_use[x_axis_trait_selected].astype(float)
            if not pd.api.types.is_float_dtype(plot_data_to_use[y_axis_trait_selected]):
                plot_data_to_use[y_axis_trait_selected] = plot_data_to_use[y_axis_trait_selected].astype(float)

            x_min_val, x_max_val = plot_data_to_use[x_axis_trait_selected].min(), plot_data_to_use[x_axis_trait_selected].max()
            y_min_val, y_max_val = plot_data_to_use[y_axis_trait_selected].min(), plot_data_to_use[y_axis_trait_selected].max()
            x_range_val = x_max_val - x_min_val if pd.notna(x_min_val) and pd.notna(x_max_val) else 0
            y_range_val = y_max_val - y_min_val if pd.notna(y_min_val) and pd.notna(y_max_val) else 0

            jitter_strength_x = x_range_val * 0.015 if x_range_val > 1e-9 else (abs(plot_data_to_use[x_axis_trait_selected].mean()) * 0.015 if abs(plot_data_to_use[x_axis_trait_selected].mean()) > 1e-9 else 0.015)
            jitter_strength_y = y_range_val * 0.015 if y_range_val > 1e-9 else (abs(plot_data_to_use[y_axis_trait_selected].mean()) * 0.015 if abs(plot_data_to_use[y_axis_trait_selected].mean()) > 1e-9 else 0.015)
            if abs(jitter_strength_x) < 1e-9: jitter_strength_x = 0.015 # Valeur minimale pour le jitter
            if abs(jitter_strength_y) < 1e-9: jitter_strength_y = 0.015

            grouped_for_jitter = plot_data_to_use[duplicates_mask].groupby([temp_x_col_grp, temp_y_col_grp])
            for _, group in grouped_for_jitter:
                num_duplicates_in_group = len(group)
                if num_duplicates_in_group > 1:
                    for i, idx in enumerate(group.index):
                        angle = 2 * np.pi * i / num_duplicates_in_group
                        offset_x = jitter_strength_x * np.cos(angle)
                        offset_y = jitter_strength_y * np.sin(angle)
                        plot_data_to_use.loc[idx, x_axis_trait_selected] += offset_x
                        plot_data_to_use.loc[idx, y_axis_trait_selected] += offset_y
        plot_data_to_use.drop(columns=[temp_x_col_grp, temp_y_col_grp], inplace=True)

        fig_interactive_scatter = px.scatter(
            plot_data_to_use, x=x_axis_trait_selected, y=y_axis_trait_selected,
            color="Cluster", text="Espece_User", hover_name="Espece_User",
            custom_data=["Espece_User", "Ecologie", x_axis_trait_selected, y_axis_trait_selected], # Pour le hovertemplate
            template="plotly_dark", height=600, color_discrete_sequence=COLOR_SEQUENCE
        )
        fig_interactive_scatter.update_traces(
            textposition="top center", marker=dict(opacity=0.8, size=8), textfont=dict(size=10),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>" +
                f"{x_axis_trait_selected}: %{{customdata[2]:.2f}}<br>" + # Affichage des valeurs des axes
                f"{y_axis_trait_selected}: %{{customdata[3]:.2f}}<br>" +
                "<br><i>Écologie:</i><br>%{customdata[1]}<extra></extra>"
            )
        )
        unique_clusters_interactive = sorted(plot_data_to_use["Cluster"].unique())
        cluster_color_map_interactive = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_interactive)}
        for cluster_label in unique_clusters_interactive:
            cluster_points_df_interactive = plot_data_to_use[plot_data_to_use["Cluster"] == cluster_label]
            if x_axis_trait_selected in cluster_points_df_interactive and y_axis_trait_selected in cluster_points_df_interactive:
                points_for_hull_interactive = cluster_points_df_interactive[[x_axis_trait_selected, y_axis_trait_selected]].drop_duplicates().values
                if len(points_for_hull_interactive) >= MIN_POINTS_FOR_HULL:
                    try:
                        hull_interactive = ConvexHull(points_for_hull_interactive)
                        hull_path_interactive = points_for_hull_interactive[np.append(hull_interactive.vertices, hull_interactive.vertices[0])]
                        clr_int = cluster_color_map_interactive.get(cluster_label, COLOR_SEQUENCE[0])
                        fig_interactive_scatter.add_trace(go.Scatter(
                            x=hull_path_interactive[:, 0], y=hull_path_interactive[:, 1], fill="toself", fillcolor=clr_int,
                            line=dict(color=clr_int, width=1.5), mode='lines', name=f'Cluster {cluster_label} Hull (Interactive)',
                            opacity=0.2, showlegend=False, hoverinfo='skip' ))
                    except Exception as e: print(f"Erreur calcul enveloppe convexe interactive pour cluster {cluster_label}: {e}")
        fig_interactive_scatter.update_layout(
            title_text=f"Exploration interactive : {y_axis_trait_selected} en fonction de {x_axis_trait_selected}", title_x=0.5,
            xaxis_title=x_axis_trait_selected, yaxis_title=y_axis_trait_selected, dragmode='pan'
        )
        st.markdown("---")
        st.plotly_chart(fig_interactive_scatter, use_container_width=True, config={'scrollZoom': True})

    elif st.session_state.run_main_analysis_once: # Si l'analyse a tourné mais conditions non remplies pour ce graphique
        if pdf.empty:
            st.info("Les données de l'ACP (pdf) sont vides, le graphique d'exploration interactive ne peut être généré.")
        elif len(sub) != len(pdf):
             st.warning("Incohérence entre le nombre d'espèces dans 'sub' et 'pdf'. Graphique interactif non généré.")
        elif not (x_axis_trait_selected in sub.columns and y_axis_trait_selected in sub.columns):
            st.warning(f"Les traits sélectionnés pour l'exploration interactive ('{x_axis_trait_selected}', '{y_axis_trait_selected}') ne sont pas valides ou présents dans les données.")


# ---------------------------------------------------------------------------- #
# SECTION 3: AFFICHAGE DU GRAPHIQUE ACP (PLEINE LARGEUR)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and fig_pca is not None:
    st.markdown("---")
    st.plotly_chart(fig_pca, use_container_width=True, config={'scrollZoom': True})
elif st.session_state.run_main_analysis_once and not sub.empty and fig_pca is None:
    st.info("Le graphique ACP n'a pas pu être généré. Vérifiez les données d'entrée et le nombre de clusters. Il se peut qu'il n'y ait pas assez de composantes principales (ex: si une seule espèce est analysée ou si toutes les espèces ont les mêmes valeurs pour les traits utilisés par l'ACP).")
# Si sub.empty, l'erreur est déjà affichée plus haut.

# ---------------------------------------------------------------------------- #
# SECTION 4: COMPOSITION DES CLUSTERS (EN COLONNES, PLEINE LARGEUR)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty and cluster_compositions_data:
    st.markdown("---")
    st.subheader("Composition des Clusters (ACP)")
    if any(d['count'] > 0 for d in cluster_compositions_data):
        num_clusters_found_display = len([d for d in cluster_compositions_data if d['count'] > 0])
        num_display_cols = min(num_clusters_found_display, 4) # Max 4 colonnes

        if num_display_cols > 0:
            cluster_cols_display = st.columns(num_display_cols)
            current_col_idx_display = 0
            for comp_data in cluster_compositions_data:
                if comp_data['count'] > 0:
                    with cluster_cols_display[current_col_idx_display % num_display_cols]:
                        st.markdown(f"**Cluster {comp_data['cluster_label']}** ({comp_data['count']} espèce{'s' if comp_data['count'] > 1 else ''})")
                        for species_name in comp_data['species_list']: st.markdown(f"- {species_name}")
                    current_col_idx_display += 1
        else: # Ce cas ne devrait pas arriver si any(d['count'] > 0) est vrai
            st.info("Aucun cluster (ACP) avec des espèces à afficher.")
    else:
        st.info("Aucune espèce n'a été assignée aux clusters ou les clusters sont vides.")
elif st.session_state.run_main_analysis_once and not sub.empty and not cluster_compositions_data:
     st.info("Les données de composition des clusters (ACP) ne sont pas disponibles.")


# ---------------------------------------------------------------------------- #
# SECTION 5: AFFICHAGE DU DENDROGRAMME (PLEINE LARGEUR, EN DERNIER)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and fig_dend is not None:
    st.markdown("---")
    st.plotly_chart(fig_dend, use_container_width=True)
elif st.session_state.run_main_analysis_once and not sub.empty and fig_dend is None:
    if X_for_dendro.shape[0] <= 1:
        st.info("Le dendrogramme n'a pas pu être généré car il nécessite au moins 2 espèces pour la classification hiérarchique.")
    elif pdf.empty :
        st.info("Le dendrogramme n'a pas pu être généré car les données des espèces (pdf) sont manquantes pour les étiquettes.")
    else:
        st.info("Le dendrogramme n'a pas pu être généré pour une autre raison (ex: problème avec les données pour la fonction `linkage`).")
