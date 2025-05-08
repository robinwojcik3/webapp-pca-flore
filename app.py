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
col_input, col_pca_plot_container = st.columns([1, 3])

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

fig_pca = None
fig_dend = None
vip_styled = None
vip_data_df = pd.DataFrame()
cluster_compositions_data = []
sub = pd.DataFrame()
pdf = pd.DataFrame()
X_for_dendro = np.array([])

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
        found_ref_binom_values_in_sub = ( sub["Espece"].str.split().str[:2].str.join(" ").str.lower().tolist() )

    not_found_user_raw_names = [species_raw_unique[i] for i, user_binom_name in enumerate(species_binom_user_unique) if user_binom_name not in found_ref_binom_values_in_sub]
    if not_found_user_raw_names:
        with col_input:
            st.warning("Non trouvées dans la base de traits : " + ", ".join(not_found_user_raw_names), icon="⚠️")

    if sub.empty:
        st.error("Aucune des espèces saisies n'a pu être utilisée pour l'analyse.")
        st.session_state.run_main_analysis_once = False; st.stop()
    if sub.shape[0] < n_clusters_selected and n_clusters_selected > 0 :
        st.error(f"Le nombre d'espèces trouvées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters_selected}).");
        st.session_state.run_main_analysis_once = False; st.stop()
    if sub.shape[0] < 2:
        st.error(f"Au moins 2 espèces sont nécessaires. {sub.shape[0]} trouvée(s).");
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
            elif len(top_vars_from_vip) == 1: default_y_init = ([t for t in numeric_trait_names_init if t != default_x_init] or [default_x_init])[0]
        if default_x_init is None and len(numeric_trait_names_init) >= 1: default_x_init = numeric_trait_names_init[0]
        if default_y_init is None:
            if len(numeric_trait_names_init) >= 2: default_y_init = numeric_trait_names_init[1] if numeric_trait_names_init[0] != numeric_trait_names_init[1] else numeric_trait_names_init[0]
            elif default_x_init: default_y_init = default_x_init
        st.session_state.x_axis_trait_interactive, st.session_state.y_axis_trait_interactive = default_x_init, default_y_init
    except Exception as e:
        st.error(f"Erreur analyse ACP : {e}"); st.exception(e)
        st.session_state.run_main_analysis_once = False; st.stop()

if st.session_state.run_main_analysis_once:
    sub = st.session_state.get('sub', pd.DataFrame())
    pdf = st.session_state.get('pdf', pd.DataFrame())
    vip_data_df = st.session_state.get('vip_data_df', pd.DataFrame())
    X_for_dendro = st.session_state.get('X_for_dendro', np.array([]))

    if not pdf.empty:
        if "Cluster" not in pdf.columns: pdf["Cluster"] = "0"
        fig_pca = px.scatter(pdf, x="PC1", y="PC2" if "PC2" in pdf.columns else None, color="Cluster", text="Espece_User", hover_name="Espece_User", custom_data=["Espece_User", "Ecologie"], template="plotly_dark", height=600, color_discrete_sequence=COLOR_SEQUENCE)
        fig_pca.update_traces(textposition="top center", marker=dict(opacity=0.7), hovertemplate=("<b>%{customdata[0]}</b><br><br><i>Écologie:</i><br>%{customdata[1]}<extra></extra>"))
        unique_clusters_pca = sorted(pdf["Cluster"].unique())
        cluster_color_map_pca = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_pca)}
        if "PC2" in pdf.columns:
            for cluster_label in unique_clusters_pca:
                cluster_points_df_pca = pdf[pdf["Cluster"] == cluster_label]
                if "PC1" in cluster_points_df_pca.columns and "PC2" in cluster_points_df_pca.columns:
                    unique_cluster_points_pca = cluster_points_df_pca[["PC1", "PC2"]].drop_duplicates().values
                    if len(unique_cluster_points_pca) >= MIN_POINTS_FOR_HULL:
                        try:
                            hull_pca = ConvexHull(unique_cluster_points_pca)
                            hull_path = unique_cluster_points_pca[np.append(hull_pca.vertices, hull_pca.vertices[0])] # Closed path
                            clr = cluster_color_map_pca.get(cluster_label, COLOR_SEQUENCE[0])
                            fig_pca.add_trace(go.Scatter(x=hull_path[:, 0], y=hull_path[:, 1], fill="toself", fillcolor=clr, line=dict(color=clr, width=1.5), mode='lines', name=f'Cluster {cluster_label} Hull', opacity=0.2, showlegend=False, hoverinfo='skip'))
                        except Exception as e: print(f"Hull ACP {cluster_label}: {e}")
        fig_pca.update_layout(title_text="Clusters d'espèces (ACP)", title_x=0.5, legend_title_text='Cluster')

        if not vip_data_df.empty: vip_styled = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)']).format({"Communalité (%)": "{:}%"})
        cluster_compositions_data = [{"cluster_label": c, "count": len(pdf.loc[pdf["Cluster"] == c, "Espece_User"].unique()), "species_list": sorted(list(pdf.loc[pdf["Cluster"] == c, "Espece_User"].unique()))} for c in sorted(pdf["Cluster"].unique())]
        if X_for_dendro.shape[0] > 1:
            Z = linkage(X_for_dendro, method="ward")
            dyn_thresh = 0
            if n_clusters_selected > 1 and (n_clusters_selected -1) <= Z.shape[0] : # Check index bounds
                threshold_idx = max(0, Z.shape[0] - (n_clusters_selected -1) ) # Ensure non-negative index for Z
                if threshold_idx < Z.shape[0]: # Ensure index is within bounds
                     dyn_thresh = Z[threshold_idx, 2] * 0.99 # Access distance for cutting
                     # A common heuristic for color_threshold with n_clusters: Z[-(n_clusters_selected-1), 2] if n_clusters_selected > 1
                     # If n_clusters_selected = 2, we want Z[-1,2]. If n_clusters_selected = N, Z[-(N-1),2]
                     if n_clusters_selected > 1 and (n_clusters_selected -1) <= Z.shape[0] :
                         dyn_thresh = Z[-(n_clusters_selected-1), 2] * 0.99 # distance of the (k-1)th last merge
                     elif Z.shape[0] > 0: dyn_thresh = Z[0, 2] / 2 # Fallback
                elif Z.shape[0] > 0: dyn_thresh = Z[0,2] / 2
            fig_dend = ff.create_dendrogram(X_for_dendro, orientation="left", labels=pdf["Espece_User"].tolist(), linkagefun=lambda _: Z, color_threshold=dyn_thresh if n_clusters_selected > 1 else 0, colorscale=COLOR_SEQUENCE)
            fig_dend.update_layout(template="plotly_dark", height=max(650, sub.shape[0] * 20), title_text="Dendrogramme", title_x=0.5)
        else: fig_dend = None

# ---------------------------------------------------------------------------- #
# SECTION 1: AFFICHAGE DU GRAPHIQUE ACP
# ---------------------------------------------------------------------------- #
with col_pca_plot_container:
    if fig_pca: st.plotly_chart(fig_pca, use_container_width=True)
    elif run_main_analysis_button and ref.empty: st.warning("Chargez les données de traits pour l'ACP.")
    elif run_main_analysis_button and sub.empty : st.warning("Aucune espèce valide pour l'analyse ACP.")
    elif st.session_state.run_main_analysis_once and not fig_pca: st.info("Graphique ACP affiché ici après analyse.")
    elif not st.session_state.run_main_analysis_once and not ref.empty: st.info("Prêt à lancer l'analyse.")
if not st.session_state.run_main_analysis_once and ref.empty:
    with col_input: st.warning("Données de référence non chargées ('data_ref.csv').")

# ---------------------------------------------------------------------------- #
# SECTION 2: EXPLORATION INTERACTIVE DES VARIABLES
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty:
    st.markdown("---"); st.subheader("🔬 Exploration interactive des variables")
    numeric_trait_names = [col for col in sub.columns if col.lower() != "espece" and pd.api.types.is_numeric_dtype(sub[col])]

    if len(numeric_trait_names) >= 2:
        st.markdown("##### Sélectionnez les variables pour les axes :");
        default_x = st.session_state.get('x_axis_trait_interactive', numeric_trait_names[0])
        default_y = st.session_state.get('y_axis_trait_interactive', numeric_trait_names[1] if len(numeric_trait_names)>1 else numeric_trait_names[0])
        if default_x not in numeric_trait_names: default_x = numeric_trait_names[0]
        if default_y not in numeric_trait_names: default_y = numeric_trait_names[1] if len(numeric_trait_names)>1 else numeric_trait_names[0]
        
        col_sel_x, col_sel_y = st.columns(2)
        with col_sel_x: x_axis = st.radio("Axe X:", numeric_trait_names, index=numeric_trait_names.index(default_x), key="ixr")
        with col_sel_y: y_axis = st.radio("Axe Y:", numeric_trait_names, index=numeric_trait_names.index(default_y), key="iyr")
        st.session_state.x_axis_trait_interactive, st.session_state.y_axis_trait_interactive = x_axis, y_axis

        if x_axis and y_axis and not pdf.empty and len(sub) == len(pdf):
            # plot_data_interactive contient les positions originales des points
            plot_data_interactive = pd.DataFrame({
                'Espece_User': pdf['Espece_User'].values, 'Ecologie': pdf['Ecologie'].values,
                x_axis: sub[x_axis].values, y_axis: sub[y_axis].values,
                'Cluster': pdf['Cluster'].values
            })

            # MODIFICATION: Gestion des étiquettes de texte via annotations
            custom_annotations = []
            # Identifier les groupes de points avec les mêmes coordonnées (basé sur les valeurs originales)
            # Utiliser des noms de colonnes temporaires sûrs pour groupby
            temp_x_col_grp = "_temp_x_grp_"
            temp_y_col_grp = "_temp_y_grp_"
            plot_data_interactive[temp_x_col_grp] = plot_data_interactive[x_axis]
            plot_data_interactive[temp_y_col_grp] = plot_data_interactive[y_axis]
            
            duplicates_mask = plot_data_interactive.duplicated(subset=[temp_x_col_grp, temp_y_col_grp], keep=False)

            # Points non superposés : texte simple au-dessus du point
            for idx, row in plot_data_interactive[~duplicates_mask].iterrows():
                custom_annotations.append(go.layout.Annotation(
                        x=row[x_axis], y=row[y_axis], text=row['Espece_User'],
                        showarrow=False, xanchor='center', yanchor='bottom', yshift=7, font=dict(size=9) ))
            
            # Points superposés : décaler les textes circulairement
            if duplicates_mask.any():
                points_to_adjust_text = plot_data_interactive[duplicates_mask]
                
                x_min, x_max = plot_data_interactive[x_axis].min(), plot_data_interactive[x_axis].max()
                y_min, y_max = plot_data_interactive[y_axis].min(), plot_data_interactive[y_axis].max()
                x_range = x_max - x_min; y_range = y_max - y_min

                offset_scale_x = (x_range * 0.025 if x_range > 1e-9 else (abs(plot_data_interactive[x_axis].mean()) * 0.025 if abs(plot_data_interactive[x_axis].mean()) > 1e-9 else 0.025))
                offset_scale_y = (y_range * 0.025 if y_range > 1e-9 else (abs(plot_data_interactive[y_axis].mean()) * 0.025 if abs(plot_data_interactive[y_axis].mean()) > 1e-9 else 0.025))
                if abs(offset_scale_x) < 1e-9: offset_scale_x = 0.025
                if abs(offset_scale_y) < 1e-9: offset_scale_y = 0.025

                grouped_for_text = points_to_adjust_text.groupby([temp_x_col_grp, temp_y_col_grp])
                for (original_x_val, original_y_val), group_df in grouped_for_text:
                    n_duplicates = len(group_df)
                    for i, (idx_g, row_g) in enumerate(group_df.iterrows()):
                        angle = (2 * np.pi * i / n_duplicates) + (np.pi / (2*n_duplicates)) # Offset angle slightly for better spacing
                        # Position du texte décalée par rapport au point original
                        text_x = original_x_val + offset_scale_x * np.cos(angle)
                        text_y = original_y_val + offset_scale_y * np.sin(angle)
                        custom_annotations.append(go.layout.Annotation(
                            x=text_x, y=text_y, ax=original_x_val, ay=original_y_val, # Ancres pour flèches (si showarrow=True)
                            text=row_g['Espece_User'], showarrow=False, # Mettre showarrow=True pour des petites flèches
                            # arrowhead=2, arrowsize=0.8, arrowwidth=1, # Options de flèche
                            xanchor='center', yanchor='middle', font=dict(size=9) ))
            
            plot_data_interactive.drop(columns=[temp_x_col_grp, temp_y_col_grp], inplace=True)
            # FIN MODIFICATION ANNOTATIONS

            # Créer la figure SANS le paramètre 'text' pour px.scatter, car géré par les annotations
            fig_interactive_scatter = px.scatter(
                plot_data_interactive, x=x_axis, y=y_axis, color="Cluster",
                hover_name="Espece_User",
                custom_data=["Espece_User", "Ecologie", x_axis, y_axis], # Valeurs originales pour hover
                template="plotly_dark", height=600, color_discrete_sequence=COLOR_SEQUENCE
            )
            fig_interactive_scatter.update_layout(annotations=custom_annotations) # Ajouter les annotations calculées
            fig_interactive_scatter.update_traces(
                marker=dict(opacity=0.8, size=8), # Pas de textposition, géré par annotations
                hovertemplate=("<b>%{customdata[0]}</b><br>" +
                               f"<br><i>{x_axis}:</i> %{{customdata[2]}}<br>" +
                               f"<i>{y_axis}:</i> %{{customdata[3]}}<br>" +
                               "<br><i>Écologie:</i><br>%{customdata[1]}<extra></extra>")
            )
            # Convex Hull basé sur les positions originales des points (plot_data_interactive)
            unique_clusters_interactive = sorted(plot_data_interactive["Cluster"].unique())
            cluster_color_map_interactive = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_interactive)}
            for cluster_label in unique_clusters_interactive:
                cluster_points_df = plot_data_interactive[plot_data_interactive["Cluster"] == cluster_label]
                if x_axis in cluster_points_df and y_axis in cluster_points_df:
                    points_for_hull = cluster_points_df[[x_axis, y_axis]].drop_duplicates().values
                    if len(points_for_hull) >= MIN_POINTS_FOR_HULL:
                        try:
                            hull_interactive = ConvexHull(points_for_hull)
                            hull_path_interactive = points_for_hull[np.append(hull_interactive.vertices, hull_interactive.vertices[0])]
                            clr_int = cluster_color_map_interactive.get(cluster_label, COLOR_SEQUENCE[0])
                            fig_interactive_scatter.add_trace(go.Scatter(
                                x=hull_path_interactive[:, 0], y=hull_path_interactive[:, 1], fill="toself", fillcolor=clr_int,
                                line=dict(color=clr_int, width=1.5), mode='lines', name=f'Hull Cl. {cluster_label}',
                                opacity=0.2, showlegend=False, hoverinfo='skip' ))
                        except Exception as e: print(f"Hull Inter.: {cluster_label} ({x_axis}, {y_axis}): {e}")
            fig_interactive_scatter.update_layout(title_text=f"Variables: {y_axis} / {x_axis}", title_x=0.5, xaxis_title=x_axis, yaxis_title=y_axis)
            st.plotly_chart(fig_interactive_scatter, use_container_width=True)
        elif not (x_axis and y_axis): st.warning("Sélectionnez variables X et Y.")
        elif pdf.empty or len(sub) != len(pdf): st.warning("Données pour graphique interactif non prêtes.")
    elif len(numeric_trait_names) == 1: st.warning("Au moins deux traits numériques requis pour nuage de points 2D.")
    else: st.warning("Aucun trait numérique pour exploration interactive.")

# ---------------------------------------------------------------------------- #
# SECTION 3: IMPORTANCE DES VARIABLES ET COMPOSITION DES CLUSTERS
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty:
    col_vars, col_cluster_comp = st.columns([1, 2])
    with col_vars:
        st.subheader("Importance des Variables (ACP)")
        if vip_styled is not None: st.write(vip_styled.to_html(escape=False), unsafe_allow_html=True)
        else: st.info("Tableau d'importance des variables affiché ici.")
    with col_cluster_comp:
        st.subheader("Composition des Clusters (ACP)")
        if cluster_compositions_data and any(d['count'] > 0 for d in cluster_compositions_data):
            num_cl_found = len([d for d in cluster_compositions_data if d['count']>0])
            cols_cl = st.columns(min(num_cl_found, 3))
            col_idx = 0
            for comp_data in cluster_compositions_data:
                if comp_data['count'] > 0:
                    with cols_cl[col_idx % min(num_cl_found, 3)]:
                        st.markdown(f"**Cluster {comp_data['cluster_label']}** ({comp_data['count']} espèces)")
                        for sp_name in comp_data['species_list']: st.markdown(f"- {sp_name}")
                    col_idx +=1
        else: st.info("Composition des clusters affichée ici.")

# ---------------------------------------------------------------------------- #
# SECTION 4: AFFICHAGE DU DENDROGRAMME
# ---------------------------------------------------------------------------- #
if fig_dend: st.plotly_chart(fig_dend, use_container_width=True)
elif st.session_state.run_main_analysis_once and not sub.empty and species_binom_user_unique:
    st.info("Dendrogramme non généré (conditions non remplies).")
