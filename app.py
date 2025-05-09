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

st.markdown("""
<style>
div[data-testid="stDataEditor"] {
    font-size: 14px;
}
div[data-testid="stDataEditor"] .glideDataEditor-header {
    font-size: 15px !important;
}
div[data-testid="stDataEditor"] table, 
div[data-testid="stDataEditor"] th, 
div[data-testid="stDataEditor"] td {
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------- #
# CONSTANTES ET CHARGEMENT DE DONNÉES INITIALES
# ---------------------------------------------------------------------------- #
MIN_POINTS_FOR_HULL = 3
COLOR_SEQUENCE = px.colors.qualitative.Plotly
LABEL_FONT_SIZE_ON_PLOTS = 15 # Augmenté pour une meilleure lisibilité
HOVER_SPECIES_FONT_SIZE = 15    # Taille du nom de l'espèce dans le hover
HOVER_ECOLOGY_TITLE_FONT_SIZE = 14 # Taille du titre "Écologie"
HOVER_ECOLOGY_TEXT_FONT_SIZE = 13  # Taille du texte de l'écologie

@st.cache_data
def load_data(file_path="data_ref.csv"):
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
    if pd.isna(text) or text.strip() == "":
        return "Description écologique non disponible."
    wrapped_lines = textwrap.wrap(text, width=line_width_chars)
    return "<br>".join(wrapped_lines)

# ---------------------------------------------------------------------------- #
# CHARGEMENT DE LA BASE ECOLOGIQUE
# ---------------------------------------------------------------------------- #
@st.cache_data
def load_ecology_data(file_path="data_ecologie_espece.csv"):
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
if 'vip_data_df_interactive' not in st.session_state: 
    st.session_state.vip_data_df_interactive = pd.DataFrame()
if 'vip_data_df_interactive_snapshot_for_comparison' not in st.session_state:
    st.session_state.vip_data_df_interactive_snapshot_for_comparison = pd.DataFrame()


# ---------------------------------------------------------------------------- #
# SECTION 1: ENTRÉES UTILISATEUR ET GRAPHIQUE ACP
# ---------------------------------------------------------------------------- #
col_input_user, col_pca_plot = st.columns([1, 2]) 

with col_input_user:
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
vip_data_df_for_calc = pd.DataFrame()
cluster_compositions_data = []
sub = pd.DataFrame()
pdf = pd.DataFrame()
X_for_dendro = np.array([])
numeric_trait_names_for_interactive_plot = []


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
        with col_input_user: 
            st.warning("Non trouvées dans la base de traits : " + ", ".join(not_found_user_raw_names), icon="⚠️")

    if sub.empty:
        st.error("Aucune des espèces saisies (après déduplication et recherche dans la base de traits) n'a pu être utilisée pour l'analyse.")
        st.session_state.run_main_analysis_once = False; st.stop()
    if sub.shape[0] < n_clusters_selected and n_clusters_selected > 0 :
        st.error(f"Le nombre d'espèces uniques trouvées et utilisées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters_selected}).");
        st.session_state.run_main_analysis_once = False; st.stop()
    if sub.shape[0] < 2:
        st.error(f"Au moins 2 espèces uniques sont nécessaires pour l'analyse. {sub.shape[0]} espèce(s) trouvée(s) et utilisée(s).");
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
        
        st.session_state.vip_data_df_for_calc = pd.DataFrame({
            "Variable": trait_columns,
            "Communalité (%)": (communal * 100).round(0).astype(int),
        }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
        
        st.session_state.X_for_dendro = X

        all_trait_names_from_sub = [col for col in sub.columns if col.lower() != "espece"]
        st.session_state.numeric_trait_names_for_interactive_plot = [
            col for col in all_trait_names_from_sub if pd.api.types.is_numeric_dtype(sub[col])
        ]
        
        numeric_trait_names_init = st.session_state.numeric_trait_names_for_interactive_plot
        default_x_init, default_y_init = None, None

        if not st.session_state.vip_data_df_for_calc.empty and len(numeric_trait_names_init) >= 1:
            top_vars_from_vip_numeric = [
                var for var in st.session_state.vip_data_df_for_calc["Variable"].tolist()
                if var in numeric_trait_names_init
            ]
            if len(top_vars_from_vip_numeric) >= 1: default_x_init = top_vars_from_vip_numeric[0]
            if len(top_vars_from_vip_numeric) >= 2: default_y_init = top_vars_from_vip_numeric[1]
            elif len(top_vars_from_vip_numeric) == 1: 
                other_numeric_traits = [t for t in numeric_trait_names_init if t != default_x_init]
                default_y_init = other_numeric_traits[0] if other_numeric_traits else default_x_init
        
        if default_x_init is None and len(numeric_trait_names_init) >= 1:
            default_x_init = numeric_trait_names_init[0]
        if default_y_init is None:
            if len(numeric_trait_names_init) >= 2:
                default_y_init = numeric_trait_names_init[1]
            elif default_x_init and len(numeric_trait_names_init) == 1: 
                 default_y_init = default_x_init

        st.session_state.x_axis_trait_interactive = default_x_init
        st.session_state.y_axis_trait_interactive = default_y_init
        
        if not st.session_state.vip_data_df_for_calc.empty and numeric_trait_names_init:
            temp_interactive_df = st.session_state.vip_data_df_for_calc[
                st.session_state.vip_data_df_for_calc["Variable"].isin(numeric_trait_names_init)
            ].copy()
            temp_interactive_df["Axe X"] = temp_interactive_df["Variable"] == st.session_state.x_axis_trait_interactive
            temp_interactive_df["Axe Y"] = temp_interactive_df["Variable"] == st.session_state.y_axis_trait_interactive
            st.session_state.vip_data_df_interactive = temp_interactive_df[["Variable", "Communalité (%)", "Axe X", "Axe Y"]].reset_index(drop=True)
            st.session_state.vip_data_df_interactive_snapshot_for_comparison = st.session_state.vip_data_df_interactive.copy()
        else:
            st.session_state.vip_data_df_interactive = pd.DataFrame(columns=["Variable", "Communalité (%)", "Axe X", "Axe Y"])
            st.session_state.vip_data_df_interactive_snapshot_for_comparison = pd.DataFrame(columns=["Variable", "Communalité (%)", "Axe X", "Axe Y"])


    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'analyse ACP : {e}"); st.exception(e)
        st.session_state.run_main_analysis_once = False; st.stop()

if st.session_state.run_main_analysis_once:
    sub = st.session_state.get('sub', pd.DataFrame())
    pdf = st.session_state.get('pdf', pd.DataFrame())
    X_for_dendro = st.session_state.get('X_for_dendro', np.array([]))
    numeric_trait_names_for_interactive_plot = st.session_state.get('numeric_trait_names_for_interactive_plot', [])

    if not pdf.empty:
        if "Cluster" not in pdf.columns: pdf["Cluster"] = "0" 
        fig_pca = px.scatter(pdf, x="PC1", y="PC2" if pdf.shape[1] > 1 and "PC2" in pdf.columns else None, 
                             color="Cluster", text="Espece_User", hover_name="Espece_User", 
                             custom_data=["Espece_User", "Ecologie"], template="plotly_dark", height=500, 
                             color_discrete_sequence=COLOR_SEQUENCE)
        fig_pca.update_traces(textposition="top center", marker=dict(opacity=0.7), 
                              hovertemplate=(
                                  f"<span style='font-size: {HOVER_SPECIES_FONT_SIZE}px;'><b>%{{customdata[0]}}</b></span><br><br>"
                                  f"<span style='font-size: {HOVER_ECOLOGY_TITLE_FONT_SIZE}px;'><i>Écologie:</i></span><br>"
                                  f"<span style='font-size: {HOVER_ECOLOGY_TEXT_FONT_SIZE}px;'>%{{customdata[1]}}</span>"
                                  "<extra></extra>"
                              ),
                              textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS)) 
        unique_clusters_pca = sorted(pdf["Cluster"].unique())
        cluster_color_map_pca = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_pca)}
        
        if "PC2" in pdf.columns and pdf.shape[1] > 1 : 
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
                        except Exception as e: print(f"Erreur calcul Hull ACP pour cluster {cluster_label}: {e}")
        fig_pca.update_layout(title_text="Plot PCA", title_x=0.5, legend_title_text='Cluster')
        fig_pca.update_layout(dragmode='pan')

    cluster_compositions_data = [{"cluster_label": c, "count": len(pdf.loc[pdf["Cluster"] == c, "Espece_User"].unique()), "species_list": sorted(list(pdf.loc[pdf["Cluster"] == c, "Espece_User"].unique()))} for c in sorted(pdf["Cluster"].unique())]
    
    if X_for_dendro.shape[0] > 1:
        Z = linkage(X_for_dendro, method="ward")
        dyn_thresh = 0
        if n_clusters_selected > 1 and (n_clusters_selected -1) <= Z.shape[0] : 
            if (-(n_clusters_selected-1) + Z.shape[0] >=0): 
                dyn_thresh = Z[-(n_clusters_selected-1), 2] * 0.99 
            elif Z.shape[0] > 0 : dyn_thresh = Z[0, 2] / 2 
        elif Z.shape[0] > 0: dyn_thresh = Z[0, 2] / 2
        
        dendro_labels = pdf["Espece_User"].tolist() if not pdf.empty and "Espece_User" in pdf.columns and len(pdf) == X_for_dendro.shape[0] else [f"Esp {i+1}" for i in range(X_for_dendro.shape[0])]

        fig_dend = ff.create_dendrogram(X_for_dendro, orientation="left", labels=dendro_labels, 
                                        linkagefun=lambda _: Z, color_threshold=dyn_thresh if n_clusters_selected > 1 else 0, 
                                        colorscale=COLOR_SEQUENCE)
        fig_dend.update_layout(template="plotly_dark", height=max(400, sub.shape[0] * 20 if not sub.empty else 400), title_text="Dendrogramme", title_x=0.5)
    else: fig_dend = None

with col_pca_plot:
    if fig_pca: 
        st.plotly_chart(fig_pca, use_container_width=True, config={'scrollZoom': True}) 
    elif run_main_analysis_button and ref.empty: st.warning("Veuillez d'abord charger des données de traits pour afficher le graphique ACP.")
    elif run_main_analysis_button and (sub.empty if 'sub' in locals() else True) : st.warning("Aucune espèce valide pour l'analyse ACP.")
    elif st.session_state.run_main_analysis_once and not fig_pca: st.info("Le graphique ACP sera affiché ici après une analyse principale réussie.")
    elif not st.session_state.run_main_analysis_once and not ref.empty: st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse principale'.")

if not st.session_state.run_main_analysis_once and ref.empty:
    with col_input_user: st.warning("Les données de référence n'ont pas pu être chargées. Vérifiez le fichier 'data_ref.csv'.")

# ---------------------------------------------------------------------------- #
# SECTION 2: EXPLORATION INTERACTIVE DES VARIABLES (MILIEU DE PAGE)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty:
    st.markdown("---")
    col_interactive_table, col_interactive_graph = st.columns([2, 3]) 

    with col_interactive_table:
        st.markdown("##### Tableau d'exploration interactif des variables")
        df_editor_source = st.session_state.get('vip_data_df_interactive', pd.DataFrame(columns=["Variable", "Communalité (%)", "Axe X", "Axe Y"]))

        if not df_editor_source.empty:
            if 'vip_data_df_interactive_snapshot_for_comparison' not in st.session_state or \
               st.session_state.vip_data_df_interactive_snapshot_for_comparison.empty:
                st.session_state.vip_data_df_interactive_snapshot_for_comparison = df_editor_source.copy()

            edited_df = st.data_editor(
                df_editor_source,
                column_config={
                    "Variable": st.column_config.TextColumn("Variable", disabled=True, help="Nom de la variable (trait)"),
                    "Communalité (%)": st.column_config.NumberColumn("Communalité (%)", format="%d%%", disabled=True, help="Communalité de la variable dans l'ACP"),
                    "Axe X": st.column_config.CheckboxColumn("Axe X", help="Sélectionner cette variable pour l'axe X du graphique d'exploration"),
                    "Axe Y": st.column_config.CheckboxColumn("Axe Y", help="Sélectionner cette variable pour l'axe Y du graphique d'exploration")
                },
                key="interactive_exploration_editor",
                use_container_width=True,
                hide_index=True,
                num_rows="fixed"
            )

            df_before_edit = st.session_state.vip_data_df_interactive_snapshot_for_comparison

            new_x_trait_from_editor = st.session_state.x_axis_trait_interactive
            for _, row in edited_df.iterrows():
                var_name = row["Variable"]
                is_checked_now = row["Axe X"]
                was_checked_before_series = df_before_edit[df_before_edit["Variable"] == var_name]["Axe X"]
                was_checked_before = was_checked_before_series.iloc[0] if not was_checked_before_series.empty else False
                if is_checked_now and not was_checked_before:
                    new_x_trait_from_editor = var_name
                    break 
                elif not is_checked_now and was_checked_before and var_name == st.session_state.x_axis_trait_interactive:
                    new_x_trait_from_editor = None

            new_y_trait_from_editor = st.session_state.y_axis_trait_interactive
            for _, row in edited_df.iterrows():
                var_name = row["Variable"]
                is_checked_now = row["Axe Y"]
                was_checked_before_series = df_before_edit[df_before_edit["Variable"] == var_name]["Axe Y"]
                was_checked_before = was_checked_before_series.iloc[0] if not was_checked_before_series.empty else False
                if is_checked_now and not was_checked_before:
                    new_y_trait_from_editor = var_name
                    break
                elif not is_checked_now and was_checked_before and var_name == st.session_state.y_axis_trait_interactive:
                    new_y_trait_from_editor = None

            needs_rerun = False
            if new_x_trait_from_editor != st.session_state.x_axis_trait_interactive or \
               new_y_trait_from_editor != st.session_state.y_axis_trait_interactive:
                st.session_state.x_axis_trait_interactive = new_x_trait_from_editor
                st.session_state.y_axis_trait_interactive = new_y_trait_from_editor
                needs_rerun = True
            
            num_x_checked_in_editor = edited_df["Axe X"].sum()
            num_y_checked_in_editor = edited_df["Axe Y"].sum()
            expected_x_checked = 1 if st.session_state.x_axis_trait_interactive else 0
            expected_y_checked = 1 if st.session_state.y_axis_trait_interactive else 0
            
            if num_x_checked_in_editor != expected_x_checked or num_y_checked_in_editor != expected_y_checked:
                needs_rerun = True

            if needs_rerun:
                updated_df_for_editor = st.session_state.vip_data_df_for_calc[
                    st.session_state.vip_data_df_for_calc["Variable"].isin(numeric_trait_names_for_interactive_plot)
                ].copy()
                updated_df_for_editor["Axe X"] = (updated_df_for_editor["Variable"] == st.session_state.x_axis_trait_interactive)
                updated_df_for_editor["Axe Y"] = (updated_df_for_editor["Variable"] == st.session_state.y_axis_trait_interactive)
                
                st.session_state.vip_data_df_interactive = updated_df_for_editor[["Variable", "Communalité (%)", "Axe X", "Axe Y"]].reset_index(drop=True)
                st.session_state.vip_data_df_interactive_snapshot_for_comparison = st.session_state.vip_data_df_interactive.copy()
                st.rerun()
            else:
                st.session_state.vip_data_df_interactive_snapshot_for_comparison = edited_df.copy()
        else:
            st.info("Le tableau d'exploration sera disponible après l'analyse si des traits numériques sont identifiés.")

    with col_interactive_graph:
        st.markdown("##### Graphique d'exploration des variables")
        x_axis_trait_selected_for_plot = st.session_state.x_axis_trait_interactive
        y_axis_trait_selected_for_plot = st.session_state.y_axis_trait_interactive

        if not numeric_trait_names_for_interactive_plot:
             st.warning("Aucun trait numérique trouvé pour l'exploration interactive.")
        elif not x_axis_trait_selected_for_plot or not y_axis_trait_selected_for_plot:
            st.info("Veuillez sélectionner une variable pour l'Axe X et une pour l'Axe Y dans le tableau à gauche.")
        elif x_axis_trait_selected_for_plot not in numeric_trait_names_for_interactive_plot or \
             y_axis_trait_selected_for_plot not in numeric_trait_names_for_interactive_plot:
            st.warning("Une ou les deux variables sélectionnées ne sont plus valides (ou pas numériques). Veuillez re-sélectionner.")
        elif sub.empty or pdf.empty or len(sub) != len(pdf):
             st.warning("Données pour le graphique interactif non prêtes ou incohérentes.")
        else:
            plot_data_interactive = pd.DataFrame({
                'Espece_User': pdf['Espece_User'].values,
                'Ecologie': pdf['Ecologie'].values,
                x_axis_trait_selected_for_plot: sub[x_axis_trait_selected_for_plot].values.copy(),
                y_axis_trait_selected_for_plot: sub[y_axis_trait_selected_for_plot].values.copy(),
                'Cluster': pdf['Cluster'].values
            })

            plot_data_to_use = plot_data_interactive.copy()
            temp_x_col_grp = "_temp_x_group_col_" 
            temp_y_col_grp = "_temp_y_group_col_"
            plot_data_to_use[temp_x_col_grp] = plot_data_to_use[x_axis_trait_selected_for_plot] 
            plot_data_to_use[temp_y_col_grp] = plot_data_to_use[y_axis_trait_selected_for_plot]
            duplicates_mask = plot_data_to_use.duplicated(subset=[temp_x_col_grp, temp_y_col_grp], keep=False)

            if duplicates_mask.any():
                x_min_val = plot_data_to_use[x_axis_trait_selected_for_plot].min()
                x_max_val = plot_data_to_use[x_axis_trait_selected_for_plot].max()
                y_min_val = plot_data_to_use[y_axis_trait_selected_for_plot].min()
                y_max_val = plot_data_to_use[y_axis_trait_selected_for_plot].max()
                x_range_val = x_max_val - x_min_val
                y_range_val = y_max_val - y_min_val
                
                jitter_strength_x = x_range_val * 0.015 if x_range_val > 1e-9 else (abs(plot_data_to_use[x_axis_trait_selected_for_plot].mean()) * 0.015 if abs(plot_data_to_use[x_axis_trait_selected_for_plot].mean()) > 1e-9 else 0.015)
                jitter_strength_y = y_range_val * 0.015 if y_range_val > 1e-9 else (abs(plot_data_to_use[y_axis_trait_selected_for_plot].mean()) * 0.015 if abs(plot_data_to_use[y_axis_trait_selected_for_plot].mean()) > 1e-9 else 0.015)
                if abs(jitter_strength_x) < 1e-9: jitter_strength_x = 0.015 
                if abs(jitter_strength_y) < 1e-9: jitter_strength_y = 0.015 

                grouped_for_jitter = plot_data_to_use[duplicates_mask].groupby([temp_x_col_grp, temp_y_col_grp])
                for _, group in grouped_for_jitter:
                    num_duplicates_in_group = len(group)
                    if num_duplicates_in_group > 1:
                        for i, idx in enumerate(group.index):
                            angle = 2 * np.pi * i / num_duplicates_in_group
                            offset_x = jitter_strength_x * np.cos(angle)
                            offset_y = jitter_strength_y * np.sin(angle)
                            
                            if not pd.api.types.is_float_dtype(plot_data_to_use[x_axis_trait_selected_for_plot]):
                                plot_data_to_use[x_axis_trait_selected_for_plot] = plot_data_to_use[x_axis_trait_selected_for_plot].astype(float)
                            if not pd.api.types.is_float_dtype(plot_data_to_use[y_axis_trait_selected_for_plot]):
                                plot_data_to_use[y_axis_trait_selected_for_plot] = plot_data_to_use[y_axis_trait_selected_for_plot].astype(float)
                            
                            plot_data_to_use.loc[idx, x_axis_trait_selected_for_plot] += offset_x
                            plot_data_to_use.loc[idx, y_axis_trait_selected_for_plot] += offset_y
            
            plot_data_to_use.drop(columns=[temp_x_col_grp, temp_y_col_grp], inplace=True) 

            fig_interactive_scatter = px.scatter(
                plot_data_to_use, x=x_axis_trait_selected_for_plot, y=y_axis_trait_selected_for_plot,
                color="Cluster", text="Espece_User", hover_name="Espece_User",
                custom_data=["Espece_User", "Ecologie"], 
                template="plotly_dark", height=600, color_discrete_sequence=COLOR_SEQUENCE
            )
            
            fig_interactive_scatter.update_traces(
                textposition="top center", 
                marker=dict(opacity=0.8, size=8),
                textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS),
                hovertemplate=(
                    f"<span style='font-size: {HOVER_SPECIES_FONT_SIZE}px;'><b>%{{customdata[0]}}</b></span><br>"
                    f"<br><span style='font-size: {HOVER_ECOLOGY_TITLE_FONT_SIZE}px;'><i>Écologie:</i></span><br>"
                    f"<span style='font-size: {HOVER_ECOLOGY_TEXT_FONT_SIZE}px;'>%{{customdata[1]}}</span>"
                    "<extra></extra>" 
                )
            )

            unique_clusters_interactive = sorted(plot_data_to_use["Cluster"].unique())
            cluster_color_map_interactive = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_interactive)}
            for cluster_label in unique_clusters_interactive:
                cluster_points_df_interactive = plot_data_to_use[plot_data_to_use["Cluster"] == cluster_label]
                if x_axis_trait_selected_for_plot in cluster_points_df_interactive and y_axis_trait_selected_for_plot in cluster_points_df_interactive:
                    points_for_hull = cluster_points_df_interactive[[x_axis_trait_selected_for_plot, y_axis_trait_selected_for_plot]].drop_duplicates().values
                    if len(points_for_hull) >= MIN_POINTS_FOR_HULL:
                        try:
                            hull_interactive = ConvexHull(points_for_hull) 
                            hull_path_interactive = points_for_hull[np.append(hull_interactive.vertices, hull_interactive.vertices[0])]
                            clr_int = cluster_color_map_interactive.get(cluster_label, COLOR_SEQUENCE[0])
                            fig_interactive_scatter.add_trace(go.Scatter(
                                x=hull_path_interactive[:, 0], y=hull_path_interactive[:, 1], fill="toself", fillcolor=clr_int,
                                line=dict(color=clr_int, width=1.5), mode='lines', name=f'Cluster {cluster_label} Hull', 
                                opacity=0.2, showlegend=False, hoverinfo='skip' ))
                        except Exception as e: print(f"Erreur calcul Hull interactif {cluster_label} ({x_axis_trait_selected_for_plot}, {y_axis_trait_selected_for_plot}): {e}")
            
            fig_interactive_scatter.update_layout(
                title_text=f"{y_axis_trait_selected_for_plot} vs. {x_axis_trait_selected_for_plot}", title_x=0.5,
                xaxis_title=x_axis_trait_selected_for_plot, yaxis_title=y_axis_trait_selected_for_plot
            )
            fig_interactive_scatter.update_layout(dragmode='pan')
            st.plotly_chart(fig_interactive_scatter, use_container_width=True, config={'scrollZoom': True})

# ---------------------------------------------------------------------------- #
# SECTION 3: COMPOSITION DES CLUSTERS (SOUS LE GRAPHIQUE INTERACTIF)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty: 
    st.markdown("---")
    st.subheader("Composition des Clusters (ACP)")
    if cluster_compositions_data and any(d['count'] > 0 for d in cluster_compositions_data):
        num_clusters_found_display = len([d for d in cluster_compositions_data if d['count']>0]) 
        num_display_cols = min(num_clusters_found_display, 3) 
        
        if num_display_cols > 0: 
            cluster_cols = st.columns(num_display_cols)
            current_col_idx = 0
            for comp_data in cluster_compositions_data:
                if comp_data['count'] > 0: 
                    with cluster_cols[current_col_idx % num_display_cols]:
                        st.markdown(f"**Cluster {comp_data['cluster_label']}** ({comp_data['count']} espèces)")
                        for species_name in comp_data['species_list']: st.markdown(f"- {species_name}")
                    current_col_idx += 1
        else:
            st.info("Aucun cluster (ACP) avec des espèces à afficher.")
    else: 
        st.info("La composition des clusters (ACP) sera affichée ici après l'analyse.")

# ---------------------------------------------------------------------------- #
# SECTION 4: AFFICHAGE DU DENDROGRAMME (PLEINE LARGEUR, EN DERNIER)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty : 
    st.markdown("---") 
    if fig_dend: 
        st.plotly_chart(fig_dend, use_container_width=True)
    elif species_binom_user_unique: 
        st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces uniques après traitement ou problème de données pour le linkage).")
