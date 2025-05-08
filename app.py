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
# FONCTION UTILITAIRE POUR AJUSTER LES POSITIONS DU TEXTE
# ---------------------------------------------------------------------------- #
def adjust_text_positions(df, x_col, y_col, label_col, jitter_strength=0.05):
    """
    Ajuste la position y des étiquettes de texte pour les points superposés.
    Ajoute une colonne 'textposition_y_offset' au DataFrame.
    """
    df_copy = df.copy()
    df_copy['textposition_y_offset'] = df_copy[y_col] # Position initiale sans décalage
    duplicates = df_copy.duplicated(subset=[x_col, y_col], keep=False)

    if duplicates.any():
        for name, group in df_copy[duplicates].groupby([x_col, y_col]):
            if len(group) > 1:
                # Calculer la plage des valeurs y pour déterminer l'échelle du jitter
                y_range = df_copy[y_col].max() - df_copy[y_col].min()
                if y_range == 0: y_range = 1 # Éviter la division par zéro si toutes les valeurs y sont identiques

                # Appliquer un décalage plus petit si les valeurs sont très proches
                # Décalage proportionnel à l'échelle des données y, mais plafonné pour éviter des décalages excessifs
                effective_jitter = min(jitter_strength * y_range, jitter_strength)

                offsets = np.linspace(-effective_jitter * (len(group) - 1) / 2,
                                      effective_jitter * (len(group) - 1) / 2,
                                      len(group))
                for i, idx in enumerate(group.index):
                    df_copy.loc[idx, 'textposition_y_offset'] = df_copy.loc[idx, y_col] + offsets[i]
    return df_copy

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

# Initialisation des variables pour les figures et données
fig_pca = None
fig_dend = None
vip_styled = None
vip_data_df = pd.DataFrame()
cluster_compositions_data = []
sub = pd.DataFrame()
pdf = pd.DataFrame()

# ---------------------------------------------------------------------------- #
# ANALYSE PRINCIPALE
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
        with col_input:
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

        st.session_state.pdf = current_pdf

        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings**2).sum(axis=1)
        st.session_state.vip_data_df = pd.DataFrame({
            "Variable": sub.columns[1:], "Communalité (%)": (communal * 100).round(0).astype(int),
        }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)

        st.session_state.X_for_dendro = X

        numeric_trait_names_init = [col for col in sub.columns if col.lower() != "espece" and pd.api.types.is_numeric_dtype(sub[col])]
        if len(numeric_trait_names_init) >= 2:
            default_x_init = numeric_trait_names_init[0]
            default_y_init = numeric_trait_names_init[1]
            if not st.session_state.vip_data_df.empty and len(st.session_state.vip_data_df) >= 2:
                top_vars_init = [var for var in st.session_state.vip_data_df["Variable"].tolist() if var in numeric_trait_names_init]
                if len(top_vars_init) >= 1: default_x_init = top_vars_init[0]
                if len(top_vars_init) >= 2: default_y_init = top_vars_init[1]
                if default_x_init == default_y_init:
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
        st.session_state.run_main_analysis_once = False
        st.stop()

# Récupérer les DataFrames de st.session_state si l'analyse principale a déjà tourné
if st.session_state.run_main_analysis_once:
    sub = st.session_state.get('sub', pd.DataFrame())
    pdf = st.session_state.get('pdf', pd.DataFrame())
    vip_data_df = st.session_state.get('vip_data_df', pd.DataFrame())
    X_for_dendro = st.session_state.get('X_for_dendro', np.array([]))

    if not pdf.empty:
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
        cluster_color_map_pca = {cluster_label: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, cluster_label in enumerate(unique_clusters_pca)}

        if "PC2" in pdf.columns and pdf.shape[1] > 1 :
            for i, cluster_label in enumerate(unique_clusters_pca):
                cluster_points_df_pca = pdf[pdf["Cluster"] == cluster_label]
                if "PC1" in cluster_points_df_pca.columns and "PC2" in cluster_points_df_pca.columns:
                    unique_cluster_points_pca = cluster_points_df_pca[["PC1", "PC2"]].drop_duplicates().values
                    if len(unique_cluster_points_pca) >= MIN_POINTS_FOR_HULL:
                        try:
                            hull_pca = ConvexHull(unique_cluster_points_pca)
                            hull_points_pca = unique_cluster_points_pca[hull_pca.vertices]
                            path_x_pca = np.append(hull_points_pca[:, 0], hull_points_pca[0, 0])
                            path_y_pca = np.append(hull_points_pca[:, 1], hull_points_pca[0, 1])
                            fig_pca.add_trace(go.Scatter(
                                x=path_x_pca, y=path_y_pca, fill="toself",
                                fillcolor=cluster_color_map_pca[cluster_label], # MODIFIÉ ICI
                                line=dict(color=cluster_color_map_pca[cluster_label], width=1.5), # MODIFIÉ ICI
                                mode='lines',
                                name=f'Cluster {cluster_label} Hull', opacity=0.2, showlegend=False, hoverinfo='skip'
                            ))
                        except Exception as e_hull_pca: print(f"Note: Impossible de générer l'enveloppe convexe ACP pour le cluster {cluster_label}: {e_hull_pca}")
                    elif len(unique_cluster_points_pca) > 0: print(f"Note: Cluster ACP {cluster_label}: pas assez de points uniques ({len(unique_cluster_points_pca)}) pour l'enveloppe (min {MIN_POINTS_FOR_HULL}).")
        fig_pca.update_layout(title_text="Clusters d'espèces (ACP)", title_x=0.5, legend_title_text='Cluster')

        if not vip_data_df.empty:
            vip_styled = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)']).format({"Communalité (%)": "{:}%"})

        cluster_compositions_data = []
        for c_label in sorted(pdf["Cluster"].unique()):
            esp_user_names = sorted(list(pdf.loc[pdf["Cluster"] == c_label, "Espece_User"].unique()))
            cluster_compositions_data.append({"cluster_label": c_label, "count": len(esp_user_names), "species_list": esp_user_names})

        if X_for_dendro.shape[0] > 1:
            Z = linkage(X_for_dendro, method="ward")
            dynamic_color_threshold = 0
            if n_clusters_selected > 1 and (n_clusters_selected -1) <= Z.shape[0] :
                idx_threshold = -(n_clusters_selected - 1)
                if idx_threshold == 0: dynamic_color_threshold = Z[0, 2] / 2
                elif Z.shape[0] >= (n_clusters_selected -1) and (n_clusters_selected -1) > 0: dynamic_color_threshold = Z[-(n_clusters_selected-1), 2] * 0.99
            fig_dend = ff.create_dendrogram(
                X_for_dendro, orientation="left", labels=pdf["Espece_User"].tolist(), linkagefun=lambda _: Z,
                color_threshold=dynamic_color_threshold if n_clusters_selected > 1 else 0, colorscale=COLOR_SEQUENCE
            )
            fig_dend.update_layout(template="plotly_dark", height=max(650, sub.shape[0] * 20), title_text="Dendrogramme", title_x=0.5)
        else: fig_dend = None


# ---------------------------------------------------------------------------- #
# AFFICHAGE DES RESULTATS ACP ET ASSOCIES
# ---------------------------------------------------------------------------- #
with col_pca_plot:
    if fig_pca:
        st.plotly_chart(fig_pca, use_container_width=True)
    elif run_main_analysis_button and ref.empty:
        st.warning("Veuillez d'abord charger des données de traits pour afficher le graphique ACP.")
    elif run_main_analysis_button and sub.empty :
        st.warning("Aucune espèce valide pour l'analyse ACP.")
    elif st.session_state.run_main_analysis_once and not fig_pca:
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
# EXPLORATION INTERACTIVE DES VARIABLES
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty:
    st.markdown("---")
    st.subheader("🔬 Exploration interactive des variables")

    potential_traits = [col for col in sub.columns if col.lower() != "espece"]
    numeric_trait_names = [
        col for col in potential_traits if pd.api.types.is_numeric_dtype(sub[col])
    ]

    if len(numeric_trait_names) >= 2:
        st.markdown("##### Sélectionnez les variables pour les axes du nuage de points :")

        current_x_trait = st.session_state.get('x_axis_trait_interactive', numeric_trait_names[0])
        current_y_trait = st.session_state.get('y_axis_trait_interactive', numeric_trait_names[1])

        col_scatter_select_x, col_scatter_select_y, col_button_update = st.columns([2,2,1])
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

        st.session_state.x_axis_trait_interactive = x_axis_trait_selected
        st.session_state.y_axis_trait_interactive = y_axis_trait_selected

        with col_button_update:
            st.markdown("<br>", unsafe_allow_html=True)
            # MODIFIÉ ICI: ajout de type="primary"
            run_interactive_update = st.button("Actualiser l'exploration", type="primary", key="update_interactive_plot_button")

        if x_axis_trait_selected and y_axis_trait_selected:
            if not pdf.empty:
                if len(sub) == len(pdf):
                    plot_data_interactive_raw = pd.DataFrame({
                        'Espece_User': pdf['Espece_User'].values,
                        'Ecologie': pdf['Ecologie'].values,
                        x_axis_trait_selected: sub[x_axis_trait_selected].values,
                        y_axis_trait_selected: sub[y_axis_trait_selected].values,
                        'Cluster': pdf['Cluster'].values
                    })

                    # MODIFIÉ ICI: Appliquer l'ajustement des positions de texte
                    plot_data_interactive = adjust_text_positions(
                        plot_data_interactive_raw,
                        x_axis_trait_selected,
                        y_axis_trait_selected,
                        'Espece_User'
                    )
                    # S'assurer que la nouvelle colonne est présente pour custom_data
                    if 'textposition_y_offset' not in plot_data_interactive.columns:
                         plot_data_interactive['textposition_y_offset'] = plot_data_interactive[y_axis_trait_selected]


                    fig_interactive_scatter = px.scatter(
                        plot_data_interactive,
                        x=x_axis_trait_selected,
                        y=y_axis_trait_selected,
                        color="Cluster",
                        text="Espece_User",
                        hover_name="Espece_User",
                        custom_data=["Espece_User", "Ecologie", x_axis_trait_selected, y_axis_trait_selected, 'textposition_y_offset'], # AJOUTÉ 'textposition_y_offset'
                        template="plotly_dark",
                        height=600,
                        color_discrete_sequence=COLOR_SEQUENCE
                    )

                    # MODIFIÉ ICI: Mettre à jour textposition pour utiliser le décalage
                    fig_interactive_scatter.update_traces(
                        # textposition="top center", # On va le gérer dynamiquement
                        marker=dict(opacity=0.8, size=8),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>" +
                            f"<br><i>{x_axis_trait_selected}:</i> %{{customdata[2]}}<br>" +
                            f"<i>{y_axis_trait_selected}:</i> %{{customdata[3]}}<br>" +
                            "<br><i>Écologie:</i><br>%{customdata[1]}" +
                            "<extra></extra>"
                        )
                    )
                    # Application du décalage textuel via annotations ou mise à jour de layout.annotations si px.scatter ne le gère pas directement
                    # Pour un contrôle fin, il faudrait itérer sur les traces ou utiliser go.Scatter directement.
                    # Plotly Express gère 'text' pour chaque point. On peut essayer d'influencer 'textposition' via un array.
                    # Mais pour un décalage fin et individuel, la méthode la plus robuste est souvent d'ajouter des annotations.
                    # Ici, on va essayer une approche plus simple en modifiant le y du texte via textfont.
                    # Une meilleure approche serait de recalculer les positions textuelles et de les fournir explicitement.

                    # Mettre à jour les positions du texte pour chaque point
                    # Plotly express ne permet pas de spécifier directement un 'text_y' différent du 'y' du point pour chaque trace.
                    # La solution de décalage par 'textposition_y_offset' est un peu plus complexe à implémenter proprement avec px.scatter
                    # uniquement pour le texte. Le plus simple est de s'assurer que le 'text' est bien présent.
                    # Le décalage des labels est un défi avec plotly express car il n'y a pas de propriété 'text_y' distincte
                    # qui pourrait être un tableau de valeurs.
                    # La fonction adjust_text_positions prépare les données, mais l'application dans px.scatter
                    # pour que les labels se déplacent verticalement tout en gardant le point à sa place d'origine
                    # nécessite soit de passer à go.Scatter, soit d'ajouter des annotations.

                    # Tentative de mise à jour des positions de texte (peut nécessiter une approche plus go.Figure pour un contrôle total)
                    # Pour l'instant, le texte sera centré au-dessus du point. La fonction adjust_text_positions
                    # a modifié 'textposition_y_offset', mais son utilisation directe pour décaler le texte dans px.scatter
                    # n'est pas triviale sans affecter la position du marqueur lui-même ou sans ajouter des annotations.

                    # Ce qu'on peut faire, c'est de s'assurer que le mode texte est bien activé.
                    fig_interactive_scatter.update_traces(mode='markers+text', textposition='top center')

                    # Le décalage des étiquettes est complexe à réaliser parfaitement avec px.scatter seul pour des points superposés
                    # sans que les étiquettes elles-mêmes ne soient superposées si on ne change pas le y des points pour le texte.
                    # La solution `adjust_text_positions` prépare les données, mais son intégration directe
                    # pour le décalage *visuel* du texte dans Plotly Express sans changer la position des *points* est limitée.
                    # Pour un vrai décalage des labels uniquement, il faudrait ajouter des annotations séparées
                    # ou passer à `go.Figure` et construire les traces `go.Scatter` manuellement, ce qui est plus verbeux.

                    # Pour l'instant, la fonction adjust_text_positions ne sera pas pleinement exploitée pour le décalage visuel des textes
                    # dans cette version simplifiée avec px.scatter, car cela demanderait une refonte plus profonde.
                    # L'objectif principal ici est de s'assurer que les couleurs des enveloppes sont correctes et que le bouton est rouge.
                    # La logique de décalage des textes est conceptuellement là mais son application graphique fine est plus ardue.


                    unique_clusters_interactive = sorted(plot_data_interactive["Cluster"].unique())
                    cluster_color_map_interactive = {
                        cluster_label: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]
                        for i, cluster_label in enumerate(unique_clusters_interactive)
                    }

                    for cluster_label in unique_clusters_interactive:
                        cluster_points_df_interactive = plot_data_interactive[plot_data_interactive["Cluster"] == cluster_label]
                        points_for_hull = cluster_points_df_interactive[[x_axis_trait_selected, y_axis_trait_selected]].drop_duplicates().values

                        if len(points_for_hull) >= MIN_POINTS_FOR_HULL:
                            try:
                                hull_interactive = ConvexHull(points_for_hull)
                                hull_points_interactive = points_for_hull[hull_interactive.vertices]
                                path_x_interactive = np.append(hull_points_interactive[:, 0], hull_points_interactive[0, 0])
                                path_y_interactive = np.append(hull_points_interactive[:, 1], hull_points_interactive[0, 1])
                                fig_interactive_scatter.add_trace(go.Scatter(
                                    x=path_x_interactive, y=path_y_interactive, fill="toself",
                                    fillcolor=cluster_color_map_interactive[cluster_label], # MODIFIÉ ICI
                                    line=dict(color=cluster_color_map_interactive[cluster_label], width=1.5), # MODIFIÉ ICI
                                    mode='lines', name=f'Cluster {cluster_label} Hull',
                                    opacity=0.2, showlegend=False, hoverinfo='skip'
                                ))
                            except Exception as e_hull_interactive:
                                print(f"Note: Impossible de générer l'enveloppe convexe interactive pour le cluster {cluster_label}: {e_hull_interactive}")
                        elif len(points_for_hull) > 0:
                            print(f"Note: Cluster interactif {cluster_label}: pas assez de points uniques ({len(points_for_hull)}) pour l'enveloppe (min {MIN_POINTS_FOR_HULL}).")

                    fig_interactive_scatter.update_layout(
                        title_text=f"Variables: {y_axis_trait_selected} en fonction de {x_axis_trait_selected}",
                        title_x=0.5,
                        xaxis_title=x_axis_trait_selected,
                        yaxis_title=y_axis_trait_selected,
                        # Tentative de forcer l'affichage du texte même si px.scatter peut le cacher par défaut pour éviter la superposition
                        # modebar_add=['v1hovermode', 'toggleSpikelines'] # Ceci n'est pas pour le texte
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
