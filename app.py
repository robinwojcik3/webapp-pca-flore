import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
from scipy.spatial import ConvexHull
import numpy as np
import textwrap

# -----------------------------------------------------------------------------
# MODULES DE SUPPORT (core.py simulé si absent)
# -----------------------------------------------------------------------------
try:
    import core
except ImportError:
    st.warning("Le module 'core.py' est introuvable. Une fonction d'analyse simulée sera utilisée. L'ACP et le dendrogramme réels ne fonctionneront pas.")

    class MockPCA:
        def __init__(self):
            self.components_ = np.array([[0.5, 0.5], [-0.5, 0.5]])
            self.explained_variance_ = np.array([0.6, 0.4])

    def mock_analyse(sub_df, n_clusters):
        n_samples = len(sub_df)
        if n_samples == 0:
            return np.array([]), MockPCA(), pd.DataFrame(columns=['PC1', 'PC2']), np.array([])

        coords_array = np.random.rand(n_samples, 2) * 10
        coords_df = pd.DataFrame(coords_array, columns=[f"PC{i+1}" for i in range(coords_array.shape[1])])

        labels = np.arange(n_samples) if n_samples <= n_clusters else np.random.randint(0, n_clusters, n_samples)

        numeric_cols = sub_df.select_dtypes(include=np.number)
        if not numeric_cols.empty:
            X_scaled = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
            X_scaled = X_scaled.fillna(0).values
        else:
            X_scaled = np.random.rand(n_samples, 2)

        mock_pca_obj = MockPCA()
        num_numeric_traits = X_scaled.shape[1]
        if num_numeric_traits > 0:
            mock_pca_obj.components_ = np.random.rand(num_numeric_traits, min(2, num_numeric_traits))
            mock_pca_obj.explained_variance_ = np.random.rand(min(2, num_numeric_traits))
        else:
            mock_pca_obj.components_ = np.array([])
            mock_pca_obj.explained_variance_ = np.array([])

        return labels, mock_pca_obj, coords_df, X_scaled

    core = type('CoreModule', (object,), {
        'analyse': mock_analyse,
        'read_reference': lambda fp: pd.DataFrame()
    })

# -----------------------------------------------------------------------------
# CONFIGURATION UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PCA flore interactive", layout="wide")
st.markdown("<h1 style='text-align: center;'>Analyse interactive des clusters botaniques</h1>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CONSTANTES
# -----------------------------------------------------------------------------
MIN_POINTS_FOR_HULL = 3
COLOR_SEQUENCE = px.colors.qualitative.Plotly
LABEL_FONT_SIZE_ON_PLOTS = 15
HOVER_SPECIES_FONT_SIZE = 15
HOVER_ECOLOGY_TITLE_FONT_SIZE = 14
HOVER_ECOLOGY_TEXT_FONT_SIZE = 13

# -----------------------------------------------------------------------------
# CHARGEMENTS INITIAUX
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_path="data_ref.csv"):
    try:
        if callable(getattr(core, "read_reference", None)) and core.read_reference.__name__ == "<lambda>":
            st.warning("Simulation du chargement de 'data_ref.csv'.")
            example_species = [f"Espece {i}" for i in range(30)]
            return pd.DataFrame({
                'Espece': example_species,
                'Trait1': np.random.rand(30) * 10,
                'Trait2': np.random.randint(1, 100, 30),
                'Trait3': np.random.randint(1, 10, 30)
            })
        data = core.read_reference(file_path)
        if data.empty:
            st.warning("Le fichier de référence de traits est vide.")
        return data
    except FileNotFoundError:
        st.error("Fichier de référence introuvable.")
        return pd.DataFrame()

ref = load_data()

ref_binom_series = pd.Series(dtype=str)
if not ref.empty and 'Espece' in ref.columns:
    ref_binom_series = (
        ref['Espece'].astype(str).str.split().str[:2].str.join(' ').str.lower()
    )

@st.cache_data
def load_ecology_data(file_path="data_ecologie_espece.csv"):
    try:
        eco_df = pd.read_csv(file_path, sep=';', header=None, usecols=[0, 1], names=['Espece', 'Description_Ecologie'], encoding='utf-8-sig')
        eco_df['Espece_norm'] = eco_df['Espece'].astype(str).str.split().str[:2].str.join(' ').str.lower()
        eco_df = eco_df.drop_duplicates('Espece_norm')
        eco_df = eco_df.set_index('Espece_norm')
        return eco_df[['Description_Ecologie']]
    except FileNotFoundError:
        st.toast("Fichier écologique absent.", icon="⚠️")
        return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))

ecology_df = load_ecology_data()

# formatter pour le hover

def format_ecology_for_hover(text, line_width_chars=65):
    if pd.isna(text) or str(text).strip() == "":
        return "Description écologique non disponible."
    wrapped = textwrap.wrap(str(text), width=line_width_chars)
    return "<br>".join(wrapped)

# -----------------------------------------------------------------------------
# ETATS DE SESSION
# -----------------------------------------------------------------------------
def_ss = {
    'selected_habitat_index': None,     # colonne actuellement sélectionnée
    'prev_selected_habitat_index': None,
    'x_axis_trait_interactive': None,
    'y_axis_trait_interactive': None,
    'vip_data_df_interactive': pd.DataFrame(),
    'vip_data_df_interactive_snapshot_for_comparison': pd.DataFrame(),
    'sub': pd.DataFrame(),
    'pdf': pd.DataFrame(),
    'X_for_dendro': np.array([]),
    'numeric_trait_names_for_interactive_plot': [],
    'analysis_done_once': False,
    'prev_n_clusters': None,
    'releves_df': None,
    'previous_num_cols': 0,
}

for k, v in def_ss.items():
    if k not in st.session_state:
        st.session_state[k] = v

# initialisation tableau releves au premier chargement
if st.session_state.releves_df is None or not isinstance(st.session_state.releves_df, pd.DataFrame):
    n_cols_placeholder = 15
    n_rows_placeholder = 11  # 1 ligne titres + 10 lignes espèces
    header = ["" for _ in range(n_cols_placeholder)]
    body = [["" for _ in range(n_cols_placeholder)] for _ in range(n_rows_placeholder - 1)]
    st.session_state.releves_df = pd.DataFrame([header] + body)
    st.session_state.releves_df.columns = [str(c) for c in st.session_state.releves_df.columns]
    st.session_state.previous_num_cols = n_cols_placeholder

# -----------------------------------------------------------------------------
# ETAPE 1 : TABLEAU + SELECTION HABITAT (cliquer sur le nom d'habitat ligne 0)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Étape 1 : Importation et sélection du relevé floristique")

st.info("Copiez/collez vos relevés : la première ligne contient les noms d'habitat, les suivantes les espèces.")

# Data Editor principal
edited_df = st.data_editor(
    st.session_state.releves_df,
    num_rows="dynamic",
    use_container_width=True,
    key="releves_data_editor",
)

# Mise à jour du dataframe en session
if not edited_df.equals(st.session_state.releves_df):
    st.session_state.releves_df = edited_df.copy()
    # ajuster index sélectionné si nb colonnes change
    if st.session_state.selected_habitat_index is not None and st.session_state.selected_habitat_index >= len(edited_df.columns):
        st.session_state.selected_habitat_index = None
    st.session_state.previous_num_cols = len(edited_df.columns)

# -----------------------------------------------------------------------------
# RANGÉE DE BOUTONS ALIGNÉE SUR LA PREMIÈRE LIGNE DU TABLEAU
# -----------------------------------------------------------------------------
if len(st.session_state.releves_df.columns) > 0:

    # conteneur pour boutons
    btn_container = st.container()
    with btn_container:
        st.markdown("""
        <style>
        div[data-testid="habitat-btn-row"] {margin-top:-42px;} /* remonter le conteneur pour l'aligner sur la 1ère ligne */
        </style>
        """, unsafe_allow_html=True)

        # layout colonnes identique au tableau
        cols_layout = st.columns(len(st.session_state.releves_df.columns))
        first_row_vals = st.session_state.releves_df.iloc[0].astype(str).tolist()

        for i, col in enumerate(cols_layout):
            habitat_label = first_row_vals[i] if first_row_vals[i].strip() else f"Relevé {i+1}"
            is_sel = (i == st.session_state.selected_habitat_index)
            if col.button(habitat_label, key=f"hab_btn_{i}", type=("primary" if is_sel else "secondary")):
                if st.session_state.selected_habitat_index != i:
                    st.session_state.selected_habitat_index = i
                else:
                    # on peut décider de désélectionner en recliquant; ici on garde sélection unique
                    pass
                st.experimental_rerun()
else:
    st.warning("Aucune colonne disponible dans le tableau.")

# colonne sélectionnée (liste pour compatibilité avec analyse)
selected_col_indices_for_analysis = [st.session_state.selected_habitat_index] if st.session_state.selected_habitat_index is not None else []

# -----------------------------------------------------------------------------
# PARAMÈTRES D'ANALYSE (slider clusters) – pas de bouton d'exécution
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Étape 2 : Paramètres d'analyse et exécution automatique")

n_clusters_selected = st.slider("Nombre de clusters", 2, 8, 3, key="n_clusters_slider", disabled=ref.empty)

# -----------------------------------------------------------------------------
# TRIGGER AUTOMATIQUE DE L'ANALYSE
# -----------------------------------------------------------------------------
should_run_analysis = False
if (
    st.session_state.selected_habitat_index is not None and
    not ref.empty and
    (st.session_state.prev_selected_habitat_index != st.session_state.selected_habitat_index or
     st.session_state.prev_n_clusters != n_clusters_selected or
     not st.session_state.analysis_done_once)
):
    should_run_analysis = True

if should_run_analysis:
    # mémoriser les nouveaux états pour prochaine comparaison
    st.session_state.prev_selected_habitat_index = st.session_state.selected_habitat_index
    st.session_state.prev_n_clusters = n_clusters_selected

    # ---------------------------------------------------------------------------------
    # ANALYSE PRINCIPALE (ancien contenu du bouton « Lancer l'analyse principale »)
    # ---------------------------------------------------------------------------------
    st.session_state.analysis_done_once = True

    # remise à zéro des sorties précédentes
    st.session_state.sub = pd.DataFrame()
    st.session_state.pdf = pd.DataFrame()
    st.session_state.X_for_dendro = np.array([])
    st.session_state.vip_data_df_interactive = pd.DataFrame()

    # extraction des espèces dans la colonne sélectionnée
    species_raw = []
    df_src = st.session_state.releves_df.copy()
    col_idx = st.session_state.selected_habitat_index
    if not df_src.empty and len(df_src) > 1 and col_idx is not None:
        species_col = (
            df_src.iloc[1:, col_idx]
            .dropna()
            .astype(str)
            .str.strip()
            .replace('', np.nan)
            .dropna()
            .tolist()
        )
        species_raw.extend(species_col)

    species_raw_unique = sorted(list(set([s for s in species_raw if s])))
    species_binom_user_unique = [" ".join(s.split()[:2]).lower() for s in species_raw_unique]

    if not species_binom_user_unique:
        st.error("Aucune espèce valide extraite.")
        st.stop()

    # correspondance avec base de référence
    indices_to_keep = []
    if not ref_binom_series.empty:
        df_match = ref_binom_series.reset_index()
        df_match.columns = ['Orig_Index', 'binom']
        for binom in species_binom_user_unique:
            matches = df_match[df_match['binom'] == binom]
            if not matches.empty:
                indices_to_keep.append(matches['Orig_Index'].iloc[0])

    indices_to_keep = sorted(list(set(indices_to_keep)))
    st.session_state.sub = ref.loc[indices_to_keep].copy() if indices_to_keep else pd.DataFrame(columns=ref.columns)

    # message si espèces manquantes
    found_binoms = st.session_state.sub['Espece'].astype(str).str.split().str[:2].str.join(' ').str.lower().tolist() if not st.session_state.sub.empty else []
    not_found = [species_raw_unique[i] for i, b in enumerate(species_binom_user_unique) if b not in found_binoms]
    if not_found:
        st.warning("Non trouvées dans la base : " + ", ".join(not_found))

    # validations
    if st.session_state.sub.empty:
        st.error("Aucune espèce correspondante dans la base de traits.")
        st.stop()
    if st.session_state.sub.shape[0] < n_clusters_selected:
        st.error(f"Espèces trouvées ({st.session_state.sub.shape[0]}) < clusters demandés ({n_clusters_selected}).")
        st.stop()
    if st.session_state.sub.shape[0] < 2:
        st.error("Deux espèces minimum nécessaires.")
        st.stop()

    # appel analyse
    try:
        labels, pca_res, coords_df, X_scaled = core.analyse(st.session_state.sub, n_clusters_selected)
    except Exception as e:
        st.error(f"Erreur core.analyse : {e}")
        st.stop()

    # normalisation des coordonnées PCA
    if isinstance(coords_df, np.ndarray):
        coords_df = pd.DataFrame(coords_df, columns=[f"PC{i+1}" for i in range(coords_df.shape[1])])
    if not isinstance(coords_df, pd.DataFrame):
        st.error("Format des coordonnées PCA incorrect.")
        st.stop()

    pdf = coords_df.copy()
    pdf['Cluster'] = labels.astype(str) if len(labels) == len(pdf) else '0'
    pdf['Espece_Ref'] = st.session_state.sub['Espece'].values[:len(pdf)]
    mapping_raw = {" ".join(s.split()[:2]).lower(): s for s in species_raw_unique}
    pdf['Espece_User'] = pdf['Espece_Ref'].apply(lambda n: mapping_raw.get(" ".join(str(n).split()[:2]).lower(), n))

    # écologie
    if not ecology_df.empty:
        pdf['Espece_norm'] = pdf['Espece_Ref'].astype(str).str.split().str[:2].str.join(' ').str.lower()
        pdf['Ecologie'] = pdf['Espece_norm'].map(ecology_df['Description_Ecologie']).apply(format_ecology_for_hover)
    else:
        pdf['Ecologie'] = format_ecology_for_hover(None)

    st.session_state.pdf = pdf.copy()

    # communalités & VIP
    vip_df = pd.DataFrame()
    if hasattr(pca_res, 'components_') and hasattr(pca_res, 'explained_variance_'):
        comps = np.asarray(pca_res.components_)
        if comps.ndim == 1:
            comps = comps.reshape(1, -1)
        loads = comps.T * np.sqrt(pca_res.explained_variance_[:comps.shape[0]])
        communal = (loads ** 2).sum(axis=1)
        num_traits = st.session_state.sub.select_dtypes(include=np.number).columns.tolist()
        if len(communal) == len(num_traits):
            vip_df = pd.DataFrame({
                'Variable': num_traits,
                'Communalité (%)': (communal * 100).round(0).astype(int)
            }).sort_values('Communalité (%)', ascending=False).reset_index(drop=True)
    st.session_state.vip_data_df_for_calc = vip_df

    # dendro
    st.session_state.X_for_dendro = X_scaled if isinstance(X_scaled, np.ndarray) else np.array([])

    # variables numériques pour étape 3
    st.session_state.numeric_trait_names_for_interactive_plot = [c for c in st.session_state.sub.columns if c.lower() != 'espece' and pd.api.types.is_numeric_dtype(st.session_state.sub[c])]

    # choix automatiques X / Y
    auto_x = auto_y = None
    if not vip_df.empty:
        inter_vars = [v for v in vip_df['Variable'] if v in st.session_state.numeric_trait_names_for_interactive_plot]
        if inter_vars:
            auto_x = inter_vars[0]
            auto_y = inter_vars[1] if len(inter_vars) > 1 else (st.session_state.numeric_trait_names_for_interactive_plot[1] if len(st.session_state.numeric_trait_names_for_interactive_plot) > 1 else auto_x)
    if auto_x is None and st.session_state.numeric_trait_names_for_interactive_plot:
        auto_x = st.session_state.numeric_trait_names_for_interactive_plot[0]
    if auto_y is None and len(st.session_state.numeric_trait_names_for_interactive_plot) > 1:
        auto_y = st.session_state.numeric_trait_names_for_interactive_plot[1]

    st.session_state.x_axis_trait_interactive = auto_x
    st.session_state.y_axis_trait_interactive = auto_y

    # tableau interactif init
    if not vip_df.empty:
        inter_df = vip_df[vip_df['Variable'].isin(st.session_state.numeric_trait_names_for_interactive_plot)].copy()
        inter_df['Axe X'] = inter_df['Variable'] == auto_x
        inter_df['Axe Y'] = inter_df['Variable'] == auto_y
        st.session_state.vip_data_df_interactive = inter_df[['Variable', 'Communalité (%)', 'Axe X', 'Axe Y']]
        st.session_state.vip_data_df_interactive_snapshot_for_comparison = st.session_state.vip_data_df_interactive.copy()

    st.experimental_rerun()

# -----------------------------------------------------------------------------
# ÉTAPE 3 : EXPLORATION INTERACTIVE DES VARIABLES
# -----------------------------------------------------------------------------
if st.session_state.analysis_done_once and not st.session_state.sub.empty:
    st.markdown("---")
    st.subheader("Étape 3 : Exploration interactive des variables")

    col_table, col_plot = st.columns([1, 2])

    with col_table:
        st.markdown("##### Tableau d'exploration")
        df_source = st.session_state.vip_data_df_interactive.copy()
        if df_source.empty:
            st.info("Aucun trait numérique disponible.")
        else:
            edited_vip = st.data_editor(
                df_source,
                column_config={
                    "Variable": st.column_config.TextColumn("Variable", disabled=True),
                    "Communalité (%)": st.column_config.NumberColumn("Communalité (%)", format="%d%%", disabled=True),
                    "Axe X": st.column_config.CheckboxColumn("Axe X"),
                    "Axe Y": st.column_config.CheckboxColumn("Axe Y"),
                },
                key="vip_editor",
                use_container_width=True,
                hide_index=True,
                num_rows="fixed"
            )

            # exclusivité X
            x_selected = edited_vip[edited_vip['Axe X']]['Variable'].tolist()
            y_selected = edited_vip[edited_vip['Axe Y']]['Variable'].tolist()
            new_x = x_selected[0] if x_selected else None
            new_y = y_selected[0] if y_selected else None

            if new_x != st.session_state.x_axis_trait_interactive or new_y != st.session_state.y_axis_trait_interactive:
                st.session_state.x_axis_trait_interactive = new_x
                st.session_state.y_axis_trait_interactive = new_y
                # mise à jour exclusivité visuelle
                edited_vip['Axe X'] = edited_vip['Variable'] == new_x
                edited_vip['Axe Y'] = edited_vip['Variable'] == new_y
                st.session_state.vip_data_df_interactive = edited_vip.copy()
                st.session_state.vip_data_df_interactive_snapshot_for_comparison = edited_vip.copy()
                st.experimental_rerun()

    with col_plot:
        st.markdown("##### Nuage de points")
        x_trait = st.session_state.x_axis_trait_interactive
        y_trait = st.session_state.y_axis_trait_interactive
        pdf_plot = st.session_state.pdf.copy()
        sub_plot = st.session_state.sub.copy()

        if not x_trait or not y_trait:
            st.info("Sélectionnez une variable X et Y dans le tableau à gauche.")
        elif x_trait not in sub_plot.columns or y_trait not in sub_plot.columns:
            st.warning("Variables non valides.")
        else:
            data_plot = pd.DataFrame({
                x_trait: sub_plot[x_trait],
                y_trait: sub_plot[y_trait],
                'Cluster': pdf_plot['Cluster'],
                'Espece_User': pdf_plot['Espece_User'],
                'Ecologie': pdf_plot['Ecologie'],
            })

            # jitter si points dupliqués
            dup_mask = data_plot.duplicated(subset=[x_trait, y_trait], keep=False)
            if dup_mask.any():
                rng_x = data_plot[x_trait].max() - data_plot[x_trait].min()
                rng_y = data_plot[y_trait].max() - data_plot[y_trait].min()
                jitter_x = rng_x * 0.015 if rng_x > 1e-9 else 0.015
                jitter_y = rng_y * 0.015 if rng_y > 1e-9 else 0.015
                for _, grp in data_plot[dup_mask].groupby([x_trait, y_trait]):
                    for j, idx in enumerate(grp.index):
                        ang = 2 * np.pi * j / len(grp)
                        data_plot.loc[idx, x_trait] += jitter_x * np.cos(ang)
                        data_plot.loc[idx, y_trait] += jitter_y * np.sin(ang)

            fig_sc = px.scatter(
                data_plot,
                x=x_trait,
                y=y_trait,
                color='Cluster',
                text='Espece_User',
                hover_name='Espece_User',
                custom_data=['Espece_User', 'Ecologie'],
                template='plotly_dark',
                height=600,
                color_discrete_sequence=COLOR_SEQUENCE,
            )
            fig_sc.update_traces(
                textposition="top center",
                marker=dict(opacity=0.8, size=8),
                textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS),
                hovertemplate=(f"<span style='font-size:{HOVER_SPECIES_FONT_SIZE}px'><b>%{{customdata[0]}}</b></span><br><br>"
                               f"<span style='font-size:{HOVER_ECOLOGY_TITLE_FONT_SIZE}px'><i>Écologie :</i></span><br>"
                               f"<span style='font-size:{HOVER_ECOLOGY_TEXT_FONT_SIZE}px'>%{{customdata[1]}}</span><extra></extra>"),
            )
            # convex hull par cluster
            uniq_clusters = sorted(data_plot['Cluster'].unique())
            col_map = {c: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, c in enumerate(uniq_clusters)}
            for cl in uniq_clusters:
                pts = data_plot[data_plot['Cluster'] == cl][[x_trait, y_trait]].drop_duplicates().values
                if len(pts) >= MIN_POINTS_FOR_HULL:
                    hull = ConvexHull(pts)
                    hull_path = pts[np.append(hull.vertices, hull.vertices[0])]
                    clr = col_map[cl]
                    fig_sc.add_trace(go.Scatter(x=hull_path[:, 0], y=hull_path[:, 1], fill='toself', fillcolor=clr,
                                                 line=dict(color=clr), mode='lines', opacity=0.2, showlegend=False, hoverinfo='skip'))
            fig_sc.update_layout(title_text=f"{y_trait} vs {x_trait}", title_x=0.5, dragmode='pan')
            st.plotly_chart(fig_sc, use_container_width=True, config={'scrollZoom': True})

# -----------------------------------------------------------------------------
# ÉTAPE 4 : ACP – graphique PCA principal + dendrogramme + composition clusters
# -----------------------------------------------------------------------------
if st.session_state.analysis_done_once and not st.session_state.sub.empty:
    st.markdown("---")
    st.subheader("Étape 4 : Résultats ACP & dendrogramme")

    pdf_disp = st.session_state.pdf.copy()
    # PCA plot
    if not pdf_disp.empty and 'PC1' in pdf_disp.columns and 'PC2' in pdf_disp.columns:
        fig_pca = px.scatter(
            pdf_disp,
            x='PC1',
            y='PC2',
            color='Cluster',
            text='Espece_User',
            hover_name='Espece_User',
            custom_data=['Espece_User', 'Ecologie'],
            template='plotly_dark',
            height=500,
            color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig_pca.update_traces(
            textposition="top center",
            marker=dict(opacity=0.7),
            textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS),
            hovertemplate=(f"<span style='font-size:{HOVER_SPECIES_FONT_SIZE}px'><b>%{{customdata[0]}}</b></span><br><br>"
                           f"<span style='font-size:{HOVER_ECOLOGY_TITLE_FONT_SIZE}px'><i>Écologie :</i></span><br>"
                           f"<span style='font-size:{HOVER_ECOLOGY_TEXT_FONT_SIZE}px'>%{{customdata[1]}}</span><extra></extra>")
        )
        st.plotly_chart(fig_pca, use_container_width=True, config={'scrollZoom': True})
    else:
        st.info("Graphique PCA non disponible (moins de 2 composantes ou données manquantes).")

    # Dendrogramme
    if isinstance(st.session_state.X_for_dendro, np.ndarray) and st.session_state.X_for_dendro.shape[0] > 1:
        try:
            Z = linkage(st.session_state.X_for_dendro, method='ward')
            thresh = 0
            if n_clusters_selected > 1 and (n_clusters_selected - 1) < Z.shape[0]:
                thresh = Z[-(n_clusters_selected - 1), 2] * 0.99
            labels_den = pdf_disp['Espece_User'].tolist()
            fig_den = ff.create_dendrogram(
                st.session_state.X_for_dendro,
                orientation='left',
                labels=labels_den,
                linkagefun=lambda _: Z,
                color_threshold=thresh,
                colorscale=COLOR_SEQUENCE,
            )
            fig_den.update_layout(template='plotly_dark', height=max(400, 20 * st.session_state.sub.shape[0]))
            st.plotly_chart(fig_den, use_container_width=True)
        except Exception as e:
            st.warning(f"Erreur dendrogramme : {e}")
    else:
        st.info("Dendrogramme non généré (au moins 2 espèces nécessaires).")

    # Composition des clusters
    st.markdown("##### Composition des clusters")
    if 'Cluster' in pdf_disp.columns:
        comps = pdf_disp.groupby('Cluster')['Espece_User'].apply(list).to_dict()
        n_cols_comp = min(len(comps), 3) if comps else 1
        cols_comp = st.columns(n_cols_comp)
        for idx, (cl, esp_list) in enumerate(sorted(comps.items(), key=lambda x: x[0])):
            with cols_comp[idx % n_cols_comp]:
                st.markdown(f"**Cluster {cl}** ({len(esp_list)} espèces)")
                for s in sorted(esp_list):
                    st.markdown(f"- {s}")
    else:
        st.info("Clusters non disponibles dans les résultats.")
