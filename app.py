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
# Pour les besoins de cet exemple, si core.py n'est pas disponible,
# nous allons simuler sa fonction analyse.
try:
    import core
except ImportError:
    st.warning("Le module 'core.py' est introuvable. Une fonction d'analyse simulée sera utilisée. L'ACP et le dendrogramme réels ne fonctionneront pas.")
    # Simulation de la fonction core.analyse pour permettre à l'UI de fonctionner
    class MockPCA:
        def __init__(self):
            # Simuler des composantes et variances expliquées plus réalistes
            self.components_ = np.array([[0.707, 0.707], [-0.707, 0.707]]) 
            self.explained_variance_ = np.array([0.6, 0.3]) # Somme < nombre de variables initiales

    def mock_analyse(sub_df, n_clusters):
        n_samples = len(sub_df)
        numeric_cols = sub_df.select_dtypes(include=np.number)
        n_features = numeric_cols.shape[1]

        if n_samples == 0 or n_features == 0:
            # Retourner des structures vides mais conformes
            return np.array([]), MockPCA(), pd.DataFrame(columns=['PC1', 'PC2']), pd.DataFrame()
        
        # Simuler des coordonnées PCA (2 composantes)
        coords_array = np.random.rand(n_samples, min(2, n_features)) * 10
        pc_cols = [f"PC{i+1}" for i in range(coords_array.shape[1])]
        coords_df = pd.DataFrame(coords_array, columns=pc_cols)

        # Simuler des labels de cluster
        if n_samples < n_clusters and n_samples > 0 : 
            labels = np.arange(n_samples)
        elif n_samples >= n_clusters and n_clusters > 0:
            labels = np.random.randint(0, n_clusters, n_samples)
        else: 
            labels = np.array([])
        
        # Simuler X (données normalisées pour le dendrogramme)
        if not numeric_cols.empty:
            X_scaled = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
            X_scaled = X_scaled.fillna(0).values
        else: 
            X_scaled = np.random.rand(n_samples, min(2, n_features)) if n_samples > 0 else np.array([]).reshape(0,min(2,n_features))

        mock_pca_obj = MockPCA()
        
        if n_features > 0:
            # Ajuster la taille des composantes simulées au nombre de traits réels
            # Les composantes sont (n_components, n_features) dans scikit-learn
            # Ici, on simule components_ comme (n_features, n_components) pour correspondre à l'usage
            # pca_results.components_ (qui est (n_components, n_features)) puis transposé.
            # Pour la simulation, components_ sera (n_features, n_pcs_simulated)
            n_pcs_simulated = min(2, n_features)
            sim_components = np.random.rand(n_features, n_pcs_simulated)
            # Normaliser les vecteurs propres simulés (colonnes de sim_components)
            if sim_components.size > 0:
                 sim_components = sim_components / np.linalg.norm(sim_components, axis=0, keepdims=True)
            
            mock_pca_obj.components_ = sim_components.T # Transposer pour correspondre à scikit-learn (n_pcs_simulated, n_features)
            
            # Simuler explained_variance_ de manière décroissante et somme <= n_pcs_simulated (pour données std)
            sim_explained_variance = np.sort(np.random.rand(n_pcs_simulated))[::-1]
            sim_explained_variance = sim_explained_variance / sim_explained_variance.sum() * n_pcs_simulated * 0.7 # Assurer que c'est plausible
            mock_pca_obj.explained_variance_ = sim_explained_variance

        else: 
            mock_pca_obj.components_ = np.array([])
            mock_pca_obj.explained_variance_ = np.array([])
            if n_samples > 0 and coords_df.empty: 
                coords_df = pd.DataFrame(np.random.rand(n_samples, 2), columns=['PC1', 'PC2'])
        
        # S'assurer que coords_df a le bon nombre de colonnes si pca_obj a moins de 2 composantes
        if mock_pca_obj.explained_variance_.shape[0] < 2 and not coords_df.empty:
            if mock_pca_obj.explained_variance_.shape[0] == 1 and 'PC2' in coords_df.columns:
                coords_df = coords_df[['PC1']] # Garder seulement PC1
            elif mock_pca_obj.explained_variance_.shape[0] == 0:
                 coords_df = pd.DataFrame(index=coords_df.index)


        return labels, mock_pca_obj, coords_df, X_scaled

    core = type('CoreModule', (object,), {'analyse': mock_analyse, 'read_reference': lambda fp: pd.DataFrame()})


# ---------------------------------------------------------------------------- #
# CONFIGURATION UI
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="PCA flore interactive", layout="wide")
st.markdown("<h1 style='text-align: center;'>Analyse interactive des clusters botaniques</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
/* Style général pour l'éditeur de données */
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

/* Style pour la première ligne du st.data_editor (noms des habitats) - CONSERVÉ POUR L'ASPECT VISUEL */
div[data-testid="stDataEditor"] .glideDataEditor-body .dvn-scroll-inner > div:first-child > div[data-cell^="[0,"] > div {
    background-color: #22272f !important; /* Couleur de fond pour thème sombre */
    color: #e1e1e1 !important;           /* Couleur de texte pour thème sombre */
    font-weight: bold !important;
}
/* Style pour la cellule de la première ligne en mode édition */
div[data-testid="stDataEditor"] .glideDataEditor-body .dvn-scroll-inner > div:first-child > div[data-cell^="[0,"] > div > .gdg-input {
    background-color: #ffffff !important; /* Fond blanc pour l'éditeur */
    color: #000000 !important;           /* Texte noir pour l'éditeur */
    font-weight: normal !important;      /* Poids normal pour l'éditeur */
}

/* Ajuster la taille des boutons de sélection d'habitat (si on les utilise ailleurs) */
/* Pour les boutons de sélection d'habitat spécifiques, nous utiliserons des clés ou des classes CSS si nécessaire */
.habitat-select-button button {
    font-size: 13px !important;
    padding: 0.25rem 0.6rem !important; /* Ajuster le padding */
    line-height: 1.4;
    width: 100%; /* Pour qu'ils prennent toute la largeur de la colonne */
    border-radius: 0.5rem; /* Coins arrondis */
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------- #
# CONSTANTES ET CHARGEMENT DE DONNÉES INITIALES
# ---------------------------------------------------------------------------- #
MIN_POINTS_FOR_HULL = 3
COLOR_SEQUENCE = px.colors.qualitative.Plotly
LABEL_FONT_SIZE_ON_PLOTS = 15 
HOVER_SPECIES_FONT_SIZE = 15    
HOVER_ECOLOGY_TITLE_FONT_SIZE = 14 
HOVER_ECOLOGY_TEXT_FONT_SIZE = 13  

@st.cache_data
def load_data(file_path="data_ref.csv"):
    try:
        if not hasattr(core, "read_reference") or callable(getattr(core, "read_reference", None)) and core.read_reference.__name__ == '<lambda>': # Check if using mock
            st.warning(f"Simulation du chargement de '{file_path}'. Le fichier réel n'est pas utilisé.")
            example_species = [f"Espece Alpha {i}" for i in range(1, 11)] + \
                                [f"Espece Beta {i}" for i in range(1, 11)] + \
                                [f"Espece Gamma {i}" for i in range(1, 11)]
            data = pd.DataFrame({
                'Espece': example_species,
                'Trait_Num_1': np.random.rand(30) * 10,
                'Trait_Num_2': np.random.randint(1, 100, 30),
                'Trait_Cat_1': np.random.choice(['X', 'Y', 'Z'], 30),
                'Humidité_édaphique': np.random.rand(30) * 5 + 1, # Ajout de traits simulés
                'Matière_organique': np.random.rand(30) * 10,
                'Lumière': np.random.rand(30) * 1000
            })
            # Ajouter quelques espèces spécifiques pour le test des avertissements
            data.loc[len(data)] = ['Rhamnus pumila', 5.0, 50, 'X', 3, 5, 500]
            data.loc[len(data)] = ['Vulpia sp.', 2.0, 20, 'Y', 2, 2, 800]
            return data
        
        data = core.read_reference(file_path) 
        if data.empty:
            st.warning(f"Le fichier de données de traits '{file_path}' est vide ou n'a pas pu être lu correctement par core.read_reference.")
        return data
    except FileNotFoundError:
        st.error(f"ERREUR CRITIQUE: Fichier de données de traits '{file_path}' non trouvé. L'application ne peut pas fonctionner.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"ERREUR CRITIQUE: Impossible de charger les données de traits depuis '{file_path}': {e}")
        return pd.DataFrame()

ref = load_data()

ref_binom_series = pd.Series(dtype='str')
if not ref.empty and 'Espece' in ref.columns:
    ref_binom_series = (
        ref["Espece"]
        .astype(str) 
        .str.split()
        .str[:2]
        .str.join(" ")
        .str.lower()
    )
else:
    if ref.empty:
        st.warning("Le DataFrame de référence 'ref' est vide. Impossible de créer 'ref_binom_series'.")
    elif 'Espece' not in ref.columns:
        st.warning("La colonne 'Espece' est manquante dans le DataFrame de référence 'ref'. Impossible de créer 'ref_binom_series'.")


# ---------------------------------------------------------------------------- #
# FONCTION UTILITAIRE POUR FORMATER L'ÉCOLOGIE
# ---------------------------------------------------------------------------- #
def format_ecology_for_hover(text, line_width_chars=65):
    if pd.isna(text) or str(text).strip() == "":
        return "Description écologique non disponible."
    wrapped_lines = textwrap.wrap(str(text), width=line_width_chars) 
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
            encoding='utf-8-sig',
            keep_default_na=False, # Important pour ne pas interpréter "" comme NaN trop tôt
            na_values=[''] # Définir explicitement les chaînes vides comme NaN si nécessaire après lecture
        )
        eco_data = eco_data.dropna(subset=['Espece']) # Supprimer les lignes où l'espèce est NaN
        eco_data['Espece'] = eco_data['Espece'].astype(str).str.strip()
        eco_data = eco_data[eco_data['Espece'] != ""] # Filtrer les noms d'espèces vides après strip

        if eco_data.empty:
            st.warning(f"Le fichier écologique '{file_path}' est vide ou ne contient aucune donnée d'espèce valide.")
            return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))

        eco_data['Espece_norm'] = (
            eco_data['Espece']
            .str.split()
            .str[:2]
            .str.join(" ")
            .str.lower()
        )
        eco_data = eco_data.drop_duplicates(subset=['Espece_norm'], keep='first')
        eco_data = eco_data.set_index('Espece_norm') # Espece_norm devient l'index
        return eco_data[["Description_Ecologie"]] # Retourne DF avec Espece_norm comme index
    
    except FileNotFoundError:
        print(f"AVERTISSEMENT: Fichier de données écologiques '{file_path}' non trouvé.")
        st.toast(f"Fichier écologique '{file_path}' non trouvé.", icon="⚠️")
        return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))
    except pd.errors.EmptyDataError:
        st.warning(f"Le fichier écologique '{file_path}' est vide.")
        return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))
    except ValueError as ve: # Peut arriver si usecols ne correspond pas, etc.
        print(f"AVERTISSEMENT: Erreur de valeur lors de la lecture du fichier '{file_path}'. Détails: {ve}.")
        st.toast(f"Erreur format fichier écologique '{file_path}'.", icon="🔥")
        return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))
    except Exception as e:
        print(f"AVERTISSEMENT: Impossible de charger les données écologiques depuis '{file_path}': {e}.")
        st.toast(f"Erreur chargement fichier écologique.", icon="🔥")
        return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))

ecology_df = load_ecology_data()

# ---------------------------------------------------------------------------- #
# INITIALISATION DES ETATS DE SESSION
# ---------------------------------------------------------------------------- #
default_session_states = {
    'x_axis_trait_interactive': None,
    'y_axis_trait_interactive': None,
    'run_main_analysis_once': False, 
    'vip_data_df_interactive': pd.DataFrame(),
    'vip_data_df_interactive_snapshot_for_comparison': pd.DataFrame(), 
    'sub': pd.DataFrame(), 
    'pdf': pd.DataFrame(), 
    'X_for_dendro': np.array([]), 
    'numeric_trait_names_for_interactive_plot': [],
    'selected_habitats_indices': None, 
    'previous_num_cols': 0,
    'analysis_has_run_for_current_selection': False 
}

for key, value in default_session_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

if 'releves_df' not in st.session_state or not isinstance(st.session_state.releves_df, pd.DataFrame):
    num_placeholder_cols = 15
    num_placeholder_rows_total = 11 
    header = [f"Habitat {j+1}" for j in range(num_placeholder_cols)] 
    placeholder_rows = [["" for _ in range(num_placeholder_cols)] for _ in range(num_placeholder_rows_total -1)]
    st.session_state.releves_df = pd.DataFrame([header] + placeholder_rows)
    st.session_state.releves_df.columns = [str(col) for col in st.session_state.releves_df.columns] 
    st.session_state.previous_num_cols = num_placeholder_cols


# ---------------------------------------------------------------------------- #
# ÉTAPE 1: IMPORTATION ET SÉLECTION DES RELEVÉS FLORISTIQUES
# ---------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("Étape 1: Importation et Sélection des Relevés Floristiques")

current_releves_df_for_selection = st.session_state.releves_df.copy()

if not current_releves_df_for_selection.empty and \
   len(current_releves_df_for_selection.columns) > 0 and \
   len(current_releves_df_for_selection) > 0:
    
    habitat_names_from_df = current_releves_df_for_selection.iloc[0].astype(str).tolist()
    num_actual_cols = len(current_releves_df_for_selection.columns)
    
    st.markdown("**Cliquez sur le nom d'un habitat ci-dessous pour le sélectionner pour l'analyse :**")
    
    if st.session_state.selected_habitats_indices is not None and \
       st.session_state.selected_habitats_indices >= num_actual_cols:
        st.session_state.selected_habitats_indices = None
        st.session_state.analysis_has_run_for_current_selection = False

    if num_actual_cols > 0:
        button_cols_layout = st.columns(num_actual_cols) 
        
        for i in range(num_actual_cols):
            habitat_name_for_button = habitat_names_from_df[i] if pd.notna(habitat_names_from_df[i]) and str(habitat_names_from_df[i]).strip() != "" else f"Relevé {i+1}"
            is_selected = (st.session_state.selected_habitats_indices == i)
            
            button_type = "primary" if is_selected else "secondary"
            button_key = f"habitat_select_button_{i}"

            with button_cols_layout[i]:
                st.markdown(f'<div class="habitat-select-button">', unsafe_allow_html=True)
                if st.button(habitat_name_for_button, key=button_key, type=button_type, use_container_width=True):
                    if st.session_state.selected_habitats_indices != i:
                        st.session_state.selected_habitats_indices = i
                        st.session_state.run_main_analysis_once = False 
                        st.session_state.analysis_has_run_for_current_selection = False 
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Ajoutez des colonnes au tableau pour pouvoir sélectionner des relevés.")
else:
    st.warning("Le tableau de données est vide ou ne contient pas de colonnes pour la sélection.")

st.info("Copiez-collez vos données de relevés ici (Ctrl+V ou Cmd+V). La première ligne doit contenir les noms des habitats/relevés. Les lignes suivantes contiennent les espèces.")

if not all(isinstance(col, str) for col in st.session_state.releves_df.columns):
    st.session_state.releves_df.columns = [str(col) for col in st.session_state.releves_df.columns]

edited_releves_df_from_editor = st.data_editor(
    st.session_state.releves_df,
    num_rows="dynamic",
    use_container_width=True,
    key="releves_data_editor_key" 
)

if not edited_releves_df_from_editor.equals(st.session_state.releves_df):
    st.session_state.releves_df = edited_releves_df_from_editor.copy()
    if len(st.session_state.releves_df.columns) != st.session_state.previous_num_cols:
        if st.session_state.selected_habitats_indices is not None and \
           st.session_state.selected_habitats_indices >= len(st.session_state.releves_df.columns):
            st.session_state.selected_habitats_indices = None
            st.session_state.analysis_has_run_for_current_selection = False
            st.session_state.run_main_analysis_once = False 
        st.session_state.previous_num_cols = len(st.session_state.releves_df.columns)
    st.rerun()

fig_pca = None
fig_dend = None
species_binom_user_unique = [] 

selected_habitat_idx_for_analysis = st.session_state.get('selected_habitats_indices', None)

if selected_habitat_idx_for_analysis is not None and \
   not ref.empty and \
   not st.session_state.get('analysis_has_run_for_current_selection', False):

    st.session_state.run_main_analysis_once = True 
    st.session_state.analysis_has_run_for_current_selection = True 

    st.session_state.sub = pd.DataFrame()
    st.session_state.pdf = pd.DataFrame()
    st.session_state.X_for_dendro = np.array([])
    st.session_state.vip_data_df_interactive = pd.DataFrame()
    st.session_state.numeric_trait_names_for_interactive_plot = []

    species_raw_from_table = []
    df_for_species_extraction = st.session_state.releves_df.copy() 
    
    if not df_for_species_extraction.empty and len(df_for_species_extraction) > 1:
        if selected_habitat_idx_for_analysis < len(df_for_species_extraction.columns):
            species_in_col_series = df_for_species_extraction.iloc[1:, selected_habitat_idx_for_analysis]
            species_in_col_cleaned = species_in_col_series.dropna().astype(str).str.strip().replace('', np.nan).dropna().tolist()
            species_raw_from_table.extend(s for s in species_in_col_cleaned if s)

    species_raw_unique_temp = sorted(list(set(s for s in species_raw_from_table if s)))
    species_binom_user_unique = [" ".join(s.split()[:2]).lower() for s in species_raw_unique_temp if s and len(s.split()) >=1] 

    if not species_binom_user_unique:
        st.error("Aucune espèce valide (nom binomial) extraite du relevé sélectionné. Vérifiez vos données et sélection.")
        st.session_state.run_main_analysis_once = False 
        st.session_state.analysis_has_run_for_current_selection = False 
    else:
        indices_to_keep_from_ref = []
        if not ref_binom_series.empty:
            if isinstance(ref_binom_series, pd.Series):
                ref_indexed_binom = ref_binom_series.reset_index()
                if 'index' in ref_indexed_binom.columns and ref_binom_series.name is not None and ref_binom_series.name in ref_indexed_binom.columns:
                    ref_indexed_binom.columns = ['Original_Ref_Index', 'ref_binom_val']
                elif len(ref_indexed_binom.columns) == 2:
                    ref_indexed_binom.columns = ['Original_Ref_Index', 'ref_binom_val']
                else:
                    st.error("Structure inattendue de ref_binom_series après reset_index.")
                    st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;
                    
                if st.session_state.run_main_analysis_once : 
                    for user_binom_specie in species_binom_user_unique:
                        matches_in_ref = ref_indexed_binom[ref_indexed_binom['ref_binom_val'] == user_binom_specie]
                        if not matches_in_ref.empty:
                            indices_to_keep_from_ref.append(matches_in_ref['Original_Ref_Index'].iloc[0])
            else:
                st.error("ref_binom_series n'est pas une Series Pandas.")
                st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;

        if st.session_state.run_main_analysis_once: 
            indices_to_keep_from_ref = sorted(list(set(indices_to_keep_from_ref)))

            if indices_to_keep_from_ref:
                st.session_state.sub = ref.loc[indices_to_keep_from_ref].copy()
            else:
                st.session_state.sub = pd.DataFrame(columns=ref.columns if not ref.empty else None)
        
            found_ref_binom_values_in_sub = []
            if not st.session_state.sub.empty and 'Espece' in st.session_state.sub.columns:
                found_ref_binom_values_in_sub = ( st.session_state.sub["Espece"].astype(str).str.split().str[:2].str.join(" ").str.lower().tolist() )
        
            raw_names_corresponding_to_binom_user_unique = [raw_name for raw_name in species_raw_unique_temp if " ".join(raw_name.split()[:2]).lower() in species_binom_user_unique]
            not_found_user_raw_names = [raw_names_corresponding_to_binom_user_unique[i] for i, user_binom_name in enumerate(species_binom_user_unique) if user_binom_name not in found_ref_binom_values_in_sub]

            if not_found_user_raw_names:
                st.warning("Non trouvées dans la base de traits : " + ", ".join(not_found_user_raw_names), icon="⚠️")

            n_clusters_selected_main = st.session_state.get('n_clusters_slider_main_value', 3) 

            if st.session_state.sub.empty:
                st.error("Aucune des espèces sélectionnées n'a été trouvée dans la base de traits. L'analyse ne peut continuer.")
                st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;
            elif st.session_state.sub.shape[0] < n_clusters_selected_main and n_clusters_selected_main > 0 :
                st.error(f"Espèces trouvées ({st.session_state.sub.shape[0]}) < clusters demandés ({n_clusters_selected_main}). Ajustez le nombre de clusters ou vérifiez les espèces.");
                st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;
            elif st.session_state.sub.shape[0] < 2:
                st.error(f"Au moins 2 espèces sont nécessaires pour l'analyse. {st.session_state.sub.shape[0]} trouvée(s).");
                st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;
            else:
                user_input_binom_to_raw_map = { " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_raw_unique_temp if s_raw and len(s_raw.split()) >=1}
                try:
                    sub_for_analysis = st.session_state.sub.select_dtypes(include=np.number)
                    if sub_for_analysis.empty or sub_for_analysis.shape[1] == 0:
                        st.error(f"Aucun trait numérique trouvé pour les espèces sélectionnées. L'ACP est impossible.")
                        st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;
                    else:
                        labels, pca_results, coords_df, X_scaled_data = core.analyse(st.session_state.sub, n_clusters_selected_main)
                        
                        if not isinstance(coords_df, pd.DataFrame):
                            if isinstance(coords_df, np.ndarray) and coords_df.ndim == 2 and coords_df.shape[0] == len(st.session_state.sub):
                                num_pcs = coords_df.shape[1]
                                coords_df = pd.DataFrame(coords_df, columns=[f"PC{i+1}" for i in range(num_pcs)], index=st.session_state.sub.index)
                            else: 
                                st.error("Coordonnées PCA (core.analyse) : format ou longueur incorrecte.")
                                st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;
                                coords_df = pd.DataFrame() 

                        if st.session_state.run_main_analysis_once: 
                            current_pdf = coords_df.copy()
                            if not current_pdf.empty:
                                if len(labels) == len(current_pdf): current_pdf["Cluster"] = labels.astype(str)
                                else: current_pdf["Cluster"] = np.zeros(len(current_pdf)).astype(str) if len(current_pdf) > 0 else pd.Series(dtype=str)
                                
                                if 'Espece' in st.session_state.sub.columns:
                                    current_pdf["Espece_Ref"] = st.session_state.sub["Espece"].values[:len(current_pdf)]
                                    current_pdf["Espece_User"] = current_pdf["Espece_Ref"].apply(lambda full_ref_name: user_input_binom_to_raw_map.get(" ".join(str(full_ref_name).split()[:2]).lower(),str(full_ref_name)))
                                else: 
                                    current_pdf["Espece_Ref"] = [f"Espèce_Ref_{i}" for i in range(len(current_pdf))]
                                    current_pdf["Espece_User"] = [f"Espèce_User_{i}" for i in range(len(current_pdf))]

                                if not ecology_df.empty:
                                    current_pdf['Espece_Ref_norm_for_eco'] = current_pdf['Espece_Ref'].astype(str).str.strip().str.split().str[:2].str.join(" ").str.lower()
                                    if ecology_df.index.name == 'Espece_norm' and 'Description_Ecologie' in ecology_df.columns:
                                        current_pdf['Ecologie_raw'] = current_pdf['Espece_Ref_norm_for_eco'].map(ecology_df['Description_Ecologie'])
                                    else:
                                        current_pdf['Ecologie_raw'] = pd.Series([np.nan] * len(current_pdf), index=current_pdf.index)
                                    current_pdf['Ecologie'] = current_pdf['Ecologie_raw'].apply(lambda x: format_ecology_for_hover(x))
                                    current_pdf['Ecologie'] = current_pdf['Ecologie'].fillna(format_ecology_for_hover(None))
                                else:
                                    current_pdf['Ecologie'] = format_ecology_for_hover(None) 
                                st.session_state.pdf = current_pdf.copy()

                                if hasattr(pca_results, 'components_') and hasattr(pca_results, 'explained_variance_') and \
                                   isinstance(pca_results.components_, np.ndarray) and isinstance(pca_results.explained_variance_, np.ndarray) and \
                                   pca_results.components_.size > 0 and pca_results.explained_variance_.size > 0 :
                                    
                                    pca_components_values = pca_results.components_ # (n_components, n_features)
                                    explained_variance_values = pca_results.explained_variance_ # (n_components,)
                                    
                                    # Loadings: (n_features, n_components)
                                    # Chaque colonne est un vecteur de loadings pour une composante.
                                    # Chaque ligne est un vecteur de loadings pour une variable.
                                    # Correct calculation for loadings: components.T * sqrt(explained_variance)
                                    # components_ are (n_components, n_features)
                                    # explained_variance_ is (n_components,)
                                    # loadings should be (n_features, n_components)
                                    
                                    # Si pca_components_values est (n_features, n_components) comme dans la simulation précédente
                                    # loadings = pca_components_values * np.sqrt(explained_variance_values[np.newaxis, :])
                                    
                                    # Si pca_components_values est (n_components, n_features) comme dans scikit-learn
                                    loadings = pca_components_values.T * np.sqrt(explained_variance_values[:, np.newaxis])
                                    loadings = loadings.T # Transposer pour avoir (n_features, n_components)
                                                                        
                                    communal = (loadings**2).sum(axis=1) # Somme sur les composantes pour chaque variable
                                    
                                    trait_columns_for_communal = st.session_state.sub.select_dtypes(include=np.number).columns.tolist()
                                    
                                    if len(communal) == len(trait_columns_for_communal):
                                        communal_percent = (communal * 100).round(0).astype(int)
                                        # CORRECTION: Borner les communalités entre 0 et 100%
                                        communal_percent_clipped = np.clip(communal_percent, 0, 100)
                                        
                                        st.session_state.vip_data_df_for_calc = pd.DataFrame({
                                            "Variable": trait_columns_for_communal,
                                            "Communalité (%)": communal_percent_clipped, # Utiliser la valeur bornée
                                        }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
                                    else: 
                                        st.session_state.vip_data_df_for_calc = pd.DataFrame(columns=["Variable", "Communalité (%)"])
                                        st.warning(f"Communalités non calculées (dimensions des loadings/traits incohérentes: {len(communal)} vs {len(trait_columns_for_communal)}).")
                                else:
                                    st.session_state.vip_data_df_for_calc = pd.DataFrame(columns=["Variable", "Communalité (%)"])
                                    st.warning("Résultats PCA incomplets pour communalités (components_ ou explained_variance_ manquants/incorrects).")
                                
                                st.session_state.X_for_dendro = X_scaled_data if isinstance(X_scaled_data, np.ndarray) else np.array([])
                                all_trait_names_from_sub = [col for col in st.session_state.sub.columns if col.lower() != "espece"]
                                st.session_state.numeric_trait_names_for_interactive_plot = [col for col in all_trait_names_from_sub if pd.api.types.is_numeric_dtype(st.session_state.sub[col])]
                                
                                numeric_trait_names_init = st.session_state.numeric_trait_names_for_interactive_plot
                                default_x_init, default_y_init = None, None

                                if not st.session_state.vip_data_df_for_calc.empty and len(numeric_trait_names_init) >= 1:
                                    top_vars_from_vip_numeric = [var for var in st.session_state.vip_data_df_for_calc["Variable"].tolist() if var in numeric_trait_names_init]
                                    if len(top_vars_from_vip_numeric) >= 1: default_x_init = top_vars_from_vip_numeric[0]
                                    if len(top_vars_from_vip_numeric) >= 2: default_y_init = top_vars_from_vip_numeric[1]
                                    elif len(top_vars_from_vip_numeric) == 1: 
                                        other_numeric_traits = [t for t in numeric_trait_names_init if t != default_x_init]
                                        default_y_init = other_numeric_traits[0] if other_numeric_traits else default_x_init
                                
                                if default_x_init is None and len(numeric_trait_names_init) >= 1: default_x_init = numeric_trait_names_init[0]
                                if default_y_init is None:
                                    if len(numeric_trait_names_init) >= 2: default_y_init = numeric_trait_names_init[1]
                                    elif default_x_init and len(numeric_trait_names_init) == 1: default_y_init = default_x_init

                                st.session_state.x_axis_trait_interactive = default_x_init
                                st.session_state.y_axis_trait_interactive = default_y_init
                                
                                if not st.session_state.vip_data_df_for_calc.empty and numeric_trait_names_init:
                                    temp_interactive_df = st.session_state.vip_data_df_for_calc[st.session_state.vip_data_df_for_calc["Variable"].isin(numeric_trait_names_init)].copy()
                                    temp_interactive_df["Axe X"] = temp_interactive_df["Variable"] == st.session_state.x_axis_trait_interactive
                                    temp_interactive_df["Axe Y"] = temp_interactive_df["Variable"] == st.session_state.y_axis_trait_interactive
                                    st.session_state.vip_data_df_interactive = temp_interactive_df[["Variable", "Communalité (%)", "Axe X", "Axe Y"]].reset_index(drop=True)
                                else:
                                    st.session_state.vip_data_df_interactive = pd.DataFrame(columns=["Variable", "Communalité (%)", "Axe X", "Axe Y"])
                                st.session_state.vip_data_df_interactive_snapshot_for_comparison = st.session_state.vip_data_df_interactive.copy()
                            else: 
                                st.warning("L'analyse n'a pas produit de coordonnées PCA (coords_df vide).")
                                st.session_state.run_main_analysis_once = False
                                st.session_state.analysis_has_run_for_current_selection = False
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse principale : {e}"); st.exception(e)
                    st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;

if selected_habitat_idx_for_analysis is None and not ref.empty:
    st.info("Veuillez sélectionner un habitat à l'Étape 1 pour lancer l'analyse.")
elif ref.empty:
     st.warning("Les données de référence ('data_ref.csv') n'ont pas pu être chargées ou sont simulées. L'analyse est désactivée si les données réelles manquent.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 2: EXPLORATION INTERACTIVE DES VARIABLES
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty:
    st.markdown("---"); st.subheader("Étape 2: Exploration Interactive des Variables")
    col_interactive_table, col_interactive_graph = st.columns([1, 2]) 

    with col_interactive_table:
        st.markdown("##### Tableau d'exploration interactif")
        df_editor_source_interactive = st.session_state.get('vip_data_df_interactive', pd.DataFrame())

        if not df_editor_source_interactive.empty:
            snapshot_cols = list(st.session_state.get('vip_data_df_interactive_snapshot_for_comparison', pd.DataFrame()).columns)
            current_cols = list(df_editor_source_interactive.columns)
            if 'vip_data_df_interactive_snapshot_for_comparison' not in st.session_state or snapshot_cols != current_cols:
                st.session_state.vip_data_df_interactive_snapshot_for_comparison = df_editor_source_interactive.copy()

            edited_df_interactive = st.data_editor(
                df_editor_source_interactive, 
                column_config={
                    "Variable": st.column_config.TextColumn("Variable", disabled=True),
                    "Communalité (%)": st.column_config.NumberColumn("Communalité (%)", format="%d%%", disabled=True),
                    "Axe X": st.column_config.CheckboxColumn("Axe X"),
                    "Axe Y": st.column_config.CheckboxColumn("Axe Y")
                }, 
                key="interactive_exploration_editor", 
                use_container_width=True, 
                hide_index=True, 
                num_rows="fixed"
            )
            
            made_change_in_interactive_axes = False
            new_x_trait_interactive = st.session_state.x_axis_trait_interactive
            new_y_trait_interactive = st.session_state.y_axis_trait_interactive

            selected_x_vars_interactive = edited_df_interactive[edited_df_interactive["Axe X"]]["Variable"].tolist()
            if selected_x_vars_interactive:
                chosen_x = selected_x_vars_interactive[-1] 
                if chosen_x != st.session_state.x_axis_trait_interactive:
                    new_x_trait_interactive = chosen_x
                    made_change_in_interactive_axes = True
                elif len(selected_x_vars_interactive) > 1 : 
                    made_change_in_interactive_axes = True 
            elif st.session_state.x_axis_trait_interactive is not None: 
                new_x_trait_interactive = None
                made_change_in_interactive_axes = True
            
            selected_y_vars_interactive = edited_df_interactive[edited_df_interactive["Axe Y"]]["Variable"].tolist()
            if selected_y_vars_interactive:
                chosen_y = selected_y_vars_interactive[-1]
                if chosen_y != st.session_state.y_axis_trait_interactive:
                    new_y_trait_interactive = chosen_y
                    made_change_in_interactive_axes = True
                elif len(selected_y_vars_interactive) > 1 :
                    made_change_in_interactive_axes = True
            elif st.session_state.y_axis_trait_interactive is not None:
                new_y_trait_interactive = None
                made_change_in_interactive_axes = True

            if made_change_in_interactive_axes:
                st.session_state.x_axis_trait_interactive = new_x_trait_interactive
                st.session_state.y_axis_trait_interactive = new_y_trait_interactive
                
                df_updated_for_editor = df_editor_source_interactive.copy()
                df_updated_for_editor["Axe X"] = (df_updated_for_editor["Variable"] == new_x_trait_interactive)
                df_updated_for_editor["Axe Y"] = (df_updated_for_editor["Variable"] == new_y_trait_interactive)
                st.session_state.vip_data_df_interactive = df_updated_for_editor
                st.session_state.vip_data_df_interactive_snapshot_for_comparison = df_updated_for_editor.copy()
                st.rerun()
            elif not edited_df_interactive.equals(st.session_state.vip_data_df_interactive_snapshot_for_comparison):
                 st.session_state.vip_data_df_interactive_snapshot_for_comparison = edited_df_interactive.copy()
        else: st.info("Le tableau d'exploration sera disponible après l'analyse si des traits numériques sont identifiés.")

    with col_interactive_graph:
        st.markdown("##### Graphique d'exploration")
        x_axis_plot = st.session_state.x_axis_trait_interactive
        y_axis_plot = st.session_state.y_axis_trait_interactive
        numeric_traits_plot = st.session_state.get('numeric_trait_names_for_interactive_plot', [])
        sub_plot = st.session_state.get('sub', pd.DataFrame())
        pdf_plot = st.session_state.get('pdf', pd.DataFrame())

        if not numeric_traits_plot: st.warning("Aucun trait numérique trouvé pour l'exploration interactive.")
        elif not x_axis_plot or not y_axis_plot: st.info("Veuillez sélectionner une variable pour l'Axe X et une pour l'Axe Y dans le tableau à gauche.")
        elif x_axis_plot not in numeric_traits_plot or y_axis_plot not in numeric_traits_plot: st.warning("Une ou les deux variables sélectionnées ne sont plus valides. Veuillez re-sélectionner.")
        elif sub_plot.empty or pdf_plot.empty or len(sub_plot) != len(pdf_plot) or x_axis_plot not in sub_plot.columns or y_axis_plot not in sub_plot.columns: st.warning("Données pour le graphique interactif non prêtes, incohérentes ou variables sélectionnées non trouvées. Vérifiez l'analyse principale.")
        else:
            required_pdf_cols = ['Espece_User', 'Ecologie', 'Cluster']
            if not all(col in pdf_plot.columns for col in required_pdf_cols): st.warning("Colonnes requises (Espece_User, Ecologie, Cluster) manquent dans les données PCA.")
            else:
                plot_data_interactive = pd.DataFrame({'Espece_User': pdf_plot['Espece_User'],'Ecologie': pdf_plot['Ecologie'],x_axis_plot: sub_plot[x_axis_plot].copy(),y_axis_plot: sub_plot[y_axis_plot].copy(),'Cluster': pdf_plot['Cluster']})
                plot_data_to_use = plot_data_interactive.copy()
                temp_x_col_grp = "_temp_x"; temp_y_col_grp = "_temp_y"
                plot_data_to_use[temp_x_col_grp] = plot_data_to_use[x_axis_plot]; plot_data_to_use[temp_y_col_grp] = plot_data_to_use[y_axis_plot]
                duplicates_mask = plot_data_to_use.duplicated(subset=[temp_x_col_grp, temp_y_col_grp], keep=False)
                if duplicates_mask.any():
                    x_min_val, x_max_val = plot_data_to_use[x_axis_plot].min(), plot_data_to_use[x_axis_plot].max()
                    y_min_val, y_max_val = plot_data_to_use[y_axis_plot].min(), plot_data_to_use[y_axis_plot].max()
                    x_range_val = (x_max_val - x_min_val) if pd.notna(x_max_val) and pd.notna(x_min_val) else 0
                    y_range_val = (y_max_val - y_min_val) if pd.notna(y_max_val) and pd.notna(y_min_val) else 0
                    jitter_x = x_range_val*0.015 if x_range_val >1e-9 else (abs(plot_data_to_use[x_axis_plot].mean())*0.015 if abs(plot_data_to_use[x_axis_plot].mean()) >1e-9 else 0.015)
                    jitter_y = y_range_val*0.015 if y_range_val >1e-9 else (abs(plot_data_to_use[y_axis_plot].mean())*0.015 if abs(plot_data_to_use[y_axis_plot].mean()) >1e-9 else 0.015)
                    if abs(jitter_x) <1e-9: jitter_x=0.015
                    if abs(jitter_y) <1e-9: jitter_y=0.015
                    
                    for _, group in plot_data_to_use[duplicates_mask].groupby([temp_x_col_grp, temp_y_col_grp]):
                        if len(group) > 1:
                            if not pd.api.types.is_float_dtype(plot_data_to_use[x_axis_plot]): plot_data_to_use[x_axis_plot] = plot_data_to_use[x_axis_plot].astype(float)
                            if not pd.api.types.is_float_dtype(plot_data_to_use[y_axis_plot]): plot_data_to_use[y_axis_plot] = plot_data_to_use[y_axis_plot].astype(float)
                            for i, idx in enumerate(group.index):
                                angle = 2 * np.pi * i / len(group)
                                plot_data_to_use.loc[idx, x_axis_plot] += jitter_x * np.cos(angle)
                                plot_data_to_use.loc[idx, y_axis_plot] += jitter_y * np.sin(angle)
                plot_data_to_use.drop(columns=[temp_x_col_grp, temp_y_col_grp], inplace=True) 

                fig_interactive_scatter = px.scatter(plot_data_to_use, x=x_axis_plot, y=y_axis_plot,color="Cluster", text="Espece_User", hover_name="Espece_User",custom_data=["Espece_User", "Ecologie"], template="plotly_dark", height=600, color_discrete_sequence=COLOR_SEQUENCE)
                fig_interactive_scatter.update_traces(textposition="top center", marker=dict(opacity=0.8, size=8),textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS),hovertemplate=(f"<span style='font-size: {HOVER_SPECIES_FONT_SIZE}px;'><b>%{{customdata[0]}}</b></span><br>"f"<br><span style='font-size: {HOVER_ECOLOGY_TITLE_FONT_SIZE}px;'><i>Écologie:</i></span><br>"f"<span style='font-size: {HOVER_ECOLOGY_TEXT_FONT_SIZE}px;'>%{{customdata[1]}}</span>" "<extra></extra>" ))
                unique_clusters_interactive = sorted(plot_data_to_use["Cluster"].unique())
                cluster_color_map_interactive = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_interactive)}
                for cluster_label in unique_clusters_interactive:
                    cluster_points_df_interactive = plot_data_to_use[plot_data_to_use["Cluster"] == cluster_label]
                    if x_axis_plot in cluster_points_df_interactive and y_axis_plot in cluster_points_df_interactive:
                        points_for_hull = cluster_points_df_interactive[[x_axis_plot, y_axis_plot]].drop_duplicates().values
                        if len(points_for_hull) >= MIN_POINTS_FOR_HULL:
                            try:
                                hull_interactive = ConvexHull(points_for_hull) 
                                hull_path_interactive = points_for_hull[np.append(hull_interactive.vertices, hull_interactive.vertices[0])]
                                clr_int = cluster_color_map_interactive.get(cluster_label, COLOR_SEQUENCE[0])
                                fig_interactive_scatter.add_trace(go.Scatter(x=hull_path_interactive[:, 0], y=hull_path_interactive[:, 1], fill="toself", fillcolor=clr_int,line=dict(color=clr_int, width=1.5), mode='lines', name=f'Cluster {cluster_label} Hull', opacity=0.2, showlegend=False, hoverinfo='skip' ))
                            except Exception as e: print(f"Erreur calcul Hull interactif {cluster_label} ({x_axis_plot}, {y_axis_plot}): {e}")
                fig_interactive_scatter.update_layout(title_text=f"{y_axis_plot} vs. {x_axis_plot}", title_x=0.5,xaxis_title=x_axis_plot, yaxis_title=y_axis_plot, dragmode='pan')
                st.plotly_chart(fig_interactive_scatter, use_container_width=True, config={'scrollZoom': True})
elif st.session_state.run_main_analysis_once and st.session_state.get('sub', pd.DataFrame()).empty :
    st.markdown("---")
    st.subheader("Étape 2: Exploration Interactive des Variables")
    st.warning("L'analyse principale n'a pas abouti à des données suffisantes pour cette section (aucune espèce trouvée ou traitée). Veuillez vérifier les étapes précédentes.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 3: PARAMÈTRES D'ANALYSE ET VISUALISATION PRINCIPALE (ACP)
# ---------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("Étape 3: Paramètres d'Analyse et Visualisation Principale (ACP)")
col_controls_area, col_pca_plot_area = st.columns([1, 2]) 

with col_controls_area:
    st.markdown("##### Paramètres")
    n_clusters_selected_val = st.slider(
        "Nombre de clusters (pour ACP)", 2, 8, 
        value=st.session_state.get('n_clusters_slider_main_value', 3), 
        key="n_clusters_slider_main_key", 
        disabled=ref.empty
    )
    if n_clusters_selected_val != st.session_state.get('n_clusters_slider_main_value', 3):
        st.session_state.n_clusters_slider_main_value = n_clusters_selected_val
        st.session_state.analysis_has_run_for_current_selection = False 
        st.rerun()

if st.session_state.run_main_analysis_once: 
    pdf_display_pca = st.session_state.get('pdf', pd.DataFrame())
    
    if not pdf_display_pca.empty and "PC1" in pdf_display_pca.columns and "Cluster" in pdf_display_pca.columns and \
       "Espece_User" in pdf_display_pca.columns and "Ecologie" in pdf_display_pca.columns:
        
        y_pca_col = None
        if "PC2" in pdf_display_pca.columns:
            y_pca_col = "PC2"
        elif len(pdf_display_pca.columns) > 1 and pdf_display_pca.columns[1].startswith("PC"):
             y_pca_col = pdf_display_pca.columns[1]


        if y_pca_col: 
            fig_pca = px.scatter(pdf_display_pca, x="PC1", y=y_pca_col, color="Cluster", text="Espece_User", 
                                 hover_name="Espece_User", custom_data=["Espece_User", "Ecologie"], 
                                 template="plotly_dark", height=500, color_discrete_sequence=COLOR_SEQUENCE)
            fig_pca.update_traces(textposition="top center", marker=dict(opacity=0.7), 
                                   hovertemplate=(f"<span style='font-size: {HOVER_SPECIES_FONT_SIZE}px;'><b>%{{customdata[0]}}</b></span><br><br>"
                                                  f"<span style='font-size: {HOVER_ECOLOGY_TITLE_FONT_SIZE}px;'><i>Écologie:</i></span><br>"
                                                  f"<span style='font-size: {HOVER_ECOLOGY_TEXT_FONT_SIZE}px;'>%{{customdata[1]}}</span>"
                                                  "<extra></extra>"), 
                                   textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS)) 
            unique_clusters_pca = sorted(pdf_display_pca["Cluster"].unique())
            cluster_color_map_pca = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_pca)}
            
            for cluster_label in unique_clusters_pca:
                cluster_points_df_pca = pdf_display_pca[pdf_display_pca["Cluster"] == cluster_label]
                if not cluster_points_df_pca.empty and "PC1" in cluster_points_df_pca.columns and y_pca_col in cluster_points_df_pca.columns:
                    unique_cluster_points_pca = cluster_points_df_pca[["PC1", y_pca_col]].drop_duplicates().values
                    if len(unique_cluster_points_pca) >= MIN_POINTS_FOR_HULL:
                        try:
                            hull_pca = ConvexHull(unique_cluster_points_pca)
                            hull_path = unique_cluster_points_pca[np.append(hull_pca.vertices, hull_pca.vertices[0])] 
                            clr = cluster_color_map_pca.get(cluster_label, COLOR_SEQUENCE[0])
                            fig_pca.add_trace(go.Scatter(x=hull_path[:, 0], y=hull_path[:, 1], fill="toself", fillcolor=clr, 
                                                         line=dict(color=clr, width=1.5), mode='lines', 
                                                         name=f'Cluster {cluster_label} Hull', opacity=0.2, 
                                                         showlegend=False, hoverinfo='skip'))
                        except Exception as e: print(f"Erreur calcul Hull ACP pour cluster {cluster_label}: {e}")
            fig_pca.update_layout(title_text="Plot PCA des espèces", title_x=0.5, legend_title_text='Cluster', dragmode='pan')
        else:
            fig_pca = None 
            if not pdf_display_pca.empty : 
                 with col_pca_plot_area: st.warning("Moins de deux composantes principales disponibles pour le graphique PCA.")

    X_for_dendro_display = st.session_state.get('X_for_dendro', np.array([]))
    sub_display_dendro = st.session_state.get('sub', pd.DataFrame())
    pdf_display_dendro_labels = st.session_state.get('pdf', pd.DataFrame())

    if isinstance(X_for_dendro_display, np.ndarray) and X_for_dendro_display.ndim == 2 and \
       X_for_dendro_display.shape[0] > 1 and X_for_dendro_display.shape[1] > 0:
        try:
            Z = linkage(X_for_dendro_display, method="ward")
            dyn_thresh = 0
            n_clust_for_dendro_color = st.session_state.get('n_clusters_slider_main_value', 3) 

            if n_clust_for_dendro_color > 1 and (n_clust_for_dendro_color -1) < Z.shape[0] : 
                dyn_thresh = Z[-(n_clust_for_dendro_color-1), 2] * 0.99 if Z.shape[0] >= (n_clust_for_dendro_color-1) else (Z[-1,2] * 0.5 if Z.shape[0]>0 else 0)
            elif Z.shape[0] > 0: dyn_thresh = Z[0, 2] / 2
            
            dendro_labels = pdf_display_dendro_labels["Espece_User"].tolist() if not pdf_display_dendro_labels.empty and "Espece_User" in pdf_display_dendro_labels.columns and len(pdf_display_dendro_labels) == X_for_dendro_display.shape[0] else [f"Esp {i+1}" for i in range(X_for_dendro_display.shape[0])]
            fig_dend = ff.create_dendrogram(X_for_dendro_display, orientation="left", labels=dendro_labels, 
                                            linkagefun=lambda _: Z, 
                                            color_threshold=dyn_thresh if n_clust_for_dendro_color > 1 else 0, 
                                            colorscale=COLOR_SEQUENCE)
            fig_dend.update_layout(template="plotly_dark", 
                                   height=max(400, sub_display_dendro.shape[0] * 20 if not sub_display_dendro.empty else 400), 
                                   title_text="Dendrogramme des espèces", title_x=0.5)
        except Exception as e: 
            print(f"Erreur lors de la création du dendrogramme: {e}")
            fig_dend = None
    else: fig_dend = None


with col_pca_plot_area: 
    if fig_pca: st.plotly_chart(fig_pca, use_container_width=True, config={'scrollZoom': True}) 
    elif st.session_state.run_main_analysis_once and st.session_state.get('sub', pd.DataFrame()).empty:
        st.warning("L'analyse n'a pas produit de résultats affichables pour le PCA (pas d'espèces traitées ou PCA impossible).")
    elif st.session_state.run_main_analysis_once and fig_pca is None and not st.session_state.get('pdf', pd.DataFrame()).empty : 
        # Ce cas est pour quand pdf existe mais fig_pca n'a pas pu être créé (ex: <2 PCs)
        # Le message est déjà affiché dans le bloc de création de fig_pca
        pass
    elif st.session_state.run_main_analysis_once :
         st.warning("Le graphique PCA n'a pas pu être généré. Vérifiez les données d'entrée et les paramètres.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 4: COMPOSITION DES CLUSTERS (ACP)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty: 
    st.markdown("---"); st.subheader("Étape 4: Composition des Clusters (ACP)")
    pdf_compo = st.session_state.get('pdf', pd.DataFrame())
    if not pdf_compo.empty and 'Cluster' in pdf_compo.columns and 'Espece_User' in pdf_compo.columns:
        compositions_display = [{"cluster_label": c, "count": len(pdf_compo.loc[pdf_compo["Cluster"] == c, "Espece_User"].unique()), "species_list": sorted(list(pdf_compo.loc[pdf_compo["Cluster"] == c, "Espece_User"].unique()))} for c in sorted(pdf_compo["Cluster"].unique())]
        if compositions_display and any(d['count'] > 0 for d in compositions_display):
            num_clusters_disp = len([d for d in compositions_display if d['count']>0]) 
            num_cols_disp = min(num_clusters_disp, 3) if num_clusters_disp > 0 else 1
            cluster_cols_layout = st.columns(num_cols_disp)
            col_idx = 0
            for comp_data in compositions_display:
                if comp_data['count'] > 0: 
                    with cluster_cols_layout[col_idx % num_cols_disp]:
                        st.markdown(f"**Cluster {comp_data['cluster_label']}** ({comp_data['count']} espèces)")
                        for species_name in comp_data['species_list']: st.markdown(f"- {species_name}")
                    col_idx += 1
            if col_idx == 0 : st.info("Aucun cluster (ACP) avec des espèces à afficher.")
        else: st.info("La composition des clusters (ACP) sera affichée ici après l'analyse (pas de données de cluster).")
    else: st.info("La composition des clusters (ACP) sera affichée ici après l'analyse (données de PCA non disponibles ou incomplètes).")
elif st.session_state.run_main_analysis_once: 
    st.markdown("---"); st.subheader("Étape 4: Composition des Clusters (ACP)")
    st.info("Analyse lancée, mais aucune donnée d'espèce n'a pu être traitée pour la composition des clusters.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 5: AFFICHAGE DU DENDROGRAMME
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty : 
    st.markdown("---"); st.subheader("Étape 5: Dendrogramme") 
    if fig_dend: st.plotly_chart(fig_dend, use_container_width=True)
    elif st.session_state.get('X_for_dendro', np.array([])).shape[0] <= 1 and \
         len(st.session_state.get('pdf', pd.DataFrame())) > 0 : 
        st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces uniques après traitement ou problème de données pour le linkage).")
    elif st.session_state.run_main_analysis_once : 
        st.info("Le dendrogramme n'a pas pu être généré. Vérifiez les données d'entrée (nombre d'espèces > 1, traits numériques).")
elif st.session_state.run_main_analysis_once: 
    st.markdown("---"); st.subheader("Étape 5: Dendrogramme")
    st.info("Analyse lancée, mais aucune donnée d'espèce n'a pu être traitée pour le dendrogramme.")

