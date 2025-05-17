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
        def __init__(self, n_features_actual, n_components_to_simulate):
            if n_features_actual > 0 and n_components_to_simulate > 0:
                sim_components_t = np.random.rand(n_features_actual, n_components_to_simulate) 
                if sim_components_t.size > 0: 
                    sim_components_t = sim_components_t / np.linalg.norm(sim_components_t, axis=0, keepdims=True)
                self.components_ = sim_components_t.T 
                
                sim_explained_variance = np.sort(np.random.rand(n_components_to_simulate))[::-1]
                if sim_explained_variance.sum() > 0:
                    sim_explained_variance = (sim_explained_variance / sim_explained_variance.sum()) * n_components_to_simulate * 0.7 
                else: 
                    sim_explained_variance = np.zeros(n_components_to_simulate)
                self.explained_variance_ = sim_explained_variance
            else:
                self.components_ = np.array([])
                self.explained_variance_ = np.array([])

    def mock_analyse(sub_df_prepared_for_core, n_clusters): # Renamed to reflect its structure
        # sub_df_prepared_for_core is expected to have 'Espece' as first col, rest numeric for PCA
        n_samples = len(sub_df_prepared_for_core)
        
        # Numeric columns for PCA are all columns after 'Espece'
        if n_samples == 0 or sub_df_prepared_for_core.shape[1] <= 1: # No data or only 'Espece' column
            mock_pca_obj = MockPCA(n_features_actual=0, n_components_to_simulate=0)
            # Return empty but conforming structures
            return np.array([]), mock_pca_obj, pd.DataFrame(index=sub_df_prepared_for_core.index), np.array([]).reshape(0,1)

        numeric_cols_for_pca_df = sub_df_prepared_for_core.iloc[:, 1:]
        n_features = numeric_cols_for_pca_df.shape[1]

        if n_features == 0: # Only 'Espece' column was present, but no numeric traits after it
            mock_pca_obj = MockPCA(n_features_actual=0, n_components_to_simulate=0)
            return np.array([]), mock_pca_obj, pd.DataFrame(index=sub_df_prepared_for_core.index), np.array([]).reshape(0,1)

        n_pcs_to_simulate_coords = min(2, n_features)
        coords_array = np.random.rand(n_samples, n_pcs_to_simulate_coords) * 10
        pc_cols = [f"PC{i+1}" for i in range(coords_array.shape[1])]
        # Index of coords_df should match the input sub_df_prepared_for_core for proper alignment later
        coords_df = pd.DataFrame(coords_array, columns=pc_cols, index=sub_df_prepared_for_core.index)

        labels = np.array([])
        if n_samples > 0 and n_clusters > 0:
            if n_samples < n_clusters : 
                labels = np.arange(n_samples)
            else:
                labels = np.random.randint(0, n_clusters, n_samples)
        
        X_scaled = np.array([]).reshape(n_samples, 0)
        if not numeric_cols_for_pca_df.empty:
            X_scaled_temp = (numeric_cols_for_pca_df - numeric_cols_for_pca_df.mean()) / numeric_cols_for_pca_df.std()
            X_scaled = X_scaled_temp.fillna(0).values
        elif n_samples > 0 :
             X_scaled = np.random.rand(n_samples, 1) 

        mock_pca_obj = MockPCA(n_features_actual=n_features, n_components_to_simulate=n_pcs_to_simulate_coords)
        
        if mock_pca_obj.components_.shape[0] < coords_df.shape[1]:
            coords_df = coords_df.iloc[:, :mock_pca_obj.components_.shape[0]]
            new_pc_cols = [f"PC{i+1}" for i in range(mock_pca_obj.components_.shape[0])]
            coords_df.columns = new_pc_cols
        
        if X_scaled.ndim == 1 and n_samples > 0 :
            X_scaled = X_scaled.reshape(-1,1)
        elif X_scaled.size == 0 and n_samples == 0: # Corrected condition for empty X_scaled
             X_scaled = np.array([]).reshape(0,n_features if n_features > 0 else 1)


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
    color: #e1e1e1 !important;          /* Couleur de texte pour thème sombre */
    font-weight: bold !important;
}
/* Style pour la cellule de la première ligne en mode édition */
div[data-testid="stDataEditor"] .glideDataEditor-body .dvn-scroll-inner > div:first-child > div[data-cell^="[0,"] > div > .gdg-input {
    background-color: #ffffff !important; /* Fond blanc pour l'éditeur */
    color: #000000 !important;          /* Texte noir pour l'éditeur */
    font-weight: normal !important;      /* Poids normal pour l'éditeur */
}

.habitat-select-button button {
    font-size: 13px !important;
    padding: 0.25rem 0.6rem !important; 
    line-height: 1.4;
    width: 100%; 
    border-radius: 0.5rem; 
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
        if not hasattr(core, "read_reference") or callable(getattr(core, "read_reference", None)) and core.read_reference.__name__ == '<lambda>': 
            st.warning(f"Simulation du chargement de '{file_path}'. Le fichier réel n'est pas utilisé.")
            example_species = [f"Espece Alpha {i}" for i in range(1, 11)] + \
                                [f"Espece Beta {i}" for i in range(1, 11)] + \
                                [f"Espece Gamma {i}" for i in range(1, 11)]
            data = pd.DataFrame({
                'Espece': example_species, # Assumed to be the identifier by core.py
                'Trait_Num_1': np.random.rand(30) * 10,
                'Trait_Num_2': np.random.randint(1, 100, 30),
                'Trait_Cat_1': np.random.choice(['X', 'Y', 'Z'], 30), # Categorical trait
                'Humidité_édaphique': np.random.rand(30) * 5 + 1, 
                'Matière_organique': np.random.rand(30) * 10,
                'Lumière': np.random.rand(30) * 1000
            })
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
            keep_default_na=False, 
            na_values=[''] 
        )
        eco_data = eco_data.dropna(subset=['Espece']) 
        eco_data['Espece'] = eco_data['Espece'].astype(str).str.strip()
        eco_data = eco_data[eco_data['Espece'] != ""] 

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
        eco_data = eco_data.set_index('Espece_norm') 
        return eco_data[["Description_Ecologie"]] 
    
    except FileNotFoundError:
        print(f"AVERTISSEMENT: Fichier de données écologiques '{file_path}' non trouvé.")
        st.toast(f"Fichier écologique '{file_path}' non trouvé.", icon="⚠️")
        return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))
    except pd.errors.EmptyDataError:
        st.warning(f"Le fichier écologique '{file_path}' est vide.")
        return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))
    except ValueError as ve: 
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
    'selected_habitats_indices': [], 
    'previous_num_cols': 0,
    'analysis_has_run_for_current_selection': False,
    'n_clusters_slider_main_value': 3 
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

st.info("Copiez-collez vos données de relevés ci-dessus (Ctrl+V ou Cmd+V). La première ligne doit contenir les noms des habitats/relevés. Les lignes suivantes contiennent les espèces.")

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
        current_max_col_index = len(st.session_state.releves_df.columns) - 1
        st.session_state.selected_habitats_indices = [
            idx for idx in st.session_state.selected_habitats_indices if idx <= current_max_col_index
        ]
        if not st.session_state.selected_habitats_indices: 
             st.session_state.analysis_has_run_for_current_selection = False
             st.session_state.run_main_analysis_once = False 
        st.session_state.previous_num_cols = len(st.session_state.releves_df.columns)
    st.rerun()

current_releves_df_for_selection = st.session_state.releves_df.copy() 

if not current_releves_df_for_selection.empty and \
    len(current_releves_df_for_selection.columns) > 0 and \
    len(current_releves_df_for_selection) > 0:
    
    habitat_names_from_df = current_releves_df_for_selection.iloc[0].astype(str).tolist()
    num_actual_cols = len(current_releves_df_for_selection.columns)
    
    st.markdown("**Cliquez sur le nom d'un habitat ci-dessous pour le sélectionner/désélectionner pour l'analyse :**") 
    
    st.session_state.selected_habitats_indices = [
        idx for idx in st.session_state.selected_habitats_indices if idx < num_actual_cols
    ]

    if num_actual_cols > 0:
        button_cols_layout = st.columns(num_actual_cols) 
        
        for i in range(num_actual_cols):
            habitat_name_for_button = habitat_names_from_df[i] if pd.notna(habitat_names_from_df[i]) and str(habitat_names_from_df[i]).strip() != "" else f"Relevé {i+1}"
            is_selected = (i in st.session_state.selected_habitats_indices) 
            
            button_type = "primary" if is_selected else "secondary"
            button_key = f"habitat_select_button_{i}" 

            with button_cols_layout[i]:
                st.markdown(f'<div class="habitat-select-button">', unsafe_allow_html=True)
                if st.button(habitat_name_for_button, key=button_key, type=button_type, use_container_width=True):
                    if is_selected:
                        st.session_state.selected_habitats_indices.remove(i)
                    else:
                        st.session_state.selected_habitats_indices.append(i)
                    st.session_state.run_main_analysis_once = False 
                    st.session_state.analysis_has_run_for_current_selection = False 
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Ajoutez des colonnes au tableau pour pouvoir sélectionner des relevés.")
else:
    st.warning("Le tableau de données est vide ou ne contient pas de colonnes pour la sélection.")


fig_pca = None
fig_dend = None

if st.session_state.selected_habitats_indices and \
    not ref.empty and \
    not st.session_state.get('analysis_has_run_for_current_selection', False):

    st.session_state.run_main_analysis_once = True 
    st.session_state.analysis_has_run_for_current_selection = True 

    st.session_state.sub = pd.DataFrame()
    st.session_state.pdf = pd.DataFrame()
    st.session_state.X_for_dendro = np.array([])
    st.session_state.vip_data_df_interactive = pd.DataFrame()
    st.session_state.numeric_trait_names_for_interactive_plot = []

    all_species_data_for_analysis = [] 
    species_not_found_in_ref_detailed = {} 

    df_for_species_extraction = st.session_state.releves_df.copy() 
    habitat_names_from_header = df_for_species_extraction.iloc[0].astype(str).tolist() if not df_for_species_extraction.empty else []

    for habitat_idx in st.session_state.selected_habitats_indices:
        if habitat_idx < len(df_for_species_extraction.columns):
            habitat_name = habitat_names_from_header[habitat_idx] if habitat_idx < len(habitat_names_from_header) and pd.notna(habitat_names_from_header[habitat_idx]) and str(habitat_names_from_header[habitat_idx]).strip() != "" else f"Relevé {habitat_idx+1}"
            species_in_col_series = df_for_species_extraction.iloc[1:, habitat_idx]
            species_raw_in_current_habitat = species_in_col_series.dropna().astype(str).str.strip().replace('', np.nan).dropna().tolist()
            
            species_not_found_in_ref_detailed[habitat_name] = []

            if not species_raw_in_current_habitat:
                st.warning(f"Aucune espèce listée dans l'habitat sélectionné : {habitat_name}")
                continue

            for raw_species_name in species_raw_in_current_habitat:
                if not raw_species_name or len(raw_species_name.split()) == 0: 
                    continue
                
                binom_species_name = " ".join(raw_species_name.split()[:2]).lower()
                
                if not ref_binom_series.empty:
                    match_in_ref = ref_binom_series[ref_binom_series == binom_species_name]
                    if not match_in_ref.empty:
                        ref_idx = match_in_ref.index[0] 
                        trait_data = ref.loc[ref_idx].to_dict() # This includes 'Espece' and all traits from ref
                        trait_data['Source_Habitat'] = habitat_name
                        trait_data['Espece_Ref_Original'] = ref.loc[ref_idx, 'Espece'] # Explicitly store original ref species name
                        trait_data['Espece_User_Input_Raw'] = raw_species_name
                        all_species_data_for_analysis.append(trait_data)
                    else:
                        species_not_found_in_ref_detailed[habitat_name].append(raw_species_name)
                else:
                     species_not_found_in_ref_detailed[habitat_name].append(raw_species_name)

    if not all_species_data_for_analysis:
        st.error("Aucune espèce valide correspondante aux traits n'a été trouvée dans les relevés sélectionnés. Vérifiez vos données et sélections.")
        st.session_state.run_main_analysis_once = False 
        st.session_state.analysis_has_run_for_current_selection = False 
    else:
        # st.session_state.sub will contain ALL columns: original traits from ref (numeric and categorical),
        # plus 'Source_Habitat', 'Espece_Ref_Original', 'Espece_User_Input_Raw'.
        # The original 'Espece' column from ref is also present if it was in ref.loc[ref_idx].to_dict().
        st.session_state.sub = pd.DataFrame(all_species_data_for_analysis)
        # Ensure the index is reset for st.session_state.sub so it's a simple range index,
        # which will be inherited by coords_df and allow direct merging/mapping.
        st.session_state.sub.reset_index(drop=True, inplace=True)

        for habitat_name, not_found_list in species_not_found_in_ref_detailed.items():
            if not_found_list:
                st.warning(f"Espèces de '{habitat_name}' non trouvées dans la base de traits : " + ", ".join(not_found_list), icon="⚠️")

        n_clusters_selected_main = st.session_state.get('n_clusters_slider_main_value', 3) 

        if st.session_state.sub.empty: 
            st.error("Aucune des espèces sélectionnées n'a été trouvée dans la base de traits. L'analyse ne peut continuer.")
            st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;
        elif st.session_state.sub.shape[0] < n_clusters_selected_main and n_clusters_selected_main > 0 :
            st.error(f"Nombre total d'instances d'espèces trouvées ({st.session_state.sub.shape[0]}) < clusters demandés ({n_clusters_selected_main}). Ajustez le nombre de clusters ou vérifiez les espèces.");
            st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;
        elif st.session_state.sub.shape[0] < 2: 
            st.error(f"Au moins 2 instances d'espèces (total sur les habitats) sont nécessaires pour l'analyse. {st.session_state.sub.shape[0]} trouvée(s).");
            st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;
        else:
            try:
                # **CORRECTION APPLIQUÉE ICI**
                # Préparer la DataFrame spécifiquement pour core.analyse
                # Elle doit avoir 'Espece' comme première colonne, suivie UNIQUEMENT des traits numériques.

                # 1. Identifier les noms des traits numériques à partir du DataFrame de référence 'ref'
                if ref.empty:
                    st.error("Le DataFrame de référence 'ref' est vide. Impossible de déterminer les traits numériques.")
                    st.session_state.run_main_analysis_once = False
                    st.session_state.analysis_has_run_for_current_selection = False
                    raise ValueError("DataFrame 'ref' vide, impossible de préparer les données pour core.analyse.")

                numeric_trait_names_from_ref = ref.select_dtypes(include=np.number).columns.tolist()

                # 2. Créer une copie de st.session_state.sub pour la préparation
                df_for_core_preparation = st.session_state.sub.copy()

                # 3. S'assurer qu'une colonne 'Espece' (identifiant) existe. Utiliser 'Espece_Ref_Original'.
                if 'Espece_Ref_Original' in df_for_core_preparation.columns:
                    df_for_core_preparation['Espece_ID_Core'] = df_for_core_preparation['Espece_Ref_Original']
                elif 'Espece' in df_for_core_preparation.columns: # Fallback si Espece_Ref_Original n'est pas là
                    df_for_core_preparation['Espece_ID_Core'] = df_for_core_preparation['Espece']
                else:
                    st.error("Colonne d'identification 'Espece' ou 'Espece_Ref_Original' manquante pour l'analyse.")
                    raise ValueError("Identifiant 'Espece' manquant pour core.analyse.")

                # 4. Définir la liste des traits numériques réels à utiliser pour l'ACP.
                # Ce sont les traits numériques de 'ref' qui sont aussi présents dans les données collectées.
                actual_numeric_traits_for_pca = [
                    trait for trait in numeric_trait_names_from_ref if trait in df_for_core_preparation.columns
                ]
                
                # 5. Construire la DataFrame finale pour core.analyse : 'Espece_ID_Core' puis les traits numériques.
                # L'ordre est important si core.py utilise iloc.
                columns_for_core_call = ['Espece_ID_Core'] + actual_numeric_traits_for_pca
                sub_for_analysis_call_prepared = df_for_core_preparation[columns_for_core_call]
                # Renommer 'Espece_ID_Core' en 'Espece' si core.py s'attend à ce nom exact.
                sub_for_analysis_call_prepared = sub_for_analysis_call_prepared.rename(columns={'Espece_ID_Core': 'Espece'})


                # 6. Vérifier s'il y a des traits numériques pour l'ACP (après la colonne 'Espece')
                if not actual_numeric_traits_for_pca: # La liste est vide
                    st.error("Aucun trait numérique disponible pour l'ACP après filtrage. L'analyse est impossible.")
                    st.session_state.run_main_analysis_once = False
                    st.session_state.analysis_has_run_for_current_selection = False
                    raise ValueError("Aucun trait numérique pour l'ACP.")
                else:
                    # L'analyse peut continuer avec sub_for_analysis_call_prepared
                    labels, pca_results, coords_df, X_scaled_data = core.analyse(sub_for_analysis_call_prepared, n_clusters_selected_main)
                    
                    # Vérifier que coords_df a un index qui correspond à sub_for_analysis_call_prepared
                    # et donc à st.session_state.sub (car ils partagent le même RangeIndex après reset_index)
                    if not coords_df.index.equals(sub_for_analysis_call_prepared.index):
                        st.error("L'index des coordonnées PCA ne correspond pas aux données d'entrée. L'alignement des données a échoué.")
                        # Potentiellement recréer coords_df avec le bon index si c'est juste un problème d'objet index
                        if len(coords_df) == len(sub_for_analysis_call_prepared):
                             coords_df.index = sub_for_analysis_call_prepared.index
                             st.warning("Index des coordonnées PCA réaligné.")
                        else:
                            st.session_state.run_main_analysis_once = False
                            st.session_state.analysis_has_run_for_current_selection = False
                            raise ValueError("Incohérence d'index PCA.")


                    if st.session_state.run_main_analysis_once: 
                        current_pdf = coords_df.copy() # Index de coords_df est crucial ici
                        if not current_pdf.empty:
                            if len(labels) == len(current_pdf): current_pdf["Cluster"] = labels.astype(str)
                            else: current_pdf["Cluster"] = np.zeros(len(current_pdf)).astype(str) if len(current_pdf) > 0 else pd.Series(dtype=str)
                            
                            # Fusionner les informations supplémentaires de st.session_state.sub en utilisant l'index partagé
                            # st.session_state.sub et current_pdf (via coords_df et sub_for_analysis_call_prepared) doivent avoir le même index.
                            current_pdf["Source_Habitat"] = st.session_state.sub.loc[current_pdf.index, "Source_Habitat"]
                            current_pdf["Espece_Ref"] = st.session_state.sub.loc[current_pdf.index, "Espece_Ref_Original"]
                            current_pdf["Espece_User"] = st.session_state.sub.loc[current_pdf.index, "Espece_User_Input_Raw"]

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
                                
                                pca_components_values = pca_results.components_ 
                                explained_variance_values = pca_results.explained_variance_ 
                                eigenvectors_matrix = pca_components_values.T  
                                sqrt_eigenvalues_vector = np.sqrt(explained_variance_values) 
                                loadings = eigenvectors_matrix * sqrt_eigenvalues_vector 
                                communal = (loadings**2).sum(axis=1) 
                                
                                # Utiliser la liste des traits numériques réellement passés à l'ACP pour les communalités
                                trait_columns_for_communal = actual_numeric_traits_for_pca 
                                
                                if len(communal) == len(trait_columns_for_communal):
                                    communal_percent = (communal * 100).round(0).astype(int) 
                                    communal_percent_clipped = np.clip(communal_percent, 0, 100)
                                    
                                    st.session_state.vip_data_df_for_calc = pd.DataFrame({
                                        "Variable": trait_columns_for_communal,
                                        "Communalité (%)": communal_percent_clipped, 
                                    }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
                                else: 
                                    st.session_state.vip_data_df_for_calc = pd.DataFrame(columns=["Variable", "Communalité (%)"])
                                    st.warning(f"Communalités non calculées (dimensions des loadings/traits incohérentes: {len(communal)} vs {len(trait_columns_for_communal)}).")
                            else:
                                st.session_state.vip_data_df_for_calc = pd.DataFrame(columns=["Variable", "Communalité (%)"])
                                st.warning("Résultats PCA incomplets pour communalités (components_ ou explained_variance_ manquants/incorrects).")
                            
                            st.session_state.X_for_dendro = X_scaled_data if isinstance(X_scaled_data, np.ndarray) else np.array([])
                            # Les traits numériques pour le graphique interactif sont ceux utilisés dans l'ACP
                            st.session_state.numeric_trait_names_for_interactive_plot = actual_numeric_traits_for_pca
                            
                            default_x_init, default_y_init = None, None
                            if not st.session_state.vip_data_df_for_calc.empty and actual_numeric_traits_for_pca: # Utiliser actual_numeric_traits_for_pca
                                top_vars_from_vip_numeric = [var for var in st.session_state.vip_data_df_for_calc["Variable"].tolist() if var in actual_numeric_traits_for_pca]
                                if len(top_vars_from_vip_numeric) >= 1: default_x_init = top_vars_from_vip_numeric[0]
                                if len(top_vars_from_vip_numeric) >= 2: default_y_init = top_vars_from_vip_numeric[1]
                                elif len(top_vars_from_vip_numeric) == 1: 
                                    other_numeric_traits = [t for t in actual_numeric_traits_for_pca if t != default_x_init]
                                    default_y_init = other_numeric_traits[0] if other_numeric_traits else default_x_init
                            
                            if default_x_init is None and actual_numeric_traits_for_pca: default_x_init = actual_numeric_traits_for_pca[0]
                            if default_y_init is None:
                                if len(actual_numeric_traits_for_pca) >= 2: default_y_init = actual_numeric_traits_for_pca[1]
                                elif default_x_init and len(actual_numeric_traits_for_pca) == 1: default_y_init = default_x_init

                            st.session_state.x_axis_trait_interactive = default_x_init
                            st.session_state.y_axis_trait_interactive = default_y_init
                            
                            if not st.session_state.vip_data_df_for_calc.empty and actual_numeric_traits_for_pca:
                                temp_interactive_df = st.session_state.vip_data_df_for_calc[st.session_state.vip_data_df_for_calc["Variable"].isin(actual_numeric_traits_for_pca)].copy()
                                temp_interactive_df["Axe X"] = temp_interactive_df["Variable"] == st.session_state.x_axis_trait_interactive
                                temp_interactive_df["Axe Y"] = temp_interactive_df["Variable"] == st.session_state.y_axis_trait_interactive
                                st.session_state.vip_data_df_interactive = temp_interactive_df[["Variable", "Communalité (%)", "Axe X", "Axe Y"]].reset_index(drop=True)
                            else:
                                st.session_state.vip_data_df_interactive = pd.DataFrame(columns=["Variable", "Communalité (%)", "Axe X", "Axe Y"])
                            st.session_state.vip_data_df_interactive_snapshot_for_comparison = st.session_state.vip_data_df_interactive.copy()
                        else: 
                            st.warning("L'analyse n'a pas produit de coordonnées PCA (coords_df vide ou invalide).")
                            st.session_state.run_main_analysis_once = False 
                            st.session_state.analysis_has_run_for_current_selection = False
            except Exception as e:
                st.error(f"Erreur lors de l'analyse principale : {e}"); st.exception(e)
                st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;

elif not st.session_state.selected_habitats_indices and not ref.empty:
    st.info("Veuillez sélectionner un ou plusieurs habitats à l'Étape 1 pour lancer l'analyse.")
elif ref.empty:
    st.warning("Les données de référence ('data_ref.csv') n'ont pas pu être chargées ou sont simulées. L'analyse est désactivée si les données réelles manquent.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 2: EXPLORATION INTERACTIVE DES VARIABLES ET PARAM ACP
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty:
    st.markdown("---"); st.subheader("Étape 2: Exploration Interactive et Paramètres ACP")
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

            current_x_selection_from_state = st.session_state.x_axis_trait_interactive
            x_vars_checked_in_editor = edited_df_interactive[edited_df_interactive["Axe X"]]["Variable"].tolist()
            
            new_x_selection_candidate = current_x_selection_from_state 

            if not x_vars_checked_in_editor: 
                if current_x_selection_from_state is not None: 
                    new_x_selection_candidate = None
                    made_change_in_interactive_axes = True
            elif len(x_vars_checked_in_editor) == 1: 
                single_checked_x = x_vars_checked_in_editor[0]
                if single_checked_x != current_x_selection_from_state:
                    new_x_selection_candidate = single_checked_x
                    made_change_in_interactive_axes = True
            else: 
                potential_new_x_selections = [v for v in x_vars_checked_in_editor if v != current_x_selection_from_state]
                if potential_new_x_selections:
                    new_x_selection_candidate = potential_new_x_selections[0] 
                else: 
                    new_x_selection_candidate = x_vars_checked_in_editor[-1] 
                made_change_in_interactive_axes = True 

            st.session_state.x_axis_trait_interactive = new_x_selection_candidate


            current_y_selection_from_state = st.session_state.y_axis_trait_interactive
            y_vars_checked_in_editor = edited_df_interactive[edited_df_interactive["Axe Y"]]["Variable"].tolist()

            new_y_selection_candidate = current_y_selection_from_state 

            if not y_vars_checked_in_editor: 
                if current_y_selection_from_state is not None:
                    new_y_selection_candidate = None
                    made_change_in_interactive_axes = True
            elif len(y_vars_checked_in_editor) == 1: 
                single_checked_y = y_vars_checked_in_editor[0]
                if single_checked_y != current_y_selection_from_state:
                    new_y_selection_candidate = single_checked_y
                    made_change_in_interactive_axes = True
            else: 
                potential_new_y_selections = [v for v in y_vars_checked_in_editor if v != current_y_selection_from_state]
                if potential_new_y_selections:
                    new_y_selection_candidate = potential_new_y_selections[0]
                else:
                    new_y_selection_candidate = y_vars_checked_in_editor[-1]
                made_change_in_interactive_axes = True
            
            st.session_state.y_axis_trait_interactive = new_y_selection_candidate

            if made_change_in_interactive_axes:
                df_updated_for_editor = df_editor_source_interactive.copy() 
                df_updated_for_editor["Axe X"] = (df_updated_for_editor["Variable"] == st.session_state.x_axis_trait_interactive)
                df_updated_for_editor["Axe Y"] = (df_updated_for_editor["Variable"] == st.session_state.y_axis_trait_interactive)
                
                st.session_state.vip_data_df_interactive = df_updated_for_editor 
                st.session_state.vip_data_df_interactive_snapshot_for_comparison = df_updated_for_editor.copy() 
                st.rerun()
            elif not edited_df_interactive.equals(st.session_state.vip_data_df_interactive_snapshot_for_comparison):
                st.session_state.vip_data_df_interactive_snapshot_for_comparison = edited_df_interactive.copy()
        else: 
            st.info("Le tableau d'exploration sera disponible après l'analyse si des traits numériques sont identifiés.")

        st.markdown("---") 
        st.markdown("##### Paramètres ACP")
        n_clusters_selected_val = st.slider(
            "Nombre de clusters (pour ACP)", 2, 8, 
            value=st.session_state.get('n_clusters_slider_main_value', 3), 
            key="n_clusters_slider_main_key_moved", 
            disabled=ref.empty or st.session_state.get('sub', pd.DataFrame()).empty, 
            help="Choisissez le nombre de groupes à former lors de l'Analyse en Composantes Principales."
        )
        if n_clusters_selected_val != st.session_state.get('n_clusters_slider_main_value', 3):
            st.session_state.n_clusters_slider_main_value = n_clusters_selected_val
            st.session_state.analysis_has_run_for_current_selection = False 
            st.rerun()


    with col_interactive_graph:
        st.markdown("##### Graphique d'exploration")
        x_axis_plot = st.session_state.x_axis_trait_interactive
        y_axis_plot = st.session_state.y_axis_trait_interactive
        # numeric_traits_plot is now st.session_state.numeric_trait_names_for_interactive_plot, which is actual_numeric_traits_for_pca
        numeric_traits_plot = st.session_state.get('numeric_trait_names_for_interactive_plot', [])
        sub_plot = st.session_state.get('sub', pd.DataFrame()) 
        pdf_plot = st.session_state.get('pdf', pd.DataFrame()) 

        if not numeric_traits_plot: st.warning("Aucun trait numérique trouvé pour l'exploration interactive.")
        elif not x_axis_plot or not y_axis_plot: st.info("Veuillez sélectionner une variable pour l'Axe X et une pour l'Axe Y dans le tableau à gauche.")
        elif x_axis_plot not in numeric_traits_plot or y_axis_plot not in numeric_traits_plot: st.warning("Une ou les deux variables sélectionnées ne sont plus valides. Veuillez re-sélectionner.")
        # sub_plot is the full data with all traits (numeric and categorical) and Source_Habitat
        # pdf_plot has PCA coordinates and Source_Habitat
        # For interactive plot, x_axis_plot and y_axis_plot refer to original trait names from sub_plot
        elif sub_plot.empty or pdf_plot.empty or x_axis_plot not in sub_plot.columns or y_axis_plot not in sub_plot.columns: 
            st.warning("Données pour le graphique interactif non prêtes, incohérentes ou variables sélectionnées non trouvées. Vérifiez l'analyse principale.")
        elif not pdf_plot.index.equals(sub_plot.index): # Critical check for alignment
             st.warning("Désalignement des données entre les résultats PCA (pdf_plot) et les données de traits (sub_plot). Le graphique interactif peut être incorrect.")
        else:
            required_pdf_cols_interactive = ['Espece_User', 'Ecologie', 'Cluster', 'Source_Habitat'] 
            if not all(col in pdf_plot.columns for col in required_pdf_cols_interactive): st.warning("Colonnes requises (Espece_User, Ecologie, Cluster, Source_Habitat) manquent dans les données PCA pour le graphique interactif.")
            else:
                # Construct plot_data_interactive carefully, ensuring indices align
                # pdf_plot and sub_plot should share the same RangeIndex at this point
                plot_data_interactive = pd.DataFrame({
                    'Espece_User': pdf_plot['Espece_User'], # From pdf_plot
                    'Ecologie': pdf_plot['Ecologie'],       # From pdf_plot
                    x_axis_plot: sub_plot[x_axis_plot],     # From sub_plot, using its index
                    y_axis_plot: sub_plot[y_axis_plot],     # From sub_plot, using its index
                    'Cluster': pdf_plot['Cluster'],         # From pdf_plot
                    'Source_Habitat': pdf_plot['Source_Habitat'] # From pdf_plot
                }).set_index(pdf_plot.index) # Ensure index is preserved if any implicit reset happened
                
                plot_data_to_use = plot_data_interactive.copy()
                # Jittering logic (remains the same)
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

                color_by_interactive = "Source_Habitat" if len(st.session_state.selected_habitats_indices) > 1 else "Cluster"
                legend_title_interactive = "Habitat d'Origine" if len(st.session_state.selected_habitats_indices) > 1 else "Cluster PCA"

                fig_interactive_scatter = px.scatter(
                    plot_data_to_use, x=x_axis_plot, y=y_axis_plot,
                    color=color_by_interactive, 
                    text="Espece_User", hover_name="Espece_User",
                    custom_data=["Espece_User", "Ecologie", "Source_Habitat", "Cluster"], 
                    template="plotly_dark", height=600, color_discrete_sequence=COLOR_SEQUENCE
                )
                fig_interactive_scatter.update_traces(
                    textposition="top center", marker=dict(opacity=0.8, size=8),
                    textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS),
                    hovertemplate=(
                        f"<span style='font-size: {HOVER_SPECIES_FONT_SIZE}px;'><b>%{{customdata[0]}}</b></span><br>"
                        f"Habitat: %{{customdata[2]}}<br>"
                        f"Cluster PCA: %{{customdata[3]}}<br>"
                        f"<br><span style='font-size: {HOVER_ECOLOGY_TITLE_FONT_SIZE}px;'><i>Écologie:</i></span><br>"
                        f"<span style='font-size: {HOVER_ECOLOGY_TEXT_FONT_SIZE}px;'>%{{customdata[1]}}</span>"
                        "<extra></extra>" 
                    )
                )
                
                unique_groups_interactive = sorted(plot_data_to_use[color_by_interactive].unique())
                extended_color_sequence_interactive = COLOR_SEQUENCE * (len(unique_groups_interactive) // len(COLOR_SEQUENCE) + 1)
                group_color_map_interactive = {
                    lbl: extended_color_sequence_interactive[i % len(extended_color_sequence_interactive)] for i, lbl in enumerate(unique_groups_interactive)
                }

                for group_label in unique_groups_interactive:
                    group_points_df_interactive = plot_data_to_use[plot_data_to_use[color_by_interactive] == group_label]
                    if x_axis_plot in group_points_df_interactive and y_axis_plot in group_points_df_interactive:
                        points_for_hull = group_points_df_interactive[[x_axis_plot, y_axis_plot]].drop_duplicates().values
                        if len(points_for_hull) >= MIN_POINTS_FOR_HULL:
                            try:
                                hull_interactive = ConvexHull(points_for_hull) 
                                hull_path_interactive = points_for_hull[np.append(hull_interactive.vertices, hull_interactive.vertices[0])]
                                clr_int = group_color_map_interactive.get(group_label, COLOR_SEQUENCE[0])
                                fig_interactive_scatter.add_trace(go.Scatter(
                                    x=hull_path_interactive[:, 0], y=hull_path_interactive[:, 1], 
                                    fill="toself", fillcolor=clr_int,
                                    line=dict(color=clr_int, width=1.5), mode='lines', 
                                    name=f'{legend_title_interactive} {group_label} Hull', opacity=0.2, 
                                    showlegend=False, hoverinfo='skip' 
                                ))
                            except Exception as e: print(f"Erreur calcul Hull interactif {group_label} ({x_axis_plot}, {y_axis_plot}): {e}")
                
                fig_interactive_scatter.update_layout(
                    title_text=f"{y_axis_plot} vs. {x_axis_plot}", title_x=0.5,
                    xaxis_title=x_axis_plot, yaxis_title=y_axis_plot, dragmode='pan',
                    legend_title_text=legend_title_interactive 
                )
                st.plotly_chart(fig_interactive_scatter, use_container_width=True, config={'scrollZoom': True})

elif st.session_state.run_main_analysis_once and st.session_state.get('sub', pd.DataFrame()).empty :
    st.markdown("---")
    st.subheader("Étape 2: Exploration Interactive et Paramètres ACP")
    st.warning("L'analyse principale n'a pas abouti à des données suffisantes pour cette section (aucune espèce trouvée ou traitée). Veuillez vérifier les étapes précédentes.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 3: VISUALISATION PRINCIPALE (ACP)
# ---------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("Étape 3: Visualisation Principale (ACP)")
_, col_pca_plot_area = st.columns([0.01, 0.99]) 

if st.session_state.run_main_analysis_once: 
    pdf_display_pca = st.session_state.get('pdf', pd.DataFrame())
    
    if not pdf_display_pca.empty and "PC1" in pdf_display_pca.columns and "Cluster" in pdf_display_pca.columns and \
        "Espece_User" in pdf_display_pca.columns and "Ecologie" in pdf_display_pca.columns and "Source_Habitat" in pdf_display_pca.columns:
        
        y_pca_col = "PC2" if "PC2" in pdf_display_pca.columns else None
        
        if "PC1" in pdf_display_pca.columns and y_pca_col : 
            color_by_pca = "Source_Habitat" if len(st.session_state.selected_habitats_indices) > 1 else "Cluster"
            legend_title_pca = "Habitat d'Origine" if len(st.session_state.selected_habitats_indices) > 1 else "Cluster PCA"

            fig_pca = px.scatter(
                pdf_display_pca, x="PC1", y=y_pca_col, 
                color=color_by_pca, 
                text="Espece_User", 
                hover_name="Espece_User", 
                custom_data=["Espece_User", "Ecologie", "Source_Habitat", "Cluster"], 
                template="plotly_dark", height=500, color_discrete_sequence=COLOR_SEQUENCE
            )
            fig_pca.update_traces(
                textposition="top center", marker=dict(opacity=0.7), 
                hovertemplate=(
                    f"<span style='font-size: {HOVER_SPECIES_FONT_SIZE}px;'><b>%{{customdata[0]}}</b></span><br>"
                    f"Habitat: %{{customdata[2]}}<br>"
                    f"Cluster PCA: %{{customdata[3]}}<br>"
                    f"<br><span style='font-size: {HOVER_ECOLOGY_TITLE_FONT_SIZE}px;'><i>Écologie:</i></span><br>"
                    f"<span style='font-size: {HOVER_ECOLOGY_TEXT_FONT_SIZE}px;'>%{{customdata[1]}}</span>"
                    "<extra></extra>"
                ), 
                textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS)
            ) 
            
            unique_groups_pca = sorted(pdf_display_pca[color_by_pca].unique())
            extended_color_sequence_pca = COLOR_SEQUENCE * (len(unique_groups_pca) // len(COLOR_SEQUENCE) + 1)
            group_color_map_pca = {
                lbl: extended_color_sequence_pca[i % len(extended_color_sequence_pca)] for i, lbl in enumerate(unique_groups_pca)
            }
            
            for group_label_pca in unique_groups_pca:
                group_points_df_pca = pdf_display_pca[pdf_display_pca[color_by_pca] == group_label_pca]
                if not group_points_df_pca.empty and "PC1" in group_points_df_pca.columns and y_pca_col in group_points_df_pca.columns:
                    unique_group_points_pca = group_points_df_pca[["PC1", y_pca_col]].drop_duplicates().values
                    if len(unique_group_points_pca) >= MIN_POINTS_FOR_HULL:
                        try:
                            hull_pca_calc = ConvexHull(unique_group_points_pca)
                            hull_path_pca = unique_group_points_pca[np.append(hull_pca_calc.vertices, hull_pca_calc.vertices[0])] 
                            clr_pca = group_color_map_pca.get(group_label_pca, COLOR_SEQUENCE[0])
                            fig_pca.add_trace(go.Scatter(
                                x=hull_path_pca[:, 0], y=hull_path_pca[:, 1], fill="toself", fillcolor=clr_pca, 
                                line=dict(color=clr_pca, width=1.5), mode='lines', 
                                name=f'{legend_title_pca} {group_label_pca} Hull', opacity=0.2, 
                                showlegend=False, hoverinfo='skip'
                            ))
                        except Exception as e: print(f"Erreur calcul Hull ACP pour groupe {group_label_pca}: {e}")
            fig_pca.update_layout(
                title_text="Plot PCA des espèces", title_x=0.5, 
                legend_title_text=legend_title_pca, 
                dragmode='pan'
            )
        else: 
            fig_pca = None 
            if not pdf_display_pca.empty : 
                with col_pca_plot_area: st.warning("Moins de deux composantes principales disponibles pour le graphique PCA. Le graphique ne peut être affiché.")

    X_for_dendro_display = st.session_state.get('X_for_dendro', np.array([]))
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
            
            if not pdf_display_dendro_labels.empty and "Espece_User" in pdf_display_dendro_labels.columns and \
               "Source_Habitat" in pdf_display_dendro_labels.columns and \
               len(pdf_display_dendro_labels) == X_for_dendro_display.shape[0]:
                dendro_labels = [f"{row['Espece_User']} ({row['Source_Habitat']})" for _, row in pdf_display_dendro_labels.iterrows()]
            else: 
                dendro_labels = [f"Esp {i+1}" for i in range(X_for_dendro_display.shape[0])]

            fig_dend = ff.create_dendrogram(X_for_dendro_display, orientation="left", labels=dendro_labels, 
                                            linkagefun=lambda _: Z, 
                                            color_threshold=dyn_thresh if n_clust_for_dendro_color > 1 else 0, 
                                            colorscale=COLOR_SEQUENCE)
            fig_dend.update_layout(template="plotly_dark", 
                                height=max(400, X_for_dendro_display.shape[0] * 20), 
                                title_text="Dendrogramme des espèces (instances par habitat)", title_x=0.5)
        except Exception as e: 
            print(f"Erreur lors de la création du dendrogramme: {e}")
            fig_dend = None
    else: fig_dend = None


with col_pca_plot_area: 
    if fig_pca: st.plotly_chart(fig_pca, use_container_width=True, config={'scrollZoom': True}) 
    elif st.session_state.run_main_analysis_once and st.session_state.get('sub', pd.DataFrame()).empty:
        st.warning("L'analyse n'a pas produit de résultats affichables pour le PCA (pas d'espèces traitées ou PCA impossible).")
    elif st.session_state.run_main_analysis_once and fig_pca is None and not st.session_state.get('pdf', pd.DataFrame()).empty : 
        pass 
    elif st.session_state.run_main_analysis_once :
        st.warning("Le graphique PCA n'a pas pu être généré. Vérifiez les données d'entrée et les paramètres.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 4: COMPOSITION DES CLUSTERS (ACP)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty: 
    st.markdown("---"); st.subheader("Étape 4: Composition des Clusters (issus de l'ACP)")
    pdf_compo = st.session_state.get('pdf', pd.DataFrame())
    if not pdf_compo.empty and 'Cluster' in pdf_compo.columns and 'Espece_User' in pdf_compo.columns and 'Source_Habitat' in pdf_compo.columns:
        pdf_compo['Species_Instance_Display'] = pdf_compo['Espece_User'] + " (" + pdf_compo['Source_Habitat'] + ")"
        
        compositions_display = []
        for c_pca in sorted(pdf_compo["Cluster"].unique()):
            cluster_data = pdf_compo[pdf_compo["Cluster"] == c_pca]
            unique_species_instances_in_cluster = cluster_data["Species_Instance_Display"].unique()
            compositions_display.append({
                "cluster_label": c_pca, 
                "count": len(unique_species_instances_in_cluster), 
                "species_list": sorted(list(unique_species_instances_in_cluster))
            })

        if compositions_display and any(d['count'] > 0 for d in compositions_display):
            num_clusters_disp = len([d for d in compositions_display if d['count']>0]) 
            num_cols_disp = min(num_clusters_disp, 3) if num_clusters_disp > 0 else 1
            cluster_cols_layout = st.columns(num_cols_disp)
            col_idx = 0
            for comp_data in compositions_display:
                if comp_data['count'] > 0: 
                    with cluster_cols_layout[col_idx % num_cols_disp]:
                        st.markdown(f"**Cluster PCA {comp_data['cluster_label']}** ({comp_data['count']} instances d'espèces)")
                        for species_instance_name in comp_data['species_list']: st.markdown(f"- {species_instance_name}")
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
        st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 instances d'espèces uniques après traitement ou problème de données pour le linkage).")
    elif st.session_state.run_main_analysis_once : 
        st.info("Le dendrogramme n'a pas pu être généré. Vérifiez les données d'entrée (nombre d'instances d'espèces > 1, traits numériques).")
elif st.session_state.run_main_analysis_once: 
    st.markdown("---"); st.subheader("Étape 5: Dendrogramme")
    st.info("Analyse lancée, mais aucune donnée d'espèce n'a pu être traitée pour le dendrogramme.")

