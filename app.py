import streamlit as st
import pandas as pd
import plotly.express as px
# import plotly.figure_factory as ff # Retiré car non utilisé après suppression dendrogramme
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import numpy as np
import textwrap # Importé pour la mise en forme du texte de survol
from collections import defaultdict # Ajouté pour l'analyse de co-occurrence
import re # Ajouté pour parser les comptes dans les chaînes de caractères

# Assurez-vous que le fichier core.py est dans le même répertoire ou accessible
# Pour les besoins de cet exemple, si core.py n'est pas disponible,
# nous allons simuler sa fonction analyse.
try:
    import core
    # Si core.py existe mais n'a plus la fonction analyse ou si elle est différente,
    # il faudra l'adapter ou la supprimer si elle n'est plus utile.
    # Pour l'instant, on assume que si core.py est importé, il est conforme aux attentes
    # ou que les appels à ses fonctions seront conditionnels.
except ImportError:
    st.warning("Le module 'core.py' est introuvable. Certaines fonctionnalités pourraient être limitées ou simulées.")
    # Simulation de la fonction core.analyse pour permettre à l'UI de fonctionner
    # Cette fonction est grandement simplifiée car l'ACP n'est plus utilisée.
    # Elle pourrait retourner des structures vides ou ne pas être appelée du tout.
    class MockCoreModule:
        def analyse(self, sub_df_prepared, n_clusters_unused):
            # L'ACP et le clustering associé ne sont plus effectués.
            # Retourner des structures vides ou non pertinentes pour l'ancienne logique ACP.
            # Les 'labels' PCA, 'pca_obj', 'coords_df' ne sont plus nécessaires.
            # 'X_scaled_data' pour le dendrogramme n'est plus nécessaire.
            print("MockCoreModule.analyse appelée, mais l'ACP est désactivée. Retour de structures vides.")
            
            # Simuler un DataFrame de coordonnées vide ou avec une structure minimale si absolument nécessaire
            # pour éviter des erreurs en aval, bien que l'objectif soit de ne plus utiliser ces sorties.
            # Si sub_df_prepared a un index, on peut le réutiliser pour la cohérence.
            idx = sub_df_prepared.index if hasattr(sub_df_prepared, 'index') else pd.Index([])
            
            # labels_sim (pour clusters PCA) - non utilisé
            labels_sim = np.array([])
            if len(idx) > 0: # Simuler des labels si des données d'entrée existent
                 labels_sim = np.zeros(len(idx), dtype=int)


            # pca_obj_sim (objet PCA) - non utilisé
            class MockPCAObj:
                def __init__(self):
                    self.components_ = np.array([])
                    self.explained_variance_ = np.array([])
            pca_obj_sim = MockPCAObj()

            # coords_df_sim (coordonnées PCA) - non utilisé
            coords_df_sim = pd.DataFrame(index=idx) # Vide, ou avec des colonnes PC vides si besoin
            # if len(idx) > 0:
            #     coords_df_sim['PC1_sim'] = np.random.rand(len(idx))
            #     coords_df_sim['PC2_sim'] = np.random.rand(len(idx))


            # X_scaled_sim (pour dendrogramme) - non utilisé
            n_samples_sim = len(idx)
            n_features_sim = sub_df_prepared.shape[1] -1 if sub_df_prepared.shape[1] > 0 else 0 # Exclure 'Espece'
            X_scaled_sim = np.array([]).reshape(n_samples_sim, n_features_sim if n_features_sim > 0 else (1 if n_samples_sim > 0 else 0) )


            # Retourner des valeurs qui ne causeront pas d'erreurs si l'appel est toujours fait,
            # mais ces valeurs ne devraient plus être utilisées pour la logique ACP.
            return labels_sim, pca_obj_sim, coords_df_sim, X_scaled_sim

        def read_reference(self, fp):
            st.warning(f"Simulation du chargement de '{fp}' via MockCoreModule.read_reference. Le fichier réel n'est pas utilisé.")
            example_species = [f"Espece Alpha {i}" for i in range(1, 11)] + \
                              [f"Espece Beta {i}" for i in range(1, 11)] + \
                              [f"Espece Gamma {i}" for i in range(1, 11)]
            data = pd.DataFrame({
                'Espece': example_species,
                'Trait_Num_1': np.random.rand(30) * 10,
                'Trait_Num_2': np.random.randint(1, 100, 30),
                'Trait_Cat_1': np.random.choice(['X', 'Y', 'Z'], 30),
                'Humidité_édaphique': np.random.rand(30) * 5 + 1,
                'Matière_organique': np.random.rand(30) * 10,
                'Lumière': np.random.rand(30) * 1000
            })
            data.loc[len(data)] = ['Rhamnus pumila', 5.0, 50, 'X', 3, 5, 500]
            data.loc[len(data)] = ['Vulpia sp.', 2.0, 20, 'Y', 2, 2, 800]
            return data

    core = MockCoreModule()


# ---------------------------------------------------------------------------- #
# CONFIGURATION UI
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="Analyse de Relevés Floristiques", layout="wide") # Titre de page mis à jour
st.markdown("<h1 style='text-align: center;'>Analyse Interactive de Relevés Floristiques et Syntaxons</h1>", unsafe_allow_html=True) # Titre principal mis à jour

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

/* Style pour la première ligne du st.data_editor (noms des habitats) */
div[data-testid="stDataEditor"] .glideDataEditor-body .dvn-scroll-inner > div:first-child > div[data-cell^="[0,"] > div {
    background-color: #22272f !important; 
    color: #e1e1e1 !important;        
    font-weight: bold !important;
}
/* Style pour la cellule de la première ligne en mode édition */
div[data-testid="stDataEditor"] .glideDataEditor-body .dvn-scroll-inner > div:first-child > div[data-cell^="[0,"] > div > .gdg-input {
    background-color: #ffffff !important; 
    color: #000000 !important;       
    font-weight: normal !important;   
}

.habitat-select-button button {
    font-size: 13px !important;
    padding: 0.25rem 0.6rem !important; 
    line-height: 1.4;
    width: 100%; 
    border-radius: 0.5rem; 
}
/* Style pour surligner les espèces en commun dans le tableau des syntaxons */
.common-species {
    color: red;
    font-weight: bold;
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
        # Utiliser core.read_reference si disponible et non la simulation par défaut de MockCoreModule
        if hasattr(core, "read_reference") and not (isinstance(core, MockCoreModule) and core.read_reference.__name__ == 'read_reference'):
             data = core.read_reference(file_path)
        else: # Fallback sur la simulation si core.read_reference n'est pas la vraie ou si core est le mock
            st.warning(f"Simulation du chargement de '{file_path}' (core.read_reference non disponible ou mock).")
            example_species = [f"Espece Alpha {i}" for i in range(1, 11)] + \
                              [f"Espece Beta {i}" for i in range(1, 11)] + \
                              [f"Espece Gamma {i}" for i in range(1, 11)]
            data = pd.DataFrame({
                'Espece': example_species, 
                'Trait_Num_1': np.random.rand(30) * 10,
                'Trait_Num_2': np.random.randint(1, 100, 30),
                'Trait_Cat_1': np.random.choice(['X', 'Y', 'Z'], 30), 
                'Humidité_édaphique': np.random.rand(30) * 5 + 1, 
                'Matière_organique': np.random.rand(30) * 10,
                'Lumière': np.random.rand(30) * 1000
            })
            data.loc[len(data)] = ['Rhamnus pumila', 5.0, 50, 'X', 3, 5, 500]
            data.loc[len(data)] = ['Vulpia sp.', 2.0, 20, 'Y', 2, 2, 800]
        
        if data.empty:
            st.warning(f"Le fichier de données de traits '{file_path}' est vide ou n'a pas pu être lu correctement.")
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
        .str[:2] # Prend les deux premiers mots (Genre et espèce)
        .str.join(" ")
        .str.lower() # Convertit en minuscules pour la comparaison
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
            header=None, # Pas de ligne d'en-tête dans le fichier
            usecols=[0, 1], # Utiliser seulement les deux premières colonnes
            names=['Espece', 'Description_Ecologie'], # Nommer les colonnes
            encoding='utf-8-sig', # Gérer le BOM (Byte Order Mark)
            keep_default_na=False, # Ne pas interpréter 'NA' comme NaN
            na_values=[''] # Interpréter les chaînes vides comme NaN
        )
        eco_data = eco_data.dropna(subset=['Espece']) # Supprimer les lignes où 'Espece' est NaN
        eco_data['Espece'] = eco_data['Espece'].astype(str).str.strip() # Nettoyer les noms d'espèces
        eco_data = eco_data[eco_data['Espece'] != ""] # Supprimer les espèces vides après nettoyage

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
        st.toast(f"Erreur chargement fichier écologique.", icon=" ")
        return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))

ecology_df = load_ecology_data()


# ---------------------------------------------------------------------------- #
# FONCTION UTILITAIRE POUR NORMALISER LES NOMS D'ESPECES
# ---------------------------------------------------------------------------- #
def normalize_species_name(species_name): # Renommée pour usage général
    """Normalise un nom d'espèce en prenant les deux premiers mots et en convertissant en minuscules."""
    if pd.isna(species_name) or str(species_name).strip() == "":
        return None
    return " ".join(str(species_name).strip().split()[:2]).lower()

# ---------------------------------------------------------------------------- #
# CHARGEMENT DES DONNÉES DES SYNTAXONS
# ---------------------------------------------------------------------------- #
@st.cache_data
def load_syntaxon_data(file_path="data_villaret.csv"):
    try:
        df = pd.read_csv(file_path, sep=';', header=None, encoding='utf-8-sig', keep_default_na=False, na_values=[''])
        if df.empty:
            st.warning(f"Le fichier des syntaxons '{file_path}' est vide.")
            return []

        processed_syntaxons = []
        for index, row in df.iterrows():
            syntaxon_id = str(row.iloc[0]).strip()
            syntaxon_name_latin = str(row.iloc[1]).strip() # Nom latin du syntaxon
            
            species_in_row_set = set()
            # Les espèces commencent à partir de la 3ème colonne (index 2)
            for species_cell_value in row.iloc[2:]:
                normalized_species = normalize_species_name(species_cell_value) # Utilisation de la fonction normalisée
                if normalized_species: 
                    species_in_row_set.add(normalized_species)
            
            if syntaxon_id and syntaxon_name_latin and species_in_row_set:
                processed_syntaxons.append({
                    'id': syntaxon_id,
                    'name_latin': syntaxon_name_latin, # Stocker le nom latin
                    'species_set': species_in_row_set 
                })
        
        if not processed_syntaxons:
            st.warning(f"Aucun syntaxon valide (avec ID, nom et espèces) n'a été trouvé dans '{file_path}'.")
        return processed_syntaxons
        
    except FileNotFoundError:
        st.error(f"ERREUR CRITIQUE: Fichier des syntaxons '{file_path}' non trouvé. L'identification des syntaxons et l'analyse de co-occurrence ne pourront pas être effectuées.")
        return []
    except pd.errors.EmptyDataError: 
        st.warning(f"Le fichier des syntaxons '{file_path}' est vide (EmptyDataError).")
        return []
    except Exception as e: 
        st.error(f"ERREUR CRITIQUE: Impossible de charger les données des syntaxons depuis '{file_path}': {e}")
        return []

syntaxon_data_list = load_syntaxon_data()


# ---------------------------------------------------------------------------- #
# INITIALISATION DES ETATS DE SESSION
# ---------------------------------------------------------------------------- #
default_session_states = {
    'x_axis_trait_interactive': None,
    'y_axis_trait_interactive': None,
    'run_main_processing_once': False, # Renommé pour refléter l'absence de PCA
    'trait_exploration_df': pd.DataFrame(), # Remplacement de vip_data_df_interactive
    'trait_exploration_df_snapshot': pd.DataFrame(), # Remplacement de vip_data_df_interactive_snapshot
    'sub': pd.DataFrame(), 
    # 'pdf': pd.DataFrame(), # Supprimé, car c'était pour les résultats PCA
    'numeric_trait_names_for_interactive_plot': [],
    'selected_habitats_indices': [], 
    'previous_num_cols': 0,
    'processing_has_run_for_current_selection': False, # Renommé
    # 'n_clusters_slider_main_value': 3, # Supprimé, car lié à l'ACP
    'top_matching_syntaxons': [] # Pour stocker les syntaxons identifiés
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
            st.session_state.processing_has_run_for_current_selection = False
            st.session_state.run_main_processing_once = False 
        st.session_state.previous_num_cols = len(st.session_state.releves_df.columns) 
    st.rerun() 

current_releves_df_for_selection = st.session_state.releves_df.copy() 

if not current_releves_df_for_selection.empty and \
   len(current_releves_df_for_selection.columns) > 0 and \
   len(current_releves_df_for_selection) > 0:
    
    habitat_names_from_df = current_releves_df_for_selection.iloc[0].astype(str).tolist()
    num_actual_cols = len(current_releves_df_for_selection.columns) 
    
    st.markdown("**Cliquez sur le nom d'un habitat ci-dessous pour le sélectionner/désélectionner :**") 
    
    st.session_state.selected_habitats_indices = [
        idx for idx in st.session_state.selected_habitats_indices if idx < num_actual_cols
    ]

    valid_habitat_buttons_info = []
    for i in range(num_actual_cols):
        species_in_col = current_releves_df_for_selection.iloc[1:, i].dropna().astype(str).str.strip().replace('', np.nan).dropna()
        if not species_in_col.empty: 
            habitat_name_for_button = habitat_names_from_df[i] if pd.notna(habitat_names_from_df[i]) and str(habitat_names_from_df[i]).strip() != "" else f"Relevé {i+1}"
            valid_habitat_buttons_info.append({'index': i, 'name': habitat_name_for_button})

    if valid_habitat_buttons_info: 
        num_buttons_to_show = len(valid_habitat_buttons_info)
        button_cols_layout = st.columns(num_buttons_to_show) 
        
        for k, habitat_info in enumerate(valid_habitat_buttons_info):
            col_idx = habitat_info['index'] 
            habitat_name_display = habitat_info['name'] 
            is_selected = (col_idx in st.session_state.selected_habitats_indices) 
            
            button_type = "primary" if is_selected else "secondary" 
            button_key = f"habitat_select_button_{col_idx}" 

            with button_cols_layout[k]: 
                st.markdown(f'<div class="habitat-select-button">', unsafe_allow_html=True)
                if st.button(habitat_name_display, key=button_key, type=button_type, use_container_width=True):
                    if is_selected:
                        st.session_state.selected_habitats_indices.remove(col_idx) 
                    else:
                        st.session_state.selected_habitats_indices.append(col_idx) 
                    st.session_state.run_main_processing_once = False 
                    st.session_state.processing_has_run_for_current_selection = False 
                    st.rerun() 
                st.markdown('</div>', unsafe_allow_html=True)
    elif num_actual_cols > 0 : 
        st.info("Aucune colonne ne contient de données d'espèces pour la sélection. Veuillez ajouter des espèces sous les noms d'habitats.")
    else: 
        st.info("Ajoutez des colonnes au tableau pour pouvoir sélectionner des relevés.")
else:
    st.warning("Le tableau de données est vide ou ne contient pas de colonnes pour la sélection.")


# Condition pour lancer le traitement principal (anciennement analyse principale)
if st.session_state.selected_habitats_indices and \
   not ref.empty and \
   not st.session_state.get('processing_has_run_for_current_selection', False): 

    st.session_state.run_main_processing_once = True 
    st.session_state.processing_has_run_for_current_selection = True 

    st.session_state.sub = pd.DataFrame() 
    # st.session_state.pdf = pd.DataFrame() # Supprimé (lié à l'ACP)
    st.session_state.trait_exploration_df = pd.DataFrame() 
    st.session_state.numeric_trait_names_for_interactive_plot = [] 
    st.session_state.top_matching_syntaxons = [] # Réinitialiser les syntaxons

    all_species_data_for_processing = [] 
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
                
                binom_species_name = normalize_species_name(raw_species_name) # Utilisation de la fonction normalisée
                
                if not ref_binom_series.empty: 
                    match_in_ref = ref_binom_series[ref_binom_series == binom_species_name]
                    if not match_in_ref.empty: 
                        ref_idx = match_in_ref.index[0] 
                        trait_data = ref.loc[ref_idx].to_dict() 
                        trait_data['Source_Habitat'] = habitat_name 
                        trait_data['Espece_Ref_Original'] = ref.loc[ref_idx, 'Espece'] 
                        trait_data['Espece_User_Input_Raw'] = raw_species_name 
                        # Ajouter l'information d'écologie directement à 'sub'
                        normalized_ref_for_eco = normalize_species_name(trait_data['Espece_Ref_Original'])
                        if not ecology_df.empty and normalized_ref_for_eco in ecology_df.index:
                             trait_data['Ecologie_raw'] = ecology_df.loc[normalized_ref_for_eco, 'Description_Ecologie']
                        else:
                             trait_data['Ecologie_raw'] = None # Ou une chaîne vide/message par défaut
                        trait_data['Ecologie'] = format_ecology_for_hover(trait_data['Ecologie_raw'])

                        all_species_data_for_processing.append(trait_data)
                    else: 
                        species_not_found_in_ref_detailed[habitat_name].append(raw_species_name)
                else: 
                    species_not_found_in_ref_detailed[habitat_name].append(raw_species_name)

    if not all_species_data_for_processing: 
        st.error("Aucune espèce valide correspondante aux traits n'a été trouvée dans les relevés sélectionnés. Vérifiez vos données et sélections.")
        st.session_state.run_main_processing_once = False 
        st.session_state.processing_has_run_for_current_selection = False 
    else: 
        st.session_state.sub = pd.DataFrame(all_species_data_for_processing) 
        st.session_state.sub.reset_index(drop=True, inplace=True) 

        for habitat_name, not_found_list in species_not_found_in_ref_detailed.items():
            if not_found_list:
                st.warning(f"Espèces de '{habitat_name}' non trouvées dans la base de traits : " + ", ".join(not_found_list), icon="⚠️")

        # Le code appelant core.analyse pour l'ACP est supprimé ici.
        # La logique PCA (labels, pca_results, coords_df) n'est plus nécessaire.

        if st.session_state.sub.empty: 
            st.error("Aucune des espèces sélectionnées n'a été trouvée dans la base de traits. Le traitement ne peut continuer.")
            st.session_state.run_main_processing_once = False; st.session_state.processing_has_run_for_current_selection = False;
        elif st.session_state.sub.shape[0] < 1: # Besoin d'au moins 1 espèce pour l'exploration des traits
            st.error(f"Au moins 1 instance d'espèce (total sur les habitats) est nécessaire. {st.session_state.sub.shape[0]} trouvée(s).");
            st.session_state.run_main_processing_once = False; st.session_state.processing_has_run_for_current_selection = False;
        else:
            try:
                if ref.empty: 
                    st.error("Le DataFrame de référence 'ref' est vide. Impossible de déterminer les traits numériques.")
                    st.session_state.run_main_processing_once = False
                    st.session_state.processing_has_run_for_current_selection = False
                    raise ValueError("DataFrame 'ref' vide.")

                numeric_trait_names_from_ref = ref.select_dtypes(include=np.number).columns.tolist()
                actual_numeric_traits = [
                    trait for trait in numeric_trait_names_from_ref if trait in st.session_state.sub.columns
                ]
                
                if not actual_numeric_traits:
                    st.error("Aucun trait numérique disponible pour l'exploration. Le traitement est impossible.")
                    st.session_state.run_main_processing_once = False
                    st.session_state.processing_has_run_for_current_selection = False
                    raise ValueError("Aucun trait numérique pour l'exploration.")
                else:
                    st.session_state.numeric_trait_names_for_interactive_plot = actual_numeric_traits
                    
                    # Préparer le DataFrame pour l'éditeur de sélection des axes du graphique interactif
                    # Ce DataFrame ne contiendra plus de communalités.
                    exploration_df_data = []
                    for trait_name in actual_numeric_traits:
                        exploration_df_data.append({"Variable": trait_name, "Axe X": False, "Axe Y": False})
                    
                    st.session_state.trait_exploration_df = pd.DataFrame(exploration_df_data)

                    # Définir les axes par défaut pour le graphique interactif
                    default_x_init, default_y_init = None, None
                    if actual_numeric_traits:
                        default_x_init = actual_numeric_traits[0]
                        if len(actual_numeric_traits) >= 2:
                            default_y_init = actual_numeric_traits[1]
                        else:
                            default_y_init = actual_numeric_traits[0] # Si un seul trait, utiliser pour X et Y

                    st.session_state.x_axis_trait_interactive = default_x_init
                    st.session_state.y_axis_trait_interactive = default_y_init
                    
                    # Mettre à jour les cases à cocher dans trait_exploration_df
                    if not st.session_state.trait_exploration_df.empty:
                        st.session_state.trait_exploration_df["Axe X"] = (st.session_state.trait_exploration_df["Variable"] == default_x_init)
                        st.session_state.trait_exploration_df["Axe Y"] = (st.session_state.trait_exploration_df["Variable"] == default_y_init)
                    
                    st.session_state.trait_exploration_df_snapshot = st.session_state.trait_exploration_df.copy()

            except Exception as e: 
                st.error(f"Erreur lors du traitement principal : {e}"); st.exception(e)
                st.session_state.run_main_processing_once = False; st.session_state.processing_has_run_for_current_selection = False;

elif not st.session_state.selected_habitats_indices and not ref.empty:
    st.info("Veuillez sélectionner un ou plusieurs habitats à l'Étape 1 pour lancer le traitement.")
elif ref.empty: 
    st.warning("Les données de référence ('data_ref.csv') n'ont pas pu être chargées ou sont simulées. Le traitement est désactivé si les données réelles manquent.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 2: EXPLORATION INTERACTIVE DES TRAITS (anciennement Étape 2 avec ACP)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_processing_once and not st.session_state.get('sub', pd.DataFrame()).empty: 
    st.markdown("---"); st.subheader("Étape 2: Exploration Interactive des Traits")
    col_interactive_table, col_interactive_graph = st.columns([1, 2]) 

    with col_interactive_table: 
        st.markdown("##### Sélection des traits pour le graphique")
        df_editor_source_interactive = st.session_state.get('trait_exploration_df', pd.DataFrame())

        if not df_editor_source_interactive.empty:
            snapshot_cols = list(st.session_state.get('trait_exploration_df_snapshot', pd.DataFrame()).columns)
            current_cols = list(df_editor_source_interactive.columns)
            if 'trait_exploration_df_snapshot' not in st.session_state or snapshot_cols != current_cols:
                st.session_state.trait_exploration_df_snapshot = df_editor_source_interactive.copy()

            edited_df_interactive = st.data_editor(
                df_editor_source_interactive, 
                column_config={
                    "Variable": st.column_config.TextColumn("Trait disponible", disabled=True), 
                    # "Communalité (%)": Supprimé
                    "Axe X": st.column_config.CheckboxColumn("Axe X"), 
                    "Axe Y": st.column_config.CheckboxColumn("Axe Y") 
                }, 
                key="interactive_trait_exploration_editor", 
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
                
                st.session_state.trait_exploration_df = df_updated_for_editor 
                st.session_state.trait_exploration_df_snapshot = df_updated_for_editor.copy() 
                st.rerun() 
            elif not edited_df_interactive.equals(st.session_state.trait_exploration_df_snapshot):
                   st.session_state.trait_exploration_df_snapshot = edited_df_interactive.copy() 
        else: 
            st.info("Le tableau de sélection des traits sera disponible après le traitement si des traits numériques sont identifiés.")

        # Le slider pour n_clusters (ACP) est supprimé.
        # st.markdown("---") 
        # st.markdown("##### Paramètres ACP") # Supprimé

    with col_interactive_graph: 
        st.markdown("##### Graphique d'exploration des traits")
        x_axis_plot = st.session_state.x_axis_trait_interactive 
        y_axis_plot = st.session_state.y_axis_trait_interactive 
        numeric_traits_plot = st.session_state.get('numeric_trait_names_for_interactive_plot', []) 
        sub_plot_releve = st.session_state.get('sub', pd.DataFrame()) # Espèces du relevé utilisateur
        
        # Récupérer les espèces des syntaxons identifiés pour le graphique
        top_syntaxons_for_plot = st.session_state.get('top_matching_syntaxons', [])
        all_plot_data_list = []

        # 1. Ajouter les données du relevé utilisateur
        if not sub_plot_releve.empty and x_axis_plot and y_axis_plot and \
           x_axis_plot in sub_plot_releve.columns and y_axis_plot in sub_plot_releve.columns:
            
            # S'assurer que les colonnes nécessaires pour le hovertemplate sont présentes
            required_cols_releve = ['Espece_User_Input_Raw', 'Ecologie', 'Source_Habitat']
            if all(col in sub_plot_releve.columns for col in required_cols_releve):
                releve_plot_df = sub_plot_releve[[x_axis_plot, y_axis_plot] + required_cols_releve].copy()
                releve_plot_df['Source_Donnee'] = 'Relevé Utilisateur'
                releve_plot_df['Nom_Affichage'] = releve_plot_df['Espece_User_Input_Raw']
                releve_plot_df['Groupe_Affichage'] = releve_plot_df['Source_Habitat'] # Pour coloration/hulls potentiels
                releve_plot_df['Symbole'] = 'circle'
                all_plot_data_list.append(releve_plot_df)
            # else:
                # st.warning("Colonnes manquantes dans 'sub_plot_releve' pour le graphique interactif.")


        # 2. Ajouter les données des espèces des syntaxons
        if top_syntaxons_for_plot and not ref.empty and 'Espece' in ref.columns:
            for i, syntaxon_info in enumerate(top_syntaxons_for_plot):
                syntaxon_name_for_graph = syntaxon_info.get('name_latin_short', f"Syntaxon {syntaxon_info.get('id', i+1)}")
                species_to_plot_from_syntaxon = []
                for species_norm in syntaxon_info.get('species_set', []):
                    match_in_ref = ref[ref_binom_series == species_norm]
                    if not match_in_ref.empty:
                        ref_idx = match_in_ref.index[0]
                        trait_data_syntaxon_sp = ref.loc[ref_idx].to_dict()
                        
                        # Vérifier si les traits X et Y sont présents pour cette espèce
                        if x_axis_plot in trait_data_syntaxon_sp and y_axis_plot in trait_data_syntaxon_sp:
                            # Récupérer l'écologie
                            eco_desc_raw = None
                            if not ecology_df.empty and species_norm in ecology_df.index:
                                eco_desc_raw = ecology_df.loc[species_norm, 'Description_Ecologie']
                            
                            species_to_plot_from_syntaxon.append({
                                x_axis_plot: trait_data_syntaxon_sp[x_axis_plot],
                                y_axis_plot: trait_data_syntaxon_sp[y_axis_plot],
                                'Espece_User_Input_Raw': ref.loc[ref_idx, 'Espece'], # Nom original de ref
                                'Ecologie': format_ecology_for_hover(eco_desc_raw),
                                'Source_Habitat': syntaxon_name_for_graph, # Utiliser le nom du syntaxon comme "habitat"
                                'Source_Donnee': f"Syntaxon: {syntaxon_name_for_graph}",
                                'Nom_Affichage': ref.loc[ref_idx, 'Espece'],
                                'Groupe_Affichage': f"Syntaxon: {syntaxon_name_for_graph}",
                                'Symbole': 'triangle-up' # Symbole différent pour les espèces de syntaxon
                            })
                if species_to_plot_from_syntaxon:
                    all_plot_data_list.append(pd.DataFrame(species_to_plot_from_syntaxon))
        
        # Combiner toutes les données pour le graphique
        if all_plot_data_list:
            final_plot_df = pd.concat(all_plot_data_list, ignore_index=True)

            # Vérifications avant de tracer
            if not numeric_traits_plot: st.warning("Aucun trait numérique trouvé pour l'exploration interactive.")
            elif not x_axis_plot or not y_axis_plot: st.info("Veuillez sélectionner un trait pour l'Axe X et un pour l'Axe Y dans le tableau à gauche.")
            elif x_axis_plot not in numeric_traits_plot or y_axis_plot not in numeric_traits_plot: st.warning("Un ou les deux traits sélectionnés ne sont plus valides. Veuillez re-sélectionner.")
            elif final_plot_df.empty or x_axis_plot not in final_plot_df.columns or y_axis_plot not in final_plot_df.columns: 
                st.warning("Données pour le graphique interactif non prêtes, incohérentes ou traits sélectionnés non trouvés.")
            else:
                plot_data_to_use = final_plot_df.copy()
                # Gestion du jitter (identique à l'ancien code, mais appliqué à plot_data_to_use)
                temp_x_col_grp = "_temp_x_for_grouping"; temp_y_col_grp = "_temp_y_for_grouping"
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

                # Coloration et légende
                color_by_interactive = "Groupe_Affichage" 
                legend_title_interactive = "Groupe"

                fig_interactive_scatter = px.scatter(
                    plot_data_to_use, x=x_axis_plot, y=y_axis_plot,
                    color=color_by_interactive, 
                    symbol='Symbole', # Utilisation de la colonne Symbole
                    text="Nom_Affichage", hover_name="Nom_Affichage", 
                    custom_data=["Nom_Affichage", "Ecologie", "Source_Habitat", "Source_Donnee"], 
                    template="plotly_dark", height=600, color_discrete_sequence=COLOR_SEQUENCE
                )
                fig_interactive_scatter.update_traces(
                    textposition="top center", marker=dict(opacity=0.8, size=8), 
                    textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS), 
                    hovertemplate=( 
                        f"<span style='font-size: {HOVER_SPECIES_FONT_SIZE}px;'><b>%{{customdata[0]}}</b></span><br>" 
                        f"Source: %{{customdata[3]}}<br>" # Source (Relevé ou Syntaxon X)
                        f"Habitat/Syntaxon: %{{customdata[2]}}<br>" # Habitat d'origine ou nom du syntaxon
                        f"<br><span style='font-size: {HOVER_ECOLOGY_TITLE_FONT_SIZE}px;'><i>Écologie:</i></span><br>" 
                        f"<span style='font-size: {HOVER_ECOLOGY_TEXT_FONT_SIZE}px;'>%{{customdata[1]}}</span>" 
                        "<extra></extra>" 
                    )
                )
                
                # Enveloppes convexes
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
                                    name=f'{group_label} Hull', opacity=0.2, 
                                    showlegend=False, hoverinfo='skip' 
                                ))
                            except Exception as e: print(f"Erreur calcul Hull interactif {group_label} ({x_axis_plot}, {y_axis_plot}): {e}")
                
                fig_interactive_scatter.update_layout(
                    title_text=f"{y_axis_plot} vs. {x_axis_plot}", title_x=0.5, 
                    xaxis_title=x_axis_plot, yaxis_title=y_axis_plot, dragmode='pan', 
                    legend_title_text=legend_title_interactive 
                )
                st.plotly_chart(fig_interactive_scatter, use_container_width=True, config={'scrollZoom': True})
        else:
            st.info("Préparez les données et sélectionnez les axes pour afficher le graphique interactif.")

elif st.session_state.run_main_processing_once and st.session_state.get('sub', pd.DataFrame()).empty :
    st.markdown("---")
    st.subheader("Étape 2: Exploration Interactive des Traits")
    st.warning("Le traitement principal n'a pas abouti à des données suffisantes pour cette section (aucune espèce trouvée ou traitée). Veuillez vérifier les étapes précédentes.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 3: IDENTIFICATION DES SYNTAXONS PERTINENTS (Nouvelle Étape)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_processing_once and not st.session_state.get('sub', pd.DataFrame()).empty and syntaxon_data_list:
    st.markdown("---")
    st.subheader("Étape 3: Identification des Syntaxons Pertinents")

    releve_species_normalized = set(normalize_species_name(sp) for sp in st.session_state.sub['Espece_Ref_Original'].unique())
    releve_species_normalized.discard(None) # Retirer les None si certains noms n'ont pas pu être normalisés

    if not releve_species_normalized:
        st.warning("Aucune espèce normalisée issue de vos relevés n'a pu être identifiée pour la comparaison avec les syntaxons.")
    else:
        syntaxon_matches = []
        for syntaxon in syntaxon_data_list:
            common_species = releve_species_normalized.intersection(syntaxon['species_set'])
            score = len(common_species) # Score basé sur le nombre d'espèces communes
            if score > 0: # Considérer uniquement les syntaxons avec au moins une espèce commune
                syntaxon_matches.append({
                    'id': syntaxon['id'],
                    'name_latin': syntaxon['name_latin'],
                    'name_latin_short': ' '.join(syntaxon['name_latin'].split()[:2]), # Pour affichage court
                    'species_set': syntaxon['species_set'],
                    'common_species_set': common_species,
                    'score': score
                })
        
        # Trier par score (décroissant) puis par nom de syntaxon (alphabétique)
        sorted_syntaxons = sorted(syntaxon_matches, key=lambda x: (-x['score'], x['name_latin']))
        st.session_state.top_matching_syntaxons = sorted_syntaxons[:5] # Garder les 5 meilleurs

        if not st.session_state.top_matching_syntaxons:
            st.info("Aucun syntaxon correspondant n'a été trouvé pour les espèces de vos relevés avec les données de syntaxons fournies.")
        else:
            st.markdown(f"Les **{len(st.session_state.top_matching_syntaxons)} syntaxons les plus probables** (basé sur le nombre d'espèces en commun) sont :")
            
            # Créer des colonnes pour afficher les syntaxons
            num_syntaxons_to_show = len(st.session_state.top_matching_syntaxons)
            cols_syntaxon_display = st.columns(num_syntaxons_to_show if num_syntaxons_to_show > 0 else 1)

            for i, matched_syntaxon in enumerate(st.session_state.top_matching_syntaxons):
                with cols_syntaxon_display[i % num_syntaxons_to_show if num_syntaxons_to_show > 0 else 0]:
                    st.markdown(f"**{i+1}. ID: {matched_syntaxon['id']}**")
                    st.markdown(f"*{matched_syntaxon['name_latin']}*")
                    st.markdown(f"Score de correspondance: {matched_syntaxon['score']} espèces communes")
                    
                    html_species_list = "<ul>"
                    # Trier les espèces du syntaxon pour un affichage cohérent, en mettant les communes en premier
                    sorted_syntaxon_species = sorted(list(matched_syntaxon['species_set']), 
                                                     key=lambda sp_name: (sp_name not in matched_syntaxon['common_species_set'], sp_name))

                    for species_name_norm in sorted_syntaxon_species:
                        species_display_name = species_name_norm.capitalize() # Mettre en majuscule la première lettre
                        if species_name_norm in matched_syntaxon['common_species_set']:
                            # Espèce commune, surligner en rouge
                            html_species_list += f"<li><span class='common-species'>{species_display_name}</span></li>"
                        else:
                            html_species_list += f"<li>{species_display_name}</li>"
                    html_species_list += "</ul>"
                    st.markdown(html_species_list, unsafe_allow_html=True)
                    st.markdown("---") # Séparateur entre les syntaxons dans la même colonne si plusieurs
elif st.session_state.run_main_processing_once and not syntaxon_data_list:
    st.markdown("---")
    st.subheader("Étape 3: Identification des Syntaxons Pertinents")
    st.warning("Les données des syntaxons ('data_villaret.csv') n'ont pas pu être chargées ou sont vides. L'identification des syntaxons ne peut être effectuée.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 3 & 4: VISUALISATION PRINCIPALE (ACP) & COMPOSITION DES CLUSTERS (ACP) - SUPPRIMÉES
# ---------------------------------------------------------------------------- #
# st.markdown("---")
# st.subheader("Étape 3: Visualisation Principale (ACP)") -> Supprimé
# ... (code de fig_pca) ... -> Supprimé

# st.markdown("---"); st.subheader("Étape 4: Composition des Clusters (issus de l'ACP)") -> Supprimé
# ... (code de composition des clusters) ... -> Supprimé


# ---------------------------------------------------------------------------- #
# ÉTAPE 4: ANALYSE DES CO-OCCURRENCES D'ESPÈCES (anciennement Étape 5)
# ---------------------------------------------------------------------------- #
def style_cooccurrence_row_parsing(row, max_overall_count, vmin_count=1):
    styles = pd.Series('', index=row.index) 
    color_start_rgb = (40, 40, 40) 
    color_end_rgb = (200, 50, 50)   

    for col_name in ['Voisin 1', 'Voisin 2', 'Voisin 3']: 
        if col_name in row.index:
            cell_value = str(row[col_name]) 
            current_count = 0 

            match = re.search(r' - (\d+)$', cell_value)
            if match:
                current_count = int(match.group(1))
            
            if current_count > 0:
                if max_overall_count == vmin_count: 
                    ratio = 1.0 if current_count >= vmin_count else 0.0
                elif max_overall_count > vmin_count:
                    ratio = (current_count - vmin_count) / (max_overall_count - vmin_count)
                    ratio = max(0.0, min(ratio, 1.0)) 
                else: 
                    ratio = 0.0
                
                r = int(color_start_rgb[0] + ratio * (color_end_rgb[0] - color_start_rgb[0]))
                g = int(color_start_rgb[1] + ratio * (color_end_rgb[1] - color_start_rgb[1]))
                b = int(color_start_rgb[2] + ratio * (color_end_rgb[2] - color_start_rgb[2]))
                styles[col_name] = f'background-color: rgb({r},{g},{b})'
            else:
                styles[col_name] = 'background-color: none' 
    return styles

if st.session_state.run_main_processing_once and \
   not st.session_state.get('sub', pd.DataFrame()).empty and \
   syntaxon_data_list:

    st.markdown("---")
    st.subheader("Étape 4: Analyse des Co-occurrences d'Espèces (basée sur les listes de syntaxons)")

    principal_species_original_names_from_sub = st.session_state.sub['Espece_Ref_Original'].unique()
    
    raw_cooccurrence_data_for_max_calc = []
    for principal_species_original in principal_species_original_names_from_sub:
        principal_species_normalized = normalize_species_name(principal_species_original)
        if not principal_species_normalized:
            continue

        co_occurrence_counts_for_this_principal = defaultdict(int)
        for syntaxon_record in syntaxon_data_list:
            if principal_species_normalized in syntaxon_record['species_set']:
                for other_species_in_syntaxon_normalized in syntaxon_record['species_set']:
                    if other_species_in_syntaxon_normalized != principal_species_normalized:
                        co_occurrence_counts_for_this_principal[other_species_in_syntaxon_normalized] += 1
        
        if co_occurrence_counts_for_this_principal:
            sorted_co_occurrences = sorted(co_occurrence_counts_for_this_principal.items(), key=lambda item: item[1], reverse=True)
            for i in range(3):
                if i < len(sorted_co_occurrences):
                    _, count = sorted_co_occurrences[i]
                    raw_cooccurrence_data_for_max_calc.append({'count': count}) 
    
    all_counts_for_styling = [item['count'] for item in raw_cooccurrence_data_for_max_calc if item['count'] > 0]
    max_overall_cooccurrence = max(all_counts_for_styling) if all_counts_for_styling else 0
    min_cooccurrence_for_color = 1 

    cooccurrence_display_list = []
    for principal_species_original in principal_species_original_names_from_sub:
        principal_species_normalized = normalize_species_name(principal_species_original)
        if not principal_species_normalized: continue 

        co_occurrence_counts_for_this_principal = defaultdict(int) 
        for syntaxon_record in syntaxon_data_list:
            if principal_species_normalized in syntaxon_record['species_set']:
                for other_species_in_syntaxon_normalized in syntaxon_record['species_set']:
                    if other_species_in_syntaxon_normalized != principal_species_normalized:
                        co_occurrence_counts_for_this_principal[other_species_in_syntaxon_normalized] += 1
        
        display_row_dict = {'Espèce Principale (issue des relevés)': principal_species_original}
        if co_occurrence_counts_for_this_principal:
            sorted_co_occurrences = sorted(co_occurrence_counts_for_this_principal.items(), key=lambda item: item[1], reverse=True)
            for i in range(3): 
                neighbor_num = i + 1 
                display_col_name = f'Voisin {neighbor_num}'
                if i < len(sorted_co_occurrences):
                    name, count = sorted_co_occurrences[i]
                    display_row_dict[display_col_name] = f"{str(name).capitalize()} - {count}"
                else: 
                    display_row_dict[display_col_name] = "-"
        else: 
            for neighbor_num in [1, 2, 3]:
                display_row_dict[f'Voisin {neighbor_num}'] = "-"
        cooccurrence_display_list.append(display_row_dict)

    if cooccurrence_display_list:
        cooccurrence_df_for_display = pd.DataFrame(cooccurrence_display_list)
        
        st.markdown("Ce tableau présente, pour chaque espèce de vos relevés (colonne 1), les trois espèces qui lui sont le plus fréquemment associées au sein des listes d'espèces caractéristiques des syntaxons de référence (`data_villaret.csv`). Le nombre après le tiret indique le nombre de syntaxons partagés. La couleur de fond indique l'intensité de cette co-occurrence (du gris foncé au rouge).")
        
        styled_object = cooccurrence_df_for_display.style.apply(
            style_cooccurrence_row_parsing, 
            max_overall_count=max_overall_cooccurrence,
            vmin_count=min_cooccurrence_for_color,
            axis=1, 
            subset=None 
        ).format(na_rep="-")
        
        st.dataframe(styled_object, use_container_width=True)

    else:
        st.info("Aucune donnée de co-occurrence à afficher pour les espèces sélectionnées et les syntaxons disponibles.")

elif st.session_state.run_main_processing_once and not syntaxon_data_list:
    st.markdown("---")
    st.subheader("Étape 4: Analyse des Co-occurrences d'Espèces (basée sur les listes de syntaxons)")
    st.warning("Les données des syntaxons ('data_villaret.csv') n'ont pas pu être chargées, sont vides, ou ne contiennent aucun syntaxon valide. L'analyse des co-occurrences ne peut pas être effectuée.")

