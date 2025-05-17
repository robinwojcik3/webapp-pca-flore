import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff # Gardé au cas où core.py l'utilise encore
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import numpy as np
import textwrap # Importé pour la mise en forme du texte de survol
from collections import defaultdict # Ajouté pour l'analyse de co-occurrence

# Assurez-vous que le fichier core.py est dans le même répertoire ou accessible
# Pour les besoins de cet exemple, si core.py n'est pas disponible,
# nous allons simuler sa fonction analyse.
try:
    import core
except ImportError:
    st.warning("Le module 'core.py' est introuvable. Une fonction d'analyse simulée sera utilisée. L'ACP réelle ne fonctionnera pas.")
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

    def mock_analyse(sub_df_prepared_for_core, n_clusters):
        n_samples = len(sub_df_prepared_for_core)
        
        if n_samples == 0 or sub_df_prepared_for_core.shape[1] <= 1: # Moins de 2 colonnes (Espece + au moins 1 trait)
            mock_pca_obj = MockPCA(n_features_actual=0, n_components_to_simulate=0)
            # X_scaled_data n'est plus utilisé par app.py, mais on le retourne pour la compatibilité de la signature
            return np.array([]), mock_pca_obj, pd.DataFrame(index=sub_df_prepared_for_core.index), np.array([]).reshape(0,1)

        numeric_cols_for_pca_df = sub_df_prepared_for_core.iloc[:, 1:] # Exclure la colonne 'Espece'
        n_features = numeric_cols_for_pca_df.shape[1]

        if n_features == 0: # Aucune colonne numérique pour l'ACP
            mock_pca_obj = MockPCA(n_features_actual=0, n_components_to_simulate=0)
            return np.array([]), mock_pca_obj, pd.DataFrame(index=sub_df_prepared_for_core.index), np.array([]).reshape(0,1)

        n_pcs_to_simulate_coords = min(2, n_features) # Simuler au plus 2 PCs pour les coordonnées
        coords_array = np.random.rand(n_samples, n_pcs_to_simulate_coords) * 10
        pc_cols = [f"PC{i+1}" for i in range(coords_array.shape[1])]
        coords_df = pd.DataFrame(coords_array, columns=pc_cols, index=sub_df_prepared_for_core.index)

        labels = np.array([])
        if n_samples > 0 and n_clusters > 0:
            if n_samples < n_clusters : 
                labels = np.arange(n_samples) # Chaque échantillon est son propre cluster
            else:
                labels = np.random.randint(0, n_clusters, n_samples)
        
        # X_scaled (pour dendrogramme) n'est plus utilisé par app.py
        # On le simule quand même pour la compatibilité de la signature de la fonction mock_analyse
        X_scaled_sim = np.array([]).reshape(n_samples, 0) 
        if not numeric_cols_for_pca_df.empty:
            X_scaled_temp_sim = (numeric_cols_for_pca_df - numeric_cols_for_pca_df.mean()) / numeric_cols_for_pca_df.std()
            X_scaled_sim = X_scaled_temp_sim.fillna(0).values 
        elif n_samples > 0 : # Si pas de traits numériques mais des échantillons, simuler avec une colonne
             X_scaled_sim = np.random.rand(n_samples, 1) 
        if X_scaled_sim.ndim == 1 and n_samples > 0 : # Assurer 2D si 1D
            X_scaled_sim = X_scaled_sim.reshape(-1,1)
        elif X_scaled_sim.size == 0 and n_samples == 0: # Cas où il n'y a aucun échantillon
             X_scaled_sim = np.array([]).reshape(0,n_features if n_features > 0 else 1)


        mock_pca_obj = MockPCA(n_features_actual=n_features, n_components_to_simulate=n_pcs_to_simulate_coords)
        
        # Assurer que coords_df a autant de colonnes que pca_obj.components_ (qui est n_pcs_to_simulate_coords)
        if mock_pca_obj.components_.shape[0] < coords_df.shape[1]:
            coords_df = coords_df.iloc[:, :mock_pca_obj.components_.shape[0]]
            new_pc_cols = [f"PC{i+1}" for i in range(mock_pca_obj.components_.shape[0])]
            coords_df.columns = new_pc_cols
        
        return labels, mock_pca_obj, coords_df, X_scaled_sim # X_scaled_sim est retourné mais non utilisé

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
    color: #e1e1e1 !important;         /* Couleur de texte pour thème sombre */
    font-weight: bold !important;
}
/* Style pour la cellule de la première ligne en mode édition */
div[data-testid="stDataEditor"] .glideDataEditor-body .dvn-scroll-inner > div:first-child > div[data-cell^="[0,"] > div > .gdg-input {
    background-color: #ffffff !important; /* Fond blanc pour l'éditeur */
    color: #000000 !important;         /* Texte noir pour l'éditeur */
    font-weight: normal !important;     /* Poids normal pour l'éditeur */
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
        # Simuler si core.read_reference n'est pas la vraie fonction ou si core n'est pas importé
        if not hasattr(core, "read_reference") or callable(getattr(core, "read_reference", None)) and core.read_reference.__name__ == '<lambda>': 
            st.warning(f"Simulation du chargement de '{file_path}'. Le fichier réel n'est pas utilisé.")
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
            # Ajouter des espèces spécifiques pour des tests potentiels
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

        # Normaliser les noms d'espèces pour la jointure (genre + espèce, minuscule)
        eco_data['Espece_norm'] = (
            eco_data['Espece']
            .str.split()
            .str[:2]
            .str.join(" ")
            .str.lower()
        )
        eco_data = eco_data.drop_duplicates(subset=['Espece_norm'], keep='first') # Garder la première occurrence en cas de doublons normalisés
        eco_data = eco_data.set_index('Espece_norm') # Définir l'index sur les noms normalisés
        return eco_data[["Description_Ecologie"]] # Retourner seulement la description
    
    except FileNotFoundError:
        print(f"AVERTISSEMENT: Fichier de données écologiques '{file_path}' non trouvé.")
        st.toast(f"Fichier écologique '{file_path}' non trouvé.", icon="⚠️")
        return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))
    except pd.errors.EmptyDataError: # Si le fichier est complètement vide
        st.warning(f"Le fichier écologique '{file_path}' est vide.")
        return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))
    except ValueError as ve: # Problèmes de parsing CSV liés aux types/valeurs
        print(f"AVERTISSEMENT: Erreur de valeur lors de la lecture du fichier '{file_path}'. Détails: {ve}.")
        st.toast(f"Erreur format fichier écologique '{file_path}'.", icon="🔥")
        return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))
    except Exception as e: # Autres erreurs
        print(f"AVERTISSEMENT: Impossible de charger les données écologiques depuis '{file_path}': {e}.")
        st.toast(f"Erreur chargement fichier écologique.", icon=" ")
        return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))

ecology_df = load_ecology_data()


# ---------------------------------------------------------------------------- #
# FONCTION UTILITAIRE POUR NORMALISER LES NOMS D'ESPECES (pour data_villaret)
# ---------------------------------------------------------------------------- #
def normalize_species_name_for_villaret(species_name):
    """Normalise un nom d'espèce en prenant les deux premiers mots et en convertissant en minuscules."""
    if pd.isna(species_name) or str(species_name).strip() == "":
        return None
    return " ".join(str(species_name).strip().split()[:2]).lower()

# ---------------------------------------------------------------------------- #
# CHARGEMENT DES DONNÉES DES SYNTAXONS (POUR CO-OCCURRENCE)
# ---------------------------------------------------------------------------- #
@st.cache_data
def load_syntaxon_data(file_path="data_villaret.csv"):
    """Charge et prétraite les données des syntaxons à partir d'un fichier CSV."""
    try:
        # Lire le CSV ; pas d'en-tête, séparateur point-virgule
        df = pd.read_csv(file_path, sep=';', header=None, encoding='utf-8-sig', keep_default_na=False, na_values=[''])
        if df.empty:
            st.warning(f"Le fichier des syntaxons '{file_path}' est vide.")
            return []

        processed_syntaxons = []
        # Itérer sur chaque ligne (chaque syntaxon)
        for index, row in df.iterrows():
            # Colonne 0: ID du syntaxon, Colonne 1: Nom du syntaxon
            syntaxon_id = str(row.iloc[0]).strip()
            syntaxon_name = str(row.iloc[1]).strip()
            
            species_in_row_set = set()
            # Les espèces commencent à partir de la 3ème colonne (index 2)
            for species_cell_value in row.iloc[2:]:
                normalized_species = normalize_species_name_for_villaret(species_cell_value)
                if normalized_species: # Ajouter seulement si le nom normalisé est valide
                    species_in_row_set.add(normalized_species)
            
            # Ajouter le syntaxon seulement s'il a un ID, un nom, et au moins une espèce caractéristique
            if syntaxon_id and syntaxon_name and species_in_row_set:
                processed_syntaxons.append({
                    'id': syntaxon_id,
                    'name': syntaxon_name,
                    'species_set': species_in_row_set # Ensemble des espèces normalisées pour ce syntaxon
                })
        
        if not processed_syntaxons:
            st.warning(f"Aucun syntaxon valide (avec ID, nom et espèces) n'a été trouvé dans '{file_path}'.")
        return processed_syntaxons
        
    except FileNotFoundError:
        st.error(f"ERREUR CRITIQUE: Fichier des syntaxons '{file_path}' non trouvé. L'analyse de co-occurrence ne pourra pas être effectuée.")
        return []
    except pd.errors.EmptyDataError: # Si le fichier CSV est complètement vide
        st.warning(f"Le fichier des syntaxons '{file_path}' est vide (EmptyDataError).")
        return []
    except Exception as e: # Capturer d'autres erreurs potentielles
        st.error(f"ERREUR CRITIQUE: Impossible de charger les données des syntaxons depuis '{file_path}': {e}")
        return []

syntaxon_data_list = load_syntaxon_data() # Charger les données ici pour qu'elles soient disponibles globalement


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
    'numeric_trait_names_for_interactive_plot': [],
    'selected_habitats_indices': [], 
    'previous_num_cols': 0, # Pour suivre les changements dans le nombre de colonnes de l'éditeur de relevés
    'analysis_has_run_for_current_selection': False, # Pour éviter de relancer l'analyse si rien n'a changé
    'n_clusters_slider_main_value': 3 # Valeur par défaut pour le slider du nombre de clusters
}

for key, value in default_session_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialisation du DataFrame des relevés avec des placeholders s'il n'existe pas
if 'releves_df' not in st.session_state or not isinstance(st.session_state.releves_df, pd.DataFrame):
    num_placeholder_cols = 15 # Nombre de colonnes de relevés par défaut
    num_placeholder_rows_total = 11 # Nombre de lignes (1 pour nom habitat + 10 pour espèces)
    # Créer la ligne d'en-tête (noms des habitats)
    header = [f"Habitat {j+1}" for j in range(num_placeholder_cols)] 
    # Créer les lignes de données vides pour les espèces
    placeholder_rows = [["" for _ in range(num_placeholder_cols)] for _ in range(num_placeholder_rows_total -1)]
    st.session_state.releves_df = pd.DataFrame([header] + placeholder_rows)
    # S'assurer que les noms de colonnes sont des chaînes (important pour st.data_editor)
    st.session_state.releves_df.columns = [str(col) for col in st.session_state.releves_df.columns] 
    st.session_state.previous_num_cols = num_placeholder_cols


# ---------------------------------------------------------------------------- #
# ÉTAPE 1: IMPORTATION ET SÉLECTION DES RELEVÉS FLORISTIQUES
# ---------------------------------------------------------------------------- #
st.markdown("---") # Ligne de séparation visuelle
st.subheader("Étape 1: Importation et Sélection des Relevés Floristiques")

st.info("Copiez-collez vos données de relevés ci-dessus (Ctrl+V ou Cmd+V). La première ligne doit contenir les noms des habitats/relevés. Les lignes suivantes contiennent les espèces.")

# S'assurer que les noms de colonnes sont des chaînes avant d'utiliser st.data_editor
if not all(isinstance(col, str) for col in st.session_state.releves_df.columns):
    st.session_state.releves_df.columns = [str(col) for col in st.session_state.releves_df.columns]

# Éditeur de données pour les relevés floristiques
edited_releves_df_from_editor = st.data_editor(
    st.session_state.releves_df,
    num_rows="dynamic", # Permettre à l'utilisateur d'ajouter/supprimer des lignes
    use_container_width=True,
    key="releves_data_editor_key" # Clé unique pour l'éditeur
)

# Si les données ont été modifiées dans l'éditeur
if not edited_releves_df_from_editor.equals(st.session_state.releves_df):
    st.session_state.releves_df = edited_releves_df_from_editor.copy() # Mettre à jour l'état de session
    # Si le nombre de colonnes a changé, ajuster les indices sélectionnés et réinitialiser l'analyse
    if len(st.session_state.releves_df.columns) != st.session_state.previous_num_cols:
        current_max_col_index = len(st.session_state.releves_df.columns) - 1
        # Conserver uniquement les indices de colonnes sélectionnés qui sont encore valides
        st.session_state.selected_habitats_indices = [
            idx for idx in st.session_state.selected_habitats_indices if idx <= current_max_col_index
        ]
        # Si plus aucun habitat n'est sélectionné (ou si la liste est vide), réinitialiser l'état de l'analyse
        if not st.session_state.selected_habitats_indices: 
            st.session_state.analysis_has_run_for_current_selection = False
            st.session_state.run_main_analysis_once = False 
        st.session_state.previous_num_cols = len(st.session_state.releves_df.columns) # Mettre à jour le nombre de colonnes précédent
    st.rerun() # Relancer le script pour refléter les changements

current_releves_df_for_selection = st.session_state.releves_df.copy() 

# Vérifier si le DataFrame des relevés est prêt pour la sélection des habitats
if not current_releves_df_for_selection.empty and \
   len(current_releves_df_for_selection.columns) > 0 and \
   len(current_releves_df_for_selection) > 0:
    
    # Extraire les noms des habitats de la première ligne du DataFrame
    habitat_names_from_df = current_releves_df_for_selection.iloc[0].astype(str).tolist()
    num_actual_cols = len(current_releves_df_for_selection.columns) # Nombre actuel de colonnes
    
    st.markdown("**Cliquez sur le nom d'un habitat ci-dessous pour le sélectionner/désélectionner pour l'analyse :**") 
    
    # S'assurer que les indices d'habitats sélectionnés sont valides par rapport au nombre actuel de colonnes
    st.session_state.selected_habitats_indices = [
        idx for idx in st.session_state.selected_habitats_indices if idx < num_actual_cols
    ]

    # Filtrer les colonnes pour ne montrer que celles avec des données d'espèces valides
    valid_habitat_buttons_info = []
    for i in range(num_actual_cols):
        # Vérifier si la colonne (à partir de la deuxième ligne) contient des données d'espèces
        # (non-NA, non-vide après strip)
        species_in_col = current_releves_df_for_selection.iloc[1:, i].dropna().astype(str).str.strip().replace('', np.nan).dropna()
        if not species_in_col.empty: # Si la colonne contient au moins une espèce
            # Utiliser le nom de l'habitat de la première ligne, ou un nom par défaut si vide/NA
            habitat_name_for_button = habitat_names_from_df[i] if pd.notna(habitat_names_from_df[i]) and str(habitat_names_from_df[i]).strip() != "" else f"Relevé {i+1}"
            valid_habitat_buttons_info.append({'index': i, 'name': habitat_name_for_button})

    if valid_habitat_buttons_info: # Si au moins un habitat valide avec des espèces existe
        num_buttons_to_show = len(valid_habitat_buttons_info)
        button_cols_layout = st.columns(num_buttons_to_show) # Créer des colonnes pour les boutons
        
        # Créer un bouton pour chaque habitat valide
        for k, habitat_info in enumerate(valid_habitat_buttons_info):
            col_idx = habitat_info['index'] # Index original de la colonne dans le DataFrame
            habitat_name_display = habitat_info['name'] # Nom à afficher sur le bouton
            is_selected = (col_idx in st.session_state.selected_habitats_indices) # Statut de sélection
            
            button_type = "primary" if is_selected else "secondary" # Style du bouton
            button_key = f"habitat_select_button_{col_idx}" # Clé unique pour le bouton

            with button_cols_layout[k]: # Placer le bouton dans sa colonne
                st.markdown(f'<div class="habitat-select-button">', unsafe_allow_html=True)
                if st.button(habitat_name_display, key=button_key, type=button_type, use_container_width=True):
                    if is_selected:
                        st.session_state.selected_habitats_indices.remove(col_idx) # Désélectionner
                    else:
                        st.session_state.selected_habitats_indices.append(col_idx) # Sélectionner
                    # Réinitialiser l'état de l'analyse car la sélection a changé
                    st.session_state.run_main_analysis_once = False 
                    st.session_state.analysis_has_run_for_current_selection = False 
                    st.rerun() # Relancer pour mettre à jour l'UI et potentiellement l'analyse
                st.markdown('</div>', unsafe_allow_html=True)
    elif num_actual_cols > 0 : # Il y a des colonnes, mais aucune avec des données d'espèces
        st.info("Aucune colonne ne contient de données d'espèces pour la sélection. Veuillez ajouter des espèces sous les noms d'habitats.")
    else: # Aucune colonne du tout
        st.info("Ajoutez des colonnes au tableau pour pouvoir sélectionner des relevés.")
else:
    st.warning("Le tableau de données est vide ou ne contient pas de colonnes pour la sélection.")


# Initialisation des figures (évite les erreurs si elles ne sont pas créées)
fig_pca = None 
# fig_dend = None # SUPPRIMÉ: Dendrogramme retiré

# Condition pour lancer l'analyse principale
if st.session_state.selected_habitats_indices and \
   not ref.empty and \
   not st.session_state.get('analysis_has_run_for_current_selection', False): # Si des habitats sont sélectionnés, ref est chargé, et l'analyse n'a pas encore tourné pour cette sélection

    st.session_state.run_main_analysis_once = True # Indiquer que l'analyse va être (ou a été) lancée
    st.session_state.analysis_has_run_for_current_selection = True # Marquer que l'analyse pour cette sélection est faite

    # Réinitialiser les DataFrames de résultats de session
    st.session_state.sub = pd.DataFrame() # DataFrame des espèces sélectionnées avec leurs traits
    st.session_state.pdf = pd.DataFrame() # DataFrame des coordonnées PCA et clusters
    st.session_state.vip_data_df_interactive = pd.DataFrame() # Pour le tableau d'exploration interactif
    st.session_state.numeric_trait_names_for_interactive_plot = [] # Liste des traits numériques

    all_species_data_for_analysis = [] # Pour stocker les données des espèces trouvées
    species_not_found_in_ref_detailed = {} # Pour suivre les espèces non trouvées par habitat

    df_for_species_extraction = st.session_state.releves_df.copy() 
    # Noms des habitats à partir de la première ligne du DataFrame des relevés
    habitat_names_from_header = df_for_species_extraction.iloc[0].astype(str).tolist() if not df_for_species_extraction.empty else []

    # Itérer sur les indices des habitats sélectionnés par l'utilisateur
    for habitat_idx in st.session_state.selected_habitats_indices:
        if habitat_idx < len(df_for_species_extraction.columns): # Vérifier la validité de l'index
            # Déterminer le nom de l'habitat
            habitat_name = habitat_names_from_header[habitat_idx] if habitat_idx < len(habitat_names_from_header) and pd.notna(habitat_names_from_header[habitat_idx]) and str(habitat_names_from_header[habitat_idx]).strip() != "" else f"Relevé {habitat_idx+1}"
            # Extraire les espèces de la colonne de l'habitat (à partir de la 2ème ligne)
            species_in_col_series = df_for_species_extraction.iloc[1:, habitat_idx]
            # Nettoyer la liste des espèces : supprimer NA, convertir en str, strip, remplacer vide par NA, puis supprimer NA
            species_raw_in_current_habitat = species_in_col_series.dropna().astype(str).str.strip().replace('', np.nan).dropna().tolist()
            
            species_not_found_in_ref_detailed[habitat_name] = [] # Initialiser la liste des non-trouvées pour cet habitat

            if not species_raw_in_current_habitat: # Si aucune espèce n'est listée pour cet habitat
                st.warning(f"Aucune espèce listée dans l'habitat sélectionné : {habitat_name}")
                continue # Passer à l'habitat suivant

            # Pour chaque espèce brute extraite de l'habitat
            for raw_species_name in species_raw_in_current_habitat:
                if not raw_species_name or len(raw_species_name.split()) == 0: # Ignorer si nom vide ou invalide
                    continue
                
                # Normaliser le nom de l'espèce (genre + espèce, minuscule) pour la comparaison
                binom_species_name = " ".join(raw_species_name.split()[:2]).lower()
                
                if not ref_binom_series.empty: # Si la série de noms binomiaux de référence existe
                    # Chercher une correspondance dans la base de référence
                    match_in_ref = ref_binom_series[ref_binom_series == binom_species_name]
                    if not match_in_ref.empty: # Si une correspondance est trouvée
                        ref_idx = match_in_ref.index[0] # Index de l'espèce dans le DataFrame 'ref'
                        trait_data = ref.loc[ref_idx].to_dict() # Obtenir les données de traits
                        trait_data['Source_Habitat'] = habitat_name # Ajouter l'habitat d'origine
                        trait_data['Espece_Ref_Original'] = ref.loc[ref_idx, 'Espece'] # Nom original de référence
                        trait_data['Espece_User_Input_Raw'] = raw_species_name # Nom entré par l'utilisateur
                        all_species_data_for_analysis.append(trait_data)
                    else: # Si non trouvée dans la référence
                        species_not_found_in_ref_detailed[habitat_name].append(raw_species_name)
                else: # Si ref_binom_series est vide (ne devrait pas arriver si 'ref' est chargé)
                     species_not_found_in_ref_detailed[habitat_name].append(raw_species_name)

    if not all_species_data_for_analysis: # Si aucune espèce correspondante n'a été trouvée au total
        st.error("Aucune espèce valide correspondante aux traits n'a été trouvée dans les relevés sélectionnés. Vérifiez vos données et sélections.")
        st.session_state.run_main_analysis_once = False 
        st.session_state.analysis_has_run_for_current_selection = False 
    else: # Si des espèces ont été trouvées et leurs traits récupérés
        st.session_state.sub = pd.DataFrame(all_species_data_for_analysis) # Créer le DataFrame 'sub'
        st.session_state.sub.reset_index(drop=True, inplace=True) # Réinitialiser l'index

        # Afficher les avertissements pour les espèces non trouvées
        for habitat_name, not_found_list in species_not_found_in_ref_detailed.items():
            if not_found_list:
                st.warning(f"Espèces de '{habitat_name}' non trouvées dans la base de traits : " + ", ".join(not_found_list), icon="⚠️")

        # Récupérer le nombre de clusters sélectionné par l'utilisateur
        n_clusters_selected_main = st.session_state.get('n_clusters_slider_main_value', 3) 

        # Vérifications avant de lancer core.analyse
        if st.session_state.sub.empty: 
            st.error("Aucune des espèces sélectionnées n'a été trouvée dans la base de traits. L'analyse ne peut continuer.")
            st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;
        elif st.session_state.sub.shape[0] < n_clusters_selected_main and n_clusters_selected_main > 0 :
            st.error(f"Nombre total d'instances d'espèces trouvées ({st.session_state.sub.shape[0]}) < clusters demandés ({n_clusters_selected_main}). Ajustez le nombre de clusters ou vérifiez les espèces.");
            st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;
        elif st.session_state.sub.shape[0] < 2: # Besoin d'au moins 2 échantillons pour l'ACP
            st.error(f"Au moins 2 instances d'espèces (total sur les habitats) sont nécessaires pour l'analyse. {st.session_state.sub.shape[0]} trouvée(s).");
            st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;
        else: # Si tout est OK, procéder à l'analyse
            try:
                if ref.empty: # Double vérification, bien que déjà gérée plus haut
                    st.error("Le DataFrame de référence 'ref' est vide. Impossible de déterminer les traits numériques.")
                    st.session_state.run_main_analysis_once = False
                    st.session_state.analysis_has_run_for_current_selection = False
                    raise ValueError("DataFrame 'ref' vide, impossible de préparer les données pour core.analyse.")

                # Identifier les traits numériques à partir du DataFrame de référence 'ref'
                numeric_trait_names_from_ref = ref.select_dtypes(include=np.number).columns.tolist()
                df_for_core_preparation = st.session_state.sub.copy() # Copie de 'sub' pour la préparation

                # Déterminer la colonne d'identification de l'espèce pour core.analyse
                if 'Espece_Ref_Original' in df_for_core_preparation.columns:
                    df_for_core_preparation['Espece_ID_Core'] = df_for_core_preparation['Espece_Ref_Original']
                elif 'Espece' in df_for_core_preparation.columns: 
                    df_for_core_preparation['Espece_ID_Core'] = df_for_core_preparation['Espece']
                else: # Si aucune colonne d'identification n'est trouvée
                    st.error("Colonne d'identification 'Espece' ou 'Espece_Ref_Original' manquante pour l'analyse.")
                    raise ValueError("Identifiant 'Espece' manquant pour core.analyse.")

                # Sélectionner les traits numériques qui sont effectivement présents dans 'sub' (et donc dans df_for_core_preparation)
                actual_numeric_traits_for_pca = [
                    trait for trait in numeric_trait_names_from_ref if trait in df_for_core_preparation.columns
                ]
                
                # Préparer le DataFrame final pour l'appel à core.analyse
                columns_for_core_call = ['Espece_ID_Core'] + actual_numeric_traits_for_pca
                sub_for_analysis_call_prepared = df_for_core_preparation[columns_for_core_call]
                # Renommer la colonne d'ID en 'Espece' comme attendu par core.analyse (selon la simulation)
                sub_for_analysis_call_prepared = sub_for_analysis_call_prepared.rename(columns={'Espece_ID_Core': 'Espece'})

                if not actual_numeric_traits_for_pca: # Si aucun trait numérique n'est disponible
                    st.error("Aucun trait numérique disponible pour l'ACP après filtrage. L'analyse est impossible.")
                    st.session_state.run_main_analysis_once = False
                    st.session_state.analysis_has_run_for_current_selection = False
                    raise ValueError("Aucun trait numérique pour l'ACP.")
                else:
                    # Appel à la fonction d'analyse (réelle ou simulée)
                    # X_scaled_data n'est plus assigné à st.session_state.X_for_dendro
                    labels, pca_results, coords_df_from_core, _ = core.analyse(sub_for_analysis_call_prepared, n_clusters_selected_main)
                    
                    # Vérifier et convertir coords_df_from_core si c'est un NumPy array
                    if not isinstance(coords_df_from_core, pd.DataFrame):
                        if isinstance(coords_df_from_core, np.ndarray) and \
                           coords_df_from_core.ndim == 2 and \
                           coords_df_from_core.shape[0] == len(sub_for_analysis_call_prepared):
                            num_pcs = coords_df_from_core.shape[1]
                            pc_cols = [f"PC{i+1}" for i in range(num_pcs)]
                            coords_df = pd.DataFrame(coords_df_from_core, columns=pc_cols, index=sub_for_analysis_call_prepared.index)
                            st.info("Les coordonnées PCA (coords_df) ont été converties de NumPy array en DataFrame.")
                        else:
                            st.error("Format des coordonnées PCA (coords_df) inattendu après retour de core.analyse. Doit être un DataFrame ou un NumPy array 2D compatible.")
                            st.session_state.run_main_analysis_once = False
                            st.session_state.analysis_has_run_for_current_selection = False
                            raise TypeError("Format coords_df incorrect depuis core.analyse.")
                    else: 
                        coords_df = coords_df_from_core 
                    
                    # Vérifier la cohérence de l'index après la conversion potentielle
                    if not coords_df.index.equals(sub_for_analysis_call_prepared.index):
                        st.error("L'index des coordonnées PCA (coords_df) ne correspond pas aux données d'entrée après conversion/vérification. L'alignement des données a échoué.")
                        if len(coords_df) == len(sub_for_analysis_call_prepared): 
                            coords_df.index = sub_for_analysis_call_prepared.index # Forcer le réalignement si les longueurs correspondent
                            st.warning("Index des coordonnées PCA (coords_df) forcé au réalignement.")
                        else: # Si les longueurs ne correspondent pas, c'est un problème plus grave
                            st.session_state.run_main_analysis_once = False
                            st.session_state.analysis_has_run_for_current_selection = False
                            # Informations de débogage
                            print(f"coords_df length: {len(coords_df)}, sub_for_analysis_call_prepared length: {len(sub_for_analysis_call_prepared)}")
                            print(f"coords_df index: {coords_df.index}")
                            print(f"sub_for_analysis_call_prepared index: {sub_for_analysis_call_prepared.index}")
                            raise ValueError("Incohérence de longueur d'index PCA persistante, impossible de réaligner.")

                    # Si l'analyse s'est bien déroulée jusqu'ici
                    if st.session_state.run_main_analysis_once: 
                        current_pdf = coords_df.copy() # DataFrame pour les résultats PCA
                        if not current_pdf.empty:
                            # Ajouter les labels de cluster (s'ils existent et correspondent en longueur)
                            if len(labels) == len(current_pdf): current_pdf["Cluster"] = labels.astype(str)
                            else: current_pdf["Cluster"] = np.zeros(len(current_pdf)).astype(str) if len(current_pdf) > 0 else pd.Series(dtype=str) # Fallback
                            
                            # Ajouter les informations d'habitat, d'espèce de référence, et d'espèce utilisateur
                            # Utiliser .loc pour s'assurer de l'alignement correct des index
                            current_pdf["Source_Habitat"] = st.session_state.sub.loc[current_pdf.index, "Source_Habitat"]
                            current_pdf["Espece_Ref"] = st.session_state.sub.loc[current_pdf.index, "Espece_Ref_Original"]
                            current_pdf["Espece_User"] = st.session_state.sub.loc[current_pdf.index, "Espece_User_Input_Raw"]

                            # Ajouter les informations écologiques
                            if not ecology_df.empty:
                                # Normaliser 'Espece_Ref' pour la jointure avec ecology_df
                                current_pdf['Espece_Ref_norm_for_eco'] = current_pdf['Espece_Ref'].astype(str).str.strip().str.split().str[:2].str.join(" ").str.lower()
                                if ecology_df.index.name == 'Espece_norm' and 'Description_Ecologie' in ecology_df.columns:
                                    current_pdf['Ecologie_raw'] = current_pdf['Espece_Ref_norm_for_eco'].map(ecology_df['Description_Ecologie'])
                                else: # Fallback si la structure de ecology_df n'est pas celle attendue
                                    current_pdf['Ecologie_raw'] = pd.Series([np.nan] * len(current_pdf), index=current_pdf.index)
                                current_pdf['Ecologie'] = current_pdf['Ecologie_raw'].apply(lambda x: format_ecology_for_hover(x))
                                current_pdf['Ecologie'] = current_pdf['Ecologie'].fillna(format_ecology_for_hover(None)) # S'assurer qu'il n'y a pas de NaN
                            else: # Si ecology_df est vide
                                current_pdf['Ecologie'] = format_ecology_for_hover(None) 
                            st.session_state.pdf = current_pdf.copy() # Stocker le DataFrame PCA final

                            # Calcul des communalités (contribution des variables aux axes PCA)
                            if hasattr(pca_results, 'components_') and hasattr(pca_results, 'explained_variance_') and \
                               isinstance(pca_results.components_, np.ndarray) and isinstance(pca_results.explained_variance_, np.ndarray) and \
                               pca_results.components_.size > 0 and pca_results.explained_variance_.size > 0 :
                                
                                pca_components_values = pca_results.components_ # Vecteurs propres (lignes = composantes, colonnes = variables)
                                explained_variance_values = pca_results.explained_variance_ # Variance expliquée par chaque composante
                                # Les loadings sont les corrélations entre les variables originales et les composantes principales.
                                # loadings = eigenvectors * sqrt(eigenvalues)
                                # Ici, pca_results.components_ sont les eigenvectors. explained_variance_ sont les eigenvalues.
                                eigenvectors_matrix = pca_components_values.T # Transposer pour avoir variables en lignes
                                sqrt_eigenvalues_vector = np.sqrt(explained_variance_values) 
                                loadings = eigenvectors_matrix * sqrt_eigenvalues_vector 
                                communal = (loadings**2).sum(axis=1) # Somme des carrés des loadings par variable
                                
                                trait_columns_for_communal = actual_numeric_traits_for_pca # Noms des traits numériques utilisés
                                
                                if len(communal) == len(trait_columns_for_communal):
                                    communal_percent = (communal * 100).round(0).astype(int) # En pourcentage
                                    communal_percent_clipped = np.clip(communal_percent, 0, 100) # S'assurer que c'est entre 0 et 100
                                    
                                    st.session_state.vip_data_df_for_calc = pd.DataFrame({
                                        "Variable": trait_columns_for_communal,
                                        "Communalité (%)": communal_percent_clipped, 
                                    }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
                                else: 
                                    st.session_state.vip_data_df_for_calc = pd.DataFrame(columns=["Variable", "Communalité (%)"])
                                    st.warning(f"Communalités non calculées (dimensions des loadings/traits incohérentes: {len(communal)} vs {len(trait_columns_for_communal)}).")
                            else: # Si les résultats PCA sont incomplets
                                st.session_state.vip_data_df_for_calc = pd.DataFrame(columns=["Variable", "Communalité (%)"])
                                st.warning("Résultats PCA incomplets pour communalités (components_ ou explained_variance_ manquants/incorrects).")
                            
                            st.session_state.numeric_trait_names_for_interactive_plot = actual_numeric_traits_for_pca
                            
                            # Définir les axes par défaut pour le graphique interactif (basé sur les communalités)
                            default_x_init, default_y_init = None, None
                            if not st.session_state.vip_data_df_for_calc.empty and actual_numeric_traits_for_pca: 
                                # Variables triées par communalité, présentes parmi les traits numériques
                                top_vars_from_vip_numeric = [var for var in st.session_state.vip_data_df_for_calc["Variable"].tolist() if var in actual_numeric_traits_for_pca]
                                if len(top_vars_from_vip_numeric) >= 1: default_x_init = top_vars_from_vip_numeric[0]
                                if len(top_vars_from_vip_numeric) >= 2: default_y_init = top_vars_from_vip_numeric[1]
                                elif len(top_vars_from_vip_numeric) == 1: # Si une seule variable avec communalité
                                    other_numeric_traits = [t for t in actual_numeric_traits_for_pca if t != default_x_init]
                                    default_y_init = other_numeric_traits[0] if other_numeric_traits else default_x_init # Utiliser une autre ou la même
                            
                            # Fallback si les communalités n'ont pas aidé
                            if default_x_init is None and actual_numeric_traits_for_pca: default_x_init = actual_numeric_traits_for_pca[0]
                            if default_y_init is None:
                                if len(actual_numeric_traits_for_pca) >= 2: default_y_init = actual_numeric_traits_for_pca[1]
                                elif default_x_init and len(actual_numeric_traits_for_pca) == 1: default_y_init = default_x_init # Si un seul trait, utiliser pour X et Y

                            st.session_state.x_axis_trait_interactive = default_x_init
                            st.session_state.y_axis_trait_interactive = default_y_init
                            
                            # Préparer le DataFrame pour l'éditeur interactif des axes
                            if not st.session_state.vip_data_df_for_calc.empty and actual_numeric_traits_for_pca:
                                temp_interactive_df = st.session_state.vip_data_df_for_calc[st.session_state.vip_data_df_for_calc["Variable"].isin(actual_numeric_traits_for_pca)].copy()
                                temp_interactive_df["Axe X"] = temp_interactive_df["Variable"] == st.session_state.x_axis_trait_interactive
                                temp_interactive_df["Axe Y"] = temp_interactive_df["Variable"] == st.session_state.y_axis_trait_interactive
                                st.session_state.vip_data_df_interactive = temp_interactive_df[["Variable", "Communalité (%)", "Axe X", "Axe Y"]].reset_index(drop=True)
                            else: # Si pas de communalités ou pas de traits numériques
                                st.session_state.vip_data_df_interactive = pd.DataFrame(columns=["Variable", "Communalité (%)", "Axe X", "Axe Y"])
                            st.session_state.vip_data_df_interactive_snapshot_for_comparison = st.session_state.vip_data_df_interactive.copy() # Snapshot pour détecter les changements
                        else: # Si current_pdf est vide (coords_df était vide)
                            st.warning("L'analyse n'a pas produit de coordonnées PCA (coords_df vide ou invalide).")
                            st.session_state.run_main_analysis_once = False 
                            st.session_state.analysis_has_run_for_current_selection = False
            except Exception as e: # Erreur globale pendant l'analyse
                st.error(f"Erreur lors de l'analyse principale : {e}"); st.exception(e)
                st.session_state.run_main_analysis_once = False; st.session_state.analysis_has_run_for_current_selection = False;

# Messages si l'analyse n'a pas été lancée
elif not st.session_state.selected_habitats_indices and not ref.empty:
    st.info("Veuillez sélectionner un ou plusieurs habitats à l'Étape 1 pour lancer l'analyse.")
elif ref.empty: # Si les données de référence n'ont pas pu être chargées
    st.warning("Les données de référence ('data_ref.csv') n'ont pas pu être chargées ou sont simulées. L'analyse est désactivée si les données réelles manquent.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 2: EXPLORATION INTERACTIVE DES VARIABLES ET PARAM ACP
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty: # Si l'analyse a tourné et 'sub' n'est pas vide
    st.markdown("---"); st.subheader("Étape 2: Exploration Interactive et Paramètres ACP")
    col_interactive_table, col_interactive_graph = st.columns([1, 2]) # Layout en deux colonnes

    with col_interactive_table: # Colonne de gauche : tableau et slider
        st.markdown("##### Tableau d'exploration interactif")
        df_editor_source_interactive = st.session_state.get('vip_data_df_interactive', pd.DataFrame())

        if not df_editor_source_interactive.empty:
            # S'assurer que le snapshot est à jour si la structure des colonnes change
            snapshot_cols = list(st.session_state.get('vip_data_df_interactive_snapshot_for_comparison', pd.DataFrame()).columns)
            current_cols = list(df_editor_source_interactive.columns)
            if 'vip_data_df_interactive_snapshot_for_comparison' not in st.session_state or snapshot_cols != current_cols:
                st.session_state.vip_data_df_interactive_snapshot_for_comparison = df_editor_source_interactive.copy()

            # Éditeur pour sélectionner les variables des axes X et Y du graphique interactif
            edited_df_interactive = st.data_editor(
                df_editor_source_interactive, 
                column_config={
                    "Variable": st.column_config.TextColumn("Variable", disabled=True), # Non modifiable
                    "Communalité (%)": st.column_config.NumberColumn("Communalité (%)", format="%d%%", disabled=True), # Non modifiable
                    "Axe X": st.column_config.CheckboxColumn("Axe X"), # Sélection pour l'axe X
                    "Axe Y": st.column_config.CheckboxColumn("Axe Y")  # Sélection pour l'axe Y
                }, 
                key="interactive_exploration_editor", 
                use_container_width=True, 
                hide_index=True, # Cacher l'index du DataFrame
                num_rows="fixed" # Empêcher l'ajout/suppression de lignes ici
            )
            
            made_change_in_interactive_axes = False # Flag pour détecter si un rerun est nécessaire

            # Logique de mise à jour de l'axe X basé sur l'éditeur
            current_x_selection_from_state = st.session_state.x_axis_trait_interactive
            x_vars_checked_in_editor = edited_df_interactive[edited_df_interactive["Axe X"]]["Variable"].tolist()
            
            new_x_selection_candidate = current_x_selection_from_state 

            if not x_vars_checked_in_editor: # Si aucune case X n'est cochée
                if current_x_selection_from_state is not None: 
                    new_x_selection_candidate = None # Désélectionner l'axe X
                    made_change_in_interactive_axes = True
            elif len(x_vars_checked_in_editor) == 1: # Si une seule case X est cochée
                single_checked_x = x_vars_checked_in_editor[0]
                if single_checked_x != current_x_selection_from_state:
                    new_x_selection_candidate = single_checked_x # Mettre à jour l'axe X
                    made_change_in_interactive_axes = True
            else: # Si plusieurs cases X sont cochées (prioriser la nouvelle sélection)
                # Prendre la première nouvelle sélection qui n'est pas l'actuelle, ou la dernière cochée
                potential_new_x_selections = [v for v in x_vars_checked_in_editor if v != current_x_selection_from_state]
                if potential_new_x_selections:
                    new_x_selection_candidate = potential_new_x_selections[0] 
                else: # Si toutes les cochées sont l'actuelle (ne devrait pas arriver si une seule est permise logiquement)
                    new_x_selection_candidate = x_vars_checked_in_editor[-1] 
                made_change_in_interactive_axes = True 

            st.session_state.x_axis_trait_interactive = new_x_selection_candidate

            # Logique de mise à jour de l'axe Y (similaire à X)
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

            # Si les sélections d'axes ont changé, mettre à jour le DataFrame de l'éditeur et relancer
            if made_change_in_interactive_axes:
                df_updated_for_editor = df_editor_source_interactive.copy() 
                # Mettre à jour les colonnes "Axe X" et "Axe Y" pour ne refléter qu'une seule sélection
                df_updated_for_editor["Axe X"] = (df_updated_for_editor["Variable"] == st.session_state.x_axis_trait_interactive)
                df_updated_for_editor["Axe Y"] = (df_updated_for_editor["Variable"] == st.session_state.y_axis_trait_interactive)
                
                st.session_state.vip_data_df_interactive = df_updated_for_editor 
                st.session_state.vip_data_df_interactive_snapshot_for_comparison = df_updated_for_editor.copy() 
                st.rerun() # Relancer pour que le graphique interactif se mette à jour
            # Si les données ont changé d'une autre manière (ne devrait pas arriver avec num_rows="fixed" et disabled cols)
            elif not edited_df_interactive.equals(st.session_state.vip_data_df_interactive_snapshot_for_comparison):
                 st.session_state.vip_data_df_interactive_snapshot_for_comparison = edited_df_interactive.copy() # Mettre à jour le snapshot
        else: 
            st.info("Le tableau d'exploration sera disponible après l'analyse si des traits numériques sont identifiés.")

        st.markdown("---") 
        st.markdown("##### Paramètres ACP")
        # Slider pour le nombre de clusters
        n_clusters_selected_val = st.slider(
            "Nombre de clusters (pour ACP)", 2, 8, # Min, Max
            value=st.session_state.get('n_clusters_slider_main_value', 3), # Valeur actuelle/défaut
            key="n_clusters_slider_main_key_moved", 
            disabled=ref.empty or st.session_state.get('sub', pd.DataFrame()).empty, # Désactivé si pas de données
            help="Choisissez le nombre de groupes à former lors de l'Analyse en Composantes Principales."
        )
        # Si la valeur du slider change, réinitialiser l'analyse et relancer
        if n_clusters_selected_val != st.session_state.get('n_clusters_slider_main_value', 3):
            st.session_state.n_clusters_slider_main_value = n_clusters_selected_val
            st.session_state.analysis_has_run_for_current_selection = False # Forcer la relance de l'analyse
            st.rerun()


    with col_interactive_graph: # Colonne de droite : graphique interactif
        st.markdown("##### Graphique d'exploration")
        x_axis_plot = st.session_state.x_axis_trait_interactive # Trait pour l'axe X
        y_axis_plot = st.session_state.y_axis_trait_interactive # Trait pour l'axe Y
        numeric_traits_plot = st.session_state.get('numeric_trait_names_for_interactive_plot', []) # Liste des traits numériques disponibles
        sub_plot = st.session_state.get('sub', pd.DataFrame()) # Données de traits des espèces sélectionnées
        pdf_plot = st.session_state.get('pdf', pd.DataFrame()) # Données PCA

        # Vérifications avant de tracer le graphique interactif
        if not numeric_traits_plot: st.warning("Aucun trait numérique trouvé pour l'exploration interactive.")
        elif not x_axis_plot or not y_axis_plot: st.info("Veuillez sélectionner une variable pour l'Axe X et une pour l'Axe Y dans le tableau à gauche.")
        elif x_axis_plot not in numeric_traits_plot or y_axis_plot not in numeric_traits_plot: st.warning("Une ou les deux variables sélectionnées ne sont plus valides. Veuillez re-sélectionner.")
        elif sub_plot.empty or pdf_plot.empty or x_axis_plot not in sub_plot.columns or y_axis_plot not in sub_plot.columns: 
            st.warning("Données pour le graphique interactif non prêtes, incohérentes ou variables sélectionnées non trouvées. Vérifiez l'analyse principale.")
        elif not pdf_plot.index.equals(sub_plot.index): # Vérifier l'alignement des index
             st.warning("Désalignement des données entre les résultats PCA (pdf_plot) et les données de traits (sub_plot). Le graphique interactif peut être incorrect.")
        else: # Si tout est OK pour le graphique interactif
            required_pdf_cols_interactive = ['Espece_User', 'Ecologie', 'Cluster', 'Source_Habitat'] 
            if not all(col in pdf_plot.columns for col in required_pdf_cols_interactive): st.warning("Colonnes requises (Espece_User, Ecologie, Cluster, Source_Habitat) manquent dans les données PCA pour le graphique interactif.")
            else:
                # Préparer les données pour le graphique interactif
                # Utiliser sub_plot pour les valeurs des traits X et Y, et pdf_plot pour les métadonnées (Cluster, Habitat, etc.)
                # S'assurer que l'index est cohérent pour la jointure implicite lors de la création du DataFrame
                plot_data_interactive = pd.DataFrame({
                    'Espece_User': pdf_plot['Espece_User'], 
                    'Ecologie': pdf_plot['Ecologie'],    
                    x_axis_plot: sub_plot[x_axis_plot],    
                    y_axis_plot: sub_plot[y_axis_plot],    
                    'Cluster': pdf_plot['Cluster'],        
                    'Source_Habitat': pdf_plot['Source_Habitat'] 
                }).set_index(pdf_plot.index) # Conserver l'index original pour la cohérence
                
                # Gestion des points superposés (jittering)
                plot_data_to_use = plot_data_interactive.copy()
                # Utiliser des noms temporaires pour éviter les conflits si x_axis_plot ou y_axis_plot sont '_temp_x'/'_temp_y'
                temp_x_col_grp = "_temp_x_for_grouping"; temp_y_col_grp = "_temp_y_for_grouping"
                plot_data_to_use[temp_x_col_grp] = plot_data_to_use[x_axis_plot]; plot_data_to_use[temp_y_col_grp] = plot_data_to_use[y_axis_plot]
                duplicates_mask = plot_data_to_use.duplicated(subset=[temp_x_col_grp, temp_y_col_grp], keep=False) # Identifier les points avec mêmes coordonnées X,Y
                
                if duplicates_mask.any(): # S'il y a des points superposés
                    # Calculer l'amplitude du jitter en fonction de la plage des données
                    x_min_val, x_max_val = plot_data_to_use[x_axis_plot].min(), plot_data_to_use[x_axis_plot].max()
                    y_min_val, y_max_val = plot_data_to_use[y_axis_plot].min(), plot_data_to_use[y_axis_plot].max()
                    x_range_val = (x_max_val - x_min_val) if pd.notna(x_max_val) and pd.notna(x_min_val) else 0
                    y_range_val = (y_max_val - y_min_val) if pd.notna(y_max_val) and pd.notna(y_min_val) else 0
                    # Jitter proportionnel à la plage, avec un fallback si la plage est nulle
                    jitter_x = x_range_val*0.015 if x_range_val >1e-9 else (abs(plot_data_to_use[x_axis_plot].mean())*0.015 if abs(plot_data_to_use[x_axis_plot].mean()) >1e-9 else 0.015)
                    jitter_y = y_range_val*0.015 if y_range_val >1e-9 else (abs(plot_data_to_use[y_axis_plot].mean())*0.015 if abs(plot_data_to_use[y_axis_plot].mean()) >1e-9 else 0.015)
                    if abs(jitter_x) <1e-9: jitter_x=0.015 # Valeur minimale de jitter
                    if abs(jitter_y) <1e-9: jitter_y=0.015

                    # Appliquer le jitter en cercle pour les points superposés
                    for _, group in plot_data_to_use[duplicates_mask].groupby([temp_x_col_grp, temp_y_col_grp]):
                        if len(group) > 1: # S'il y a plus d'un point dans le groupe superposé
                            # S'assurer que les colonnes sont de type float pour l'ajout du jitter
                            if not pd.api.types.is_float_dtype(plot_data_to_use[x_axis_plot]): plot_data_to_use[x_axis_plot] = plot_data_to_use[x_axis_plot].astype(float)
                            if not pd.api.types.is_float_dtype(plot_data_to_use[y_axis_plot]): plot_data_to_use[y_axis_plot] = plot_data_to_use[y_axis_plot].astype(float)
                            for i, idx in enumerate(group.index): # Pour chaque point du groupe
                                angle = 2 * np.pi * i / len(group) # Répartir les points en cercle
                                plot_data_to_use.loc[idx, x_axis_plot] += jitter_x * np.cos(angle)
                                plot_data_to_use.loc[idx, y_axis_plot] += jitter_y * np.sin(angle)
                plot_data_to_use.drop(columns=[temp_x_col_grp, temp_y_col_grp], inplace=True) # Supprimer les colonnes temporaires

                # Déterminer la coloration et le titre de la légende
                color_by_interactive = "Source_Habitat" if len(st.session_state.selected_habitats_indices) > 1 else "Cluster"
                legend_title_interactive = "Habitat d'Origine" if len(st.session_state.selected_habitats_indices) > 1 else "Cluster PCA"

                # Créer le scatter plot interactif
                fig_interactive_scatter = px.scatter(
                    plot_data_to_use, x=x_axis_plot, y=y_axis_plot,
                    color=color_by_interactive, 
                    text="Espece_User", hover_name="Espece_User", # Texte sur les points et au survol
                    custom_data=["Espece_User", "Ecologie", "Source_Habitat", "Cluster"], # Données pour le hovertemplate
                    template="plotly_dark", height=600, color_discrete_sequence=COLOR_SEQUENCE
                )
                fig_interactive_scatter.update_traces(
                    textposition="top center", marker=dict(opacity=0.8, size=8), # Style des marqueurs
                    textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS), # Taille de la police du texte sur les points
                    hovertemplate=( # Template HTML pour le survol
                        f"<span style='font-size: {HOVER_SPECIES_FONT_SIZE}px;'><b>%{{customdata[0]}}</b></span><br>" # Nom de l'espèce
                        f"Habitat: %{{customdata[2]}}<br>" # Habitat d'origine
                        f"Cluster PCA: %{{customdata[3]}}<br>" # Cluster PCA
                        f"<br><span style='font-size: {HOVER_ECOLOGY_TITLE_FONT_SIZE}px;'><i>Écologie:</i></span><br>" # Titre écologie
                        f"<span style='font-size: {HOVER_ECOLOGY_TEXT_FONT_SIZE}px;'>%{{customdata[1]}}</span>" # Description écologique
                        "<extra></extra>" # Masquer les informations supplémentaires par défaut de Plotly
                    )
                )
                
                # Ajouter les enveloppes convexes (Convex Hulls)
                unique_groups_interactive = sorted(plot_data_to_use[color_by_interactive].unique())
                # Étendre la séquence de couleurs si nécessaire
                extended_color_sequence_interactive = COLOR_SEQUENCE * (len(unique_groups_interactive) // len(COLOR_SEQUENCE) + 1)
                group_color_map_interactive = { # Mapper chaque groupe à une couleur
                    lbl: extended_color_sequence_interactive[i % len(extended_color_sequence_interactive)] for i, lbl in enumerate(unique_groups_interactive)
                }

                for group_label in unique_groups_interactive: # Pour chaque groupe (habitat ou cluster)
                    group_points_df_interactive = plot_data_to_use[plot_data_to_use[color_by_interactive] == group_label]
                    if x_axis_plot in group_points_df_interactive and y_axis_plot in group_points_df_interactive:
                        # Points uniques pour l'enveloppe (éviter les erreurs avec des points dupliqués par le jitter)
                        points_for_hull = group_points_df_interactive[[x_axis_plot, y_axis_plot]].drop_duplicates().values 
                        if len(points_for_hull) >= MIN_POINTS_FOR_HULL: # Besoin d'au moins 3 points pour une enveloppe
                            try:
                                hull_interactive = ConvexHull(points_for_hull) 
                                # Chemin de l'enveloppe (fermer le polygone)
                                hull_path_interactive = points_for_hull[np.append(hull_interactive.vertices, hull_interactive.vertices[0])]
                                clr_int = group_color_map_interactive.get(group_label, COLOR_SEQUENCE[0]) # Couleur de l'enveloppe
                                fig_interactive_scatter.add_trace(go.Scatter(
                                    x=hull_path_interactive[:, 0], y=hull_path_interactive[:, 1], 
                                    fill="toself", fillcolor=clr_int, # Remplissage
                                    line=dict(color=clr_int, width=1.5), mode='lines', 
                                    name=f'{legend_title_interactive} {group_label} Hull', opacity=0.2, # Opacité
                                    showlegend=False, hoverinfo='skip' # Ne pas montrer dans la légende ni au survol
                                ))
                            except Exception as e: print(f"Erreur calcul Hull interactif {group_label} ({x_axis_plot}, {y_axis_plot}): {e}")
                
                fig_interactive_scatter.update_layout(
                    title_text=f"{y_axis_plot} vs. {x_axis_plot}", title_x=0.5, # Titre du graphique
                    xaxis_title=x_axis_plot, yaxis_title=y_axis_plot, dragmode='pan', # Titres des axes, mode de drag
                    legend_title_text=legend_title_interactive # Titre de la légende
                )
                st.plotly_chart(fig_interactive_scatter, use_container_width=True, config={'scrollZoom': True}) # Afficher le graphique

# Message si l'analyse a été lancée mais 'sub' est vide (aucune espèce traitée)
elif st.session_state.run_main_analysis_once and st.session_state.get('sub', pd.DataFrame()).empty :
    st.markdown("---")
    st.subheader("Étape 2: Exploration Interactive et Paramètres ACP")
    st.warning("L'analyse principale n'a pas abouti à des données suffisantes pour cette section (aucune espèce trouvée ou traitée). Veuillez vérifier les étapes précédentes.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 3: VISUALISATION PRINCIPALE (ACP)
# ---------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("Étape 3: Visualisation Principale (ACP)")
_, col_pca_plot_area = st.columns([0.01, 0.99]) # Colonne principale pour le graphique ACP

if st.session_state.run_main_analysis_once: 
    pdf_display_pca = st.session_state.get('pdf', pd.DataFrame()) # Données PCA à afficher
    
    # Vérifier si les données PCA sont prêtes et contiennent les colonnes nécessaires
    if not pdf_display_pca.empty and "PC1" in pdf_display_pca.columns and "Cluster" in pdf_display_pca.columns and \
       "Espece_User" in pdf_display_pca.columns and "Ecologie" in pdf_display_pca.columns and "Source_Habitat" in pdf_display_pca.columns:
        
        y_pca_col = "PC2" if "PC2" in pdf_display_pca.columns else None # Utiliser PC2 s'il existe
        
        if "PC1" in pdf_display_pca.columns and y_pca_col : # Si au moins PC1 et PC2 sont disponibles
            # Déterminer la coloration et le titre de la légende pour le graphique PCA
            color_by_pca = "Source_Habitat" if len(st.session_state.selected_habitats_indices) > 1 else "Cluster"
            legend_title_pca = "Habitat d'Origine" if len(st.session_state.selected_habitats_indices) > 1 else "Cluster PCA"

            # Créer le scatter plot PCA
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
                hovertemplate=( # Template HTML pour le survol
                    f"<span style='font-size: {HOVER_SPECIES_FONT_SIZE}px;'><b>%{{customdata[0]}}</b></span><br>"
                    f"Habitat: %{{customdata[2]}}<br>"
                    f"Cluster PCA: %{{customdata[3]}}<br>"
                    f"<br><span style='font-size: {HOVER_ECOLOGY_TITLE_FONT_SIZE}px;'><i>Écologie:</i></span><br>"
                    f"<span style='font-size: {HOVER_ECOLOGY_TEXT_FONT_SIZE}px;'>%{{customdata[1]}}</span>"
                    "<extra></extra>"
                ), 
                textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS) # Taille de la police du texte sur les points
            ) 
            
            # Ajouter les enveloppes convexes (Convex Hulls) pour le graphique PCA
            unique_groups_pca = sorted(pdf_display_pca[color_by_pca].unique())
            extended_color_sequence_pca = COLOR_SEQUENCE * (len(unique_groups_pca) // len(COLOR_SEQUENCE) + 1)
            group_color_map_pca = {
                lbl: extended_color_sequence_pca[i % len(extended_color_sequence_pca)] for i, lbl in enumerate(unique_groups_pca)
            }
            
            for group_label_pca in unique_groups_pca: # Pour chaque groupe
                group_points_df_pca = pdf_display_pca[pdf_display_pca[color_by_pca] == group_label_pca]
                if not group_points_df_pca.empty and "PC1" in group_points_df_pca.columns and y_pca_col in group_points_df_pca.columns:
                    # Utiliser les points uniques pour l'enveloppe
                    unique_group_points_pca = group_points_df_pca[["PC1", y_pca_col]].drop_duplicates().values
                    if len(unique_group_points_pca) >= MIN_POINTS_FOR_HULL: # Besoin d'au moins 3 points
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
        else: # Si moins de deux composantes principales sont disponibles
            fig_pca = None # Réinitialiser fig_pca
            if not pdf_display_pca.empty : 
                with col_pca_plot_area: st.warning("Moins de deux composantes principales disponibles pour le graphique PCA. Le graphique ne peut être affiché.")
    # else: fig_pca reste None si pdf_display_pca est vide ou mal formé

# Affichage du graphique PCA (ou des messages d'avertissement)
with col_pca_plot_area: 
    if fig_pca: st.plotly_chart(fig_pca, use_container_width=True, config={'scrollZoom': True}) 
    elif st.session_state.run_main_analysis_once and st.session_state.get('sub', pd.DataFrame()).empty:
        st.warning("L'analyse n'a pas produit de résultats affichables pour le PCA (pas d'espèces traitées ou PCA impossible).")
    elif st.session_state.run_main_analysis_once and fig_pca is None and not st.session_state.get('pdf', pd.DataFrame()).empty : 
        pass # Le message d'avertissement pour PCA (moins de 2 PCs) est déjà géré ci-dessus
    elif st.session_state.run_main_analysis_once : # Cas général si fig_pca n'a pas été créé
        st.warning("Le graphique PCA n'a pas pu être généré. Vérifiez les données d'entrée et les paramètres.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 4: COMPOSITION DES CLUSTERS (ACP)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty: 
    st.markdown("---"); st.subheader("Étape 4: Composition des Clusters (issus de l'ACP)")
    pdf_compo = st.session_state.get('pdf', pd.DataFrame()) # Données PCA pour la composition des clusters
    # Vérifier si les données sont prêtes
    if not pdf_compo.empty and 'Cluster' in pdf_compo.columns and 'Espece_User' in pdf_compo.columns and 'Source_Habitat' in pdf_compo.columns:
        # Créer une colonne pour l'affichage combinant espèce et habitat source
        pdf_compo['Species_Instance_Display'] = pdf_compo['Espece_User'] + " (" + pdf_compo['Source_Habitat'] + ")"
        
        compositions_display = [] # Pour stocker les informations de chaque cluster
        for c_pca in sorted(pdf_compo["Cluster"].unique()): # Pour chaque label de cluster unique
            cluster_data = pdf_compo[pdf_compo["Cluster"] == c_pca] # Données du cluster actuel
            # Instances uniques d'espèces (espèce + habitat) dans le cluster
            unique_species_instances_in_cluster = cluster_data["Species_Instance_Display"].unique()
            compositions_display.append({
                "cluster_label": c_pca, 
                "count": len(unique_species_instances_in_cluster), 
                "species_list": sorted(list(unique_species_instances_in_cluster)) # Liste triée des instances
            })

        if compositions_display and any(d['count'] > 0 for d in compositions_display): # Si des clusters avec des espèces existent
            # Déterminer le nombre de colonnes pour l'affichage (max 3)
            num_clusters_disp = len([d for d in compositions_display if d['count']>0]) 
            num_cols_disp = min(num_clusters_disp, 3) if num_clusters_disp > 0 else 1
            cluster_cols_layout = st.columns(num_cols_disp) # Créer les colonnes
            col_idx = 0
            for comp_data in compositions_display: # Pour chaque cluster à afficher
                if comp_data['count'] > 0: 
                    with cluster_cols_layout[col_idx % num_cols_disp]: # Placer dans la colonne appropriée
                        st.markdown(f"**Cluster PCA {comp_data['cluster_label']}** ({comp_data['count']} instances d'espèces)")
                        for species_instance_name in comp_data['species_list']: st.markdown(f"- {species_instance_name}")
                    col_idx += 1
            if col_idx == 0 : st.info("Aucun cluster (ACP) avec des espèces à afficher.") # Si aucun cluster n'avait d'espèces
        else: st.info("La composition des clusters (ACP) sera affichée ici après l'analyse (pas de données de cluster).")
    else: st.info("La composition des clusters (ACP) sera affichée ici après l'analyse (données de PCA non disponibles ou incomplètes).")
# Message si l'analyse a été lancée mais 'sub' est vide
elif st.session_state.run_main_analysis_once: 
    st.markdown("---"); st.subheader("Étape 4: Composition des Clusters (ACP)")
    st.info("Analyse lancée, mais aucune donnée d'espèce n'a pu être traitée pour la composition des clusters.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 5: ANALYSE DES CO-OCCURRENCES D'ESPÈCES (basée sur les syntaxons)
# ---------------------------------------------------------------------------- #
def style_cooccurrence_row(row, max_overall_count, vmin_count=1):
    """
    Applique un style de fond coloré aux cellules des voisins en fonction de leur compte de co-occurrence.
    La fonction `row` est une Series pandas représentant une ligne du DataFrame auquel le style est appliqué.
    Elle doit contenir les colonnes 'Voisin 1 Compte', 'Voisin 2 Compte', 'Voisin 3 Compte'.
    La fonction retourne une Series de chaînes de style CSS, avec le même index que la `row` d'entrée.
    """
    # Initialise une Series pour les styles, avec le même index que la ligne d'entrée.
    # Les styles non spécifiés resteront des chaînes vides (pas de style).
    styles = pd.Series('', index=row.index) 
    
    # Couleurs pour le dégradé: du gris foncé (proche du noir) au rouge foncé
    color_start_rgb = (40, 40, 40)  # Gris très foncé
    color_end_rgb = (200, 50, 50)    # Rouge foncé modéré

    for col_num in [1, 2, 3]: # Correspond à Voisin 1, Voisin 2, Voisin 3
        # Colonne à laquelle le style sera appliqué (ex: 'Voisin 1', 'Voisin 2', etc.)
        target_style_col_name = f'Voisin {col_num}' 
        # Colonne d'où lire la valeur du compte (ex: 'Voisin 1 Compte')
        count_val_col_name = f'Voisin {col_num} Compte'
        
        # Vérifier si la colonne de compte existe dans la ligne (elle devrait si subset=None)
        if count_val_col_name in row.index:
            count_val = row[count_val_col_name] 

            if pd.notna(count_val) and count_val > 0:
                current_count = int(count_val)
                # Calculer le ratio pour l'interpolation des couleurs
                if max_overall_count == vmin_count: 
                    ratio = 1.0 if current_count >= vmin_count else 0.0
                elif max_overall_count > vmin_count:
                    ratio = (current_count - vmin_count) / (max_overall_count - vmin_count)
                    ratio = max(0.0, min(ratio, 1.0)) 
                else: 
                    ratio = 0.0
                
                # Interpoler les couleurs RGB
                r = int(color_start_rgb[0] + ratio * (color_end_rgb[0] - color_start_rgb[0]))
                g = int(color_start_rgb[1] + ratio * (color_end_rgb[1] - color_start_rgb[1]))
                b = int(color_start_rgb[2] + ratio * (color_end_rgb[2] - color_start_rgb[2]))
                
                # Appliquer le style à la colonne cible (ex: 'Voisin 1')
                styles[target_style_col_name] = f'background-color: rgb({r},{g},{b})'
            else:
                # Pas de style particulier si pas de co-occurrence ou compte nul pour la colonne cible
                styles[target_style_col_name] = 'background-color: none' 
        # else: si la colonne de compte n'est pas dans la ligne, ne rien faire pour cette colonne de style.
        # Cela ne devrait pas arriver si la fonction reçoit la ligne complète.
    return styles

if st.session_state.run_main_analysis_once and \
   not st.session_state.get('sub', pd.DataFrame()).empty and \
   syntaxon_data_list:

    st.markdown("---")
    st.subheader("Étape 5: Analyse des Co-occurrences d'Espèces (basée sur les listes de syntaxons)")

    principal_species_original_names_from_sub = st.session_state.sub['Espece_Ref_Original'].unique()
    cooccurrence_results_list = [] 

    for principal_species_original in principal_species_original_names_from_sub:
        principal_species_normalized = normalize_species_name_for_villaret(principal_species_original)
        if not principal_species_normalized:
            continue

        co_occurrence_counts_for_this_principal = defaultdict(int)
        for syntaxon_record in syntaxon_data_list:
            if principal_species_normalized in syntaxon_record['species_set']:
                for other_species_in_syntaxon_normalized in syntaxon_record['species_set']:
                    if other_species_in_syntaxon_normalized != principal_species_normalized:
                        co_occurrence_counts_for_this_principal[other_species_in_syntaxon_normalized] += 1
        
        current_result_row_dict = {'Espèce Principale (issue des relevés)': principal_species_original}
        if co_occurrence_counts_for_this_principal:
            sorted_co_occurrences = sorted(co_occurrence_counts_for_this_principal.items(), key=lambda item: item[1], reverse=True)
            
            for i in range(3): 
                neighbor_num = i + 1 
                if i < len(sorted_co_occurrences):
                    name, count = sorted_co_occurrences[i]
                    current_result_row_dict[f'Voisin {neighbor_num} Nom'] = name
                    current_result_row_dict[f'Voisin {neighbor_num} Compte'] = count
                else: 
                    current_result_row_dict[f'Voisin {neighbor_num} Nom'] = None
                    current_result_row_dict[f'Voisin {neighbor_num} Compte'] = pd.NA 
        else: 
            for neighbor_num in [1, 2, 3]:
                current_result_row_dict[f'Voisin {neighbor_num} Nom'] = None
                current_result_row_dict[f'Voisin {neighbor_num} Compte'] = pd.NA
        cooccurrence_results_list.append(current_result_row_dict)

    if cooccurrence_results_list:
        raw_cooccurrence_df = pd.DataFrame(cooccurrence_results_list)

        all_counts_for_styling = []
        for neighbor_num in [1, 2, 3]:
            counts_in_col = raw_cooccurrence_df[f'Voisin {neighbor_num} Compte'].dropna().astype(int)
            all_counts_for_styling.extend(counts_in_col[counts_in_col > 0].tolist())
        
        max_overall_cooccurrence = max(all_counts_for_styling) if all_counts_for_styling else 0
        min_cooccurrence_for_color = 1 

        # Création du DataFrame qui sera passé au Styler.
        # Ce DataFrame doit contenir toutes les colonnes nécessaires à la fonction de style pour la lecture,
        # et les colonnes cibles pour l'application du style.
        df_for_styling_input_and_display = []
        for _, row_from_raw_df in raw_cooccurrence_df.iterrows():
            # Dictionnaire pour la nouvelle ligne du DataFrame à styler
            styled_row_dict = {'Espèce Principale (issue des relevés)': row_from_raw_df['Espèce Principale (issue des relevés)']}
            for neighbor_num_loop in [1, 2, 3]: 
                nom_col_in_raw = f'Voisin {neighbor_num_loop} Nom'
                compte_col_in_raw = f'Voisin {neighbor_num_loop} Compte'
                
                nom_val = row_from_raw_df[nom_col_in_raw]
                compte_val = row_from_raw_df[compte_col_in_raw]
                
                # Colonne pour affichage combiné (ex: 'Voisin 1')
                display_col_combined = f'Voisin {neighbor_num_loop}'
                # Colonne pour le compte brut, utilisée par la fonction de style (ex: 'Voisin 1 Compte')
                data_col_compte = f'Voisin {neighbor_num_loop} Compte'

                if pd.notna(nom_val) and pd.notna(compte_val) and compte_val > 0:
                    styled_row_dict[display_col_combined] = f"{str(nom_val).capitalize()} - {compte_val}"
                else:
                    styled_row_dict[display_col_combined] = "-" 
                
                # Ajouter la colonne de compte brut. La fonction de style lira cette colonne.
                styled_row_dict[data_col_compte] = compte_val if pd.notna(compte_val) else 0 
            df_for_styling_input_and_display.append(styled_row_dict)
        
        # Ce DataFrame contient maintenant les colonnes d'affichage ET les colonnes de compte brut
        cooccurrence_df_for_styler = pd.DataFrame(df_for_styling_input_and_display)

        st.markdown("Ce tableau présente, pour chaque espèce de vos relevés (colonne 1), les trois espèces qui lui sont le plus fréquemment associées au sein des listes d'espèces caractéristiques des syntaxons de référence (`data_villaret.csv`). Le nombre après le tiret indique le nombre de syntaxons partagés. La couleur de fond indique l'intensité de cette co-occurrence (du gris foncé au rouge).")
        
        styled_object = cooccurrence_df_for_styler.style.apply(
            style_cooccurrence_row, # La fonction de style
            max_overall_count=max_overall_cooccurrence,
            vmin_count=min_cooccurrence_for_color,
            axis=1, # Appliquer par ligne
            subset=None # La fonction de style reçoit la ligne entière de cooccurrence_df_for_styler
        ).format(na_rep="-")

        # Colonnes à masquer après le style (les colonnes de compte brut)
        columns_to_hide = [f'Voisin {i} Compte' for i in [1,2,3]]
        
        st.dataframe(styled_object.hide(axis="columns", subset=columns_to_hide), use_container_width=True)

    else:
        st.info("Aucune donnée de co-occurrence à afficher pour les espèces sélectionnées et les syntaxons disponibles.")

elif st.session_state.run_main_analysis_once and not syntaxon_data_list:
    st.markdown("---")
    st.subheader("Étape 5: Analyse des Co-occurrences d'Espèces (basée sur les listes de syntaxons)")
    st.warning("Les données des syntaxons ('data_villaret.csv') n'ont pas pu être chargées, sont vides, ou ne contiennent aucun syntaxon valide. L'analyse des co-occurrences ne peut pas être effectuée.")

