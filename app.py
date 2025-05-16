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
            self.components_ = np.array([[0.5, 0.5], [-0.5, 0.5]]) # Exemple
            self.explained_variance_ = np.array([0.6, 0.4]) # Exemple

    def mock_analyse(sub_df, n_clusters):
        n_samples = len(sub_df)
        if n_samples == 0:
            return np.array([]), MockPCA(), pd.DataFrame(), pd.DataFrame()
        
        # Simuler des coordonnées PCA (2 composantes)
        coords = np.random.rand(n_samples, 2) * 10
        
        # Simuler des labels de cluster
        if n_samples < n_clusters :
             labels = np.arange(n_samples)
        else:
            labels = np.random.randint(0, n_clusters, n_samples)
        
        # Simuler X (données normalisées pour le dendrogramme)
        X_scaled = (sub_df.select_dtypes(include=np.number) - sub_df.select_dtypes(include=np.number).mean()) / sub_df.select_dtypes(include=np.number).std()
        X_scaled = X_scaled.fillna(0).values
        if X_scaled.shape[1] == 0: # S'il n'y a pas de colonnes numériques
            X_scaled = np.random.rand(n_samples, 2) # Données aléatoires pour éviter les erreurs

        mock_pca_obj = MockPCA()
        # Ajuster la taille des components_ en fonction du nombre de traits numériques
        num_numeric_traits = X_scaled.shape[1]
        if num_numeric_traits > 0:
            mock_pca_obj.components_ = np.random.rand(num_numeric_traits, min(2, num_numeric_traits))
            mock_pca_obj.explained_variance_ = np.random.rand(min(2, num_numeric_traits))
        else: # Pas de traits numériques, PCA ne peut pas vraiment tourner
            mock_pca_obj.components_ = np.array([])
            mock_pca_obj.explained_variance_ = np.array([])
            coords = pd.DataFrame(np.random.rand(n_samples, 2), columns=['PC1', 'PC2']) # Coords aléatoires si pas de traits

        return labels, mock_pca_obj, coords, X_scaled

    core = type('CoreModule', (object,), {'analyse': mock_analyse})


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
/* Ajuster la taille des cases à cocher et de leur label */
div[data-testid="stCheckbox"] label {
    font-size: 13px !important; /* Taille du texte du label */
    line-height: 1.2;
}
div[data-testid="stCheckbox"] span { /* Conteneur du label */
    padding-top: 0px !important;
    margin-top: -2px; /* Ajustement vertical si nécessaire */
}

/* Cibler spécifiquement les checkboxes pour la sélection d'habitat si possible */
/* Cela nécessiterait une classe CSS parente ou un ID plus spécifique si disponible */
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
        # Simuler la lecture si core.py n'est pas là et donc data_ref.csv non plus
        if "core" not in globals() or not hasattr(core, "read_reference"):
            st.warning(f"Simulation du chargement de '{file_path}'. Le fichier réel n'est pas utilisé.")
            # Créer un DataFrame de référence simulé
            example_species = [f"Espece Alpha {i}" for i in range(1, 6)] + \
                              [f"Espece Beta {i}" for i in range(1, 6)] + \
                              [f"Espece Gamma {i}" for i in range(1, 6)]
            data = pd.DataFrame({
                'Espece': example_species,
                'Trait1_num': np.random.rand(15) * 10,
                'Trait2_num': np.random.randint(1, 100, 15),
                'Trait3_cat': np.random.choice(['A', 'B', 'C'], 15)
            })
            return data
        
        data = core.read_reference(file_path) # Ligne originale
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
        .astype(str) # Assurer que c'est une chaîne avant les opérations de chaîne
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
    wrapped_lines = textwrap.wrap(str(text), width=line_width_chars) # Assurer que text est une chaîne
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
        # Retourner un DataFrame vide avec la colonne attendue pour éviter les erreurs .map plus tard
        return pd.DataFrame(columns=['Description_Ecologie'])
    except ValueError as ve:
        print(f"AVERTISSEMENT: Erreur de valeur lors de la lecture du fichier '{file_path}'. Détails: {ve}.")
        st.toast(f"Erreur format fichier écologique '{file_path}'.", icon="🔥")
        return pd.DataFrame(columns=['Description_Ecologie'])
    except Exception as e:
        print(f"AVERTISSEMENT: Impossible de charger les données écologiques depuis '{file_path}': {e}.")
        st.toast(f"Erreur chargement fichier écologique.", icon="🔥")
        return pd.DataFrame(columns=['Description_Ecologie'])

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
}

for key, value in default_session_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

if 'releves_df' not in st.session_state:
    num_placeholder_cols = 15
    num_placeholder_rows_total = 11 # 1 pour les noms d'habitats + 10 pour les espèces
    # Créer des données de placeholder avec des types mixtes pour éviter les erreurs de data_editor
    header = [f"NomHabitat {c_idx+1}" for c_idx in range(num_placeholder_cols)]
    placeholder_rows = [["" for _ in range(num_placeholder_cols)] for _ in range(num_placeholder_rows_total -1)]
    st.session_state.releves_df = pd.DataFrame([header] + placeholder_rows)


# ---------------------------------------------------------------------------- #
# SECTION 0: INPUT TABLE AND HABITAT SELECTION (Full Width)
# ---------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("Étape 1: Importation et Sélection des Relevés Floristiques")

st.info("Copiez-collez vos données de relevés ici (Ctrl+V ou Cmd+V). La première ligne de vos données doit contenir les noms des habitats/relevés. Les lignes suivantes contiennent les espèces pour chaque relevé.")
# Pour data_editor, il est préférable que les colonnes aient des noms (même des entiers)
# Si st.session_state.releves_df est vide ou n'a pas de colonnes, data_editor peut avoir des problèmes.
# Assurons-nous qu'il y a des colonnes par défaut.
if st.session_state.releves_df.empty:
    num_placeholder_cols = 15
    num_placeholder_rows_total = 11
    header = [f"NomHabitat {c_idx+1}" for c_idx in range(num_placeholder_cols)]
    placeholder_rows = [["" for _ in range(num_placeholder_cols)] for _ in range(num_placeholder_rows_total -1)]
    st.session_state.releves_df = pd.DataFrame([header] + placeholder_rows)
else:
    # S'assurer que les noms de colonnes sont des chaînes pour éviter les problèmes avec data_editor
    st.session_state.releves_df.columns = [str(col) for col in st.session_state.releves_df.columns]


edited_releves_df_from_editor = st.data_editor(
    st.session_state.releves_df,
    num_rows="dynamic", # Permet à l'utilisateur d'ajouter/supprimer des lignes
    use_container_width=True,
    key="releves_data_editor_key" 
)

# Update session state if editor is used, to make changes persistent for checkbox generation
if not edited_releves_df_from_editor.equals(st.session_state.releves_df):
    st.session_state.releves_df = edited_releves_df_from_editor.copy() # Utiliser .copy()
    # Réinitialiser les sélections si la structure du tableau change (par exemple, nombre de colonnes)
    # Cela est implicitement géré car les clés des cases à cocher dépendent du nombre de colonnes.
    # Cependant, il est bon de réinitialiser explicitement selected_habitats_indices si le nombre de colonnes change.
    if len(st.session_state.releves_df.columns) != len(st.session_state.get('previous_num_cols', [])):
         st.session_state.selected_habitats_indices = []
    st.session_state.previous_num_cols = len(st.session_state.releves_df.columns)
    st.rerun() 

current_releves_df_for_checkboxes = st.session_state.releves_df
selected_col_indices_for_analysis = [] # Sera rempli par les sélections de checkbox

if not current_releves_df_for_checkboxes.empty and \
   len(current_releves_df_for_checkboxes.columns) > 0 and \
   len(current_releves_df_for_checkboxes) > 0:
    
    # Les noms d'habitat sont sur la première ligne (index 0)
    habitat_names_from_df = current_releves_df_for_checkboxes.iloc[0].astype(str).tolist()
    num_actual_cols = len(current_releves_df_for_checkboxes.columns)
    
    # S'assurer que habitat_names_full a la bonne longueur et des noms par défaut si vides
    habitat_names_full = [habitat_names_from_df[i] if i < len(habitat_names_from_df) else f"Relevé {i+1}" for i in range(num_actual_cols)]
    habitat_names_full = [name if pd.notna(name) and str(name).strip() != "" else f"Relevé {i+1}" for i, name in enumerate(habitat_names_full)]

    st.markdown("**Sélectionnez les habitats (colonnes) à inclure dans l'analyse :**")
    
    # Déterminer le nombre de colonnes pour l'affichage des checkboxes
    cols_per_row_for_checkboxes = min(5, num_actual_cols if num_actual_cols > 0 else 1)
    if num_actual_cols == 0: # S'il n'y a pas de colonnes, ne pas essayer de créer st.columns(0)
        st.warning("Le tableau de données est vide ou ne contient pas de colonnes.")
    else:
        checkbox_cols_layout = st.columns(cols_per_row_for_checkboxes)
        
        temp_selected_indices_from_checkboxes = []
        for i in range(num_actual_cols):
            habitat_display_name = habitat_names_full[i]
            # La valeur par défaut de la checkbox est si son index est dans selected_habitats_indices
            is_selected_in_ui = checkbox_cols_layout[i % cols_per_row_for_checkboxes].checkbox(
                habitat_display_name, 
                key=f"select_habitat_col_{i}", 
                value = i in st.session_state.selected_habitats_indices # Utiliser l'état de session ici
            )
            if is_selected_in_ui:
                temp_selected_indices_from_checkboxes.append(i)
        
        # Mettre à jour l'état de session si les sélections ont changé
        if sorted(temp_selected_indices_from_checkboxes) != sorted(st.session_state.selected_habitats_indices):
            st.session_state.selected_habitats_indices = temp_selected_indices_from_checkboxes
            st.rerun() 
            
    selected_col_indices_for_analysis = st.session_state.selected_habitats_indices


# ---------------------------------------------------------------------------- #
# SECTION 1: CONTROLS ET GRAPHIQUE ACP
# ---------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("Étape 2: Paramètres d'Analyse et Visualisation Principale")
col_controls_area, col_pca_plot_area = st.columns([1, 2]) 

with col_controls_area:
    st.markdown("##### Paramètres")
    n_clusters_selected = st.slider("Nombre de clusters (pour ACP)", 2, 8, 3, key="n_clusters_slider_main", disabled=ref.empty)
    
    run_main_analysis_button = st.button(
        "Lancer l'analyse principale", 
        type="primary", 
        disabled=ref.empty or not selected_col_indices_for_analysis, # Désactivé si pas de ref ou pas de sélection
        key="main_analysis_button_trigger"
    )
    if ref.empty:
        st.warning("Les données de référence ('data_ref.csv') n'ont pas pu être chargées ou sont simulées. L'analyse est désactivée si les données réelles manquent.")
    if not selected_col_indices_for_analysis and not ref.empty:
        st.info("Veuillez sélectionner au moins un habitat (colonne) dans le tableau ci-dessus pour activer l'analyse.")

# Variables locales pour les résultats d'analyse, seront mises à jour depuis st.session_state si l'analyse a déjà eu lieu
fig_pca = None
fig_dend = None
# species_binom_user_unique est recalculé à chaque fois que le bouton est cliqué

# ---------------------------------------------------------------------------- #
# ANALYSE PRINCIPALE (CALCULS)
# ---------------------------------------------------------------------------- #
if run_main_analysis_button and not ref.empty and selected_col_indices_for_analysis:
    st.session_state.run_main_analysis_once = True
    
    species_raw_from_table = []
    # Utiliser la version la plus à jour du DataFrame des relevés (depuis l'éditeur ou l'état de session)
    df_for_species_extraction = st.session_state.releves_df.copy() # Travailler sur une copie
    
    if not df_for_species_extraction.empty and len(df_for_species_extraction) > 1: # S'assurer qu'il y a au moins une ligne pour les noms d'habitat ET une pour les espèces
        for col_idx_int in selected_col_indices_for_analysis: # col_idx_int est un entier
            if col_idx_int < len(df_for_species_extraction.columns):
                # Les espèces sont à partir de la deuxième ligne (index 1)
                # .iloc attend des indexeurs entiers pour les lignes et les colonnes.
                species_in_col_series = df_for_species_extraction.iloc[1:, col_idx_int]
                
                species_in_col_cleaned = species_in_col_series.dropna()\
                                        .astype(str)\
                                        .str.strip()\
                                        .replace('', np.nan)\
                                        .dropna()\
                                        .tolist()
                species_raw_from_table.extend(s for s in species_in_col_cleaned if s) # s'assurer que s n'est pas une chaîne vide après strip

    species_raw_unique_temp = sorted(list(set(s for s in species_raw_from_table if s))) # Filtrer les chaînes vides ici aussi
    species_binom_user_unique = [" ".join(s.split()[:2]).lower() for s in species_raw_unique_temp if s and len(s.split()) >=1] # Gérer les chaînes vides ou avec moins de 2 mots

    if not species_binom_user_unique:
        with col_controls_area:
            st.error("Aucune espèce valide (nom binomial) extraite des relevés sélectionnés. Vérifiez vos données et sélections.")
        st.session_state.run_main_analysis_once = False
        st.stop()

    indices_to_keep_from_ref = []
    if not ref_binom_series.empty:
        ref_indexed_binom = ref_binom_series.reset_index()
        # S'assurer que les colonnes ont les bons noms si reset_index a créé 'index'
        if 'index' in ref_indexed_binom.columns and 'Espece' in ref_indexed_binom.columns and len(ref_indexed_binom.columns) == 2 : # Cas simple
             ref_indexed_binom.columns = ['Original_Ref_Index', 'ref_binom_val']
        elif len(ref_indexed_binom.columns) == 2 : # Si déjà 2 colonnes, on assume qu'elles sont correctes
             ref_indexed_binom.columns = ['Original_Ref_Index', 'ref_binom_val']
        else: # Cas plus complexe, essayer de trouver la colonne 'Espece'
            if 'Espece' in ref_indexed_binom.columns:
                original_indices = ref_indexed_binom.index
                ref_binom_values = ref_indexed_binom['Espece']
                ref_indexed_binom = pd.DataFrame({'Original_Ref_Index': original_indices, 'ref_binom_val': ref_binom_values})
            else: # Fallback si la structure est inattendue
                with col_controls_area: st.error("Structure inattendue de ref_binom_series après reset_index.")
                st.session_state.run_main_analysis_once = False; st.stop()


        for user_binom_specie in species_binom_user_unique:
            matches_in_ref = ref_indexed_binom[ref_indexed_binom['ref_binom_val'] == user_binom_specie]
            if not matches_in_ref.empty:
                indices_to_keep_from_ref.append(matches_in_ref['Original_Ref_Index'].iloc[0])
    
    # Utiliser les indices uniques pour éviter les doublons si une espèce est trouvée plusieurs fois
    indices_to_keep_from_ref = sorted(list(set(indices_to_keep_from_ref)))


    if indices_to_keep_from_ref:
        st.session_state.sub = ref.loc[indices_to_keep_from_ref].copy()
    else:
        st.session_state.sub = pd.DataFrame(columns=ref.columns)
    
    # `sub` est maintenant dans st.session_state.sub

    found_ref_binom_values_in_sub = []
    if not st.session_state.sub.empty:
        found_ref_binom_values_in_sub = ( st.session_state.sub["Espece"].astype(str).str.split().str[:2].str.join(" ").str.lower().tolist() )
    
    raw_names_corresponding_to_binom_user_unique = [
        raw_name for raw_name in species_raw_unique_temp 
        if " ".join(raw_name.split()[:2]).lower() in species_binom_user_unique
    ]
    
    not_found_user_raw_names = [
        raw_names_corresponding_to_binom_user_unique[i] 
        for i, user_binom_name in enumerate(species_binom_user_unique) 
        if user_binom_name not in found_ref_binom_values_in_sub
    ]

    if not_found_user_raw_names:
        with col_controls_area: 
            st.warning("Non trouvées dans la base de traits : " + ", ".join(not_found_user_raw_names), icon="⚠️")

    if st.session_state.sub.empty:
        with col_controls_area:
            st.error("Aucune des espèces sélectionnées n'a été trouvée dans la base de traits de référence. L'analyse ne peut continuer.")
        st.session_state.run_main_analysis_once = False; st.stop()
    if st.session_state.sub.shape[0] < n_clusters_selected and n_clusters_selected > 0 :
        with col_controls_area:
            st.error(f"Le nombre d'espèces uniques trouvées et utilisables ({st.session_state.sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters_selected}).");
        st.session_state.run_main_analysis_once = False; st.stop()
    if st.session_state.sub.shape[0] < 2: # PCA et dendrogramme nécessitent au moins 2 points
        with col_controls_area:
            st.error(f"Au moins 2 espèces uniques sont nécessaires pour l'analyse. {st.session_state.sub.shape[0]} espèce(s) trouvée(s) et utilisée(s).");
        st.session_state.run_main_analysis_once = False; st.stop()

    # S'assurer que sub ne contient que les traits numériques pour core.analyse si c'est ce qu'il attend
    # Ou que core.analyse gère les types de données mixtes.
    # Pour la simulation, on suppose que core.analyse peut prendre sub tel quel.
    
    user_input_binom_to_raw_map = { " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_raw_unique_temp if s_raw and len(s_raw.split()) >=1}
    try:
        # S'assurer que sub a des traits numériques pour l'analyse PCA
        sub_for_analysis = st.session_state.sub.select_dtypes(include=np.number)
        if sub_for_analysis.empty or sub_for_analysis.shape[1] == 0:
            with col_controls_area:
                st.error(f"Aucun trait numérique trouvé pour les espèces sélectionnées dans la base de référence. L'analyse ACP ne peut pas être effectuée.")
            st.session_state.run_main_analysis_once = False; st.stop()

        labels, pca_results, coords_df, X_scaled_data = core.analyse(st.session_state.sub, n_clusters_selected)
        
        # S'assurer que coords_df est un DataFrame, sinon le créer
        if not isinstance(coords_df, pd.DataFrame):
            if isinstance(coords_df, np.ndarray) and coords_df.ndim == 2:
                 num_pcs = coords_df.shape[1]
                 coords_df = pd.DataFrame(coords_df, columns=[f"PC{i+1}" for i in range(num_pcs)])
            else: # Fallback si coords_df n'est pas ce qui est attendu
                with col_controls_area: st.error("Les coordonnées PCA retournées par core.analyse ne sont pas dans un format attendu.")
                st.session_state.run_main_analysis_once = False; st.stop()


        current_pdf = coords_df.copy() # current_pdf est maintenant basé sur coords_df
        current_pdf["Cluster"] = labels.astype(str) if len(labels) == len(current_pdf) else np.zeros(len(current_pdf)).astype(str) # Gérer la taille
        current_pdf["Espece_Ref"] = st.session_state.sub["Espece"].values[:len(current_pdf)] # Assurer la même longueur
        current_pdf["Espece_User"] = current_pdf["Espece_Ref"].apply(
            lambda full_ref_name: user_input_binom_to_raw_map.get(
                " ".join(str(full_ref_name).split()[:2]).lower(), # str() pour la robustesse
                str(full_ref_name) # str() pour la robustesse
            )
        )


        if not ecology_df.empty:
            current_pdf['Espece_Ref_norm_for_eco'] = ( current_pdf['Espece_Ref'].astype(str).str.strip().str.split().str[:2].str.join(" ").str.lower() )
            # Gérer le cas où ecology_df.index.name est différent de 'Espece_norm' ou si l'index n'est pas défini
            if ecology_df.index.name == 'Espece_norm' or 'Espece_norm' in ecology_df.columns:
                 map_source = ecology_df['Description_Ecologie'] if ecology_df.index.name != 'Espece_norm' else ecology_df.set_index('Espece_norm')['Description_Ecologie']
                 current_pdf['Ecologie_raw'] = current_pdf['Espece_Ref_norm_for_eco'].map(map_source)
            else: # Si l'index n'est pas 'Espece_norm', essayer de le définir
                try:
                    current_pdf['Ecologie_raw'] = current_pdf['Espece_Ref_norm_for_eco'].map(ecology_df.set_index('Espece_norm')['Description_Ecologie'])
                except KeyError: # Si 'Espece_norm' n'est pas une colonne valide pour devenir index
                    current_pdf['Ecologie_raw'] = pd.Series([np.nan] * len(current_pdf)) # Pas de données éco
                    st.warning("Impossible de joindre les données écologiques car 'Espece_norm' n'a pas pu être utilisé comme index.")


            current_pdf['Ecologie'] = current_pdf['Ecologie_raw'].apply(lambda x: format_ecology_for_hover(x))
            current_pdf['Ecologie'] = current_pdf['Ecologie'].fillna(format_ecology_for_hover("Description écologique non disponible."))
        else:
            current_pdf['Ecologie'] = format_ecology_for_hover("Description écologique non disponible (fichier non chargé ou vide).")
        st.session_state.pdf = current_pdf

        # Communalités
        if hasattr(pca_results, 'components_') and hasattr(pca_results, 'explained_variance_') and \
           pca_results.components_ is not None and pca_results.explained_variance_ is not None and \
           len(pca_results.components_) > 0 :
            loadings = pca_results.components_.T * (pca_results.explained_variance_ ** 0.5)
            communal = (loadings**2).sum(axis=1)
            trait_columns_for_communal = st.session_state.sub.select_dtypes(include=np.number).columns.tolist() # Utiliser uniquement les traits numériques pour les communalités
            
            # S'assurer que communal a la même longueur que trait_columns_for_communal
            if len(communal) == len(trait_columns_for_communal):
                st.session_state.vip_data_df_for_calc = pd.DataFrame({
                    "Variable": trait_columns_for_communal,
                    "Communalité (%)": (communal * 100).round(0).astype(int),
                }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
            else: # Si les longueurs ne correspondent pas, initialiser un DataFrame vide
                 st.session_state.vip_data_df_for_calc = pd.DataFrame(columns=["Variable", "Communalité (%)"])
                 st.warning(f"Les communalités n'ont pas pu être calculées correctement (incohérence de dimensions: {len(communal)} vs {len(trait_columns_for_communal)}).")
        else:
            st.session_state.vip_data_df_for_calc = pd.DataFrame(columns=["Variable", "Communalité (%)"])
            st.warning("Les résultats PCA ne contiennent pas les informations nécessaires pour calculer les communalités.")

        
        st.session_state.X_for_dendro = X_scaled_data

        all_trait_names_from_sub = [col for col in st.session_state.sub.columns if col.lower() != "espece"]
        st.session_state.numeric_trait_names_for_interactive_plot = [
            col for col in all_trait_names_from_sub if pd.api.types.is_numeric_dtype(st.session_state.sub[col])
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
        with col_controls_area:
            st.error(f"Une erreur est survenue lors de l'analyse principale : {e}"); st.exception(e)
        st.session_state.run_main_analysis_once = False; st.stop()

# --- Affichage des résultats après l'analyse (si elle a eu lieu) ---
if st.session_state.run_main_analysis_once:
    # Récupérer les données de session pour l'affichage
    pdf_display = st.session_state.get('pdf', pd.DataFrame())
    sub_display = st.session_state.get('sub', pd.DataFrame())
    X_for_dendro_display = st.session_state.get('X_for_dendro', np.array([]))
    
    # Construction du graphique PCA
    if not pdf_display.empty and "PC1" in pdf_display.columns:
        # S'assurer que PC2 existe si plus d'une colonne PCA
        y_pca_col = "PC2" if "PC2" in pdf_display.columns and pdf_display.shape[1] > 1 else None
        if y_pca_col is None and pdf_display.shape[1] > 1 and len(pdf_display.columns) >1 and pdf_display.columns[1].startswith("PC"):
            # Si PC2 n'est pas nommé explicitement mais qu'il y a une 2e colonne PCA
            y_pca_col = pdf_display.columns[1]

        fig_pca = px.scatter(pdf_display, x="PC1", y=y_pca_col, 
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
        unique_clusters_pca = sorted(pdf_display["Cluster"].unique())
        cluster_color_map_pca = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_pca)}
        
        if y_pca_col and "PC1" in pdf_display.columns: # S'assurer que les deux colonnes existent pour le hull
            for cluster_label in unique_clusters_pca:
                cluster_points_df_pca = pdf_display[pdf_display["Cluster"] == cluster_label]
                if not cluster_points_df_pca.empty and "PC1" in cluster_points_df_pca.columns and y_pca_col in cluster_points_df_pca.columns:
                    unique_cluster_points_pca = cluster_points_df_pca[["PC1", y_pca_col]].drop_duplicates().values
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

    # Construction du dendrogramme
    if X_for_dendro_display.ndim == 2 and X_for_dendro_display.shape[0] > 1 and X_for_dendro_display.shape[1] > 0:
        try:
            Z = linkage(X_for_dendro_display, method="ward")
            dyn_thresh = 0
            if n_clusters_selected > 1 and (n_clusters_selected -1) < Z.shape[0] : # -1 car Z est indexé à partir de 0, et on veut la (n-1)ème fusion pour n clusters
                dyn_thresh = Z[-(n_clusters_selected-1), 2] * 0.99 if Z.shape[0] >= (n_clusters_selected-1) else (Z[-1,2] * 0.5 if Z.shape[0]>0 else 0)
            elif Z.shape[0] > 0: 
                dyn_thresh = Z[0, 2] / 2 # Fallback si peu de clusters demandés ou peu de données
            
            dendro_labels = pdf_display["Espece_User"].tolist() if not pdf_display.empty and "Espece_User" in pdf_display.columns and len(pdf_display) == X_for_dendro_display.shape[0] else [f"Esp {i+1}" for i in range(X_for_dendro_display.shape[0])]

            fig_dend = ff.create_dendrogram(X_for_dendro_display, orientation="left", labels=dendro_labels, 
                                            linkagefun=lambda _: Z, color_threshold=dyn_thresh if n_clusters_selected > 1 else 0, 
                                            colorscale=COLOR_SEQUENCE)
            fig_dend.update_layout(template="plotly_dark", height=max(400, sub_display.shape[0] * 20 if not sub_display.empty else 400), title_text="Dendrogramme", title_x=0.5)
        except Exception as e:
            print(f"Erreur lors de la création du dendrogramme: {e}")
            fig_dend = None # Assurer que fig_dend est None en cas d'erreur
    else: 
        fig_dend = None
        if X_for_dendro_display.ndim != 2 or X_for_dendro_display.shape[1] == 0:
            print("Données X_for_dendro_display non valides pour linkage (pas 2D ou pas de features).")


# --- Affichage du graphique PCA dans la colonne dédiée ---
with col_pca_plot_area:
    if fig_pca: 
        st.plotly_chart(fig_pca, use_container_width=True, config={'scrollZoom': True}) 
    elif st.session_state.run_main_analysis_once : # Si l'analyse a tourné mais pas de fig_pca
        st.warning("Le graphique PCA n'a pas pu être généré. Vérifiez les données d'entrée et les paramètres.")
    # Messages initiaux ou d'erreur déjà gérés dans col_controls_area ou par le bouton


# ---------------------------------------------------------------------------- #
# SECTION 2: EXPLORATION INTERACTIVE DES VARIABLES
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty:
    st.markdown("---")
    st.subheader("Étape 3: Exploration Interactive des Variables")
    col_interactive_table, col_interactive_graph = st.columns([2, 3]) 

    with col_interactive_table:
        st.markdown("##### Tableau d'exploration interactif des variables")
        df_editor_source = st.session_state.get('vip_data_df_interactive', pd.DataFrame(columns=["Variable", "Communalité (%)", "Axe X", "Axe Y"]))

        if not df_editor_source.empty:
            if 'vip_data_df_interactive_snapshot_for_comparison' not in st.session_state or \
               st.session_state.vip_data_df_interactive_snapshot_for_comparison.empty or \
               not isinstance(st.session_state.vip_data_df_interactive_snapshot_for_comparison, pd.DataFrame) or \
               list(st.session_state.vip_data_df_interactive_snapshot_for_comparison.columns) != list(df_editor_source.columns): # Vérifier la structure
                st.session_state.vip_data_df_interactive_snapshot_for_comparison = df_editor_source.copy()

            edited_df = st.data_editor(
                df_editor_source, # Utiliser la source de l'état de session
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
            
            # Logique pour gérer une seule sélection pour Axe X et Axe Y
            # Si une nouvelle case X est cochée, déselectionner les autres pour X
            selected_x_vars = edited_df[edited_df["Axe X"]]["Variable"].tolist()
            current_x_trait = st.session_state.x_axis_trait_interactive
            new_x_trait = current_x_trait

            if len(selected_x_vars) > 1: # Plus d'une case X cochée
                # Garder la dernière cochée si possible, ou la première de la liste
                # Ou, si l'ancienne était dans la liste, la garder. Sinon, prendre la première.
                if current_x_trait in selected_x_vars:
                    new_x_trait = current_x_trait
                else:
                    new_x_trait = selected_x_vars[0] # Ou la dernière: selected_x_vars[-1]
            elif len(selected_x_vars) == 1:
                new_x_trait = selected_x_vars[0]
            elif len(selected_x_vars) == 0: # Aucune case X cochée
                new_x_trait = None
            
            # Idem pour Y
            selected_y_vars = edited_df[edited_df["Axe Y"]]["Variable"].tolist()
            current_y_trait = st.session_state.y_axis_trait_interactive
            new_y_trait = current_y_trait

            if len(selected_y_vars) > 1:
                if current_y_trait in selected_y_vars:
                    new_y_trait = current_y_trait
                else:
                    new_y_trait = selected_y_vars[0]
            elif len(selected_y_vars) == 1:
                new_y_trait = selected_y_vars[0]
            elif len(selected_y_vars) == 0:
                new_y_trait = None

            # Si les sélections ont changé, mettre à jour l'état et rerun
            if new_x_trait != current_x_trait or new_y_trait != current_y_trait:
                st.session_state.x_axis_trait_interactive = new_x_trait
                st.session_state.y_axis_trait_interactive = new_y_trait

                # Mettre à jour df_editor_source pour refléter la sélection unique
                df_editor_source["Axe X"] = (df_editor_source["Variable"] == new_x_trait)
                df_editor_source["Axe Y"] = (df_editor_source["Variable"] == new_y_trait)
                st.session_state.vip_data_df_interactive = df_editor_source.copy()
                st.session_state.vip_data_df_interactive_snapshot_for_comparison = df_editor_source.copy()
                st.rerun()
            
            # Mettre à jour le snapshot si l'éditeur a été modifié mais n'a pas déclenché de rerun ci-dessus
            # (par exemple, si l'utilisateur a coché puis décoché la même case)
            elif not edited_df.equals(st.session_state.vip_data_df_interactive_snapshot_for_comparison) :
                 st.session_state.vip_data_df_interactive_snapshot_for_comparison = edited_df.copy()


        else:
            st.info("Le tableau d'exploration sera disponible après l'analyse si des traits numériques sont identifiés.")

    with col_interactive_graph:
        st.markdown("##### Graphique d'exploration des variables")
        x_axis_trait_selected_for_plot = st.session_state.x_axis_trait_interactive
        y_axis_trait_selected_for_plot = st.session_state.y_axis_trait_interactive
        
        current_numeric_traits = st.session_state.get('numeric_trait_names_for_interactive_plot', [])
        current_sub_df_interactive = st.session_state.get('sub', pd.DataFrame()) # sub contient les données brutes des traits
        current_pdf_df_interactive = st.session_state.get('pdf', pd.DataFrame()) # pdf contient les infos de cluster et Espece_User

        if not current_numeric_traits:
            st.warning("Aucun trait numérique trouvé pour l'exploration interactive.")
        elif not x_axis_trait_selected_for_plot or not y_axis_trait_selected_for_plot:
            st.info("Veuillez sélectionner une variable pour l'Axe X et une pour l'Axe Y dans le tableau à gauche.")
        elif x_axis_trait_selected_for_plot not in current_numeric_traits or \
             y_axis_trait_selected_for_plot not in current_numeric_traits:
            st.warning("Une ou les deux variables sélectionnées ne sont plus valides (ou pas numériques). Veuillez re-sélectionner.")
        elif current_sub_df_interactive.empty or current_pdf_df_interactive.empty or \
             len(current_sub_df_interactive) != len(current_pdf_df_interactive) or \
             x_axis_trait_selected_for_plot not in current_sub_df_interactive.columns or \
             y_axis_trait_selected_for_plot not in current_sub_df_interactive.columns:
            st.warning("Données pour le graphique interactif non prêtes, incohérentes ou variables sélectionnées non trouvées dans `sub`. Vérifiez l'analyse principale.")
        else:
            # S'assurer que les colonnes de traits existent dans sub_df_interactive
            # Et que les colonnes Espece_User, Ecologie, Cluster existent dans pdf_df_interactive
            required_pdf_cols = ['Espece_User', 'Ecologie', 'Cluster']
            if not all(col in current_pdf_df_interactive.columns for col in required_pdf_cols):
                st.warning("Certaines colonnes requises (Espece_User, Ecologie, Cluster) manquent dans les données PCA.")
            else:
                plot_data_interactive = pd.DataFrame({
                    'Espece_User': current_pdf_df_interactive['Espece_User'].values,
                    'Ecologie': current_pdf_df_interactive['Ecologie'].values,
                    x_axis_trait_selected_for_plot: current_sub_df_interactive[x_axis_trait_selected_for_plot].values.copy(),
                    y_axis_trait_selected_for_plot: current_sub_df_interactive[y_axis_trait_selected_for_plot].values.copy(),
                    'Cluster': current_pdf_df_interactive['Cluster'].values
                })

                plot_data_to_use = plot_data_interactive.copy()
                # Jittering (comme avant)
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
                    x_range_val = x_max_val - x_min_val if pd.notna(x_max_val) and pd.notna(x_min_val) else 0
                    y_range_val = y_max_val - y_min_val if pd.notna(y_max_val) and pd.notna(y_min_val) else 0
                    
                    jitter_strength_x = x_range_val * 0.015 if x_range_val > 1e-9 else (abs(plot_data_to_use[x_axis_trait_selected_for_plot].mean()) * 0.015 if abs(plot_data_to_use[x_axis_trait_selected_for_plot].mean()) > 1e-9 else 0.015)
                    jitter_strength_y = y_range_val * 0.015 if y_range_val > 1e-9 else (abs(plot_data_to_use[y_axis_trait_selected_for_plot].mean()) * 0.015 if abs(plot_data_to_use[y_axis_trait_selected_for_plot].mean()) > 1e-9 else 0.015)
                    if abs(jitter_strength_x) < 1e-9: jitter_strength_x = 0.015 
                    if abs(jitter_strength_y) < 1e-9: jitter_strength_y = 0.015 

                    grouped_for_jitter = plot_data_to_use[duplicates_mask].groupby([temp_x_col_grp, temp_y_col_grp])
                    for _, group in grouped_for_jitter:
                        num_duplicates_in_group = len(group)
                        if num_duplicates_in_group > 1:
                            if not pd.api.types.is_float_dtype(plot_data_to_use[x_axis_trait_selected_for_plot]):
                                plot_data_to_use[x_axis_trait_selected_for_plot] = plot_data_to_use[x_axis_trait_selected_for_plot].astype(float)
                            if not pd.api.types.is_float_dtype(plot_data_to_use[y_axis_trait_selected_for_plot]):
                                plot_data_to_use[y_axis_trait_selected_for_plot] = plot_data_to_use[y_axis_trait_selected_for_plot].astype(float)

                            for i, idx in enumerate(group.index):
                                angle = 2 * np.pi * i / num_duplicates_in_group
                                offset_x = jitter_strength_x * np.cos(angle)
                                offset_y = jitter_strength_y * np.sin(angle)
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
                    textposition="top center", marker=dict(opacity=0.8, size=8),
                    textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS),
                    hovertemplate=(
                        f"<span style='font-size: {HOVER_SPECIES_FONT_SIZE}px;'><b>%{{customdata[0]}}</b></span><br>"
                        f"<br><span style='font-size: {HOVER_ECOLOGY_TITLE_FONT_SIZE}px;'><i>Écologie:</i></span><br>"
                        f"<span style='font-size: {HOVER_ECOLOGY_TEXT_FONT_SIZE}px;'>%{{customdata[1]}}</span>"
                        "<extra></extra>" ))
                # Convex Hull (comme avant)
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
                    xaxis_title=x_axis_trait_selected_for_plot, yaxis_title=y_axis_trait_selected_for_plot)
                fig_interactive_scatter.update_layout(dragmode='pan')
                st.plotly_chart(fig_interactive_scatter, use_container_width=True, config={'scrollZoom': True})

# ---------------------------------------------------------------------------- #
# SECTION 3: COMPOSITION DES CLUSTERS
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty: 
    st.markdown("---")
    st.subheader("Composition des Clusters (ACP)")
    current_cluster_compositions_pdf = st.session_state.get('pdf', pd.DataFrame())
    if not current_cluster_compositions_pdf.empty and \
       'Cluster' in current_cluster_compositions_pdf.columns and \
       'Espece_User' in current_cluster_compositions_pdf.columns:
        
        cluster_compositions_data_display = [
            {"cluster_label": c, 
             "count": len(current_cluster_compositions_pdf.loc[current_cluster_compositions_pdf["Cluster"] == c, "Espece_User"].unique()), 
             "species_list": sorted(list(current_cluster_compositions_pdf.loc[current_cluster_compositions_pdf["Cluster"] == c, "Espece_User"].unique()))
            } for c in sorted(current_cluster_compositions_pdf["Cluster"].unique())
        ]

        if cluster_compositions_data_display and any(d['count'] > 0 for d in cluster_compositions_data_display):
            num_clusters_found_display = len([d for d in cluster_compositions_data_display if d['count']>0]) 
            num_display_cols = min(num_clusters_found_display, 3) if num_clusters_found_display > 0 else 1
            
            cluster_cols = st.columns(num_display_cols)
            current_col_idx = 0
            for comp_data in cluster_compositions_data_display:
                if comp_data['count'] > 0: 
                    with cluster_cols[current_col_idx % num_display_cols]:
                        st.markdown(f"**Cluster {comp_data['cluster_label']}** ({comp_data['count']} espèces)")
                        for species_name in comp_data['species_list']: st.markdown(f"- {species_name}")
                    current_col_idx += 1
            if current_col_idx == 0 : # Si aucun cluster n'avait d'espèces
                 st.info("Aucun cluster (ACP) avec des espèces à afficher.")
        else: 
            st.info("La composition des clusters (ACP) sera affichée ici après l'analyse (pas de données de cluster).")
    else:
         st.info("La composition des clusters (ACP) sera affichée ici après l'analyse (données de PCA non disponibles).")

# ---------------------------------------------------------------------------- #
# SECTION 4: AFFICHAGE DU DENDROGRAMME
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty : 
    st.markdown("---") 
    if fig_dend: # fig_dend est défini globalement et mis à jour dans le bloc d'analyse
        st.plotly_chart(fig_dend, use_container_width=True)
    elif st.session_state.get('X_for_dendro', np.array([])).shape[0] <= 1 and species_binom_user_unique: 
        # species_binom_user_unique est recalculé lors du clic, il faut donc le récupérer depuis l'état si besoin
        # ou se baser sur X_for_dendro
        st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces uniques après traitement ou problème de données pour le linkage).")
    elif st.session_state.run_main_analysis_once : # Si l'analyse a tourné mais fig_dend est None
        st.info("Le dendrogramme n'a pas pu être généré. Vérifiez les données d'entrée (nombre d'espèces > 1, traits numériques).")

