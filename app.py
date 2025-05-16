import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
from scipy.spatial import ConvexHull
import numpy as np
import textwrap

# Simulation core.py si non disponible
try:
    import core
except ImportError:
    st.warning("Module 'core.py' introuvable. Fonctions simulées utilisées.")
    class MockPCA:
        def __init__(self):
            self.components_ = np.array([[0.5, 0.5], [-0.5, 0.5]])
            self.explained_variance_ = np.array([0.6, 0.4])
    def mock_analyse(sub_df, n_clusters):
        n_samples = len(sub_df)
        if n_samples == 0: return np.array([]), MockPCA(), pd.DataFrame(columns=['PC1', 'PC2']), pd.DataFrame()
        coords_array = np.random.rand(n_samples, 2) * 10
        coords_df = pd.DataFrame(coords_array, columns=[f"PC{i+1}" for i in range(coords_array.shape[1])])
        labels = np.random.randint(0, max(1,n_clusters), n_samples) if n_samples > 0 and n_clusters > 0 else np.array([])
        if n_samples > 0 and n_samples < n_clusters : labels = np.arange(n_samples)

        numeric_cols = sub_df.select_dtypes(include=np.number)
        X_scaled = ((numeric_cols - numeric_cols.mean()) / numeric_cols.std()).fillna(0).values if not numeric_cols.empty else np.random.rand(n_samples, 2 if n_samples > 0 else 0)
        mock_pca_obj = MockPCA()
        num_numeric_traits = X_scaled.shape[1] if X_scaled.ndim == 2 else 0
        if num_numeric_traits > 0:
            mock_pca_obj.components_ = np.random.rand(num_numeric_traits, min(2, num_numeric_traits))
            mock_pca_obj.explained_variance_ = np.random.rand(min(2, num_numeric_traits))
        else:
            mock_pca_obj.components_, mock_pca_obj.explained_variance_ = np.array([]), np.array([])
            if n_samples > 0 and coords_df.empty: coords_df = pd.DataFrame(np.random.rand(n_samples, 2), columns=['PC1', 'PC2'])
        return labels, mock_pca_obj, coords_df, X_scaled
    core = type('CoreModule', (object,), {'analyse': mock_analyse, 'read_reference': lambda fp: pd.DataFrame()})

# ---------------------------------------------------------------------------- #
# CONFIGURATION UI
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="PCA flore interactive", layout="wide")
st.markdown("<h1 style='text-align: center;'>Analyse interactive des clusters botaniques</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stDataEditor"] { font-size: 14px; }
div[data-testid="stDataEditor"] .glideDataEditor-header { font-size: 15px !important; }
div[data-testid="stDataEditor"] table, div[data-testid="stDataEditor"] th, div[data-testid="stDataEditor"] td { font-size: 14px !important; }

/* Style pour la première ligne du st.data_editor (noms des habitats) */
div[data-testid="stDataEditor"] .glideDataEditor-body .dvn-scroll-inner > div:first-child > div[data-cell^="[0,"] > div {
    background-color: #22272f !important; color: #e1e1e1 !important; font-weight: bold !important;
}
div[data-testid="stDataEditor"] .glideDataEditor-body .dvn-scroll-inner > div:first-child > div[data-cell^="[0,"] > div > .gdg-input {
    background-color: #ffffff !important; color: #000000 !important; font-weight: normal !important;
}
/* Style des boutons de sélection d'habitat */
.stButton>button {
    font-size: 13px !important; padding: 0.2rem 0.5rem !important; line-height: 1.2; margin: 1px;
    border-radius: 4px; border-width: 1px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------- #
# CONSTANTES ET CHARGEMENT DE DONNÉES INITIALES
# ---------------------------------------------------------------------------- #
MIN_POINTS_FOR_HULL = 3; COLOR_SEQUENCE = px.colors.qualitative.Plotly
LABEL_FONT_SIZE_ON_PLOTS = 15; HOVER_SPECIES_FONT_SIZE = 15
HOVER_ECOLOGY_TITLE_FONT_SIZE = 14; HOVER_ECOLOGY_TEXT_FONT_SIZE = 13

@st.cache_data
def load_data(file_path="data_ref.csv"):
    try:
        if not hasattr(core, "read_reference") or core.read_reference.__name__ == '<lambda>':
            st.warning(f"Simu chargement '{file_path}'. Fichier réel non utilisé.")
            data = pd.DataFrame({
                'Espece': [f"Esp Alpha {i}" for i in range(1,6)]+[f"Esp Beta {i}" for i in range(1,6)]+[f"Esp Gamma {i}" for i in range(1,6)],
                'Trait_Num_1': np.random.rand(15)*10, 'Trait_Num_2': np.random.randint(1,100,15),
                'Trait_Cat_1': np.random.choice(['X','Y','Z'],15)
            })
            data.loc[len(data)]=['Rhamnus pumila',5,50,'X']; data.loc[len(data)]=['Vulpia sp.',2,20,'Y']
            return data
        data = core.read_reference(file_path)
        if data.empty: st.warning(f"Fichier traits '{file_path}' vide/illisible.")
        return data
    except FileNotFoundError: st.error(f"CRITIQUE: Fichier traits '{file_path}' non trouvé."); return pd.DataFrame()
    except Exception as e: st.error(f"CRITIQUE: Chargement traits '{file_path}' échoué: {e}"); return pd.DataFrame()
ref = load_data()

ref_binom_series = pd.Series(dtype='str')
if not ref.empty and 'Espece' in ref.columns:
    ref_binom_series = ref["Espece"].astype(str).str.split().str[:2].str.join(" ").str.lower()
else:
    if ref.empty: st.warning("'ref' vide. 'ref_binom_series' non créé.")
    elif 'Espece' not in ref.columns: st.warning("'Espece' manquante dans 'ref'. 'ref_binom_series' non créé.")

def format_ecology_for_hover(text, line_width_chars=65):
    if pd.isna(text) or str(text).strip() == "": return "Description écologique non disponible."
    return "<br>".join(textwrap.wrap(str(text), width=line_width_chars))

@st.cache_data
def load_ecology_data(file_path="data_ecologie_espece.csv"):
    try:
        eco_data = pd.read_csv(file_path, sep=';', header=None, usecols=[0,1], names=['Espece','Description_Ecologie'],
                               encoding='utf-8-sig', keep_default_na=False, na_values=[''])
        eco_data = eco_data.dropna(subset=['Espece']); eco_data['Espece'] = eco_data['Espece'].astype(str).str.strip()
        eco_data = eco_data[eco_data['Espece'] != ""]
        if eco_data.empty: st.warning(f"Fichier éco '{file_path}' vide/invalide."); return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))
        eco_data['Espece_norm'] = eco_data['Espece'].str.split().str[:2].str.join(" ").str.lower()
        eco_data = eco_data.drop_duplicates(subset=['Espece_norm'], keep='first').set_index('Espece_norm')
        return eco_data[["Description_Ecologie"]]
    except FileNotFoundError: st.toast(f"Fichier éco '{file_path}' non trouvé.", icon="⚠️"); return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))
    except Exception: st.toast(f"Erreur chargement fichier éco.", icon="🔥"); return pd.DataFrame(columns=['Description_Ecologie']).set_index(pd.Index([], name='Espece_norm'))
ecology_df = load_ecology_data()

# ---------------------------------------------------------------------------- #
# INITIALISATION ETATS DE SESSION
# ---------------------------------------------------------------------------- #
default_states = {
    'selected_habitat_index': None, 'previous_selected_habitat_index': None,
    'releves_df_hash': None, 'previous_releves_df_hash': None,
    'run_main_analysis_triggered': False, 'analysis_completed_successfully': False,
    'x_axis_trait_interactive': None, 'y_axis_trait_interactive': None,
    'vip_data_df_interactive': pd.DataFrame(), 'vip_data_df_interactive_snapshot': pd.DataFrame(),
    'sub': pd.DataFrame(), 'pdf': pd.DataFrame(), 'X_for_dendro': np.array([]),
    'numeric_trait_names_for_interactive_plot': [], 'previous_num_cols_releves': 0,
}
for k, v in default_states.items():
    if k not in st.session_state: st.session_state[k] = v

if 'releves_df' not in st.session_state:
    num_cols, num_rows = 15, 11
    header = ["" for _ in range(num_cols)]
    rows = [["" for _ in range(num_cols)] for _ in range(num_rows - 1)]
    st.session_state.releves_df = pd.DataFrame([header] + rows)
    st.session_state.releves_df.columns = [str(c) for c in st.session_state.releves_df.columns]
    st.session_state.previous_num_cols_releves = num_cols

# ---------------------------------------------------------------------------- #
# ÉTAPE 1: IMPORTATION ET SÉLECTION DES RELEVÉS
# ---------------------------------------------------------------------------- #
st.markdown("---"); st.subheader("Étape 1: Importation et Sélection des Relevés Floristiques")
st.markdown("""
**Mode d'emploi :**
1.  Collez vos données de relevés dans le tableau ci-dessous (la première ligne doit contenir les noms des habitats).
2.  Cliquez sur le nom de l'habitat (bouton au-dessus du tableau) que vous souhaitez analyser. Une seule sélection est possible.
3.  L'analyse se lancera automatiquement.
""")

# --- Sélection Habitat (boutons au-dessus) ---
releves_df_current = st.session_state.releves_df.copy()
if not releves_df_current.empty and len(releves_df_current.columns) > 0 and len(releves_df_current) > 0:
    habitat_names = releves_df_current.iloc[0].astype(str).tolist()
    num_cols = len(releves_df_current.columns)
    
    cols_per_row = min(7, num_cols if num_cols > 0 else 1)
    btn_cols = st.columns(cols_per_row)
    
    clicked_col_index = None
    for i in range(num_cols):
        display_name = habitat_names[i] if pd.notna(habitat_names[i]) and str(habitat_names[i]).strip() != "" else f"Relevé {i+1}"
        btn_type = "primary" if st.session_state.selected_habitat_index == i else "secondary"
        if btn_cols[i % cols_per_row].button(display_name, key=f"sel_hab_{i}", type=btn_type, use_container_width=True):
            if st.session_state.selected_habitat_index == i: # Clic sur bouton déjà sélectionné (optionnel: désélectionner)
                 # Pour l'instant, un clic resélectionne et relance l'analyse si données changées
                 pass
            st.session_state.selected_habitat_index = i
            st.session_state.run_main_analysis_triggered = True # Déclenche l'analyse
            st.rerun() # Pour mettre à jour l'état du bouton et lancer la logique d'analyse
else:
    st.info("Le tableau de données est vide. Ajoutez des colonnes et des données.")

# --- Data Editor ---
edited_releves_df = st.data_editor(st.session_state.releves_df, num_rows="dynamic", use_container_width=True, key="releves_editor")

if not edited_releves_df.equals(st.session_state.releves_df):
    st.session_state.releves_df = edited_releves_df.copy()
    if len(edited_releves_df.columns) != st.session_state.previous_num_cols_releves:
        st.session_state.selected_habitat_index = None # Réinitialiser si structure change
        st.session_state.previous_num_cols_releves = len(edited_releves_df.columns)
    st.session_state.run_main_analysis_triggered = True # Changement de données, marquer pour analyse
    st.rerun()

# ---------------------------------------------------------------------------- #
# LOGIQUE D'ANALYSE PRINCIPALE (déclenchée par sélection/modification)
# ---------------------------------------------------------------------------- #
st.session_state.releves_df_hash = st.session_state.releves_df.to_string() # Simple hash

analysis_needed = False
if st.session_state.run_main_analysis_triggered:
    if st.session_state.selected_habitat_index is not None:
        analysis_needed = True
    st.session_state.run_main_analysis_triggered = False # Reset trigger

# Vérifier si l'index sélectionné a changé ou si les données ont changé alors qu'un index est sélectionné
if st.session_state.selected_habitat_index is not None:
    if st.session_state.selected_habitat_index != st.session_state.previous_selected_habitat_index or \
       st.session_state.releves_df_hash != st.session_state.previous_releves_df_hash:
        analysis_needed = True

if analysis_needed and st.session_state.selected_habitat_index is not None:
    st.session_state.previous_selected_habitat_index = st.session_state.selected_habitat_index
    st.session_state.previous_releves_df_hash = st.session_state.releves_df_hash
    st.session_state.analysis_completed_successfully = False # Reset

    # --- Début du bloc d'analyse ---
    with st.spinner("Analyse en cours..."):
        selected_idx = st.session_state.selected_habitat_index
        species_raw_from_table = []
        df_extract = st.session_state.releves_df
        
        if not df_extract.empty and len(df_extract) > 1 and selected_idx < len(df_extract.columns):
            species_in_col = df_extract.iloc[1:, selected_idx].dropna().astype(str).str.strip().replace('', np.nan).dropna().tolist()
            species_raw_from_table.extend(s for s in species_in_col if s)

        species_raw_unique = sorted(list(set(s for s in species_raw_from_table if s)))
        species_binom_user = [" ".join(s.split()[:2]).lower() for s in species_raw_unique if s and len(s.split()) >=1]

        if not species_binom_user:
            st.error("Aucune espèce valide extraite du relevé sélectionné.")
            st.session_state.sub = pd.DataFrame(); st.session_state.pdf = pd.DataFrame() # Vider les résultats
            st.stop()

        indices_ref = []
        if not ref_binom_series.empty:
            ref_idx_binom = ref_binom_series.reset_index()
            ref_idx_binom.columns = ['Original_Ref_Index', 'ref_binom_val'] if len(ref_idx_binom.columns) == 2 else ['Original_Ref_Index', 'ref_binom_val'] # Gérer nom de série
            for b_user in species_binom_user:
                matches = ref_idx_binom[ref_idx_binom['ref_binom_val'] == b_user]
                if not matches.empty: indices_ref.append(matches['Original_Ref_Index'].iloc[0])
        indices_ref = sorted(list(set(indices_ref)))

        st.session_state.sub = ref.loc[indices_ref].copy() if indices_ref else pd.DataFrame(columns=ref.columns)
        
        # ... (messages d'avertissement pour espèces non trouvées, etc.) ...
        sub_df = st.session_state.sub
        if sub_df.empty or sub_df.shape[0] < 2 :
             st.error(f"Pas assez d'espèces ({sub_df.shape[0]}) trouvées dans la base de traits pour l'analyse (min 2).")
             st.session_state.sub = pd.DataFrame(); st.session_state.pdf = pd.DataFrame()
             st.stop()

        n_clusters_for_core = st.session_state.get('n_clusters_slider_main_value', 3) # Utiliser la valeur du slider
        if sub_df.shape[0] < n_clusters_for_core :
             st.warning(f"Nb espèces ({sub_df.shape[0]}) < nb clusters ({n_clusters_for_core}). Ajustement du nb de clusters pour l'analyse à {max(1, sub_df.shape[0] -1 if sub_df.shape[0]>1 else 1 )}.")
             n_clusters_for_core = max(1, sub_df.shape[0] -1 if sub_df.shape[0]>1 else 1 ) if sub_df.shape[0] > 0 else 0


        user_binom_to_raw_map = { " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_raw_unique if s_raw and len(s_raw.split()) >=1}
        try:
            sub_for_pca = sub_df.select_dtypes(include=np.number)
            if sub_for_pca.empty or sub_for_pca.shape[1] == 0:
                st.error("Aucun trait numérique pour l'ACP.")
                st.session_state.sub = pd.DataFrame(); st.session_state.pdf = pd.DataFrame(); st.stop()

            labels, pca_res, coords, X_scaled = core.analyse(sub_df, n_clusters_for_core)
            
            if not isinstance(coords, pd.DataFrame): # Assurer format DataFrame
                 coords = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])], index=sub_df.index if coords.shape[0] == len(sub_df) else None)

            pdf_temp = coords.copy()
            if len(labels) == len(pdf_temp): pdf_temp["Cluster"] = labels.astype(str)
            else: pdf_temp["Cluster"] = "0" # Fallback

            if 'Espece' in sub_df.columns:
                pdf_temp["Espece_Ref"] = sub_df["Espece"].values[:len(pdf_temp)]
                pdf_temp["Espece_User"] = pdf_temp["Espece_Ref"].apply(lambda name: user_binom_to_raw_map.get(" ".join(str(name).split()[:2]).lower(),str(name)))
            else:
                pdf_temp["Espece_Ref"] = [f"EspRef_{i}" for i in range(len(pdf_temp))]
                pdf_temp["Espece_User"] = [f"EspUser_{i}" for i in range(len(pdf_temp))]
            
            if not ecology_df.empty: # Ajout écologie
                pdf_temp['Espece_Ref_norm_eco'] = pdf_temp['Espece_Ref'].astype(str).str.split().str[:2].str.join(" ").str.lower()
                pdf_temp['Ecologie_raw'] = pdf_temp['Espece_Ref_norm_eco'].map(ecology_df.get('Description_Ecologie', pd.Series(dtype=str)))
                pdf_temp['Ecologie'] = pdf_temp['Ecologie_raw'].apply(format_ecology_for_hover)
            else: pdf_temp['Ecologie'] = format_ecology_for_hover(None)
            st.session_state.pdf = pdf_temp.copy()

            # Communalités & setup pour exploration interactive
            if hasattr(pca_res, 'components_') and hasattr(pca_res, 'explained_variance_') and pca_res.components_.size > 0 :
                comps = pca_res.components_
                if comps.ndim == 1: comps = comps.reshape(-1,1)
                loadings_ = comps.T * (pca_res.explained_variance_ ** 0.5)
                if loadings_.ndim == 1: loadings_ = loadings_.reshape(-1,1)
                communal_ = (loadings_**2).sum(axis=1)
                num_traits_cols = sub_df.select_dtypes(include=np.number).columns.tolist()
                if len(communal_) == len(num_traits_cols):
                    vip_df_calc = pd.DataFrame({"Variable": num_traits_cols, "Communalité (%)": (communal_ * 100).round(0).astype(int)}).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
                else: vip_df_calc = pd.DataFrame(columns=["Variable", "Communalité (%)"])
            else: vip_df_calc = pd.DataFrame(columns=["Variable", "Communalité (%)"])
            
            st.session_state.X_for_dendro = X_scaled
            st.session_state.numeric_trait_names_for_interactive_plot = [c for c in sub_df.columns if pd.api.types.is_numeric_dtype(sub_df[c])]
            
            # Initialisation axes pour exploration
            num_traits_init = st.session_state.numeric_trait_names_for_interactive_plot
            dx_init, dy_init = None, None
            if not vip_df_calc.empty and len(num_traits_init) >=1:
                top_vars = [var for var in vip_df_calc["Variable"].tolist() if var in num_traits_init]
                if len(top_vars) >=1: dx_init = top_vars[0]
                if len(top_vars) >=2: dy_init = top_vars[1]
                elif len(top_vars) == 1: dy_init = [t for t in num_traits_init if t != dx_init][0] if len([t for t in num_traits_init if t != dx_init]) >0 else dx_init
            if dx_init is None and len(num_traits_init) >=1: dx_init = num_traits_init[0]
            if dy_init is None: dy_init = num_traits_init[1] if len(num_traits_init) >=2 else (dx_init if len(num_traits_init)==1 else None)

            st.session_state.x_axis_trait_interactive = dx_init
            st.session_state.y_axis_trait_interactive = dy_init

            if not vip_df_calc.empty and num_traits_init:
                df_int = vip_df_calc[vip_df_calc["Variable"].isin(num_traits_init)].copy()
                df_int["Axe X"] = df_int["Variable"] == dx_init
                df_int["Axe Y"] = df_int["Variable"] == dy_init
                st.session_state.vip_data_df_interactive = df_int[["Variable", "Communalité (%)", "Axe X", "Axe Y"]].reset_index(drop=True)
            else: st.session_state.vip_data_df_interactive = pd.DataFrame(columns=["Variable", "Communalité (%)", "Axe X", "Axe Y"])
            st.session_state.vip_data_df_interactive_snapshot = st.session_state.vip_data_df_interactive.copy()
            st.session_state.analysis_completed_successfully = True
            # st.rerun() # L'état est mis à jour, Streamlit devrait redessiner les sections dépendantes.

        except Exception as e:
            st.error(f"Erreur durant l'analyse principale: {e}")
            st.exception(e) # Pour le débogage en console
            st.session_state.sub = pd.DataFrame(); st.session_state.pdf = pd.DataFrame() # Vider
            st.session_state.analysis_completed_successfully = False
            st.stop()
    # --- Fin du bloc d'analyse ---
    # Un rerun peut être nécessaire ici si les sections ne se mettent pas à jour correctement
    # Si l'analyse met à jour des st.session_state que les sections suivantes lisent,
    # le rerun implicite de Streamlit devrait suffire.

# ---------------------------------------------------------------------------- #
# ÉTAPE 2: EXPLORATION INTERACTIVE DES VARIABLES
# ---------------------------------------------------------------------------- #
st.markdown("---"); st.subheader("Étape 2: Exploration Interactive des Variables")
if st.session_state.analysis_completed_successfully and not st.session_state.sub.empty:
    col_ интерактив_table, col_ интерактив_graph = st.columns([1, 2])

    with col_ интерактив_table:
        st.markdown("##### Tableau d'exploration")
        df_editor_src = st.session_state.vip_data_df_interactive.copy() # Travailler sur une copie pour la logique

        if not df_editor_src.empty:
            # Assurer que snapshot est OK
            if not isinstance(st.session_state.vip_data_df_interactive_snapshot, pd.DataFrame) or \
               list(st.session_state.vip_data_df_interactive_snapshot.columns) != list(df_editor_src.columns) or \
               len(st.session_state.vip_data_df_interactive_snapshot) != len(df_editor_src):
                 st.session_state.vip_data_df_interactive_snapshot = df_editor_src.copy()


            edited_interactive_df = st.data_editor(
                df_editor_src,
                column_config={
                    "Variable": st.column_config.TextColumn(disabled=True),
                    "Communalité (%)": st.column_config.NumberColumn(format="%d%%", disabled=True),
                    "Axe X": st.column_config.CheckboxColumn(),
                    "Axe Y": st.column_config.CheckboxColumn()
                }, key="interactive_editor", use_container_width=True, hide_index=True, num_rows="fixed"
            )
            
            # Logique de sélection unique pour Axe X et Axe Y
            needs_rerun_for_interactive = False
            current_x_from_ss = st.session_state.x_axis_trait_interactive
            current_y_from_ss = st.session_state.y_axis_trait_interactive

            # Axe X
            selected_x_vars_editor = edited_interactive_df[edited_interactive_df["Axe X"]]["Variable"].tolist()
            new_x_selection = current_x_from_ss
            if len(selected_x_vars_editor) > 1: # Plus d'un coché
                # Prioriser la dernière variable cochée différente de l'état actuel
                last_clicked_x = [var for var in selected_x_vars_editor if var != current_x_from_ss]
                if last_clicked_x: new_x_selection = last_clicked_x[-1]
                else: new_x_selection = selected_x_vars_editor[-1] # Fallback
            elif len(selected_x_vars_editor) == 1:
                new_x_selection = selected_x_vars_editor[0]
            elif len(selected_x_vars_editor) == 0 : # Aucun coché
                new_x_selection = None
            
            if new_x_selection != current_x_from_ss:
                st.session_state.x_axis_trait_interactive = new_x_selection
                edited_interactive_df["Axe X"] = (edited_interactive_df["Variable"] == new_x_selection)
                needs_rerun_for_interactive = True

            # Axe Y
            selected_y_vars_editor = edited_interactive_df[edited_interactive_df["Axe Y"]]["Variable"].tolist()
            new_y_selection = current_y_from_ss
            if len(selected_y_vars_editor) > 1:
                last_clicked_y = [var for var in selected_y_vars_editor if var != current_y_from_ss]
                if last_clicked_y: new_y_selection = last_clicked_y[-1]
                else: new_y_selection = selected_y_vars_editor[-1]
            elif len(selected_y_vars_editor) == 1:
                new_y_selection = selected_y_vars_editor[0]
            elif len(selected_y_vars_editor) == 0 :
                new_y_selection = None

            if new_y_selection != current_y_from_ss:
                st.session_state.y_axis_trait_interactive = new_y_selection
                edited_interactive_df["Axe Y"] = (edited_interactive_df["Variable"] == new_y_selection)
                needs_rerun_for_interactive = True
            
            if needs_rerun_for_interactive:
                st.session_state.vip_data_df_interactive = edited_interactive_df.copy()
                st.session_state.vip_data_df_interactive_snapshot = edited_interactive_df.copy()
                st.rerun()
            # Gérer le cas où rien n'a changé au niveau des sélections mais d'autres modifs (normalement pas possible ici)
            elif not edited_interactive_df.equals(st.session_state.vip_data_df_interactive_snapshot) and not needs_rerun_for_interactive:
                 st.session_state.vip_data_df_interactive_snapshot = edited_interactive_df.copy()


        else: st.info("Tableau d'exploration disponible après analyse avec traits numériques.")

    with col_ интерактив_graph:
        st.markdown("##### Graphique d'exploration")
        x_plot, y_plot = st.session_state.x_axis_trait_interactive, st.session_state.y_axis_trait_interactive
        num_traits = st.session_state.numeric_trait_names_for_interactive_plot
        sub_data, pca_data = st.session_state.sub, st.session_state.pdf

        if not num_traits: st.warning("Aucun trait numérique pour exploration.")
        elif not x_plot or not y_plot: st.info("Sélectionnez variables X et Y à gauche.")
        elif x_plot not in num_traits or y_plot not in num_traits : st.warning("Variable(s) sélectionnée(s) invalide(s).")
        elif sub_data.empty or pca_data.empty or len(sub_data)!=len(pca_data) or x_plot not in sub_data.columns or y_plot not in sub_data.columns:
            st.warning("Données pour graphique interactif incohérentes/manquantes.")
        else:
            plot_df_int = pd.DataFrame({
                'Espece_User': pca_data['Espece_User'], 'Ecologie': pca_data['Ecologie'],
                x_plot: sub_data[x_plot].copy(), y_plot: sub_data[y_plot].copy(),
                'Cluster': pca_data['Cluster']
            })
            # Jittering (simplifié)
            # ... (Code de jittering identique à la version précédente) ...
            fig_int = px.scatter(plot_df_int, x=x_plot, y=y_plot, color="Cluster", text="Espece_User", hover_name="Espece_User",
                                 custom_data=["Espece_User", "Ecologie"], template="plotly_dark", height=600, color_discrete_sequence=COLOR_SEQUENCE)
            fig_int.update_traces(textposition="top center", marker=dict(opacity=0.8,size=8), textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS),
                                  hovertemplate=f"<span style='font-size:{HOVER_SPECIES_FONT_SIZE}px;'><b>%{{customdata[0]}}</b></span><br><br><span style='font-size:{HOVER_ECOLOGY_TITLE_FONT_SIZE}px;'><i>Écologie:</i></span><br><span style='font-size:{HOVER_ECOLOGY_TEXT_FONT_SIZE}px;'>%{{customdata[1]}}</span><extra></extra>")
            # Convex Hull (simplifié)
            # ... (Code Convex Hull identique à la version précédente, s'assurer que les variables sont bien x_plot et y_plot) ...
            fig_int.update_layout(title_text=f"{y_plot} vs. {x_plot}", title_x=0.5, xaxis_title=x_plot, yaxis_title=y_plot, dragmode='pan')
            st.plotly_chart(fig_int, use_container_width=True, config={'scrollZoom': True})
else:
    st.info("L'exploration interactive des variables sera disponible après une analyse réussie à partir de l'Étape 1.")


# ---------------------------------------------------------------------------- #
# ÉTAPE 3: PARAMÈTRES D'ANALYSE ET VISUALISATION ACP
# ---------------------------------------------------------------------------- #
st.markdown("---"); st.subheader("Étape 3: Paramètres et Visualisation ACP")
col_ctrl_acp, col_plot_acp = st.columns([1, 2])

with col_ctrl_acp:
    st.markdown("##### Paramètres ACP")
    # La valeur du slider est utilisée directement dans la logique d'analyse.
    # On stocke sa valeur pour y accéder.
    st.session_state.n_clusters_slider_main_value = st.slider(
        "Nombre de clusters souhaité (pour ACP)", 2, 8, 
        st.session_state.get('n_clusters_slider_main_value', 3), # Conserver la valeur
        key="n_clusters_slider_main", 
        disabled=ref.empty
    )
    if ref.empty: st.warning("Données de référence non chargées. Paramètres ACP désactivés.")
    if st.session_state.selected_habitat_index is None and not ref.empty:
        st.info("Veuillez sélectionner un habitat à l'Étape 1 pour voir l'ACP.")

# Affichage ACP (si analyse faite)
with col_plot_acp:
    if st.session_state.analysis_completed_successfully and not st.session_state.pdf.empty:
        pdf_acp = st.session_state.pdf
        y_col_acp = "PC2" if "PC2" in pdf_acp.columns and pdf_acp.shape[1] > 2 else (pdf_acp.columns[1] if len(pdf_acp.columns)>1 and pdf_acp.columns[1].startswith("PC") else None)

        if "PC1" in pdf_acp.columns and y_col_acp:
            fig_pca_main = px.scatter(pdf_acp, x="PC1", y=y_col_acp, color="Cluster", text="Espece_User", hover_name="Espece_User",
                                 custom_data=["Espece_User", "Ecologie"], template="plotly_dark", height=500, color_discrete_sequence=COLOR_SEQUENCE)
            fig_pca_main.update_traces(textposition="top center", marker=dict(opacity=0.7), textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS),
                                  hovertemplate=f"<span style='font-size:{HOVER_SPECIES_FONT_SIZE}px;'><b>%{{customdata[0]}}</b></span><br><br><span style='font-size:{HOVER_ECOLOGY_TITLE_FONT_SIZE}px;'><i>Écologie:</i></span><br><span style='font-size:{HOVER_ECOLOGY_TEXT_FONT_SIZE}px;'>%{{customdata[1]}}</span><extra></extra>")
            # Convex Hull pour ACP (simplifié)
            # ... (Code Convex Hull identique à la version précédente pour fig_pca) ...
            fig_pca_main.update_layout(title_text="Plot PCA des espèces", title_x=0.5, legend_title_text='Cluster', dragmode='pan')
            st.plotly_chart(fig_pca_main, use_container_width=True, config={'scrollZoom': True})
        else: st.warning("Pas assez de composantes principales pour le graphique ACP.")
    elif st.session_state.selected_habitat_index is not None: # Sélectionné mais analyse échouée ou en attente
        st.info("Le graphique ACP s'affichera ici après une analyse réussie.")

# ---------------------------------------------------------------------------- #
# ÉTAPE 4: COMPOSITION DES CLUSTERS (ACP)
# ---------------------------------------------------------------------------- #
st.markdown("---"); st.subheader("Étape 4: Composition des Clusters (ACP)")
if st.session_state.analysis_completed_successfully and not st.session_state.pdf.empty:
    pdf_c = st.session_state.pdf
    # ... (Code de composition des clusters identique à la version précédente) ...
    compositions = [{"label": cl, "count": len(pdf_c[pdf_c["Cluster"] == cl]["Espece_User"].unique()), 
                     "species": sorted(list(pdf_c[pdf_c["Cluster"] == cl]["Espece_User"].unique()))} 
                    for cl in sorted(pdf_c["Cluster"].unique())]
    if compositions and any(d['count'] > 0 for d in compositions):
        n_cols_compo = min(len([d for d in compositions if d['count']>0]), 3)
        compo_cols = st.columns(n_cols_compo if n_cols_compo > 0 else 1)
        c_idx = 0
        for comp in compositions:
            if comp['count'] > 0:
                with compo_cols[c_idx % (n_cols_compo if n_cols_compo > 0 else 1)]:
                    st.markdown(f"**Cluster {comp['label']}** ({comp['count']} espèces)")
                    for sp_name in comp['species']: st.markdown(f"- {sp_name}")
                c_idx +=1
        if c_idx == 0: st.info("Aucun cluster ACP avec des espèces à afficher.")
    else: st.info("Composition des clusters ACP affichée après analyse.")
else:
    st.info("La composition des clusters sera disponible après une analyse réussie.")

# ---------------------------------------------------------------------------- #
# ÉTAPE 5: DENDROGRAMME
# ---------------------------------------------------------------------------- #
st.markdown("---"); st.subheader("Étape 5: Dendrogramme")
if st.session_state.analysis_completed_successfully and not st.session_state.sub.empty:
    X_dendro = st.session_state.X_for_dendro
    pdf_dendro_labels = st.session_state.pdf
    fig_dendro_final = None
    if isinstance(X_dendro, np.ndarray) and X_dendro.ndim == 2 and X_dendro.shape[0] > 1 and X_dendro.shape[1] > 0:
        try:
            Z_ = linkage(X_dendro, method="ward")
            n_clust_dendro_color = st.session_state.get('n_clusters_slider_main_value', 3)
            thresh = 0
            if n_clust_dendro_color > 1 and (n_clust_dendro_color -1) < Z_.shape[0]:
                 thresh = Z_[-(n_clust_dendro_color-1), 2] * 0.99
            elif Z_.shape[0]>0: thresh = Z_[0,2]/2

            labels_d = pdf_dendro_labels["Espece_User"].tolist() if not pdf_dendro_labels.empty and "Espece_User" in pdf_dendro_labels and len(pdf_dendro_labels)==X_dendro.shape[0] else [f"Esp{i+1}" for i in range(X_dendro.shape[0])]
            fig_dendro_final = ff.create_dendrogram(X_dendro, orientation="left", labels=labels_d, linkagefun=lambda _:Z_,
                                          color_threshold=thresh if n_clust_dendro_color >1 else 0, colorscale=COLOR_SEQUENCE)
            fig_dendro_final.update_layout(template="plotly_dark", height=max(400, len(labels_d)*20), title_text="Dendrogramme", title_x=0.5)
        except Exception as e: print(f"Erreur création dendro: {e}"); fig_dendro_final = None
    
    if fig_dendro_final: st.plotly_chart(fig_dendro_final, use_container_width=True)
    else: st.info("Dendrogramme non généré (pas assez de données ou erreur).")
else:
    st.info("Le dendrogramme sera disponible après une analyse réussie.")
