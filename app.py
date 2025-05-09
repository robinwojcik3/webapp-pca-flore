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
# import core # Commentez ou décommentez si core.py est disponible

# ---------------------------------------------------------------------------- #
# SIMULATION DE core.py pour l'exemple (si core.py n'est pas fourni)
# ---------------------------------------------------------------------------- #
# Décommentez cette section si vous n'avez pas core.py et souhaitez exécuter l'exemple
# Vous devrez adapter les données de sortie pour qu'elles correspondent à ce que votre core.py produirait.
if 'core' not in globals():
    print("Simulation de core.py car le module n'est pas trouvé.")
    class PCA: # Simulation de l'objet PCA
        def __init__(self, n_components):
            self.n_components_ = n_components
            self.components_ = np.random.rand(n_components, 5) # Supposons 5 traits
            self.explained_variance_ = np.random.rand(n_components)

    def analyse_simulation(sub_df, n_clusters):
        n_samples = len(sub_df)
        if n_samples == 0:
            raise ValueError("Le DataFrame d'entrée 'sub_df' est vide.")
        
        # Simuler les labels de cluster
        if n_samples < n_clusters :
             labels = np.arange(n_samples) # Moins d'échantillons que de clusters demandés
        else:
             labels = np.random.randint(0, n_clusters, n_samples)


        # Simuler les coordonnées PCA (PC1, PC2)
        # Utiliser les données de sub_df si possible pour rendre la simulation un peu plus réaliste
        # Ici, nous allons juste générer des données aléatoires pour les coordonnées
        coords = np.random.rand(n_samples, 2) * 10 # Deux composantes principales

        # Simuler l'objet PCA
        # Le nombre de composants de l'ACP doit correspondre au nombre de traits numériques dans sub_df
        # Exclure la colonne 'Espece' et ne garder que les traits numériques pour déterminer n_features
        numeric_trait_cols = [col for col in sub_df.columns if col.lower() != 'espece' and pd.api.types.is_numeric_dtype(sub_df[col])]
        n_features = len(numeric_trait_cols)
        if n_features == 0: # S'il n'y a pas de traits numériques, l'ACP ne peut pas vraiment être simulée correctement
            print("Avertissement: Aucun trait numérique trouvé dans sub_df pour la simulation PCA.")
            # Créer un objet PCA avec un nombre arbitraire de composants si aucun trait numérique
            pca_instance = PCA(n_components=2)
            pca_instance.components_ = np.random.rand(2,1) # Simuler des composants pour une caractéristique fictive
            pca_instance.explained_variance_ = np.random.rand(2)

        else:
            pca_instance = PCA(n_components=min(2, n_features)) # Simuler jusqu'à 2 composantes ou n_features
            pca_instance.components_ = np.random.rand(pca_instance.n_components_, n_features)
            pca_instance.explained_variance_ = np.random.rand(pca_instance.n_components_)


        # Simuler X (données utilisées pour le dendrogramme, souvent les données normalisées ou les coordonnées PCA)
        # Ici, nous utilisons les coordonnées PCA simulées comme X pour simplifier
        X_for_dendro = coords.copy()

        return labels, pca_instance, coords, X_for_dendro

    class CoreSimulator:
        @staticmethod
        def read_reference(file_path):
            # Simuler la lecture de data_ref.csv
            # Créez un DataFrame de démonstration si le fichier n'existe pas
            try:
                return pd.read_csv(file_path, sep=';', encoding='utf-8-sig')
            except FileNotFoundError:
                print(f"Fichier {file_path} non trouvé, création de données de démo pour ref.")
                data = {
                    'Espece': [f'Genre{i} espece{i} var{i}' for i in range(1, 21)],
                    'TraitA': np.random.rand(20) * 10,
                    'TraitB': np.random.randint(1, 100, 20),
                    'TraitC': np.random.choice(['Faible', 'Moyen', 'Fort'], 20),
                    'Humidite_atmospherique': np.random.rand(20) * 100,
                    'Matiere_organique': np.random.rand(20) * 20,
                    'Niveau_trophique': np.random.rand(20) * 5,
                    'Humidite_edaphique': np.random.rand(20) * 80,
                    'Temperature': np.random.rand(20) * 30,
                    'Texture': np.random.rand(20) * 10,
                    'Continentalite': np.random.rand(20) * 7,
                    'Lumiere': np.random.rand(20) * 10000,
                    'Salinite': np.random.rand(20) * 3,
                    'Reaction_du_sol_pH': np.random.rand(20) * 5 + 4, # pH entre 4 et 9
                }
                return pd.DataFrame(data)

        @staticmethod
        def analyse(sub_df, n_clusters):
            return analyse_simulation(sub_df, n_clusters)

    core = CoreSimulator() # Remplacer le module core par notre simulateur
# FIN DE LA SIMULATION de core.py
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# CONFIGURATION UI
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="PCA flore interactive", layout="wide")
st.markdown("<h1 style='text-align: center;'>Analyse interactive des clusters botaniques</h1>", unsafe_allow_html=True)

# CSS pour la taille de police du data_editor (tentative)
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
LABEL_FONT_SIZE_ON_PLOTS = 13 

@st.cache_data
def load_data(file_path="data_ref.csv"):
    """Charge les données de référence (traits) à partir du chemin spécifié."""
    try:
        # Utilise core.read_reference qui est soit le vrai, soit le simulateur
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
        print(f"AVERTISSEMENT: Fichier de données écologiques '{file_path}' non trouvé. Création de données de démo pour l'écologie.")
        st.toast(f"Fichier écologique '{file_path}' non trouvé. Utilisation de données de démo.", icon="⚠️")
        # Créer un DataFrame de démo pour l'écologie
        demo_eco_data = {
            'Espece_norm': [name.lower() for name in ref_binom_series.unique() if pd.notna(name)],
            'Description_Ecologie': [f"Description écologique simulée pour {name.split()[0]} {name.split()[1]}." for name in ref_binom_series.unique() if pd.notna(name)]
        }
        if demo_eco_data['Espece_norm']: # S'assurer qu'il y a des espèces de référence
            df_demo_eco = pd.DataFrame(demo_eco_data)
            df_demo_eco = df_demo_eco.set_index('Espece_norm')
            return df_demo_eco[["Description_Ecologie"]]
        return pd.DataFrame(columns=['Description_Ecologie']) # Retourner vide si pas d'espèces de réf

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
if 'numeric_trait_names_for_interactive_plot' not in st.session_state:
    st.session_state.numeric_trait_names_for_interactive_plot = []
if 'vip_data_df_for_calc' not in st.session_state: # Assurer l'initialisation
    st.session_state.vip_data_df_for_calc = pd.DataFrame()


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
        # valeur par défaut pour test rapide
        value="Genre1 espece1\nGenre2 espece2\nGenre3 espece3\nGenre4 espece4\nGenre5 espece5\nGenre6 espece6" if 'core' not in globals() else "", 
        disabled=ref.empty
    )
    species_raw_unique = sorted(list(set(s.strip() for s in species_txt.splitlines() if s.strip())))
    species_binom_user_unique = [" ".join(s.split()[:2]).lower() for s in species_raw_unique]

    run_main_analysis_button = st.button("Lancer l'analyse principale", type="primary", disabled=ref.empty, key="main_analysis_button")

fig_pca = None
fig_dend = None
cluster_compositions_data = [] # Initialisation ici
sub = pd.DataFrame() # Initialisation ici
pdf = pd.DataFrame() # Initialisation ici
X_for_dendro = np.array([]) # Initialisation ici


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
    sub = st.session_state.sub # Rendre 'sub' accessible localement

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
    if sub.shape[0] < 2: # Dendrogramme et ACP nécessitent au moins 2 points
        st.error(f"Au moins 2 espèces uniques sont nécessaires pour l'analyse. {sub.shape[0]} espèce(s) trouvée(s) et utilisée(s).");
        st.session_state.run_main_analysis_once = False; st.stop()

    user_input_binom_to_raw_map = { " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_raw_unique }
    try:
        # Utilise core.analyse qui est soit le vrai, soit le simulateur
        labels, pca, coords, X = core.analyse(sub, n_clusters_selected)
        st.session_state.X_for_dendro = X # Sauvegarder X pour le dendrogramme

        current_pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
        current_pdf["Cluster"] = labels.astype(str)
        current_pdf["Espece_Ref"] = sub["Espece"].values # Assurer que l'index correspond
        current_pdf["Espece_User"] = current_pdf["Espece_Ref"].apply(lambda full_ref_name: user_input_binom_to_raw_map.get(" ".join(full_ref_name.split()[:2]).lower(), full_ref_name))

        if not ecology_df.empty:
            current_pdf['Espece_Ref_norm_for_eco'] = ( current_pdf['Espece_Ref'].astype(str).str.strip().str.split().str[:2].str.join(" ").str.lower() )
            current_pdf['Ecologie_raw'] = current_pdf['Espece_Ref_norm_for_eco'].map(ecology_df['Description_Ecologie'])
            current_pdf['Ecologie'] = current_pdf['Ecologie_raw'].apply(lambda x: format_ecology_for_hover(x))
            current_pdf['Ecologie'] = current_pdf['Ecologie'].fillna(format_ecology_for_hover("Description écologique non disponible."))
        else:
            current_pdf['Ecologie'] = format_ecology_for_hover("Description écologique non disponible (fichier non chargé ou vide).")
        st.session_state.pdf = current_pdf # Sauvegarder pdf pour les graphiques

        # Calcul des communalités
        # S'assurer que pca.components_ et pca.explained_variance_ sont disponibles et corrects
        if hasattr(pca, 'components_') and hasattr(pca, 'explained_variance_'):
            loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
            communal = (loadings**2).sum(axis=1)
            trait_columns_for_communalities = [col for col in sub.columns if col.lower() != "espece" and pd.api.types.is_numeric_dtype(sub[col])] # Uniquement traits numériques

            # S'assurer que communal a la bonne longueur par rapport à trait_columns_for_communalities
            if len(communal) == len(trait_columns_for_communalities):
                st.session_state.vip_data_df_for_calc = pd.DataFrame({
                    "Variable": trait_columns_for_communalities,
                    "Communalité (%)": (communal * 100).round(0).astype(int),
                }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
            else:
                st.warning(f"Discordance de longueur pour communalités: {len(communal)} vs {len(trait_columns_for_communalities)} traits. Tableau des communalités pourrait être incorrect.")
                # Créer un DataFrame vide ou avec des valeurs par défaut pour éviter des erreurs en aval
                st.session_state.vip_data_df_for_calc = pd.DataFrame({"Variable": trait_columns_for_communalities, "Communalité (%)": 0})

        else:
            st.warning("Attributs 'components_' ou 'explained_variance_' non trouvés dans l'objet PCA. Communalités non calculées.")
            st.session_state.vip_data_df_for_calc = pd.DataFrame(columns=["Variable", "Communalité (%)"])


        # Détermination des traits numériques et des axes par défaut pour le graphique interactif
        all_trait_names_from_sub = [col for col in sub.columns if col.lower() != "espece"]
        st.session_state.numeric_trait_names_for_interactive_plot = [
            col for col in all_trait_names_from_sub if pd.api.types.is_numeric_dtype(sub[col])
        ]
        
        numeric_trait_names_init = st.session_state.numeric_trait_names_for_interactive_plot
        default_x_init, default_y_init = None, None

        # Utiliser vip_data_df_for_calc qui contient uniquement des variables numériques pour les communalités
        if not st.session_state.vip_data_df_for_calc.empty:
            # Les variables dans vip_data_df_for_calc sont déjà numériques et triées par communalité
            top_vars_from_vip_numeric = st.session_state.vip_data_df_for_calc["Variable"].tolist()
            
            if len(top_vars_from_vip_numeric) >= 1: default_x_init = top_vars_from_vip_numeric[0]
            if len(top_vars_from_vip_numeric) >= 2: default_y_init = top_vars_from_vip_numeric[1]
            elif len(top_vars_from_vip_numeric) == 1: # Si une seule var numérique avec communalité
                 # Essayer de trouver une autre var numérique globale si disponible
                other_numeric_overall = [t for t in numeric_trait_names_init if t != default_x_init]
                default_y_init = other_numeric_overall[0] if other_numeric_overall else default_x_init
        
        # Fallback si vip_data_df_for_calc était vide ou n'a pas aidé
        if default_x_init is None and len(numeric_trait_names_init) >= 1:
            default_x_init = numeric_trait_names_init[0]
        if default_y_init is None:
            if len(numeric_trait_names_init) >= 2:
                # S'assurer que default_y_init est différent de default_x_init si possible
                default_y_init = numeric_trait_names_init[1] if numeric_trait_names_init[0] != numeric_trait_names_init[1] else numeric_trait_names_init[0]
            elif default_x_init and len(numeric_trait_names_init) == 1: 
                 default_y_init = default_x_init # Si une seule var numérique, utiliser la même pour X et Y

        st.session_state.x_axis_trait_interactive = default_x_init
        st.session_state.y_axis_trait_interactive = default_y_init
        
        # Préparer st.session_state.vip_data_df_interactive basé sur les sélections par défaut
        # Il ne doit contenir que les traits numériques présents dans numeric_trait_names_init
        if not st.session_state.vip_data_df_for_calc.empty and numeric_trait_names_init:
            # Filtrer vip_data_df_for_calc pour ne garder que les variables qui sont aussi dans numeric_trait_names_init
            # (normalement, elles devraient toutes y être si vip_data_df_for_calc a été construit à partir des traits numériques de sub)
            interactive_df_init_base = st.session_state.vip_data_df_for_calc[
                st.session_state.vip_data_df_for_calc["Variable"].isin(numeric_trait_names_init)
            ].copy()

            if not interactive_df_init_base.empty :
                interactive_df_init_base["Axe X"] = (interactive_df_init_base["Variable"] == st.session_state.x_axis_trait_interactive)
                interactive_df_init_base["Axe Y"] = (interactive_df_init_base["Variable"] == st.session_state.y_axis_trait_interactive)
                st.session_state.vip_data_df_interactive = interactive_df_init_base[["Variable", "Communalité (%)", "Axe X", "Axe Y"]].reset_index(drop=True)
            else: # Si aucun trait numérique commun ou vip_data_df_for_calc était vide
                st.session_state.vip_data_df_interactive = pd.DataFrame(columns=["Variable", "Communalité (%)", "Axe X", "Axe Y"])
        else: # Si pas de communalités ou pas de traits numériques
            if numeric_trait_names_init: # S'il y a des traits numériques mais pas de communalités
                 temp_df = pd.DataFrame({"Variable": numeric_trait_names_init, "Communalité (%)": 0})
                 temp_df["Axe X"] = (temp_df["Variable"] == st.session_state.x_axis_trait_interactive)
                 temp_df["Axe Y"] = (temp_df["Variable"] == st.session_state.y_axis_trait_interactive)
                 st.session_state.vip_data_df_interactive = temp_df
            else:
                st.session_state.vip_data_df_interactive = pd.DataFrame(columns=["Variable", "Communalité (%)", "Axe X", "Axe Y"])


    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'analyse principale : {e}"); st.exception(e)
        st.session_state.run_main_analysis_once = False; st.stop()

# Recharger les données depuis session_state si l'analyse a déjà été lancée
if st.session_state.run_main_analysis_once:
    sub = st.session_state.get('sub', pd.DataFrame()) # Local 'sub'
    pdf = st.session_state.get('pdf', pd.DataFrame()) # Local 'pdf'
    X_for_dendro = st.session_state.get('X_for_dendro', np.array([])) # Local 'X_for_dendro'
    # numeric_trait_names_for_interactive_plot est déjà dans session_state

    if not pdf.empty:
        if "Cluster" not in pdf.columns: pdf["Cluster"] = "0" 
        fig_pca = px.scatter(pdf, x="PC1", y="PC2" if pdf.shape[1] > 1 and "PC2" in pdf.columns else None, 
                             color="Cluster", text="Espece_User", hover_name="Espece_User", 
                             custom_data=["Espece_User", "Ecologie"], template="plotly_dark", height=500, 
                             color_discrete_sequence=COLOR_SEQUENCE)
        fig_pca.update_traces(textposition="top center", marker=dict(opacity=0.7), 
                              hovertemplate=("<b>%{customdata[0]}</b><br><br><i>Écologie:</i><br>%{customdata[1]}<extra></extra>"),
                              textfont=dict(size=LABEL_FONT_SIZE_ON_PLOTS)) 
        unique_clusters_pca = sorted(pdf["Cluster"].unique())
        cluster_color_map_pca = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_pca)}
        
        if "PC2" in pdf.columns and pdf.shape[1] > 1 : 
            for cluster_label in unique_clusters_pca:
                cluster_points_df_pca = pdf[pdf["Cluster"] == cluster_label]
                if "PC1" in cluster_points_df_pca.columns and "PC2" in cluster_points_df_pca.columns: # Vérifier la présence des colonnes
                    # S'assurer que les colonnes PC1 et PC2 existent avant de les utiliser
                    if not cluster_points_df_pca[["PC1", "PC2"]].empty:
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
                            except Exception as e: print(f"Erreur calcul Hull ACP pour cluster {cluster_label}: {e}") # Afficher l'erreur pour débogage
        fig_pca.update_layout(title_text="Plot PCA", title_x=0.5, legend_title_text='Cluster')
        fig_pca.update_layout(dragmode='pan')

    # Calculer cluster_compositions_data seulement si pdf n'est pas vide
    if not pdf.empty and "Cluster" in pdf.columns and "Espece_User" in pdf.columns:
        cluster_compositions_data = [{"cluster_label": c, "count": len(pdf.loc[pdf["Cluster"] == c, "Espece_User"].unique()), "species_list": sorted(list(pdf.loc[pdf["Cluster"] == c, "Espece_User"].unique()))} for c in sorted(pdf["Cluster"].unique())]
    else:
        cluster_compositions_data = []

    
    if X_for_dendro.ndim == 2 and X_for_dendro.shape[0] > 1 and X_for_dendro.shape[1] > 0: # S'assurer que X_for_dendro est 2D et non vide
        Z = linkage(X_for_dendro, method="ward")
        dyn_thresh = 0
        if n_clusters_selected > 1 and (n_clusters_selected -1) < Z.shape[0] : # Correction: n-1 et non n_clusters_selected seulement
            if Z.shape[0] - (n_clusters_selected - 1) >= 0: # S'assurer que l'index est valide
                dyn_thresh = Z[Z.shape[0] - (n_clusters_selected - 1), 2] * 0.99
            elif Z.shape[0] > 0 : dyn_thresh = Z[0, 2] / 2 
        elif Z.shape[0] > 0: dyn_thresh = Z[0, 2] / 2
        
        dendro_labels = pdf["Espece_User"].tolist() if not pdf.empty and "Espece_User" in pdf.columns and len(pdf) == X_for_dendro.shape[0] else [f"Esp {i+1}" for i in range(X_for_dendro.shape[0])]

        fig_dend = ff.create_dendrogram(X_for_dendro, orientation="left", labels=dendro_labels, 
                                        linkagefun=lambda _: Z, color_threshold=dyn_thresh if n_clusters_selected > 1 else 0, 
                                        colorscale=COLOR_SEQUENCE)
        fig_dend.update_layout(template="plotly_dark", height=max(400, sub.shape[0] * 20 if not sub.empty else 400), title_text="Dendrogramme", title_x=0.5)
    else: 
        fig_dend = None
        if X_for_dendro.shape[0] <=1 : print("Pas assez de données pour le dendrogramme (X_for_dendro).")


# AFFICHAGE DU GRAPHIQUE ACP (COLONNE DE DROITE - HAUT)
with col_pca_plot:
    if fig_pca: 
        st.plotly_chart(fig_pca, use_container_width=True, config={'scrollZoom': True}) 
    elif run_main_analysis_button and ref.empty: st.warning("Veuillez d'abord charger des données de traits pour afficher le graphique ACP.")
    elif run_main_analysis_button and (st.session_state.get('sub', pd.DataFrame()).empty) : st.warning("Aucune espèce valide pour l'analyse ACP.")
    elif st.session_state.run_main_analysis_once and not fig_pca: st.info("Le graphique ACP sera affiché ici après une analyse principale réussie.")
    elif not st.session_state.run_main_analysis_once and not ref.empty: st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse principale'.")

if not st.session_state.run_main_analysis_once and ref.empty:
    with col_input_user: st.warning("Les données de référence n'ont pas pu être chargées. Vérifiez le fichier 'data_ref.csv'.")

# ---------------------------------------------------------------------------- #
# SECTION 2: EXPLORATION INTERACTIVE DES VARIABLES (MILIEU DE PAGE)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty:
    st.markdown("---")
    col_interactive_table, col_interactive_graph = st.columns([2, 3]) 

    with col_interactive_table:
        st.markdown("##### Tableau d'exploration interactif des variables")
        df_editor_source = st.session_state.get('vip_data_df_interactive', pd.DataFrame()).copy()

        if not df_editor_source.empty:
            edited_df = st.data_editor(
                df_editor_source, 
                column_config={
                    "Variable": st.column_config.TextColumn("Variable", disabled=True, help="Nom de la variable (trait)"),
                    "Communalité (%)": st.column_config.NumberColumn("Communalité (%)", format="%d%%", disabled=True, help="Communalité de la variable dans l'ACP"),
                    "Axe X": st.column_config.CheckboxColumn("Axe X", help="Sélectionner cette variable pour l'axe X du graphique d'exploration"),
                    "Axe Y": st.column_config.CheckboxColumn("Axe Y", help="Sélectionner cette variable pour l'axe Y du graphique d'exploration")
                },
                key="interactive_exploration_editor_v2", # Clé unique et mise à jour si la structure change
                use_container_width=True,
                hide_index=True,
                num_rows="fixed" 
            )

            old_ss_x = st.session_state.x_axis_trait_interactive
            old_ss_y = st.session_state.y_axis_trait_interactive

            # Déterminer l'intention de l'utilisateur à partir de l'éditeur
            # (la variable que l'utilisateur a effectivement tenté de sélectionner/désélectionner)
            def get_user_intent(edited_column_series, previous_selection_value):
                current_selections_in_editor = edited_column_series[edited_column_series == True].index.tolist()
                variable_names_selected = df_editor_source.loc[current_selections_in_editor, "Variable"].tolist()

                if not variable_names_selected: # Aucune sélection
                    return None
                if len(variable_names_selected) == 1: # Une seule sélection claire
                    return variable_names_selected[0]
                
                # Plusieurs sélections dans l'éditeur (état transitoire ou clic rapide)
                # Prioriser une variable qui n'était PAS la sélection précédente
                newly_toggled_vars = [var for var in variable_names_selected if var != previous_selection_value]
                if newly_toggled_vars:
                    return newly_toggled_vars[-1] # Prendre la dernière "nouvelle"
                
                # Si toutes les variables sélectionnées incluent la précédente (ou si la précédente était None)
                # et qu'il y en a plusieurs, cela signifie que l'utilisateur a cliqué sur une nouvelle.
                # On prend la dernière de la liste des sélections actuelles.
                return variable_names_selected[-1]

            editor_intent_x = get_user_intent(edited_df["Axe X"], old_ss_x)
            editor_intent_y = get_user_intent(edited_df["Axe Y"], old_ss_y)

            new_ss_x = editor_intent_x
            new_ss_y = editor_intent_y

            # Résolution de conflit : si X et Y sont la même variable
            if new_ss_x is not None and new_ss_x == new_ss_y:
                # Si X a été activement changé pour créer ce conflit, Y est désélectionné de cette variable.
                if new_ss_x != old_ss_x: 
                    new_ss_y = None
                # Si Y a été activement changé pour créer ce conflit, X est désélectionné.
                elif new_ss_y != old_ss_y:
                    new_ss_x = None
                # Si aucun n'a été "activement" changé pour créer le conflit (par ex. état initial ou erreur)
                # Désélectionner arbitrairement Y pour résoudre.
                else:
                    new_ss_y = None 
            
            # Vérifier si l'état de session effectif doit changer
            session_needs_update = (new_ss_x != old_ss_x) or (new_ss_y != old_ss_y)

            if session_needs_update:
                st.session_state.x_axis_trait_interactive = new_ss_x
                st.session_state.y_axis_trait_interactive = new_ss_y

                # Reconstruire le DataFrame source pour data_editor pour refléter l'état propre
                numeric_vars_list = st.session_state.get('numeric_trait_names_for_interactive_plot', [])
                base_df = st.session_state.get('vip_data_df_for_calc', pd.DataFrame())

                if not base_df.empty and numeric_vars_list:
                    rebuilt_df = base_df[base_df["Variable"].isin(numeric_vars_list)].copy()
                    rebuilt_df["Axe X"] = (rebuilt_df["Variable"] == st.session_state.x_axis_trait_interactive)
                    rebuilt_df["Axe Y"] = (rebuilt_df["Variable"] == st.session_state.y_axis_trait_interactive)
                    st.session_state.vip_data_df_interactive = rebuilt_df[["Variable", "Communalité (%)", "Axe X", "Axe Y"]].reset_index(drop=True)
                else: # Fallback
                    temp_df_fallback = pd.DataFrame({"Variable": numeric_vars_list, "Communalité (%)": 0.0}) # Communalité en float
                    temp_df_fallback["Axe X"] = (temp_df_fallback["Variable"] == st.session_state.x_axis_trait_interactive)
                    temp_df_fallback["Axe Y"] = (temp_df_fallback["Variable"] == st.session_state.y_axis_trait_interactive)
                    st.session_state.vip_data_df_interactive = temp_df_fallback

                st.rerun()
        else:
            st.info("Le tableau d'exploration sera disponible après l'analyse si des traits numériques sont identifiés.")

    with col_interactive_graph:
        st.markdown("##### Graphique d'exploration des variables")
        # Utiliser les valeurs de session_state qui sont maintenant "propres"
        x_axis_trait_selected_for_plot = st.session_state.x_axis_trait_interactive
        y_axis_trait_selected_for_plot = st.session_state.y_axis_trait_interactive
        
        current_numeric_traits = st.session_state.get('numeric_trait_names_for_interactive_plot', [])
        local_sub_df = st.session_state.get('sub', pd.DataFrame())
        local_pdf_df = st.session_state.get('pdf', pd.DataFrame())


        if not current_numeric_traits:
             st.warning("Aucun trait numérique trouvé pour l'exploration interactive.")
        elif not x_axis_trait_selected_for_plot or not y_axis_trait_selected_for_plot:
            st.info("Veuillez sélectionner une variable pour l'Axe X et une pour l'Axe Y dans le tableau à gauche.")
        elif x_axis_trait_selected_for_plot not in current_numeric_traits or \
             y_axis_trait_selected_for_plot not in current_numeric_traits:
            st.warning(f"Variable(s) sélectionnée(s) non valide(s) ou non numérique(s): X='{x_axis_trait_selected_for_plot}', Y='{y_axis_trait_selected_for_plot}'. Veuillez re-sélectionner.")
        elif local_sub_df.empty or local_pdf_df.empty or len(local_sub_df) != len(local_pdf_df):
             st.warning("Données pour le graphique interactif non prêtes ou incohérentes.")
        else:
            # Vérifier si les colonnes sélectionnées existent dans sub_df
            if x_axis_trait_selected_for_plot not in local_sub_df.columns or \
               y_axis_trait_selected_for_plot not in local_sub_df.columns:
                st.error(f"Erreur interne: Les variables sélectionnées ('{x_axis_trait_selected_for_plot}', '{y_axis_trait_selected_for_plot}') ne sont pas dans les données traitées. Veuillez relancer l'analyse.")
            else:
                plot_data_interactive = pd.DataFrame({
                    'Espece_User': local_pdf_df['Espece_User'].values,
                    'Ecologie': local_pdf_df['Ecologie'].values,
                    x_axis_trait_selected_for_plot: local_sub_df[x_axis_trait_selected_for_plot].values.copy(),
                    y_axis_trait_selected_for_plot: local_sub_df[y_axis_trait_selected_for_plot].values.copy(),
                    'Cluster': local_pdf_df['Cluster'].values
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
                        "<b>%{customdata[0]}</b><br>" +  
                        "<br><i>Écologie:</i><br>%{customdata[1]}" + 
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
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty: 
    st.markdown("---")
    st.subheader("Composition des Clusters (ACP)")
    # Utiliser la variable locale cluster_compositions_data qui a été calculée après l'analyse
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
    elif st.session_state.run_main_analysis_once: # Si l'analyse a eu lieu mais pas de données de cluster
        st.info("La composition des clusters (ACP) sera affichée ici si des clusters sont formés.")


# ---------------------------------------------------------------------------- #
# SECTION 4: AFFICHAGE DU DENDROGRAMME (PLEINE LARGEUR, EN DERNIER)
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not st.session_state.get('sub', pd.DataFrame()).empty : 
    st.markdown("---") 
    # Utiliser la variable locale fig_dend
    if fig_dend: 
        st.plotly_chart(fig_dend, use_container_width=True)
    elif species_binom_user_unique and st.session_state.get('X_for_dendro', np.array([])).shape[0] > 1 : # Si des espèces ont été entrées et X_for_dendro est prêt mais pas de fig_dend
        st.info("Le dendrogramme n'a pas pu être généré (problème de linkage ou données X_for_dendro).")
    elif st.session_state.get('X_for_dendro', np.array([])).shape[0] <= 1 :
         st.info("Pas assez de données (espèces uniques > 1) pour générer un dendrogramme.")

