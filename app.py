"""
Web-app Streamlit : PCA + clustering botaniques
Reconnaît les noms saisis au format « Genre épithète » même si la base
contient les auteurs (« Acacia mearnsii » ↔ « Acacia mearnsii De Wild. »)

Modifications v6 (basées sur v5):
- Ajout de l'interactivité sur le graphique PCA :
    - Un clic sur un point (espèce) du PCA permet de sélectionner cette espèce.
    - Un bouton apparaît pour confirmer la suppression de l'espèce sélectionnée.
    - Si la suppression est confirmée, l'espèce est retirée de l'analyse,
      le champ de saisie est mis à jour, et toutes les analyses sont relancées.
- Utilisation de st.session_state pour gérer la liste des espèces et les résultats.
- Nécessite la bibliothèque 'streamlit-plotly-events'.
- Maintien des améliorations précédentes (déduplication, centrage, colonnes, etc.).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
from scipy.spatial import ConvexHull
import numpy as np
# Assurez-vous que streamlit-plotly-events est installé : pip install streamlit-plotly-events
# Si ModuleNotFoundError, exécutez la commande ci-dessus dans votre terminal.
try:
    from streamlit_plotly_events import plotly_events
except ModuleNotFoundError:
    st.error("La bibliothèque 'streamlit-plotly-events' n'est pas installée. "
             "Veuillez l'installer en exécutant : pip install streamlit-plotly-events")
    st.stop()


# Assurez-vous que le fichier core.py est dans le même répertoire ou accessible
# Pour les besoins de cet exemple, nous allons simuler les fonctions de core.py
# Si vous avez votre fichier core.py, décommentez la ligne suivante et commentez/supprimez les fonctions simulées.
# import core

# --- SIMULATION DE core.py ---
def simulate_read_reference(file_path="data_ref.csv"):
    """Simule la lecture d'un fichier de référence."""
    # Crée un DataFrame de démonstration si le fichier n'existe pas
    # Adaptez ceci pour correspondre à la structure attendue de vos données réelles
    data = {
        'Espece': [
            'Quercus robur L.', 'Fagus sylvatica L.', 'Pinus sylvestris L.',
            'Betula pendula Roth', 'Acer pseudoplatanus L.', 'Fraxinus excelsior L.',
            'Sorbus aucuparia L.', 'Populus tremula L.', 'Salix caprea L.',
            'Ulmus glabra Huds.', 'Tilia cordata Mill.', 'Alnus glutinosa (L.) Gaertn.',
            'Corylus avellana L.', 'Prunus avium (L.) L.', 'Crataegus monogyna Jacq.',
            'Teucrium chamaedrys L.', 'Potentilla verna L.', 'Astragalus monspessulanus L.'
        ],
        # Ajoutez autant de colonnes de traits que nécessaire pour votre analyse PCA
        'Trait1': np.random.rand(18) * 10,
        'Trait2': np.random.rand(18) * 5,
        'Trait3': np.random.rand(18) * 20,
        'Trait4': np.random.rand(18) * 15,
        'Trait5': np.random.rand(18) * 8,
    }
    df = pd.DataFrame(data)
    # Simuler une sauvegarde et un rechargement pour correspondre au comportement de core.read_reference
    # df.to_csv(file_path, index=False) # Optionnel: sauvegarder pour tester la lecture réelle
    # return pd.read_csv(file_path)
    return df

def simulate_analyse(sub_df, n_clusters_selected):
    """Simule la fonction d'analyse PCA et de clustering."""
    if sub_df.empty or sub_df.shape[0] < 2:
        # Pas assez de données pour PCA
        return np.array([]), None, np.array([[]]), pd.DataFrame()

    # Sélectionner uniquement les colonnes numériques pour PCA (en excluant 'Espece')
    X_data = sub_df.select_dtypes(include=np.number)

    if X_data.shape[1] < 2: # Pas assez de traits pour PCA
         return np.array([]), None, np.array([[]]), X_data # ou sub_df selon ce que core.analyse retourne

    # Simulation PCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering

    # S'assurer qu'il y a au moins 2 composantes si possible
    n_components = min(2, X_data.shape[0], X_data.shape[1])
    
    if n_components < 1: # Si après min, on a 0 ou 1, on ne peut pas faire de PCA utile pour 2D plot
        return np.array([]), None, np.array([[]]), X_data

    pca_instance = PCA(n_components=n_components)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    coords_pca = pca_instance.fit_transform(X_scaled)

    # S'assurer que coords_pca a toujours 2 colonnes si n_components était 1
    if coords_pca.shape[1] == 1:
        coords_pca = np.hstack([coords_pca, np.zeros_like(coords_pca)])


    # Simulation Clustering
    # S'assurer que n_clusters n'est pas plus grand que le nombre d'échantillons
    actual_n_clusters = min(n_clusters_selected, X_data.shape[0])
    if actual_n_clusters < 1: # Ne devrait pas arriver si sub_df.shape[0] >= 2
        labels_cluster = np.zeros(X_data.shape[0], dtype=int)
    else:
        clustering = AgglomerativeClustering(n_clusters=actual_n_clusters)
        labels_cluster = clustering.fit_predict(X_scaled)

    return labels_cluster, pca_instance, coords_pca, X_scaled # X_scaled est X dans votre code original

# Remplacer les appels à core par les simulations
core_read_reference = simulate_read_reference
core_analyse = simulate_analyse
# --- FIN DE LA SIMULATION DE core.py ---


# ---------------------------------------------------------------------------- #
# CONFIGURATION UI
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="PCA flore interactive", layout="wide")
st.markdown("<h1 style='text-align: center;'>Analyse interactive des clusters botaniques</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------- #
# INITIALISATION DE L'ETAT DE SESSION
# ---------------------------------------------------------------------------- #
if 'species_text_area_content' not in st.session_state:
    st.session_state.species_text_area_content = "Teucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus\nQuercus robur" # Contenu initial pour démo
if 'current_analyzed_species_user_names' not in st.session_state:
    st.session_state.current_analyzed_species_user_names = []
if 'species_selected_for_removal' not in st.session_state:
    st.session_state.species_selected_for_removal = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {
        "fig_pca": None, "fig_dend": None, "vip_styled": None,
        "cluster_compositions_data": [], "not_found_user_raw_names": [],
        "error_message": None, "sub_data_shape_0": 0, "pdf_data": pd.DataFrame()
    }
if 'run_counter' not in st.session_state:
    st.session_state.run_counter = 0
if 'initial_run_done' not in st.session_state: # Pour gérer l'affichage initial
    st.session_state.initial_run_done = False


# ---------------------------------------------------------------------------- #
# CHARGEMENT DE LA BASE
# ---------------------------------------------------------------------------- #
@st.cache_data
def load_data(file_path="data_ref.csv"):
    """Charge les données de référence à partir du chemin spécifié."""
    try:
        # Utiliser la fonction simulée ou votre propre core.read_reference
        data = core_read_reference(file_path)
        if data.empty:
            st.warning(f"Le fichier de données '{file_path}' est vide ou n'a pas pu être chargé correctement.")
        return data
    except FileNotFoundError:
        st.error(f"ERREUR CRITIQUE: Fichier de données '{file_path}' non trouvé. L'application ne peut pas fonctionner.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"ERREUR CRITIQUE: Impossible de charger les données depuis '{file_path}': {e}")
        return pd.DataFrame()

ref_data = load_data()

ref_binom_series = pd.Series(dtype='str')
if not ref_data.empty:
    ref_binom_series = (
        ref_data["Espece"]
        .str.split()
        .str[:2]
        .str.join(" ")
        .str.lower()
    )
else:
    st.error("La base de données de référence est vide. L'application ne peut pas fonctionner correctement.")


# ---------------------------------------------------------------------------- #
# FONCTION D'ANALYSE PRINCIPALE
# ---------------------------------------------------------------------------- #
def perform_analysis(species_user_names_list, ref_df, ref_binom_series_data, n_clusters):
    """
    Exécute l'ensemble du pipeline d'analyse PCA et clustering.
    Retourne un dictionnaire contenant tous les résultats (figures, tables, messages).
    """
    results = {
        "fig_pca": None, "fig_dend": None, "vip_styled": None,
        "cluster_compositions_data": [], "not_found_user_raw_names": [],
        "error_message": None, "sub_data_shape_0": 0, "pdf_data": pd.DataFrame()
    }

    if ref_df.empty:
        results["error_message"] = "Les données de référence sont vides. Impossible de procéder à l'analyse."
        return results

    if not species_user_names_list:
        results["error_message"] = "Veuillez saisir au moins un nom d'espèce."
        return results

    species_binom_user_unique = [" ".join(s.split()[:2]).lower() for s in species_user_names_list]

    indices_to_keep_from_ref = []
    if not ref_binom_series_data.empty:
        ref_indexed_binom = ref_binom_series_data.reset_index()
        ref_indexed_binom.columns = ['Original_Ref_Index', 'ref_binom_val']
        
        for user_binom_specie in species_binom_user_unique:
            matches_in_ref = ref_indexed_binom[ref_indexed_binom['ref_binom_val'] == user_binom_specie]
            if not matches_in_ref.empty:
                indices_to_keep_from_ref.append(matches_in_ref['Original_Ref_Index'].iloc[0])
    
    indices_to_keep_from_ref = sorted(list(set(indices_to_keep_from_ref)))

    if indices_to_keep_from_ref:
        sub = ref_df.loc[indices_to_keep_from_ref].copy()
    else:
        sub = pd.DataFrame(columns=ref_df.columns)

    results["sub_data_shape_0"] = sub.shape[0]

    found_ref_binom_values_in_sub = []
    if not sub.empty:
        found_ref_binom_values_in_sub = (
            sub["Espece"]
            .str.split().str[:2].str.join(" ").str.lower()
            .tolist()
        )

    not_found_user_raw_names_list = []
    for i, user_binom_name in enumerate(species_binom_user_unique):
        if user_binom_name not in found_ref_binom_values_in_sub:
            not_found_user_raw_names_list.append(species_user_names_list[i])
    results["not_found_user_raw_names"] = not_found_user_raw_names_list
    
    if sub.empty:
        results["error_message"] = "Aucune des espèces valides pour l'analyse n'a été trouvée dans la base de référence."
        return results
    
    if sub.shape[0] < n_clusters and sub.shape[0] > 0 : # Permettre l'analyse si n_clusters > nb_especes, mais avertir
        st.warning(f"Le nombre d'espèces ({sub.shape[0]}) est inférieur au nombre de clusters ({n_clusters}). Le nombre de clusters sera ajusté à {sub.shape[0]}.")
        n_clusters = sub.shape[0] # Ajuster n_clusters
    elif sub.shape[0] == 0: # Redondant avec sub.empty mais pour être sûr
        results["error_message"] = "Aucune espèce à analyser."
        return results

    if sub.shape[0] < 2:
        results["error_message"] = f"Au moins 2 espèces sont nécessaires pour l'analyse PCA. {sub.shape[0]} espèce(s) trouvée(s) et utilisée(s)."
        return results

    min_points_for_hull = 3
    
    user_input_binom_to_raw_map = {
        " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_user_names_list
    }

    try:
        # Utiliser la fonction simulée ou votre propre core.analyse
        labels, pca, coords, X = core_analyse(sub, n_clusters)

        if X.shape[0] == 0 : # Si core_analyse retourne des données vides
             results["error_message"] = "L'analyse PCA n'a pas pu être effectuée (pas assez de données ou de traits)."
             return results

        pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
        pdf["Cluster"] = labels.astype(str)
        pdf["Espece_Ref"] = sub["Espece"].values

        def get_user_input_name(full_ref_name):
            binom_ref_name = " ".join(full_ref_name.split()[:2]).lower()
            return user_input_binom_to_raw_map.get(binom_ref_name, full_ref_name)

        pdf["Espece_User"] = pdf["Espece_Ref"].apply(get_user_input_name)
        results["pdf_data"] = pdf.copy() # Stocker pdf pour le débogage ou l'utilisation future

        color_sequence = px.colors.qualitative.Plotly  
        
        y_axis_param = "PC2" if coords.shape[1] > 1 else None
        if y_axis_param is None and "PC1" in pdf.columns: # Si 1D PCA, on peut simuler un axe Y pour le scatter
            pdf["PC2_dummy"] = 0 
            y_axis_param = "PC2_dummy"


        results["fig_pca"] = px.scatter(
            pdf, x="PC1", y=y_axis_param,
            color="Cluster", text="Espece_User",
            template="plotly_dark", height=600, color_discrete_sequence=color_sequence
        )
        results["fig_pca"].update_traces(
            textposition="top center", marker=dict(size=10, opacity=0.8), # Augmenter la taille des points
            hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>" + (f"PC2: %{{y:.2f}}<extra></extra>" if y_axis_param != "PC2_dummy" else "<extra></extra>")
        )
        
        unique_clusters = sorted(pdf["Cluster"].unique())
        cluster_color_map = {cluster_label: color_sequence[i % len(color_sequence)] for i, cluster_label in enumerate(unique_clusters)}

        if coords.shape[1] > 1: # Enveloppes convexes seulement si 2D+
            for i, cluster_label in enumerate(unique_clusters):
                cluster_points_df = pdf[pdf["Cluster"] == cluster_label]
                unique_cluster_points = cluster_points_df[["PC1", "PC2"]].drop_duplicates().values
                
                if len(unique_cluster_points) >= min_points_for_hull:
                    try:
                        hull = ConvexHull(unique_cluster_points)
                        hull_points = unique_cluster_points[hull.vertices]
                        path_x = np.append(hull_points[:, 0], hull_points[0, 0])
                        path_y = np.append(hull_points[:, 1], hull_points[0, 1])
                        results["fig_pca"].add_trace(go.Scatter(
                            x=path_x, y=path_y, fill="toself",
                            fillcolor=cluster_color_map[cluster_label],
                            line=dict(color=cluster_color_map[cluster_label], width=1.5),
                            mode='lines', name=f'Cluster {cluster_label} Hull',
                            opacity=0.2, showlegend=False, hoverinfo='skip'
                        ))
                    except Exception as e_hull:
                        print(f"Avertissement: Impossible de générer l'enveloppe convexe pour le cluster {cluster_label}: {e_hull}")
                elif len(unique_cluster_points) > 0 :
                     print(f"Info: Le cluster {cluster_label} a {len(unique_cluster_points)} points uniques (minimum {min_points_for_hull} requis pour l'enveloppe), enveloppe ignorée.")


        results["fig_pca"].update_layout(
            title_text="Clusters d'espèces (PCA)", title_x=0.5, legend_title_text='Cluster'
        )
        if y_axis_param == "PC2_dummy": # Cacher l'axe Y si c'est un dummy
            results["fig_pca"].update_yaxes(visible=False, showticklabels=False)


        if X.shape[0] > 1: # X est X_scaled ici
            Z = linkage(X, method="ward")
            dynamic_color_threshold = 0
            # S'assurer que n_clusters (ajusté) est utilisé pour le seuil
            if n_clusters > 1 and (n_clusters -1) <= Z.shape[0] :
                idx_threshold = -(n_clusters - 1)
                if idx_threshold == 0: 
                    dynamic_color_threshold = Z[0, 2] / 2 
                elif Z.shape[0] >= (n_clusters -1) and (n_clusters -1) > 0:
                    # S'assurer que l'index n'est pas hors limites
                    actual_idx_for_threshold = max(0, Z.shape[0] - (n_clusters -1) )
                    dynamic_color_threshold = Z[actual_idx_for_threshold, 2] * 0.99


            results["fig_dend"] = ff.create_dendrogram(
                X, orientation="left", labels=pdf["Espece_User"].tolist(),
                linkagefun=lambda _: Z,
                color_threshold=dynamic_color_threshold if n_clusters > 1 else 0,
                colorscale=color_sequence
            )
            results["fig_dend"].update_layout(
                template="plotly_dark", height=max(650, sub.shape[0] * 25), # Augmenter l'espace par espèce
                title_text="Dendrogramme", title_x=0.5
            )
        
        if pca is not None and hasattr(pca, 'components_') and hasattr(pca, 'explained_variance_'):
            loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
            communal = (loadings**2).sum(axis=1)
            # S'assurer que les colonnes de traits existent dans sub avant de les utiliser
            trait_columns = sub.select_dtypes(include=np.number).columns
            if len(trait_columns) == len(communal):
                vip_data_df = pd.DataFrame({
                    "Variable": trait_columns, 
                    "Communalité (%)": (communal * 100).round(0).astype(int),
                }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
                results["vip_styled"] = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)'])\
                                                   .format({"Communalité (%)": "{:}%"})
            else:
                print("Avertissement: Discordance entre le nombre de traits et les communalités. VIP non calculé.")
        else:
            print("Info: PCA non effectué ou attributs manquants, VIP non calculé.")


        cluster_comp_data = []
        for c_label in sorted(pdf["Cluster"].unique()):
            esp_user_names = sorted(list(pdf.loc[pdf["Cluster"] == c_label, "Espece_User"].unique()))
            cluster_comp_data.append({
                "cluster_label": c_label, "count": len(esp_user_names), "species_list": esp_user_names
            })
        results["cluster_compositions_data"] = cluster_comp_data

    except Exception as e:
        results["error_message"] = f"Une erreur est survenue lors de l'analyse : {e}"
        st.exception(e)
    
    return results

# ---------------------------------------------------------------------------- #
# CALLBACKS ET LOGIQUE D'INTERACTION
# ---------------------------------------------------------------------------- #
def trigger_analysis_from_button():
    """Prépare les espèces depuis le text_area et lance l'analyse complète."""
    st.session_state.initial_run_done = True # Marquer que le bouton a été cliqué au moins une fois
    species_raw_unique_from_text = sorted(list(set(s.strip() for s in st.session_state.species_text_area_content.splitlines() if s.strip())))
    st.session_state.current_analyzed_species_user_names = species_raw_unique_from_text
    st.session_state.species_selected_for_removal = None
    
    if not ref_data.empty:
        n_clusters = st.session_state.get("n_clusters_slider_val", 3)
        st.session_state.analysis_results = perform_analysis(
            st.session_state.current_analyzed_species_user_names,
            ref_data,
            ref_binom_series,
            n_clusters
        )
        st.session_state.run_counter += 1
    else:
        st.session_state.analysis_results["error_message"] = "Les données de référence ne sont pas chargées."
        st.session_state.analysis_results["fig_pca"] = None # S'assurer de nettoyer les figures
        st.session_state.analysis_results["fig_dend"] = None


def handle_species_removal(species_name_to_remove):
    """Supprime une espèce de la liste et relance l'analyse."""
    if species_name_to_remove in st.session_state.current_analyzed_species_user_names:
        st.session_state.current_analyzed_species_user_names.remove(species_name_to_remove)
    
    st.session_state.species_text_area_content = "\n".join(st.session_state.current_analyzed_species_user_names)
    st.session_state.species_selected_for_removal = None
    
    if not ref_data.empty:
        n_clusters = st.session_state.get("n_clusters_slider_val", 3)
        st.session_state.analysis_results = perform_analysis(
            st.session_state.current_analyzed_species_user_names,
            ref_data,
            ref_binom_series,
            n_clusters
        )
        st.session_state.run_counter += 1
    else:
        st.session_state.analysis_results["error_message"] = "Les données de référence ne sont pas chargées."
    st.rerun()


# ---------------------------------------------------------------------------- #
# LAYOUT DE LA PAGE
# ---------------------------------------------------------------------------- #
col_input, col_pca_plot = st.columns([1, 3]) # Ratio ajusté pour plus d'espace au PCA

with col_input:
    st.subheader("CORTEGE FLORISTIQUE")
    n_clusters_selected_val = st.slider(
        "Nombre de clusters", min_value=2, max_value=8, 
        value=st.session_state.get("n_clusters_slider_val", 3),
        key="n_clusters_slider_val",
        disabled=ref_data.empty
    )
    
    st.text_area(
        "Liste d'espèces (une par ligne)", height=250,
        placeholder="Teucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus\n…",
        key="species_text_area_content",
        disabled=ref_data.empty
    )

    st.button(
        "Lancer l'analyse", 
        type="primary", 
        disabled=ref_data.empty,
        on_click=trigger_analysis_from_button,
        use_container_width=True
    )

    # Afficher les avertissements pour les espèces non trouvées
    # seulement si une analyse a été tentée
    if st.session_state.initial_run_done and st.session_state.analysis_results.get("not_found_user_raw_names"):
        st.warning(
            "Non trouvées dans la base : " + ", ".join(st.session_state.analysis_results["not_found_user_raw_names"]),
            icon="⚠️"
        )

    if st.session_state.initial_run_done and st.session_state.analysis_results.get("error_message"):
        st.error(st.session_state.analysis_results["error_message"])

    if st.session_state.species_selected_for_removal:
        species_to_remove = st.session_state.species_selected_for_removal
        st.markdown("---")
        st.markdown(f"Voulez-vous supprimer **{species_to_remove}** de l'analyse ?")
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            st.button(f"Oui, supprimer", 
                      on_click=handle_species_removal, 
                      args=(species_to_remove,),
                      key=f"confirm_remove_{species_to_remove.replace(' ', '_').replace('.', '_')}", # Clé plus robuste
                      type="primary", use_container_width=True)
        with col_cancel:
            if st.button("Annuler", key=f"cancel_remove_{species_to_remove.replace(' ', '_').replace('.', '_')}", use_container_width=True):
                st.session_state.species_selected_for_removal = None
                st.rerun()


with col_pca_plot:
    fig_pca_result = st.session_state.analysis_results.get("fig_pca")
    if fig_pca_result:
        selected_points = plotly_events(
            fig_pca_result, 
            click_event=True, 
            key=f"pca_plot_events_{st.session_state.run_counter}",
            override_height=fig_pca_result.layout.height if fig_pca_result.layout else 600, # S'assurer que layout existe
            override_width="100%"
        )
        if selected_points:
            # plotly_events retourne une liste de dictionnaires, un par point cliqué (généralement un seul pour click_event)
            # Le 'text' correspond à ce que nous avons défini dans px.scatter (Espece_User)
            clicked_species_name = selected_points[0].get('text') 
            
            if clicked_species_name:
                # Vérifier si l'espèce cliquée est bien dans la liste actuellement analysée
                # Cela évite des problèmes si le graphe est "périmé" par rapport à la liste des espèces
                # (par exemple, si l'utilisateur modifie le text_area sans relancer l'analyse)
                current_species_in_plot = st.session_state.analysis_results.get("pdf_data", pd.DataFrame())
                if not current_species_in_plot.empty and clicked_species_name in current_species_in_plot["Espece_User"].tolist():
                    st.session_state.species_selected_for_removal = clicked_species_name
                    st.rerun()
                elif clicked_species_name in st.session_state.current_analyzed_species_user_names:
                    # Fallback si pdf_data n'est pas à jour mais le nom est dans la liste principale
                    st.session_state.species_selected_for_removal = clicked_species_name
                    st.rerun()
                else:
                    st.warning(f"L'espèce '{clicked_species_name}' n'est plus dans la liste d'analyse active ou le graphique est obsolète. Veuillez relancer l'analyse si besoin.")
                    st.session_state.species_selected_for_removal = None # Nettoyer au cas où

    elif st.session_state.initial_run_done and not st.session_state.analysis_results.get("error_message"):
        if ref_data.empty:
             st.warning("Veuillez d'abord charger des données pour afficher le graphique PCA.")
        else:
            st.info("Le graphique PCA sera affiché ici après une analyse réussie.")
    elif not st.session_state.initial_run_done and not ref_data.empty :
         st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse'.")


# Affichage des résultats en dehors des colonnes principales pour occuper toute la largeur si besoin
st.markdown("---") # Séparateur visuel

results_cols = st.columns([1, 2]) # Réutiliser une structure de colonnes pour les résultats tabulaires

with results_cols[0]:
    st.subheader("Importance des Variables")
    vip_styled_result = st.session_state.analysis_results.get("vip_styled")
    if vip_styled_result is not None:
        st.write(vip_styled_result.to_html(escape=False), unsafe_allow_html=True)
    elif st.session_state.initial_run_done and not st.session_state.analysis_results.get("error_message") and st.session_state.analysis_results.get("sub_data_shape_0", 0) > 0:
        st.info("Le tableau d'importance des variables sera affiché ici.")

with results_cols[1]:
    st.subheader("Composition des Clusters")
    cluster_compositions_data_result = st.session_state.analysis_results.get("cluster_compositions_data")
    if cluster_compositions_data_result:
        num_clusters_found = len(cluster_compositions_data_result)
        if num_clusters_found > 0:
            # Afficher les clusters en ligne s'il y en a peu, sinon en colonne
            if num_clusters_found <= 3:
                 cluster_display_cols = st.columns(num_clusters_found)
            else: # Si plus de 3 clusters, les afficher verticalement pour éviter trop de colonnes étroites
                 cluster_display_cols = [st] * num_clusters_found # Simule une seule colonne pour l'itération

            for i, comp_data in enumerate(cluster_compositions_data_result):
                target_col = cluster_display_cols[i] if num_clusters_found <=3 else st # Utilise la i-ème colonne ou st directement
                with target_col:
                    st.markdown(f"**Cluster {comp_data['cluster_label']}** ({comp_data['count']} espèces)")
                    # Utiliser st.expander pour les longues listes d'espèces
                    if len(comp_data['species_list']) > 10:
                        with st.expander("Voir les espèces", expanded=False):
                            for species_name in comp_data['species_list']:
                                st.markdown(f"- {species_name}")
                    else:
                        for species_name in comp_data['species_list']:
                            st.markdown(f"- {species_name}")
                    if num_clusters_found > 3: st.markdown("---") # Séparateur si affichage vertical

        elif st.session_state.initial_run_done: # Si l'analyse a tourné mais pas de clusters
            st.info("Aucun cluster à afficher (vérifiez le nombre d'espèces et de clusters demandés).")
    elif st.session_state.initial_run_done and not st.session_state.analysis_results.get("error_message") and st.session_state.analysis_results.get("sub_data_shape_0", 0) > 0:
        st.info("La composition des clusters sera affichée ici.")


fig_dend_result = st.session_state.analysis_results.get("fig_dend")
if fig_dend_result:
    st.subheader("Dendrogramme Hiérarchique")
    st.plotly_chart(fig_dend_result, use_container_width=True)
elif st.session_state.initial_run_done and not st.session_state.analysis_results.get("error_message") and st.session_state.analysis_results.get("sub_data_shape_0", 0) > 0 :
    if st.session_state.analysis_results.get("sub_data_shape_0", 0) < 2 :
        st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces).")
    else:
        st.info("Le dendrogramme sera affiché ici si l'analyse le génère.")


st.markdown("---")
st.caption(f"Robin Wojcik, Améten. Date actuelle : {pd.Timestamp('now', tz='Europe/Paris').strftime('%Y-%m-%d %H:%M:%S %Z')}")
st.caption("Note : L'interactivité du graphique PCA (clic pour suppression) utilise la bibliothèque 'streamlit-plotly-events'.")
