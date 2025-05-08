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
from streamlit_plotly_events import plotly_events

# Assurez-vous que le fichier core.py est dans le même répertoire ou accessible
import core

# ---------------------------------------------------------------------------- #
# CONFIGURATION UI
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="PCA flore interactive", layout="wide")
st.markdown("<h1 style='text-align: center;'>Analyse interactive des clusters botaniques</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------- #
# INITIALISATION DE L'ETAT DE SESSION
# ---------------------------------------------------------------------------- #
if 'species_text_area_content' not in st.session_state:
    st.session_state.species_text_area_content = "Teucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus" # Contenu initial pour démo
if 'current_analyzed_species_user_names' not in st.session_state:
    # Liste des noms bruts uniques de l'utilisateur, qui est la source de vérité pour l'analyse
    st.session_state.current_analyzed_species_user_names = []
if 'species_selected_for_removal' not in st.session_state:
    st.session_state.species_selected_for_removal = None # Stocke le nom utilisateur de l'espèce cliquée
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {
        "fig_pca": None, "fig_dend": None, "vip_styled": None,
        "cluster_compositions_data": [], "not_found_user_raw_names": [],
        "error_message": None, "sub_data_shape_0": 0
    }
if 'run_counter' not in st.session_state: # Pour forcer la réinitialisation de plotly_events
    st.session_state.run_counter = 0

# ---------------------------------------------------------------------------- #
# CHARGEMENT DE LA BASE
# ---------------------------------------------------------------------------- #
@st.cache_data
def load_data(file_path="data_ref.csv"):
    """Charge les données de référence à partir du chemin spécifié."""
    try:
        data = core.read_reference(file_path)
        return data
    except FileNotFoundError:
        st.error(f"ERREUR CRITIQUE: Fichier de données '{file_path}' non trouvé. L'application ne peut pas fonctionner.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"ERREUR CRITIQUE: Impossible de charger les données depuis '{file_path}': {e}")
        return pd.DataFrame()

ref_data = load_data() # Renommé pour clarté

ref_binom_series = pd.Series(dtype='str')
if not ref_data.empty:
    ref_binom_series = (
        ref_data["Espece"]
        .str.split()
        .str[:2]
        .str.join(" ")
        .str.lower()
    )

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
        "error_message": None, "sub_data_shape_0": 0
    }

    if not species_user_names_list:
        results["error_message"] = "Veuillez saisir au moins un nom d'espèce."
        return results

    species_binom_user_unique = [" ".join(s.split()[:2]).lower() for s in species_user_names_list]

    indices_to_keep_from_ref = []
    if not ref_binom_series_data.empty:
        ref_indexed_binom = ref_binom_series_data.reset_index()
        ref_indexed_binom.columns = ['Original_Ref_Index', 'ref_binom_val']
        
        # Assurer l'unicité des indices de ref pour éviter les duplications dans 'sub'
        # Si plusieurs noms utilisateurs pointent vers le même nom binomial normalisé,
        # nous ne voulons qu'une seule entrée de 'ref' pour ce nom binomial.
        # Cependant, la logique actuelle basée sur species_user_names_list qui est déjà unique
        # et le fait de prendre le .iloc[0] par user_binom_specie garantit que 'sub'
        # aura des entrées uniques de 'ref' par 'user_binom_specie' unique.
        
        # Pour conserver une trace des indices 'ref' déjà utilisés par un 'user_binom_specie'
        # afin de garantir que chaque 'user_binom_specie' ne sélectionne qu'une fois.
        # Note : species_binom_user_unique est déjà unique, donc pas besoin ici.

        for user_binom_specie in species_binom_user_unique:
            matches_in_ref = ref_indexed_binom[ref_indexed_binom['ref_binom_val'] == user_binom_specie]
            if not matches_in_ref.empty:
                indices_to_keep_from_ref.append(matches_in_ref['Original_Ref_Index'].iloc[0])
    
    # S'assurer que les indices sont uniques (au cas où plusieurs user_binom_specie correspondraient au même index de ref, peu probable)
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
    # species_user_names_list et species_binom_user_unique sont alignés
    for i, user_binom_name in enumerate(species_binom_user_unique):
        if user_binom_name not in found_ref_binom_values_in_sub:
            not_found_user_raw_names_list.append(species_user_names_list[i])
    results["not_found_user_raw_names"] = not_found_user_raw_names_list
    
    if sub.empty:
        results["error_message"] = "Aucune des espèces valides pour l'analyse n'a été trouvée dans la base de référence."
        return results
    
    if sub.shape[0] < n_clusters:
        results["error_message"] = f"Le nombre d'espèces trouvées et utilisées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters})."
        return results
        
    if sub.shape[0] < 2:
        results["error_message"] = f"Au moins 2 espèces sont nécessaires pour l'analyse PCA. {sub.shape[0]} espèce(s) trouvée(s) et utilisée(s)."
        return results

    min_points_for_hull = 3
    
    user_input_binom_to_raw_map = {
        " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_user_names_list
    }

    try:
        labels, pca, coords, X = core.analyse(sub, n_clusters)

        pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
        pdf["Cluster"] = labels.astype(str)
        pdf["Espece_Ref"] = sub["Espece"].values # Noms de référence (avec auteurs potentiellement)

        def get_user_input_name(full_ref_name):
            binom_ref_name = " ".join(full_ref_name.split()[:2]).lower()
            # Retrouver le nom utilisateur original qui correspond à ce nom binomial de référence
            # Cela suppose que user_input_binom_to_raw_map contient une entrée pour chaque binom_ref_name trouvé dans sub
            return user_input_binom_to_raw_map.get(binom_ref_name, full_ref_name)

        # Assigner le nom utilisateur correspondant. 'sub' est indexé comme X et coords.
        # Les 'Espece_Ref' dans 'sub' doivent correspondre à ceux utilisés pour générer 'user_input_binom_to_raw_map'
        # via 'species_user_names_list'.
        pdf["Espece_User"] = pdf["Espece_Ref"].apply(get_user_input_name)
        
        color_sequence = px.colors.qualitative.Plotly  
        
        results["fig_pca"] = px.scatter(
            pdf, x="PC1", y="PC2" if coords.shape[1] > 1 else None,
            color="Cluster", text="Espece_User", # Important pour l'identification au clic
            template="plotly_dark", height=600, color_discrete_sequence=color_sequence
        )
        results["fig_pca"].update_traces(
            textposition="top center", marker=dict(opacity=0.7),
            hovertemplate="<b>%{text}</b><br>PC1: %{x}<br>PC2: %{y}<extra></extra>" # Afficher Espece_User dans le hover
        )
        
        unique_clusters = sorted(pdf["Cluster"].unique())
        cluster_color_map = {cluster_label: color_sequence[i % len(color_sequence)] for i, cluster_label in enumerate(unique_clusters)}

        if coords.shape[1] > 1:
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
                        print(f"Could not generate convex hull for cluster {cluster_label}: {e_hull}")
                elif len(unique_cluster_points) > 0 :
                     print(f"Cluster {cluster_label} has less than {min_points_for_hull} points ({len(unique_cluster_points)} unique), skipping hull.")


        results["fig_pca"].update_layout(
            title_text="Clusters d'espèces (PCA)", title_x=0.5, legend_title_text='Cluster'
        )

        if X.shape[0] > 1:
            Z = linkage(X, method="ward")
            dynamic_color_threshold = 0
            if n_clusters > 1 and (n_clusters -1) <= Z.shape[0] :
                idx_threshold = -(n_clusters - 1)
                if idx_threshold == 0: 
                    dynamic_color_threshold = Z[0, 2] / 2 
                elif Z.shape[0] >= (n_clusters -1) and (n_clusters -1) > 0:
                    dynamic_color_threshold = Z[-(n_clusters-1), 2] * 0.99 

            results["fig_dend"] = ff.create_dendrogram(
                X, orientation="left", labels=pdf["Espece_User"].tolist(),
                linkagefun=lambda _: Z,
                color_threshold=dynamic_color_threshold if n_clusters > 1 else 0,
                colorscale=color_sequence
            )
            results["fig_dend"].update_layout(
                template="plotly_dark", height=max(650, sub.shape[0] * 20),
                title_text="Dendrogramme", title_x=0.5
            )
        
        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings**2).sum(axis=1)
        vip_data_df = pd.DataFrame({
            "Variable": sub.columns[1:], 
            "Communalité (%)": (communal * 100).round(0).astype(int),
        }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
        results["vip_styled"] = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)'])\
                                           .format({"Communalité (%)": "{:}%"})

        cluster_comp_data = []
        for c_label in sorted(pdf["Cluster"].unique()):
            esp_user_names = sorted(list(pdf.loc[pdf["Cluster"] == c_label, "Espece_User"].unique()))
            cluster_comp_data.append({
                "cluster_label": c_label, "count": len(esp_user_names), "species_list": esp_user_names
            })
        results["cluster_compositions_data"] = cluster_comp_data

    except Exception as e:
        results["error_message"] = f"Une erreur est survenue lors de l'analyse : {e}"
        st.exception(e) # Affiche la trace complète dans la console Streamlit pour le débogage
    
    return results

# ---------------------------------------------------------------------------- #
# CALLBACKS ET LOGIQUE D'INTERACTION
# ---------------------------------------------------------------------------- #
def trigger_analysis_from_button():
    """Prépare les espèces depuis le text_area et lance l'analyse complète."""
    # Conserver les noms uniques saisis par l'utilisateur pour éviter les doublons dès l'entrée
    species_raw_unique_from_text = sorted(list(set(s.strip() for s in st.session_state.species_text_area_content.splitlines() if s.strip())))
    st.session_state.current_analyzed_species_user_names = species_raw_unique_from_text
    st.session_state.species_selected_for_removal = None # Réinitialiser la sélection
    
    if not ref_data.empty:
        n_clusters = st.session_state.get("n_clusters_slider_val", 3) # Utiliser la valeur du slider
        st.session_state.analysis_results = perform_analysis(
            st.session_state.current_analyzed_species_user_names,
            ref_data,
            ref_binom_series,
            n_clusters
        )
        st.session_state.run_counter += 1 # Pour rafraîchir plotly_events
    else:
        st.session_state.analysis_results["error_message"] = "Les données de référence ne sont pas chargées."

def handle_species_removal(species_name_to_remove):
    """Supprime une espèce de la liste et relance l'analyse."""
    if species_name_to_remove in st.session_state.current_analyzed_species_user_names:
        st.session_state.current_analyzed_species_user_names.remove(species_name_to_remove)
    
    # Mettre à jour le contenu du text_area
    st.session_state.species_text_area_content = "\n".join(st.session_state.current_analyzed_species_user_names)
    st.session_state.species_selected_for_removal = None # Réinitialiser après suppression
    
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
    # st.rerun() # Streamlit devrait se réexécuter automatiquement après la modification de session_state via on_click

# ---------------------------------------------------------------------------- #
# LAYOUT DE LA PAGE
# ---------------------------------------------------------------------------- #
col_input, col_pca_plot = st.columns([1, 3])

with col_input:
    st.subheader("CORTEGE FLORISTIQUE")
    # Le slider doit stocker sa valeur dans session_state pour être accessible par perform_analysis
    n_clusters_selected = st.slider(
        "Nombre de clusters", 2, 8, 
        value=st.session_state.get("n_clusters_slider_val", 3), # Conserver la valeur
        key="n_clusters_slider_val", # Clé pour accéder à la valeur via session_state
        disabled=ref_data.empty
    )
    
    # Utiliser la valeur de session_state pour le text_area pour permettre la mise à jour programmatique
    st.text_area(
        "Liste d'espèces (une par ligne)", height=250,
        placeholder="Teucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus\n…",
        key="species_text_area_content", # Lie directement au session_state
        disabled=ref_data.empty
    )

    st.button(
        "Lancer l'analyse", 
        type="primary", 
        disabled=ref_data.empty,
        on_click=trigger_analysis_from_button
    )

    # Afficher les avertissements pour les espèces non trouvées
    if st.session_state.analysis_results.get("not_found_user_raw_names"):
        st.warning(
            "Non trouvées dans la base : " + ", ".join(st.session_state.analysis_results["not_found_user_raw_names"]),
            icon="⚠️"
        )

    # Afficher le message d'erreur général de l'analyse s'il y en a un
    if st.session_state.analysis_results.get("error_message"):
        st.error(st.session_state.analysis_results["error_message"])

    # Section pour la suppression d'espèces après clic sur PCA
    if st.session_state.species_selected_for_removal:
        species_to_remove = st.session_state.species_selected_for_removal
        st.markdown("---")
        st.markdown(f"Voulez-vous supprimer **{species_to_remove}** de l'analyse ?")
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            st.button(f"Oui, supprimer {species_to_remove}", 
                      on_click=handle_species_removal, 
                      args=(species_to_remove,),
                      key=f"confirm_remove_{species_to_remove.replace(' ', '_')}") # Clé unique
        with col_cancel:
            if st.button("Annuler", key=f"cancel_remove_{species_to_remove.replace(' ', '_')}"):
                st.session_state.species_selected_for_removal = None
                st.rerun() # Pour cacher les boutons de confirmation


# Affichage des résultats
with col_pca_plot:
    fig_pca_result = st.session_state.analysis_results.get("fig_pca")
    if fig_pca_result:
        # Utilisation de plotly_events pour capturer les clics
        # La clé doit changer si la figure change pour que les événements soient correctement liés
        selected_points = plotly_events(
            fig_pca_result, 
            click_event=True, 
            key=f"pca_plot_events_{st.session_state.run_counter}", # Clé dynamique
            override_height=fig_pca_result.layout.height,
            override_width="100%"
        )
        if selected_points: # Un point a été cliqué
            clicked_species_name = selected_points[0].get('text')
            if clicked_species_name:
                # Vérifier que l'espèce cliquée est bien dans la liste analysée
                # pour éviter des erreurs si le graphe est obsolète par rapport à la liste.
                if clicked_species_name in st.session_state.current_analyzed_species_user_names:
                    st.session_state.species_selected_for_removal = clicked_species_name
                    st.rerun() # Réexécuter pour afficher les options de suppression
    elif not st.session_state.analysis_results.get("error_message"): # Si pas d'erreur mais pas de PCA
        if ref_data.empty:
             st.warning("Veuillez d'abord charger des données pour afficher le graphique PCA.")
        elif not st.session_state.current_analyzed_species_user_names and st.button_value_if_missing("Lancer l'analyse", False): # après un clic sur "Lancer l'analyse" mais sans espèces
             pass # L'erreur sera gérée dans col_input
        else:
            st.info("Le graphique PCA sera affiché ici après une analyse réussie.")


col_vars, col_cluster_comp_container = st.columns([1, 2])

with col_vars:
    st.subheader("Importance des Variables")
    vip_styled_result = st.session_state.analysis_results.get("vip_styled")
    if vip_styled_result is not None:
        st.write(vip_styled_result.to_html(escape=False), unsafe_allow_html=True)
    elif not st.session_state.analysis_results.get("error_message") and st.session_state.analysis_results.get("sub_data_shape_0", 0) > 0:
        st.info("Le tableau d'importance des variables sera affiché ici.")

with col_cluster_comp_container:
    st.subheader("Composition des Clusters")
    cluster_compositions_data_result = st.session_state.analysis_results.get("cluster_compositions_data")
    if cluster_compositions_data_result:
        num_clusters_found = len(cluster_compositions_data_result)
        if num_clusters_found > 0:
            # Dynamically adjust column widths or number of columns if many clusters
            cols_per_row = min(num_clusters_found, 4) # Max 4 clusters per row
            
            for i in range(0, num_clusters_found, cols_per_row):
                cluster_cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < num_clusters_found:
                        comp_data = cluster_compositions_data_result[i+j]
                        with cluster_cols[j]:
                            st.markdown(f"**Cluster {comp_data['cluster_label']}** ({comp_data['count']} espèces)")
                            for species_name in comp_data['species_list']:
                                st.markdown(f"- {species_name}")
        else:
            st.info("Aucun cluster à afficher.")
    elif not st.session_state.analysis_results.get("error_message") and st.session_state.analysis_results.get("sub_data_shape_0", 0) > 0:
        st.info("La composition des clusters sera affichée ici.")


fig_dend_result = st.session_state.analysis_results.get("fig_dend")
if fig_dend_result:
    st.plotly_chart(fig_dend_result, use_container_width=True)
elif not st.session_state.analysis_results.get("error_message") and st.session_state.analysis_results.get("sub_data_shape_0", 0) > 0 :
    if st.session_state.analysis_results.get("sub_data_shape_0", 0) < 2 :
        st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces).")
    else:
        st.info("Le dendrogramme sera affiché ici.")


if not st.session_state.current_analyzed_species_user_names and not st.session_state.analysis_results.get("error_message") and not ref_data.empty:
    with col_pca_plot: # Afficher ce message seulement au début ou si la liste est vide
        if not any(st.session_state.analysis_results.values()): # Si aucun résultat n'est encore présent
            st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse'.")

st.markdown("---")
st.caption(f"Robin Wojcik, Améten. Date actuelle : {pd.Timestamp('now', tz='Europe/Paris').strftime('%Y-%m-%d %H:%M:%S %Z')}")
st.caption("Note : L'interactivité du graphique PCA (clic pour suppression) utilise la bibliothèque 'streamlit-plotly-events'.")
