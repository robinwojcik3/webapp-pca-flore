"""
Web-app Streamlit : PCA + clustering botaniques
Reconnaît les noms saisis au format « Genre épithète » même si la base
contient les auteurs (« Acacia mearnsii » ↔ « Acacia mearnsii De Wild. »)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage
# Assurez-vous que le fichier core.py est dans le même répertoire ou accessible
import core 

# ---------------------------------------------------------------------------- #
# CONFIGURATION UI
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="PCA flore interactive", layout="wide")
# Centrer le titre principal en utilisant du HTML dans st.markdown
st.markdown("<h1 style='text-align: center;'>Analyse interactive de clusters botaniques</h1>", unsafe_allow_html=True)


# Données de référence chargées une seule fois au début
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    if uploaded_file:
        try:
            data = core.read_reference(uploaded_file)
            message = f"{uploaded_file.name} chargé ({data.shape[0]} lignes)."
            status = "success"
            return data, message, status
        except Exception as e:
            return pd.DataFrame(), f"Erreur lors de la lecture du fichier téléversé: {e}", "error"
    else:
        try:
            data = core.read_reference(file_path or "data_ref.csv")
            # Message pour le fichier local n'est plus affiché par défaut
            message = f"Fichier local *{file_path or 'data_ref.csv'}* utilisé ({data.shape[0]} lignes)."
            status = "info_local" # Statut spécifique pour ne pas l'afficher si non désiré
            return data, message, status
        except FileNotFoundError:
            # Pas d'erreur affichée ici si le fichier local par défaut n'est pas trouvé,
            # l'utilisateur sera invité à uploader un fichier.
            return pd.DataFrame(), f"Fichier {file_path or 'data_ref.csv'} non trouvé.", "error_local_not_found"
        except Exception as e:
            return pd.DataFrame(), f"Erreur lors de la lecture du fichier local: {e}", "error"

# Initialisation des variables d'état pour stocker les résultats entre les exécutions
if 'coords' not in st.session_state:
    st.session_state.coords = None
if 'pca_generated_once' not in st.session_state:
    st.session_state.pca_generated_once = False


# ---------------------------------------------------------------------------- #
# LAYOUT DE LA PAGE
# ---------------------------------------------------------------------------- #

# Ligne supérieure : Cortège (entrées) et Plot PCA
col_input, col_pca_plot = st.columns([1, 3]) # Ratio 1/4 (25%) et 3/4 (75%)

with col_input:
    st.subheader("CORTEGE")

    uploaded = st.file_uploader(
        "Base de données CSV (optionnel)", 
        type=["csv"],
        label_visibility="collapsed" # Masquer le label par défaut du file_uploader
    )
    
    ref = pd.DataFrame() # Initialiser ref
    load_message = ""
    load_status = ""

    if uploaded:
        ref, load_message, load_status = load_data(uploaded_file=uploaded)
        if load_status == "success":
            st.success(load_message)
        elif load_status == "error":
            st.error(load_message)
            st.stop() # Arrêter si le fichier uploadé ne peut pas être lu
    else:
        # Tente de charger data_ref.csv par défaut si aucun fichier n'est uploadé
        # N'affiche plus de message pour le fichier local par défaut
        ref, load_message, load_status = load_data(file_path="data_ref.csv")
        if load_status == "error": # Erreur critique lors du chargement du fichier local
             st.error(load_message)
             st.stop()
        # Si "error_local_not_found", ref sera vide, l'interface invitera à uploader.

    if ref.empty and not uploaded:
        st.info("Veuillez téléverser un fichier CSV de données ou vous assurer que 'data_ref.csv' est présent.")
    
    if not ref.empty:
        ref_binom = (
            ref["Espece"]
            .str.split()
            .str[:2]
            .str.join(" ")
            .str.lower()
        )
    else:
        ref_binom = pd.Series(dtype='str')

    n_clusters = st.slider("Nombre de clusters", 2, 8, 3, disabled=ref.empty)
    
    # Options interactives supplémentaires
    linkage_methods = ["ward", "average", "complete", "single"]
    linkage_method_choice = st.selectbox(
        "Méthode de linkage (dendrogramme)", 
        linkage_methods, 
        index=0, 
        disabled=ref.empty
    )

    # Sélecteurs pour les composantes PCA (se peuplent après la première analyse)
    pc_options = []
    if st.session_state.coords is not None and st.session_state.coords.shape[1] > 0:
        pc_options = [f"PC{i+1}" for i in range(st.session_state.coords.shape[1])]
    
    x_axis_choice = st.selectbox(
        "Axe X du PCA", 
        pc_options, 
        index=0, 
        disabled=not pc_options # Désactivé si pas d'options (pas encore d'analyse)
    )
    
    y_axis_choice_options = [pc for pc in pc_options if pc != x_axis_choice]
    # S'assurer que l'index pour y_axis est valide
    y_axis_index = 0
    if len(y_axis_choice_options) > 0:
        if "PC2" in y_axis_choice_options and x_axis_choice != "PC2":
             y_axis_index = y_axis_choice_options.index("PC2")
        # else, l'index 0 est conservé par défaut.
    
    y_axis_choice = st.selectbox(
        "Axe Y du PCA", 
        y_axis_choice_options, 
        index=y_axis_index, 
        disabled=not y_axis_choice_options or len(y_axis_choice_options) == 0
    )

    species_txt = st.text_area(
        "Liste d'espèces (une par ligne)", height=200, 
        placeholder="Teucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus\n…",
        disabled=ref.empty
    )
    species_raw = [s.strip() for s in species_txt.splitlines() if s.strip()]
    species_binom = [" ".join(s.split()[:2]).lower() for s in species_raw]

    run = st.button("Lancer l'analyse", type="primary", disabled=ref.empty)

# Initialisation des variables pour les graphiques et tableaux
fig_pca = None
fig_dend = None
vip = pd.DataFrame()
cluster_compositions = []
pdf_analysis_results = pd.DataFrame() # Pour stocker les résultats de l'analyse pour le dendrogramme

# ---------------------------------------------------------------------------- #
# ANALYSE (déclenchée par le bouton)
# ---------------------------------------------------------------------------- #
if run and not ref.empty:
    if not species_binom:
        with col_input: # Afficher l'erreur dans la colonne de saisie
            st.error("Veuillez saisir au moins un nom d'espèce.")
        st.stop()

    mask = ref_binom.isin(species_binom)
    sub = ref[mask].copy()

    if sub.empty:
        with col_input:
            st.error("Aucune des espèces saisies n'a été trouvée dans la base de données.")
        truly_not_found = [s for s in species_raw if " ".join(s.split()[:2]).lower() not in ref_binom.values]
        if truly_not_found:
            with col_input:
                st.warning(
                    "Espèces non trouvées dans la base de référence : " + ", ".join(truly_not_found),
                    icon="⚠️"
                )
        st.stop()

    not_found = [s for s in species_raw if " ".join(s.split()[:2]).lower() not in ref_binom.values]
    if not_found:
        with col_input:
            st.warning(
                "Non trouvées dans la base : " + ", ".join(not_found),
                icon="⚠️"
            )
    
    # La liste "Espèces trouvées et utilisées pour l'analyse" a été enlevée.

    if sub.shape[0] < n_clusters:
        with col_input:
            st.error(f"Le nombre d'espèces trouvées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters}).")
        st.stop()
    
    if sub.shape[0] < 2:
        with col_input:
            st.error(f"Au moins 2 espèces sont nécessaires pour l'analyse PCA. {sub.shape[0]} espèce(s) trouvée(s).")
        st.stop()

    try:
        labels, pca, current_coords, X = core.analyse(sub, n_clusters)
        st.session_state.coords = current_coords # Mettre à jour les coords dans session_state
        st.session_state.pca_generated_once = True


        pdf_analysis_results = pd.DataFrame(st.session_state.coords, columns=[f"PC{i+1}" for i in range(st.session_state.coords.shape[1])])
        pdf_analysis_results["Cluster"] = labels.astype(str)
        pdf_analysis_results["Espece"] = sub["Espece"].values

        # Définir une palette de couleurs pour la cohérence
        cluster_ids_sorted = sorted(pdf_analysis_results["Cluster"].unique())
        color_palette = px.colors.qualitative.Plotly 
        cluster_color_map = {
            cluster_id: color_palette[i % len(color_palette)] 
            for i, cluster_id in enumerate(cluster_ids_sorted)
        }
        pdf_analysis_results["color_code"] = pdf_analysis_results["Cluster"].map(cluster_color_map)


        # FIGURE PCA
        # Utiliser les choix de l'utilisateur pour les axes, ou les valeurs par défaut si l'analyse vient d'être lancée
        # et que les selectbox n'ont pas encore été mis à jour par un re-run.
        pc_options_current = [f"PC{i+1}" for i in range(st.session_state.coords.shape[1])]
        
        # S'assurer que x_axis_choice et y_axis_choice sont valides après l'analyse
        # Si l'utilisateur n'a pas encore interagi avec les nouveaux selectbox après la 1ère analyse,
        # on utilise PC1/PC2 par défaut.
        # Si l'utilisateur a choisi, on utilise ses choix.
        
        final_x_axis = x_axis_choice if x_axis_choice in pc_options_current else pc_options_current[0]
        
        valid_y_options = [pc for pc in pc_options_current if pc != final_x_axis]
        final_y_axis = None
        if valid_y_options:
            if y_axis_choice in valid_y_options:
                final_y_axis = y_axis_choice
            else:
                final_y_axis = valid_y_options[0] if len(valid_y_options) > 0 else None


        fig_pca = px.scatter(
            pdf_analysis_results,
            x=final_x_axis,
            y=final_y_axis if st.session_state.coords.shape[1] > 1 and final_y_axis else None,
            color="Cluster",
            color_discrete_map=cluster_color_map,
            text="Espece",
            template="plotly_dark",
            height=600,
        )
        fig_pca.update_traces(textposition="top center")
        fig_pca.update_layout(
            title_text="Clusters d'espèces (PCA)",
            title_x=0.5,
            legend_title_text='Cluster'
        )

        # DENDROGRAMME
        if X.shape[0] > 1:
            Z = linkage(X, method=linkage_method_choice) # Utilisation du choix utilisateur
            
            # Coloration des étiquettes du dendrogramme
            species_to_color_map = pd.Series(pdf_analysis_results.color_code.values, index=pdf_analysis_results.Espece).to_dict()

            fig_dend = ff.create_dendrogram(
                X,
                orientation="left",
                labels=sub["Espece"].tolist(),
                linkagefun=lambda _: Z,
                # color_threshold=0 # Pas utilisé pour la coloration par cluster externe
            )
            fig_dend.update_layout(
                template="plotly_dark", 
                height=max(650, sub.shape[0] * 20),
                title_text="Dendrogramme",
                title_x=0.5
            )
            
            # Appliquer les couleurs aux étiquettes du dendrogramme
            # Les étiquettes sont sur yaxis car orientation='left'
            dendro_labels_ordered = fig_dend.layout.yaxis.ticktext
            if dendro_labels_ordered: # S'assurer que les étiquettes existent
                tick_colors = [species_to_color_map.get(str(label), 'grey') for label in dendro_labels_ordered]
                fig_dend.update_layout(yaxis_tickfont_color=tick_colors) # Applique un tableau de couleurs
        else:
            fig_dend = None

        # TABLEAUX DESCRIPTIFS - Importance des variables
        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings**2).sum(axis=1)
        vip_data_vars = sub.columns[1:] # Exclut la colonne 'Espece'
        
        # S'assurer que communal a la bonne longueur
        if len(communal) == len(vip_data_vars):
            vip_df_data = {
                "Variable": vip_data_vars,
                "Communalité (%)": (communal * 100).round(0).astype(int),
            }
            vip = (
                pd.DataFrame(vip_df_data)
                .sort_values("Communalité (%)", ascending=False)
                .reset_index(drop=True)
            )
            # Ajouter le signe '%' pour l'affichage
            vip["Communalité (%)"] = vip["Communalité (%)"].astype(str) + "%"
        else:
            vip = pd.DataFrame(columns=["Variable", "Communalité (%)"]) # Tableau vide en cas de souci
            st.warning("Impossible de calculer l'importance des variables en raison d'une incohérence de dimensions.")


        # TABLEAUX DESCRIPTIFS - Composition des clusters
        cluster_compositions = []
        for c_label in sorted(pdf_analysis_results["Cluster"].unique()):
            esp = pdf_analysis_results.loc[pdf_analysis_results["Cluster"] == c_label, "Espece"].tolist()
            cluster_compositions.append({
                "cluster_label": c_label,
                "count": len(esp),
                "species_list": ", ".join(esp)
            })

    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'analyse : {e}")
        st.exception(e) # Affiche la trace de la pile pour le débogage
        st.stop()


# Affichage des résultats dans les colonnes appropriées
with col_pca_plot:
    if fig_pca:
        st.plotly_chart(fig_pca, use_container_width=True)
    elif run and ref.empty:
        st.warning("Veuillez d'abord charger des données pour afficher le graphique PCA.")
    elif run and not species_binom and not ref.empty:
        pass 
    elif st.session_state.pca_generated_once and not fig_pca: # Analyse lancée, mais fig_pca non généré
        st.info("Le graphique PCA sera affiché ici après une analyse réussie.")
    elif not run and not ref.empty :
         st.info("Configurez les options à gauche et cliquez sur 'Lancer l'analyse' pour voir le PCA.")


# Ligne du milieu : Importance des variables et Composition des clusters
col_vars, col_cluster_comp = st.columns([1, 2]) 

with col_vars:
    st.subheader("Importance des Variables")
    if not vip.empty:
        st.dataframe(vip, use_container_width=True)
    elif run:
        st.info("Le tableau d'importance des variables sera affiché ici.")

with col_cluster_comp:
    st.subheader("Composition des Clusters")
    if cluster_compositions:
        for comp in cluster_compositions:
            st.markdown(f"**Cluster {comp['cluster_label']}** ({cluster_color_map.get(comp['cluster_label'], '#FFFFFF')}) — {comp['count']} espèces : {comp['species_list']}")
    elif run:
        st.info("La composition des clusters sera affichée ici.")


# Ligne du bas : Dendrogramme
if fig_dend:
    st.plotly_chart(fig_dend, use_container_width=True)
elif run and not ref.empty and species_binom: 
    if X.shape[0] <= 1 : # Condition où le dendrogramme n'est pas généré
        st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces).")
elif run and ref.empty:
    st.warning("Veuillez d'abord charger des données pour afficher le dendrogramme.")

if not run and not ref.empty:
    with col_pca_plot: # Message initial si données chargées mais pas d'analyse lancée
        st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse'.")
elif not run and ref.empty and not uploaded: # Message si aucune donnée n'est chargée au démarrage
    with col_pca_plot:
        st.info("Veuillez téléverser un fichier de données pour commencer.")

