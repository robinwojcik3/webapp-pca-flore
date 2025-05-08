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
st.title("Analyse interactive de clusters botaniques")

# ---------------------------------------------------------------------------- #
# CHARGEMENT DE LA BASE
# ---------------------------------------------------------------------------- #
# Placé dans la première colonne (Cortège) pour la logique de l'interface
# uploaded = st.file_uploader("Base de données CSV (optionnel)", type=["csv"])
# if uploaded:
#     ref = core.read_reference(uploaded)
#     st.success(f"{uploaded.name} chargé ({ref.shape[0]} lignes).")
# else:
#     ref = core.read_reference("data_ref.csv") # Assurez-vous que ce fichier existe
#     st.info("Fichier local *data_ref.csv* utilisé.")

# # pré-calcul : version « binomiale » (Genre + épithète) en minuscules
# ref_binom = (
#     ref["Espece"]
#     .str.split()
#     .str[:2]
#     .str.join(" ")
#     .str.lower()
# )

# Données de référence chargées une seule fois au début
# Pour éviter les rechargements multiples lors des interactions avec les widgets
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    if uploaded_file:
        data = core.read_reference(uploaded_file)
        message = f"{uploaded_file.name} chargé ({data.shape[0]} lignes)."
        status = "success"
        return data, message, status
    else:
        try:
            data = core.read_reference(file_path or "data_ref.csv")
            message = f"Fichier local *{file_path or 'data_ref.csv'}* utilisé ({data.shape[0]} lignes)."
            status = "info"
            return data, message, status
        except FileNotFoundError:
            st.error(f"Fichier {file_path or 'data_ref.csv'} non trouvé. Veuillez téléverser un fichier.")
            return pd.DataFrame(), "Erreur: Fichier de données non trouvé.", "error" # Retourne un DataFrame vide en cas d'erreur

# Initialisation des données de référence
# ref, load_message, load_status = load_data() # Tentative de chargement initial
# if load_status == "success":
#     st.success(load_message)
# elif load_status == "info":
#     st.info(load_message)


# ref_binom = (
#     ref["Espece"]
#     .str.split()
#     .str[:2]
#     .str.join(" ")
#     .str.lower()
# ) if not ref.empty else pd.Series(dtype='str')


# ---------------------------------------------------------------------------- #
# LAYOUT DE LA PAGE
# ---------------------------------------------------------------------------- #

# Ligne supérieure : Cortège (entrées) et Plot PCA
col_input, col_pca_plot = st.columns([3, 7]) # Ratio approximatif 30% / 70%

with col_input:
    st.subheader("CORTEGE") # Titre pour la section des entrées utilisateur

    uploaded = st.file_uploader("Base de données CSV (optionnel)", type=["csv"])
    
    # Chargement des données en fonction du fichier uploadé ou local
    if uploaded:
        ref, load_message, load_status = load_data(uploaded_file=uploaded)
    else:
        # Tente de charger data_ref.csv par défaut si aucun fichier n'est uploadé
        ref, load_message, load_status = load_data(file_path="data_ref.csv")

    if load_status == "success":
        st.success(load_message)
    elif load_status == "info":
        st.info(load_message)
    elif load_status == "error" and not uploaded: # Affiche l'erreur seulement si le fichier par défaut n'est pas trouvé et rien n'est uploadé
         st.error(load_message)
    
    # Pré-calcul de ref_binom seulement si ref n'est pas vide
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

    species_txt = st.text_area(
        "Liste d'espèces (une par ligne)", height=250, # Augmentation de la hauteur pour correspondre à la maquette
        placeholder="Teucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus\n…",
        disabled=ref.empty
    )
    species_raw = [s.strip() for s in species_txt.splitlines() if s.strip()]
    species_binom = [" ".join(s.split()[:2]).lower() for s in species_raw]

    run = st.button("Lancer l'analyse", type="primary", disabled=ref.empty)

# Initialisation des variables pour les graphiques et tableaux en dehors du if run
fig_pca = None
fig_dend = None
vip = pd.DataFrame()
cluster_compositions = []

# ---------------------------------------------------------------------------- #
# ANALYSE (déclenchée par le bouton)
# ---------------------------------------------------------------------------- #
if run and not ref.empty:
    if not species_binom:
        st.error("Veuillez saisir au moins un nom d'espèce.")
        st.stop()

    mask = ref_binom.isin(species_binom)
    sub = ref[mask].copy()

    if sub.empty:
        st.error("Aucune des espèces saisies n'a été trouvée dans la base de données.")
        # Afficher les noms non trouvés même si 'sub' est vide au final
        not_found_run = [
            s_raw for s_raw, s_bin in zip(species_raw, species_binom)
            if s_bin not in ref_binom[ref_binom.isin(species_binom)].values # Vérifie contre les valeurs de ref_binom qui sont dans species_binom
        ]
        # Correction: vérifier ceux qui ne sont pas du tout dans ref_binom
        truly_not_found = [s for s in species_raw if " ".join(s.split()[:2]).lower() not in ref_binom.values]

        if truly_not_found:
            st.warning(
                "Espèces non trouvées dans la base de référence : " + ", ".join(truly_not_found),
                icon="⚠️"
            )
        st.stop()

    # Liste des noms non trouvés (ceux saisis mais pas dans ref_binom)
    not_found = [
        s for s in species_raw
        if " ".join(s.split()[:2]).lower() not in ref_binom.values
    ]
    if not_found:
        # Afficher dans la colonne de gauche (input)
        with col_input:
            st.warning(
                "Non trouvées dans la base : " + ", ".join(not_found),
                icon="⚠️"
            )
    
    # Afficher les espèces trouvées et utilisées pour l'analyse
    found_species = sub["Espece"].tolist()
    if found_species:
        with col_input:
            st.info(f"Espèces trouvées et utilisées pour l'analyse : {', '.join(found_species)}")


    if sub.shape[0] < n_clusters:
        st.error(f"Le nombre d'espèces trouvées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters}).")
        st.stop()
    
    if sub.shape[0] < 2: # PCA a besoin d'au moins 2 échantillons
        st.error(f"Au moins 2 espèces sont nécessaires pour l'analyse PCA. {sub.shape[0]} espèce(s) trouvée(s).")
        st.stop()

    try:
        labels, pca, coords, X = core.analyse(sub, n_clusters)

        # Préparation des données pour les graphiques et tableaux
        pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
        pdf["Cluster"] = labels.astype(str)
        pdf["Espece"] = sub["Espece"].values

        # FIGURE PCA
        fig_pca = px.scatter(
            pdf,
            x="PC1",
            y="PC2" if coords.shape[1] > 1 else None, # Gérer le cas où il n'y a qu'une PC
            color="Cluster",
            text="Espece",
            template="plotly_dark", # Thème sombre pour correspondre à la maquette
            height=600, # Hauteur ajustée
            # Pourrait ajouter color_discrete_map ici si les couleurs spécifiques des clusters sont connues
        )
        fig_pca.update_traces(textposition="top center")
        fig_pca.update_layout(
            title_text="Clusters d'espèces (PCA)", # Titre du graphique
            title_x=0.5, # Centrer le titre
            legend_title_text='Cluster' 
        )


        # DENDROGRAMME
        if X.shape[0] > 1: # Le dendrogramme nécessite au moins 2 points de données
            Z = linkage(X, method="ward")
            fig_dend = ff.create_dendrogram(
                X,
                orientation="left",
                labels=sub["Espece"].tolist(),
                linkagefun=lambda _: Z,
                color_threshold=0 # Pour colorer les branches différemment si souhaité (optionnel)
            )
            fig_dend.update_layout(
                template="plotly_dark", 
                height=max(650, sub.shape[0] * 20), # Hauteur dynamique
                # width=900 # La largeur sera gérée par use_container_width=True
                title_text="Dendrogramme",
                title_x=0.5
            )
        else:
            fig_dend = None # Pas de dendrogramme si pas assez de données


        # TABLEAUX DESCRIPTIFS - Importance des variables
        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings**2).sum(axis=1)
        vip_data = {
            "Variable": sub.columns[1:], # Exclut la colonne 'Espece'
            "Communalité (%)": (communal * 100).round(1),
        }
        # S'assurer que les longueurs correspondent pour créer le DataFrame
        min_len = min(len(vip_data["Variable"]), len(vip_data["Communalité (%)"]))
        vip = (
            pd.DataFrame({
                "Variable": vip_data["Variable"][:min_len],
                "Communalité (%)": vip_data["Communalité (%)"][:min_len]
            })
            .sort_values("Communalité (%)", ascending=False)
            .reset_index(drop=True)
        )
        
        # TABLEAUX DESCRIPTIFS - Composition des clusters
        cluster_compositions = []
        for c_label in sorted(pdf["Cluster"].unique()):
            esp = pdf.loc[pdf["Cluster"] == c_label, "Espece"].tolist()
            cluster_compositions.append({
                "cluster_label": c_label,
                "count": len(esp),
                "species_list": ", ".join(esp)
            })

    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'analyse : {e}")
        st.stop()


# Affichage des résultats dans les colonnes appropriées
with col_pca_plot:
    if fig_pca:
        # st.subheader("Clusters d'espèces (PCA)") # Déplacé dans la mise en page du graphique Plotly
        st.plotly_chart(fig_pca, use_container_width=True)
    elif run and ref.empty:
        st.warning("Veuillez d'abord charger des données pour afficher le graphique PCA.")
    elif run and not species_binom: # Si on a cliqué sur run sans espèces
        pass # L'erreur est déjà gérée dans col_input
    elif run: # Si 'run' est cliqué mais fig_pca n'est pas généré (ex: erreur avant)
        st.info("Le graphique PCA sera affiché ici après une analyse réussie.")


# Ligne du milieu : Importance des variables et Composition des clusters
col_vars, col_cluster_comp = st.columns([1, 2]) # Ratio 1/3 et 2/3

with col_vars:
    st.subheader("Importance des Variables")
    if not vip.empty:
        st.dataframe(vip, use_container_width=True)
        # Note: Pour un style de tableau plus avancé (ex: barres de couleur),
        # il faudrait utiliser st.markdown avec du HTML/CSS ou une bibliothèque tierce.
    elif run:
        st.info("Le tableau d'importance des variables sera affiché ici.")

with col_cluster_comp:
    st.subheader("Composition des Clusters")
    if cluster_compositions:
        for comp in cluster_compositions:
            st.markdown(f"**Cluster {comp['cluster_label']}** — {comp['count']} espèces : {comp['species_list']}")
    elif run:
        st.info("La composition des clusters sera affichée ici.")


# Ligne du bas : Dendrogramme (pleine largeur par défaut)
if fig_dend:
    # st.subheader("Dendrogramme") # Déplacé dans la mise en page du graphique Plotly
    st.plotly_chart(fig_dend, use_container_width=True)
elif run and not ref.empty and species_binom: # Si analyse lancée mais pas de dendrogramme (ex: 1 espèce)
    st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces).")
elif run and ref.empty:
    st.warning("Veuillez d'abord charger des données pour afficher le dendrogramme.")


# Pour débogage : afficher l'état des variables clés si l'analyse n'a pas tourné
if not run:
    with col_pca_plot:
        st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse'.")

