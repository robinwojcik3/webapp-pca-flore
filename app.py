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
# Titre principal centré
st.markdown("<h1 style='text-align: center;'>Analyse interactive des clusters botaniques</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------- #
# CHARGEMENT DE LA BASE
# ---------------------------------------------------------------------------- #
@st.cache_data
def load_data(file_path="data_ref.csv"):
    """Charge les données de référence à partir du chemin spécifié."""
    try:
        data = core.read_reference(file_path)
        # Message de succès/info retiré car l'utilisateur ne veut plus le voir
        return data
    except FileNotFoundError:
        st.error(f"ERREUR CRITIQUE: Fichier de données '{file_path}' non trouvé. L'application ne peut pas fonctionner.")
        return pd.DataFrame() # Retourne un DataFrame vide en cas d'erreur critique
    except Exception as e:
        st.error(f"ERREUR CRITIQUE: Impossible de charger les données depuis '{file_path}': {e}")
        return pd.DataFrame()

# Initialisation des données de référence
ref = load_data()

ref_binom = pd.Series(dtype='str')
if not ref.empty:
    ref_binom = (
        ref["Espece"]
        .str.split()
        .str[:2]
        .str.join(" ")
        .str.lower()
    )

# ---------------------------------------------------------------------------- #
# LAYOUT DE LA PAGE
# ---------------------------------------------------------------------------- #

# Ligne supérieure : Cortège (entrées) et Plot PCA
# Ratio ajusté à 1/4 (entrées) et 3/4 (PCA)
col_input, col_pca_plot = st.columns([1, 3])

with col_input:
    st.subheader("CORTEGE") # Titre pour la section des entrées utilisateur

    # Le file_uploader et les messages de chargement de fichier ont été supprimés.
    # L'application utilise 'data_ref.csv' par défaut.
    # Si ref est vide, une erreur critique aura déjà été affichée par load_data.

    n_clusters = st.slider("Nombre de clusters", 2, 8, 3, disabled=ref.empty)

    species_txt = st.text_area(
        "Liste d'espèces (une par ligne)", height=250,
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
        truly_not_found = [s_raw for s_raw, s_bin in zip(species_raw, species_binom) if s_bin not in ref_binom.values]
        if truly_not_found:
            with col_input: # Afficher le message dans la colonne de gauche
                st.warning(
                    "Espèces non trouvées dans la base de référence : " + ", ".join(truly_not_found),
                    icon="⚠️"
                )
        st.stop()

    # Liste des noms non trouvés (ceux saisis mais pas dans ref_binom)
    not_found = [
        s_raw for s_raw, s_bin in zip(species_raw, species_binom)
        if s_bin not in ref_binom.values # Utilise ref_binom directement
    ]
    if not_found:
        with col_input:
            st.warning(
                "Non trouvées dans la base : " + ", ".join(not_found),
                icon="⚠️"
            )
    
    # La section "Espèces trouvées et utilisées pour l'analyse" a été supprimée.

    if sub.shape[0] < n_clusters:
        st.error(f"Le nombre d'espèces trouvées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters}).")
        st.stop()
    
    if sub.shape[0] < 2: # PCA a besoin d'au moins 2 échantillons
        st.error(f"Au moins 2 espèces sont nécessaires pour l'analyse PCA. {sub.shape[0]} espèce(s) trouvée(s).")
        st.stop()

    try:
        labels, pca, coords, X = core.analyse(sub, n_clusters)

        pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
        pdf["Cluster"] = labels.astype(str)
        pdf["Espece"] = sub["Espece"].values

        # FIGURE PCA
        fig_pca = px.scatter(
            pdf,
            x="PC1",
            y="PC2" if coords.shape[1] > 1 else None,
            color="Cluster",
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
            Z = linkage(X, method="ward")
            dynamic_color_threshold = 0
            if n_clusters > 1 and (n_clusters -1) <= Z.shape[0] :
                # Pour k clusters, on veut que les (k-1) fusions les plus "hautes"
                # définissent les couleurs des branches principales.
                # Z est trié par distance. Z[-(k-1), 2] est la distance de la (k-1)ème plus grande fusion.
                # Les liens au-dessus de ce seuil seront colorés.
                # Cela devrait donner n_clusters groupes de couleurs distinctes.
                # Index pour Z: Z a X.shape[0]-1 lignes.
                # Si n_clusters = 2, on veut Z[-1, 2] comme référence.
                # Si n_clusters = X.shape[0], alors Z[-(X.shape[0]-1), 2] qui est Z[0,2].
                idx_threshold = -(n_clusters - 1)
                if idx_threshold == 0: # Cas où n_clusters = 1, mais déjà géré. Ou si n_clusters-1 = Z.shape[0]
                    # Si n_clusters = 1 (pas de coloration souhaitée)
                    # ou si tous les points sont des clusters, Z[0,2] est le plus petit lien
                    dynamic_color_threshold = Z[0, 2] / 2 # pour ne pas colorer
                elif Z.shape[0] >= (n_clusters -1) and (n_clusters -1) > 0: # Assure que l'index est valide
                     dynamic_color_threshold = Z[-(n_clusters-1), 2] * 0.99 # Légèrement en dessous pour assurer la coupe

            fig_dend = ff.create_dendrogram(
                X,
                orientation="left",
                labels=sub["Espece"].tolist(),
                linkagefun=lambda _: Z,
                color_threshold=dynamic_color_threshold if n_clusters > 1 else 0,
            )
            fig_dend.update_layout(
                template="plotly_dark",
                height=max(650, sub.shape[0] * 20),
                title_text="Dendrogramme",
                title_x=0.5
            )
        else:
            fig_dend = None

        # TABLEAUX DESCRIPTIFS - Importance des variables
        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        communal = (loadings**2).sum(axis=1)
        vip_data = {
            "Variable": sub.columns[1:], # Exclut la colonne 'Espece'
            # Communalité en % entier
            "Communalité (%)": (communal * 100).round(0).astype(int),
        }
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
        st.exception(e) # Affiche la trace complète pour le débogage
        st.stop()

# Affichage des résultats dans les colonnes appropriées
with col_pca_plot:
    if fig_pca:
        st.plotly_chart(fig_pca, use_container_width=True)
    elif run and ref.empty:
        # Ce cas ne devrait plus se produire si l'app s'arrête quand ref est vide au début
        st.warning("Veuillez d'abord charger des données pour afficher le graphique PCA.")
    elif run and not species_binom:
        pass 
    elif run:
        st.info("Le graphique PCA sera affiché ici après une analyse réussie.")

# Ligne du milieu : Importance des variables et Composition des clusters
col_vars, col_cluster_comp = st.columns([1, 2]) # Ratio 1/3 et 2/3

with col_vars:
    st.subheader("Importance des Variables")
    if not vip.empty:
        # Affichage avec formatage du pourcentage pour la communalité
        st.dataframe(
            vip,
            use_container_width=True,
            column_config={
                "Communalité (%)": st.column_config.NumberColumn(format="%d%%")
            }
        )
    elif run:
        st.info("Le tableau d'importance des variables sera affiché ici.")

with col_cluster_comp:
    st.subheader("Composition des Clusters")
    if cluster_compositions:
        for comp in cluster_compositions:
            st.markdown(f"**Cluster {comp['cluster_label']}** — {comp['count']} espèces : {comp['species_list']}")
    elif run:
        st.info("La composition des clusters sera affichée ici.")

# Ligne du bas : Dendrogramme
if fig_dend:
    st.plotly_chart(fig_dend, use_container_width=True)
elif run and not ref.empty and species_binom:
    st.info("Le dendrogramme n'a pas pu être généré (nécessite au moins 2 espèces ou problème de seuil).")
elif run and ref.empty:
     # Ce cas ne devrait plus se produire
    st.warning("Veuillez d'abord charger des données pour afficher le dendrogramme.")

# Message initial si l'analyse n'a pas encore été lancée
if not run and not ref.empty:
    with col_pca_plot:
        st.info("Prêt à lancer l'analyse. Configurez les options à gauche et cliquez sur 'Lancer l'analyse'.")
elif not run and ref.empty:
    # Si ref est vide, un message d'erreur critique est déjà affiché.
    # On pourrait ajouter un message ici si nécessaire, mais l'app est déjà "bloquée".
    pass
