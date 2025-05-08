import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
from scipy.spatial import ConvexHull
import numpy as np

# Assurez-vous que le fichier core.py est dans le même répertoire ou accessible
# Si core.py n'est pas disponible, certaines fonctionnalités d'analyse ne marcheront pas.
# Pour les besoins de cet exemple, nous allons simuler sa présence si non trouvé.
try:
    import core
except ImportError:
    # Simulation de core.py si non trouvé pour permettre au script de se charger
    st.warning("Le fichier `core.py` n'a pas été trouvé. Certaines fonctionnalités d'analyse seront simulées ou désactivées.")
    class MockPCA:
        def __init__(self, n_components):
            self.n_components = n_components
            self.components_ = np.random.rand(10, n_components) # Simule 10 traits
            self.explained_variance_ = np.random.rand(n_components)

        def fit_transform(self, X):
            return np.random.rand(X.shape[0], self.n_components)

    class core:
        @staticmethod
        def read_reference(file_path):
            # Crée un DataFrame de démo si le fichier n'existe pas
            try:
                return pd.read_csv(file_path)
            except FileNotFoundError:
                st.warning(f"Fichier '{file_path}' non trouvé. Utilisation de données de démonstration.")
                data = {
                    'Espece': [f'Genre{i} espece{i} Auteur{i}' for i in range(1, 21)],
                }
                # Ajout de 10 colonnes de traits numériques aléatoires
                for j in range(1, 11):
                    data[f'Trait{j}'] = np.random.rand(20) * 10
                return pd.DataFrame(data)

        @staticmethod
        def analyse(sub_df, n_clusters):
            if sub_df.shape[1] <= 1: # Pas assez de colonnes de traits
                 raise ValueError("Pas assez de colonnes de traits pour l'analyse PCA. Le DataFrame 'sub' doit contenir des colonnes numériques au-delà de 'Espece'.")
            
            # S'assurer que X ne contient que des données numériques
            X_data = sub_df.select_dtypes(include=np.number)
            
            if X_data.empty:
                raise ValueError("Aucune donnée numérique trouvée pour l'analyse PCA après sélection des types.")

            if X_data.shape[0] < 2: # Pas assez d'échantillons
                raise ValueError(f"Au moins 2 espèces sont nécessaires pour l'analyse PCA. {X_data.shape[0]} trouvée(s).")

            # Simuler PCA
            n_components_pca = min(2, X_data.shape[1], X_data.shape[0])
            if n_components_pca == 0:
                 raise ValueError("Impossible de déterminer le nombre de composants pour la PCA.")

            pca_model = MockPCA(n_components=n_components_pca)
            coords_data = pca_model.fit_transform(X_data)
            
            # Simuler les labels de cluster
            if X_data.shape[0] == 0: # Si X_data est vide après filtrage, ce qui ne devrait pas arriver ici
                labels_data = np.array([])
            elif n_clusters > X_data.shape[0]:
                 labels_data = np.arange(X_data.shape[0]) % X_data.shape[0] # Chaque point son cluster si plus de clusters que de points
            else:
                 labels_data = np.random.randint(0, n_clusters, size=X_data.shape[0])
            
            return labels_data, pca_model, coords_data, X_data

# ---------------------------------------------------------------------------- #
# CONFIGURATION UI & STYLE
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="PCA flore interactive", layout="wide")
st.markdown("<h1 style='text-align: center;'>Analyse interactive des clusters botaniques</h1>", unsafe_allow_html=True)

# CSS personnalisé pour encadrer les sections
st.markdown("""
<style>
.styled-container {
    border: 1.5px solid #F0F0F0; /* Bordure fine, très claire (presque blanche) */
    border-radius: 12px;        /* Bords arrondis */
    padding: 1.5rem;            /* Marge intérieure */
    margin-bottom: 1.5rem;      /* Marge inférieure pour espacer les sections */
    background-color: rgba(40, 40, 42, 0.5); /* Fond subtil légèrement différent du thème sombre par défaut, optionnel */
}
.styled-container .stDataFrame, .styled-container .stPlotlyChart {
    margin-bottom: 0 !important; /* Éviter double marge pour éléments internes */
}
/* Ajustement pour que les sous-titres dans les conteneurs stylés aient une bonne apparence */
.styled-container h3 {
    margin-top: 0;
    color: #FAFAFA; /* Couleur claire pour les sous-titres */
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------- #
# CHARGEMENT DE LA BASE
# ---------------------------------------------------------------------------- #
@st.cache_data
def load_data(file_path="data_ref.csv"):
    """Charge les données de référence à partir du chemin spécifié."""
    try:
        data = core.read_reference(file_path)
        if data.select_dtypes(include=np.number).empty and 'Trait1' not in data.columns: # Vérifie si des données numériques existent ou si c'est la démo
            st.error("ERREUR CRITIQUE: Les données de référence ne contiennent aucune colonne de trait numérique pour l'analyse.")
            return pd.DataFrame()
        return data
    except FileNotFoundError:
        st.error(f"ERREUR CRITIQUE: Fichier de données '{file_path}' non trouvé. L'application ne peut pas fonctionner sans `core.py` ou ce fichier.")
        return pd.DataFrame() # Retourne un DataFrame vide
    except Exception as e:
        st.error(f"ERREUR CRITIQUE: Impossible de charger les données depuis '{file_path}': {e}")
        return pd.DataFrame() # Retourne un DataFrame vide

ref = load_data() # Tente de charger data_ref.csv, sinon utilise la démo de core.py simulé

ref_binom_series = pd.Series(dtype='str')
if not ref.empty and "Espece" in ref.columns:
    ref_binom_series = (
        ref["Espece"]
        .astype(str) # Assurer que c'est une chaîne
        .str.split()
        .str[:2]
        .str.join(" ")
        .str.lower()
    )
else:
    if not ref.empty:
        st.warning("La colonne 'Espece' est manquante dans les données de référence. La normalisation des noms ne pourra pas s'effectuer.")


# ---------------------------------------------------------------------------- #
# LAYOUT DE LA PAGE
# ---------------------------------------------------------------------------- #
col_input, col_pca_plot_container = st.columns([1, 3]) # Renommé pour clarté

with col_input:
    st.subheader("PARAMÈTRES D'ANALYSE")
    n_clusters_selected = st.slider("Nombre de clusters souhaités", 2, 8, 3, key="n_clusters_slider", disabled=ref.empty)
    species_txt = st.text_area(
        "Liste d'espèces (une par ligne, format 'Genre epithete')", height=250,
        placeholder="Exemple:\nTeucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus\n...",
        disabled=ref.empty
    )
    # Nettoyage et déduplication des espèces saisies par l'utilisateur
    species_raw_unique = sorted(list(set(s.strip() for s in species_txt.splitlines() if s.strip())))
    # Normalisation des noms d'espèces utilisateur (genre + épithète en minuscule)
    species_binom_user_unique = [" ".join(s.split()[:2]).lower() for s in species_raw_unique]

    run = st.button("🚀 Lancer l'analyse", type="primary", disabled=ref.empty, use_container_width=True)

# Initialisation des variables pour les figures et données
fig_pca = None
fig_dend = None
vip_styled = None # Pour le DataFrame stylé des importances
cluster_compositions_data = []

# ---------------------------------------------------------------------------- #
# ANALYSE (déclenchée par le bouton)
# ---------------------------------------------------------------------------- #
if run and not ref.empty:
    if not species_binom_user_unique:
        st.error("Veuillez saisir au moins un nom d'espèce valide.")
        st.stop()

    # Logique de déduplication pour la création de 'sub' :
    # Pour chaque espèce unique normalisée fournie par l'utilisateur,
    # on prend la première occurrence correspondante dans la base de référence 'ref'.
    indices_to_keep_from_ref = []
    if not ref_binom_series.empty:
        ref_indexed_binom = ref_binom_series.reset_index() # Conserve l'index original de 'ref'
        ref_indexed_binom.columns = ['Original_Ref_Index', 'ref_binom_val']

        for user_binom_specie in species_binom_user_unique:
            matches_in_ref = ref_indexed_binom[ref_indexed_binom['ref_binom_val'] == user_binom_specie]
            if not matches_in_ref.empty:
                indices_to_keep_from_ref.append(matches_in_ref['Original_Ref_Index'].iloc[0])
    
    # Création du sous-ensemble de données 'sub' pour l'analyse
    if indices_to_keep_from_ref:
        sub = ref.loc[list(set(indices_to_keep_from_ref))].copy() # Utilise set pour garantir des index uniques
    else:
        sub = pd.DataFrame(columns=ref.columns)

    # Vérification des espèces non trouvées et information à l'utilisateur
    found_ref_binom_values_in_sub = []
    if not sub.empty and "Espece" in sub.columns:
        found_ref_binom_values_in_sub = (
            sub["Espece"].astype(str).str.split().str[:2].str.join(" ").str.lower().tolist()
        )

    not_found_user_raw_names = []
    for i, user_binom_name in enumerate(species_binom_user_unique):
        if user_binom_name not in found_ref_binom_values_in_sub:
            not_found_user_raw_names.append(species_raw_unique[i])
            
    if not_found_user_raw_names:
        with col_input: # Afficher l'avertissement dans la colonne de saisie
            st.warning(
                "Espèces non trouvées dans la base de référence : \n- " + "\n- ".join(not_found_user_raw_names),
                icon="⚠️"
            )

    if sub.empty:
        st.error("Aucune des espèces saisies (après déduplication et recherche) n'a pu être appariée dans la base de référence. Analyse impossible.")
        st.stop()
    
    if sub.select_dtypes(include=np.number).empty:
        st.error("Les espèces sélectionnées ne disposent pas de données de traits numériques pour l'analyse.")
        st.stop()

    # Vérifications avant de lancer l'analyse principale
    if sub.shape[0] < n_clusters_selected:
        st.error(f"Le nombre d'espèces valides trouvées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters_selected}). Veuillez réduire le nombre de clusters ou fournir plus d'espèces correspondantes.")
        st.stop()
    
    if sub.shape[0] < 2:
        st.error(f"Au moins 2 espèces valides sont nécessaires pour l'analyse PCA. {sub.shape[0]} espèce(s) trouvée(s) et utilisable(s).")
        st.stop()

    min_points_for_hull = 3 # Minimum de points pour une enveloppe convexe
    
    # Mappage des noms binomiaux de 'sub' vers les noms bruts saisis par l'utilisateur pour l'affichage
    user_input_binom_to_raw_map = {
        " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_raw_unique
    }

    try:
        with st.spinner("Analyse en cours... Veuillez patienter."):
            # L'appel à core.analyse doit retourner: labels, pca_model, coords, X_numeric
            # X_numeric doit être le DataFrame numérique utilisé pour la PCA/Dendrogramme
            labels, pca, coords, X = core.analyse(sub, n_clusters_selected)

        pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
        pdf["Cluster"] = labels.astype(str) # Convertir les labels de cluster en string pour Plotly
        
        # Assurer que 'Espece' existe dans 'sub' et que les index correspondent
        if "Espece" in sub.columns and len(sub.index) == len(pdf.index):
             pdf["Espece_Ref"] = sub["Espece"].values # Noms de la base de référence
        else: # Fallback si l'alignement est un problème
            pdf["Espece_Ref"] = [f"Espèce {i+1}" for i in range(len(pdf))]


        # Fonction pour récupérer le nom original saisi par l'utilisateur
        def get_user_input_name(full_ref_name):
            binom_ref_name = " ".join(str(full_ref_name).split()[:2]).lower()
            return user_input_binom_to_raw_map.get(binom_ref_name, str(full_ref_name))

        pdf["Espece_User"] = pdf["Espece_Ref"].apply(get_user_input_name) # Noms tels que saisis par l'utilisateur

        # Palette de couleurs pour les clusters
        color_sequence = px.colors.qualitative.Plotly 
        
        # Création du graphique PCA
        fig_pca = px.scatter(
            pdf,
            x="PC1",
            y="PC2" if coords.shape[1] > 1 else None, # Gère le cas où il n'y a qu'une PC
            color="Cluster",
            text="Espece_User", # Afficher le nom utilisateur
            template="plotly_dark",
            height=650,
            color_discrete_sequence=color_sequence,
            title="Analyse en Composantes Principales (PCA) des Espèces"
        )
        fig_pca.update_traces(
            textposition="top center",
            marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
            hovertemplate="<b>%{text}</b><br>Cluster: %{customdata[0]}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            customdata=pdf[['Cluster']] # Ajouter Cluster aux customdata pour hover
        )
        fig_pca.update_layout(
            title_x=0.5, # Centrer le titre
            legend_title_text='Cluster',
            xaxis_title="Axe Principal 1",
            yaxis_title="Axe Principal 2" if coords.shape[1] > 1 else None,
        )
        
        # Ajout des enveloppes convexes si possible (au moins 2D et assez de points)
        unique_clusters = sorted(pdf["Cluster"].unique())
        cluster_color_map = {cluster_label: color_sequence[i % len(color_sequence)] for i, cluster_label in enumerate(unique_clusters)}

        if coords.shape[1] > 1: # Nécessite au moins 2 dimensions pour l'enveloppe
            for cluster_label in unique_clusters:
                cluster_points_df = pdf[pdf["Cluster"] == cluster_label]
                unique_cluster_points = cluster_points_df[["PC1", "PC2"]].drop_duplicates().values
                
                if len(unique_cluster_points) >= min_points_for_hull:
                    try:
                        hull = ConvexHull(unique_cluster_points)
                        hull_points = unique_cluster_points[hull.vertices]
                        path_x = np.append(hull_points[:, 0], hull_points[0, 0]) # Fermer le polygone
                        path_y = np.append(hull_points[:, 1], hull_points[0, 1])

                        fig_pca.add_trace(go.Scatter(
                            x=path_x, y=path_y,
                            fill="toself", fillcolor=cluster_color_map[cluster_label],
                            line=dict(color=cluster_color_map[cluster_label], width=1.5),
                            mode='lines', name=f'Enveloppe Cluster {cluster_label}',
                            opacity=0.15, showlegend=False, hoverinfo='skip'
                        ))
                    except Exception as e_hull: # pylint: disable=broad-except
                        # Peut arriver si les points sont colinéaires
                        print(f"Avertissement: Impossible de générer l'enveloppe convexe pour le cluster {cluster_label}: {e_hull}")
                # else: (Optionnel) message si pas assez de points pour une enveloppe

        # Création du Dendrogramme
        # X doit être le DataFrame numérique utilisé pour le clustering (avant PCA pour le dendrogramme typiquement)
        # S'assurer que X (données d'entrée de linkage) et les labels pdf["Espece_User"] sont dans le même ordre et de même taille
        if X.shape[0] > 1 and X.shape[0] == len(pdf["Espece_User"]):
            Z = linkage(X, method="ward") # Utilise les données originales X, pas les coords PCA
            
            # Calcul du seuil de couleur dynamique pour le dendrogramme
            # Vise à colorer selon le nombre de clusters demandé
            dynamic_color_threshold = 0
            if n_clusters_selected > 1 and (n_clusters_selected -1) <= Z.shape[0] :
                # L'index pour Z est (nombre de points - nombre de clusters)
                # Z contient n-1 fusions. Si on veut k clusters, on coupe à la (n-k)ème fusion la plus haute.
                # Les distances sont dans Z[:, 2], triées.
                # L'index -(n_clusters_selected-1) correspond à la (n_clusters_selected-1)-ième plus grande distance de fusion
                # qui, si coupée juste en dessous, devrait donner n_clusters_selected clusters.
                # Ou, plus simplement, si on veut k clusters, on regarde la distance de la (N-k)ème fusion.
                # Z est de taille (N-1, 4). Les fusions sont ordonnées par distance.
                # Pour k clusters, on a besoin de k-1 "groupes" de branches distinctes.
                # La (N-k)ème ligne de Z (0-indexé) est la fusion qui réduit le nombre de clusters de k+1 à k.
                # Donc, on prend la distance de Z[N-k, 2].
                idx_threshold = X.shape[0] - n_clusters_selected
                if 0 <= idx_threshold < Z.shape[0]:
                    dynamic_color_threshold = Z[idx_threshold, 2] * 1.01 # Un peu au-dessus pour assurer la coupe
                elif idx_threshold < 0 and Z.shape[0] > 0: # Plus de clusters que de fusions possibles (cas N < k)
                     dynamic_color_threshold = Z[0, 2] / 2 # Colorier seulement les premières fusions
                else: # Cas où N = k, ou N=1
                    dynamic_color_threshold = 0 # Pas de coloration significative

            fig_dend = ff.create_dendrogram(
                X, # Utilise les données originales (traits)
                orientation="left",
                labels=pdf["Espece_User"].tolist(), # Labels correspondants à X
                linkagefun=lambda _: Z, # Fournir la matrice de linkage précalculée
                color_threshold=dynamic_color_threshold if n_clusters_selected > 1 else 0,
                colorscale=color_sequence # Utiliser la même palette que PCA
            )
            fig_dend.update_layout(
                template="plotly_dark",
                height=max(600, sub.shape[0] * 22), # Hauteur dynamique
                title_text="Dendrogramme Hiérarchique des Espèces",
                title_x=0.5
            )
        else:
            fig_dend = None
            if X.shape[0] <= 1:
                print("Pas assez de données pour générer un dendrogramme (moins de 2 espèces).")
            elif X.shape[0] != len(pdf["Espece_User"]):
                 print("Incohérence de taille entre les données X et les labels pour le dendrogramme.")


        # Calcul de l'importance des variables (communalités)
        # S'assurer que pca.components_ et pca.explained_variance_ sont disponibles
        if hasattr(pca, 'components_') and hasattr(pca, 'explained_variance_'):
            loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
            communal = (loadings**2).sum(axis=1)
            
            # S'assurer que X.columns existe et correspond aux communalités
            if hasattr(X, 'columns') and len(X.columns) == len(communal):
                vip_data_df = pd.DataFrame({
                    "Variable (Trait)": X.columns, # Utilise les noms de colonnes de X
                    "Communalité (%)": (communal * 100).round(0).astype(int),
                }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
                
                vip_styled = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)'])\
                                            .format({"Communalité (%)": "{:}%"})\
                                            .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
            else:
                st.warning("Impossible de calculer l'importance des variables : les noms des traits ne sont pas disponibles ou ne correspondent pas.")
                vip_styled = None
        else:
            st.warning("Impossible de calculer l'importance des variables : informations PCA manquantes.")
            vip_styled = None


        # Préparation des données pour la composition des clusters
        cluster_compositions_data = []
        for c_label in sorted(pdf["Cluster"].unique()):
            # Utiliser Espece_User pour l'affichage
            esp_user_names = sorted(list(pdf.loc[pdf["Cluster"] == c_label, "Espece_User"].unique()))
            cluster_compositions_data.append({
                "cluster_label": c_label,
                "count": len(esp_user_names),
                "species_list": esp_user_names
            })

    except ValueError as ve: # Erreurs attendues de core.analyse ou pre-check
        st.error(f"Erreur de configuration ou de données pour l'analyse : {ve}")
        st.stop()
    except Exception as e: # pylint: disable=broad-except
        st.error(f"Une erreur inattendue est survenue lors de l'analyse : {e}")
        st.exception(e) 
        st.stop()

# ---------------------------------------------------------------------------- #
# AFFICHAGE DES RESULTATS
# ---------------------------------------------------------------------------- #

# Affichage du graphique PCA
with col_pca_plot_container:
    st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
    st.subheader("Visualisation PCA des Clusters")
    if fig_pca:
        st.plotly_chart(fig_pca, use_container_width=True)
    elif run and ref.empty: # Cas où les données n'ont pas pu être chargées
        st.warning("Les données de référence n'ont pas pu être chargées. Le graphique PCA ne peut être affiché.")
    elif run and not species_binom_user_unique and not ref.empty : # Cas où aucune espèce n'est saisie
         pass # L'erreur est déjà gérée plus haut
    elif run: # Cas où l'analyse a échoué pour une autre raison avant la création de fig_pca
        st.info("Le graphique PCA sera affiché ici après une analyse réussie et si les conditions sont remplies.")
    else: # Avant le premier 'run'
        st.info("Configurez les paramètres à gauche et cliquez sur 'Lancer l'analyse' pour visualiser la PCA.")
    st.markdown("</div>", unsafe_allow_html=True)


# Affichage de l'importance des variables et de la composition des clusters sur deux colonnes
col_vars_container, col_cluster_comp_container = st.columns([1, 2])

with col_vars_container:
    st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
    st.subheader("Importance des Variables")
    if vip_styled is not None:
        st.write(vip_styled.to_html(escape=False), unsafe_allow_html=True)
    elif run: # Si 'run' a été cliqué mais vip_styled n'est pas prêt
        st.info("Le tableau d'importance des variables (communalités) sera affiché ici si l'analyse réussit.")
    else:
        st.info("Les informations sur l'importance des variables apparaîtront après l'analyse.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_cluster_comp_container:
    st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
    st.subheader("Composition des Clusters")
    if cluster_compositions_data:
        num_clusters_found = len(cluster_compositions_data)
        if num_clusters_found > 0:
            # Dynamiquement créer des colonnes pour chaque cluster
            # Limiter le nombre de colonnes pour éviter un affichage trop chargé si beaucoup de clusters
            max_cols_display = 4 
            cluster_display_cols = st.columns(min(num_clusters_found, max_cols_display))
            
            for i, comp_data in enumerate(cluster_compositions_data):
                with cluster_display_cols[i % max_cols_display]:
                    st.markdown(f"**Cluster {comp_data['cluster_label']}** ({comp_data['count']} espèce(s))")
                    # Afficher les espèces avec des puces, potentiellement dans un expander si la liste est longue
                    if len(comp_data['species_list']) > 10:
                        with st.expander("Voir les espèces", expanded=False):
                            for species_name in comp_data['species_list']:
                                st.markdown(f"- {species_name}")
                    else:
                        for species_name in comp_data['species_list']:
                            st.markdown(f"- {species_name}")
                    if i % max_cols_display == max_cols_display -1 and i < num_clusters_found -1 :
                        st.markdown("---") # Séparateur si on recommence une ligne de colonnes (non visible ici car on ne le fait pas)
        else: # cluster_compositions_data existe mais est vide
            st.info("Aucun cluster n'a été formé ou les données de composition ne sont pas disponibles.")
    elif run:
        st.info("La composition des clusters sera affichée ici après une analyse réussie.")
    else:
        st.info("Les détails sur la composition des clusters apparaîtront après l'analyse.")
    st.markdown("</div>", unsafe_allow_html=True)

# Affichage du Dendrogramme (pleine largeur, sous les autres résultats)
if fig_dend:
    st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
    st.subheader("Dendrogramme Hiérarchique")
    st.plotly_chart(fig_dend, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
elif run and not ref.empty and species_binom_user_unique: # Si analyse lancée, données chargées, espèces saisies
    # Un message d'erreur ou d'info plus spécifique sur l'échec du dendrogramme peut être ici
    # S'il y a eu une erreur spécifique lors de la création du dendrogramme, elle est déjà loggée ou affichée.
    # Ce message est un fallback.
    if sub.shape[0] > 1: # Assez d'espèces mais échec quand même
         st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
         st.subheader("Dendrogramme Hiérarchique")
         st.warning("Le dendrogramme n'a pas pu être généré. Cela peut être dû à une configuration de données ou à un nombre insuffisant de points pour certaines étapes du calcul (ex: moins de 2 espèces uniques après filtrage pour le linkage, ou problème de seuil de coloration). Vérifiez les avertissements dans la console si disponibles.")
         st.markdown("</div>", unsafe_allow_html=True)
    # Si sub.shape[0] <= 1, l'erreur est déjà gérée plus haut.

# Message de pied de page ou d'information générale
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Application d'analyse de flore v6.0</p>", unsafe_allow_html=True)

