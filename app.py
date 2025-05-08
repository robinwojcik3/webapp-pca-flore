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
            # Simule un nombre variable de traits basé sur l'entrée, au lieu de 10 fixes
            # Ceci est un placeholder, car X.shape[1] n'est pas connu ici.
            # Dans la vraie fonction core.analyse, X_data.shape[1] serait utilisé.
            # Pour la simulation, on peut supposer un nombre max de traits, ex: 10.
            num_simulated_traits = 10 
            self.components_ = np.random.rand(num_simulated_traits, n_components) 
            self.explained_variance_ = np.random.rand(n_components)

        def fit_transform(self, X):
            # S'assurer que n_components ne dépasse pas le nombre de features de X
            effective_n_components = min(self.n_components, X.shape[1])
            if effective_n_components == 0 and X.shape[1] > 0 : # Si X a des features mais n_components est 0
                 effective_n_components = 1 # Forcer au moins 1 composant si possible
            elif X.shape[1] == 0: # Si X n'a pas de features
                # Retourner un array avec 0 colonnes mais le bon nombre de lignes
                return np.empty((X.shape[0], 0))


            # Mettre à jour components_ et explained_variance_ si effective_n_components a changé
            if self.components_.shape[1] != effective_n_components:
                num_simulated_traits = self.components_.shape[0]
                self.components_ = np.random.rand(num_simulated_traits, effective_n_components)
                self.explained_variance_ = np.random.rand(effective_n_components)
            
            if effective_n_components == 0: # Si toujours 0 (ex: X.shape[1] était 0)
                 return np.empty((X.shape[0], 0))


            return np.random.rand(X.shape[0], effective_n_components)

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
            # Sélectionner uniquement les colonnes numériques pour l'analyse
            X_data = sub_df.select_dtypes(include=np.number)

            if X_data.empty:
                raise ValueError("Aucune donnée numérique trouvée pour l'analyse PCA après sélection des types.")
            
            if X_data.shape[1] == 0: # Vérification spécifique si aucune colonne numérique n'est trouvée
                 raise ValueError("Le DataFrame 'sub' ne contient aucune colonne numérique pour l'analyse PCA.")

            if X_data.shape[0] < 2: # Pas assez d'échantillons
                raise ValueError(f"Au moins 2 espèces sont nécessaires pour l'analyse PCA. {X_data.shape[0]} trouvée(s).")

            # Déterminer le nombre de composants pour la PCA
            # n_components ne peut pas être plus grand que le nombre d'échantillons ou le nombre de features
            n_components_pca = min(2, X_data.shape[1], X_data.shape[0]) 
            if n_components_pca == 0 and X_data.shape[1] > 0: # Si X_data a des features, mais min(2, features, samples) est 0 (ex: 1 feature, 1 sample)
                n_components_pca = 1 # Forcer à 1 si possible
            elif X_data.shape[1] == 0:
                 raise ValueError("Impossible de déterminer le nombre de composants pour la PCA car il n'y a pas de traits numériques.")


            pca_model = MockPCA(n_components=n_components_pca)
            coords_data = pca_model.fit_transform(X_data)
            
            # Si coords_data revient avec 0 colonnes (ex: n_components_pca était 0 ou X_data n'avait pas de features)
            if coords_data.shape[1] == 0 and n_components_pca > 0 :
                 # Cela indique un problème dans la simulation de PCA si n_components_pca était > 0
                 # ou que X_data n'avait pas de features pour commencer.
                 # Si X_data avait des features, mais coords_data n'en a pas, c'est une erreur de logique dans MockPCA.
                 # Pour ce mock, on va supposer que si n_components_pca > 0, coords_data aura des colonnes.
                 # Si X_data.shape[1] == 0, alors n_components_pca devrait être 0 et coords_data.shape[1] aussi.
                 pass # On laisse coords_data tel quel, la suite du code (ex: création de pdf) devrait gérer.


            # Simuler les labels de cluster
            if X_data.shape[0] == 0: 
                labels_data = np.array([])
            elif n_clusters > X_data.shape[0]: # Plus de clusters que de points
                # Assigner chaque point à son propre cluster si n_clusters > n_points
                # ou assigner au cluster 0, ..., n_points-1
                labels_data = np.arange(X_data.shape[0]) 
            else: # Cas standard
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
    background-color: rgba(40, 40, 42, 0.5); /* Fond subtil */
}
.styled-container .stDataFrame, .styled-container .stPlotlyChart {
    margin-bottom: 0 !important; /* Éviter double marge pour éléments internes */
}
/* Ajustement pour que les sous-titres dans les conteneurs stylés aient une bonne apparence */
.styled-container h3 { /* Cible les st.subheader */
    margin-top: 0;
    color: #FAFAFA; /* Couleur claire pour les sous-titres */
    margin-bottom: 1rem; /* Ajouter un peu d'espace sous le titre avant le contenu */
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
        # Vérifie si des données numériques existent OU si c'est la démo (qui a 'Trait1')
        if data.select_dtypes(include=np.number).empty and 'Trait1' not in data.columns: 
            st.error("ERREUR CRITIQUE: Les données de référence ne contiennent aucune colonne de trait numérique pour l'analyse.")
            return pd.DataFrame() # Retourne un DataFrame vide pour bloquer la suite
        return data
    except FileNotFoundError: # Géré par core.read_reference qui utilise des données démo
        # Ce cas ne devrait plus être atteint si core.read_reference simule les données.
        # Mais on le garde par sécurité.
        st.error(f"ERREUR CRITIQUE: Fichier de données '{file_path}' non trouvé et la simulation a échoué. L'application ne peut pas fonctionner.")
        return pd.DataFrame() 
    except Exception as e:
        st.error(f"ERREUR CRITIQUE: Impossible de charger les données depuis '{file_path}': {e}")
        return pd.DataFrame()

ref = load_data() 

ref_binom_series = pd.Series(dtype='str')
if not ref.empty and "Espece" in ref.columns:
    ref_binom_series = (
        ref["Espece"]
        .astype(str) 
        .str.split()
        .str[:2]
        .str.join(" ")
        .str.lower()
    )
else:
    if not ref.empty: # Si ref a été chargé mais manque 'Espece'
        st.warning("La colonne 'Espece' est manquante dans les données de référence. La normalisation des noms ne pourra pas s'effectuer.")
    # Si ref est vide, load_data a déjà affiché une erreur critique.

# ---------------------------------------------------------------------------- #
# LAYOUT DE LA PAGE
# ---------------------------------------------------------------------------- #
col_input, col_pca_plot_container = st.columns([1, 3]) 

with col_input:
    st.subheader("PARAMÈTRES D'ANALYSE")
    n_clusters_selected = st.slider("Nombre de clusters souhaités", 2, 8, 3, key="n_clusters_slider", disabled=ref.empty)
    species_txt = st.text_area(
        "Liste d'espèces (une par ligne, format 'Genre epithete')", height=250,
        placeholder="Exemple:\nTeucrium chamaedrys\nPotentilla verna\nAstragalus monspessulanus\n...",
        disabled=ref.empty
    )
    species_raw_unique = sorted(list(set(s.strip() for s in species_txt.splitlines() if s.strip())))
    species_binom_user_unique = [" ".join(s.split()[:2]).lower() for s in species_raw_unique]

    # Changement de l'icône du bouton ici
    run = st.button("🌸 Lancer l'analyse", type="primary", disabled=ref.empty, use_container_width=True)

# Initialisation des variables pour les figures et données
fig_pca = None
fig_dend = None
vip_styled = None 
cluster_compositions_data = []
sub = pd.DataFrame() # Initialiser sub pour qu'il existe même si l'analyse n'est pas lancée

# ---------------------------------------------------------------------------- #
# ANALYSE (déclenchée par le bouton)
# ---------------------------------------------------------------------------- #
if run and not ref.empty:
    if not species_binom_user_unique:
        st.error("Veuillez saisir au moins un nom d'espèce valide.")
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
        sub = ref.loc[list(set(indices_to_keep_from_ref))].copy() 
    else:
        sub = pd.DataFrame(columns=ref.columns)

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
        with col_input: 
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

    if sub.shape[0] < n_clusters_selected:
        st.error(f"Le nombre d'espèces valides trouvées ({sub.shape[0]}) est inférieur au nombre de clusters demandé ({n_clusters_selected}). Veuillez réduire le nombre de clusters ou fournir plus d'espèces correspondantes.")
        st.stop()
    
    if sub.shape[0] < 2: # Nécessaire pour PCA et dendrogramme (linkage)
        st.error(f"Au moins 2 espèces valides sont nécessaires pour l'analyse. {sub.shape[0]} espèce(s) trouvée(s) et utilisable(s).")
        st.stop()

    min_points_for_hull = 3 
    
    user_input_binom_to_raw_map = {
        " ".join(s_raw.split()[:2]).lower(): s_raw for s_raw in species_raw_unique
    }

    try:
        with st.spinner("Analyse en cours... Veuillez patienter."):
            labels, pca, coords, X = core.analyse(sub, n_clusters_selected)

        if coords.shape[1] == 0: # Si PCA n'a retourné aucun composant (ex: 1 seule feature)
            st.warning("L'analyse PCA n'a pas pu extraire de composantes (par exemple, s'il y a moins de 2 traits ou si les données sont constantes). Le graphique PCA ne peut pas être affiché. Le dendrogramme et l'importance des variables pourraient encore être disponibles si les données brutes le permettent.")
            # On ne stoppe pas, car le dendrogramme pourrait encore fonctionner sur X
            fig_pca = None # Assurer que fig_pca est None
        else:
            pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
            pdf["Cluster"] = labels.astype(str) 
            
            if "Espece" in sub.columns and len(sub.index) == len(pdf.index):
                pdf["Espece_Ref"] = sub["Espece"].values 
            else: 
                pdf["Espece_Ref"] = [f"Espèce {i+1}" for i in range(len(pdf))]

            def get_user_input_name(full_ref_name):
                binom_ref_name = " ".join(str(full_ref_name).split()[:2]).lower()
                return user_input_binom_to_raw_map.get(binom_ref_name, str(full_ref_name))

            pdf["Espece_User"] = pdf["Espece_Ref"].apply(get_user_input_name) 

            color_sequence = px.colors.qualitative.Plotly  
            
            fig_pca = px.scatter(
                pdf,
                x="PC1",
                y="PC2" if coords.shape[1] > 1 else None, 
                color="Cluster",
                text="Espece_User", 
                template="plotly_dark",
                height=650,
                color_discrete_sequence=color_sequence,
                title="Analyse en Composantes Principales (PCA) des Espèces" # Ce titre est interne à Plotly, pas le st.subheader
            )
            fig_pca.update_traces(
                textposition="top center",
                marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
                hovertemplate="<b>%{text}</b><br>Cluster: %{customdata[0]}<br>PC1: %{x:.2f}" + ("<br>PC2: %{y:.2f}" if coords.shape[1] > 1 else "") + "<extra></extra>",
                customdata=pdf[['Cluster']] 
            )
            fig_pca.update_layout(
                title_x=0.5, 
                legend_title_text='Cluster',
                xaxis_title="Axe Principal 1",
                yaxis_title="Axe Principal 2" if coords.shape[1] > 1 else None,
            )
            
            unique_clusters = sorted(pdf["Cluster"].unique())
            cluster_color_map = {cluster_label: color_sequence[i % len(color_sequence)] for i, cluster_label in enumerate(unique_clusters)}

            if coords.shape[1] > 1: 
                for cluster_label in unique_clusters:
                    cluster_points_df = pdf[pdf["Cluster"] == cluster_label]
                    unique_cluster_points = cluster_points_df[["PC1", "PC2"]].drop_duplicates().values
                    
                    if len(unique_cluster_points) >= min_points_for_hull:
                        try:
                            hull = ConvexHull(unique_cluster_points)
                            hull_points = unique_cluster_points[hull.vertices]
                            path_x = np.append(hull_points[:, 0], hull_points[0, 0]) 
                            path_y = np.append(hull_points[:, 1], hull_points[0, 1])

                            fig_pca.add_trace(go.Scatter(
                                x=path_x, y=path_y,
                                fill="toself", fillcolor=cluster_color_map[cluster_label],
                                line=dict(color=cluster_color_map[cluster_label], width=1.5),
                                mode='lines', name=f'Enveloppe Cluster {cluster_label}',
                                opacity=0.15, showlegend=False, hoverinfo='skip'
                            ))
                        except Exception as e_hull: 
                            print(f"Avertissement: Impossible de générer l'enveloppe convexe pour le cluster {cluster_label}: {e_hull}")
        
        # Dendrogramme (utilise X, les données numériques brutes de 'sub')
        if X.shape[0] > 1 and X.shape[0] == len(pdf.get("Espece_User", [])): # S'assurer que pdf et Espece_User existent si PCA a réussi
            Z = linkage(X, method="ward") 
            
            dynamic_color_threshold = 0
            if n_clusters_selected > 1 and (n_clusters_selected -1) <= Z.shape[0] :
                idx_threshold = X.shape[0] - n_clusters_selected
                if 0 <= idx_threshold < Z.shape[0]:
                    dynamic_color_threshold = Z[idx_threshold, 2] * 1.01 
                elif idx_threshold < 0 and Z.shape[0] > 0: 
                    dynamic_color_threshold = Z[0, 2] / 2 
                else: 
                    dynamic_color_threshold = 0

            fig_dend = ff.create_dendrogram(
                X, 
                orientation="left",
                labels=pdf["Espece_User"].tolist(), 
                linkagefun=lambda _: Z, 
                color_threshold=dynamic_color_threshold if n_clusters_selected > 1 else 0,
                colorscale=color_sequence 
            )
            fig_dend.update_layout(
                template="plotly_dark",
                height=max(600, sub.shape[0] * 22), 
                title_text="Dendrogramme Hiérarchique des Espèces", # Titre interne Plotly
                title_x=0.5
            )
        elif X.shape[0] <=1 : # Cas où il n'y a pas assez de données pour linkage
            fig_dend = None
            print("Pas assez de données pour générer un dendrogramme (moins de 2 espèces après filtrage pour X).")
        else: # Incohérence de taille ou pdf["Espece_User"] non dispo
            fig_dend = None
            print("Incohérence de taille entre les données X et les labels pour le dendrogramme, ou labels non disponibles.")


        # Importance des variables (utilisant le modèle PCA s'il existe et a des composantes)
        if pca is not None and hasattr(pca, 'components_') and hasattr(pca, 'explained_variance_') and pca.components_.shape[1] > 0 :
            # S'assurer que components_ n'est pas vide (nombre de PC > 0)
            if pca.components_.shape[0] > 0 and pca.components_.shape[1] > 0: # num_traits x num_pc
                loadings = pca.components_.T * (pca.explained_variance_ ** 0.5) # num_pc x num_traits -> num_traits x num_pc
                communal = (loadings**2).sum(axis=1) # Sum over PCs for each trait
                
                if hasattr(X, 'columns') and len(X.columns) == len(communal):
                    vip_data_df = pd.DataFrame({
                        "Variable (Trait)": X.columns, 
                        "Communalité (%)": (communal * 100).round(0).astype(int),
                    }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
                    
                    vip_styled = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)'])\
                                            .format({"Communalité (%)": "{:}%"})\
                                            .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                else:
                    st.warning("Impossible de calculer l'importance des variables : les noms des traits ne sont pas disponibles ou ne correspondent pas aux communalités.")
                    vip_styled = None
            else:
                st.warning("Impossible de calculer l'importance des variables : le modèle PCA n'a pas de composantes ou de variance expliquée (ex: pas assez de variance dans les données).")
                vip_styled = None
        elif fig_pca is None and run : # Si PCA a échoué à produire un graphique, il est probable que l'importance ne soit pas calculable
             st.warning("L'analyse PCA n'ayant pas produit de résultat, l'importance des variables ne peut être calculée.")
             vip_styled = None
        else: # Cas où pca n'a pas les attributs (ne devrait pas arriver si core.analyse est bien structuré)
            st.warning("Impossible de calculer l'importance des variables : informations PCA manquantes ou PCA non effectuée.")
            vip_styled = None

        # Composition des clusters (utilise pdf, donc dépendant de la réussite de la PCA pour les coords)
        if fig_pca is not None and 'pdf' in locals() and not pdf.empty: # S'assurer que pdf existe et n'est pas vide
            cluster_compositions_data = []
            for c_label in sorted(pdf["Cluster"].unique()):
                esp_user_names = sorted(list(pdf.loc[pdf["Cluster"] == c_label, "Espece_User"].unique()))
                cluster_compositions_data.append({
                    "cluster_label": c_label,
                    "count": len(esp_user_names),
                    "species_list": esp_user_names
                })
        elif fig_pca is None and run: # Si PCA a échoué
            st.info("La composition des clusters ne peut être déterminée car l'analyse PCA n'a pas abouti.")
            cluster_compositions_data = []


    except ValueError as ve: 
        st.error(f"Erreur de configuration ou de données pour l'analyse : {ve}")
        # Reset figs and data to avoid displaying stale results
        fig_pca, fig_dend, vip_styled, cluster_compositions_data = None, None, None, []
        st.stop()
    except Exception as e: 
        st.error(f"Une erreur inattendue est survenue lors de l'analyse : {e}")
        st.exception(e)  
        fig_pca, fig_dend, vip_styled, cluster_compositions_data = None, None, None, []
        st.stop()

# ---------------------------------------------------------------------------- #
# AFFICHAGE DES RESULTATS
# ---------------------------------------------------------------------------- #

with col_pca_plot_container:
    st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
    st.subheader("Visualisation PCA des Clusters") # Titre à l'intérieur du conteneur
    if fig_pca:
        st.plotly_chart(fig_pca, use_container_width=True)
    elif run and ref.empty: 
        st.warning("Les données de référence n'ont pas pu être chargées. Le graphique PCA ne peut être affiché.")
    elif run and not species_binom_user_unique and not ref.empty : 
        pass 
    elif run: # Analyse lancée, mais fig_pca n'est pas là (erreur gérée plus haut, ou PCA non applicable)
        st.info("Le graphique PCA sera affiché ici si l'analyse réussit et si les données le permettent (ex: au moins 2 composantes principales).")
    else: 
        st.info("Configurez les paramètres à gauche et cliquez sur '🌸 Lancer l'analyse' pour visualiser la PCA.")
    st.markdown("</div>", unsafe_allow_html=True)


col_vars_container, col_cluster_comp_container = st.columns([1, 2])

with col_vars_container:
    st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
    st.subheader("Importance des Variables") # Titre à l'intérieur
    if vip_styled is not None:
        st.write(vip_styled.to_html(escape=False), unsafe_allow_html=True)
    elif run and fig_pca is None and not sub.empty: # Si analyse lancée, PCA a échoué mais sub n'était pas vide
        st.info("L'importance des variables (communalités) n'a pas pu être calculée, potentiellement car l'analyse PCA n'a pas abouti.")
    elif run: 
        st.info("Le tableau d'importance des variables (communalités) sera affiché ici si l'analyse réussit.")
    else:
        st.info("Les informations sur l'importance des variables apparaîtront après l'analyse.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_cluster_comp_container:
    st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
    st.subheader("Composition des Clusters") # Titre à l'intérieur
    if cluster_compositions_data:
        num_clusters_found = len(cluster_compositions_data)
        if num_clusters_found > 0:
            max_cols_display = 4  
            cluster_display_cols = st.columns(min(num_clusters_found, max_cols_display))
            
            for i, comp_data in enumerate(cluster_compositions_data):
                with cluster_display_cols[i % max_cols_display]:
                    st.markdown(f"**Cluster {comp_data['cluster_label']}** ({comp_data['count']} espèce(s))")
                    if len(comp_data['species_list']) > 10:
                        with st.expander("Voir les espèces", expanded=False):
                            for species_name in comp_data['species_list']:
                                st.markdown(f"- {species_name}")
                    else:
                        for species_name in comp_data['species_list']:
                            st.markdown(f"- {species_name}")
                    # Pas besoin de st.markdown("---") ici car les colonnes gèrent la séparation visuelle
        else: 
            st.info("Aucun cluster n'a été formé ou les données de composition ne sont pas disponibles (cela peut arriver si la PCA a échoué).")
    elif run and fig_pca is None and not sub.empty :
         st.info("La composition des clusters ne peut être affichée car l'analyse PCA n'a pas abouti.")
    elif run:
        st.info("La composition des clusters sera affichée ici après une analyse réussie.")
    else:
        st.info("Les détails sur la composition des clusters apparaîtront après l'analyse.")
    st.markdown("</div>", unsafe_allow_html=True)


# Affichage du Dendrogramme (pleine largeur, sous les autres résultats)
# S'assurer que 'sub' est défini et non vide avant de vérifier sub.shape[0]
if 'sub' in locals() and not sub.empty and sub.shape[0] > 1:
    if fig_dend:
        st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
        st.subheader("Dendrogramme Hiérarchique") # Titre à l'intérieur
        st.plotly_chart(fig_dend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    elif run and species_binom_user_unique: # Analyse lancée, espèces saisies, mais fig_dend non généré
        st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
        st.subheader("Dendrogramme Hiérarchique") # Titre à l'intérieur
        st.warning("Le dendrogramme n'a pas pu être généré. Cela peut être dû à une configuration de données (ex: traits non numériques uniquement, données constantes) ou à un nombre insuffisant d'espèces uniques après filtrage pour le calcul du linkage (besoin d'au moins 2).")
        st.markdown("</div>", unsafe_allow_html=True)
elif run and 'sub' in locals() and not sub.empty and sub.shape[0] <= 1: # Moins de 2 espèces pour le dendro
    st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
    st.subheader("Dendrogramme Hiérarchique") # Titre à l'intérieur
    st.info("Le dendrogramme ne peut pas être généré avec moins de deux espèces.")
    st.markdown("</div>", unsafe_allow_html=True)
elif not run : # Avant le premier 'run', afficher un placeholder pour le dendrogramme
    # Optionnel: afficher un placeholder avant le premier run si souhaité
    # st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
    # st.subheader("Dendrogramme Hiérarchique")
    # st.info("Le dendrogramme sera affiché ici après une analyse réussie.")
    # st.markdown("</div>", unsafe_allow_html=True)
    pass # Ou ne rien afficher pour le dendrogramme avant le premier 'run'


st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Application d'analyse de flore v7.0</p>", unsafe_allow_html=True)
