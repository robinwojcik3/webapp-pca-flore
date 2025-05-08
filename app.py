import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
from scipy.spatial import ConvexHull
import numpy as np
import textwrap

import core

# ---------------------------------------------------------------------------- #
# CONFIGURATION UI
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="PCA flore interactive", layout="wide")
st.markdown("<h1 style='text-align: center;'>Analyse interactive des clusters botaniques</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------- #
# CONSTANTES ET CHARGEMENT DE DONNÉES INITIALES
# ---------------------------------------------------------------------------- #
MIN_POINTS_FOR_HULL = 3
COLOR_SEQUENCE = px.colors.qualitative.Plotly

@st.cache_data
def load_data(file_path="data_ref.csv"):
    try:
        data = core.read_reference(file_path)
        return data
    except FileNotFoundError:
        st.error(f"ERREUR CRITIQUE: Fichier de données '{file_path}' non trouvé.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"ERREUR CRITIQUE: Chargement données depuis '{file_path}': {e}")
        return pd.DataFrame()

ref = load_data()
ref_binom_series = pd.Series(dtype='str')
if not ref.empty:
    ref_binom_series = ref["Espece"].str.split().str[:2].str.join(" ").str.lower()

# ---------------------------------------------------------------------------- #
# FONCTIONS UTILITAIRES
# ---------------------------------------------------------------------------- #
def format_ecology_for_hover(text, line_width_chars=65):
    if pd.isna(text) or text.strip() == "":
        return "Description écologique non disponible."
    return "<br>".join(textwrap.wrap(text, width=line_width_chars))

@st.cache_data
def load_ecology_data(file_path="data_ecologie_espece.csv"):
    try:
        eco_data = pd.read_csv(
            file_path, sep=';', header=None, usecols=[0, 1],
            names=['Espece', 'Description_Ecologie'], encoding='utf-8-sig'
        )
        eco_data['Espece_norm'] = eco_data['Espece'].astype(str).str.strip().str.split().str[:2].str.join(" ").str.lower()
        eco_data = eco_data.drop_duplicates(subset=['Espece_norm'], keep='first').set_index('Espece_norm')
        return eco_data[["Description_Ecologie"]]
    except FileNotFoundError:
        print(f"AVERTISSEMENT: Fichier écologique '{file_path}' non trouvé.")
        st.toast(f"Fichier écologique '{file_path}' non trouvé.", icon="⚠️")
        return pd.DataFrame()
    except Exception as e:
        print(f"AVERTISSEMENT: Erreur chargement données écologiques '{file_path}': {e}")
        st.toast(f"Erreur chargement fichier écologique.", icon="🔥")
        return pd.DataFrame()

ecology_df = load_ecology_data()

def apply_jitter_to_plot_data(df, x_col, y_col, jitter_strength_factor=0.015):
    """
    Applique un décalage (jitter) aux points superposés.
    Les points sont disposés en cercle autour de la position originale.
    Le rayon du jitter est proportionnel à l'étendue des données et au nombre de points superposés.
    """
    df_jittered = df.copy()
    # Colonnes pour les coordonnées d'affichage (initialisées avec les originales)
    df_jittered['x_display'] = df_jittered[x_col].astype(float)
    df_jittered['y_display'] = df_jittered[y_col].astype(float)

    # Identifier les groupes de coordonnées dupliquées
    coord_counts = df_jittered.groupby([x_col, y_col]).size().reset_index(name='counts')
    points_to_jitter = coord_counts[coord_counts['counts'] > 1]

    if points_to_jitter.empty:
        return df_jittered

    # Calculer l'étendue des données pour normaliser le rayon du jitter
    # Eviter division par zéro si toutes les valeurs sont identiques sur un axe
    x_range = df_jittered[x_col].max() - df_jittered[x_col].min()
    y_range = df_jittered[y_col].max() - df_jittered[y_col].min()

    # Définir un rayon de base pour le jitter.
    # Si l'étendue est nulle, utiliser une petite valeur absolue (ex: 0.1 ou jitter_strength_factor lui-même).
    # Sinon, proportionnel à l'étendue.
    base_radius_x = x_range * jitter_strength_factor if x_range > 1e-9 else jitter_strength_factor
    base_radius_y = y_range * jitter_strength_factor if y_range > 1e-9 else jitter_strength_factor
    if base_radius_x == 0: base_radius_x = 0.05 # Valeur minimale si étendue nulle et facteur aussi (improbable)
    if base_radius_y == 0: base_radius_y = 0.05


    for _, row in points_to_jitter.iterrows():
        original_x = row[x_col]
        original_y = row[y_col]
        n_duplicates = row['counts']

        # Indices des points appartenant à ce groupe de superposition
        group_indices = df_jittered[
            (df_jittered[x_col] == original_x) & (df_jittered[y_col] == original_y)
        ].index

        # Ajuster le rayon pour que les points sur le cercle ne se superposent pas trop
        # Augmente légèrement avec le nombre de points. sqrt(n) est une heuristique.
        # Le but est de rendre le rayon légèrement plus grand si plus de points sont impliqués.
        effective_radius_x = base_radius_x * (1 + (np.sqrt(n_duplicates) -1 ) * 0.5) # Multiplicateur ad-hoc
        effective_radius_y = base_radius_y * (1 + (np.sqrt(n_duplicates) -1 ) * 0.5)


        for i, idx in enumerate(group_indices):
            if n_duplicates == 1: # Ne devrait pas arriver ici car on filtre counts > 1
                continue
            angle = 2 * np.pi * i / n_duplicates
            
            offset_x = effective_radius_x * np.cos(angle)
            offset_y = effective_radius_y * np.sin(angle)
            
            df_jittered.loc[idx, 'x_display'] = original_x + offset_x
            df_jittered.loc[idx, 'y_display'] = original_y + offset_y
            
    return df_jittered

# ---------------------------------------------------------------------------- #
# INITIALISATION ETATS DE SESSION
# ---------------------------------------------------------------------------- #
for key in ['x_axis_trait_interactive', 'y_axis_trait_interactive']:
    if key not in st.session_state: st.session_state[key] = None
if 'run_main_analysis_once' not in st.session_state: st.session_state.run_main_analysis_once = False

# ---------------------------------------------------------------------------- #
# LAYOUT & INPUTS
# ---------------------------------------------------------------------------- #
col_input, col_pca_plot_container = st.columns([1, 3])
with col_input:
    st.subheader("CORTEGE FLORISTIQUE")
    n_clusters_selected = st.slider("Nombre de clusters (ACP)", 2, 8, 3, key="n_clusters_slider", disabled=ref.empty)
    species_txt = st.text_area(
        "Liste d'espèces (une par ligne)", height=250,
        placeholder="Teucrium chamaedrys\nPotentilla verna\n...", disabled=ref.empty
    )
    species_raw_unique = sorted(list(set(s.strip() for s in species_txt.splitlines() if s.strip())))
    species_binom_user_unique = [" ".join(s.split()[:2]).lower() for s in species_raw_unique]
    run_main_analysis_button = st.button("Lancer l'analyse principale", type="primary", disabled=ref.empty)

fig_pca, fig_dend, vip_styled, cluster_compositions_data = None, None, None, []
sub, pdf, vip_data_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
X_for_dendro = np.array([])

# ---------------------------------------------------------------------------- #
# ANALYSE PRINCIPALE (CALCULS)
# ---------------------------------------------------------------------------- #
if run_main_analysis_button and not ref.empty:
    st.session_state.run_main_analysis_once = True
    if not species_binom_user_unique:
        st.error("Veuillez saisir au moins un nom d'espèce.")
        st.stop()

    indices_to_keep = [idx for val in species_binom_user_unique if (matches := ref_binom_series[ref_binom_series == val].index).any() for idx in [matches[0]]]
    st.session_state.sub = ref.loc[indices_to_keep].copy() if indices_to_keep else pd.DataFrame(columns=ref.columns)
    sub = st.session_state.sub

    if not sub.empty:
        found_in_sub_norm = sub["Espece"].str.split().str[:2].str.join(" ").str.lower().tolist()
        not_found_raw = [species_raw_unique[i] for i, name_norm in enumerate(species_binom_user_unique) if name_norm not in found_in_sub_norm]
        if not_found_raw:
            with col_input: st.warning(f"Non trouvées: {', '.join(not_found_raw)}", icon="⚠️")
    
    if sub.empty:
        st.error("Aucune espèce saisie n'a pu être utilisée.")
        st.session_state.run_main_analysis_once = False; st.stop()
    if sub.shape[0] < max(2, n_clusters_selected if n_clusters_selected > 0 else 2):
        st.error(f"Pas assez d'espèces ({sub.shape[0]}) pour le nombre de clusters ({n_clusters_selected}) ou pour l'analyse (min 2).")
        st.session_state.run_main_analysis_once = False; st.stop()

    user_input_binom_to_raw_map = {" ".join(s.split()[:2]).lower(): s for s in species_raw_unique}

    try:
        labels, pca_obj, coords, X = core.analyse(sub, n_clusters_selected)
        st.session_state.pdf = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
        st.session_state.pdf["Cluster"] = labels.astype(str)
        st.session_state.pdf["Espece_Ref"] = sub["Espece"].values
        st.session_state.pdf["Espece_User"] = st.session_state.pdf["Espece_Ref"].apply(lambda name: user_input_binom_to_raw_map.get(" ".join(name.split()[:2]).lower(), name))
        
        if not ecology_df.empty:
            st.session_state.pdf['Espece_Ref_norm_eco'] = st.session_state.pdf['Espece_Ref'].astype(str).str.strip().str.split().str[:2].str.join(" ").str.lower()
            st.session_state.pdf['Ecologie_raw'] = st.session_state.pdf['Espece_Ref_norm_eco'].map(ecology_df['Description_Ecologie'])
            st.session_state.pdf['Ecologie'] = st.session_state.pdf['Ecologie_raw'].apply(format_ecology_for_hover)
        else:
            st.session_state.pdf['Ecologie'] = format_ecology_for_hover(None)
        pdf = st.session_state.pdf

        loadings = pca_obj.components_.T * (np.sqrt(pca_obj.explained_variance_))
        communal = (loadings**2).sum(axis=1)
        trait_cols = [c for c in sub.columns if c.lower() != "espece"]
        st.session_state.vip_data_df = pd.DataFrame({
            "Variable": trait_cols, "Communalité (%)": (communal * 100).round(0).astype(int)
        }).sort_values("Communalité (%)", ascending=False).reset_index(drop=True)
        vip_data_df = st.session_state.vip_data_df
        st.session_state.X_for_dendro = X

        num_traits = [c for c in trait_cols if pd.api.types.is_numeric_dtype(sub[c])]
        dx_init, dy_init = None, None
        if not vip_data_df.empty and len(num_traits) >= 1:
            top_vips = [v for v in vip_data_df["Variable"].tolist() if v in num_traits]
            if top_vips: dx_init = top_vips[0]
            if len(top_vips) >= 2: dy_init = top_vips[1]
            elif dx_init and len(num_traits) > 1: # VIP X, non-VIP Y
                dy_init = next((t for t in num_traits if t != dx_init), None)
        
        if dx_init is None and num_traits: dx_init = num_traits[0]
        if dy_init is None and dx_init: dy_init = next((t for t in num_traits if t != dx_init), dx_init) # Fallback to X if only one
        
        st.session_state.x_axis_trait_interactive = dx_init
        st.session_state.y_axis_trait_interactive = dy_init

    except Exception as e:
        st.error(f"Erreur analyse ACP: {e}"); st.exception(e)
        st.session_state.run_main_analysis_once = False; st.stop()

if st.session_state.run_main_analysis_once: # Retrieve from session state if already run
    sub = st.session_state.get('sub', sub)
    pdf = st.session_state.get('pdf', pdf)
    vip_data_df = st.session_state.get('vip_data_df', vip_data_df)
    X_for_dendro = st.session_state.get('X_for_dendro', X_for_dendro)

    if not pdf.empty: # Build figures if pdf data is available
        if "Cluster" not in pdf.columns: pdf["Cluster"] = "0"
        fig_pca = px.scatter(pdf, x="PC1", y="PC2" if "PC2" in pdf.columns else None, color="Cluster", 
                             text="Espece_User", hover_name="Espece_User", custom_data=["Espece_User", "Ecologie"],
                             template="plotly_dark", height=600, color_discrete_sequence=COLOR_SEQUENCE)
        fig_pca.update_traces(textposition="top center", marker_opacity=0.7, hovertemplate="<b>%{customdata[0]}</b><br><br><i>Écologie:</i><br>%{customdata[1]}<extra></extra>")
        
        unique_clusters = sorted(pdf["Cluster"].unique())
        cmap = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters)}
        if "PC2" in pdf.columns:
            for lbl in unique_clusters:
                pts_df = pdf[pdf["Cluster"] == lbl]
                if "PC1" in pts_df.columns and "PC2" in pts_df.columns:
                    unique_pts = pts_df[["PC1", "PC2"]].drop_duplicates().values
                    if len(unique_pts) >= MIN_POINTS_FOR_HULL:
                        try:
                            hull = ConvexHull(unique_pts)
                            h_pts = unique_pts[hull.vertices]
                            px_h, py_h = np.append(h_pts[:,0],h_pts[0,0]), np.append(h_pts[:,1],h_pts[0,1])
                            clr = cmap.get(lbl, COLOR_SEQUENCE[0])
                            fig_pca.add_trace(go.Scatter(x=px_h, y=py_h, fill="toself", fillcolor=clr, 
                                                         line_color=clr, line_width=1.5, mode='lines', opacity=0.2, 
                                                         showlegend=False, hoverinfo='skip'))
                        except Exception as e: print(f"Hull ACP {lbl}: {e}")
        fig_pca.update_layout(title_text="Clusters d'espèces (ACP)", title_x=0.5, legend_title_text='Cluster')

        if not vip_data_df.empty:
            vip_styled = vip_data_df.style.set_properties(**{'text-align': 'center'}, subset=['Communalité (%)']).format({"Communalité (%)": "{:}%"})
        
        cluster_compositions_data = [{"label":lbl, "count":len(pdf[pdf.Cluster==lbl]), "species":sorted(pdf[pdf.Cluster==lbl].Espece_User.unique())} for lbl in unique_clusters]

        if X_for_dendro.shape[0] > 1:
            Z = linkage(X_for_dendro, method="ward")
            d_thresh = 0
            if n_clusters_selected > 1 and (n_clusters_selected - 1) <= Z.shape[0]:
                idx = -(n_clusters_selected - 1)
                if Z.shape[0] > 0 : d_thresh = Z[idx, 2] * 0.99 if (idx + Z.shape[0] >=0) else Z[0,2]/2
            fig_dend = ff.create_dendrogram(X_for_dendro, orientation="left", labels=pdf.Espece_User.tolist(), linkagefun=lambda _: Z,
                                            color_threshold=d_thresh, colorscale=COLOR_SEQUENCE)
            fig_dend.update_layout(template="plotly_dark", height=max(650, sub.shape[0] * 20), title_text="Dendrogramme", title_x=0.5)

# ---------------------------------------------------------------------------- #
# SECTION 1: AFFICHAGE GRAPHIQUE ACP
# ---------------------------------------------------------------------------- #
with col_pca_plot_container:
    if fig_pca: st.plotly_chart(fig_pca, use_container_width=True)
    elif run_main_analysis_button and ref.empty: st.warning("Chargez les données de traits pour l'ACP.")
    elif run_main_analysis_button and sub.empty : st.warning("Aucune espèce valide pour l'ACP.")
    elif st.session_state.run_main_analysis_once and not fig_pca: st.info("Graphique ACP ici après analyse.")
    elif not st.session_state.run_main_analysis_once and not ref.empty: st.info("Prêt à lancer l'analyse.")

if not st.session_state.run_main_analysis_once and ref.empty:
    with col_input: st.warning("Données de référence non chargées. Vérifiez 'data_ref.csv'.")

# ---------------------------------------------------------------------------- #
# SECTION 2: EXPLORATION INTERACTIVE DES VARIABLES
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty:
    st.markdown("---")
    st.subheader("🔬 Exploration interactive des variables")
    num_traits = [c for c in sub.columns if c.lower() != "espece" and pd.api.types.is_numeric_dtype(sub[c])]

    if len(num_traits) >= 1: # Modifié pour permettre l'affichage même avec 1 seul trait numérique (graphique 1D)
        st.markdown("##### Sélectionnez les variables pour les axes :")
        # S'assurer que les valeurs par défaut sont valides
        dx_int = st.session_state.get('x_axis_trait_interactive', num_traits[0] if num_traits else None)
        dy_int = st.session_state.get('y_axis_trait_interactive', (num_traits[1] if len(num_traits) > 1 else dx_int) if num_traits else None)
        if dx_int not in num_traits and num_traits: dx_int = num_traits[0]
        if dy_int not in num_traits and num_traits: dy_int = num_traits[1] if len(num_traits)>1 else dx_int


        csx, csy = st.columns([1,1])
        with csx: x_sel = st.radio("Axe X:", num_traits, index=num_traits.index(dx_int) if dx_int in num_traits else 0, key="iax")
        # Pour l'axe Y, si moins de 2 traits numériques, désactiver ou gérer l'affichage 1D
        if len(num_traits) >= 2:
            with csy: y_sel = st.radio("Axe Y:", num_traits, index=num_traits.index(dy_int) if dy_int in num_traits else (1 if len(num_traits)>1 else 0), key="iay")
        else:
            y_sel = x_sel # Pour un affichage 1D (y=x) si un seul trait
            with csy: st.markdown(f"*(Axe Y identique à Axe X: {y_sel})*")


        st.session_state.x_axis_trait_interactive, st.session_state.y_axis_trait_interactive = x_sel, y_sel

        if x_sel and y_sel and not pdf.empty and len(sub) == len(pdf):
            plot_data_raw = pd.DataFrame({
                'Espece_User': pdf['Espece_User'], 'Ecologie': pdf['Ecologie'],
                x_sel: sub[x_sel], y_sel: sub[y_sel], 'Cluster': pdf['Cluster']
            })
            
            # Appliquer le Jittering
            plot_data_display = apply_jitter_to_plot_data(plot_data_raw, x_sel, y_sel, jitter_strength_factor=0.015)

            fig_int = px.scatter(
                plot_data_display, x='x_display', y='y_display', # Utiliser les coordonnées jittered pour l'affichage
                color="Cluster", text="Espece_User", hover_name="Espece_User",
                # custom_data utilise les valeurs originales pour l'infobulle
                custom_data=["Espece_User", "Ecologie", x_sel, y_sel], 
                template="plotly_dark", height=600, color_discrete_sequence=COLOR_SEQUENCE
            )
            fig_int.update_traces(textposition="top center", marker_opacity=0.8, marker_size=8,
                                  hovertemplate=("<b>%{customdata[0]}</b><br>" +
                                                 f"<br><i>{x_sel} (original):</i> %{{customdata[2]}}<br>" +
                                                 f"<i>{y_sel} (original):</i> %{{customdata[3]}}<br>" +
                                                 "<br><i>Écologie:</i><br>%{customdata[1]}<extra></extra>"))
            
            unique_clusters_int = sorted(plot_data_display["Cluster"].unique())
            cmap_int = {lbl: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, lbl in enumerate(unique_clusters_int)}
            for lbl in unique_clusters_int:
                pts_df_int = plot_data_display[plot_data_display["Cluster"] == lbl]
                # Pour l'enveloppe, utiliser les coordonnées jittered x_display, y_display
                if 'x_display' in pts_df_int.columns and 'y_display' in pts_df_int.columns:
                    unique_pts_int = pts_df_int[['x_display', 'y_display']].drop_duplicates().values
                    if len(unique_pts_int) >= MIN_POINTS_FOR_HULL:
                        try:
                            hull_int = ConvexHull(unique_pts_int)
                            h_pts_int = unique_pts_int[hull_int.vertices]
                            px_h_int,py_h_int = np.append(h_pts_int[:,0],h_pts_int[0,0]),np.append(h_pts_int[:,1],h_pts_int[0,1])
                            clr_i = cmap_int.get(lbl, COLOR_SEQUENCE[0])
                            fig_int.add_trace(go.Scatter(x=px_h_int, y=py_h_int, fill="toself", fillcolor=clr_i,
                                                         line_color=clr_i, line_width=1.5, mode='lines', opacity=0.2,
                                                         showlegend=False, hoverinfo='skip'))
                        except Exception as e: print(f"Hull Int. {lbl}: {e}")
            
            fig_int.update_layout(title_text=f"Variables: {y_sel} vs {x_sel}", title_x=0.5, xaxis_title=x_sel, yaxis_title=y_sel)
            st.plotly_chart(fig_int, use_container_width=True)

        elif not(x_sel and y_sel): st.warning("Sélectionnez variables X et Y.")
        elif pdf.empty or len(sub)!=len(pdf): st.warning("Données pour graphique interactif non prêtes.")
    else: st.warning("Pas assez de traits numériques (min 1) pour l'exploration interactive.")

# ---------------------------------------------------------------------------- #
# SECTION 3: IMPORTANCE VARIABLES & COMPOSITION CLUSTERS
# ---------------------------------------------------------------------------- #
if st.session_state.run_main_analysis_once and not sub.empty:
    cvm, cccm = st.columns([1, 2])
    with cvm:
        st.subheader("Importance Variables (ACP)")
        if vip_styled is not None: st.write(vip_styled.to_html(escape=False), unsafe_allow_html=True)
        else: st.info("Tableau VIP ici.")
    with cccm:
        st.subheader("Composition Clusters (ACP)")
        if cluster_compositions_data:
            ncf = len(cluster_compositions_data)
            if ncf > 0:
                ndc = min(ncf, 3)
                ccols = st.columns(ndc)
                for i, cd in enumerate(cluster_compositions_data):
                    with ccols[i % ndc]:
                        st.markdown(f"**Cluster {cd['label']}** ({cd['count']} espèces)")
                        for sn in cd['species']: st.markdown(f"- {sn}")
                        if i//ndc < (ncf-1)//ndc and (i+1)%ndc == 0: st.markdown("---")
            else: st.info("Aucun cluster ACP.")
        else: st.info("Composition clusters ici.")

# ---------------------------------------------------------------------------- #
# SECTION 4: DENDROGRAMME
# ---------------------------------------------------------------------------- #
if fig_dend: st.plotly_chart(fig_dend, use_container_width=True)
elif st.session_state.run_main_analysis_once and not sub.empty: st.info("Dendrogramme non généré (conditions non remplies).")
