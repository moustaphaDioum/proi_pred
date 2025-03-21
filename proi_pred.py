import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp
import time  
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# === Chargement des images ===
lapin_img = plt.imread("lapin.png")  # Assure-toi d'avoir "rabbit.png"
renard_img = plt.imread("renard.png")    # Assure-toi d'avoir "fox.png"

# === Modèle Lotka-Volterra ===
def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# === Fonction pour exécuter la simulation ===
def run_simulation(alpha, beta, delta, gamma, x0, y0, t_max, points):
    t_span = (0, t_max)
    t_eval = np.linspace(*t_span, points)
    sol = solve_ivp(lotka_volterra, t_span, [x0, y0], args=(alpha, beta, delta, gamma), t_eval=t_eval)

    # Récupération des solutions
    t, x, y = sol.t, sol.y[0], sol.y[1]

    # Vérification et mise à jour des populations
    for i in range(len(t)):
        if x[i] < 1:  # Si les proies descendent sous 1
            x[i:] = 0  # Elles restent nulles
            y[i:] = y[i] * np.exp(-gamma * (t[i:] - t[i]))  # Décroissance exponentielle des prédateurs
            break  # On arrête la boucle car l'évolution est forcée

        if y[i] < 1:  # Si les prédateurs descendent sous 1
            y[i:] = 0  # Ils restent nuls
            x[i:] = x[i] * np.exp(alpha * (t[i:] - t[i]))  # Croissance exponentielle des proies sans prédateurs
            break  # On arrête la boucle

    return t, x, y

# === Fonction pour ajouter une image à l'affichage ===
def add_image(ax, img, x, y, zoom=0.02):
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)

# === Interface Streamlit ===
st.markdown("""
    <style>
        .stApp {
            background-color: #e0e0e0;
        }
        .stMarkdown, .stText, .stSubheader, .stTitle , .stSlider{
            color: #8e44ad;  /* Couleur de tous les textes */
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="color: #4CAF50;">Simulation Lotka-Volterra 🦊🐰</h1>', unsafe_allow_html=True)


# === Organisation en colonnes ===
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<p style="color: #8e44ad;">Modèle mathématique</p>', unsafe_allow_html=True)
    st.latex(r"""
    \begin{cases}
    \frac{dx}{dt} = \alpha x - \beta xy \\ 
    \frac{dy}{dt} = \delta xy - \gamma y
    \end{cases}
    """)

    st.markdown('<p style="color: #8e44ad;">Paramètres de simulation</p>', unsafe_allow_html=True)

    # Texte personnalisé avec couleur #8e44ad
    st.markdown('<p style="color: #8e44ad;">Taux de croissance des proies (α)</p>', unsafe_allow_html=True)
    alpha = st.slider(" ", 0.0, 1.0, 0.33, 0.05)

    # Texte personnalisé avec couleur #8e44ad
    st.markdown('<p style="color: #8e44ad;">Taux de prédation (β)</p>', unsafe_allow_html=True)
    beta = st.slider(" ", 0.0, 1.0, 0.02, 0.04)
    
    st.markdown('<p style="color: #8e44ad;">Conversion des proies en prédateurs (δ)</p>', unsafe_allow_html=True)
    delta = st.slider(" ", 0.0, 1.0, 0.02, 0.05)
    
    st.markdown('<p style="color: #8e44ad;">Mortalité des prédateurs (γ)</p>', unsafe_allow_html=True)
    gamma = st.slider(" ", 0.0, 1.0, 0.3, 0.02)
    
    st.markdown('<p style="color: #8e44ad;">Population initiale des proies</p>', unsafe_allow_html=True)
    x0 = st.number_input(" ", 0, 1000, 100)
    
    st.markdown('<p style="color: #8e44ad;">Population initiale des prédateurs</p>', unsafe_allow_html=True)
    y0 = st.number_input(" ", 0, 1000, 20)
    
    st.markdown('<p style="color: #8e44ad;">Temps de simulation</p>', unsafe_allow_html=True)
    t_max = st.slider(" ", 5, 100, 10)

   # points = st.slider("Nombre de points", 10, 100, 30)

    # Bouton pour lancer la simulation
    run_simulation_btn = st.button("Simuler 🚀")

with col2:
    if run_simulation_btn:
        with st.spinner("Simulation en cours... ⏳"):
            # Exécute la simulation
            t, x, y = run_simulation(alpha, beta, delta, gamma, x0, y0, t_max, 100)

            st.success("Simulation terminée ✅")

            # Affichage du graphique des populations (courbes)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t, x, label="Proies (Lapins)", color="blue")
            ax.plot(t, y, label="Prédateurs (Renards)", color="red")
            ax.set_xlabel("Temps")
            ax.set_ylabel("Population")
            ax.set_title("Dynamique Lotka-Volterra", color="white")
            ax.legend()
            ax.grid()
            ax.set_facecolor("black")
            st.pyplot(fig)

            # === ANIMATION AVEC IMAGES ===
            st.markdown('<p style="background-color: #8e44ad; padding: 10px; color: white; font-size: 20px;">Évolution des populations 📍</p>', unsafe_allow_html=True)

            # Création d'un espace pour l'affichage dynamique
            plot_spot = st.empty()

            for i in range(len(t)):
                # Ajustement dynamique de la taille du cadre
                max_population = max(max(x), max(y))
                lim = max(10, max_population / 5)

                fig_anim, ax_anim = plt.subplots(figsize=(10, 8))
                ax_anim.set_xlim(0, lim)
                ax_anim.set_ylim(0, lim)
                ax_anim.set_xticks([])
                ax_anim.set_yticks([])
                ax_anim.set_facecolor("white")

                # Nombre d'animaux proportionnel aux valeurs simulées
                n_lapins = max(0, round(x[i]))
                n_renards = max(0, round(y[i]))

                # Titre dynamique
                ax_anim.set_title(f"Temps: {t[i]:.1f} | Lapins: {n_lapins} | Renards: {n_renards}",
                                  fontsize=14, color="black", fontweight="bold")

                # Position aléatoire des lapins et renards
                lapin_positions = np.random.rand(n_lapins, 2) * (lim - 2) + 1
                renard_positions = np.random.rand(n_renards, 2) * (lim - 2) + 1

                # Ajouter les images des lapins
                for pos in lapin_positions:
                    add_image(ax_anim, lapin_img, pos[0], pos[1], zoom=0.05)

                # Ajouter les images des renards
                for pos in renard_positions:
                    add_image(ax_anim, renard_img, pos[0], pos[1], zoom=0.05)

                # Ajouter la légende
                #legend_labels = ["Lapins", "Renards"]
                #legend_colors = ["blue", "red"]
                #for j, label in enumerate(legend_labels):
                    #ax_anim.text(lim - 3, 1.5 - j * 0.8, f"⬤ {label}", color=legend_colors[j], fontsize=12, ha="right")

                # Affichage dans Streamlit
                plot_spot.pyplot(fig_anim)
                plt.close(fig_anim)  

                #time.sleep(0.01)  # Pause pour ralentir l'animation