"""
Marches aléatoires en 2D : libre, avec parois réfléchissantes, avec parois absorbantes.

N particules font des pas de longueur fixe dans une direction aléatoire.
Dans les cas confinés, une force de rappel harmonique les attire vers le
centre d'une boîte carrée. On compare le déplacement quadratique moyen
(DQM), l'exposant de diffusion, le coefficient de diffusion, les énergies
et, pour les parois absorbantes, le nombre de particules restantes.

Projet L3 Physique, CY Cergy Paris Université (février 2025).

Utilisation :
    python marches_aleatoires.py            # résultats, figures et animation
    python marches_aleatoires.py --sauver   # enregistre les figures et l'animation dans figures/
"""

import os
import sys

import numpy as np  # Bibliothèque pour le calcul numérique
import matplotlib.pyplot as plt  # Bibliothèque pour la visualisation
from matplotlib.animation import FuncAnimation, PillowWriter  # Animation des marches
from scipy.optimize import curve_fit  # Fonction d'ajustement pour estimer des paramètres
from scipy.ndimage import gaussian_filter1d # Filtre Gaussien pour lisser les données

SAUVER = "--sauver" in sys.argv

# =====================================
# Paramètres globaux des simulations 2D
# =====================================

N = 1000  # Nombre de particules
T = 1000  # Nombre de pas de temps
delta = 1  # Longueur d'un pas
box_size = 20  # Taille de la boîte
k = 0.01  # Raideur du potentiel harmonique (force de rappel vers le centre)
m = 1 # Masse de la particule

np.random.seed(0)  # Graine fixe : résultats reproductibles d'une exécution à l'autre

def compute_DQM(positions):
    """
    Calcule le Déplacement Quadratique Moyen (DQM) :
    DQM(T) = ⟨x² + y²⟩ permettant d'évaluer la diffusion des particules.
    """
    return np.mean(np.sum((positions - positions[:, 0, None, :])**2, axis=2), axis=0)

# Début des simulations
def random_walk_free(N, T, delta):
    """
    Simule une marche aléatoire libre en 2D sans contrainte.
    Les particules se déplacent dans toutes les directions de manière aléatoire.
    """
    positions = np.zeros((N, T, 2)) # Matrice de positions initialisée à zéro

    for t in range(1, T): # Boucle temporelle
        angles = np.random.uniform(0, 2 * np.pi, N) # Angles aléatoires pour le déplacement
        dx = delta * np.cos(angles) # Déplacement en x
        dy = delta * np.sin(angles) # Déplacement en y

        # Mise à jour des positions
        positions[:, t, 0] = positions[:, t-1, 0] + dx
        positions[:, t, 1] = positions[:, t-1, 1] + dy

    return positions

def random_walk_reflecting_2D(N, T, delta, box_size, k):
    """
    Simule une marche aléatoire avec des parois réfléchissantes.
    Lorsqu'une particule atteint un bord, elle est renvoyée dans la direction opposée.
    """
    positions = np.zeros((N, T, 2))

    for t in range(1, T): # Boucle temporelle
        angles = np.random.uniform(0, 2 * np.pi, N) # Angles aléatoires pour le déplacement
        dx = delta * np.cos(angles) # Déplacement en x
        dy = delta * np.sin(angles) # Déplacement en y

        # Ajout d'une force harmonique dirigée vers (0,0)
        Fx = - (k / m) * positions[:, t-1, 0]
        Fy = - (k / m) * positions[:, t-1, 1]

        # Mise à jour des positions
        positions[:, t, 0] = positions[:, t-1, 0] + (dx + Fx )
        positions[:, t, 1] = positions[:, t-1, 1] + (dy + Fy )

        # Gestion des réflexions
        mask_x = np.abs(positions[:, t, 0]) > box_size / 2 # Détection des chocs en x
        mask_y = np.abs(positions[:, t, 1]) > box_size / 2 # Détection des chocs en y

        # Inversion de la vitesse après réflexion
        dx[mask_x] *= -1 # Réflexion en x
        dy[mask_y] *= -1 # Réflexion en y

        # Correction après réflexion
        positions[:, t, 0] = positions[:, t-1, 0] + (dx + Fx )
        positions[:, t, 1] = positions[:, t-1, 1] + (dy + Fy )

    return positions

def random_walk_absorption_2D(N, T, delta, box_size, k):
    """
    Simule une marche aléatoire avec des parois absorbantes.
    Lorsqu'une particule est absorbée, son énergie est immédiatement mise à zéro.
    """
    # Initialisation des tableaux pour stocker les positions, vitesses et énergies
    positions = np.zeros((N, T, 2))
    active = np.ones(N, dtype=bool)  # Tableau indiquant si une particule est encore active
    particles_remaining = np.zeros(T)  # Nombre de particules restantes à chaque instant
    particles_remaining[0] = N  # Toutes les particules sont actives au départ

    vx = np.zeros((N, T)) # Vitesses en x
    vy = np.zeros((N, T)) # Vitesses en y

    E_cinetique_absorb = np.zeros((N, T)) # Énergie cinétique des particules
    E_potentielle_absorb = np.zeros((N, T)) # Énergie potentielle des particules

    for t in range(1, T): # Boucle temporelle
        angles = np.random.uniform(0, 2 * np.pi, N)  # Angles aléatoires pour le déplacement
        dx = delta * np.cos(angles) * active  # Déplacement en x ( sur les particules actives )
        dy = delta * np.sin(angles) * active  # idem en y

        # Ajout d'une force harmonique
        Fx = - (k / m) * positions[:, t-1, 0] * active # Force en x ( sur les particules actives )
        Fy = - (k / m) * positions[:, t-1, 1] * active # idem en y

        # Mise à jour des positions uniquement pour les particules actives
        positions[:, t, 0] = np.where(active, positions[:, t-1, 0] + (dx + Fx ), positions[:, t-1, 0])
        positions[:, t, 1] = np.where(active, positions[:, t-1, 1] + (dy + Fy ), positions[:, t-1, 1])

        # Absorption des particules aux bords
        absorbed = (np.abs(positions[:, t, 0]) > box_size / 2) | (np.abs(positions[:, t, 1]) > box_size / 2)
        active[absorbed] = False  # Désactivation des particules absorbées

        # Mise à jour des vitesses
        vx[:, t] = np.where(active, dx + Fx, 0)  # Mise à zéro des vitesses après absorption en x
        vy[:, t] = np.where(active, dy + Fy, 0)  # idem en y

        # Calcul de l'énergie cinétique et potentielle en s'assurant que l'énergie des particules absorbées est nulle
        E_cinetique_absorb[:, t] = np.where(active, 0.5 * m *(vx[:, t]**2 + vy[:, t]**2), 0)
        E_potentielle_absorb[:, t] = np.where(active, 0.5 * k * m * (positions[:, t, 0]**2 + positions[:, t, 1]**2), 0)

        particles_remaining[t] = np.sum(active)  # Mise à jour du nombre de particules restantes

    return positions, E_cinetique_absorb, E_potentielle_absorb, particles_remaining

# ====================================
# Exécution des simulations et calculs
# ====================================

# Exécution des simulations sans DQM
positions_libre = random_walk_free(N, T, delta) # Marche libre
positions_reflect = random_walk_reflecting_2D(N, T, delta, box_size, k) # Marche réfléchissante
positions_absorb, E_cinetique_absorb, E_potentielle_absorb, particles_remaining = random_walk_absorption_2D(N, T, delta, box_size, k) # Marche absorbante

# Calcul des vitesses : différence de position entre deux instants successifs
vx_libre = np.diff(positions_libre[:, :, 0], axis=1, prepend=0) / delta # Vitesse en x pour la marche libre
vy_libre = np.diff(positions_libre[:, :, 1], axis=1, prepend=0) / delta # En y

vx_reflect = np.diff(positions_reflect[:, :, 0], axis=1, prepend=0) / delta # Vitesse en x pour la marche réfléchissante
vy_reflect = np.diff(positions_reflect[:, :, 1], axis=1, prepend=0) / delta # En y

vx_absorb = np.diff(positions_absorb[:, :, 0], axis=1, prepend=0) / delta # Vitesse en x pour la marche absorbante
vy_absorb = np.diff(positions_absorb[:, :, 1], axis=1, prepend=0) / delta # En y

# Énergie cinétique ( = 1/2 * m * v^2 )
E_cinetique_libre = 0.5 * m * (vx_libre**2 + vy_libre**2) # Marche libre
E_cinetique_reflect = 0.5 * m * (vx_reflect**2 + vy_reflect**2) # Marche réfléchissante

# Énergie potentielle ( Force qui attire vers (0,0) )
E_potentielle_libre = 0.5 * k * m * (positions_libre[:, :, 0]**2 + positions_libre[:, :, 1]**2) # Marche libre
E_potentielle_reflect = 0.5 * k * m * (positions_reflect[:, :, 0]**2 + positions_reflect[:, :, 1]**2) # Marche réfléchissante

# Énergie totale
E_totale_libre = np.mean(E_potentielle_libre + E_cinetique_libre, axis=0) # Marche libre
E_totale_reflect = np.mean(E_potentielle_reflect + E_cinetique_reflect, axis=0) # Marche réfléchissante
E_totale_absorb = np.mean(E_potentielle_absorb + E_cinetique_absorb, axis=0) # Marche absorbante

def power_law(t, alpha, C):
    """
    Modèle de loi de puissance : ⟨x²⟩ = C * t^alpha
    Utilisé pour ajuster les données simulées et estimer l'exposant alpha.
    """
    return C * t ** alpha

# Calcul du DQM après la simulation
DQM_libre = compute_DQM(positions_libre) # Marche libre
DQM_reflect = compute_DQM(positions_reflect) # Marche réfléchissante
DQM_absorb = compute_DQM(positions_absorb) # Marche absorbante

# Lissage des courbes grâce à la fonction gaussian_filter1d
DQM_libre_smooth = gaussian_filter1d(DQM_libre, sigma=1.5) # Marche libre
DQM_reflect_smooth = gaussian_filter1d(DQM_reflect, sigma=1.5) # Marche réfléchissante
DQM_absorb_smooth = gaussian_filter1d(DQM_absorb, sigma=1.5) # Marche absorbante

# Ajustement du modèle de loi de puissance pour estimer l'exposant alpha
t_values = np.arange(1, T)
popt_libre, _ = curve_fit(power_law, t_values, DQM_libre[1:], p0=[1, 1], bounds=(0, [2, np.inf])) # Marche libre
popt_reflect, _ = curve_fit(power_law, t_values, DQM_reflect[1:], p0=[1, 1], bounds=(0, [2, np.inf])) # Marche réfléchissante
popt_absorb, _ = curve_fit(power_law, t_values, DQM_absorb[1:], p0=[1, 1], bounds=(0, [2, np.inf])) # Marche absorbante

# Calcul du coefficient de diffusion D pour chaque type de marche
D_libre = (DQM_libre[-1] - DQM_libre[T//2]) / (4 * (T - T//2)) # Marche libre
D_reflect = (DQM_reflect[200] - DQM_reflect[50]) / (4 * (200 - 50)) # Marche réfléchissante
D_absorb = (DQM_absorb[-1] - DQM_absorb[T//2]) / (4 * (T - T//2)) # Marche absorbante

# =======================
# Affichage des résultats
# =======================

# Affichage des coefficients de diffusion calculés
print(f"Coefficient de diffusion (Libre) :         {D_libre:.4f}   -----> D théorique : D ≃ 0,25  ")
print(f"Coefficient de diffusion (Réfléchissant) : {D_reflect:.4f}   -----> D théorique : 0 < D < 0,25     ")
print(f"Coefficient de diffusion (Absorbant) :     {D_absorb:.4f}   -----> D théorique : 0 < D << 0,25 \n")

# Affichage des exposants de diffusion estimés par ajustement de la loi de puissance
print(f"Exposant de diffusion (α) - Libre : {popt_libre[0]:.3f}            \n-----> Doit être très proche de 1 et donc suivre la loi d'Einstein ")
print(f"Exposant de diffusion (α) - Réfléchissant : {popt_reflect[0]:.3f}    \n-----> ( dans le cas d'une petite boîte ) Très faible → Le confinement limite fortement la diffusion. ")
print(f"Exposant de diffusion (α) - Absorbant : {popt_absorb[0]:.3f}       \n-----> ( dans le cas d'une petite boîte ) Supérieur à celui de la marche réfléchissante → le DQM inclut les particules absorbées, restées figées sur les parois (voir README). ")

# Création d'une figure avec 2x3 sous-graphes ( afin d'afficher toutes les trajectoires et graphes sur la même image )
fig, axs = plt.subplots(2, 3, figsize=(15, 10)) # Configuration de la grille de subplots
fig.suptitle("Comparaison de différentes marches aléatoires en 2D", fontsize=16) # Titre

# Tracé des trajectoires de 10 particules pour chaque type de marche :
# Trajectoires des marches libres
for i in range(10):
    axs[0, 0].plot(positions_libre[i, :, 0], positions_libre[i, :, 1], alpha=0.6)
    axs[0, 0].set_title("Trajectoire de la marche aléatoire \nlibre sans contrainte")
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")

# Trajectoires des marches avec réflexion
for i in range(10):
    axs[0, 1].plot(positions_reflect[i, :, 0], positions_reflect[i, :, 1], alpha=0.6)
    axs[0, 1].set_title("Trajectoire de la marche aléatoire \navec réflexion aux bords")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")
    axs[0, 1].grid(True) # Affichage de la "grille" pour mieux visualiser la boîte ( grid(True) )

# Trajectoires des marches avec absorption
for i in range(10):
    axs[0, 2].plot(positions_absorb[i, :, 0], positions_absorb[i, :, 1], alpha=0.6)
    axs[0, 2].set_title("Trajectoire de la marche aléatoire \navec absorption aux bords")
    axs[0, 2].set_xlabel("x")
    axs[0, 2].set_ylabel("y")
    axs[0, 2].grid(True) # Affichage de la "grille" pour mieux visualiser la boîte ( True )

# Évolution du déplacement quadratique moyen (DQM) en fonction du temps
axs[1, 0].plot(t_values, DQM_libre_smooth[1:], label=f"Libre (α={popt_libre[0]:.3f})", color='g')
axs[1, 0].plot(t_values, DQM_reflect_smooth[1:], label=f"Réflexion (α={popt_reflect[0]:.3f})", color='b')
axs[1, 0].plot(t_values, DQM_absorb_smooth[1:], label=f"Absorption (α={popt_absorb[0]:.3f})", color='r')
axs[1, 0].plot(t_values, power_law(t_values, *popt_libre), 'g--')
axs[1, 0].plot(t_values, power_law(t_values, *popt_reflect), 'b--')
axs[1, 0].plot(t_values, power_law(t_values, *popt_absorb), 'r--')
axs[1, 0].plot(t_values, t_values, 'k-', label="Loi d'Einstein (α=1)")
axs[1, 0].set_title("Évolution du DQM en 2D")
axs[1, 0].set_xlabel("Temps (t)")
axs[1, 0].set_ylabel("⟨x²⟩")
axs[1, 0].legend()
axs[1, 0].grid()

# Nombre de particules restantes pour l'absorption
axs[1, 1].plot(range(T), particles_remaining, label="Particules restantes", color='r')
axs[1, 1].set_title("Évolution du nombre de particules restantes")
axs[1, 1].set_xlabel("Temps")
axs[1, 1].set_ylabel("Nombre de particules")
axs[1, 1].grid()

# Distribution finale des particules
axs[1, 2].scatter(positions_libre[:, -1, 0], positions_libre[:, -1, 1], alpha=0.3, label="libre ( sans potentiel )", color="green")
axs[1, 2].scatter(positions_reflect[:, -1, 0], positions_reflect[:, -1, 1], alpha=0.3, label="Réfléchissant", color="blue")
axs[1, 2].scatter(positions_absorb[:, -1, 0], positions_absorb[:, -1, 1], alpha=0.3, label="absorbant", color="red")
axs[1, 2].set_xlabel("x")
axs[1, 2].set_ylabel("y")
axs[1, 2].set_title("Distribution finale des particules")
axs[1, 2].legend()
axs[1, 2].grid()

fig.tight_layout()

# Création d'une figure avec 1x3 sous-graphes ( afin d'afficher toutes les Énergies sur la même image )
fig2, axs = plt.subplots(1, 3, figsize=(15, 5)) # Configuration de la grille de subplots
fig2.suptitle("Tracé des Énergies", fontsize=16) # Titre

# Énergie cinétique
axs[0].plot(range(T), np.mean(E_cinetique_libre, axis=0), alpha=0.5, label="Libre ( sans potentiel )", color='g')
axs[0].plot(range(T), np.mean(E_cinetique_reflect, axis=0), alpha=0.5, label="Réflexion", color='b')
axs[0].plot(range(T), np.mean(E_cinetique_absorb, axis=0), alpha=0.5, label="Absorption", color='r')
axs[0].set_title("Évolution de l'énergie cinétique")
axs[0].set_xlabel("Temps")
axs[0].set_ylabel("Énergie cinétique")
axs[0].legend()
axs[0].grid()

# Énergie potentielle
axs[1].plot(range(T), np.mean(E_potentielle_libre, axis=0), alpha=0.5, label="Libre ( sans potentiel )", color='g')
axs[1].plot(range(T), np.mean(E_potentielle_reflect, axis=0), alpha=0.5, label="Réflexion", color='b')
axs[1].plot(range(T), np.mean(E_potentielle_absorb, axis=0), alpha=0.5, label="Absorption", color='r')
axs[1].set_title("Évolution de l'énergie potentielle")
axs[1].set_xlabel("Temps")
axs[1].set_ylabel("Énergie potentielle")
axs[1].legend()
axs[1].grid()

# Énergie totale
axs[2].plot(range(T), E_totale_libre, alpha=0.5, label="Libre ( sans potentiel )", color='g')
axs[2].plot(range(T), E_totale_reflect, alpha=0.5, label="Réflexion", color='b')
axs[2].plot(range(T), E_totale_absorb, alpha=0.5, label="Absorption", color='r')
axs[2].set_title("Évolution de l'énergie totale")
axs[2].set_xlabel("Temps")
axs[2].set_ylabel("Énergie totale")
axs[2].legend()
axs[2].grid()
fig2.tight_layout()

# ==================================
# Animation des 1000 particules
# ==================================

def animer_marches(fichier=None, tous_les=5):
    """
    Nuage des N particules pour les trois marches, une image tous les
    `tous_les` pas. Marche libre : le cercle a pour rayon la distance
    quadratique moyenne prévue par Einstein, sqrt(<r²>) = delta * sqrt(t).
    Marche absorbante : les particules absorbées restent en gris, figées là
    où elles ont franchi la paroi.
    """
    fig3, axs3 = plt.subplots(1, 3, figsize=(15, 5.4))
    marches = [
        (positions_libre, "Marche libre", "green", 70),
        (positions_reflect, "Parois réfléchissantes + rappel", "blue", 12),
        (positions_absorb, "Parois absorbantes + rappel", "red", 12),
    ]
    nuages = []
    for ax, (pos, titre, couleur, lim) in zip(axs3, marches):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if lim == 12:  # contour de la boîte
            ax.add_patch(plt.Rectangle((-box_size / 2, -box_size / 2), box_size, box_size,
                                       fill=False, color="black", lw=1.5))
        nuages.append(ax.scatter(pos[:, 0, 0], pos[:, 0, 1], s=5, alpha=0.5, color=couleur))
    cercle = plt.Circle((0, 0), 0, fill=False, ls="--", color="black", lw=1.2)
    axs3[0].add_patch(cercle)

    def image(k):
        t = k * tous_les
        for nuage, (pos, titre, couleur, lim) in zip(nuages, marches):
            nuage.set_offsets(pos[:, t, :])
        cercle.set_radius(delta * np.sqrt(t))
        axs3[0].set_title(f"Marche libre (t = {t})\ncercle de rayon $\\sqrt{{t}}$ (loi d'Einstein)")
        axs3[1].set_title(f"Parois réfléchissantes + rappel (t = {t})")
        actives = np.all(np.abs(positions_absorb[:, t, :]) <= box_size / 2, axis=1)
        nuages[2].set_color(np.where(actives[:, None], [[1, 0, 0, 0.5]], [[0.5, 0.5, 0.5, 0.5]]))
        axs3[2].set_title(f"Parois absorbantes + rappel (t = {t})\n{actives.sum()} particules restantes")
        return nuages

    fig3.tight_layout()
    fig3.subplots_adjust(top=0.86)  # place pour les titres sur deux lignes
    anim = FuncAnimation(fig3, image, frames=T // tous_les, interval=50, blit=False)
    if fichier:
        anim.save(fichier, writer=PillowWriter(fps=20), dpi=60)
        plt.close(fig3)
        print(f"Animation enregistrée : {fichier}")
    return anim


# Affichage ou enregistrement
if SAUVER:
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/marches_aleatoires.png", dpi=80)
    fig2.savefig("figures/energies.png", dpi=80)
    print("\nFigures enregistrées dans figures/")
    animer_marches("figures/marches_aleatoires.gif")
else:
    anim = animer_marches()
    plt.show()