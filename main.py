import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # 1. Paramètres du Problème
    T: float = 3.0         # Temps total jusqu'à l'échéance
    N: int = 9             # Nombre d'étapes de temps
    d: int = 10            # Nombre d'actifs sous-jacents
    r: float = 0.05        # Taux sans risque
    delta: float = 0.1     # Rendement des dividendes
    sigma: float = 0.2     # Volatilité des actifs
    K: float = 100.0       # Prix d'exercice (strike price)
    s_0: float = 90.0      # Prix initial de l'actif

    # 2. Paramètres d'Entraînement
    train_steps: int = 500   # Nombre d'itérations d'entraînement
    mc_runs: int = 500       # Nombre de simulations Monte Carlo pour l'évaluation
    batch_size: int = 8192    # Taille du lot pour l'entraînement et la simulation MC
    lr_boundaries: List[int] = (1000, 2000)  # Bornes pour le plan d'apprentissage
    lr_values: List[float] = (0.05, 0.005, 0.0005)  # Valeurs de taux d'apprentissage par morceaux
    optimizer_eps: float = 0.1   # Paramètre epsilon pour l'optimiseur Adam

    # 3. Configuration du Dispositif
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_piecewise_learning_rate(step: int, config: Config) -> float:
    """
    Détermine le taux d'apprentissage basé sur l'étape d'entraînement actuelle.
    
    Args:
        step (int): Étape d'entraînement actuelle.
        config (Config): Objet de configuration contenant les bornes et valeurs de taux d'apprentissage.
    
    Returns:
        float: Taux d'apprentissage approprié pour l'étape actuelle.
    """
    if step < config.lr_boundaries[0]:
        return config.lr_values[0]
    elif step < config.lr_boundaries[1]:
        return config.lr_values[1]
    else:
        return config.lr_values[2]


class StoppingNet(nn.Module):
    """
    Réseau de neurones pour approximer les décisions d'arrêt à chaque étape de temps.
    Architecture : [Linear -> BatchNorm -> ReLU] x (couches cachées) -> Linear
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [50, 50]):
        """
        Initialise le StoppingNet.
    
        Args:
            input_dim (int): Dimension des caractéristiques d'entrée.
            hidden_dims (List[int], optional): Liste contenant le nombre de neurones dans chaque couche cachée.
                                               Par défaut à [50, 50].
        """
        super(StoppingNet, self).__init__()
        layers = []
        prev_dim = input_dim
        for idx, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Couche de sortie
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du réseau.
    
        Args:
            x (torch.Tensor): Tenseur d'entrée de forme (batch_size, input_dim).
    
        Returns:
            torch.Tensor: Tenseur de sortie de forme (batch_size, 1).
        """
        return self.network(x)


class DeepOptimalStopping:
    """
    Implémente la méthode Deep Optimal Stopping pour les options Bermudan max-call.
    """

    def __init__(self, config: Config):
        """
        Initialise l'instance de DeepOptimalStopping.
    
        Args:
            config (Config): Objet de configuration contenant tous les paramètres nécessaires.
        """
        self.config = config
        self.device = config.device
        # Création d'une liste de réseaux de neurones, un par étape de temps (sauf la dernière)
        self.nets = nn.ModuleList([
            StoppingNet(input_dim=config.d + 1).to(self.device)  # d actifs + 1 payoff
            for _ in range(config.N - 1)
        ])
        # Définition de l'optimiseur Adam avec le premier taux d'apprentissage
        self.optimizer = optim.Adam(
            self.nets.parameters(),
            lr=config.lr_values[0],
            eps=config.optimizer_eps
        )
        self.loss_history = []  # Historique des pertes pour le suivi

    def simulate_asset_paths(self, batch_size: int) -> torch.Tensor:
        """
        Simule les trajectoires de prix des actifs en utilisant le modèle Black-Scholes multidimensionnel.
    
        Args:
            batch_size (int): Nombre de trajectoires simulées.
    
        Returns:
            torch.Tensor: Prix des actifs simulés de forme (batch_size, d, N).
        """
        # Génération des incréments de Wiener (mouvements browniens)
        W = torch.randn(batch_size, self.config.d, self.config.N, device=self.device) * np.sqrt(self.config.T / self.config.N)
        W = torch.cumsum(W, dim=2)  # Somme cumulative pour simuler le mouvement brownien

        # Création de la grille temporelle
        t = torch.linspace(self.config.T / self.config.N, self.config.T, self.config.N, device=self.device)
        # Calcul du drift et de la diffusion
        drift = (self.config.r - self.config.delta - 0.5 * self.config.sigma ** 2) * t
        diffusion = self.config.sigma * W

        # Calcul des prix des actifs
        X = self.config.s_0 * torch.exp(drift.view(1, 1, self.config.N) + diffusion)  # Trajectoires des actifs
        return X

    def calculate_payoffs(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calcule les payoffs actualisés pour chaque étape de temps.
    
        Args:
            X (torch.Tensor): Prix des actifs simulés de forme (batch_size, d, N).
    
        Returns:
            torch.Tensor: Payoffs actualisés de forme (batch_size, N).
        """
        # Création de la grille temporelle
        t = torch.linspace(self.config.T / self.config.N, self.config.T, self.config.N, device=self.device)
        discount_factors = torch.exp(-self.config.r * t)  # Facteurs d'actualisation

        # Calcul du payoff brut à chaque étape de temps
        max_X = torch.amax(X, dim=1)  # Prix maximum parmi les actifs à chaque étape
        payoffs = torch.maximum(max_X - self.config.K, torch.tensor(0.0, device=self.device))  # Payoff à chaque étape
        discounted_payoffs = discount_factors.view(1, self.config.N) * payoffs  # Payoffs actualisés

        return discounted_payoffs

    def train(self):
        """
        Entraîne les réseaux de neurones pour approximer la stratégie d'arrêt optimale.
        """
        self.nets.train()  # Met tous les réseaux en mode entraînement

        for step in range(self.config.train_steps):
            # 1. Mise à jour du taux d'apprentissage en fonction de l'étape actuelle
            current_lr = get_piecewise_learning_rate(step, self.config)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

            # 2. Simulation des trajectoires des actifs et calcul des payoffs
            X = self.simulate_asset_paths(self.config.batch_size)  # Forme : (batch_size, d, N)
            discounted_payoffs = self.calculate_payoffs(X)        # Forme : (batch_size, N)

            # 3. Initialisation de g_tau avec les payoffs terminaux
            g_tau = discounted_payoffs[:, -1].clone()  # Forme : (batch_size,)

            loss = 0.0  # Initialisation de la perte pour cette étape d'entraînement

            # 4. Induction arrière sur les étapes de temps
            for n in reversed(range(self.config.N - 1)):  # De N-2 jusqu'à 0
                # a. Préparation des entrées pour le réseau : prix des actifs et payoffs à l'étape n
                asset_prices_n = X[:, :, n]  # Forme : (batch_size, d)
                payoff_n = discounted_payoffs[:, n].unsqueeze(1)  # Forme : (batch_size, 1)
                network_input = torch.cat([asset_prices_n, payoff_n], dim=1)  # Forme : (batch_size, d + 1)

                # b. Passage avant à travers le réseau
                net_output = self.nets[n](network_input).squeeze(1)  # Forme : (batch_size,)
                F_n = torch.sigmoid(net_output)  # Probabilité d'arrêt

                # c. Calcul du payoff attendu
                term = discounted_payoffs[:, n] * F_n + g_tau * (1.0 - F_n)  # Forme : (batch_size,)
                loss -= torch.mean(term)  # On négative la moyenne car on veut maximiser le payoff

                # d. Mise à jour de g_tau pour l'itération suivante
                g_tau = torch.where(net_output > 0.0, discounted_payoffs[:, n], g_tau)

            # 5. Rétropropagation et mise à jour des paramètres
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 6. Journalisation des progrès
            if (step + 1) % 100 == 0 or step < 100:
                estimated_price = -loss.item()  # Interprétation directe sans mise à l'échelle
                self.loss_history.append(estimated_price)
                print(f"Étape {step + 1:4d}, Perte ~ {estimated_price:.4f}, Taux d'apprentissage : {current_lr}")

    def evaluate(self) -> float:
        """
        Évalue la stratégie d'arrêt apprise en utilisant une simulation Monte Carlo.
    
        Returns:
            float: Estimation du prix de l'option Bermudan max-call.
        """
        self.nets.eval()  # Met tous les réseaux en mode évaluation
        price_accumulator = 0.0  # Accumulateur pour le prix

        with torch.no_grad():  # Désactive le calcul des gradients pour l'efficacité
            for run in range(self.config.mc_runs):
                # 1. Simulation des trajectoires des actifs et calcul des payoffs
                X = self.simulate_asset_paths(self.config.batch_size)
                discounted_payoffs = self.calculate_payoffs(X)

                # 2. Initialisation de g_tau avec les payoffs terminaux
                g_tau = discounted_payoffs[:, -1].clone()  # Forme : (batch_size,)

                # 3. Application de la stratégie d'arrêt apprise via induction arrière
                for n in reversed(range(self.config.N - 1)):
                    asset_prices_n = X[:, :, n]  # Forme : (batch_size, d)
                    payoff_n = discounted_payoffs[:, n].unsqueeze(1)  # Forme : (batch_size, 1)
                    network_input = torch.cat([asset_prices_n, payoff_n], dim=1)  # Forme : (batch_size, d + 1)

                    net_output = self.nets[n](network_input).squeeze(1)  # Forme : (batch_size,)
                    decision = (net_output > 0.0).float()  # Décision d'arrêt stricte : 1 si arrêter, sinon 0

                    # Mise à jour de g_tau basée sur la décision d'arrêt
                    g_tau = decision * discounted_payoffs[:, n] + (1.0 - decision) * g_tau

                # 4. Accumulation du payoff moyen pour cette simulation
                price_accumulator += g_tau.mean().item()

        # Calcul du prix moyen sur toutes les simulations Monte Carlo
        estimated_price = price_accumulator / self.config.mc_runs
        print(f"\nPrix estimé de l'option Bermudan max-call : {estimated_price:.4f}")
        return estimated_price


def main():
    # Initialisation de la configuration
    config = Config(
        T=3.0,
        N=9,
        d=5,
        r=0.05,
        delta=0.1,
        sigma=0.2,
        K=100.0,
        s_0=110.0,
        train_steps=500,
        mc_runs=500,
        batch_size=8192,
        lr_boundaries=[1000, 2000],
        lr_values=[0.05, 0.005, 0.0005],
        optimizer_eps=0.1
    )

    # Initialisation du modèle Deep Optimal Stopping
    dos_model = DeepOptimalStopping(config)

    # Entraînement du modèle
    dos_model.train()

    # Évaluation du modèle entraîné
    dos_model.evaluate()


if __name__ == "__main__":
    main()



