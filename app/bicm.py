import numpy as np
from scipy.stats import norm

class BiCM:
    def __init__(self, player_degrees, opening_degrees):
        self.player_degrees = player_degrees
        self.opening_degrees = opening_degrees
        self.num_players = len(player_degrees)
        self.num_openings = len(opening_degrees)
        self.null_model = None
        self.fit()

    def fit(self):
        self.null_model = np.outer(self.player_degrees, self.opening_degrees) / (self.num_players * self.num_openings)

    def sample(self):
        return self.null_model

    def probability(self, i, j):
        return self.null_model[i, j]

    def get_p_value(self, observed_weight, i, j):
        expected_weight = self.probability(i, j)
        std_dev = np.sqrt(expected_weight * (1 - expected_weight))
        
        if std_dev == 0:
            return 0.0 if observed_weight > expected_weight else 1.0

        z_score = (observed_weight - expected_weight) / std_dev
        p_value = 1 - norm.cdf(z_score)
        return p_value
