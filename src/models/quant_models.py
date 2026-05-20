import lightgbm as lgb
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib

class AdvancedQuantModels:
    def __init__(self):
        self.regime_model = GaussianHMM(n_components=7, covariance_type="full", n_iter=100)
        self.trend_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
        self.flow_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
        self.rs_model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=200)
        self.vol_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
        self.scaler = StandardScaler()

    def save(self, path):
        joblib.dump(self, path)

class MetaModel:
    def __init__(self):
        self.regime_weights = {
            'Strong Trend': {'trend': 0.6, 'rs': 0.3, 'flow': 0.1, 'vol': 0.0},
            'Compression':  {'trend': 0.2, 'rs': 0.4, 'flow': 0.3, 'vol': 0.1},
            'Capitulation': {'trend': 0.0, 'rs': 0.0, 'flow': 0.2, 'vol': 0.8},
            'Neutral':      {'trend': 0.25, 'rs': 0.25, 'flow': 0.25, 'vol': 0.25}
        }
    def ensemble(self, scores, regime):
        w = self.regime_weights.get(regime, self.regime_weights['Neutral'])
        return sum(scores[k] * w[k] for k in w)
