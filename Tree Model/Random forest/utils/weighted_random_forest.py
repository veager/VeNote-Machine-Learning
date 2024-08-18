import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.tree import DecisionTreeRegressor

from sklearn.utils.validation import check_X_y, check_is_fitted, _check_sample_weight, check_array
from sklearn.utils._mask import indices_to_mask

from .weighted_bagging_model import WeightedBaggingRegressor


class WeightedRandomForestRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators=100,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=True,
        weighted_bootstrap=True,
        weighted_training=True,
        oob_score=False,
        random_state=None,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        monotonic_cst=None,
        ccp_alpha=0.0,
        n_jobs=None,
        verbose=0,
    ):
        # Bootstrapping parameters, passed to the `WeightedBaggingRegressor`
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.weighted_bootstrap = weighted_bootstrap
        self.weighted_training = weighted_training

        self.oob_score = oob_score
        self.random_state = random_state

        # Decision tree parameters, passed to the `DecisionTreeRegressor`
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst

        self.n_jobs = n_jobs
        self.verbose = verbose

    # -----------------------------------------------
    def model_initialization(self):

        dt = DecisionTreeRegressor(
            splitter='best',
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=1.0,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            monotonic_cst=self.monotonic_cst,
            random_state=None)

        self.estimator_ = clone(dt)

        model = WeightedBaggingRegressor(
            estimator=dt,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            bootstrap_features=self.bootstrap_features,
            weighted_bootstrap=self.weighted_bootstrap,
            weighted_training=self.weighted_training,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose)

        return model

    # ------------------------------------------------------------
    def fit(self, X, y, sample_weight=None):

        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        self._model = self.model_initialization()
        self._model.fit(X, y, sample_weight=sample_weight)

        self.n_features_in_ = self._model.n_features_in_

        self.estimators_ = self._model.estimators_
        self.estimators_samples_ = self._model.estimators_samples__
        self.estimators_features_ = self._model.estimators_features_

        if self.oob_score:
            self.oob_score_ = self._model.oob_score_
            self.oob_prediction_ = self._model.oob_prediction_

        return self
        # ------------------------------------------------------------------

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        pred = self._model.predict(X)
        return pred

    # -------------------------------------------------------------------
    @property
    def feature_importances_(self):
        '''
        feature importance of submodels must be sored and reindexed due to feature bootstrapping
        '''
        check_is_fitted(self)
        # features of submodels are sampled and shuffled
        feat_imp = np.zeros(self.n_features_in_)

        for feat_idx, submodel in zip(self.estimators_features_, self.estimators_):
            sorted_feat_idx = np.argsort(feat_idx)
            feat_imp[indices_to_mask(feat_idx, self.n_features_in_)] += submodel.feature_importances_[sorted_feat_idx]

        feat_imp = feat_imp / feat_imp.sum()

        return feat_imp
    # -------------------------------------------------------------------
# ====================================================================================================