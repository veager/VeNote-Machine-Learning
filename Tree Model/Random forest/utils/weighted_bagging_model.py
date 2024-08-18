import itertools
import numbers
from warnings import warn

import numpy as np

from sklearn.utils import _safe_indexing
# Convert list of indices to boolean mask
from sklearn.utils._mask import indices_to_mask

from sklearn.utils.validation import check_random_state, check_X_y, check_is_fitted, _check_sample_weight
from sklearn.utils.parallel import Parallel, delayed

from sklearn.base import RegressorMixin
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._bagging import BaseBagging

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, root_mean_squared_error


def _generate_indices(
        random_state,
        bootstrap,
        n_population,
        n_samples,
        population_weight=None
):
    '''
    original version: scikit-learn/sklearn/ensemble/_bagging.py
    '''
    if population_weight is not None:
        assert n_population == len(population_weight)
        p = population_weight / np.sum(population_weight)
        # print(p)
    else:
        p = None

    indices = random_state.choice(a=n_population, size=n_samples, replace=bootstrap, p=p)

    return indices
# ============================================================
def _generate_bagging_indices(
        random_state,
        bootstrap_features,
        bootstrap_samples,
        n_features,
        n_samples,
        max_features,
        max_samples,
        sample_weight=None
):
    '''
    Randomly draw feature and sample indices.
    original version: scikit-learn/sklearn/ensemble/_bagging.py
    '''
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    if bootstrap_features:
        if n_features == max_features:
            feature_indices = np.arange(n_features)
        else:
            feature_indices = _generate_indices(
                random_state, False, n_features, max_features)
            # sort the indices to make sure in the ascending order
            feature_indices = np.sort(feature_indices)
    else:
        feature_indices = np.arange(n_features)


    sample_indices = _generate_indices(
        random_state, bootstrap_samples, n_samples, max_samples, sample_weight)

    return feature_indices, sample_indices
# ==========================================================================
def _parallel_build_estimators(
        n_estimators,
        ensemble,
        X,
        y,
        sample_weight,
        seeds
):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features       # dtype: int
    max_samples = ensemble._max_samples         # dtype: int
    bootstrap = ensemble.bootstrap              # dtype: bool
    bootstrap_features = ensemble.bootstrap_features    # dtype: bool
    requires_feature_indexing = bootstrap_features or max_features != n_features

    weighted_bootstrap = ensemble.weighted_bootstrap    # dtype: bool
    weighted_training = ensemble.weighted_training      # dtype: bool


    # Build estimators
    estimators = []
    estimators_samples = []
    estimators_features = []

    for i in range(n_estimators):
        # if verbose > 1:
        #     print(f"Building estimator {i+1} of {n_estimators} for this parallel run (total {total_n_estimators})...")

        random_state = seeds[i]
        estimator = ensemble._make_estimator(append=False, random_state=random_state)

        # sample weights for bootstrapping
        bootstrap_sw_ = sample_weight if weighted_bootstrap else None

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(
            random_state,
            bootstrap_features,
            bootstrap,
            n_features,
            n_samples,
            max_features,
            max_samples,
            sample_weight = bootstrap_sw_)

        y_ = _safe_indexing(y, indices)
        X_ = _safe_indexing(X, indices)
        sw_ = _safe_indexing(sample_weight, indices) if weighted_training else None

        if requires_feature_indexing:
            X_ = X_[:, features]

        estimator.fit(X_, y_, sample_weight=sw_)

        estimators.append(estimator)
        estimators_samples.append(indices)
        estimators_features.append(features)

    return estimators, estimators_samples, estimators_features
# ==========================================================================
def _parallel_predict_regression(estimators, estimators_features, X):
    """Private function used to compute predictions within a job."""
    return sum(
        estimator.predict(X[:, features])
        for estimator, features in zip(estimators, estimators_features)
    )
# ==========================================================================
class WeightedBaggingRegressor(RegressorMixin, BaseBagging):
    def __init__(
            self,
            estimator = None,
            n_estimators = 10,
            max_samples = 1.0,
            max_features = 1.0,
            bootstrap = True,
            bootstrap_features = True,
            weighted_bootstrap = True,
            weighted_training = True,
            oob_score = False,
            n_jobs = None,
            random_state = None,
            verbose = 0
    ):
        super().__init__(
            estimator = estimator,
            n_estimators = n_estimators,
            max_samples = max_samples,
            max_features = max_features,
            bootstrap = bootstrap,
            bootstrap_features = bootstrap_features,
            oob_score = oob_score,
            warm_start = False,
            n_jobs = n_jobs,
            random_state = random_state,
            verbose = verbose
        )

        self.weighted_bootstrap = weighted_bootstrap
        self.weighted_training  = weighted_training
    # -------------------------------------------------------------------------
    def _get_estimator(self):
        """Resolve which estimator to return (default is DecisionTreeClassifier)"""
        if self.estimator is None:
            return DecisionTreeRegressor()
        return self.estimator
    # -------------------------------------------------------------------------
    def _check_bootstrap_params(self):
        '''
        Check the bootstrap related parameters:
        self.bootstrap
        self.bootstrap_features
        self.max_samples
        self.max_features
        '''

        # Check bootstrap samples
        if self.bootstrap:
            # Validate self.max_samples
            if isinstance(self.max_samples, numbers.Integral):
                max_samples = self.max_samples
            elif isinstance(self.max_samples, float):
                max_samples = int(self.max_samples * self._n_samples)
            elif self.max_samples is None:
                max_samples = self._n_samples
            else:
                raise ValueError('`max_samples` must be int or float')

            if max_samples > self._n_samples:
                raise ValueError('`max_samples` must be <= `n_samples`')

        # self.bootstrap = False
        else:
            if self.max_samples is not None:
                raise ValueError("`max_samples` cannot be set if `bootstrap=False`. Either switch to `bootstrap=True` or set `max_samples=None`.")
            else:
                max_samples = self._n_samples

        # Store validated integer row sampling value
        self._max_samples = max_samples


        # Check bootstrap features
        if self.bootstrap_features:
            # Validate max_features
            if isinstance(self.max_features, numbers.Integral):
                max_features = self.max_features
            elif isinstance(self.max_features, float):
                max_features = int(self.max_features * self.n_features_in_)
            elif self.max_features is None:
                max_features = self.n_features_in_
            else:
                raise ValueError('`max_features` must be int or float')

            if max_features > self.n_features_in_:
                raise ValueError('`max_features` must be <= `n_features`')

            max_features = max(1, int(max_features))

        # self.bootstrap_features = False
        else:
            if self.max_features is not None:
                raise ValueError('`max_features` cannot be set if `bootstrap_features=False`. Either switch to `bootstrap_features=True` or set `max_features=None`.')
            else:
                max_features = self.n_features_in_

        # Store validated integer feature sampling value
        self._max_features = max_features

        return self
    # -------------------------------------------------------------------------
    def fit(self, X, y, sample_weight=None):

        random_state = check_random_state(self.random_state)

        # check X, y
        X, y = check_X_y(X, y, accept_sparse=False, dtype=None)

        self._X = X
        self._y = y
        self._sample_weight = sample_weight

        # Remap output
        self._n_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]

        # Check parameters
        self._validate_estimator(self._get_estimator())

        # Check bootstrap parameters
        self._check_bootstrap_params()

        # Check sample weights
        # (True, True), (True, False), (False, True)
        if self.weighted_bootstrap or self.weighted_training:
            if sample_weight is None:
                raise ValueError('`sample_weight` must be provided when `bootstrap_weighted=True` or `weighted_training=True`')
            else:
                sample_weight = _check_sample_weight(sample_weight, X, dtype=None)
        # (False, False)
        else:
            if sample_weight is not None:
                raise ValueError('`sample_weight` cannot be set if both `bootstrap_weighted` and `weighted_training` are False')
            else:
                sample_weight = None

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError('Out of bag estimation only available if `bootstrap=True`')

        # Parallel loop
        total_n_estimators = self.n_estimators

        n_jobs, n_estimators, starts = _partition_estimators(
            total_n_estimators, self.n_jobs)

        seeds = random_state.randint(np.iinfo(np.int32).max, size=total_n_estimators)
        self._seeds = seeds

        # parallel fitting
        all_results = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i] : starts[i + 1]],
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ = list(itertools.chain.from_iterable(t[0] for t in all_results))
        # already exist attribute `self.estimators_samples_`
        self.estimators_samples__ = list(itertools.chain.from_iterable(t[1] for t in all_results))
        self.estimators_features_ = list(itertools.chain.from_iterable(t[2] for t in all_results))

        if self.oob_score:
            _oob_sw = sample_weight if self.weighted_training else None
            self._set_oob_score(X, y, _oob_sw)

        return self
    # -------------------------------------------------------------------------------------
    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)

        # Check data
        X = self._validate_data(X)

        # Parallel loop
        n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)

        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_regression)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
            )
            for i in range(n_jobs)
        )

        # Reduce
        y_hat = sum(all_y_hat) / self.n_estimators

        return y_hat
    # --------------------------------------------------------------------------------
    def _set_oob_score(self, X, y, sample_weight):

        n_samples = y.shape[0]

        predictions = np.zeros((n_samples,))
        n_predictions = np.zeros((n_samples,))

        for estimator, samples, features in zip(
                self.estimators_, self.estimators_samples__, self.estimators_features_
        ):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            predictions[mask] += estimator.predict((X[mask, :])[:, features])
            n_predictions[mask] += 1

        if (n_predictions == 0).any():
            warn(
                "Some inputs do not have OOB scores. "
                "This probably means too few estimators were used "
                "to compute any reliable oob estimates."
            )
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions

        self.oob_prediction_ = predictions
        self.oob_score_ = r2_score(y, predictions)
        # self.oob_score_ = root_mean_squared_error(y, predictions, sample_weight=sample_weight)
# ======================================================================================