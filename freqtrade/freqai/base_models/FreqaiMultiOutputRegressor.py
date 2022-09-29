from joblib import Parallel
from sklearn.multioutput import MultiOutputRegressor, _fit_estimator
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import has_fit_parameter


class FreqaiMultiOutputRegressor(MultiOutputRegressor):

    def fit(self, X, y, sample_weight=None, fit_params=None):
        """Fit the model to data, separately for each output variable.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.
        fit_params : A list of dicts for the fit_params
            Parameters passed to the ``estimator.fit`` method of each step.
            Each dict may contain same or different values (e.g. different
            eval_sets or init_models)
            .. versionadded:: 0.23
        Returns
        -------
        self : object
            Returns a fitted instance.
        """

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        y = self._validate_data(X="no_validation", y=y, multi_output=True)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        if sample_weight is not None and not has_fit_parameter(
            self.estimator, "sample_weight"
        ):
            raise ValueError("Underlying estimator does not support sample weights.")

        if not fit_params:
            fit_params = [None] * y.shape[1]

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator, X, y[:, i], sample_weight, **fit_params[i]
            )
            for i in range(y.shape[1])
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return
