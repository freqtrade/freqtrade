import numpy as np
from joblib import Parallel
from sklearn.base import is_classifier
from sklearn.multioutput import MultiOutputClassifier, _fit_estimator
from sklearn.utils.fixes import delayed
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, has_fit_parameter


class FreqaiMultiOutputClassifier(MultiOutputClassifier):

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
            Only supported if the underlying classifier supports sample
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

        if is_classifier(self):
            check_classification_targets(y)

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

        self.classes_ = []
        for estimator in self.estimators_:
            self.classes_.extend(estimator.classes_)

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict_proba(self, X):
        """Return prediction probabilities for each class of each output.

        This method will raise a ``ValueError`` if any of the
        estimators do not have ``predict_proba``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        p : array of shape (n_samples, n_classes), or a list of n_outputs \
                such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

            .. versionchanged:: 0.19
                This function now returns a list of arrays where the length of
                the list is ``n_outputs``, and each array is (``n_samples``,
                ``n_classes``) for that particular output.
                """
        check_is_fitted(self)
        results = np.hstack([estimator.predict_proba(X) for estimator in self.estimators_])
        return np.squeeze(results)

    def predict(self, X):
        """Predict multi-output variable using model for each target variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self)
        if not hasattr(self.estimators_[0], "predict"):
            raise ValueError("The base estimator should implement a predict method")

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X) for e in self.estimators_
        )

        results = np.asarray(y).T

        return np.squeeze(results)
