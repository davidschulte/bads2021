from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from DataTransformer import DataTransformer

"""This class is the final model. It is composed of a pipeline with our DataTransformer, a StandardScaler and the estimator
that is passed when an instance of the class is created.
If we set use_delivery_dates to True, the hardcoded rule discussed in the notebook will be enforced."""
class Model(BaseEstimator):

    def __init__(self, estimator, use_delivery_date=True):
        self.pipeline = Pipeline([
            ('transformer', DataTransformer(use_delivery_date)), # ('step_name', transfomer) always follow this format for transformers in the pipeline
            ('scaler', StandardScaler()),
            ('estimator', estimator) # ('step_name', fun()) add parantheses for other functions
            ])
        self.use_delivery_date = use_delivery_date

    def fit(self, X, y):
        self.pipeline.fit(X,y)

    def predict_proba(self, X):
        y_pred = self.pipeline.predict_proba(X)
        if self.use_delivery_date:
            y_pred[X['delivery_date'].isnull()] = 0

        return y_pred

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:,1] > threshold).astype('int32')
