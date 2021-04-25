import xgboost
import torch
import sklearn
import pandas as pd
import numpy as np
from .federate_shap import FederateShap

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        outputs = torch.sigmoid(x)
        return outputs


def test():
    import shap
    shap.initjs()

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # load data
    X, y = shap.datasets.adult()
    cols = ['Age', 'Country', 'Education-Num', 'Marital Status', 'Relationship', 'Race', 'Sex', 'Capital Gain',
            'Capital Loss', 'Workclass', 'Occupation', 'Hours per week']
    X = X[cols]
    # X_display, y_display = shap.datasets.adult(display=True)
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=7)
    # normalize data
    dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
    X_train_norm = X_train.copy()
    X_valid_norm = X_valid.copy()
    for k, dtype in dtypes:
        m = X_train[k].mean()
        s = X_train[k].std()
        X_train_norm[k] -= m
        X_train_norm[k] /= s

        X_valid_norm[k] -= m
        X_valid_norm[k] /= s
    # train model
    knn_norm = sklearn.neighbors.KNeighborsClassifier()
    knn_norm.fit(X_train_norm, y_train)
    # test score
    knn_norm.score(X_valid, y_valid)

    # Explain the model
    f_knn = lambda x: knn_norm.predict_proba(x)[:, 1]
    med = X_train_norm.median().values.reshape((1, X_train_norm.shape[1]))
    x = np.array(X_train_norm.iloc[0])
    # x = np.array(X_train_norm.loc[2583])
    M = 12
    fs = FederateShap()

    # shap
    phi = fs.kernel_shap(f_knn, x, med, M)
    base_value = phi[-1]
    shap_values = phi[:-1]
    shap_values_df = pd.DataFrame(data=np.array([shap_values]), columns=list(X_train_norm))
    print("Shap Values")
    # shap_values_df
    row = shap_values_df.iloc[0]
    row.plot(kind='bar', color='k')


# if __name__ == '__main__':
#     # train XGBoost model
#     X, y = shap.datasets.adult()
#     model = xgboost.XGBClassifier().fit(X, y)
#     # model = LogisticRegression(13, 2)
#     # compute SHAP values
#     explainer = shap.Explainer(model, X)
#     shap_values = explainer(X)
#
#     shap.plots.bar(shap_values)
#
#     clustering = shap.utils.hclust(X, y)  # by default this trains (X.shape[1] choose 2) 2-feature XGBoost models
#     shap.plots.bar(shap_values, clustering=clustering)