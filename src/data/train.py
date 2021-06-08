from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
import numpy as np

def dims(x):
    if np.array(x).ndim == 1:
        return np.array(x).reshape(-1, 1)
    else:
        return x

def train_model(X_train, X_test, y_train, y_test,algo="knn"):
    # all available models in dictionary
    models = {"rf":RandomForestRegressor,"knn": KNeighborsRegressor, "svm":LinearSVR}
    
    # select model
    model = models[algo]()
    
    # train model
    model.fit(dims(X_train),y_train)
    
    # score model
    mse = mean_squared_error(y_test, model.predict(dims(X_test)))
    print("Model {} successfully trained with final MSE of {}".format(algo,mse))
    
    # give back
    return model