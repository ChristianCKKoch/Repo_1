from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Preprocessor:
    def __init__(self,x,y):
        # hier hab ich daten bekommen
        
        # hier werden daten gesplittet
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x,y, test_size=0.2, random_state=42)
        
        # hier werden die daten an den normalizer übergeben
        self.fit_normalizer(self.X_train)
        
        # ab hier hab ich jetzt den scaler fertig
  
    def get_data(self):
        return self.normalize(self.X_train), self.normalize(self.X_test), self.y_train, self.y_test, self.scaler
        
    # trainiert den normalizer
    def fit_normalizer(self,x):
        # hier kommen die übergebenen daten an
        # hier initialisiert
        self.scaler = MinMaxScaler()
        
        # hier trainiert
        self.scaler.fit(dims(x))
        
    def normalize(self,x):
        # wendet einfach den gespeicherten scaler an
        return self.scaler.transform(dims(x))

def dims(x):
    if np.array(x).ndim == 1:
        return np.array(x).reshape(-1, 1)
    else:
        return x