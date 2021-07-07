import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from Preprocessor import Preprocessor
import sys
sys.path.append("src/models/")
from Classifier import Classifier
from operator import itemgetter, attrgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

#Verbindung zu sqlite Datenbank öffnen
connection = sqlite3.connect("data/raw/database.sqlite")

#SQL-Abfrage in pandas DataFrame
sql = "SELECT * FROM vw_Match_Team_Data"

daten = pd.read_sql_query(sql, connection)
connection.close()

onehot_enc_list = list()
daten_join = daten
#Teamname-Spalten encoden (one-hot-encoding)
for spalte in ('home_team_name','away_team_name'):
    team_name = daten[spalte]
    team_name_res = team_name.values.reshape(-1,1)
    team_name_onehot_enc = OneHotEncoder()
    team_name_onehot = team_name_onehot_enc.fit_transform(team_name_res)
    
    #onehot_enc_list.append([spalte, team_name_onehot, team_name_onehot_enc])
    
    sp_name_array = team_name_onehot_enc.categories_[0]
    for i in range(0, sp_name_array.size):
        sp_name_array[i] = sp_name_array[i] + '_H' if spalte == 'home_team_name' else sp_name_array[i] + '_A'
    daten_join = daten_join.join(pd.DataFrame(team_name_onehot.toarray(), columns=sp_name_array))

result_spalte = daten_join.pop("result")
daten_join['result'] = result_spalte
daten_prepared = daten_join.drop(["home_team_name", "away_team_name"], axis=1)

result['R_Kla'] = 0 if result['reproduction'] < 1 else 1

# print(onehot_enc_list[1][1].toarray())
# print("Array-Objekt 0")
# print(onehot_enc_list[1][2].categories_[0])

# sp_name_array = onehot_enc_list[1][2].categories_[0]
# for i in range(0, sp_name_array.size):
#     sp_name_array[i] = sp_name_array[i] + '_H'

# daten_join = daten.join(pd.DataFrame(onehot_enc_list[1][1].toarray(), columns=sp_name_array))

daten_prepared.to_excel("data/processed/Bundesliga_daten.xlsx")

X = daten_prepared.loc[:,daten_prepared.columns != "result"]
y = daten_prepared["result"]

X_train, X_test, y_train, y_test, scaler = Preprocessor(X, y).get_data()

X_train_exp = pd.DataFrame(X_train)
X_train_exp.to_excel("data/processed/Bundesliga_daten_X_normalized.xlsx")

#Models spezifieren
models = {"rf":RandomForestClassifier, "dt":DecisionTreeClassifier, "knn":KNeighborsClassifier, "svm":SVC, "mlp":MLPClassifier}

#Classifier verwenden
clf = Classifier()
resultat = clf.train_models(X_train, X_test, y_train, y_test, models)

#Bestes Ergebnis bestimmen und als Modell speichern
print("Bestes Model ist: {} mit einer Akkuranz von {}%".format(sorted(resultat, key=itemgetter(1), reverse=True)[0][0],sorted(resultat, key=itemgetter(1), reverse=True)[0][1]*100))
bestes_model = sorted(resultat, key=itemgetter(1), reverse=True)[0][2]
print("Alle Ergebnisse: {}".format(resultat))