#Dies ist das Python-File zum Bestimmen von Ecoli-Klassen / Anwenden des Ecoli-Modells
import pandas as pd

#Model und Scaler aus Dateien lesen
clf = pd.read_pickle(r'../../models/classifier_object.pickle')
sca = pd.read_pickle(r'../../models/scaler_object.pickle')

#Zu bestimmende Werte aus Excel-Datei lesen
data = pd.read_excel('../../data/raw/U bung kNN Klassifizierung Ecoli.xls', sheet_name=1)

#Normalisieren der Daten
X_output = sca.transform(data)
y_pred = clf.predict(X_output)

#Ausgabe der vorhergesagten Ergebnisse
print(y_pred)