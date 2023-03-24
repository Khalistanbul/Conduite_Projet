import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#nom
nom = 'CAMILLE'

# Charger le fichier CSV dans une dataframe Pandas
base_données = pd.read_csv('LightPrenom.csv', sep=';', decimal=',')

# Sélectionner toutes les lignes pour le nom demandé
selection1 = base_données.loc[base_données['Prenom'] == nom]
#garde que le training set = 80%
selection2 = selection1.loc[selection1['Annee'] < 2022]

print(selection2)

#graphe
x1 = list(selection2['Annee'])
y1 = list(selection2['Pourcentage'])
y2 = list(selection2['Nombre'])
# x2 = np.clip(x2, a_min=1900, a_max=1997)

print(x1, len(x1))
print(y1, len(y1))

plt.subplot(221)
plt.bar(x1, y1) 
plt.xlim([1900, 2022])
plt.xlabel('Années')
plt.ylabel('Pourcentage de naissances')
plt.title(nom)

plt.subplot(222)

plt.bar(x1, y2)
plt.xlim([1900, 2022])
plt.xlabel('Années')
plt.ylabel('Nombre de naissances')
plt.title(nom)


plt.show()


