
# Recommandation de Plantes Médicinales - Modèle Machine Learning

Ce projet implémente un modèle de machine learning permettant de recommander des plantes médicinales basées sur les symptômes décrits par un utilisateur. En utilisant la similarité cosinus et des représentations textuelles basées sur TF-IDF, le système analyse les bienfaits et utilisations des plantes pour suggérer les options les plus pertinentes.

---

## Fonctionnalités
- **Prise en compte des symptômes** : L'utilisateur entre une description de ses symptômes ou besoins, et le modèle fournit une liste de plantes appropriées.
- **Recommandations enrichies** : Chaque plante recommandée est accompagnée de ses bienfaits, utilisations et précautions d'emploi.
- **Affichage des scores de pertinence** : Les recommandations sont triées par pertinence, basée sur une analyse textuelle.
- **Support multilingue** : Le système utilise les stop words en français pour un traitement adapté des descriptions.

---

## Prérequis

### Bibliothèques Python requises
Assurez-vous que les bibliothèques suivantes sont installées :

- `pandas`
- `scikit-learn`
- `nltk`

Pour installer ces dépendances :
```bash
pip install pandas scikit-learn nltk
```

### Données d’entrée
Le modèle utilise un fichier CSV contenant des informations sur les plantes médicinales, incluant les colonnes suivantes :
- `Plante` : Le nom de la plante
- `Bienfaits` : Les bienfaits associés à cette plante
- `Utilisations` : Les différentes manières d'utiliser la plante
- `Précautions` : Les précautions à prendre lors de l'utilisation

Un exemple de fichier CSV (`plantes_medicinales.csv`) est fourni.

---

## Structure du Projet

### 1. **Chargement et Prétraitement des Données**
Le fichier CSV est chargé dans un DataFrame Pandas. Les textes longs sont tronqués pour un affichage lisible.

```python
# Charger les données depuis le fichier CSV
df = pd.read_csv('plantes_medicinales.csv')

# Tronquer les textes longs pour les rendre lisibles
df_tronque = df.apply(lambda x: x.apply(lambda y: str(y)[:10] + "..." if len(str(y)) > 10 else str(y)))
```

### 2. **Vectorisation TF-IDF**
Le texte des colonnes "Bienfaits" et "Utilisations" est combiné et transformé en vecteurs TF-IDF pour une analyse efficace de la similarité textuelle.

```python
def vectoriser_texte(texte_plantes):
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_fr)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texte_plantes)
    return tfidf_matrix
```

### 3. **Calcul de la Similarité Cosinus**
Le symptôme utilisateur est comparé aux descriptions des plantes pour calculer une similarité cosinus.

```python
similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
scores_similarite = similarity_matrix.flatten()
```

### 4. **Affichage des Recommandations**
Les plantes sont triées par pertinence, et les 5 meilleures recommandations sont affichées avec leurs détails.

```python
recommandations = df[['Plante', 'Bienfaits', 'Utilisations', 'Précautions', 'score_similarite']].sort_values(by='score_similarite', ascending=False)
```

---

## Exemple d’Utilisation

Voici un scénario d’utilisation complet :

```python
# Exemple d'utilisation : Entrer un symptôme
symptome_utilisateur = "Je cherche à réduire l'anxiété et favoriser un bon sommeil"
recommandations = recommander_plantes(symptome_utilisateur, df)

# Afficher les recommandations enrichies
print("
Plantes recommandées :")
for _, row in recommandations.iterrows():
    print(f"
Plante : {row['Plante']}")
    print(f"Bienfaits : {row['Bienfaits']}")
    print(f"Utilisations : {row['Utilisations']}")
    print(f"Précautions : {row['Précautions']}")
    print(f"Pertinence : {row['score_similarite']:.2f}")
```

### Exemple de sortie
```
Plantes recommandées :

Plante : Lavande
Bienfaits : Apaise l'anxiété, favorise le sommeil
Utilisations : Huile essentielle, infusion
Précautions : Éviter en cas de grossesse
Pertinence : 0.85

Plante : Valériane
Bienfaits : Favorise un sommeil réparateur, apaise l'anxiété
Utilisations : Infusion, teinture
Précautions : Peut interagir avec des médicaments pour le sommeil
Pertinence : 0.82

Plante : Camomille
Bienfaits : Réduit le stress, améliore le sommeil
Utilisations : Infusion, teinture
Précautions : Pas recommandé pour les personnes allergiques aux plantes de la famille des astéracées
Pertinence : 0.79
```

---

## Améliorations Futures
- **Ajout de nouvelles données** : Intégrer davantage de plantes avec des descriptions détaillées.
- **Amélioration de l'interface utilisateur** : Créer une application Web ou mobile pour faciliter l'utilisation.
- **Traduction automatique** : Ajouter un support pour les symptômes en anglais ou dans d'autres langues.
- **Explication des scores** : Fournir une justification détaillée des scores de pertinence pour chaque plante.

---

## Licence
Ce projet est open source. Vous êtes libre de l'utiliser, le modifier et le distribuer avec attribution.

---

## Créateurs
- Nom :
#### ZAHA Fatima Zahra
#### GUELBAOUI Hajar
#### TIRAOUI Doha

