#!/usr/bin/env python
# coding: utf-8

# # Adaptation de Domain

# ## Feature Based - Pred

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Notre implémentation de PRED
class SimplePRED(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=LogisticRegression()):
        self.base_estimator = base_estimator
        
    def fit(self, Xs, ys, Xt, yt):
        self.source_model = clone(self.base_estimator)
        self.source_model.fit(Xs, ys)
        
        source_preds = self.source_model.predict_proba(Xt)
        Xt_aug = np.hstack([Xt, source_preds])
        
        self.target_model = clone(self.base_estimator)
        self.target_model.fit(Xt_aug, yt)
        return self
    
    def predict(self, X, domain='target'):
        if domain == 'source':
            return self.source_model.predict(X)
        source_preds = self.source_model.predict_proba(X)
        X_aug = np.hstack([X, source_preds])
        return self.target_model.predict(X_aug)

    def score(self, X, y, domain='target'):
        return np.mean(self.predict(X, domain) == y)

# Création des données
print("Generating datasets...")
Xs, ys = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_redundant=5, random_state=42)
Xt, yt = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=43)
Xt = Xt + 2  # Ajout d'un shift pour simuler la différence de domaine

# Split des données
Xt_train, Xt_test, yt_train, yt_test = train_test_split(Xt, yt, test_size=0.2, 
                                                        random_state=42)

# Modèle de base (sans adaptation)
print("Training base model...")
base_model = LogisticRegression(random_state=42)
base_model.fit(Xs, ys)
base_score = base_model.score(Xt_test, yt_test)

# Modèle PRED
print("Training PRED model...")
pred_model = SimplePRED(LogisticRegression(random_state=42))
pred_model.fit(Xs, ys, Xt_train, yt_train)
pred_score = pred_model.score(Xt_test, yt_test)

# Résultats
results = {
    'Base Model': base_score,
    'PRED Model': pred_score
}

# Affichage des résultats
print("\nAccuracy Scores:")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

# Visualisation
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Performance Comparison")
plt.ylabel("Accuracy Score")
plt.ylim(0, 1)
plt.show()

# Rapport détaillé
y_pred = pred_model.predict(Xt_test)
print("\nClassification Report:")
print(classification_report(yt_test, y_pred))


# ## Instance Based - LinInt

# In[5]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def create_financial_data(n_samples=500, n_features=4, shift=0.2, noise=0.1):
   """
   Crée des données synthétiques qui simulent des données financières
   avec un shift réduit et des relations non-linéaires
   """
   np.random.seed(42)  # Pour reproductibilité
   
   # Données source
   Xs = np.random.randn(n_samples, n_features)
   # Relations non-linéaires plus complexes
   ys = (0.3 * np.sin(2*Xs[:,0]) + 
         0.5 * np.exp(-Xs[:,1]**2) + 
         0.2 * Xs[:,2]**2 + 
         0.1 * np.tanh(Xs[:,3]) + 
         noise * np.random.randn(n_samples))
   
   # Données cible avec shift réduit
   Xt = Xs + shift * np.random.randn(n_samples, n_features)
   yt = (0.3 * np.sin(2*Xt[:,0]) + 
         0.5 * np.exp(-Xt[:,1]**2) + 
         0.2 * Xt[:,2]**2 + 
         0.1 * np.tanh(Xt[:,3]) + 
         noise * np.random.randn(n_samples))
   
   return Xs, ys, Xt, yt

class LinIntFinance:
   def __init__(self, base_estimator=KernelRidge(kernel='rbf'), prop=0.7, random_state=42):
       self.base_estimator = base_estimator
       self.prop = prop
       self.random_state = random_state
       
   def fit(self, Xs, ys, Xt, yt):
       np.random.seed(self.random_state)
       
       # Division des données cible
       shuffle_idx = np.random.permutation(len(Xt))
       cut = int(len(Xt) * self.prop)
       
       Xt_train = Xt[shuffle_idx[:cut]]
       yt_train = yt[shuffle_idx[:cut]]
       Xt_valid = Xt[shuffle_idx[cut:]]
       yt_valid = yt[shuffle_idx[cut:]]
       
       # Entraînement des modèles
       self.source_model = clone(self.base_estimator)
       self.target_model = clone(self.base_estimator)
       
       self.source_model.fit(Xs, ys)
       self.target_model.fit(Xt_train, yt_train)
       
       # Prédictions pour l'interpolation
       source_pred = self.source_model.predict(Xt_valid)
       target_pred = self.target_model.predict(Xt_valid)
       
       # Apprentissage de l'interpolation avec régularisation
       X_combined = np.column_stack([source_pred, target_pred])
       self.interpolator = Ridge(alpha=0.1)  # Légère régularisation
       self.interpolator.fit(X_combined, yt_valid)
       
       return self
   
   def predict(self, X):
       source_pred = self.source_model.predict(X)
       target_pred = self.target_model.predict(X)
       X_combined = np.column_stack([source_pred, target_pred])
       return self.interpolator.predict(X_combined)

def evaluate_models(y_true, y_base, y_linint):
   """Évalue les modèles et retourne un dictionnaire de métriques"""
   return {
       'Base R2': r2_score(y_true, y_base),
       'LinInt R2': r2_score(y_true, y_linint),
       'Base MSE': mean_squared_error(y_true, y_base),
       'LinInt MSE': mean_squared_error(y_true, y_linint)
   }

def plot_results(y_true, y_base, y_linint, title="Performance Comparison"):
   plt.figure(figsize=(15, 7))
   
   # Subplot 1: Comparaison des prédictions
   plt.subplot(1, 2, 1)
   plt.plot(y_true, label='True Values', color='black', alpha=0.7)
   plt.plot(y_base, label='Base Model', color='blue', alpha=0.5)
   plt.plot(y_linint, label='LinInt Model', color='red', alpha=0.5)
   plt.title('Predictions Comparison')
   plt.xlabel('Sample Index')
   plt.ylabel('Value')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   # Subplot 2: Scatter plot
   plt.subplot(1, 2, 2)
   plt.scatter(y_true, y_base, alpha=0.5, label='Base Model', color='blue')
   plt.scatter(y_true, y_linint, alpha=0.5, label='LinInt Model', color='red')
   plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', label='Perfect Fit')
   plt.title('Predicted vs True Values')
   plt.xlabel('True Values')
   plt.ylabel('Predicted Values')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

if __name__ == "__main__":
   # Génération des données
   print("Generating synthetic financial data...")
   Xs, ys, Xt, yt = create_financial_data(n_samples=1000)  # Plus de données
   
   # Standardisation
   scaler_X = StandardScaler()
   scaler_y = StandardScaler()
   
   Xs = scaler_X.fit_transform(Xs)
   Xt = scaler_X.transform(Xt)
   ys = scaler_y.fit_transform(ys.reshape(-1, 1)).ravel()
   yt = scaler_y.transform(yt.reshape(-1, 1)).ravel()
   
   # Division train/test
   test_size = int(len(Xt) * 0.2)
   Xt_train, Xt_test = Xt[:-test_size], Xt[-test_size:]
   yt_train, yt_test = yt[:-test_size], yt[-test_size:]
   
   # Modèles
   base_model = KernelRidge(kernel='rbf', alpha=0.1)
   linint_model = LinIntFinance(base_model)
   
   print("Training models...")
   # Entraînement
   base_model.fit(Xs, ys)
   linint_model.fit(Xs, ys, Xt_train, yt_train)
   
   # Prédictions
   base_pred = base_model.predict(Xt_test)
   linint_pred = linint_model.predict(Xt_test)
   
   # Évaluation
   metrics = evaluate_models(yt_test, base_pred, linint_pred)
   
   print("\nPerformance Metrics:")
   for name, value in metrics.items():
       print(f"{name}: {value:.4f}")
   
   # Visualisation
   plot_results(yt_test, base_pred, linint_pred)


# ## Paramter Based - Regular Transfer

# In[6]:


pip install numpy pandas scikit-learn matplotlib seaborn scipy


# In[11]:


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns

class ImprovedRegularTransfer(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, lambda_=1.0):
        self.base_estimator = base_estimator if base_estimator is not None else LogisticRegression(class_weight='balanced')
        self.lambda_ = lambda_
        self.source_model = None
        self.target_model = None
        self.scaler = StandardScaler(with_mean=False)  # Pour supporter les matrices sparses
        
    def fit(self, Xs, ys, Xt, yt):
        """
        Entraîne le modèle avec adaptation de domaine améliorée
        """
        # Normalisation des données
        Xs_scaled = self.scaler.fit_transform(Xs)
        Xt_scaled = self.scaler.transform(Xt)
        
        # Entraînement du modèle source
        self.source_model = clone(self.base_estimator)
        self.source_model.fit(Xs_scaled, ys)
        
        # Adaptation pour le domaine cible
        self.target_model = clone(self.base_estimator)
        self.target_model.fit(Xt_scaled, yt)
        
        # Transfert des connaissances avec régularisation
        if hasattr(self.source_model, 'coef_'):
            source_coef = self.source_model.coef_.flatten()
            target_coef = self.target_model.coef_.flatten()
            
            # Adaptation des coefficients
            adapted_coef = (1 - self.lambda_) * target_coef + self.lambda_ * source_coef
            self.target_model.coef_ = adapted_coef.reshape(1, -1)
        
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.target_model.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.target_model.predict_proba(X_scaled)

def load_data():
    """
    Charge les données de 20 newsgroups
    """
    print("Chargement des données source (comp.graphics)...")
    source_data = fetch_20newsgroups(
        subset='train',
        categories=['comp.graphics'],
        shuffle=True,
        random_state=42
    )
    
    print("Chargement des données cible (sci.med)...")
    target_data = fetch_20newsgroups(
        subset='train',
        categories=['sci.med'],
        shuffle=True,
        random_state=42
    )
    
    # Création des DataFrames avec labels équilibrés
    source_df = pd.DataFrame({
        'text': source_data.data,
        'label': [1 if i < len(source_data.data)/2 else 0 for i in range(len(source_data.data))]
    })
    
    target_df = pd.DataFrame({
        'text': target_data.data,
        'label': [1 if i < len(target_data.data)/2 else 0 for i in range(len(target_data.data))]
    })
    
    # Équilibrage et échantillonnage
    min_samples = min(len(source_df), len(target_df), 5000)
    source_df = pd.concat([
        source_df[source_df['label'] == 0].sample(min_samples//2, random_state=42),
        source_df[source_df['label'] == 1].sample(min_samples//2, random_state=42)
    ])
    target_df = pd.concat([
        target_df[target_df['label'] == 0].sample(min_samples//2, random_state=42),
        target_df[target_df['label'] == 1].sample(min_samples//2, random_state=42)
    ])
    
    return {'Source': source_df, 'Target': target_df}

def preprocess_text_data(source_data, target_data):
    """
    Prétraitement amélioré des données textuelles
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Combiner les textes pour vocabulaire commun
    all_texts = list(source_data['text']) + list(target_data['text'])
    vectorizer.fit(all_texts)
    
    # Transformer les données
    Xs = vectorizer.transform(source_data['text'])
    Xt = vectorizer.transform(target_data['text'])
    
    ys = source_data['label'].values
    yt = target_data['label'].values
    
    return Xs, ys, Xt, yt, vectorizer

def optimize_lambda(Xs, ys, Xt_labeled, yt_labeled, Xt_val, yt_val):
    """
    Optimisation améliorée du paramètre lambda
    """
    lambda_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    best_score = -1
    best_lambda = 0.5  # valeur par défaut
    
    print("Testing lambda values:", lambda_values)
    for lambda_ in lambda_values:
        print(f"Testing lambda = {lambda_}")
        model = ImprovedRegularTransfer(
            base_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
            lambda_=lambda_
        )
        model.fit(Xs, ys, Xt_labeled, yt_labeled)
        score = model.score(Xt_val, yt_val)
        print(f"Score for lambda {lambda_}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_lambda = lambda_
    
    print(f"Optimization successful. Best score: {best_score:.4f}")
    return best_lambda

def evaluate_model_performance(model, X_test, y_test, model_name="Model"):
    """
    Évaluation complète du modèle
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'avg_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    return metrics

def visualize_results(source_scores, target_scores, adapted_scores):
    """
    Visualisation des résultats
    """
    plt.figure(figsize=(12, 5))
    
    # Moyennes et écarts-types
    means = [np.mean(scores) for scores in [source_scores, target_scores, adapted_scores]]
    stds = [np.std(scores) for scores in [source_scores, target_scores, adapted_scores]]
    
    # Plot
    labels = ['Source', 'Target', 'Adapted']
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.title('Model Performance Comparison')
    plt.ylabel('Accuracy Score')
    plt.ylim(0, 1)
    
    # Valeurs exactes
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                ha='center', va='bottom')
    
    plt.show()

def main():
    # Paramètres
    N_TRIALS = 5
    TARGET_RATIO = 0.1
    
    # Stockage des résultats
    source_scores = []
    target_scores = []
    adapted_scores = []
    
    # Chargement des données
    print("Loading data...")
    data = load_data()
    
    for trial in range(N_TRIALS):
        print(f"\nTrial {trial + 1}/{N_TRIALS}")
        
        # Prétraitement
        print("Preprocessing data...")
        Xs, ys, Xt, yt, vectorizer = preprocess_text_data(
            data['Source'],
            data['Target']
        )
        
        # Division des données cible
        n_target_samples = Xt.shape[0]
        n_labeled_target = int(TARGET_RATIO * n_target_samples)
        indices = np.random.permutation(n_target_samples)
        
        Xt_labeled = Xt[indices[:n_labeled_target]]
        yt_labeled = yt[indices[:n_labeled_target]]
        Xt_test = Xt[indices[n_labeled_target:]]
        yt_test = yt[indices[n_labeled_target:]]
        
        # Optimisation de lambda
        print("Optimizing lambda parameter...")
        best_lambda = optimize_lambda(Xs, ys, Xt_labeled, yt_labeled, 
                                    Xt_test[:100], yt_test[:100])
        print(f"Best lambda found: {best_lambda}")
        
        # Modèle de base (source)
        base_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42+trial)
        base_model.fit(Xs, ys)
        
        # Modèle adapté
        adapted_model = ImprovedRegularTransfer(
            base_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42+trial),
            lambda_=best_lambda
        )
        adapted_model.fit(Xs, ys, Xt_labeled, yt_labeled)
        
        # Évaluation
        source_score = base_model.score(Xt_test, yt_test)
        target_score = LogisticRegression(class_weight='balanced', max_iter=1000).fit(
            Xt_labeled, yt_labeled).score(Xt_test, yt_test)
        adapted_score = adapted_model.score(Xt_test, yt_test)
        
        source_scores.append(source_score)
        target_scores.append(target_score)
        adapted_scores.append(adapted_score)
        
        print(f"Source Model Accuracy: {source_score:.4f}")
        print(f"Target Model Accuracy: {target_score:.4f}")
        print(f"Adapted Model Accuracy: {adapted_score:.4f}")
        
        if trial == 0:
            print("\nDetailed Metrics (First Trial):")
            metrics = evaluate_model_performance(adapted_model, Xt_test, yt_test)
            for metric, value in metrics.items():
                print(f"\n{metric}:\n{value}")
    
    # Visualisation finale
    print("\nGenerating final visualization...")
    visualize_results(source_scores, target_scores, adapted_scores)
    
    # Statistiques finales
    print("\nFinal Statistics:")
    print(f"Average Source Accuracy: {np.mean(source_scores):.4f} ± {np.std(source_scores):.4f}")
    print(f"Average Target Accuracy: {np.mean(target_scores):.4f} ± {np.std(target_scores):.4f}")
    print(f"Average Adapted Accuracy: {np.mean(adapted_scores):.4f} ± {np.std(adapted_scores):.4f}")

if __name__ == "__main__":
    main()

