#%% gerekli kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%% data okunması ve anlamsız olan Id sütununun atılması 
data = pd.read_csv('C:/Users/HasanTan/Desktop/Ödev1/Iris.csv')
data = data.drop('Id', axis=1)
features = set(data)
#%% tür isimlerinin dönüştürülmesi
class_mapping = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}
data['Species'] = data['Species'].map(class_mapping)
#%% Exploratory Data Analysis 
data_correlation = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(data_correlation, annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()


sns.pairplot(data, hue="Species")
plt.title("Pair Plot")
plt.show()
#%%
def calculate_gini(data_feature):
    classes = set(data_feature)
    gini = 1.0
    print(classes)
    for cls in classes:
        p = len(data_feature[data_feature == cls]) / len(data_feature)
        gini -= p ** 2
    return gini
#%%
def split_node(X, y, feature_index, threshold):
    left_X, left_y = X[X[:, feature_index] <= threshold], y[X[:, feature_index] <= threshold]
    right_X, right_y = X[X[:, feature_index] > threshold], y[X[:, feature_index] > threshold]
    return left_X, left_y, right_X, right_y
#%%
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        # Durma koşullarını kontrol et
        if self.max_depth is not None and depth >= self.max_depth:
            return self.get_leaf_node(y)

        if len(set(y)) == 1:
            return self.get_leaf_node(y)

        # En iyi bölme kriterini bul
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = split_node(X, y, feature_index, threshold)
                gini = (len(left_y) * calculate_gini(left_y) + len(right_y) * calculate_gini(right_y)) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        # Düğümü oluştur
        node = {
            'feature_index': best_feature_index,
            'threshold': best_threshold,
            'left': None,
            'right': None
        }

        # Sol ve sağ alt düğümleri oluştur
        left_X, left_y, right_X, right_y = split_node(X, y, best_feature_index, best_threshold)
        node['left'] = self.build_tree(left_X, left_y, depth + 1)
        node['right'] = self.build_tree(right_X, right_y, depth + 1)

        return node

    def get_leaf_node(self, y):
        values, counts = np.unique(y, return_counts=True)
        dominant_class = values[np.argmax(counts)]

        leaf_node = {
            'class': dominant_class,
            'count': len(y)
            }
        return leaf_node
#%%