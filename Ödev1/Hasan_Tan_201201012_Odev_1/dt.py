#%% Karar ağacı için tanımladığım class yapısı
class DecisionTreeClassifier:
#%% Karar ağacının başlangıç değerlerini atayan init methodu
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
#%% Karar ağacının asıl oluşması için gereken methodu çağırmayı sağlayan ve train verilerini fit eden method
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
#%% Bu class'ta tanımlanan diğer methodları da kullanarak karar ağacının özyinelemeli şekilde meydana gelmesini sağlayan method   
    def build_tree(self, X, y, depth=0):
        # Stopping conditions
        if self.max_depth is not None and depth >= self.max_depth:
            return self.get_leaf_node(y)
        if len(self.unique(y)) == 1:
            return self.get_leaf_node(y)
         # Find the best splitting criterion and Create the node
        node, best_feature_index, best_threshold = self.create_node(X,y)
        # Create the left and right child nodes
        left_X, left_y, right_X, right_y = self.split_node(X, y, best_feature_index, best_threshold)
        node['left'] = self.build_tree(left_X, left_y, depth + 1)
        node['right'] = self.build_tree(right_X, right_y, depth + 1)

        return node
#%% Karar ağacının yapraklarını meydana getiren, en son değerin belirlenmesini sağlayan method   
    def get_leaf_node(self, y):
        counts = self.count_values(y)
        dominant_class = counts.index(max(counts))

        leaf_node = {
            'class': dominant_class,
            'count': len(y)
        }
        return leaf_node
#%% Test verisini kullanarak modele tahmin ettirmeye yarayan method. Yapısında ağacı özyinelemeli olarak takip eden başka bir methodu bulundurmaktadır.   
    def predict(self, X):
        predictions = []
        for sample in X:
            prediction = self.traverse_tree(sample, self.tree)
            predictions.append(prediction)
        return predictions
#%% Ağacı test verisinin çıktısını tahmin etmeye yarayacak şekilde özyinelemeli olarak takip etmeye yarayan method.   
    def traverse_tree(self, sample, node):
        if 'class' in node:
            return node['class']
        else:
            feature_index = node['feature_index']
            threshold = node['threshold']
            if sample[feature_index] <= threshold:
                return self.traverse_tree(sample, node['left'])
            else:
                return self.traverse_tree(sample, node['right'])
#%% Ağacı görselleştirmeye yarayan method yapısında başka bir methodun kullanılmasını içermektedir.           
    def print_tree(self):
        self.print_node(self.tree)
#%% Ağacın bulunulan nodeundaki threshold değerlerini, hangi featureın kullanıldığını görselleştirmeye yarayan, ayrılan dalları görselleştirmeye yarayan method
    def print_node(self, node, depth=0):
        indent = "       " * depth
        if 'class' in node:
            print(indent + "Class:", node['class'])
        else:
            feature_index = node['feature_index']
            threshold = node['threshold']
            print(indent + "Feature", feature_index, "<=", threshold)
            print(indent + "--> True:")
            self.print_node(node['left'], depth + 1)
            print(indent + "--> False:")
            self.print_node(node['right'], depth + 1)
#%% Gini indexini hesaplamaya yarayan method
    def calculate_gini(self, y):
        classes = self.unique(y)
        gini = 1.0

        for cls in classes:
            p = y.count(cls) / len(y)
            gini -= p ** 2

        return gini
#%% Node ayrımını kontrol eden ve node threshold ve feature değerlerine göre veriyi ikiye bölen method   
    def split_node(self, X, y, feature_index, threshold):
        left_X, left_y, right_X, right_y = [], [], [], []

        for i in range(len(X)):
            if X[i][feature_index] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        
        return left_X, left_y, right_X, right_y 
#%% Veride ne kadar hangi sınıftan olduğunu döndüren method. Leaf nodeundaki baskın sınıfı belirlemeye yarıyor.   
    def count_values(self,column):
        count_0 = 0
        count_1 = 0
        count_2 = 0
        for value in column:
            if value == 0.0:
                count_0 += 1
            elif value == 1.0:
                count_1 += 1
            elif value == 2.0:
                count_2 += 1
        counts = [count_0,count_1,count_2]
        return counts
#%% Hedef veri içindeki sınıfların belirlenmesini sağlayan method   
    def unique(self, input_list):
        unique = []
        for sublist in input_list:
            if sublist not in unique:
                unique.append(sublist)
        return unique
#%% İç içe for döngüleri ile en düşük gini veren threshold değerine ve feature değerine göre node oluşturmayı sağlayan method. Ağacın oluşmasını temel olarak bu method sağlamaktadır     
    def create_node(self,X,y):
        best_gini = float('inf')
        best_feature_index = 0
        best_threshold = 0
        for feature_index in range(len(X[0])):
            thresholds = self.unique([row[feature_index]for row in X])
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = self.split_node(X, y, feature_index, threshold)
                gini = (len(left_y) * self.calculate_gini(left_y) + len(right_y) * self.calculate_gini(right_y)) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold
        node = {
            'feature_index': best_feature_index,
            'threshold': best_threshold,
            'left': None,
            'right': None
        }
        return node, best_feature_index, best_threshold
#%%



    

    