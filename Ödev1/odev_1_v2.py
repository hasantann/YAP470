class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        # Stopping conditions
        if self.max_depth is not None and depth >= self.max_depth:
            return self.get_leaf_node(y)
        if len(set(y)) == 1:
            return self.get_leaf_node(y)

        # Find the best splitting criterion
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None
        for feature_index in range(X.shape[1]):
            thresholds = self.unique_values(X.iloc[:, feature_index])
            print(thresholds)
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = self.split_node(X, y, feature_index, threshold)
                gini = (len(left_y) * self.calculate_gini(left_y) + len(right_y) * self.calculate_gini(right_y)) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold
        print("best_feature_index: ", best_feature_index, "best_gini:", best_gini, "best_threshold", )
        # Create the node
        node = {
            'feature_index': best_feature_index,
            'threshold': best_threshold,
            'left': None,
            'right': None
        }

        # Create the left and right child nodes
        left_X, left_y, right_X, right_y = self.split_node(X, y, best_feature_index, best_threshold)
        node['left'] = self.build_tree(left_X, left_y, depth + 1)
        node['right'] = self.build_tree(right_X, right_y, depth + 1)

        return node

    def get_leaf_node(self, y):
        counts = self.count_values(y)
        dominant_class = max(counts, key=counts.get)

        leaf_node = {
            'class': dominant_class,
            'count': len(y)
        }
        return leaf_node
    
    def predict(self, X):
        predictions = []
        for sample in X:
            prediction = self.traverse_tree(sample, self.tree)
            predictions.append(prediction)
            print("Sample:", sample, "Prediction:", prediction)
        return predictions

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

    @staticmethod
    def calculate_gini(y):
        classes = set(y)
        gini = 1.0

        for cls in classes:
            p = y.count(cls) / len(y)
            gini -= p ** 2

        return gini

    @staticmethod
    def split_node(X, y, feature_index, threshold):
        left_X, left_y, right_X, right_y = [], [], [], []

        for i in range(len(X)):
            if X.iloc[i,feature_index] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return left_X, left_y, right_X, right_y

    @staticmethod
    def unique_values(column):
        return list(set(column))

    @staticmethod
    def count_values(column):
        counts = {}
        for value in column:
            if value not in counts:
                counts[value] = 0
            counts[value] += 1
        return counts
