import random
#%% KMeans için tanımladığım class yapısı
class KMeansClusterClassifier:
    #%% KMeans başlangıç değerlerini atayan init methodu
    def __init__(self, n_cluster=3, max_iterations=100):
        self.num_clusters = n_cluster
        self.max_iterations = max_iterations
        self.cluster_data = [[] for _ in range(n_cluster)]
        self.centroids = []
    #%% KMeans için gerekli olan fit methodu
    def fit(self, X, y):
        if len(X) < 1:
            return

        labelled_data = self.labelize(X, y)
        self.centroids, self.cluster_data = self.calculate_clusters_and_centers(labelled_data)
    #%% Unsupervised model olan KMeans için label ekleme methodu
    def labelize(self, X, y):
        labelled_data = []
        for i in range(len(y)):
            labelled_data.append(X[i] + [y[i]])
        return labelled_data
    #%% KMeans için gerekli olan cluster ve center hesaplamalarını yapan method
    def calculate_clusters_and_centers(self, data):
        centers = []
        for _ in range(self.num_clusters):
            centers.append(data[random.randint(0, len(data) - 1)])
        for _ in range(self.max_iterations):
            clusters = [[] for _ in range(self.num_clusters)]
            for point in data:
                closest_cluster = self.find_closest_cluster(point, centers)
                clusters[closest_cluster].append(point)
            for num, _ in enumerate(centers):
                if len(clusters[num]) > 0:
                    centers[num] = self.calculate_averages(clusters[num])
        return centers, clusters
    #%% Her data noktasına en yakın cluster bulmayı sağlayan method
    def find_closest_cluster(self, point, centers):
        closest_cluster = 0
        closest_distance = float('inf')
        for idx, center in enumerate(centers):
            distance = self.distance(point, center)
            if distance < closest_distance:
                closest_distance = distance
                closest_cluster = idx
        return closest_cluster
    #%% Eucledean distance hesaplamasını yapan method
    def distance(self, point1, point2):
        return sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5
    #%% Yeni merkes için ortalama hesaplaması yapan method
    def calculate_averages(self, cluster_data):
        num_dimensions = len(cluster_data[0]) - 1
        new_center = [0] * num_dimensions
        for i in range(num_dimensions):
            total = sum(row[i] for row in cluster_data)
            new_center[i] = total / len(cluster_data)
        return new_center
    #%% Yeni veriler için tahmin yapmayı sağlayan method
    def predict(self, X):
        predictions = []
        for point in X:
            closest_cluster = self.find_closest_cluster(point + [-1], self.centroids)
            most_common_label = self.count_label([row[-1] for row in self.cluster_data[closest_cluster]])
            predictions.append(most_common_label)
        return predictions
    #%% Cluster içinde en çok tekrar eden labelı bulmayı sağlayan method
    def count_label(self, labels):
        label_counts = {}
        for idx in labels:
            if idx in label_counts:
                label_counts[idx] += 1
            else:
                label_counts[idx] = 1
        return max(label_counts, key=label_counts.get)
