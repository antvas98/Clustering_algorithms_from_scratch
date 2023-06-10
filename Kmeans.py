import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, sigmoid_kernel
from itertools import compress
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, sigmoid_kernel
import matplotlib.pyplot as plt


class Standarized_Kmeans:
    def __init__(self,num_clusters):
        self.k = num_clusters

    def fit(self,data,maxiter=100):
        # Initialize centroids
        centroids = np.zeros([self.k, data.shape[1]])
        random_indices = np.random.choice(data.shape[0], size=self.k, replace=False)
        K=self.k
        for j in range(K):
            centroids[j] = data[random_indices[j]]

        # Initialize clusters
        Clusters = [[] for _ in range(K)]
        upd_centroids = np.zeros([K, data.shape[1]])
        iterations = 0
        while iterations < maxiter:
            iterations += 1
            print(f"Iteration number: {iterations}")
            centroids = upd_centroids.copy()

            # Assign observations to clusters
            dist_obs_cent = np.zeros((data.shape[0], K))
            for i in range(K):
                dist_obs_cent[:, i] = np.sum((data - centroids[i]) ** 2, axis=1)
            cluster_indices = np.argmin(dist_obs_cent, axis=1)
            Clusters = [[] for _ in range(K)]
            for j, cluster_index in enumerate(cluster_indices):
                Clusters[cluster_index].append(j)

            # Update centroids
            for j in range(K):
                if Clusters[j]:
                    upd_centroids[j] = np.mean(data[Clusters[j]], axis=0)

            # Check convergence
            # if np.sum((centroids - upd_centroids) ** 2) < tol:
            if (centroids == upd_centroids).all():
                print("Converged!")
                self.Clusters = Clusters
                return self.Clusters

        print("Max iterations reached without convergence.")
        self.Clusters = Clusters
        return Clusters
    
    @staticmethod
    def evaluate(Clusters, y):
        # Find the majority label for each cluster
        cluster_labels = []
        for cluster in Clusters:
            labels = y[cluster]
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            majority_label = unique_labels[np.argmax(label_counts)]
            cluster_labels.append(majority_label)

        # Assign the majority labels to each data point in the clusters
        assigned_labels = np.zeros_like(y)
        for i, cluster in enumerate(Clusters):
            assigned_labels[cluster] = cluster_labels[i]

        # Calculate accuracy by comparing assigned labels with true labels
        accuracy = np.mean(assigned_labels == y) * 100

        return accuracy

    @staticmethod
    def visualize(Clusters, X, num_samples=10):
        num_classes = len(Clusters) 

        fig, axs = plt.subplots(num_classes, num_samples, figsize=(num_samples, num_classes))

        # Iterate over each class
        for c in range(num_classes):
            class_samples = Clusters[c]  # Samples belonging to the current class

            # Select the specified number of samples from the class
            for i in range(num_samples):
                ind = class_samples[i]
                pixels = X[ind].reshape((28, 28))

                # Plot the image in the corresponding subplot
                axs[c, i].imshow(pixels, cmap='gray')
                axs[c, i].axis('off')

        # Adjust spacing between subplots
        fig.tight_layout()

        # Display the plot
        plt.show()

class Kernelized_Kmeans:
    def __init__(self,num_clusters, kernel='linear', gamma_gaussian=10**(-7), gamma_sigmoid=0.0045, coef_sigmoid=0.11):
        self.k = num_clusters
        self.kernel = kernel 
        self.gamma_gaussian = gamma_gaussian
        self.gamma_sigmoid = gamma_sigmoid
        self.coef_sigmoid = coef_sigmoid


    def fit(self, X, maxiter=100):

        if self.kernel == 'linear':
            KernelMatrix = (linear_kernel(X/255)+1)*5
        if self.kernel == 'gaussian':
            KernelMatrix = rbf_kernel(X, gamma=self.gamma_gaussian)  
        if self.kernel == 'sigmoid':
            KernelMatrix = sigmoid_kernel(X/250, gamma=self.gamma_sigmoid,coef0=self.coef_sigmoid)

        n = X.shape[0]
        K=self.k

        # Randomly initialize clusters
        indcs = np.random.choice(range(K), n)
        
        for l in range(maxiter): 
            print(f"Iteration number: {l+1}")  # Current iteration

            # Compute number of observations per cluster
            obser_per_cluster = np.bincount(indcs, minlength=K)
            
            # Compute distances between obs and centers
            distances = np.zeros((n, K))
            for k in range(K):
                ptr_cluster = np.where(indcs == k)[0]
                thirdterm = KernelMatrix[ptr_cluster][:, ptr_cluster].sum() / (obser_per_cluster[k] ** 2)
                secondterm = (-2 * KernelMatrix[:, ptr_cluster].sum(axis=1)) / obser_per_cluster[k]
                distances[:, k] = secondterm + thirdterm

            # Assign new clusters
            upd_indcs = np.argmin(distances, axis=1)

            if (upd_indcs == indcs).all():
                self.Clusters = [[] for _ in range(K)]
                for i, cluster_idx in enumerate(upd_indcs):
                    self.Clusters[cluster_idx].append(i)
                print("Converged!")
                return self.Clusters

            indcs = upd_indcs.copy()

            if l == maxiter-1:
                print("Max iterations reached without convergence.")
        
        self.Clusters = [[] for _ in range(K)]
        for i, cluster_idx in enumerate(upd_indcs):
            self.Clusters[cluster_idx].append(i)

        return self.Clusters
    
    @staticmethod
    def evaluate(Clusters, y):
        # Find the majority label for each cluster
        cluster_labels = []
        for cluster in Clusters:
            labels = y[cluster]
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            majority_label = unique_labels[np.argmax(label_counts)]
            cluster_labels.append(majority_label)

        # Assign the majority labels to each data point in the clusters
        assigned_labels = np.zeros_like(y)
        for i, cluster in enumerate(Clusters):
            assigned_labels[cluster] = cluster_labels[i]

        # Calculate accuracy by comparing assigned labels with true labels
        accuracy = np.mean(assigned_labels == y) * 100

        return accuracy

    @staticmethod
    def visualize(Clusters, X, num_samples=10):
        num_classes = len(Clusters) 

        fig, axs = plt.subplots(num_classes, num_samples, figsize=(num_samples, num_classes))

        # Iterate over each class
        for c in range(num_classes):
            class_samples = Clusters[c]  # Samples belonging to the current class

            # Select the specified number of samples from the class
            for i in range(num_samples):
                ind = class_samples[i]
                pixels = X[ind].reshape((28, 28))

                # Plot the image in the corresponding subplot
                axs[c, i].imshow(pixels, cmap='gray')
                axs[c, i].axis('off')

        # Adjust spacing between subplots
        fig.tight_layout()

        # Display the plot
        plt.show()



