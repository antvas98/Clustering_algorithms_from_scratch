from Kmeans import Standarized_Kmeans, Kernelized_Kmeans
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, sigmoid_kernel
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import numpy as np

class Spectral_clustering:

    def __init__(self, num_clusters, num_eigen = 3, gamma_gaussian = 10**(-7), epsilon = 0.515):
        self.gamma = gamma_gaussian
        self.num_clusters = num_clusters
        self.num_eigen = num_eigen
        self.epsilon = epsilon
    
    def fit(self, X, normalized = False, maxiter=100):

        GramMatrix = rbf_kernel(X,gamma=self.gamma) #the Gaussian kernel matrix
        graphW=np.where(GramMatrix> self.epsilon,1,0) 

        #Create D matrix 
        D=np.diag(np.sum(graphW,axis=1)) #diagonal entries of D contain the degree information of nodes in the graph
        #convert sparse matrices to csr format to make multiplications faster
        csr_graphW=csr_matrix(graphW)
        csr_D=csr_matrix(D)

        #Create the unormalized Laplacian L (in lecture notes notation)
        csr_L=csr_D-csr_graphW
        csr_L=csr_L.asfptype()  
        
        if normalized:

            #Create the normalized Laplacian Lbar (in lecture notes notation)
            diag_to_minus_half=np.diag(D)**(-1/2)
            D_to_minus_half=np.diag(diag_to_minus_half)

            csr_D_to_minus_half=csr_matrix(D_to_minus_half) #convert to csr to speed up the operations


            csr_Lbar= csr_D_to_minus_half @ csr_L @ csr_D_to_minus_half 


            #Create H for the normalized case (columns of H are the K eigenvectors of Lbar corresponding to K minimal eigenvalues)
            Lbar_eigenValues, Lbar_eigenVectors =eigsh(csr_Lbar,k=self.num_eigen)

            idx = Lbar_eigenValues.argsort()[::1]   
            Lbar_eigenValues = Lbar_eigenValues[idx]
            Lbar_eigenVectors = Lbar_eigenVectors[:,idx]

            H_Lbar=Lbar_eigenVectors[:,0:self.num_eigen]
            self.H = H_Lbar
        
        else:

            #Create H for the unormalized case
            L_eigenValues, L_eigenVectors =eigsh(csr_L,k=self.num_eigen)

            idx = L_eigenValues.argsort()[::1]   
            L_eigenValues = L_eigenValues[idx]
            L_eigenVectors = L_eigenVectors[:,idx]

            H_L = L_eigenVectors[:,0:self.num_eigen]
            self.H = H_L
        
        model = Standarized_Kmeans(num_clusters = self.num_clusters)
        self.Clusters = model.fit(self.H, maxiter=maxiter)
    
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




