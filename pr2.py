from shared_imports import *

# load data
(X_train, y_train, X_test, y_test) = load_mnist(subsample_no=1000)

print("Original dimensionality is: " + str(X_train.shape[1]))

csim_hd_vec = measure(X_train, sim_dist_method="cosine")
print("Mean of high-dim cosine similarity: " + str(round(np.mean(csim_hd_vec), round_tol)))
print("Std dev of high-dim cosine similarity: " + str(round(np.std(csim_hd_vec), round_tol)) + "\n")

component_sweep = [784, 400, 200, 100, 50, 10, 3]

for n_components in component_sweep:
    embed_method = "PCA"
    print("Embedding in " + str(n_components) + " dimensions via " + embed_method)
    (X_transformed, csim_ld_vec) = embed_and_measure(X_train, n_components, embed_method=embed_method)
    print("\tMean of low-dim similarities/distances: " + str(round(np.mean(csim_ld_vec), round_tol)))
    print("\tStd dev of low-dim similarities/distances: " + str(round(np.std(csim_ld_vec), round_tol)))
    print("\tCorrelation of similarities/distances: " + str(round(np.corrcoef(csim_hd_vec, csim_ld_vec)[0,1], round_tol)) + "\n")

for n_components in component_sweep:
    embed_method = "Laplacian eigenmaps"
    print("Embedding in " + str(n_components) + " dimensions via " + embed_method)
    (X_transformed, csim_ld_vec) = embed_and_measure(X_train, n_components, embed_method=embed_method)
    print("\tMean of low-dim similarities/distances: " + str(round(np.mean(csim_ld_vec), round_tol)))
    print("\tStd dev of low-dim similarities/distances: " + str(round(np.std(csim_ld_vec), round_tol)))
    print("\tCorrelation of similarities/distances: " + str(round(np.corrcoef(csim_hd_vec, csim_ld_vec)[0,1], round_tol)) + "\n")