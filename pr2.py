from shared_imports import *

# problem 2 part 1 - use a spectral embedding method (or two) to embed the MNIST data and evaluate
#                    how well distances are preserved

(X_train, y_train, X_test, y_test) = load_mnist(subsample_no=1000)

print("Original dimensionality is: " + str(X_train.shape[1]))

csim_hd = measure(X_train, sim_dist_method="cosine")
csim_hd_vec = vectorize_measure(csim_hd)
print("Mean of high-dim cosine similarity: " + str(round(np.mean(csim_hd_vec), round_tol)))
print("Std dev of high-dim cosine similarity: " + str(round(np.std(csim_hd_vec), round_tol)) + "\n")

component_sweep = [784, 400, 200, 100, 50, 10, 3]

# evaluate distances over a sweep of lower dimensional embeddings
for n_components in component_sweep:
    for embed_method in ["PCA", "Laplacian eigenmaps"]:
        sim_dist_method = "cosine"
        # sim_dist_method = "KNN"
        print("Embedding in " + str(n_components) + " dimensions via " + embed_method)
        (X_transformed, mean_measure, std_measure, corr_measure) = embed_and_correlate(X_train, n_components, csim_hd, embed_method=embed_method)
        print("\tMean of low-dim similarities/distances: " + str(round(mean_measure, round_tol)))
        print("\tStd dev of low-dim similarities/distances: " + str(round(std_measure, round_tol)))
        print("\tCorrelation of similarities/distances: " + str(round(corr_measure, round_tol)) + "\n")

'''
# plot the 3D embedding (PCA)
(X_transformed, mean_measure, std_measure, corr_measure) = embed_and_correlate(X_train, 3, csim_hd, embed_method="PCA")
df = pd.DataFrame({'X': X_transformed[:,0], 'Y': X_transformed[:,1], 'Z': X_transformed[:,2], 'label': y_train })

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['X'], df['Y'], df['Z'], c=df['label'], cmap='viridis', s=20)
ax.view_init(30, 60)
plt.savefig("plots/PCA_embedding.png")

# plot the 3D embedding (Laplacian eigenmaps)
(X_transformed, mean_measure, std_measure, corr_measure) = embed_and_correlate(X_train, 3, csim_hd, embed_method="Laplacian eigenmaps")
df = pd.DataFrame({'X': X_transformed[:,0], 'Y': X_transformed[:,1], 'Z': X_transformed[:,2], 'label': y_train })

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['X'], df['Y'], df['Z'], c=df['label'], cmap='viridis', s=20)
ax.view_init(30, 60)
plt.savefig("plots/Laplacian_eigenmaps_embedding.png")
'''
# problem 2 part 2 - rinse, repeat with a variational autoencoder