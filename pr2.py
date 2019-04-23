from shared_imports import *

# problem 2 part 1 - use a spectral embedding method (or two) to embed the MNIST data and evaluate
#                    how well distances are preserved

# embed X_train and report the similarity of its pairwise similarities/distances with those in the original
# space
def report_embedding(X, n_components, csim_hd, embed_method, sim_dist_method):
    print("    Embedding dimension: " + str(n_components))
    (X_transformed, mean_measure, std_measure, corr_measure) = embed_and_correlate(X_train, n_components, csim_hd, embed_method=embed_method, sim_dist_method=sim_dist_method)
    if sim_dist_method != "KNN":
        print("      Mean of low-dim similarities/distances: " + str(round(mean_measure, round_tol)))
        print("      Std dev of low-dim similarities/distances: " + str(round(std_measure, round_tol)))
    print("      Correlation of similarities/distances: " + str(round(corr_measure, round_tol)))

(X_train, y_train, X_test, y_test) = load_mnist(subsample_no=1000)

print("Original dimensionality is: " + str(X_train.shape[1]))


'''
sim_dist_method_array = ["cosine", "Euclidean distance", "KNN"]
csim_hd_array = {}
for sim_dist_method in sim_dist_method_array:
    csim_hd_array[sim_dist_method] = measure(X_train, sim_dist_method=sim_dist_method)

# top loop, for efficiency: embedding methods
# then loop similarity/distance measures
# then loop embedding dimensions (component number)

# in practice this is super difficult to parse!

component_sweep = [784, 400, 200, 100, 50, 10, 3]
# component_sweep = [10]

for embed_method in ["PCA", "Laplacian eigenmaps"]:
    print("Embedding method: " + embed_method)
    for sim_dist_method in sim_dist_method_array:
        print("  Similarity/distance: " + sim_dist_method)
        csim_hd = csim_hd_array[sim_dist_method]
        if sim_dist_method != "KNN":
            csim_hd_vec = vectorize_measure(csim_hd)
            print("  Mean of high-dim similarity/distance: " + str(round(np.mean(csim_hd_vec), round_tol)))
            print("  Std dev of high-dim similarity/distance: " + str(round(np.std(csim_hd_vec), round_tol)))
        for n_components in component_sweep:
            report_embedding(X_train, n_components, csim_hd, embed_method, sim_dist_method)

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

encoder = vae_embed(X_train, X_test, latent_dim=3, epochs=20)