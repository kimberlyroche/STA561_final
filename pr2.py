from shared_imports import *

# testing
subsample_no = 1000
epochs = 10

# full
subsample_no = 0
epochs = 100

# get similarity measure from user
sim_dist_method = "cosine"
if(len(sys.argv) > 1):
    if sys.argv[1] == "euclid":
        sim_dist_method = "Euclidean distance"
    elif sys.argv[1] == "knn":
        sim_dist_method = "KNN"

# problem 2 part 1 - use a spectral embedding method (or two) to embed the MNIST data and evaluate
#                    how well distances are preserved

# problem 2 part 2 - rinse, repeat with a variational autoencoder; we'll combine all into one script

# embed X_train and report the similarity of its pairwise similarities/distances with those in the original
# space
def report_embedding(X_train, y_train, X_test, y_test, n_components, csim_hd, embed_method, sim_dist_method):
    print("  Embedding dimension: " + str(n_components))
    (X_transformed, mean_measure, std_measure, corr_measure) = embed_and_correlate(X_train, y_train, X_test, \
        y_test, n_components, csim_hd, embed_method=embed_method, sim_dist_method=sim_dist_method, epochs=epochs)
    if sim_dist_method != "KNN":
        print("    Mean of low-dim similarities/distances: " + str(round(mean_measure, round_tol)))
        print("    Std dev of low-dim similarities/distances: " + str(round(std_measure, round_tol)))
    print("    Correlation of similarities/distances: " + str(round(corr_measure, round_tol)))

(X_train, y_train, X_test, y_test) = load_mnist(subsample_no=subsample_no)

print("Original dimensionality is: " + str(X_train.shape[1]))

csim_hd = measure(X_train, sim_dist_method=sim_dist_method)

# top loop, for efficiency: embedding methods
# then loop similarity/distance measures
# then loop embedding dimensions (component number)

# in practice this is super difficult to parse!
component_sweep = [784, 400, 200, 100, 50, 10, 3]

for embed_method in ["PCA", "Laplacian eigenmaps", "VAE"]:
    print("Embedding method: " + embed_method)
    print("Similarity/distance: " + sim_dist_method)
    if sim_dist_method != "KNN":
        csim_hd_vec = vectorize_measure(csim_hd)
        print("  Mean of high-dim similarity/distance: " + str(round(np.mean(csim_hd_vec), round_tol)))
        print("  Std dev of high-dim similarity/distance: " + str(round(np.std(csim_hd_vec), round_tol)))
    for n_components in component_sweep:
        report_embedding(X_train, y_train, X_test, y_test, n_components, csim_hd, embed_method, sim_dist_method)

# plot the 3D embedding (PCA)
X_transformed = embed(X_train, y_train, X_test, y_test, 3, embed_method="PCA")
df = pd.DataFrame({'X': X_transformed[:,0], 'Y': X_transformed[:,1], 'Z': X_transformed[:,2], 'label': y_train })

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['X'], df['Y'], df['Z'], c=df['label'], cmap='viridis', s=20)
ax.view_init(30, 60)
plt.savefig("plots/PCA_embedding.png")

# plot the 3D embedding (Laplacian eigenmaps)
X_transformed = embed(X_train, y_train, X_test, y_test, 3, embed_method="Laplacian eigenmaps")
df = pd.DataFrame({'X': X_transformed[:,0], 'Y': X_transformed[:,1], 'Z': X_transformed[:,2], 'label': y_train })

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['X'], df['Y'], df['Z'], c=df['label'], cmap='viridis', s=20)
ax.view_init(30, 60)
plt.savefig("plots/Laplacian_eigenmaps_embedding.png")

# plot the 3D embedding (VAE)
X_transformed = embed(X_train, y_train, X_test, y_test, 3, embed_method="VAE", epochs=epochs)
df = pd.DataFrame({'X': X_transformed[:,0], 'Y': X_transformed[:,1], 'Z': X_transformed[:,2], 'label': y_train })

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['X'], df['Y'], df['Z'], c=df['label'], cmap='viridis', s=20)
ax.view_init(30, 60)
plt.savefig("plots/VAE_embedding.png")
