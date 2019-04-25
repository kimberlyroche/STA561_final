from shared_imports import *

# testing
#subsample_no = 50
#batch_size = 10
#epochs = 10

# full
subsample_no = 0
batch_size = 10
epochs = 100

# problem 1 part 1 - compute LOO error for GP regression and CNN regression

# note: should take about 15-20 minutes to perform LOO CV on both model
# before doing a full run, updated epochs to 100+
# may want to try non-linear kernel

(X, y) = load_lab1_data(subsample_no=subsample_no)

no_samples = X.shape[0]

SSE_GP = 0.0 # sum of squared errors (GP)
SSE_CNN = 0.0 # sum of squared errors (CNN)
time_GP = 0.0 # total runtime (GP)
time_CNN = 0.0 # total runtime (CNN)
loo = LeaveOneOut()
loo.get_n_splits(X)
it = 1
for train_index, test_index in loo.split(X):
    if it % 50 == 0:
        print("Evaluating split #" + str(it))
    it += 1
    # subset train & test
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # GP
    start = time.time()
    y_pred = predict_GP(X_train, y_train, X_test)
    end = time.time()
    time_GP += end - start
    err = (y_pred[0,0] - y_test[0,0])**2
    SSE_GP += err
    # CNN
    start = time.time()
    y_pred = predict_CNN_regression(X_train, y_train, X_test, batch_size=batch_size, epochs=epochs)
    end = time.time()
    time_CNN += end - start
    err = (y_pred - y_test[0,0])**2
    SSE_CNN += err

print("RMSE (GP): " + str(round(np.sqrt(SSE_GP/no_samples), round_tol)))
print("Average runtime (GP): " + str(round(time_GP/no_samples, round_tol)) + " sec")

print("RMSE (CNN): " + str(round(np.sqrt(SSE_CNN/no_samples), round_tol)))
print("Average runtime (CNN): " + str(round(time_CNN/no_samples, round_tol)) + " sec")

# problem 1 part 2 - predictve (posterior) samples

omissions = [0, 49, 99, 149] # sample indices 1, 50, 100, 150

for omit in omissions:
    retain = [x for x in range(no_samples) if x != omit]

    X_train = X[retain].reshape((no_samples-1), 3)
    X_test = X[omit].reshape(1, 3)
    y_train = y[retain].reshape((no_samples-1), 1)
    y_test = y[omit].reshape(1, 1)

    # get the GP prediction + uncertainty (=normal posterior)
    y_pred, sigma = predict_GP(X_train, y_train, X_test, return_sigma=True)
    no_posterior_samples = 1000
    white_noise = np.random.normal(y_pred[0], sigma, no_posterior_samples)
    posterior_samples = []
    for i in range(no_posterior_samples):
        posterior_samples.append(y_pred[0,0] + sigma[0]*white_noise[i])

    sns.set_style('whitegrid')
    print("True value for sample " + str(omit+1) + ": " + str(round(y_test[0,0], round_tol)))
    print("\tMean, std for GP: " + str(round(y_pred[0,0], round_tol)) + ", " + str(round(sigma[0], round_tol)))
    dens_fig, length_ax = plt.subplots()
    sns.kdeplot(np.array(posterior_samples), bw=0.5)
    plt.savefig("plots/posterior_" + str(omit+1) + ".png")
    
    # get the CNN prediction
    y_pred = predict_CNN_regression(X_train, y_train, X_test, batch_size=batch_size, epochs=epochs)
    print("\tPrediction from CNN: " + str(round(float(y_pred), round_tol)))
