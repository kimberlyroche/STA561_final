from shared_imports import *

# problem 1 part 1 - compute LOO error for GP regression and CNN regression

subsample_no = 50

(X, y) = load_lab1_data(subsample_no=subsample_no)

no_samples = X.shape[0]

SSE_GP = 0.0 # sum of squared errors (GP)
SSE_CNN = 0.0 # sum of squared errors (CNN)
loo = LeaveOneOut()
loo.get_n_splits(X)
it = 1
for train_index, test_index in loo.split(X):
    if it % 10 == 0:
	    print("Evaluating split #" + str(it))
    it += 1
    # subset train & test
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # GP
    y_pred = predict_GP(X_train, y_train, X_test)
    err = (y_pred[0,0] - y_test[0,0])**2
    SSE_GP += err
    # CNN
    y_pred = predict_CNN_regression(X_train, y_train, X_test, batch_size=10)
    err = (y_pred - y_test[0,0])**2
    SSE_CNN += err

print("RMSE (GP): " + str(np.sqrt(SSE_GP/no_samples)))
print("RMSE (CNN): " + str(np.sqrt(SSE_CNN/no_samples)))