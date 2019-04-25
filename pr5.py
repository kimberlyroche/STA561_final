from shared_imports import *

# testing
subsample_no = 5000
batch_size = 10
epochs = 100

# full
#subsample_no = 5000
#batch_size = 128
#epochs = 100

# get similarity measure from user
method = "cnn"
if(len(sys.argv) > 1):
    if sys.argv[1] == "gpr":
        method = "GPR"

(X_GP, X_CNN, y) = load_cifar10(subsample_no=subsample_no)

no_samples = X_GP.shape[0]

print("Method: " + method)
print("Subsampling to " + str(subsample_no))

error_GP = 0.0 # sum of misclassifications (GP)
error_CNN = 0.0 # sum of misclassifications (CNN)
time_GP = 0.0 # total runtime (GP)
time_CNN = 0.0 # total runtime (CNN)
loo = LeaveOneOut()
loo.get_n_splits(X_GP)
it = 1
iter_limit = 100
for train_index, test_index in loo.split(X_GP):
    if it <= iter_limit:
        print("Evaluating split #" + str(it))
        it += 1
        y_train, y_test = y[train_index], y[test_index]
        if method == "GPR":
            X_train, X_test = X_GP[train_index], X_GP[test_index]
            start = time.time()
            y_pred = predict_GP(X_train, y_train, X_test)
            end = time.time()
            time_GP += end - start
            error_GP += one_hot_loss(y_test[0], y_pred[0])
        else:
            X_train, X_test = X_CNN[train_index], X_CNN[test_index]
            start = time.time()
            y_pred = predict_CNN_classification(X_train, y_train, X_test, batch_size=batch_size, epochs=epochs, verbose=1)
            end = time.time()
            time_CNN += end - start
            error_CNN += one_hot_loss(y_test[0], y_pred.tolist())

if method == "GPR":
    print("Misclassification percent (GP): " + str(error_GP/iter_limit))
    print("Average runtime (GP): " + str(round(time_GP/iter_limit, round_tol)) + " sec")
else:
    print("Misclassification percent (CNN): " + str(error_CNN/iter_limit))
    print("Average runtime (CNN): " + str(round(time_CNN/iter_limit, round_tol)) + " sec")
