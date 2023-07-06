# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

# Encoding of data


def categorical_encoding(data):
    data.replace('Iris-setosa', 0, inplace=True)
    data.replace('Iris-versicolor', 1, inplace=True)
    data.replace('Iris-virginica', 2, inplace=True)

# helper function


def sample_split(data, split_size):
    train_data = data.sample(frac=split_size)
    test_data = data.drop(train_data.index)
    return train_data, test_data

# Standard scalar normalization


def standard_scalar_normalization(data):
    for i in range(35):
        m = data.mean()
        sigma = data.std()
        data = data.subtract(m)

        data = data.divide(sigma)

    return data


def Normalization(data):
    minimum = data.min()
    maximum = data.max()

    return (data-minimum)/(maximum - minimum)


def seperate_X_Y(data):
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    X = standard_scalar_normalization(X)

    return X, Y


def compute_Accuracy(Y_prediction, Y_test):
    Accuracy = 0

    for i in range(Y_prediction.shape[0]):
        if (Y_prediction[i] == Y_test.iloc[i]):
            Accuracy = Accuracy + 1

    Accuracy = Accuracy/Y_prediction.shape[0]

    return Accuracy


def binary_SVM_Classifier(X_train, Y_train, X_test, Y_test, kernel='RadialBasisFunction'):
    if (kernel == 'Linear'):
        Linear = svm.SVC(kernel='linear', C=1,
                         decision_function_shape='ovo').fit(X_train, Y_train)
        Linear_prediction = Linear.predict(X_test)
        Accuracy = compute_Accuracy(Linear_prediction, Y_test)
        print("Accuracy using Linear Kernel:", Accuracy, file=out)

    elif (kernel == 'Quadratic'):
        Quadratic = svm.SVC(kernel='poly', degree=2, C=1,
                            decision_function_shape='ovo').fit(X_train, Y_train)
        Quadratic_prediction = Quadratic.predict(X_test)
        Accuracy = compute_Accuracy(Quadratic_prediction, Y_test)
        print("Accuracy using Quadratic Kernel:", Accuracy, file=out)

    else:
        RBF = svm.SVC(kernel='rbf', gamma=1, C=1,
                      decision_function_shape='ovo').fit(X_train, Y_train)
        RBF_prediction = RBF.predict(X_test)
        Accuracy = compute_Accuracy(RBF_prediction, Y_test)
        print("Accuracy using Radial Basis Function Kernel:", Accuracy, file=out)

    return Accuracy


def MLP_classifier(X_train, Y_train, X_test, Y_test, hidden_layers=[], weight_optimizer='sgd', learning_rate_value=0.001, batch_size_value=32, maximum_iterations=1000, flag=0):
    Classifier = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=maximum_iterations,
                               solver=weight_optimizer, learning_rate_init=learning_rate_value, batch_size=batch_size_value)
    Classifier.fit(X_train, Y_train)
    Y_prediction = Classifier.predict(X_test)
    Accuracy = compute_Accuracy(Y_prediction, Y_test)

    if (flag == 0):
        print('Accuracy with {:d} hidden layer with '.format(
            len(hidden_layers)), end='', file=out)
        for i in range(len(hidden_layers)):
            if (i+1 != len(hidden_layers)):
                print("{:d} and ".format(hidden_layers[i]), end='', file=out)
            else:
                print("{:d} nodes ".format(hidden_layers[i]), end='', file=out)

        print(Accuracy, file=out)

    elif (flag == 1):
        print('Accuracy with learning rate as {:f} using {:d} hidden layer is '.format(
            learning_rate_value, len(hidden_layers)), end='', file=out)
        print(Accuracy, file=out)

    return Accuracy


def backward_elimination(X_train, Y_train, X_test, Y_test, hidden_layers, learning_rate, Accuracy_original, threshold_value=0.5):
    max_value = Accuracy_original

    while (max_value > threshold_value and X_train.shape[1] > 1):
        Accuracy_modified = []
        for column_name in X_train.columns:
            X_train_modified = X_train.drop(column_name, axis=1)
            X_test_modified = X_test.drop(column_name, axis=1)
            Accuracy_modified.append(MLP_classifier(
                X_train_modified, Y_train, X_test_modified, Y_test, hidden_layers, 'sgd', learning_rate, 32, 5000, 2))
        max_value = max(Accuracy_modified)
        max_index = Accuracy_modified.index(max_value)

        if max_value > threshold_value:
            X_train = X_train.drop(X_train.columns[max_index], axis=1)
            X_test = X_test.drop(X_test.columns[max_index], axis=1)

    print("The best set of features are ", X_train.columns, file=out)
    print("Accuracy with the above features is", max_value, file=out)

    return X_train, X_test, max_value


def ensemble_learning_max_voting(X_train, Y_train, X_test, Y_test, models):
    
    voting_prediction = np.zeros(Y_test.shape[0])
    models_predictions = []
    vote_for = [0,0,0]

    for model in models:
      model[1].fit(X_train,Y_train)
      Y_prediction = model[1].predict(X_test)
      models_predictions.append(Y_prediction)

    for index in range(Y_test.shape[0]):
      vote_for[0] = 0
      vote_for[1] = 0
      vote_for[2] = 0
      
      for j in range(3):
        vote_for[models_predictions[j][index]] += 1
      
      if(vote_for[0]>=2): voting_prediction[index] = 0
      elif (vote_for[1]>=2): voting_prediction[index] = 1
      elif (vote_for[2]>=2): voting_prediction[index] = 2
      else: voting_prediction[index] = 0

    Accuracy = compute_Accuracy(voting_prediction, Y_test)
    print("Accuracy using ensemble learning with the models ", models[0][0], ",", models[1][0], ",", models[2][0], "with", len(
        hidden_layers), "hidden layers is", Accuracy,file=out)

    return Accuracy


# Start of main
if __name__ == "__main__":

    out = open('q2_results.txt', 'w')

    # Reading data from iris.data file using pandas
    original_data = pd.read_csv("iris.data", header=None)
    original_data.columns = ["Sepal_Length", "Sepal_Width", "Petal_Length",
                             "Petal_Width", "Class"]  # Labeling columns with names

    # Part 1

    categorical_encoding(original_data)  # Performing Categorical Encoding
    # Class as Iris-setosa -> 0
    # Iris-versicolor -> 1
    # Iris-virginica -> 2
    data = original_data.copy()
    # Dividing the dataset into two parts 80% for training and 20% for testing
    train_data, test_data = sample_split(data, 0.8)
    # Seperating X_train and Y_train and performing Standard Scalar Normalization
    X_train, Y_train = seperate_X_Y(train_data)
    # Seperating X_test and Y_test and performing Standard Scalar Normalization
    X_test, Y_test = seperate_X_Y(test_data)

    # end of Part 1

    # Part 2

    print('--------------- Start of Part 2 -------------------\n\n', file=out)
    Linear_Accuracy = binary_SVM_Classifier(
        X_train, Y_train, X_test, Y_test, 'Linear')  # Accuracy using Linear Kernel
    Quadratic_Accuracy = binary_SVM_Classifier(
        X_train, Y_train, X_test, Y_test, 'Quadratic')  # Accuracy using Quadratic Kernel
    # Accuracy using Radial Basis Function kernel
    RBF_Accuracy = binary_SVM_Classifier(
        X_train, Y_train, X_test, Y_test, 'RadialBasisFunction')
    print('\n\n--------------- End of Part 2 -------------------\n\n', file=out)

    # end of Part 2

    # Part 3

    print('--------------- Start of Part 3 -------------------\n\n', file=out)

    # Using stochastic gradient descent optimiser, learing rate = 0.001 and batch size = 32

    Accuracy_1_hidden_layer = MLP_classifier(X_train, Y_train, X_test, Y_test, [
                                             16], 'sgd', 0.001, 32, 1000)          # Accuracy for 1 hidden layer with 16 nodes
    # Accuracy for 2 hidden layers with 256 nodes and 16 nodes respectively
    Accuracy_2_hidden_layer = MLP_classifier(X_train, Y_train, X_test, Y_test, [
                                             256, 16], 'sgd', 0.001, 32, 1000)
    hidden_layers = [16]

    if (Accuracy_1_hidden_layer < Accuracy_2_hidden_layer):
        hidden_layers = [256, 16]

    best_accuracy_hidden_layers_model = hidden_layers
    print("The best accuracy model is with ", len(
        best_accuracy_hidden_layers_model), "layers", file=out)
    print('\n\n--------------- End of Part 3 -------------------\n\n', file=out)

    # end of Part 3

    # Part 4

    print('--------------- Start of Part 4 -------------------\n\n', file=out)
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    Accuracy_learning_rates = []
    for learning_rate in learning_rates:
        Accuracy_learning_rates.append(MLP_classifier(
            X_train, Y_train, X_test, Y_test, hidden_layers, 'sgd', learning_rate, 32, 5000, 1))

    # Plotting Learning rate vs Accuracy

    lr = -np.log10(learning_rates)
    plt.plot(lr, Accuracy_learning_rates,
             linestyle='dashed', marker='o', scalex=True)
    plt.xlabel("Learning Rates (in log scale)")
    plt.ylabel("Accuracy")
    plt.savefig('LearningVsAccuracy_v1.png')
    plt.close()
    plt.plot(learning_rates, Accuracy_learning_rates,
             linestyle='dashed', marker='o', scalex=True)
    plt.xscale("log")
    plt.xlabel("Learning Rates")
    plt.ylabel("Accuracy")
    plt.savefig('LearningVsAccuracy_v2.png')
    plt.close()
    print('\n\n--------------- End of Part 4 -------------------\n\n', file=out)

    # end of Part 4

    # Part 5

    print('--------------- Start of Part 5 -------------------\n\n', file=out)
    Accuracy_with_all_features = Accuracy_1_hidden_layer
    if Accuracy_1_hidden_layer < Accuracy_2_hidden_layer:
        Accuracy_with_all_features = Accuracy_2_hidden_layer
    X_temp_train, X_temp_test, Acuuracy_after_backward_elimination = backward_elimination(
        X_train, Y_train, X_test, Y_test, hidden_layers, learning_rate, Accuracy_with_all_features, 0.6)
    print('\n\n--------------- End of Part 5 -------------------\n\n', file=out)

    # end of Part 5

    # Part 6

    print('--------------- Start of Part 6 -------------------\n\n', file=out)
    models = [('SVM Quadratic', svm.SVC(kernel='poly', degree=2, C=1, decision_function_shape='ovo')),
              ('SVM Radial Basis', svm.SVC(kernel='rbf',
               gamma=1, C=1, decision_function_shape='ovo')),
              ('MLP', MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=1000, solver='sgd', learning_rate_init=0.001, batch_size=32))]
    Accuracy_with_ensemble_learning = ensemble_learning_max_voting(
        X_train, Y_train, X_test, Y_test, models)
    print('\n\n--------------- End of Part 6 -------------------\n\n', file=out)

    # end of Part 6

    out.close()
