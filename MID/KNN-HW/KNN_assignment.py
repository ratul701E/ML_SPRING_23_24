import numpy as np
import matplotlib.pyplot as plt
import data_utils
import download

def download_data():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download_dir = "./data"
    download.maybe_download_and_extract(url,download_dir)

# Class to initialize and apply K-nearest neighbour classfier
class KNearestNeighbor(object):
    def __init__(self):
        pass

    # Method to initialize classifier with training data
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    
    # Method to predict labels of test examples using 'compute_distances' and 'predict_labels' methods.
    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists, k=k)

    # Method to compute Euclidean distances from each text example to every training example  
    def compute_distances(self, X, distance_metric = 'euclidean'):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # YOUR CODE HERE
        # Compute distances from each test example (in argument 'X' of this method) to every training example and store distances in
        # dists variable given above. For each row, i, dist[i] should contain distances between test example i and every training example.
        if distance_metric == 'euclidean':
            for i in range(num_test):
                dists[i, :] = np.sqrt(np.sum((self.X_train - X[i, :])**2, axis=1))
        elif distance_metric == 'manhattan':
            for i in range(num_test):
                dists[i, :] = np.sum(np.abs(self.X_train - X[i, :]), axis=1)
        else:
            raise ValueError('Invalid distance_metric.')
        return dists

    # Method to predict labels of test examples using chosen value of k given Euclidean distances obtained from 'compute_distances' method.
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        # YOUR CODE HERE
        # Given dists computed using 'compute_distances' method above, obtain k closest distances to training examples for each test example
        # dists[i]. Use k closest distances obtained to predict label of each dists[i]. Label of each dists[i] should be stored in y_pred[i].
        for i in range(num_test):
            closest_y = []
            sorted_indices = np.argsort(dists[i, :])
            closest_y = self.y_train[sorted_indices[:k]]
            counts = np.bincount(closest_y)
            y_pred[i] = np.argmax(counts)
        return y_pred

def visualize_data(X_train, y_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

if __name__ == "__main__":

    # Download CIFAR10 data and store it in current directory if you have not done it.
    # download_data()
    cifar10_dir = './data/cifar-10-batches-py'

    # Load training and testing data from CIFAR10 dataset
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10(cifar10_dir)

    # Checking the size of the training and testing data
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    #Visualize the data if you want
    #visualize_data(X_train, y_train)

    # Memory error prevention by subsampling data. We sample 10000 training examples and 1000 test examples.
    num_training = 1000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 100
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # YOUR CODE HERE
    # Reshape data and place into rows. Flatten the training and test data so each row 
    # consists of all pixels of an example
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print(X_train.shape, X_test.shape) # X_train should be (10000, 3072) and X_test should be (1000, 3072)

    # Performing KNN
    classifier = KNearestNeighbor()
    # YOUR CODE HERE
    # Use the KNearestNeighbour classifier to do as follows:
    # 1) Initialize classifier with training data
    # 2) Use classifier to compute distances from each test example in X_test to every training example
    # 3) Use classifier to predict labels of each test example in X_test using k=5 
    # Initialize classifier with training data
    classifier.train(X_train, y_train)
    dists = classifier.compute_distances(X_test)
    k = 5
    y_test_pred = classifier.predict_labels(dists, k)

    # Calculate accuracy
    num_correct = np.sum(y_test_pred == y_test)  # number of test examples correctly predicted
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct with k=%d => accuracy: %f' % (num_correct, num_test, k, accuracy))
    # Accuracy above should be ~ 29-30%

    # Perform 5-fold cross validation to find optimal k from choices below
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10]

    X_train_folds = []
    y_train_folds = []
    # Training data is split into 5 folds
    X_train_folds = np.array_split(X_train,num_folds)
    y_train_folds = np.array_split(y_train,num_folds)
    k_to_accuracies = {} # dictionary to hold validation accuracies for each k 


    for k in k_choices:
        k_to_accuracies[k] = [] # each key, k, should hold its list of 5 validation accuracies
        
        # For each fold of cross validation
        for num_knn in range(0,num_folds):
            # YOUR CODE HERE
            # 1) Split training data into validation fold and training folds
            # 2) Inititialize classifier with training folds and compute distances between 
            #    examples in validation fold and training folds
            # 3) Use classifier to predict labels of valdation fold for given k value

            X_train_fold = np.concatenate([fold for i, fold in enumerate(X_train_folds) if i != num_knn])
            y_train_fold = np.concatenate([fold for i, fold in enumerate(y_train_folds) if i != num_knn])
            X_val_fold = X_train_folds[num_knn]
            y_val_fold = y_train_folds[num_knn]
            
            classifier.train(X_train_fold, y_train_fold)
            
            dists = classifier.compute_distances(X_val_fold)
            
            y_val_pred = classifier.predict_labels(dists, k)
            
            num_correct = np.sum(y_val_pred == y_val_fold)
            accuracy = float(num_correct) / len(y_val_fold)
            k_to_accuracies[k].append(accuracy)

            # number of test examples correctly predicted, where y_test_pred contains labels
            # predicted by classifier on validation fold
            num_correct = np.sum(y_test_pred == y_test) 
            accuracy = float(num_correct) / num_test
            # print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
            k_to_accuracies[k].append(accuracy)


    print("Printing our 5-fold accuracies for varying values of k:")
    print()
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))
    
    for k in sorted(k_to_accuracies):
        print('k = %d, avg. accuracy = %f' % (k, sum(k_to_accuracies[k])/5))
    
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation

    accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.savefig('cross-validation_accuracy.jpg')

    # YOUR CODE HERE
    # Choose best value of k based on cross-validation results
    # Intialize classifier and predict labels of test data, X_test, using best value of k
    best_k = None
    best_accuracy = -1

    for k in k_choices:
        # Calculate average accuracy for current k
        avg_accuracy = np.mean(k_to_accuracies[k])
        
        # Update best_k if current k has higher average accuracy
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_k = k

    # Use the best value of k to predict labels of test data, X_test
    dists = classifier.compute_distances(X_test)
    y_test_pred = classifier.predict_labels(dists, k=best_k)
    # Computing and displaying the accuracy for best k found during cross-validation
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct on test data => accuracy: %f' % (num_correct, num_test, accuracy))
    print('Best K value is: ', best_k)
    # Accuracy above should be ~ 57-58%


    # Manhattan 
    print("Using Manhattan Distance Metric")
    classifier.train(X_train, y_train)
    dists = classifier.compute_distances(X_test, distance_metric = 'manhattan')
    k = 5
    y_test_pred = classifier.predict_labels(dists, k)

    # Calculate accuracy
    num_correct = np.sum(y_test_pred == y_test)  # number of test examples correctly predicted
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct with k=%d => accuracy: %f' % (num_correct, num_test, k, accuracy))


