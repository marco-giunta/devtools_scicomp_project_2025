from pyclassify.classifier import kNN
from pyclassify.utils import read_config, read_file
import argparse
import os

def split_dataset(X, y, train_percent=0.8):
    n_train = int(train_percent * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    return X_train, y_train, X_test, y_test

def compute_accuracy(pred_labels, true_labels):
    return sum(([t == p for t, p in zip(true_labels, pred_labels)])) / len(pred_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    args = parser.parse_args()
    config_path = args.config

    config = read_config(config_path)
    k = config.get('k')
    dataset_path = config.get('dataset')
    X, y = read_file(dataset_path)
    X_train, y_train, X_test, y_test = split_dataset(X, y)
    model = kNN(k)
    predictions = model((X_train, y_train), X_test)
    accuracy = compute_accuracy(predictions, y_test)
    print(f'Accuracy: {accuracy:.3%}')

    # os.system("git add .")
    # os.system('git commit -m "practical2"')
    # os.system("git push")

if __name__ == '__main__':
    main()