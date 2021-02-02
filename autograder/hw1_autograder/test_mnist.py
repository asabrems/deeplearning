#!/usr/bin/env python3

import os
import sys

sys.path.append('./')

import traceback

import matplotlib.pyplot as plt
import numpy as np
from hw1.mnist import mnist


DATA_PATH = "./autograder/hw1_autograder/data"

def main():
    train_x, train_y, val_x, val_y = load_data()
    val_accuracies = mnist(train_x, train_y, val_x, val_y)
    visualize_results(val_accuracies)


def load_data():
    train_x = np.load(os.path.join(DATA_PATH, "train_data.npy"))
    train_y = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    val_x = np.load(os.path.join(DATA_PATH, "val_data.npy"))
    val_y = np.load(os.path.join(DATA_PATH, "val_labels.npy"))

    train_x = train_x / 255
    val_x = val_x / 255

    return train_x, train_y, val_x, val_y


def visualize_results(val_accuracies):
    print("Saving and showing graph")
    try:
        plt.plot(val_accuracies)
        plt.ylabel('Accuracy')
        plt.savefig('hw1/validation_accuracy.png')
        print("Accuracies", val_accuracies)
        # plt.show()
    except Exception as e:
        traceback.print_exc()
        print("Error: Problems generating plot. See if a .png was generated in hw1/. "
              "If not, check the writeup and Piazza hw1p1 thread.")


if __name__ == "__main__":
    main()
