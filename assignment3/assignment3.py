#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import sys

PRINT_USEFUL_EXTRA_TEXT = True

SHOW_PLOTS = False
RANDOM_SEED = None
# RANDOM_SEED = 12345

rng = np.random.default_rng(RANDOM_SEED)

PART1_MAX_TREE_DEPTH = 6  # max 5 splits / leaf is level 6 at max

PART2_REGULARIZATION_LAMBDA = np.exp(-10)
PART2_S_FOLD_S_VALUE = 5
PART2_PLOT_PREDICTIONS = False
PART2_PLOT_LAMBDA_VALUES = False

number_to_class = ["Iris-setosa", "Iris-versicolor"]
class_to_number = {}
for i, class_name in enumerate(number_to_class):
    class_to_number[class_name] = i

def solve_decision_tree(X, t, use_gain_ratio):
    # first 40 of each class
    X_train = np.concatenate(X[0:40, :], X[50:90, :], axis=0)
    t_train = np.concatenate(t[0:40, :], t[50:90, :], axis=0)
    # last 10 of each class
    X_test = np.concatenate(X[40:50, :], X[90:100, :], axis=0)
    t_test = np.concatenate(t[40:50, :], t[90:100, :], axis=0)
    # TODO CALCULATE DT
    # TODO CALCULATE ACCURACY


def part1(step):
    use_gain_ratio = False
    if step == 1:
        use_gain_ratio = False
    elif step == 2:
        use_gain_ratio = True
    else:
        raise Exception("On part1: Unexpected step " + str(step) + ", must be one of  1-2")

    dataset_filename = 'iris.csv'
    input = np.loadtxt(dataset_filename, delimiter=",", encoding="utf8", skiprows=0, dtype=str)
    headings = input[0, :]
    input = input[1:, :]
    n, m = input.shape
    X = input[:, 0:(m - 1)].astype(np.float)
    t = np.vectorize(class_to_number.get)(input[:, (m - 1):])

    start = time.time()
    accuracy, root_param_index = solve_decision_tree(
        X, t, use_gain_ratio)
    end = time.time()

    if PRINT_USEFUL_EXTRA_TEXT:
        print("calculated in {:.3f} ms".format((end - start)*1000))
    root_param = headings[root_param_index]
    print("DT {} {:.2f}".format(root_param, accuracy))
    return

    points = np.concatenate((
        rng.random((point_count, 1)) * PART1_POINT_X_RANGE - (PART1_POINT_X_RANGE/2),
        rng.random((point_count, 1)) * PART1_POINT_Y_RANGE - (PART1_POINT_Y_RANGE / 2),
    ), axis=1)
    classes = []
    for [x, y] in points:
        classes.append(0 if y < -3 * x + 1 else 1)
        # assume c=1 if y = -3x+1
    classes = np.array(classes)

    # points and classes are ready now, start PLA
    target_outputs = 2 * classes - 1  # map class 0 to -1, class 1 to 1 for PLA
    inputs_with_dummy_x0 = np.concatenate((np.ones((point_count, 1)), points), axis=1)
    w = np.zeros((3, 1))  # w means current weights
    t = 1  # t means current iteration index
    while True:
        if t % 1000 == 1:
            print("PLA ITERATION #{} STARTED".format(t))
        calculated = np.matmul(inputs_with_dummy_x0, w)
        misclassified = []
        for i in range(point_count):
            c = calculated[i, 0]
            c_sign = -1 if c < 0 else 1  # assume sign of 0 is 1 (+)
            if c_sign != target_outputs[i]:
                misclassified.append(i)
        if t==1000000: # prevent infinite loop just in case
            break
        if len(misclassified) == 0:
            break  # complete PLA
        chosen_misclassified_index = misclassified[rng.integers(0, len(misclassified))]
        input_misclassfied = inputs_with_dummy_x0[chosen_misclassified_index]
        target_output_misclassified = target_outputs[chosen_misclassified_index]
        # update weights
        u = np.reshape(input_misclassfied * target_output_misclassified, (3, 1))
        w += u
        t += 1  # next iteration
    print("Finished calculation in {} iterations".format(t))
    print("Calculated weights are: {}".format(w.flatten()))
    # our decision boundary is ready:    sign ( w0 * 1 + w1 * x + w2 * y )
    # which is  c = 1  if  w0 * 1 + w1 * x + w2 * y >= 0  else 0
    # thus, our decision boundary is =>  y = (- w0 - w1 * x) / w2
    # if we try to write it as y = mx + b, m = -w1/w2, b = -w0/w2
    m = -w[1, 0] / w[2, 0]
    b = -w[0, 0] / w[2, 0]

    print('Decision Boundary is y = {:.2f}x + {:.2f}'.format(m, b))

    line_x = np.linspace(- (PART1_POINT_X_RANGE/2), + (PART1_POINT_X_RANGE/2), 100)

    points_color_map = mcolors.ListedColormap(["red", "blue"])

    plt.plot(line_x, m * line_x + b, color='purple', label='Decision Boundary y = {:.2f}x + {:.2f}'.format(m, b))
    plt.plot(line_x, -3 * line_x + 1, color='green', label='Target Function y = -3x + 1')
    plt.scatter(points[:, 0], points[:, 1], s=point_size, c=classes, cmap=points_color_map)
    plt.title('Assignment 1 Part 1 Step {} - {} points '.format(step, point_count))
    plt.xlabel('x', color='#222222')
    plt.ylabel('y', color='#222222')
    plt.legend(loc='upper left')
    plt.grid()
    if SHOW_PLOTS:
        plt.show()
    plot_filename = "part1_step{}.png".format(step)
    plt.savefig(plot_filename)
    print("Saved the plot to {}".format(plot_filename))
    plt.close()

def solve_linear_regression(X, t, lambda_value = 0):
    # APPLY CLOSED FORM SOLUTION
    # w∗ = ((X'X)^-1)X't
    # w∗ = ((X'X + λI)^−1)X't
    w = np.matmul(
        np.linalg.inv(
            np.matmul(
                np.transpose(X),
                X
            ) + lambda_value
        ),
        np.matmul(
            np.transpose(X),
            t
        )
    )

    y = np.matmul(X, w)
    #erms = np.sum((y - t) ** 2) / 2 + (
    #    np.linalg.norm(w,2) * PART2_REGULARIZATION_LAMBDA / 2
    #    if apply_l2_regularization else 0
    #)
    erms = np.sqrt(np.sum((y - t) ** 2) / t.size)

    return w, erms, y

def solve_linear_regression_s_fold(X, t, lambda_value = 0):
    n, m = X.shape
    sample_per_fold = int(n / PART2_S_FOLD_S_VALUE)
    if sample_per_fold * PART2_S_FOLD_S_VALUE != n:
        raise Exception("Sample count {} is not divisible by S, {} of s-fold"
                        .format(n, PART2_S_FOLD_S_VALUE))
    erms_test_values = np.zeros(PART2_S_FOLD_S_VALUE)
    erms_train_values = np.zeros(PART2_S_FOLD_S_VALUE)
    for i in range(PART2_S_FOLD_S_VALUE):
        n_start = i * sample_per_fold
        n_end = (i+1) * sample_per_fold
        X_test = X[n_start:n_end, :]
        X_train = np.concatenate((
            X[0:n_start, :],
            X[n_end:, :]
        ), axis=0)
        t_test = t[n_start:n_end, :]
        t_train = np.concatenate((
            t[0:n_start, :],
            t[n_end:, :]
        ))
        w, erms_train, y_train = solve_linear_regression(X_train, t_train, lambda_value)
        y_test = np.matmul(X_test, w)
        erms_test = np.sqrt(np.sum((y_test - t_test) ** 2) / t_test.size)
        erms_test_values[i] = erms_test
        erms_train_values[i] = erms_train
    erms_test_average = np.sum(erms_test_values) / PART2_S_FOLD_S_VALUE
    erms_train_average = np.sum(erms_train_values) / PART2_S_FOLD_S_VALUE
    return erms_train_average, erms_test_average


def part2(step):
    dataset_filename = ''
    apply_l2_regularization = False
    if step == 1:
        dataset_filename = 'ds1.csv'
        apply_l2_regularization = False
    elif step == 2:
        dataset_filename = 'ds2.csv'
        apply_l2_regularization = False
    elif step == 3:
        dataset_filename = 'ds2.csv'
        apply_l2_regularization = True
    else:
        raise Exception("On part2: Unexpected step " + str(step) + ", must be one of  1-3")
    input = np.loadtxt(dataset_filename, delimiter=",", encoding="utf8")
    n, m = input.shape
    X = input[:, 0:(m-1)]
    t = input[:, (m-1):]
    start = time.time()
    erms_train, erms_test = solve_linear_regression_s_fold(X, t, PART2_REGULARIZATION_LAMBDA if apply_l2_regularization else 0)
    end = time.time()

    print("Applying {}-fold cross validation".format(PART2_S_FOLD_S_VALUE))
    print("There were {} independent variables and 1 dependent variables".format(m-1))
    print("There were {} samples in total".format(n))
    print("Completed in {:.3f} milliseconds".format((end - start)*1000))
    print("Average Erms_train = {}".format(erms_train))
    print("Average Erms_test = {}".format(erms_test))

    if PART2_PLOT_PREDICTIONS:
        print("Calculating without cross-validation")
        w, erms, y = solve_linear_regression(X, t, PART2_REGULARIZATION_LAMBDA if apply_l2_regularization else 0)
        print("Weights = {}".format(w.flatten()))
        plt.scatter(range(n), y, s=1, color='purple', label='Predicted y')
        plt.scatter(range(n), t, s=1, color='green', label='Target')
        plt.plot(range(n), y - t, color='red', label='Root Mean Square Error')
        plt.title('Assignment 1 Part 2 Step {} - Predictions - No Cross-Validation'.format(step))
        plt.xlabel('Sample Number', color='#222222')
        plt.ylabel('Target Value', color='#222222')
        plt.legend(loc='upper left')
        plt.grid()
        if SHOW_PLOTS:
            plt.show()
        plot_filename = "part2_step{}_predictions.png".format(step)
        plt.savefig(plot_filename)
        print("Saved the plot to {}".format(plot_filename))
        plt.close()

    if PART2_PLOT_LAMBDA_VALUES:
        print("Calculating for a range of lambda values")
        erms_train_values = []
        erms_test_values = []
        ln_lambda_values = range(-60, 5)
        for lambda_value in ln_lambda_values:
            erms_train, erms_test = solve_linear_regression_s_fold(X, t, np.exp(lambda_value))
            erms_train_values.append(erms_train)
            erms_test_values.append(erms_test)
        plt.plot(ln_lambda_values, erms_train_values, color='blue', label='Training')
        plt.plot(ln_lambda_values, erms_test_values, color='red', label='Test')
        plt.title('Assignment 1 Part 2 Step {} - Regularization - {}-Fold Cross-Validation'.format(step, PART2_S_FOLD_S_VALUE))
        plt.xlabel('ln(lambda)', color='#222222')
        plt.ylabel('Root Mean Square Error', color='#222222')
        plt.legend(loc='upper left')
        plt.grid()
        if SHOW_PLOTS:
            plt.show()
        plot_filename = "part2_step{}_regularization.png".format(step)
        plt.savefig(plot_filename)
        print("Saved the plot to {}".format(plot_filename))
        plt.close()


if __name__ == "__main__":
    usageError = False
    if len(sys.argv) != 3:
        usageError = True
    elif not sys.argv[1] in ['part1', 'part2']:
        usageError = True
    elif not sys.argv[2] in ['step1', 'step2']:
        usageError = True
    if usageError:
        print("""
        Error! Please run the program as follows:
        python3 assignment1.py (part1|part2) (step1|step2)
        eg: "python3 assignment1.py part1 step2" 
        """)
        exit(1)

    step_number = int(sys.argv[2][4:])
    if sys.argv[1] == 'part1':
        part1(step_number)
    elif sys.argv[1] == 'part2':
        part2(step_number)
    else:
        raise Exception("On main: Unexpected part '" + sys.argv[1] + "', 'must be one of  part1, part2"'')
