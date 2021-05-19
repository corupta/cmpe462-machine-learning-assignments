#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import numpy as np
import time
import sys
import os

SHOW_PLOTS = False
RANDOM_SEED = None
# RANDOM_SEED = 12345

rng = np.random.default_rng(RANDOM_SEED)

# PART1_POINT_XY_RANGE = 200  # center is 0
PART1_POINT_X_RANGE = 200  # center is 0
PART1_POINT_Y_RANGE = 600  # center is 0

PART2_REGULARIZATION_LAMBDA = np.exp(-10)
PART2_PLOT_PREDICTIONS = False
PART2_PLOT_LAMBDA_VALUES = False
S_FOLD_S_VALUE = 5

MIN_LOSS_CHANGE_FOR_STOP = 0.00001
STEP_SIZE_DEFAULT = 0.000001

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def solve_logistic_regression(X, t, batch_size = np.inf, step_size=STEP_SIZE_DEFAULT):
    prev_loss = np.inf
    current_loss = np.inf
    current_iteration = 0
    n, m = X.shape
    loss_list = []
    w = np.zeros((m, 1))
    while prev_loss - (current_loss if current_loss != np.inf else 0) > MIN_LOSS_CHANGE_FOR_STOP:
    # while current_loss > MIN_LOSS_CHANGE_FOR_STOP:
        current_iteration += 1
        prev_loss = current_loss
        current_loss = 0
        # print("Running iteration {}, prev_lost={}".format(current_iteration, prev_loss))
        for i in range(0, n, batch_size if batch_size != np.inf else n):
            chosen_indexes = [index for index in range(i, min(i+batch_size, n))]
            elems = np.take(X, chosen_indexes, axis=0)
            targets = np.take(t, chosen_indexes, axis=0)
            # loss = sum ( ln(1 + exp (−y * wT * x)) ) / N
            # but in our case X is not vector but a matrix (one dim is each input row)
            # and y is not scalar but a vector (y for all input rows)
            # thus, use below slightly modified, short version
            a = np.matmul(
                                elems,
                                w
                            )
            b = - np.multiply(
                            targets,
                            np.matmul(
                                elems,
                                w
                            )
                        )
            c = 1 + np.exp(
                        - np.multiply(
                            targets,
                            np.matmul(
                                elems,
                                w
                            )
                        )
                    )
            d = np.log(
                    1 + np.exp(
                        - np.multiply(
                            targets,
                            np.matmul(
                                elems,
                                w
                            )
                        )
                    )
                )
            batch_loss = np.sum(
                np.log(
                    1 + np.exp(
                        - np.multiply(
                            targets,
                            np.matmul(
                                elems,
                                w
                            )
                        )
                    )
                )
            ) / n
            current_loss += batch_loss
            # calculate gradient (modified formula to work on batch_size elements)
            g = - np.dot(
                elems.transpose(),
                np.multiply(
                    targets,
                    sigmoid(
                        - np.multiply(
                            targets,
                            np.matmul(
                                elems,
                                w
                            )
                        )
                    )
                )
            ) / targets.size
            # update weights
            w = w - step_size * g
        loss_list.append(current_loss)
    return w, current_iteration


metrics_path = 'metrics'
def solve_logistic_regression_s_fold(X, t, batch_size = np.inf, step_size=STEP_SIZE_DEFAULT):
    n, m = X.shape
    sample_per_fold = int(n / S_FOLD_S_VALUE)
    # if sample_per_fold * S_FOLD_S_VALUE != n:
    #    raise Exception("Sample count {} is not divisible by S, {} of s-fold"
    #                    .format(n, S_FOLD_S_VALUE))
    err_test_values = np.zeros(S_FOLD_S_VALUE)
    err_train_values = np.zeros(S_FOLD_S_VALUE)
    iteration_count_values = np.zeros(S_FOLD_S_VALUE)
    for i in range(S_FOLD_S_VALUE):
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
        print("Running for Fold #{}".format(i))
        w, iteration_count = solve_logistic_regression(X_train, t_train, batch_size, step_size)
        print("Finished in {} iterations".format(iteration_count))
        y_train = sigmoid(np.matmul(X_train, w))
        y_test = sigmoid(np.matmul(X_test, w))
        err_train = np.sum((y_train - (1 + t_train)/2)**2) / t_train.size
        err_test = np.sum((y_test - (1 + t_test)/2)**2) / t_test.size
        err_train_values[i] = err_train
        err_test_values[i] = err_test
        iteration_count_values[i] = iteration_count
    err_test_average = np.sum(err_test_values) / S_FOLD_S_VALUE
    err_train_average = np.sum(err_train_values) / S_FOLD_S_VALUE
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    metric_filename = "batch-{}_step-{}.txt".format(batch_size, step_size)
    with open(os.path.join(metrics_path, metric_filename), 'w') as f:
        lines = [
            "Average err_train = {}".format(err_train_average),
            "Average err_test = {}".format(err_test_average),
            "Stats by Fold:"
        ]
        for i in range(S_FOLD_S_VALUE):
            lines.append("Fold #{} : iterations = {} , err_train = {:.3f} , err_test = {:.3f}".format(
                i+1,iteration_count_values[i], err_train_values[i], err_test_values[i]))
        for line in lines:
            print(line)
        f.write("\n".join(lines))

    return err_train_average, err_test_average, err_train_values, err_test_values, iteration_count_values
# van or not?
target_name_to_number = {
    "van": 1,
    "saab": -1
}
def part1(batch_size):
    dataset_filename = 'vehicle.csv'
    input = np.loadtxt(dataset_filename, delimiter=",", encoding="utf8", skiprows=1, dtype=str)
    # select those with target_classes only
    input = input[(input[:, -1] == "saab") | (input[:, -1] == "van")]
    n, m = input.shape
    X = input[:, 0:(m-1)].astype(np.float)
    t = np.vectorize(target_name_to_number.get)(input[:, (m-1):])
    start = time.time()
    err_train_average, err_test_average, err_train_values, err_test_values, iteration_count_values = solve_logistic_regression_s_fold(X, t, batch_size)
    end = time.time()

    print("Applying {}-fold cross validation".format(S_FOLD_S_VALUE))
    print("There were {} independent variables and 1 dependent variables".format(m - 1))
    print("There were {} samples in total".format(n))
    print("Completed in {:.3f} milliseconds".format((end - start) * 1000))


    pass
    # TODO

def part1_old(step):
    # pick random (x,y) points where both x and y are in range [-100,100)
    point_count = 0
    point_size = 1
    if step == 1:
        point_count = 50
        point_size = 8
    elif step == 2:
        point_count = 100
        point_size = 4
    elif step == 3:
        point_count = 5000
        point_size = 1
    else:
        raise Exception("On part1: Unexpected step " + str(step) + ", must be one of  1-3")
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


def calculate_all():
    step_sizes = [0.0000001, 0.000001, 0.00001]
    batch_sizes = [1, np.inf, 32]
    for batch_size in batch_sizes:
        for step_size in step_sizes:
            print("Running for batch={}, step={}".format(batch_size, step_size))
            part1(batch_size, step_size)


if __name__ == "__main__":
    # calculate_all()
    # exit(0)
    usageError = False
    if len(sys.argv) != 3:
        usageError = True
    elif not sys.argv[1] in ['part1']:
        usageError = True
    elif (not sys.argv[2] in ['step1', 'step2']) and sys.argv[2] != str(int(sys.argv[2])):
        usageError = True
    if usageError:
        print("""
        Error! Please run the program as follows:
        python3 assignment1.py (part1) (step1|step2|<number>)
        eg: "python3 assignment1.py part1 step2" 
        """)
        exit(1)

    batch_size = None
    if sys.argv[2] == "step1":
        batch_size = math.inf
    elif sys.argv[2] == "step2":
        batch_size = 1
    else:
        batch_size = int(sys.argv[2])
    if sys.argv[1] == 'part1':
        part1(batch_size)
    else:
        raise Exception("On main: Unexpected part '" + sys.argv[1] + "', 'must be one of  part1"'')
