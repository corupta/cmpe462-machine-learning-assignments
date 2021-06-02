#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import sys
from libsvm.svm import svm_problem, svm_parameter, svm_model
from libsvm.svmutil import svm_train, svm_predict

PRINT_USEFUL_EXTRA_TEXT = False

SHOW_PLOTS = False
RANDOM_SEED = None
# RANDOM_SEED = 12345

rng = np.random.default_rng(RANDOM_SEED)

PART1_MAX_TREE_DEPTH = 6  # max 5 splits / leaf is level 6 at max

PART2_REGULARIZATION_LAMBDA = np.exp(-10)
PART2_S_FOLD_S_VALUE = 5
PART2_PLOT_PREDICTIONS = False
PART2_PLOT_LAMBDA_VALUES = False

part1_number_to_class = ["Iris-setosa", "Iris-versicolor"]
part1_class_to_number = {}
for ind, class_name in enumerate(part1_number_to_class):
    part1_class_to_number[class_name] = ind

def prob_to_entropy(prob):
    return prob if prob == 0 else - prob * np.log2(prob)

def entropy(p, n):
    return prob_to_entropy(p/(p+n)) + prob_to_entropy(n/(p+n))

def entropy_for_t(t):
    counts = np.zeros(2)
    for c in t:
        counts[c] += 1
    return entropy(counts[0], counts[1])

# tree elements: (feature_index, threshold, lte_node, gt_node)

def solve_decision_tree_recurse(X, t, use_gain_ratio, depth = 0):
    n, m = X.shape

    counts = np.zeros(2)
    for c in t:
        counts[c] += 1

    if depth >= PART1_MAX_TREE_DEPTH or np.max(counts) == t.size:
        return np.argmax(counts)

    current_entropy = entropy(counts[0], counts[1])
    # separate by feature mean values as thresholds
    thresholds = np.mean(X, axis=0)
    split_feature_index = -1
    prev_score = -np.inf
    for i in range(m):
        lte_t = t[(X[:, i] <= thresholds[i])]
        gt_t = t[(X[:, i] > thresholds[i])]
        lte_entropy = entropy_for_t(lte_t)
        gt_entropy = entropy_for_t(gt_t)
        weighted_entropy = (lte_t.size * lte_entropy + gt_t.size * gt_entropy) / t.size
        info_gain = current_entropy - weighted_entropy
        score = info_gain
        if use_gain_ratio:
            gain_ratio = info_gain / (
                    prob_to_entropy(lte_t.size/t.size)
                    + prob_to_entropy(gt_t.size/t.size)
            )
            score = gain_ratio
        if score > prev_score:
            split_feature_index = i
            prev_score = score
    lte_t = t[(X[:, split_feature_index] <= thresholds[split_feature_index])]
    gt_t = t[(X[:, split_feature_index] > thresholds[split_feature_index])]
    lte_X = X[(X[:, split_feature_index] <= thresholds[split_feature_index])]
    gt_X = X[(X[:, split_feature_index] > thresholds[split_feature_index])]
    if split_feature_index == -1:
        # shouldn't end up here, but put it just in case.
        return np.argmax(counts)
    return (
        split_feature_index, 
        thresholds[split_feature_index],
        solve_decision_tree_recurse(lte_X, lte_t, use_gain_ratio),
        solve_decision_tree_recurse(gt_X, gt_t, use_gain_ratio)
    )

def decide_for_sample(dt, x):
    # x has 1 rows!!!
    current = dt
    while isinstance(current, tuple):
        split_feature_index, threshold, lte_node, gt_node = current
        current = lte_node if x[split_feature_index] <= threshold else gt_node
    return current

def solve_decision_tree(X, t, use_gain_ratio):
    # first 40 of each class
    X_train = np.concatenate((X[0:40, :], X[50:90, :]), axis=0)
    t_train = np.concatenate((t[0:40, :], t[50:90, :]), axis=0)
    # last 10 of each class
    X_test = np.concatenate((X[40:50, :], X[90:100, :]), axis=0)
    t_test = np.concatenate((t[40:50, :], t[90:100, :]), axis=0)
    test_count = 20
    dt = solve_decision_tree_recurse(X_train, t_train, use_gain_ratio)
    y_predict = [decide_for_sample(dt, X_test[i, :]) for i in range(20)]
    correct_count = 0
    for i in range(20):
        if y_predict[i] == t_test[i]:
            correct_count += 1
    accuracy = correct_count / test_count

    if PRINT_USEFUL_EXTRA_TEXT:
        print("full dt: ", dt)
    return accuracy, dt[0]


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
    t = np.vectorize(part1_class_to_number.get)(input[:, (m - 1):])

    start = time.time()
    accuracy, root_param_index = solve_decision_tree(
        X, t, use_gain_ratio)
    end = time.time()

    if PRINT_USEFUL_EXTRA_TEXT:
        print("calculated in {:.3f} ms".format((end - start)*1000))
    root_param = headings[root_param_index]
    print("DT {} {:.2f}".format(root_param, accuracy))
    return




part2_number_to_class = ["M", "B"]
part2_class_to_number = {}
for ind, class_name in enumerate(part2_number_to_class):
    part2_class_to_number[class_name] = ind

def solve_svm(X_train, t_train, X_test, t_test, C, kernel):
    # normalize data to avoid warnings related to reaching max iteration for some c values
    # WARNING: reaching max number of iterations
    norm = np.linalg.norm(X_train, axis=0)
    norm_X_train = X_train / norm
    norm_X_test = X_test / norm
    m = svm_train(t_train, norm_X_train, "-q -t {} -c {}".format(kernel, C))
    n  = t_test.size
    number_of_support_vectors = m.get_nr_sv()
    p_labels, p_acc, p_vals = svm_predict(t_test, norm_X_test, m, "-q")
    accuracy = p_acc[0] / 100
#    manual_accuracy = 0
#    for i in range(n):
#        if  int(p_labels[i]) == t_test[i]:
#            manual_accuracy += 1
#    manual_accuracy  /= n
    return accuracy,  number_of_support_vectors

def part2(step):
    dataset_filename = 'wbcd.csv'

    kernels = [0]
    c_values = [10.0]
    # c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    # kernels = [0, 1, 2, 3]
    if step == 1:
        c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    elif step == 2:
        kernels = [0, 1, 2, 3]
    else:
        raise Exception("On part2: Unexpected step " + str(step) + ", must be one of  1-2")

    input = np.loadtxt(dataset_filename, delimiter=",", encoding="utf8", skiprows=0, dtype=str)
    headings = input[0, 2:] # remove id and diagnosis field, too.
    input = input[1:, :]
    n, m = input.shape
    X = input[:, 2:].astype(np.float)
    t = np.vectorize(part2_class_to_number.get)(input[:, 1])
    X_train = X[:400, :]
    X_test  = X[400:, :]
    t_train = t[:400]
    t_test = t[400:]

    if PRINT_USEFUL_EXTRA_TEXT:
        print("There are {} independent variables and 1 dependent variables".format(m - 2))
        print("There are {} samples in total".format(n))

    for c in c_values:
        for k in kernels:
            if PRINT_USEFUL_EXTRA_TEXT:
                print("Calculating svm for C={} Kernel={}".format(c, k))
            start = time.time()
            accuracy, support_vector_count = solve_svm(X_train, t_train, X_test, t_test, c, k)
            end = time.time()
            print("SVM kernel={} C={} acc={} n={}".format(k, c, accuracy, support_vector_count))
            if PRINT_USEFUL_EXTRA_TEXT:
                print("Calculated svm for C={} Kernel={} in {:.3f} milliseconds".format(c, k, (end - start)*1000))


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
