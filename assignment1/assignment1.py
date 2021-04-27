#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import sys

SHOW_PLOTS = True
# RANDOM_SEED = None
RANDOM_SEED = 12345

rng = np.random.default_rng(RANDOM_SEED)

# PART1_POINT_XY_RANGE = 200  # center is 0
PART1_POINT_X_RANGE = 200  # center is 0
PART1_POINT_Y_RANGE = 600  # center is 0

PART2_REGULARIZATION_LAMBDA = np.exp(-10)
PART2_PRINT_WEIGHTS = True

def part1(step):
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
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.legend(loc='upper left')
    plt.grid()
    if SHOW_PLOTS:
        plt.show()

def solve_linear_regression(X, t, apply_l2_regularization = False):
    # APPLY CLOSED FORM SOLUTION
    # w∗ = ((X'X)^-1)X't
    # w∗ = ((X'X + λI)^−1)X't
    w = np.matmul(
        np.linalg.inv(
            np.matmul(
                np.transpose(X),
                X
            ) + (PART2_REGULARIZATION_LAMBDA if apply_l2_regularization else 0)
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
    # todo
    n,m = input.shape
    X = input[:, 0:(m-1)]
    t = input[:, (m-1):]
    start = time.time()
    w, erms, y = solve_linear_regression(X, t, apply_l2_regularization)
    end = time.time()

    plt.scatter(range(n), y, s=1, color='purple', label='Predicted y')
    plt.scatter(range(n), t, s=1, color='green', label='Target')
    plt.plot(range(n), y-t, color='red', label='Error')
    plt.title('Assignment 1 Part 2 Step {} '.format(step))
    plt.xlabel('Sample Number', color='#1C2833')
    plt.ylabel('Target Value', color='#1C2833')
    plt.legend(loc='upper left')
    plt.grid()
    if SHOW_PLOTS:
        plt.show()

    print("There were {} independent variables and 1 dependent variables".format(m-1))
    print("There were {} samples in total".format(n))
    print("Completed in {} milliseconds".format(end - start))
    print("Erms = {}".format(erms))
    if PART2_PRINT_WEIGHTS:
        print("Weights = {}".format(w.flatten()))
    pass

if __name__ == "__main__":
    usageError = False
    if len(sys.argv) != 3:
        usageError = True
    elif not sys.argv[1] in ['part1', 'part2']:
        usageError = True
    elif not sys.argv[2] in ['step1', 'step2', 'step3']:
        usageError = True
    if usageError:
        print("""
        Error! Please run the program as follows:
        python3 assignment1.py (part1|part2) (step1|step2|step3)
        eg: "python3 assignment1.py part1 step2" 
        """)
        exit(1)
    plot_filename = sys.argv[1] + "_" + sys.argv[2] + ".png"
    step_number = int(sys.argv[2][4:])
    if sys.argv[1] == 'part1':
        part1(step_number)
    elif sys.argv[1] == 'part2':
        part2(step_number)
    else:
        raise Exception("On main: Unexpected part '" + sys.argv[1] + "', 'must be one of  part1, part2"'')
    plt.savefig(plot_filename)
    print("Saved the plot to {}".format(plot_filename))
