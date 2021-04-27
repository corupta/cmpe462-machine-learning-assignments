#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys

def part1(step):
    pointCount = 0
    if step == 1:
        pointCount = 50
    elif step == 2:
        pointCount = 100
    elif step == 3:
        pointCount = 5000
    else:
        raise Exception("On part1: Unexpected step " + str(step) + ", must be one of  1-3")
    pass

def part2(step):
    dataset_filename = ''
    apply_i2_regularization = False
    if step == 1:
        dataset_filename = 'ds1.csv'
        apply_i2_regularization = False
    elif step == 2:
        dataset_filename = 'ds2.csv'
        apply_i2_regularization = False
    elif step == 3:
        dataset_filename = 'ds2.csv'
        apply_i2_regularization = True
    else:
        raise Exception("On part2: Unexpected step " + str(step) + ", must be one of  1-3")
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
    print(plot_filename)
