# CMPE 462 Assignment 1 Report

## Part 1

I have implemented a single function taking `batch_size` parameter to solve both step 1 and step2. 
* When `batch_size=np.inf` or `batch_size>=sample_count` it works as full gradient descent
* When `batch_size=1` it works as stochastic gradient descent
* When `1<batch_size<sample_count` it works as mini batch gradient descent

That function also have an optional `step_size` parameter.
And when I tried a few values, I found out that if I keep it somewhat bigger than `1e-5` (such as `1e-4`) its loss value is increased on each iteration rather than decreasing and quickly arrives to infinity.
Thus, I have chosen small, medium, big `step_size` values as `1e-7`, `1e-6`, and `1e-5` respectively for testing and kept the medium one as default.

Applied 5-fold cross-validation, collected loss over iteration plots for each of them.
Calculated error in cross-validation via the "simple error formula" shown in 
`Lec04-LogisticRegression.pdf` page 22-23 rather than the loss function used elsewhere.
Collected train and test errors for each fold, and their average.

Below are cross-validation results and loss over iteration plots for each `batch_size` x `step_size` x `fold` combination.







### Batch Size inf (FGD) - Step Size 1e-7 (Small)
```
Average err_train = 0.15055732515442818
Average err_test = 0.15261621809682047
Stats by Fold:
Fold #1 : iterations = 6473.0 , err_train = 0.145 , err_test = 0.178
Fold #2 : iterations = 6485.0 , err_train = 0.149 , err_test = 0.155
Fold #3 : iterations = 6358.0 , err_train = 0.150 , err_test = 0.153
Fold #4 : iterations = 6576.0 , err_train = 0.149 , err_test = 0.150
Fold #5 : iterations = 6050.0 , err_train = 0.160 , err_test = 0.128
```
#### Batch Size inf (FGD) - Step Size  1e-7 (Small) - Fold #1
![plots/batch-inf_step-1e-07_fold-1.png](plots/batch-inf_step-1e-07_fold-1.png "Batch Size inf (FGD) - Step 1e-7 - Fold #1")
#### Batch Size inf (FGD) - Step Size  1e-7 (Small) - Fold #2
![plots/batch-inf_step-1e-07_fold-2.png](plots/batch-inf_step-1e-07_fold-2.png "Batch Size inf (FGD) - Step 1e-7 - Fold #2")
#### Batch Size inf (FGD) - Step Size  1e-7 (Small) - Fold #3
![plots/batch-inf_step-1e-07_fold-3.png](plots/batch-inf_step-1e-07_fold-3.png "Batch Size inf (FGD) - Step 1e-7 - Fold #3")
#### Batch Size inf (FGD) - Step Size  1e-7 (Small) - Fold #4
![plots/batch-inf_step-1e-07_fold-4.png](plots/batch-inf_step-1e-07_fold-4.png "Batch Size inf (FGD) - Step 1e-7 - Fold #4")
#### Batch Size inf (FGD) - Step Size  1e-7 (Small) - Fold #5
![plots/batch-inf_step-1e-07_fold-5.png](plots/batch-inf_step-1e-07_fold-5.png "Batch Size inf (FGD) - Step 1e-7 - Fold #5")




### Batch Size inf (FGD) - Step Size  1e-6 (Medium)
```
Average err_train = 0.13103323850981358
Average err_test = 0.13333879646475127
Stats by Fold:
Fold #1 : iterations = 2724.0 , err_train = 0.126 , err_test = 0.159
Fold #2 : iterations = 2825.0 , err_train = 0.130 , err_test = 0.136
Fold #3 : iterations = 2678.0 , err_train = 0.131 , err_test = 0.135
Fold #4 : iterations = 3038.0 , err_train = 0.127 , err_test = 0.138
Fold #5 : iterations = 2685.0 , err_train = 0.141 , err_test = 0.098
```
#### Batch Size inf (FGD) - Step Size  1e-6 (Medium) - Fold #1
![plots/batch-inf_step-1e-06_fold-1.png](plots/batch-inf_step-1e-06_fold-1.png "Batch Size inf (FGD) - Step 1e-6 - Fold #1")
#### Batch Size inf (FGD) - Step Size  1e-6 (Medium) - Fold #2
![plots/batch-inf_step-1e-06_fold-2.png](plots/batch-inf_step-1e-06_fold-2.png "Batch Size inf (FGD) - Step 1e-6 - Fold #2")
#### Batch Size inf (FGD) - Step Size  1e-6 (Medium) - Fold #3
![plots/batch-inf_step-1e-06_fold-3.png](plots/batch-inf_step-1e-06_fold-3.png "Batch Size inf (FGD) - Step 1e-6 - Fold #3")
#### Batch Size inf (FGD) - Step Size  1e-6 (Medium) - Fold #4
![plots/batch-inf_step-1e-06_fold-4.png](plots/batch-inf_step-1e-06_fold-4.png "Batch Size inf (FGD) - Step 1e-6 - Fold #4")
#### Batch Size inf (FGD) - Step Size  1e-6 (Medium) - Fold #5
![plots/batch-inf_step-1e-06_fold-5.png](plots/batch-inf_step-1e-06_fold-5.png "Batch Size inf (FGD) - Step 1e-6 - Fold #5")




### Batch Size inf (FGD) - Step Size 1e-5 (Big)
```
Average err_train = 0.0720774984544428
Average err_test = 0.07694272484894403
Stats by Fold:
Fold #1 : iterations = 6927.0 , err_train = 0.074 , err_test = 0.087
Fold #2 : iterations = 7126.0 , err_train = 0.072 , err_test = 0.090
Fold #3 : iterations = 7552.0 , err_train = 0.072 , err_test = 0.067
Fold #4 : iterations = 7162.0 , err_train = 0.070 , err_test = 0.096
Fold #5 : iterations = 7998.0 , err_train = 0.073 , err_test = 0.045
```
#### Batch Size inf (FGD) - Step Size 1e-5 (Big) - Fold #1
![plots/batch-inf_step-1e-05_fold-1.png](plots/batch-inf_step-1e-05_fold-1.png "Batch Size inf (FGD) - Step 1e-5 - Fold #1")
#### Batch Size inf (FGD) - Step Size 1e-5 (Big) - Fold #2
![plots/batch-inf_step-1e-05_fold-2.png](plots/batch-inf_step-1e-05_fold-2.png "Batch Size inf (FGD) - Step 1e-5 - Fold #2")
#### Batch Size inf (FGD) - Step Size 1e-5 (Big) - Fold #3
![plots/batch-inf_step-1e-05_fold-3.png](plots/batch-inf_step-1e-05_fold-3.png "Batch Size inf (FGD) - Step 1e-5 - Fold #3")
#### Batch Size inf (FGD) - Step Size 1e-5 (Big) - Fold #4
![plots/batch-inf_step-1e-05_fold-4.png](plots/batch-inf_step-1e-05_fold-4.png "Batch Size inf (FGD) - Step 1e-5 - Fold #4")
#### Batch Size inf (FGD) - Step Size 1e-5 (Big) - Fold #5
![plots/batch-inf_step-1e-05_fold-5.png](plots/batch-inf_step-1e-05_fold-5.png "Batch Size inf (FGD) - Step 1e-5 - Fold #5")





















### Batch Size 32 (MiniBatch) - Step Size 1e-7 (Small)
```
Average err_train = 0.1301209234288046
Average err_test = 0.13237757609349568
Stats by Fold:
Fold #1 : iterations = 2728.0 , err_train = 0.126 , err_test = 0.159
Fold #2 : iterations = 2853.0 , err_train = 0.129 , err_test = 0.135
Fold #3 : iterations = 2694.0 , err_train = 0.130 , err_test = 0.134
Fold #4 : iterations = 3095.0 , err_train = 0.126 , err_test = 0.137
Fold #5 : iterations = 2751.0 , err_train = 0.139 , err_test = 0.097
```
#### Batch Size 32 (MiniBatch) - Step Size  1e-7 (Small) - Fold #1
![plots/batch-32_step-1e-07_fold-1.png](plots/batch-32_step-1e-07_fold-1.png "Batch Size 32 (MiniBatch) - Step 1e-7 - Fold #1")
#### Batch Size 32 (MiniBatch) - Step Size  1e-7 (Small) - Fold #2
![plots/batch-32_step-1e-07_fold-2.png](plots/batch-32_step-1e-07_fold-2.png "Batch Size 32 (MiniBatch) - Step 1e-7 - Fold #2")
#### Batch Size 32 (MiniBatch) - Step Size  1e-7 (Small) - Fold #3
![plots/batch-32_step-1e-07_fold-3.png](plots/batch-32_step-1e-07_fold-3.png "Batch Size 32 (MiniBatch) - Step 1e-7 - Fold #3")
#### Batch Size 32 (MiniBatch) - Step Size  1e-7 (Small) - Fold #4
![plots/batch-32_step-1e-07_fold-4.png](plots/batch-32_step-1e-07_fold-4.png "Batch Size 32 (MiniBatch) - Step 1e-7 - Fold #4")
#### Batch Size 32 (MiniBatch) - Step Size  1e-7 (Small) - Fold #5
![plots/batch-32_step-1e-07_fold-5.png](plots/batch-32_step-1e-07_fold-5.png "Batch Size 32 (MiniBatch) - Step 1e-7 - Fold #5")




### Batch Size 32 (MiniBatch) - Step Size  1e-6 (Medium)
```
Average err_train = 0.06850718108092783
Average err_test = 0.0733356749825757
Stats by Fold:
Fold #1 : iterations = 7020.0 , err_train = 0.070 , err_test = 0.084
Fold #2 : iterations = 7248.0 , err_train = 0.068 , err_test = 0.085
Fold #3 : iterations = 7596.0 , err_train = 0.068 , err_test = 0.064
Fold #4 : iterations = 7228.0 , err_train = 0.066 , err_test = 0.092
Fold #5 : iterations = 7958.0 , err_train = 0.070 , err_test = 0.043
```
#### Batch Size 32 (MiniBatch) - Step Size  1e-6 (Medium) - Fold #1
![plots/batch-32_step-1e-06_fold-1.png](plots/batch-32_step-1e-06_fold-1.png "Batch Size 32 (MiniBatch) - Step 1e-6 - Fold #1")
#### Batch Size 32 (MiniBatch) - Step Size  1e-6 (Medium) - Fold #2
![plots/batch-32_step-1e-06_fold-2.png](plots/batch-32_step-1e-06_fold-2.png "Batch Size 32 (MiniBatch) - Step 1e-6 - Fold #2")
#### Batch Size 32 (MiniBatch) - Step Size  1e-6 (Medium) - Fold #3
![plots/batch-32_step-1e-06_fold-3.png](plots/batch-32_step-1e-06_fold-3.png "Batch Size 32 (MiniBatch) - Step 1e-6 - Fold #3")
#### Batch Size 32 (MiniBatch) - Step Size  1e-6 (Medium) - Fold #4
![plots/batch-32_step-1e-06_fold-4.png](plots/batch-32_step-1e-06_fold-4.png "Batch Size 32 (MiniBatch) - Step 1e-6 - Fold #4")
#### Batch Size 32 (MiniBatch) - Step Size  1e-6 (Medium) - Fold #5
![plots/batch-32_step-1e-06_fold-5.png](plots/batch-32_step-1e-06_fold-5.png "Batch Size 32 (MiniBatch) - Step 1e-6 - Fold #5")




### Batch Size 32 (MiniBatch) - Step Size 1e-5 (Big)
```
Average err_train = 0.03502782140028424
Average err_test = 0.04064824741488453
Stats by Fold:
Fold #1 : iterations = 4251.0 , err_train = 0.036 , err_test = 0.040
Fold #2 : iterations = 4287.0 , err_train = 0.034 , err_test = 0.045
Fold #3 : iterations = 4362.0 , err_train = 0.035 , err_test = 0.037
Fold #4 : iterations = 4268.0 , err_train = 0.033 , err_test = 0.054
Fold #5 : iterations = 4352.0 , err_train = 0.037 , err_test = 0.027
```
#### Batch Size 32 (MiniBatch) - Step Size 1e-5 (Big) - Fold #1
![plots/batch-32_step-1e-05_fold-1.png](plots/batch-32_step-1e-05_fold-1.png "Batch Size 32 (MiniBatch) - Step 1e-5 - Fold #1")
#### Batch Size 32 (MiniBatch) - Step Size 1e-5 (Big) - Fold #2
![plots/batch-32_step-1e-05_fold-2.png](plots/batch-32_step-1e-05_fold-2.png "Batch Size 32 (MiniBatch) - Step 1e-5 - Fold #2")
#### Batch Size 32 (MiniBatch) - Step Size 1e-5 (Big) - Fold #3
![plots/batch-32_step-1e-05_fold-3.png](plots/batch-32_step-1e-05_fold-3.png "Batch Size 32 (MiniBatch) - Step 1e-5 - Fold #3")
#### Batch Size 32 (MiniBatch) - Step Size 1e-5 (Big) - Fold #4
![plots/batch-32_step-1e-05_fold-4.png](plots/batch-32_step-1e-05_fold-4.png "Batch Size 32 (MiniBatch) - Step 1e-5 - Fold #4")
#### Batch Size 32 (MiniBatch) - Step Size 1e-5 (Big) - Fold #5
![plots/batch-32_step-1e-05_fold-5.png](plots/batch-32_step-1e-05_fold-5.png "Batch Size 32 (MiniBatch) - Step 1e-5 - Fold #5")



















### Batch Size inf (SGD) - Step Size 1e-7 (Small)
```
Average err_train = 0.04778974257571252
Average err_test = 0.05313534848255619
Stats by Fold:
Fold #1 : iterations = 6024.0 , err_train = 0.049 , err_test = 0.056
Fold #2 : iterations = 6171.0 , err_train = 0.047 , err_test = 0.062
Fold #3 : iterations = 6226.0 , err_train = 0.048 , err_test = 0.046
Fold #4 : iterations = 6004.0 , err_train = 0.047 , err_test = 0.070
Fold #5 : iterations = 6367.0 , err_train = 0.049 , err_test = 0.032
```
#### Batch Size inf (SGD) - Step Size  1e-7 (Small) - Fold #1
![plots/batch-1_step-1e-07_fold-1.png](plots/batch-1_step-1e-07_fold-1.png "Batch Size inf (SGD) - Step 1e-7 - Fold #1")
#### Batch Size inf (SGD) - Step Size  1e-7 (Small) - Fold #2
![plots/batch-1_step-1e-07_fold-2.png](plots/batch-1_step-1e-07_fold-2.png "Batch Size inf (SGD) - Step 1e-7 - Fold #2")
#### Batch Size inf (SGD) - Step Size  1e-7 (Small) - Fold #3
![plots/batch-1_step-1e-07_fold-3.png](plots/batch-1_step-1e-07_fold-3.png "Batch Size inf (SGD) - Step 1e-7 - Fold #3")
#### Batch Size inf (SGD) - Step Size  1e-7 (Small) - Fold #4
![plots/batch-1_step-1e-07_fold-4.png](plots/batch-1_step-1e-07_fold-4.png "Batch Size inf (SGD) - Step 1e-7 - Fold #4")
#### Batch Size inf (SGD) - Step Size  1e-7 (Small) - Fold #5
![plots/batch-1_step-1e-07_fold-5.png](plots/batch-1_step-1e-07_fold-5.png "Batch Size inf (SGD) - Step 1e-7 - Fold #5")




### Batch Size inf (SGD) - Step Size  1e-6 (Medium)
```
Average err_train = 0.028376515438427414
Average err_test = 0.0348464826120811
Stats by Fold:
Fold #1 : iterations = 3066.0 , err_train = 0.029 , err_test = 0.034
Fold #2 : iterations = 2971.0 , err_train = 0.028 , err_test = 0.036
Fold #3 : iterations = 3177.0 , err_train = 0.028 , err_test = 0.033
Fold #4 : iterations = 3174.0 , err_train = 0.027 , err_test = 0.047
Fold #5 : iterations = 3217.0 , err_train = 0.030 , err_test = 0.024
```
#### Batch Size inf (SGD) - Step Size  1e-6 (Medium) - Fold #1
![plots/batch-1_step-1e-06_fold-1.png](plots/batch-1_step-1e-06_fold-1.png "Batch Size inf (SGD) - Step 1e-6 - Fold #1")
#### Batch Size inf (SGD) - Step Size  1e-6 (Medium) - Fold #2
![plots/batch-1_step-1e-06_fold-2.png](plots/batch-1_step-1e-06_fold-2.png "Batch Size inf (SGD) - Step 1e-6 - Fold #2")
#### Batch Size inf (SGD) - Step Size  1e-6 (Medium) - Fold #3
![plots/batch-1_step-1e-06_fold-3.png](plots/batch-1_step-1e-06_fold-3.png "Batch Size inf (SGD) - Step 1e-6 - Fold #3")
#### Batch Size inf (SGD) - Step Size  1e-6 (Medium) - Fold #4
![plots/batch-1_step-1e-06_fold-4.png](plots/batch-1_step-1e-06_fold-4.png "Batch Size inf (SGD) - Step 1e-6 - Fold #4")
#### Batch Size inf (SGD) - Step Size  1e-6 (Medium) - Fold #5
![plots/batch-1_step-1e-06_fold-5.png](plots/batch-1_step-1e-06_fold-5.png "Batch Size inf (SGD) - Step 1e-6 - Fold #5")




### Batch Size inf (SGD) - Step Size 1e-5 (Big)
```
Average err_train = 0.017028992194655757
Average err_test = 0.026250552778676543
Stats by Fold:
Fold #1 : iterations = 2324.0 , err_train = 0.016 , err_test = 0.030
Fold #2 : iterations = 2225.0 , err_train = 0.019 , err_test = 0.017
Fold #3 : iterations = 2386.0 , err_train = 0.015 , err_test = 0.029
Fold #4 : iterations = 2344.0 , err_train = 0.016 , err_test = 0.039
Fold #5 : iterations = 2342.0 , err_train = 0.018 , err_test = 0.017
```
#### Batch Size inf (SGD) - Step Size 1e-5 (Big) - Fold #1
![plots/batch-1_step-1e-05_fold-1.png](plots/batch-1_step-1e-05_fold-1.png "Batch Size inf (SGD) - Step 1e-5 - Fold #1")
#### Batch Size inf (SGD) - Step Size 1e-5 (Big) - Fold #2
![plots/batch-1_step-1e-05_fold-2.png](plots/batch-1_step-1e-05_fold-2.png "Batch Size inf (SGD) - Step 1e-5 - Fold #2")
#### Batch Size inf (SGD) - Step Size 1e-5 (Big) - Fold #3
![plots/batch-1_step-1e-05_fold-3.png](plots/batch-1_step-1e-05_fold-3.png "Batch Size inf (SGD) - Step 1e-5 - Fold #3")
#### Batch Size inf (SGD) - Step Size 1e-5 (Big) - Fold #4
![plots/batch-1_step-1e-05_fold-4.png](plots/batch-1_step-1e-05_fold-4.png "Batch Size inf (SGD) - Step 1e-5 - Fold #4")
#### Batch Size inf (SGD) - Step Size 1e-5 (Big) - Fold #5
![plots/batch-1_step-1e-05_fold-5.png](plots/batch-1_step-1e-05_fold-5.png "Batch Size inf (SGD) - Step 1e-5 - Fold #5")


 



Samples are randomly chosen from `[-100,100)` range for `x` and `[-300,300)` range for `y`. 
Range of y is 3 times that of x, in order to make sure the final plot looks full 
while the target function plot line is fully visible for that range.
 
Random seed can be set by changing the global variable `RANDOM_SEED`
By default it is kept as None (numpy requests a random seed from os)

Below are the results and plots for each step with `RANDOM_SEED=12345` 

### Step 1
```
Finished calculation in 25 iterations
Calculated weights are: [  0.         531.02365676 197.25030511]
Decision Boundary is y = -2.69x + -0.00
```
![part1_step1.png](part1_step1.png "Part1 Step1 Plot")

\
&nbsp;

### Step 2
```
Finished calculation in 119 iterations
Calculated weights are: [ -50.         1122.35262344  371.32963463]
Decision Boundary is y = -3.02x + 0.13
```
![part1_step2.png](part1_step2.png "Part1 Step2 Plot")

\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
&nbsp;
### Step 3
```
Finished calculation in 23867 iterations
Calculated weights are: [-6418.         20547.05277251  6856.03386458]
Decision Boundary is y = -3.00x + 0.94
```
![part1_step3.png](part1_step3.png "Part1 Step3 Plot")
    
### Part 1 Overview

When we check out above results, we can see 
that the decision boundary gets closer to the original target function 
as the number of sampled points are increased.

Also, as the number of sampled points are increased, 
PLA needs to make much more iterations to fully converge.

See below table for comparing above results.


|Step| Samples | Iterations |y = -3x + 1|
|--|--|--|--|
|1|50|25|y = -2.69x + -0.00|
|2|100|119|y = -3.02x + 0.13|
|3|5000|23867|y = -3.00x + 0.94|


## Part 2

Implemented closed form solution, so there will not be any loss over iterations graphs. 

Chose to apply 5-fold cross validation, 
this can be changed via the global variable, `PART2_S_FOLD_S_VALUE` 

Created a few supplementary graphs in order to get a better grasp of how well it predicts, and which lambda value to choose.

Timing includes cross validation, but none of the extra-supplementary graph generations.

Set `PART2_PLOT_PREDICTIONS=True` to create `part2_stepN_predictions.png` which 
shows predictions vs actual target values for each sample.

Set `PART2_PLOT_LAMBDA_VALUES=True` to create `part2_stepN_regularization.png`
a graph of training & testing results (root mean square errors) over varying ln-lambda value. 
This will be same for step2 and step3 as expected, because they are only varied by regularization. 

Chosen the lambda value for regularization as `e^-10` yet 
it is neither beneficial nor harmful, as it seems. 

When we choose to do 2-fold cross validation instead, 
regularization becomes useful because of tolerating 
the lack of many training values causing over-fitting. 
But such analysis is out of this report's scope. 
If interested, one can set `PART2_S_FOLD_S_VALUE=2` and 
`PART2_PLOT_LAMBDA_VALUES=True` and run the program on part2, 
and check out the generated graph, `part2_stepN_regularization.png`.   

\
\
\
\
\
&nbsp;

### Step 1
```
Applying 5-fold cross validation
There were 100 independent variables and 1 dependent variables
There were 1000 samples in total
Completed in 3.160 milliseconds
Average Erms_train = 36.257771814312136
Average Erms_test = 41.3584787667631
```
![part2_step1_predictions.png](part2_step1_predictions.png "Part2 Step1 Predictions Plot")
![part2_step1_regularization.png](part2_step1_regularization.png "Part2 Step1 Regularization Plot")
### Step 2
```
Applying 5-fold cross validation
There were 500 independent variables and 1 dependent variables
There were 1000 samples in total
Completed in 48.580 milliseconds
Average Erms_train = 23.6657974544233
Average Erms_test = 65.28337321386573
```
![part2_step2_predictions.png](part2_step2_predictions.png "Part2 Step2 Predictions Plot")
![part2_step2_regularization.png](part2_step2_regularization.png "Part2 Step2 Regularization Plot")
### Step 3
```
Applying 5-fold cross validation
There were 500 independent variables and 1 dependent variables
There were 1000 samples in total
Completed in 52.595 milliseconds
Average Erms_train = 23.665797456855017
Average Erms_test = 65.28335649367837
```
![part2_step3_predictions.png](part2_step3_predictions.png "Part2 Step3 Predictions Plot")
![part2_step3_regularization.png](part2_step3_regularization.png "Part2 Step3 Regularization Plot")


### Part 2

Below is the given input data

|Name|GiveBirth|CanFly|LiveInWater|HaveLegs|Class|
|--|--|--|--|--|--|
| human | yes | no | no | yes | mammals |
| python | no | no | no | no | non-mammals |
| salmon | no | no | yes | no | non-mammals |
| whale | yes | no | yes | no | mammals |
| frog | no | no | sometimes | yes | non-mammals |
| komodo | no | no | no | yes | non-mammals |
| bat | yes | yes | no | yes | mammals |
| pigeon | no | yes | no | yes | non-mammals |
| cat | yes | no | no | yes | mammals |
| leopard shark | yes | no | yes | no | non-mammals |
| turtle | no | no | sometimes | yes | non-mammals |
| penguin | no | no | sometimes | yes | non-mammals |
| porcupine | yes | no | no | yes | mammals |
| eel | no | no | yes | no | non-mammals |
| salamander | no | no | sometimes | yes | non-mammals |
| gila monster | no | no | no | yes | non-mammals |
| platypus | no | no | no | yes | mammals |
| owl | no | yes | no | yes | non-mammals |
| dolphin | yes | no | yes | no | mammals |
| eagle | no | yes | no | yes | non-mammals |
| test | yes | no | yes | no | ??? |

When we count mammal vs non-mammal count per class we can obtain below table:

| feature | # mammals | # non-mammals |
|--|--|--|
| GiveBirth (yes) |6|1|
| GiveBirth (no) |1|12|
| CanFly (yes) |1|3|
| CanFly (no) |6|10|
| LiveInWater (yes) |2|3|
| LiveInWater (sometimes) |0|4|
| LiveInWater (no) |5|6|
| HaveLegs (yes) |5|9|
| HaveLegs (no) |2|4|


We are asked to guess whether the "test" belongs to "mammals" or "non-mammals" class.
```
P(mammals | GiveBirth=yes, CanFly=no, LiveInWater=yes, HaveLegs=no)
P(non-mammals | GiveBirth=yes, CanFly=no, LiveInWater=yes, HaveLegs=no)
```
Apply Bayes Rule:
```
P(Class|GiveBirth, CanFly, LiveInWater, HaveLegs)
∝ 
P(GiveBirth, CanFly, LiveInWater, HaveLegs|Class) * P(Class)
```
We can rewrite then rewrite it as follows:
```
P(GiveBirth, CanFly, LiveInWater, HaveLegs|Class) * P(Class)
∝ 
P(GiveBirth|Class) * P(CanFly|Class)
* P(LiveInWater|Class) * P(HaveLegs|Class) *  P(Class)
```
For our case:
```
P(mammals | GiveBirth=yes, CanFly=no, LiveInWater=yes, HaveLegs=no)
∝ 
P(GiveBirth=yes|mammals) * P(CanFly=no|mammals)
* P(LiveInWater=yes|mammals) * P(HaveLegs=no|mammals) 
* P(mammals)

and

P(non-mammals | GiveBirth=yes, CanFly=no, LiveInWater=yes, HaveLegs=no)
∝ 
P(GiveBirth=yes|non-mammals) * P(CanFly=no|non-mammals)
* P(LiveInWater=yes|non-mammals) * P(HaveLegs=no|non-mammals) 
* P(non-mammals)
```

Using the frequency table we can compute all these:
```
P(GiveBirth=yes|mammals) = 6/7
P(CanFly=no|mammals) = 6/7
P(LiveInWater=yes|mammals) = 2/7
P(HaveLegs=no|mammals) = 2/7
P(mammals) = 7/20
--------------------
P(GiveBirth=yes|non-mammals) = 1/13
P(CanFly=no|non-mammals) = 10/13
P(LiveInWater=yes|non-mammals) = 3/13
P(HaveLegs=no|non-mammals) = 4/13
P(non-mammals) = 13/20
``` 
When we multiply these values as in previous formula we obtain below results:
```
P(mammals | GiveBirth=yes, CanFly=no, LiveInWater=yes, HaveLegs=no)
∝ 
6/7 * 6/7 * 2/7 * 2/7 * 7/20 = 36/1715 ~ 0.0209912536

and

P(non-mammals | GiveBirth=yes, CanFly=no, LiveInWater=yes, HaveLegs=no)
∝ 
1/13 * 10/13 * 3/13 * 4/13 * 13/20 = 30/10985 ~ 0.00273099681
```
Now we can compare these two results to make our decision:
```
36/1715 > 30/10985
0.0209912536 > 0.00273099681
P(mammals | GiveBirth=yes, CanFly=no, LiveInWater=yes, HaveLegs=no)
>
P(non-mammals | GiveBirth=yes, CanFly=no, LiveInWater=yes, HaveLegs=no)
```
Thus, we would guess that "test" belongs to "mammals" class.