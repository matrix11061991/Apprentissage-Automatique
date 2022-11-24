# SVM Algorithm
Example of using Support Vector Machine
# What is SVM ?
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

> Effective in high dimensional spaces.

> Still effective in cases where number of dimensions is greater than the number of samples.

> Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

> Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:
>If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

>SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

