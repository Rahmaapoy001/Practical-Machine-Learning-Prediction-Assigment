# Practical-Machine-Learning-Prediction-Assigment

## Introduction

The goal of this assignment is to predict the manner in which participants performed a specific physical activity using data collected from wearable accelerometers. The data was gathered from devices placed on the belt, forearm, arm, and dumbbell of subjects while they performed barbell lifts. The target variable is `classe`, which represents five different classes of movement (A-E). The task is to build a machine learning model that can accurately classify these movements.

## Algorithm Description

To build the predictive model, we used the **Random Forest** algorithm, which is suitable for classification tasks with high dimensional data and can handle noisy features well. The data preprocessing steps included:

- Removing near-zero variance variables  
- Eliminating variables with too many missing values  
- Dropping non-predictive variables such as timestamps and user names  
- Converting the target variable `classe` into a factor

The dataset was split into two parts:
- **Training set** (70%)
- **Validation set** (30%)

Using the `caret` package in R, we trained the model with 5-fold cross-validation to ensure robustness.

## Cross-Validation and Error Estimation

We used `trainControl(method = "cv", number = 5)` in the `train()` function to perform 5-fold cross-validation during model training. This method ensures that the model is validated on multiple subsets of the training data, reducing the risk of overfitting.

After training, the model's accuracy on the validation set was calculated, and the **out-of-sample error** was estimated using:

```r
out_of_sample_error <- 1 - accuracy
```

Where `accuracy` is the percentage of correct predictions on the validation set. A confusion matrix was also used to evaluate the performance across all classes.

## Conclusion

The Random Forest model achieved high accuracy and a low out-of-sample error, demonstrating strong predictive performance for the `classe` variable. This validates the use of ensemble methods and careful preprocessing in handling real-world sensor data. The final model was also tested on the 20 test cases provided and produced reliable predictions.
