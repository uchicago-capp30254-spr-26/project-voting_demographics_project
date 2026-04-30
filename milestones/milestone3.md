# <Voting Demographics>
# MileStone 3

# Linear Model Selection
Our prediction model is to understand whether features related to identified characteristics can explain their voting behavior in terms of whether they vote or not, and the medium of voting.

The linear model we picked for this purpose is the Logistic Regression model. We picked this because the classification we target is binary and we can have additional information in the form of probabilities apart from the final classification.

# Regularizer
We picked the L2 (Ridge) Regression as we have limited number of features (7) and expect multicollinearity between them. We shall use the Train (70%) set, Validation (10%) set and Test (20%) sets to come up with a learning rate and hyper-parameter, based on accuracy of the model at various combinations.


# Interpretation
We would like to explain the output of this model based on the weight of coefficients for our features: Age, Family Income, Race, Region, Education. In our project, the weights are interpreted as follows: because features are encoded using dummy variables, “young” serves as the baseline category for AGE. Therefore, a positive coefficient on “elderly” indicates that elderly individuals are more likely to vote than young individuals, holding all other variables constant.

# Metrics of Analysis
We analyze the performance of our model on test data to come up with the Confusion Matrix and associated metrics like Precision as we aim to minimize false positives. False positives are especially important in our project because the goal is to support policies aimed at increasing turnout among individuals with a low likelihood of voting. Misclassifying such individuals as likely voters would result in missing key targets for intervention. We would also look into overall metrics like F1 to balance accuracy and precision.

# Organizing Code
At this stage, we plan to have separate folders with .py files and csv files for data cleaning, linear model, non-linear model and visualization.