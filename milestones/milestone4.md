# <Voting Demographics>
# Milestone 4

## Progress on the Project
We cleaned the data by removing all rows with missing values and selecting relevant features. During this process, we performed feature engineering, such as grouping ages into 5-year intervals, creating household income categories based on medians, and generating a new feature representing household income per person.  
For the linear model, we selected a Logistic Regression model and have nearly completed the coding step. We are currently working on visualization and making minor revisions to the model to reflect recent updates to our feature set.  
For the non-linear model, we chose a random forest model and have begun implementing it.  
We have not yet started working on neural networks.  

## Non-linear Model
We chose a random forest model because it is relatively easy to interpret compared to other non-linear models, and it provides useful metrics such as feature importance, which help us understand the impact of each variable. Additionally, random forests handle non-linear relationships effectively and perform well with mixed types of features, making them suitable for our dataset.  
We did not choose K-Means because it is an unsupervised learning method, whereas our dataset is labeled and our objective is to predict outcomes.  
K-Nearest Neighbors (K-NN) was also not selected because our training data is imbalanced, with significantly more “voted” observations than “not voted” ones. This imbalance can bias K-NN predictions toward the majority class.

## Interpretation
We aim to interpret the model’s output by examining which features are most important in determining voter turnout. In a random forest, features that are used for the first splits across many trees are considered more important, as they contribute the most to reducing entropy and increasing information gain. This indicates that these features play a significant role in separating the data and predicting voting behavior.  
As we move further down the trees, the importance of features generally decreases because they contribute less to reducing uncertainty. Therefore, features appearing at higher levels of the trees are more influential in determining the model’s predictions.

## Metrics of Analysis
We plan to evaluate our model’s performance on the test data using a confusion matrix and related metrics such as the F1 score. Initially, we considered using accuracy as our primary metric; however, because our dataset is imbalanced, we observed consistently high accuracy across different hyperparameter settings, making it less informative. Therefore, we chose to use the F1 score, which better captures the balance between precision and recall in imbalanced classification problems.

## Organizing Code
At this stage, we are developing our code in Jupyter Notebook. Before submission, we plan to organize our project into separate folders containing .py files and CSV files for different components, including data cleaning, the linear model, the non-linear model, and the neural network model. Visualizations will be included within the respective model files to keep related outputs and analyses together.