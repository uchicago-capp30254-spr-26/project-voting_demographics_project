
# Voting Demographics


## Abstract

This project explores how voter demographics influence participation in elections using voter turnout data from the 2024 U.S. presidential election. Specifically, we use demographic variables to predict whether an individual is likely not to vote, addressing the policy issue of unequal voter turnout across different groups.

Our analysis applies three machine learning models, Logistic Regression, Random Forests, and Neural Networks, to predict voter turnout based on demographic characteristics. We evaluate model performance using the F1 score to account for class imbalance in the dataset, defining not_voted as 1 and voted as 0 in order to prioritize identifying individuals who do not participate.

The results, presented in a final report, highlight the predictive power of demographic factors in explaining voter participation. This study provides insights into disparities in electoral engagement and may help inform targeted outreach strategies aimed at increasing turnout among underrepresented populations.


## Members

- Srinath Rao Pathangae (<srinath@uchicago.edu>)
- Ashanti Hatchett (<ashantih@uchicago.edu>)
- Kumhyun Song (<songk1122@uchicago.edu>) 


## Instructions on How to Run the Code

1. In the `project` folder, run the following commands in the terminal to see the results of each model:
- Logistic Regression: `python logistic_regression.py`
- Random Forest: `python random_forest.py`
- Neural Networks: `python neural_networks.py`   
After each script finishes running, the results will be printed in the terminal, and all plots will be saved in the `final_report/plots` folder.

2. To view the cleaned data, run the following command in the `project` folder: `python data_cleaning.py`  
The cleaned dataset (`cleaned_data.csv`) will be saved in the data folder.

3. To generate feature-related plots, run the following command in the `project` folder: `python data_plots.py`
All plots will be saved in the `final_report/plots` folder.


## Data Source

Name: IPUMS CPS (Current Population Survey) November 2024 Voting Supplement  

Source URL: https://cps.ipums.org/  
  - To download the raw data, users must create an account and log in. For this project, the raw data is provided in the data folder.   

Source Type: Bulk Data   

Summary: The CPS is a nationally representative survey conducted by the U.S. Census Bureau. The voting supplement includes individual-level responses related to voting behavior and voter turnout in the 2024 election cycle. It also contains demographic and socioeconomic variables such as age, gender, education, income, and employment status, which can be used to model and predict voter participation.


## Project Report

[Open PDF](./final_report/final_report.pdf)