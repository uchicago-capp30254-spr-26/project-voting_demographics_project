import numpy as np
import pandas as pd
import itertools

import matplotlib.pylab as plt
from data_processing import ProcessedData

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate 
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import warnings
from pathlib import Path

# add path for visualizations
BASE_DIR = Path(__file__).resolve().parent.parent
PATH = BASE_DIR / "final_report" / "plots"

# supress penalty deprecation warning, since it still works with liblinear
warnings.filterwarnings('ignore', message=".*penalty.*deprecated.*")

class LinearModel:

    def __init__(self):
        self.data = ProcessedData(random_state=1234)
        self.model = None
        self.scores = None
        self.coef_df = None

    def logreg_variations(self, X, y, penalty = 'l2', C = 1.0, cv = 5, class_weight = None):
        '''
        Run cross validation for the model.
        '''
        logreg = LogisticRegression(penalty=penalty, C=C, solver='liblinear', class_weight=class_weight)
        scores = cross_validate(logreg, X, y, cv=cv, return_train_score=True, scoring = 'f1')

        return scores

    def find_hyperparams(self):
        '''
        Use the hyperparameters of interest to run cross validation and find the
        best model.
        '''
        regularizers = ['l1', 'l2']
        C_list = list(np.logspace(-4, 1, 50))  # 0.0001 to 10, 50 points
        # class_weight='balanced' can help with imbalanced dataset, which
        # may bias predictions toward majority class (voted)
        class_weight_list = [None, 'balanced']

        hyperparam_combos = list(itertools.product(regularizers, C_list, class_weight_list))

        # create a dictionary to track the scores for different combinations
        scores = {}

        for penalty, C, class_weight in hyperparam_combos:
            # validate the model with the different combinations of eta and lambda
            all_scores = self.logreg_variations(self.data.train_x, self.data.train_y, penalty=penalty, C=C, class_weight=class_weight)
            mean_train_score = np.mean(all_scores["train_score"])
            mean_test_score =  np.mean(all_scores["test_score"])

            scores[(penalty, C, class_weight)] = (mean_train_score, mean_test_score)

        # save to the model to plot later
        self.scores = scores
        (best_penalty, best_C, best_class_weight), (mean_train, mean_test) = max(scores.items(), key=lambda item: item[1][1])

        # print results to the terminal
        print(f'''----- Cross Validation Results -----
              \n penalty = {best_penalty} 
              \n C = {best_C} 
              \n class_weight = {best_class_weight} 
              \n mean train score: {mean_train} 
              \n mean test score: {mean_test} \n''')
        
        # return the best hyperparameters to pass into the model
        return best_penalty, best_C, best_class_weight, mean_train, mean_test
    
    def plot_hyperparams(self, scores):
        '''
        Plot the cross validation results of all hyperparameter combinations.
        '''
        combos = [('l1', None), ('l1', 'balanced'), ('l2', None), ('l2', 'balanced')]
        colors = ['blue', 'orange', 'green', 'red']

        fig, ax = plt.subplots(figsize=(10, 6))

        for (penalty, class_weight), color in zip(combos, colors):
            filtered = {C: v for (p, C, cw), v in scores.items() 
                        if p == penalty and cw == class_weight}
            
            C_vals = sorted(filtered.keys())
            # only including the test scores because train and test scores were
            # virtually the same, so the lines completely overlapped
            test_scores = [filtered[C][1] for C in C_vals]

            label = f"{penalty}, {class_weight}"
            ax.plot(C_vals, test_scores, color=color, linestyle='-', label=f"{label}")

        ax.set_xscale('log')  # C_list uses logspace
        ax.set_xlabel('C')
        ax.set_ylabel('F1 Score')
        ax.set_title('Hyperparameter Test Scores')
        ax.legend()
        plt.tight_layout()
        plt.savefig(PATH / 'logistic_regression_hyperparams.png')
    
    def train(self):
        '''
        Train the logistic regression model.
        '''
        best_penalty, best_C, best_class_weight, _, _ = self.find_hyperparams()
        
        self.model = LogisticRegression(penalty=best_penalty, C=best_C, solver='liblinear', class_weight=best_class_weight)
        self.model.fit(self.data.train_x, self.data.train_y)

        # save the coefficients to a dataframe for returning and plotting results
        self.coef_df = pd.DataFrame(zip(self.data.feature_order, self.model.coef_[0]), columns=['features', 'coefficients'])
        self.coef_df.to_csv(PATH / 'logistic_regression_coefficients.csv', index=False)

        return self.model
    
    def plot_coefficients(self):
        '''
        Plot the coefficients for the final report.
        '''
        # drop zero coefficients
        coefs = self.coef_df[self.coef_df['coefficients'] != 0]
        coefs = coefs.sort_values('coefficients')

        # print the top coefficients to the terminal
        print("---- Training Results ----- \n")
        top_features = coefs.reindex(coefs['coefficients'].abs().sort_values(ascending=False).index).head(5)
        print(f"Top features: \n {top_features} \n")

        # plot the coefficients for the final report
        colors = ['red' if c > 0 else 'blue' for c in coefs['coefficients']]

        plt.figure(figsize=(10, 8))
        plt.barh(coefs['features'], coefs['coefficients'], color=colors)
        plt.axvline(x=0, color='black', linewidth=0.8)
        plt.ylabel('Feature')
        plt.xlabel('Coefficient')
        plt.title('Logistic Regression Coefficients')
        plt.tight_layout()
        plt.savefig(PATH / 'logistic_regression_coefficients.png')  

    def test(self):
        '''
        Test the logistic regression model.
        '''
        predictions = self.model.predict(self.data.test_x)
        # calculate the F1 score
        f1 = f1_score(self.data.test_y, predictions)

        # create the confustion matrix
        cm = confusion_matrix(self.data.test_y, predictions)
        # access false positive and false negative rates
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

        # plot the confusion matrix and save file for the final report
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['voted', 'not_voted'])
        display.plot()
        plt.savefig(PATH / 'logistic_regression_confusion_matrix.png')

        # print the results
        print("----- Test Results ----- \n")
        print(f"F1 Score: {f1} \n")

        print(f"Confusion matrix: \n {cm} \n")
        print(f"False negative rate: {fn / (fn + tp)} \n")
        print(f"False positive rate: {fp / (fp + tn)} \n")
        
        # print the classification report that includes other metrics
        report = classification_report(self.data.test_y, predictions)
        print(f"Other test metrics: \n {report}")

 

if __name__ == "__main__":
    lm = LinearModel()
    lm.train()
    lm.plot_coefficients()
    lm.plot_hyperparams(lm.scores)
    lm.test()