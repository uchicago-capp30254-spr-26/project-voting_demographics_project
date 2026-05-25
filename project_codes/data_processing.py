import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from data_cleaning import cleaned_data

class ProcessedData:
    """
    Prepare the dataset into training, validation, and test sets.
    """
    def __init__(self, random_state = 123):
        self.data = cleaned_data()
        y_df = self.data["VOTED"].map({"voted": 0, "not_voted": 1}).to_numpy()
        X_df = self.data.drop(columns=["VOTED"])

        categorical_feature = ["SEX", "RACE", "EDUC", "EMPSTAT", "NATIVITY", "REGION",
                               "METRO", "MARST", "DIFFMOB"]
        numerical_feature = ["AGE", "NCHILD", "INCOME_PER_PERSON"]

        ## training set 70%, validation set 10%, test set 20%
        #added stratify in addition to neural networks code
        train_x, rest_x, self.train_y, rest_y = train_test_split(
            X_df, y_df, test_size=0.3, random_state=random_state, stratify=y_df)
        val_x, test_x, self.val_y, self.test_y = train_test_split(
            rest_x, rest_y, test_size=(2/3), random_state=random_state, stratify=rest_y)
        
        self.ohe = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()

        train_x_categorical = self.ohe.fit_transform(train_x[categorical_feature])
        train_x_numerical = self.scaler.fit_transform(train_x[numerical_feature])
        val_x_categorical = self.ohe.transform(val_x[categorical_feature])
        val_x_numerical = self.scaler.transform(val_x[numerical_feature])
        test_x_categorical = self.ohe.transform(test_x[categorical_feature])
        test_x_numerical = self.scaler.transform(test_x[numerical_feature])

        self.train_x = np.concatenate([train_x_categorical, train_x_numerical], axis=1)
        self.val_x = np.concatenate([val_x_categorical, val_x_numerical], axis=1)
        self.test_x = np.concatenate([test_x_categorical, test_x_numerical], axis=1)

        categorical_feature_name = self.ohe.get_feature_names_out(categorical_feature)
        numerical_feature_name = self.scaler.get_feature_names_out(numerical_feature)
        self.feature_order = np.concatenate([categorical_feature_name, numerical_feature_name])