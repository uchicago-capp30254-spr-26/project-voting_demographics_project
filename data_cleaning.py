import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PATH = "data/cps_00006.dat"
COLSPECS = [(0,4), (4,9), (9,11), (11,21), (21,35), (35,37), (37,39), (39,42),
            (42,44), (44,58), (58,73), (73,87), (87,89), (89,90), (90,93),
            (93,94), (94,97), (97,99), (99,101), (101,104)]
NAMES = ["YEAR", "SERIAL", "MONTH", "HWTFINL", "CPSID", "GQTYPE", "REGION",
         "FAMINC", "PERNUM", "WTFINL", "CPSIDV", "CPSIDP", "AGE", "SEX", "RACE",
         "NATIVITY", "EDUC", "VOTED", "VOTEHOW", "VOTERES"]

class Dataset:
    """
    Load and preprocess IPUMS Current Population Survey dataset containing
    demographics and voting turnout of individuals.
    """

    def __init__(self, path, colspecs, colnames):
        self.data = pd.read_fwf(path, colspecs=colspecs, names=colnames)

    def _exclude_codes_(self, colname, codes):
        """
        Remove rows where the column contains invalid or non-informative codes.
        """
        self.data = self.data[~self.data[colname].isin(codes)]
    
    def _change_codes_(self, colname, changes):
        """
        Replace codes in a column using a changes dict.
        """
        self.data[colname] = self.data[colname].replace(changes)
    
    def _extract_feature_(self, colname, group_details):
        """
        Group numeric values of a column into categories.
        """
        for group, (low, high) in group_details.items():
            self.data.loc[self.data[colname].between(low, high), colname] = group


def cleaned_data():
    data = Dataset(PATH, COLSPECS, NAMES)

    ## VOTED: 1 Did not vote, 2 Voted
    data._exclude_codes_("VOTED", [96, 97, 98, 99])
    data._change_codes_("VOTED", {1: "not_voted", 2: "voted"})

    ## VOTEHOW: 1 In person, 2 By mail, 99 Did not vote
    data._exclude_codes_("VOTEHOW", [96, 97, 98])
    data._change_codes_("VOTEHOW", {1: "in_person", 2: "by_mail", 99: "not_voted"})

    ## AGE: 100 Young (18-39), 200 Middle Age (40-64), 300 Elderly (65+)
    age_group = {900: (0, 17), 100: (18, 39), 200: (40, 64), 300: (65, 85)}
    data._extract_feature_("AGE", age_group)
    data._exclude_codes_("AGE", [900])
    data._change_codes_("AGE", {100: "young", 200: "middle_age", 300: "elderly"})

    ## SEX: 1 Male, 2 Female
    data._change_codes_("SEX", {1: "male", 2: "female"})

    ## RACE: 100 White, 200 Black, 300 American Indian/Aleut/Eskimo, 651 Asian, 900 Others
    data._exclude_codes_("RACE", [999])
    data._change_codes_("RACE", {code : "others" for code in [801, 802, 803, 652, 804, 809, 813, 805, 810, 806, 811, 812, 817, 816, 808, 815, 807, 814, 830, 818, 820]})
    data._change_codes_("RACE", {100: "white", 200: "black", 300: "indian_aleut_eskimo", 651: "asian"})

    ## EDUCATION: 100 Below high school, 200 High school graduate, 300 College graduate, 400 Master's degree or higher
    data._exclude_codes_("EDUC", [1])
    education_group = {100: (2,72), 200: (73, 81), 300: (91, 122), 400: (123, 125)}
    data._extract_feature_("EDUC", education_group)
    data._change_codes_("EDUC", {200: "hs_grad", 300: "college_grad", 400: "master_higher"})

    ## NATIVITY: 100 At least one parent native-born, 200 Individual foreign born/Both parents foreign-born
    data._exclude_codes_("NATIVITY", [0])
    nativity_group = {100: (1, 3), 200: (4, 5)}
    data._extract_feature_("NATIVITY", nativity_group)
    data._change_codes_("NATIVITY", {100: "native_born", 200: "foreign_born"})

    ## REGION: 1 Northeast, 2 Midwest, 3 South, 4 West
    region_group = {1: (11, 12), 2: (21, 22), 3: (31, 33), 4: (41, 42)}
    data._extract_feature_("REGION", region_group)
    data._change_codes_("REGION", {1: "northeast", 2: "midwest", 3: "south", 4: "west"})

    ## HOUSEHOLD INCOME: 1 Lower ($0-74,999), 2 Middle ($75,000-149,999), 3 Upper ($150,000+)
    data._exclude_codes_("FAMINC", [995, 996, 997, 999])
    householdincome_group = {1: (100,830), 2: (841, 842), 3: (843, 843)}
    data._extract_feature_("FAMINC", householdincome_group)
    data._change_codes_("FAMINC", {1: "lower", 2: "middle", 3: "upper"})

    using_cols = ["VOTED", "VOTEHOW", "AGE", "SEX", "RACE", "EDUC", "NATIVITY", "REGION", "FAMINC"]

    df = data.data[using_cols].copy()

    df["AGE"] = pd.Categorical(df["AGE"], categories=["young", "middle_age", "elderly"], ordered=True)
    df["SEX"] = pd.Categorical(df["SEX"], categories=["male", "female"], ordered=True)
    df["RACE"] = pd.Categorical(df["RACE"], categories=["white", "black", "indian_aleut_eskimo", "asian", "others"], ordered=True)
    df["EDUC"] = pd.Categorical(df["EDUC"], categories=["hs_grad", "college_grad", "master_higher"], ordered=True)
    df["NATIVITY"] = pd.Categorical(df["NATIVITY"], categories=["native_born", "foreign_born"], ordered=True)
    df["REGION"] = pd.Categorical(df["REGION"], categories=["northeast", "midwest", "south", "west"], ordered=True)
    df["FAMINC"] = pd.Categorical(df["FAMINC"], categories=["lower", "middle", "upper"], ordered=True)

    return df


if __name__ == "__main__":
    result = cleaned_data()
    result.to_csv("data/cleaned_data.csv", index = False)