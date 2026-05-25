from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PATH = BASE_DIR / "data" / "cps_00007.dat"

COLSPECS = [(0,4), (4,9), (9,11), (11,21), (21,35), (35,37), (37,39), (39,40),
            (40,43), (43,45), (45,59), (59,74), (74,88), (88,90), (90,91),
            (91,94), (94,95), (95,97), (97, 98), (98,99), (99,101), (101,104),
            (104,105), (105,107), (107,109), (109,112)]

NAMES = ["YEAR", "SERIAL", "MONTH", "HWTFINL", "CPSID", "GQTYPE", "REGION",
         "METRO", "FAMINC", "PERNUM", "WTFINL", "CPSIDV", "CPSIDP", "AGE",
         "SEX", "RACE", "MARST", "FAMSIZE", "NCHILD", "NATIVITY", "EMPSTAT",
         "EDUC", "DIFFMOB", "VOTED", "VOTEHOW", "VOTERES"]

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
    """
    Clean the dataset, performs feature engineering, and creates the final dataset used for modeling.
    """
    
    data = Dataset(PATH, COLSPECS, NAMES)

    ## VOTED: 1 Did not vote, 2 Voted
    data._exclude_codes_("VOTED", [96, 97, 98, 99])
    data._change_codes_("VOTED", {1: "not_voted", 2: "voted"})

    ## (No longer using it) VOTEHOW: 1 In person, 2 By mail, 99 Did not vote
    ## data._exclude_codes_("VOTEHOW", [96, 97, 98])
    ## data._change_codes_("VOTEHOW", {1: "in_person", 2: "by_mail", 99: "not_voted"})

    ## SEX: 1 Male, 2 Female
    data._change_codes_("SEX", {1: "male", 2: "female"})

    ## RACE: 100 White, 200 Black, 300 American Indian/Aleut/Eskimo, 651 Asian, 900 Others
    data._exclude_codes_("RACE", [999])
    data._change_codes_("RACE", {code: "others" for code in [801, 802, 803, 652, 804, 809, 813, 805, 810, 806, 811, 812, 817, 816, 808, 815, 807, 814, 830, 818, 820]})
    data._change_codes_("RACE", {100: "white", 200: "black", 300: "indian_aleut_eskimo", 651: "asian"})

    ## EDUCATION: 100 Below high school graduate, 200 High school graduate, 300 College graduate, 400 Master's degree or higher
    data._exclude_codes_("EDUC", [1])
    education_group = {100: (2, 72), 200: (73, 81), 300: (91, 122), 400: (123, 125)}
    data._extract_feature_("EDUC", education_group)
    data._change_codes_("EDUC", {100: "below_hs", 200: "hs_grad", 300: "college_grad", 400: "master_higher"})

    ## EMPLOYMENT STATUS: 100 Employed, 200 Unemployed, 300 Retired, 400 Not in labor force
    data._exclude_codes_("EMPSTAT", [0])
    employment_group = {100: (1, 12), 200:(21, 22), 300: (36, 36), 400: (32, 34)}
    data._extract_feature_("EMPSTAT", employment_group)
    data._change_codes_("EMPSTAT", {100: "employed", 200: "unemployed", 300: "retired", 400: "not_in_labor_force"})

    ## NATIVITY: 100 At least one parent native-born, 200 Individual foreign born/Both parents foreign-born
    data._exclude_codes_("NATIVITY", [0])
    nativity_group = {100: (1, 3), 200: (4, 5)}
    data._extract_feature_("NATIVITY", nativity_group)
    data._change_codes_("NATIVITY", {100: "native_born", 200: "foreign_born"})

    ## REGION: 1 Northeast, 2 Midwest, 3 South, 4 West
    region_group = {1: (11, 12), 2: (21, 22), 3: (31, 33), 4: (41, 42)}
    data._extract_feature_("REGION", region_group)
    data._change_codes_("REGION", {1: "northeast", 2: "midwest", 3: "south", 4: "west"})

    ## METROPOLITAN: 10 Not in metropolitan area, 20 In metropolitan area
    data._exclude_codes_("METRO", [0])
    metropolitan_group = {10: (1, 1), 20: (2, 4)}
    data._extract_feature_("METRO", metropolitan_group)
    data._change_codes_("METRO", {10: "not_metropolitan", 20: "metropolitan"})

    ## MARITAL STATUS: 10 Has spouse, 20 No spouse
    data._exclude_codes_("MARST", [9])
    marital_group = {10: (1, 1), 20: (2, 6)}
    data._extract_feature_("MARST", marital_group)
    data._change_codes_("MARST", {10: "has_spouse", 20: "no_spouse"})

    ## DISABILITY LIMITING MOBILITY
    data._exclude_codes_("DIFFMOB", [0])
    data._change_codes_("DIFFMOB", {1: "no_mobility_limitation", 2: "mobility_limitation"})

    ## AGE: Categorized into 5-year intervals and represented by the median age
    age_group = {900: (0, 17), 119: (18, 20)}
    key = 123
    for age in range(21, 86, 5):
        age_group[key] = (age, age + 4)
        key += 5
    data._extract_feature_("AGE", age_group)
    data._exclude_codes_("AGE", [900])
    age_categories = {k: k-100 for k in age_group.keys() if k >= 100 and k != 900}
    data._change_codes_("AGE", age_categories)

    ## FAMILY SIZE: Number of family members in household
    ## (No feature engineering required)

    ## CHILDREN: Number of children in household
    ## (No feature engineering required)

    ## HOUSEHOLD INCOME: Represented by the median income within each category
    householdincome_group = {2500: (100, 100), 6250: (210, 210), 8750: (300, 300),
                             11250: (430, 430), 13750: (470, 470), 17500: (500, 500),
                             22500: (600, 600), 27500: (710, 710), 32500: (720, 720),
                             37500:(730, 730), 45000: (740, 740), 55000: (820, 820),
                             67500: (830, 830), 87500: (841,841), 125000: (842, 842),
                             150000: (843, 843)}
    data._extract_feature_("FAMINC", householdincome_group)

    df = data.data.copy()

    df["SEX"] = pd.Categorical(df["SEX"], categories=["male", "female"], ordered=True)
    df["RACE"] = pd.Categorical(df["RACE"], categories=["white", "black", "indian_aleut_eskimo", "asian", "others"], ordered=True)
    df["EDUC"] = pd.Categorical(df["EDUC"], categories=["below_hs", "hs_grad", "college_grad", "master_higher"], ordered=True)
    df["EMPSTAT"] = pd.Categorical(df["EMPSTAT"], categories=["employed", "unemployed", "retired", "not_in_labor_force"], ordered=True)
    df["NATIVITY"] = pd.Categorical(df["NATIVITY"], categories=["native_born", "foreign_born"], ordered=True)
    df["REGION"] = pd.Categorical(df["REGION"], categories=["northeast", "midwest", "south", "west"], ordered=True)
    df["METRO"] = pd.Categorical(df["METRO"], categories=["not_metropolitan", "metropolitan"], ordered=True)
    df["MARST"] = pd.Categorical(df["MARST"], categories=["has_spouse", "no_spouse"], ordered=True)
    df["DIFFMOB"] = pd.Categorical(df["DIFFMOB"], categories=["no_mobility_limitation", "mobility_limitation"], ordered=True)

    ## Adding extract feature "INCOME_PER_PERSON"
    df["INCOME_PER_PERSON"] = df["FAMINC"].astype(float) / df["FAMSIZE"].astype(float)

    using_cols = ["VOTED", "SEX", "RACE", "EDUC", "EMPSTAT", "NATIVITY", "REGION",
                  "METRO", "MARST", "DIFFMOB", "AGE", "NCHILD", "INCOME_PER_PERSON"]
    
    df = df[using_cols].copy()

    return df


if __name__ == "__main__":
    result = cleaned_data()
    result.to_csv(BASE_DIR / "data" / "cleaned_data.csv", index = False)