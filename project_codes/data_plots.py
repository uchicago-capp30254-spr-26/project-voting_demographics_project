
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from data_cleaning import cleaned_data

BASE_DIR = Path(__file__).resolve().parent.parent
PATH = BASE_DIR / "final_report" / "plots"

def main():
    df = cleaned_data()
    df["not_voted"] = df["VOTED"].map({"voted": 0, "not_voted": 1})

    table1 = df.pivot_table(index="SEX", columns="RACE", values="not_voted", aggfunc="mean")
    plt.figure(figsize=(8,4))
    sns.heatmap(table1, annot=True, fmt=".2f", cmap="Reds")
    plt.title("Voting Rate by Sex and Race")
    plt.savefig(PATH / "data1.png")
    plt.close()

    table2 = df.pivot_table(index="EDUC", columns="EMPSTAT", values="not_voted", aggfunc="mean")
    plt.figure(figsize=(7,4))
    sns.heatmap(table2, annot=True, fmt=".2f", cmap="Reds")
    plt.title("Voting Rate by Education and Employment Status")
    plt.savefig(PATH / "data2.png")
    plt.close()

    table3 = df.pivot_table(index="NATIVITY", columns="REGION", values="not_voted", aggfunc="mean")
    plt.figure(figsize=(8,4))
    sns.heatmap(table3, annot=True, fmt=".2f", cmap="Reds")
    plt.title("Voting Rate by Nativity and Region")
    plt.savefig(PATH / "data3.png")
    plt.close()

    table4 = df.pivot_table(index="DIFFMOB", columns="METRO", values="not_voted", aggfunc="mean")
    plt.figure(figsize=(8,4))
    sns.heatmap(table4, annot=True, fmt=".2f", cmap="Reds")
    plt.title("Voting Rate by Mobility Disability and Residence in Metropolitan Areas")
    plt.savefig(PATH / "data4.png")
    plt.close()

    table5 = df.pivot_table(columns="MARST", values="not_voted", aggfunc="mean")
    plt.figure(figsize=(8.5,2))
    sns.heatmap(table5, annot=True, fmt=".2f", cmap="Reds")
    plt.title("Voting Rate by Spouse Status")
    plt.savefig(PATH / "data5.png")
    plt.close()

    table6 = df.pivot_table(index=pd.cut(df["AGE"],
                            bins=[18,20,25,30,35,40,45,50,55,60,65,70,75,80,85]),
                            values="not_voted", aggfunc="mean")
    plt.figure(figsize=(7,4))
    sns.heatmap(table6, annot=True, fmt=".2f", cmap="Reds")
    plt.title("Voting Rate by Age Group")
    plt.savefig(PATH / "data6.png")
    plt.close()

    table7 = df.pivot_table(index=pd.cut(df["FAMSIZE"], bins=[0,1.01,2.01,3.01,4.01,5.01,6.01,7.01,8.01,15], labels=[1,2,3,4,5,6,7,8,"9+"]),
                            columns=pd.cut(df["NCHILD"], bins=[0,1.01,2.01,3.01,4.01,5.01,6.01,7.01,8.01,9], labels=[1,2,3,4,5,6,7,8,9]),
                            values="not_voted", aggfunc="mean")
    plt.figure(figsize=(7,4))
    sns.heatmap(table7, annot=True, fmt=".2f", cmap="Reds")
    plt.title("Voting Rate by Family Size and Number of Children")
    plt.savefig(PATH / "data7.png")
    plt.close()

    table8 = df.pivot_table(index=pd.cut(df["FAMINC"], bins=30),
                            columns=pd.cut(df["INCOME_PER_PERSON"], bins=40),
                            values="not_voted", aggfunc="mean")
    plt.figure(figsize=(7,3))
    sns.heatmap(table8, annot=False, cmap="Reds", xticklabels=False, yticklabels=False)
    plt.title("Voting Rate by Family Income and Income Per Person")
    plt.savefig(PATH / "data8.png")
    plt.close()

    plt.figure(figsize=(7,2))
    sns.countplot(x='VOTED', data=df)
    plt.title('Voting Distribution')
    plt.savefig(PATH / "data9.png")
    plt.close()

if __name__ == "__main__":
    main()