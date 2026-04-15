# <Voting Demographics>

## Updated Proposal

This project will explore how the demographics of registered voters affects their participation in elections. That is, we want to predict whether an individual voted, and perhaps their mode of voting (in-person or mail).

## Data

Our dataset is from the IPUMS Current Population Survey, which includes individual household-level data from monthly surveys that are conducted by the Census Bureau. We have downloaded the data from November 2024, which includes columns for whether a person voted and how they voted.

The features that we plan on using are age, sex, race, educational attainment, nativity, region, household type, and household income. The new feature that we plan on extracting is a categorical one based on either income or age. Currently, household income is measured in a variety of ranges, so we’d like to simplify this into 3 categories: low income, middle income, and high income. Similarly, we could simplify age into 3 categories.
