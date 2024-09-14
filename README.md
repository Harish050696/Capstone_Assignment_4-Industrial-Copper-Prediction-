# Capstone Assignment 4
A Machine Learning Classification model to predict the status of the copper and the Regression model to predict the selling price of the copper.

# Data Extraction:
Necessary libraries are imported (e.g., NumPy, pandas, Matplotlib, Seaborn, and pickle).
Data is loaded from an Excel file (Copper_Set.xlsx), and the structure and content of the dataframe are explored.

# Feature Engineering:
The dataset is filtered based on the "status" column values ("Won" and "Lost").
Various columns are manipulated, such as dropping the "id" column, rounding numeric values, and creating a new "area" column by multiplying "width" and "thickness."

# Data Cleaning:
Rows with null values in specific columns (e.g., customer, application, item_date) are removed.
Values starting with "0000" in the "material reference" column are replaced with NaN.

# Modelling:
Classification models like RandomForestClassifier, ExtraTreesClassifier, and XGBClassifier are trained using grid search with cross-validation.
Regression models like ExtraTreesRegressor, XGBRegressor, and RandomForestRegressor are also hypertuned.
XGB Classifier and ExtraTrees Regressor are selected based on accuracy and R² scores.

# Model Saving:
The trained XGB Classifier and ExtraTrees Regressor are saved using the pickle module for further use in a Streamlit application.

# Streamlit Application:
A Streamlit file is created to load the models and make predictions.
The code for running the Streamlit application is included.
