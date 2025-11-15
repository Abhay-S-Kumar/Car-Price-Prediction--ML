# Car-Price-Prediction--ML



This project uses machine learning to predict the price of cars based on various features. The primary dataset is `CarPrice_Assignment.csv`, and the analysis is contained in the `CarPrice_ML.ipynb` notebook.

The goal is to build a regression model that can accurately estimate the `price` of a car given its attributes.

---

## 🚀 Project Workflow

The project follows a standard machine learning workflow:

1.  **Data Loading:** The `CarPrice_Assignment.csv` dataset is loaded into a pandas DataFrame.
2.  **Data Cleaning & Feature Engineering:**
    * **Brand Extraction:** The car's `brand` is extracted from the `CarName` column to create a new, usable feature.
    * **Column Dropping:** Irrelevant columns (`car_ID`, `CarName`) are removed.
    * **Multicollinearity:** Feature correlation is analyzed. `highwaympg` is dropped due to its high correlation (0.97) with `citympg`, preventing multicollinearity.
3.  **Data Preprocessing (Encoding):**
    * **Manual Mapping:** Ordinal features (`doornumber`, `cylindernumber`) are manually mapped from text to their logical numeric values (e.g., 'four' -> 4).
    * **One-Hot Encoding:** All other nominal categorical features (`brand`, `fueltype`, `carbody`, etc.) are converted into numerical format using `pd.get_dummies()`.
4.  **Model Training:**
    * The dataset is split into training (80%) and testing (20%) sets.
    * A **Random Forest Regressor** is selected as the primary model.
5.  **Hyperparameter Tuning:**
    * `RandomisedSearchCV` is used to systematically test a range of hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`) to find the best-performing model.
6.  **Model Evaluation:**
    * The final model is evaluated on the unseen test set using standard regression metrics.

---

## 🏁 Results

The final tuned Random Forest model performed well on the test data:

* **R² Score:** **0.887** (or 88.7%)
    * *This indicates the model can explain 88.7% of the variance in car prices, which is a strong result.*
* **Mean Absolute Error (MAE):** **$1386.68**
    * *On average, the model's prediction is off by approximately $1,387.*
* **Root Mean Squared Error (RMSE):** **$2072.51**
    * *This metric penalizes larger errors more heavily and provides another view of the model's accuracy.*

The best-performing model parameters were found to be:
* `max_depth: 10`
* `min_samples_split: 5`
* `n_estimators: 100`

---

## 🛠️ How to Run

To run this project, clone the repository and install the required libraries.

**Libraries Used:**
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

You can install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
