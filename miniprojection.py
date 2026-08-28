import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def show(value):
    """Display values in both scripts and notebooks."""
    print(value)


# ============================================================
# LOAD DATA
# ============================================================

base_dir = Path(__file__).resolve().parent
csv_files = list(base_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file was found in the project folder.")

file_path = csv_files[0]
df = pd.read_csv(file_path)

original_records = len(df)

print("Dataset loaded successfully!")
print("File:", file_path.name)
print("Shape:", df.shape)

show(df.head())

# ============================================================
# CLEAN COLUMN NAMES AND DATA
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)

required_columns = [
    "Car_Name",
    "Year",
    "Selling_Price",
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

numeric_columns = [
    "Year",
    "Selling_Price",
    "Present_Price",
    "Kms_Driven",
    "Owner",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

duplicate_count = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# Remove invalid records
df = df[
    (df["Selling_Price"] > 0)
    & (df["Present_Price"] > 0)
    & (df["Kms_Driven"] >= 0)
    & (df["Year"].between(1900, 2026))
].copy()

df = df.dropna(subset=["Selling_Price", "Present_Price", "Year"])

print("\nDataset shape after cleaning:", df.shape)
print("Duplicate rows removed:", duplicate_count)

# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
missing_table = pd.DataFrame({
    "Column": df.columns,
    "Missing_Values": df.isna().sum().values,
    "Missing_Percentage": (
        df.isna().sum().values / len(df) * 100
    ).round(2),
})

show(missing_table)

print("\n========== DESCRIPTIVE STATISTICS ==========")
show(df.describe(include="all").T)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

REFERENCE_YEAR = 2018

df["Car_Age"] = REFERENCE_YEAR - df["Year"]

df["Price_Retention_pct"] = (
    df["Selling_Price"] / df["Present_Price"] * 100
)

df["Depreciation_pct"] = 100 - df["Price_Retention_pct"]

df["Depreciation_Value"] = (
    df["Present_Price"] - df["Selling_Price"]
)

# ============================================================
# BUSINESS KPIs
# ============================================================

print("\n========== BUSINESS KPIs ==========")
print("Total listings:", len(df))
print("Unique car models:", df["Car_Name"].nunique())
print("Average selling price:", round(df["Selling_Price"].mean(), 2))
print("Median selling price:", round(df["Selling_Price"].median(), 2))
print("Average present price:", round(df["Present_Price"].mean(), 2))
print("Average kilometers driven:", round(df["Kms_Driven"].mean(), 2))
print("Average price retention:",
      round(df["Price_Retention_pct"].mean(), 2), "%")

# ============================================================
# CATEGORY ANALYSIS
# ============================================================

categorical_columns = [
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
]

for column in categorical_columns:
    print(f"\n========== {column.upper()} ==========")
    print(df[column].value_counts())

    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x=column,
        order=df[column].value_counts().index,
    )
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Number of Cars")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# ============================================================
# DISTRIBUTIONS
# ============================================================

for column, title in [
    ("Selling_Price", "Distribution of Selling Price"),
    ("Present_Price", "Distribution of Present Price"),
    ("Kms_Driven", "Distribution of Kilometers Driven"),
]:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, bins=30, kde=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ============================================================
# CORRELATION
# ============================================================

numeric_features_for_corr = [
    "Year",
    "Selling_Price",
    "Present_Price",
    "Kms_Driven",
    "Owner",
    "Car_Age",
    "Price_Retention_pct",
    "Depreciation_pct",
]

correlation = df[numeric_features_for_corr].corr()

print("\n========== CORRELATION WITH SELLING PRICE ==========")
show(correlation["Selling_Price"].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ============================================================
# GROUP ANALYSIS
# ============================================================

def create_summary(group_column):
    return df.groupby(group_column).agg(
        Listings=("Selling_Price", "count"),
        Average_Selling_Price=("Selling_Price", "mean"),
        Median_Selling_Price=("Selling_Price", "median"),
        Average_Retention=("Price_Retention_pct", "mean"),
        Average_Kms=("Kms_Driven", "mean"),
    ).sort_values("Average_Selling_Price", ascending=False)


fuel_summary = create_summary("Fuel_Type")
seller_summary = create_summary("Seller_Type")
transmission_summary = create_summary("Transmission")

for title, summary in [
    ("FUEL TYPE ANALYSIS", fuel_summary),
    ("SELLER TYPE ANALYSIS", seller_summary),
    ("TRANSMISSION ANALYSIS", transmission_summary),
]:
    print(f"\n========== {title} ==========")
    show(summary.round(2))

# ============================================================
# MODEL ANALYSIS
# ============================================================

model_summary = df.groupby("Car_Name").agg(
    Listings=("Selling_Price", "count"),
    Average_Selling_Price=("Selling_Price", "mean"),
    Median_Selling_Price=("Selling_Price", "median"),
    Average_Present_Price=("Present_Price", "mean"),
    Average_Retention=("Price_Retention_pct", "mean"),
)

model_3plus = model_summary[model_summary["Listings"] >= 3]

print("\n========== TOP MODELS BY SELLING PRICE ==========")
show(
    model_3plus
    .sort_values("Average_Selling_Price", ascending=False)
    .head(15)
    .round(2)
)

print("\n========== TOP MODELS BY PRICE RETENTION ==========")
show(
    model_3plus
    .sort_values("Average_Retention", ascending=False)
    .head(15)
    .round(2)
)

# ============================================================
# OUTLIER ANALYSIS
# ============================================================

q1 = df["Selling_Price"].quantile(0.25)
q3 = df["Selling_Price"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[
    (df["Selling_Price"] < lower_bound)
    | (df["Selling_Price"] > upper_bound)
]

print("\n========== OUTLIER ANALYSIS ==========")
print("Lower bound:", round(lower_bound, 2))
print("Upper bound:", round(upper_bound, 2))
print("Number of outliers:", len(outliers))

# ============================================================
# MACHINE LEARNING
# ============================================================

features = [
    "Present_Price",
    "Kms_Driven",
    "Car_Age",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
]

target = "Selling_Price"

X = df[features]
y = df[target]

numeric_features = [
    "Present_Price",
    "Kms_Driven",
    "Car_Age",
    "Owner",
]

categorical_features = [
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            categorical_features,
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        min_samples_leaf=2,
    ),
}

results = []
predictions = {}

for model_name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)

    predictions[model_name] = predicted

    results.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, predicted),
        "RMSE": np.sqrt(mean_squared_error(y_test, predicted)),
        "R2": r2_score(y_test, predicted),
    })

model_results = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
show(model_results.round(3))

# ============================================================
# CROSS-VALIDATION
# ============================================================

linear_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

cv_scores = cross_val_score(
    linear_pipeline,
    X,
    y,
    cv=5,
    scoring="r2",
)

print("\n========== CROSS VALIDATION ==========")
print("R² scores:", np.round(cv_scores, 3))
print("Mean CV R²:", round(cv_scores.mean(), 3))
print("Std CV R²:", round(cv_scores.std(), 3))

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

rf_predictions = predictions["Random Forest"]

plt.figure(figsize=(9, 6))
plt.scatter(y_test, rf_predictions, alpha=0.7)

minimum = min(y_test.min(), rf_predictions.min())
maximum = max(y_test.max(), rf_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    color="red",
)

plt.title("Actual vs Predicted Selling Price")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.tight_layout()
plt.show()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            min_samples_leaf=2,
        ),
    ),
])

rf_pipeline.fit(X, y)

fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]
fitted_model = rf_pipeline.named_steps["model"]

feature_names = fitted_preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": fitted_model.feature_importances_,
}).sort_values("Importance", ascending=False).head(15)

print("\n========== TOP FEATURE IMPORTANCES ==========")
show(importance_df)

plt.figure(figsize=(10, 7))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()

# ============================================================
# BUSINESS INSIGHTS
# ============================================================

highest_fuel = fuel_summary.index[0]
highest_seller = seller_summary.index[0]
highest_transmission = transmission_summary.index[0]

print("\n========== BUSINESS INSIGHTS ==========")
print(f"1. {highest_fuel} cars have the highest average selling price.")
print(f"2. {highest_seller} listings have the highest average selling price.")
print(
    f"3. {highest_transmission} vehicles have the highest average "
    "selling price."
)

if not model_3plus.empty:
    highest_retention_model = model_3plus["Average_Retention"].idxmax()
    highest_value_model = model_3plus["Average_Selling_Price"].idxmax()

    print(
        f"4. {highest_retention_model} has the highest average "
        "price retention."
    )
    print(
        f"5. {highest_value_model} has the highest average selling price."
    )

# ============================================================
# EXPORT RESULTS
# ============================================================

output_file = base_dir / "car_dekho_cleaned_analysis.csv"
df.to_csv(output_file, index=False)

print("\nCleaned dataset saved as:", output_file)
print("Project completed successfully.")