# Airbnb Listing Price Prediction & Pricing Simulation

Predicts Airbnb listing prices from property, host, location and review features, then uses the model to compare estimated revenue under different pricing strategies.

## Data

- Airbnb listings dataset from Kaggle (`Listings.csv`, about 280K listings across 10 cities). Not included here because of size (150 MB).
- After cleaning and trimming price outliers: **276,874 listings, 18 features**.

## What I did

- **Cleaning (pandas, NumPy):** filled missing review scores (about 33% missing) and host response/acceptance rates with medians, gave missing categorical values an `Unknown` category, dropped the mostly empty `district` column, and removed prices above the 99th percentile.
- **Feature engineering:** `host_years` (how long the host has been active) and a `bedrooms_missing` flag so the model keeps the signal that bedroom count was not reported.
- **Modeling (scikit-learn):** one-hot encoding inside a `ColumnTransformer` + `Pipeline`, an 80/20 train/test split, and a Linear Regression baseline compared with a Random Forest (100 trees).
- **Pricing simulation:** took the model's predicted price for a listing as the baseline and compared estimated revenue for price changes from -20% to +20% under an assumed demand response.
- **Reuse:** saved the trained pipeline with `joblib` and scored a new example listing with it.

## Results

| Model | RMSE | R² |
| --- | --- | --- |
| Linear Regression (baseline) | 730.1 | 0.00 |
| Random Forest | **424.9** | **0.66** |

RMSE is in the dataset's price units. The strongest predictors (grouped feature importance) were city (0.39), bedrooms (0.14), accommodates, property type and the host's total listing count. These show what predicts price, not what causes it.

## Limitations

- There is no occupancy or booking data, so the demand response in the pricing simulation is an assumption, not an estimate.
- Listings come from several cities, so one model is learning very different markets at once; per-city models are a natural next step.

## Run it

```bash
pip install pandas numpy scikit-learn joblib jupyter
```

Put `Listings.csv` next to `Airbnb_Listing.ipynb` and run the notebook top to bottom. The trained model file (1.6 GB) is not stored in the repo; the notebook rebuilds it.

**Tools:** Python, pandas, NumPy, scikit-learn, Jupyter
