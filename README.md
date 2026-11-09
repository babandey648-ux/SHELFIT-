# ShelfIt — Data Analytics Individual Submission

Run the steps in this order from inside this folder:

| Step | File | What it does |
|---|---|---|
| 1 | `generate_data.py` | Generates the synthetic ShelfIt data from the business-plan assumptions |
| 2 | `make_raw.py` | Adds realistic data-quality issues to create the raw export |
| 3 | `ShelfIt_Data_Cleaning_and_EDA.ipynb` | Audit, cleaning, transformation, descriptive & diagnostic EDA (all 12 charts explained) |
| 4 | `app.py` | Interactive Streamlit dashboard: `streamlit run app.py` |

Setup: `pip install -r requirements.txt`

## data/
- `shelfit_sellers_raw.csv`, `shelfit_orders_raw.csv` — raw input (with issues)
- `shelfit_sellers_clean.csv`, `shelfit_orders_clean.csv` — cleaned output
- `shelfit_sellers_model_ready.csv` — encoded and scaled, for the group-phase models
- `*_generated.csv`, `ShelfIt_Simulated_Data.xlsx` — output of step 1 (with data dictionary)

## figures/
All 12 EDA charts as PNGs (used in the report).

`ShelfIt_Data_Cleaning_and_EDA.html` is the executed notebook, viewable in any browser.
