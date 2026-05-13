# UW–Madison Spring 2026 Data Challenge — Yelp Restaurant Success

> 🏆 **Best Overall · Best Report · Best Presentation**
> Team Mysa — Spring 2026 Data Challenge, UW–Madison

We acted as data consultants for a hypothetical restaurant owner preparing to open a new location, using ~150K businesses and ~7M reviews from the Yelp Open Dataset. The brief had two parts: **(1)** recommend up to 5 attributes the owner should invest in before opening, and **(2)** build a dashboard the owner can use after opening. We also took on the optional **predictive modeling** component, training a stacking ensemble to classify whether a restaurant stays open.

## Results at a glance

| | |
|---|---|
| Final model | Stacking ensemble (RF + LightGBM + XGBoost → Logistic Regression) |
| Accuracy | **81.8%** on a 21,945-row held-out test set |
| ROC-AUC | **0.771** |
| Top 5 recommended attributes | Casual Ambience, Street Parking, Lunch-Friendly, Outdoor Seating, Bicycle Parking |
| Largest practical effects | Casual ambience (+19.5pp), street parking (+17.9pp), lunch-friendly (+17.5pp) |
| Unlabeled restaurants predicted | 24,911 open / 2,411 closed |

## Dashboard

The client-facing deliverable is a Tableau dashboard summarizing distribution, ratings, top categories, and the attribute success rates the owner should care about.

![Dashboard](Dashboard_1.png)

Open `Data_Challenge_Dashboard.twb` in Tableau Desktop (it reads from `output/restaurant_merged.csv`).

## Approach

**Part 1a — Attribute selection.** The naive approach (rank attributes by `is_open` alone) gets hijacked by fast-food chains where DriveThru looks like the winner. We defined a composite success metric instead:

> `success = (stars >= 4.0) AND (review_count >= 20) AND (is_open == 1)`

This filters for genuinely well-established, high-quality restaurants. From the 81 raw attribute keys we extracted, we kept the 21 with ≥5% prevalence, then screened each one with:

- **Percentage-point (PP) difference** between success rates with vs without the attribute
- **Chi-square independence test** (significance at p < 0.05)
- A **composite rank** averaging PP rank and chi-square rank
- One attribute per category group (Ambience, GoodForMeal, BusinessParking) to avoid double-counting

We cross-validated the ranking with **permutation importance** from a Random Forest trained on the same target, which produced the same top group.

**Part 1b — Dashboard.** Five panels: open vs closed counts, avg hours/days, geographic distribution, star distribution, top categories, and the attribute success-rate comparison that ties back to Part 1a.

**Predictive modeling.** Used the merged restaurant dataset with engineered features (top-10 attribute count, permutation-weighted attribute score, sentiment percentages from review aggregation, city density tier). 80/20 stratified split, `class_weight="balanced"` and `scale_pos_weight` for the 78.7% open/21.3% closed imbalance. Tuned LR, RF, LightGBM, and XGBoost via `RandomizedSearchCV` (5-fold, ROC-AUC scoring), then stacked the three tree models under a Logistic Regression meta-learner.

Tuning slightly reduced raw accuracy but meaningfully improved recall on the minority (closed) class, which is the right tradeoff for the consulting question — the client cares more about catching the closure signal than chasing the majority class.

## Repository structure

```
.
├── README.md
├── notebooks/
│   ├── DataCleaning_Q1.ipynb         # Cleaning, attribute extraction, Part 1a analysis
│   └── predictive_modelling.ipynb    # Feature engineering, baselines, tuning, final stack
├── dashboard/
│   ├── Data_Challenge_Dashboard.twb  # Tableau workbook
│   └── Dashboard_1.png               # Static export
├── report/
│   ├── Mysa_DC_Report.pdf            # Final written report
│   └── AI_Tool_Usage.pdf             # AI usage disclosure
├── docs/
│   ├── Yelp_Dataset_Documentation___ToS.pdf
│   └── mermaid-diagram.png           # Yelp schema diagram
└── output/
    └── predictions.csv               # is_open predictions for unlabeled restaurants
```

## Data

This project uses the [Yelp Open Dataset](https://www.yelp.com/dataset). The raw JSON files are **not** included here per the Yelp Dataset Terms of Use — download them yourself and place them in a `data/` folder. The schema we used:

![Yelp schema](mermaid-diagram.png)

We worked with five files: `business`, `review`, `tip`, `checkin`, `user` (the data challenge organizers provided a version with `is_open` removed for a subset of businesses, which became our unlabeled prediction target).

## Reproducing

```bash
# 1. Drop Yelp JSON files into data/ and convert business/review/tip to parquet
# 2. Run cleaning + Part 1a analysis
jupyter notebook notebooks/DataCleaning_Q1.ipynb
# Produces output/restaurant_merged.csv

# 3. Run modeling
jupyter notebook notebooks/predictive_modelling.ipynb
# Produces output/predictions.csv
```

Main dependencies: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `imbalanced-learn`, `scipy`, `statsmodels`, `matplotlib`, `missingno`.

## AI tool usage

Claude was used as a supporting tool for code structure on the attribute parsing function, the average-daily-hours helper, the chi-square contingency setup, and chart formatting. All statistical methods, modeling decisions, and interpretations were done by the team. Full disclosure in `report/AI_Tool_Usage.pdf`.

## Team Mysa

Spring 2026 UW–Madison Data Challenge. Won Best Overall, Best Report, and Best Presentation.
