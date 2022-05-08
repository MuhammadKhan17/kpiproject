# KPI Project

## Project Overview

- In order to measure the success of an advertisement and take effective data-driven business decisions, we use key performance indicators (KPI).
- Exploratory data analysis was used to find two primary learning objectives.
  - Optimal mix of creatives per funnel + publisher
  - Impact of number of creatives in market on performance
- Random forests and light gradient boosting machine were utilized in predicting average funnel-specific click through rate (CTR), site visit conversion rate (CVRSV), buy flow entry conversion rate (CVRBF) and sales conversion rate (CVRS) for a given creative/advertisement prior to market launch.
- Evaluation metrics on hold-out data
  - CTR: R-squared (0.59), MAE (0.051)
  - CVR(SV): R-squared (0.65), MAE (0.12)
  - CVR(BF): R-squared (0.53), MAE (1.15)
  - CVR(S): R-squared (0.37), MAE (0.30)

## Code and Resources Used

Packages: pandas, numpy, matplotlib, seaborn, statsmodels, sklearn, lightgbm, tkinter, holidays

## Data

The data set contains over 13,000 creatives/ads, each with selected features and KPI results.

### Variable Descriptions

|                  | Description                                                             |
| ---------------- | ----------------------------------------------------------------------- |
| Creative ID      | Unique Identifier for each creative                                     |
| Channel          | Type of Media: Social ( eg. FB, Twitter) or Display (eg Amazon, Google) |
| Publisher        | Where the creative is served                                            |
| Funnel           | UF, MF, LF                                                              |
| LOB              | Line of business                                                        |
| Product          | Item advertised in the creative                                         |
| Theme            | High-level concept for a series of ads - a theme has numerous versions  |
| Creative Version | A theme can have various creative versions                              |
| KPI Audience     | Audience targeted by creative                                           |
| Price            | Advertised value of the item                                            |
| Price Placement  | Position of the price on the creative                                   |
| Discount         | Advertised value of the incentive                                       |
| Offer Placement  | Position of the discount on the creative                                |
| Offer Group      | Advertised extra incentives                                             |
| Length           | Ad length measured in seconds; no length for display                    |
| Asset Type       | Video or display/static creative                                        |
| Video Type       | Characteristics of video creatives                                      |
| Ad Size          | Physical dimensions of creative                                         |
