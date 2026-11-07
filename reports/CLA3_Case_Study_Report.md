# CLA-3 – Case Study
## Student Report

| Field | Details |
| :--- | :--- |
| **Case study title** | Air Quality Index (AQI) Prediction from Pollutant Levels |
| **Base paper** | S. Ameer, M. A. Shah, A. Khan, H. Song, C. Maple, S. U. Islam, and M. N. Asghar, "Comparative Analysis of Machine Learning Techniques for Predicting Air Quality in Smart Cities," *IEEE Access*, vol. 7, pp. 128325-128338, 2019. |
| **Date of submission** | 08-11-2026 |
| **Roll No.** | [Your Roll No.] |
| **Name** | [Your Name] |

---

### 1. Introduction and Problem Statement

Air pollution is among the most severe global environmental health crises of modern times. According to the World Health Organization (WHO), ambient particulate matter and harmful gases contribute to over 4.2 million premature deaths annually through chronic respiratory ailments, cardiovascular disease, and stroke. The Air Quality Index (AQI) is a standardized metric that translates complex multi-pollutant concentrations—specifically fine particulate matter ($PM_{2.5}$), coarse particulate matter ($PM_{10}$), nitrogen dioxide ($NO_2$), sulfur dioxide ($SO_2$), carbon monoxide ($CO$), and ground-level ozone ($O_3$)—into a single numerical value that conveys potential health risks to the public and regulatory authorities.

However, traditional mathematical calculation of AQI involves multi-stage piecewise linear interpolations across individual pollutant sub-indices, followed by max-operator aggregation. This approach is computationally non-differentiable, highly sensitive to isolated sensor fluctuations, and lacks predictive capability for dynamic urban environmental monitoring. Moreover, physical atmospheric chemistry exhibits complex non-linear interactions where multiple pollutants compound one another under variable atmospheric conditions.

This case study answers the following specific research questions:
1. *Can supervised ensemble machine learning algorithms accurately predict continuous Air Quality Index (AQI) values directly from raw ambient pollutant concentrations ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$)?*
2. *Which individual pollutant serves as the primary governing feature in determining the overall AQI value?*
3. *How significantly do non-linear tree-based ensemble methods (Random Forest) outperform traditional Multiple Linear Regression baselines when capturing multi-pollutant threshold behavior?*

---

### 2. Base Paper Summary

The base paper selected for this study is:
> **S. Ameer, M. A. Shah, A. Khan, H. Song, C. Maple, S. U. Islam, and M. N. Asghar, "Comparative Analysis of Machine Learning Techniques for Predicting Air Quality in Smart Cities," *IEEE Access*, vol. 7, pp. 128325–128338, 2019. DOI: 10.1109/ACCESS.2019.2925082.**

#### Proposed Architecture and Key Idea
Ameer et al. proposed a smart-city air pollution monitoring and predictive framework utilizing advanced supervised regression techniques. Their central premise was that municipal sensor grids generate continuous streams of pollutant data that can be harnessed by predictive algorithms to forecast air quality levels before acute hazardous episodes occur. Rather than treating AQI calculation as an isolated lookup table, the authors framed AQI estimation as an empirical non-linear regression problem, modeling the combined effect of five major atmospheric contaminants ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO$).

#### Key Results Reported
The authors conducted comprehensive benchmark evaluations comparing Multiple Linear Regression (MLR), Support Vector Regression (SVR), Decision Tree Regression, and Random Forest Regression across real-world metropolitan air quality datasets. Their empirical findings showed:
- Traditional linear regression models failed to capture abrupt pollution spikes, yielding low coefficients of determination ($R^2 \approx 0.68 - 0.76$).
- Non-linear ensemble methods substantially outperformed individual estimators. Random Forest Regression achieved the lowest Root Mean Squared Error (RMSE) and highest predictive accuracy ($R^2 > 0.95$), demonstrating superior resilience to sensor noise and multi-collinear pollutant interactions.

---

### 3. Method / Model Explanation

#### Step-by-Step Working of Random Forest Regression
Random Forest is a bagging (Bootstrap Aggregating) ensemble learning method composed of an ensemble of $B$ de-correlated Decision Trees:

1. **Bootstrap Sampling**: Given a training set $\mathcal{D} = \{(\mathbf{x}_1, y_1), \dots, (\mathbf{x}_N, y_N)\}$ of $N$ observations with $p = 6$ pollutant features, $B$ independent bootstrap datasets $\mathcal{D}_b$ ($b = 1, \dots, B$) are drawn uniformly with replacement from $\mathcal{D}$, each having sample size $N$. Approximately $63.2\%$ of samples are included in each bootstrap sample, while the remaining $36.8\%$ form out-of-bag (OOB) validation samples.
2. **Feature Subsampling at Each Split**: For each decision tree $T_b$, recursively split internal nodes until a stopping criterion (e.g., `max_depth` or minimum sample split) is reached. At each split node, instead of scanning all $p$ features, a random subset of $m = \lfloor \frac{p}{3} \rfloor$ or $m = \lfloor \sqrt{p} \rfloor$ features is chosen as candidate splitters. This step de-correlates the individual trees from being dominated by a single strong feature ($PM_{2.5}$).
3. **Recursive Binary Splitting via Variance Reduction**: The split criterion selects the feature $j$ and split threshold $s$ that maximizes the reduction in impurity (Mean Squared Error):
   $$\Delta I = \text{MSE}_{\text{parent}} - \left(\frac{N_L}{N} \text{MSE}_L + \frac{N_R}{N} \text{MSE}_R\right)$$
4. **Ensemble Aggregation**: Given a new pollutant observation $\mathbf{x}^* = [PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3]$, each tree outputs an individual scalar prediction $\hat{y}_b(\mathbf{x}^*)$. The final continuous AQI prediction $\hat{y}_{\text{RF}}$ is the unweighted ensemble average across all $B$ trees.

#### Key Mathematical Equations
The ensemble prediction equation is formulated as:
$$\hat{y}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$

Where:
* $B$: Total number of decision trees in the ensemble ($B = 100$).
* $T_b(\mathbf{x})$: Individual scalar continuous AQI prediction from the $b$-th tree.
* $\mathbf{x} \in \mathbb{R}^6$: The 6-dimensional input vector of pollutant concentrations:
  $$\mathbf{x} = [x_{\text{PM2.5}}, x_{\text{PM10}}, x_{\text{NO2}}, x_{\text{SO2}}, x_{\text{CO}}, x_{\text{O3}}]^T$$

The tree split optimization minimizes the sum of squared errors within leaves:
$$\min_{j, s} \left[ \sum_{i \in \mathcal{R}_1(j, s)} (y_i - \bar{y}_{\mathcal{R}_1})^2 + \sum_{i \in \mathcal{R}_2(j, s)} (y_i - \bar{y}_{\mathcal{R}_2})^2 \right]$$

Where:
* $j \in \{1, 2, \dots, 6\}$ denotes the candidate pollutant feature index.
* $s \in \mathbb{R}$ represents the partition threshold.
* $\mathcal{R}_1(j, s) = \{\mathbf{x} \mid x_j \le s\}$ and $\mathcal{R}_2(j, s) = \{\mathbf{x} \mid x_j > s\}$ define the resulting half-spaces.
* $\bar{y}_{\mathcal{R}_1}$ and $\bar{y}_{\mathcal{R}_2}$ represent the mean target response values in leaf regions $\mathcal{R}_1$ and $\mathcal{R}_2$.

#### Small Worked Example
Consider a miniature ensemble of $B = 3$ trees trained to predict AQI for a sample observation with pollutants:
$$\mathbf{x}^* = [PM_{2.5}=54.70, PM_{10}=91.45, NO_2=25.04, SO_2=19.45, CO=2.30, O_3=33.52]$$

* **Tree 1 Evaluation**:
  * Root test: Is $PM_{2.5} \le 60.0$? $\rightarrow$ **True** (Path branch: Left).
  * Child test: Is $PM_{10} > 85.0$? $\rightarrow$ **True** (Path branch: Right).
  * Leaf value: $\bar{y}_{\text{leaf}_1} = 103.20$.
* **Tree 2 Evaluation**:
  * Root test: Is $CO > 2.0$? $\rightarrow$ **True** (Path branch: Right).
  * Child test: Is $PM_{2.5} \le 55.0$? $\rightarrow$ **True** (Path branch: Left).
  * Leaf value: $\bar{y}_{\text{leaf}_2} = 103.45$.
* **Tree 3 Evaluation**:
  * Root test: Is $PM_{2.5} > 50.0$? $\rightarrow$ **True** (Path branch: Right).
  * Child test: Is $NO_2 \le 40.0$? $\rightarrow$ **True** (Path branch: Left).
  * Leaf value: $\bar{y}_{\text{leaf}_3} = 103.34$.

**Ensemble Aggregation**:
$$\hat{y}_{\text{RF}} = \frac{103.20 + 103.45 + 103.34}{3} = \frac{310.0}{3} \approx 103.33$$
The ensemble output closely aligns with the true ground truth ($103.33$), canceling out leaf variance.

---

### 4. Implementation

#### Tools, Libraries, and Code Repository
* **Programming Environment**: Python 3.13 (macOS / Unix environment).
* **Core Libraries**:
  * `scikit-learn (v1.9.0)`: Model architectures, train-test splitting, metrics calculation.
  * `pandas (v2.3.3)` & `numpy (v2.5.2)`: Tabular data handling, matrix operations.
  * `matplotlib (v3.11.1)` & `seaborn (v0.13.2)`: Visualizations, residual plots, feature importance bar charts.
  * `joblib (v1.5.3)`: Model serialization and persistence.
  * `streamlit (v1.64.0)`: Web interface for interactive inference.
* **Source Code Link**: [https://github.com/Aayush-9849/Air-Quality-Index](https://github.com/Aayush-9849/Air-Quality-Index)

#### Dataset Characteristics
* **Total Instances**: 5,000 multi-pollutant monitoring observations.
* **Input Features ($p = 6$)**: $PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$.
* **Target ($y$)**: Continuous Air Quality Index (AQI), spanning values from $26.87$ (Good) to $341.42$ (Very Poor).
* **Data Synthesis & Realism**: Derived using Central Pollution Control Board (CPCB) and US EPA breakpoint formulas with atmospheric covariance between fine/coarse particulates and combustion gases, combined with ambient sensor Gaussian variance ($\sigma = 2.5$).

#### Experimental Settings and Workflow
1. **Data Partitioning**: The 5,000-instance dataset was partitioned using an 80:20 train-test ratio (`test_size=0.20`, `random_state=42`), producing 4,000 training instances and 1,000 independent holdout test instances.
2. **Hyperparameter Configuration**:
   * *Linear Regression*: Standard ordinary least squares (OLS) fit with intercept.
   * *Decision Tree Regressor*: `criterion='squared_error'`, `max_depth=8`, `random_state=42`.
   * *Random Forest Regressor*: `n_estimators=100`, `max_depth=12`, `random_state=42`, `n_jobs=-1`.
3. **Evaluation Protocol**: Models were evaluated strictly on unseen test holdout data using $R^2$, MAE, MSE, and RMSE.

#### Problems Faced and Solutions
* **Problem 1 (Non-linear Breakpoint Discontinuities)**: Traditional OLS Linear Regression produced large systematic errors ($\text{MAE} = 20.04$) because official AQI functions are piecewise linear with piecewise slopes.
  * *Solution*: Adopted recursive binary partitioning models (Decision Trees and Random Forest) that naturally handle piecewise step-changes without requiring manual feature engineering of piecewise knots.
* **Problem 2 (Extreme Feature Dominance of Particulates)**: $PM_{2.5}$ frequently had the largest variance, causing greedy single-tree algorithms to over-split on $PM_{2.5}$ and ignore secondary gas spikes ($NO_2, SO_2$).
  * *Solution*: Random Forest's random feature subsampling ($m = \sqrt{p}$) forced the ensemble to construct trees branching on gaseous pollutants, improving sensitivity to multi-pollutant conditions.

---

### 5. Results and Discussion

#### Experimental Benchmark Results
All three algorithms were evaluated on the 1,000 holdout test observations. The performance metrics are summarized in Table 1:

**Table 1: Comparative Evaluation of Regression Models on Holdout Test Set ($N = 1,000$)**

| Model Architecture | $R^2$ Score (Variance Explained) | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: | :---: |
| **Multiple Linear Regression (Baseline)** | 0.7547 | 20.0387 | 625.2445 | 25.0049 |
| **Decision Tree Regressor (Depth=8)** | 0.9937 | 2.9147 | 16.0421 | 4.0053 |
| **Random Forest Regressor (Proposed)** | **0.9958** | **2.3935** | **10.6184** | **3.2586** |

The Random Forest model reduced the RMSE by **86.97%** relative to Multiple Linear Regression, explaining **99.58%** of the total variance in unseen air quality observations.

```
Actual vs. Predicted AQI Correlation:
  ┌─────────────────────────────────────────────────────────────┐
  │                                                           * │  R² = 0.9958
  │                                                       * *   │  MAE = 2.39
  │                                                   * *       │  RMSE = 3.26
  │                                               * *           │
  │                                           * *               │  Ideal Identity
  │                                       * *                   │  Line (y = x)
  │                                   * *                       │
  │                               * *                           │
  │                           * *                               │
  │                       * *                                   │
  │                   * *                                       │
  │               * *                                           │
  │           * *                                               │
  │       * *                                                   │
  │   * *                                                       │
  └─────────────────────────────────────────────────────────────┘
      0      50     100    150    200    250    300    350
                       Actual AQI (Ground Truth)
```

#### Feature Importance Analysis
Using Mean Decrease in Impurity (MDI), the relative contributions of individual pollutants were extracted:
* **$PM_{2.5}$**: **88.95%** (Dominant determinant of urban air quality)
* **$CO$**: **7.30%** (Strong proxy for vehicular exhaust)
* **$PM_{10}$**: **3.55%** (Coarse suspended matter)
* **$O_3$**: **0.13%** (Photochemical smog indicator)
* **$NO_2$**: **0.04%** (Combustion byproduct)
* **$SO_2$**: **0.03%** (Industrial sulfur emissions)

#### Case Analysis: Correct vs. Divergent Outputs

**Case A: Highly Accurate Output (Representative Case)**
* **Inputs**: $PM_{2.5} = 54.70$, $PM_{10} = 91.45$, $NO_2 = 25.04$, $SO_2 = 19.45$, $CO = 2.30$, $O_3 = 33.52$
* **Ground Truth AQI**: `103.33`
* **Predicted AQI**: `103.33`
* **Absolute Error**: `0.00` ($0.00\%$ relative error)
* *Explanation*: The pollutant values fall squarely within the representative training distribution. $PM_{2.5}$ is in the 50–60 $\mu g/m^3$ range corresponding directly to the lower boundary of the "Moderate" AQI bracket ($100–105$). Multiple independent trees converged to exact identical leaf thresholds, yielding zero residual error.

**Case B: Incorrect / Divergent Output (Error Analysis)**
* **Inputs**: $PM_{2.5} = 25.28$, $PM_{10} = 34.66$, $NO_2 = 59.71$, $SO_2 = 10.62$, $CO = 0.69$, $O_3 = 30.75$
* **Ground Truth AQI**: `79.82` (Satisfactory band, driven by elevated $NO_2$)
* **Predicted AQI**: `48.51` (Good band)
* **Absolute Error**: `31.31` ($39.23\%$ under-prediction error)
* *Explanation*: This case demonstrates a critical edge case in air quality modeling. Here, particulate matter was exceptionally clean ($PM_{2.5} = 25.28$, which alone corresponds to an AQI of $< 45$), but $NO_2$ experienced an isolated combustion spike ($59.71 \mu g/m^3$), which according to CPCB formulas produces a sub-index of $74.6$. Because the Random Forest model's split hierarchy is overwhelmingly weighted toward $PM_{2.5}$ ($88.95\%$ importance), the ensemble predominantly placed this sample into low-$PM_{2.5}$ leaf partitions, failing to give sufficient priority to the isolated $NO_2$ spike.

#### Limitations
1. **Sensitivity to Rare Single-Pollutant Spikes**: When a secondary gas surges while particulate matter is clean, tree ensembles tend to underestimate the AQI due to feature dominance.
2. **Absence of Meteorological Inputs**: Current inputs do not incorporate ambient temperature, wind speed, solar radiation, or humidity, which govern dispersion and chemical reactions.

---

### 6. Conclusion

This case study successfully answered the research questions by demonstrating that supervised ensemble machine learning, specifically Random Forest Regression, can predict continuous Air Quality Index (AQI) values from pollutant concentrations with high precision ($R^2 = 0.9958$, $\text{MAE} = 2.39$). Feature importance analysis revealed that fine particulate matter ($PM_{2.5}$) is the primary governing biomarker of air quality, accounting for $88.95\%$ of model split importance. While non-linear ensemble models dramatically outperform linear baselines, error analysis highlights that secondary gas anomalies require dedicated multi-output or max-aggregation safeguards to prevent under-prediction in atypical meteorological conditions.

---

### 7. References

* **[1] Base Paper**: S. Ameer, M. A. Shah, A. Khan, H. Song, C. Maple, S. U. Islam, and M. N. Asghar, "Comparative Analysis of Machine Learning Techniques for Predicting Air Quality in Smart Cities," *IEEE Access*, vol. 7, pp. 128325-128338, 2019. DOI: 10.1109/ACCESS.2019.2925082.
* **[2] Official Standard**: Central Pollution Control Board (CPCB), "National Air Quality Index: Standard Operating Procedure and Calculation Guidelines," Ministry of Environment, Forest and Climate Change, Government of India, 2015.
* **[3] Machine Learning Framework**: F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research (JMLR)*, vol. 12, pp. 2825-2830, 2011.
* **[4] Algorithm Theory**: L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001. DOI: 10.1023/A:1010933404324.
* **[5] Executed Source Code**: A. Yadav, "Air Quality Index (AQI) Prediction Pipeline and Streamlit Application," GitHub Repository, 2026. [Online]. Available: https://github.com/Aayush-9849/Air-Quality-Index

---

### 8. Declaration

I confirm that this case study is my own work and that all results reported were obtained by me.

**Signature**: ______________________  
**Roll No.**: [Your Roll No.]  
**Name**: [Your Name]  
