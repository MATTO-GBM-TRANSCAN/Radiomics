# Time-to-Event Analysis Workflow Using Radiomic and Clinical Features

This workflow outlines a pipeline for performing survival (time-to-event) analysis by integrating **radiomic features (RF)** from medical imaging with **clinical variables**. It is structured in sequential steps, including data preprocessing, feature selection, univariate and multivariable modeling, and validation.

---

## **Redundancy Analysis**

* **Method:** *Pearson Correlation* between RF.
* **Goal:** Identify and reduce redundant features by evaluating pairwise linear correlations.

---

## **Clinical Impact Analysis**

* **Correlation Assessment:**
  * *Spearman Correlation* is used to evaluate monotonic relationships between each RF and clinical outcomes.
* **Survival Analysis:**
  * *Log-Rank Test* is performed for each RF, stratified by its median value, to assess differences in survival distributions.

---

## **Feature Preprocessing**

* RF and numeric clinical variables undergo normalization and filtering:

  * **Scaling to \[0,1]**
  * **Low Variance Removal:** Features with variance < 0.01 are excluded.
  * **Z-score Normalization** applied post-scaling.

---

## **Selection of Representative Radiomic Features**

* **Clustering:** Features are clustered using `1 - Pearson Correlation` as a distance metric.
* **Representative Selection:** One RF from each cluster is selected based on strongest clinical impact (highest statistic with p-value < 0.05 from **Spearman** or **Log-Rank Test**).

---

## **Univariate Modelling**

* **Cross-Validation:**
  * Each representative RF is validated via **5-fold cross-validation** of clinical impact analysis.
  * Consistently impactful RFs across folds are retained.
* **Survival Visualization:**
  * **Kaplan-Meier plots** stratified by RF median value are generated for the validated RFs.

---

## **Multivariable Modelling**

* **Radiomic Signature Generation:**
  * All possible combinations of 2 to 8 representative RFs are used to form **radiomic signatures**.
* **Model Evaluation:**
  * For each signature, a **Cox Proportional Hazards (CoxPH) model** is fitted using **5-fold cross-validation**.
  * The model with the highest average **log-likelihood** is selected.
* **Performance Assessment:**
  * The best model is evaluated using the **Concordance Index (C-index)** under 5-fold cross-validation.

---

## **External Validation**
