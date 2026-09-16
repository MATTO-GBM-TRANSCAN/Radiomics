# Event Analysis Workflow Using Radiomic and Clinical Features

This workflow outlines a pipeline for performing **event-based (binary classification)** analysis by integrating **radiomic features (RF)** from medical imaging with **clinical variables**. It includes steps for data preprocessing, feature selection, univariate and multivariable modeling, and evaluation on external data.

---

## **Redundancy Analysis**

* **Method:** *Pearson Correlation* between RF.
* **Goal:** Identify and eliminate redundant features by evaluating pairwise linear correlations.

---

## **Clinical Impact Analysis**

* **Discriminative Analysis:**
  * *Mann–Whitney U Test* is used to compare the distributions of each RF across binary clinical outcome groups.
  * RFs with p-value < 0.05 are retained for further processing.

---

## **Feature Preprocessing**

* Radiomic and numeric clinical variables undergo normalization and filtering:
  * **Scaling to \[0,1]**
  * **Low Variance Removal:** Features with variance < 0.01 are excluded.
  * **Z-score Normalization** is applied post-scaling.

---

## **Selection of Representative Radiomic Features**

* **Clustering:** Features are clustered using `1 - Pearson Correlation` as a distance metric.
* **Representative Selection:** One RF is selected from each cluster based on lowest p-value from **Mann–Whitney U Test** (p < 0.05), reflecting stronger association with clinical outcomes.

---

## **Univariate Modelling**

* **Model Evaluation:**
  * Each representative RF is evaluated using a **Logistic Regression model** with **5-fold cross-validation**.
  * The model with the highest average **Accuracy** is selected.
* **Performance Assessment:**
  * For the best Logistic Regression model, compute the **ROC-AUC score** using **5-fold cross-validation**.

---

## **Multivariable Modelling**

* **Radiomic Signature Generation:**
  * All possible combinations of 2 to 7 representative RFs are used to form **radiomic signatures**.
* **Model Evaluation:**
  * For each radiomic signature, a **Logistic Regression model** is trained using **5-fold cross-validation**.
  * The model with the highest average **Accuracy** is selected.
* **Performance Assessment:**
  * For the best-performing Logistic Regression model, compute the **ROC-AUC score** using **5-fold cross-validation**.

---

## **External Validation**
