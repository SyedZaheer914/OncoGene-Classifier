# OncoGene Classifier: Lung Cancer Prediction via Gene Expression

![App Interface](./AppInterface.png)

## 📋 Overview
OncoGene Classifier is a machine learning pipeline designed to distinguish between lung cancer samples and normal tissue using high-dimensional gene expression data. By integrating feature selection with a Random Forest architecture, the tool identifies key biomarkers (genes) that drive the classification, providing both diagnostic predictions and biological insights.

## 📊 Results & Visualization

### Model Performance
The classifier's reliability is validated through several metrics. The ROC curve indicates strong sensitivity and specificity, while the confusion matrix reveals the model's precision in identifying true positives (Cancer) vs. true negatives (Normal).

| Classification Metrics | Prediction Confidence |
|:---:|:---:|
| ![Confusion Matrix](./confusion_matrix.png) | ![ROC Curve](./roc_curve.png) |
| *Figure 1: Confusion Matrix* | *Figure 2: ROC Curve (AUC Analysis)* |

### Biomarker Identification
Using **ANOVA F-test feature selection**, the pipeline narrows down thousands of genes to the most statistically significant contributors. 

![Top Cancer Genes](./top_cancer_genes.png)
*Figure 3: Top 20 Genes ranked by importance in the classification model.*

### Clinical Context
The project also explores the correlation between gene expression profiles and **Smoking Status**, visualizing how environmental factors align with the molecular data.

![Smoking Data Analysis](./SmokingDataAnalysis.png)
*Figure 4: Correlation analysis of smoking status vs. genomic expression.*

---

## 🚀 Key Features
* **Intelligent Preprocessing:** Automatic detection of file orientation (Samples as rows or columns) and standard scaling.
* **Feature Selection:** Reduces noise by selecting the most informative genes, preventing overfitting.
* **Interactive UI:** A user-friendly interface for data upload and real-time visualization generation.
* **Comprehensive Exports:** Generates `cancer_predictions.csv` containing final classification results and probability scores.

## 📂 Project Structure
* `LungCancerPredictionViaGeneExpression.py`: Main application script.
* `cancer_predictions.csv`: Final output predictions per sample.
* `RandomForestPredictions.png`: Overview of model output across the dataset.
* `top_cancer_genes.png`: Horizontal bar chart of prioritized biomarkers.

## 🛠️ Technical Stack & Installation
This project is built using:
* **Python 3.x**
* **Scikit-Learn:** Random Forest Classifier & Feature Selection.
* **Pandas/NumPy:** Data manipulation and matrix normalization.
* **Matplotlib/Seaborn:** Advanced genomic data visualization.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
