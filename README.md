# OncoGene Classifier

A machine learning application for classifying lung cancer samples using gene expression data.  
This project leverages Random Forests and feature selection to identify the most informative genes, enabling accurate cancer vs. normal classification and gene ranking.

---

## 🚀 Features
- Upload **expression matrix** (genes × samples).
- Upload **labels file** (SampleID → Cancer/Normal).
- Optional: Upload **smoking status file** for correlation analysis.
- Automatic detection of file orientation (samples in rows vs. columns).
- Preprocessing with scaling and feature selection (ANOVA F-test).
- Random Forest classification with cross-validation.
- Outputs:
  - Confusion matrix (`confusion_matrix.png`)
  - ROC curve (`roc_curve.png`)
  - Prediction probabilities (`prediction_probabilities.png`)
  - Classification report
  - Ranked list of top cancer-associated genes (`top_cancer_genes.png`)
  - Predictions per sample (`cancer_predictions.csv`)

---

## 📂 Input Files
1. **Expression Matrix**  
   - CSV/TSV file with genes as rows and samples as columns.  
   - Example shape: `(22000, 107)`.

2. **Labels File**  
   - CSV file with two columns: `SampleID, Label`.  
   - Labels must be `Cancer` or `Normal`.

3. **Smoking Status File** *(optional)*  
   - CSV file with `SampleID` and smoking-related columns.

---

## 🖥️ Usage
1. Run the application:
   ```bash
   python LungCancerPredictionViaGeneExpression.py
