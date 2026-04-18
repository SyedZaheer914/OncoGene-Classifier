import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score
import numpy as np
import os

class GeneExpressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lung Cancer Gene Expression Classifier")
        self.root.configure(bg="#1e1e2f")  # Dark purple/blue background

        btn_style = {
            "bg": "#4ecca3",  # Mint green
            "fg": "black",
            "font": ("Helvetica", 10, "bold"),
            "activebackground": "#3fa98c",
            "activeforeground": "white",
            "bd": 0,
            "relief": tk.FLAT,
            "width": 30,
            "height": 2
        }

        self.upload_expr_btn = tk.Button(root, text="Upload Expression Matrix", command=self.load_expression_csv, **btn_style)
        self.upload_expr_btn.pack(pady=10)

        self.upload_label_btn = tk.Button(root, text="Upload Labels CSV", command=self.load_labels_csv, **btn_style)
        self.upload_label_btn.pack(pady=10)

        self.analyze_btn = tk.Button(root, text="Run Cancer Classification", command=self.run_analysis, **btn_style)
        self.analyze_btn.pack(pady=10)

        self.upload_smoking_btn = tk.Button(root, text="Upload Smoking Status", command=self.load_smoking_status_csv, **btn_style)
        self.upload_smoking_btn.pack(pady=10)

        self.output_text = tk.Text(root, bg="#2b2d42", fg="#ffffff", insertbackground='white',font=("Courier New", 10), bd=1, relief=tk.FLAT, wrap="word")
        self.output_text.pack(padx=10, pady=10, expand=True, fill='both')

        self.expr_df = None
        self.labels_df = None
        self.smoking_df = None


    def log(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def load_expression_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text/CSV Files", "*.csv *.tsv *.txt")])
        if file_path:
            try:
                df = None
                try:
                    df = pd.read_csv(file_path, sep='\t', index_col=0, engine='python', quotechar='"')
                except Exception as fallback_err:
                    df = pd.read_csv(file_path, sep=',', index_col=0, engine='python', quotechar='"')

                if df is not None:
                    self.log("" + f"\n[DEBUG] Raw shape: {df.shape}")
                    if all(str(idx).startswith("GSM") for idx in df.index[:5]):
                        self.log("\n[INFO] Detected GSM IDs in rows. Transposing to match expected format...")
                        df = df.transpose()
                        self.log(f"\n[DEBUG] Shape after transpose: {df.shape}")

                if df.empty:
                    self.log("\n[ERROR] Loaded file is empty. Please check the format.")
                    return

                numeric_df = df.apply(pd.to_numeric, errors='coerce')
                numeric_df.dropna(axis=0, how='all', inplace=True)
                numeric_df.dropna(axis=1, how='all', inplace=True)

                if numeric_df.empty:
                    self.log("\n[ERROR] Expression matrix contains no numeric data.")
                    return

                self.expr_df = numeric_df
                self.log("\n[INFO] Expression matrix loaded successfully.")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load expression matrix: {e}")

    def load_labels_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.labels_df = pd.read_csv(file_path)
                self.log("\n[INFO] Labels file loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load labels file: {e}")

    def load_smoking_status_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                smoking_df = pd.read_csv(file_path)
                self.smoking_df = smoking_df
                self.log("\n[INFO] Smoking status file loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load smoking status file: {e}")

    def find_smoking_column(self, df):
        for col in df.columns:
            if "smoking" in col.lower():
                return col
        return None

    def run_analysis(self):
        if self.expr_df is None:
            messagebox.showwarning("Missing Data", "Please upload the expression matrix file.")
            return

        if self.labels_df is None:
            messagebox.showwarning("Missing Data", "Please upload the Labels file.")
            return

        try:
            self.expr_df.columns = self.expr_df.columns.str.replace('.CEL.gz', '', regex=False)

            data = self.expr_df.transpose().reset_index()
            common_ids = set(data['index']) & set(self.labels_df['SampleID'])
            if len(common_ids) == 0:
                self.log("\n[ERROR] No matching SampleIDs between expression matrix and label file. Check your sample IDs.")
                return

            merged = pd.merge(data, self.labels_df, left_on='index', right_on='SampleID')
            X = merged.drop(columns=['index', 'SampleID', 'Label'])
            y = merged['Label'].map({'Cancer': 1, 'Normal': 0})

            if y.nunique() < 2:
                self.log("\n[ERROR] Not enough classes to perform classification.")
                return

            from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            sel = SelectKBest(f_classif, k=100)
            X_selected = sel.fit_transform(X_scaled, y)
            self.log(f"\n[INFO] Selected {X_selected.shape[1]} high-variance features from {X.shape[1]} total genes.")

            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)

            min_class_size = y.value_counts().min()
            n_splits = min(5, min_class_size)
            if n_splits < 2:
                self.log(f"\n[ERROR] Not enough samples to perform cross-validation. At least 2 samples per class needed.")
                return

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X_selected, y, cv=cv)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1] if clf.n_classes_ > 1 else None

            merged_test = pd.DataFrame({
                'SampleID': y_test.reset_index(drop=True).index,
                'True_Label': y_test.values,
                'Predicted': y_pred
            })
            merged_test.to_csv("cancer_predictions.csv", index=False)
            self.log("\n[INFO] Predictions saved to cancer_predictions.csv")

            cm = confusion_matrix(y_test, y_pred)
            self.log("\n[INFO] Confusion Matrix:\n\n" + str(cm))
            # Create labeled DataFrame for the confusion matrix
            cm_df = pd.DataFrame(cm, index=["Actual Normal", "Actual Cancer"],
                          columns=["Predicted Normal", "Predicted Cancer"])

            # Plot it as a heatmap
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False,
                        linewidths=0.5, linecolor='black')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            plt.close()

            self.log("\n[INFO] Saved confusion_matrix.png")

            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig("roc_curve.png")
                plt.close()
                self.log("\n[INFO] Saved roc_curve.png")

                plt.figure(figsize=(8, 4))
                plt.hist(y_proba, bins=20, color='skyblue', edgecolor='black')
                plt.title("Prediction Probabilities")
                plt.xlabel("Probability of Cancer")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig("prediction_probabilities.png")
                plt.close()
                self.log("\n[INFO] Saved prediction_probabilities.png")
            else:
                self.log("\n[WARNING] ROC curve skipped: only one class present.")

            report = classification_report(y_test, y_pred, zero_division=0)
            self.log("\n[INFO] Cancer Classification Report:\n\n" + report)

            selected_gene_names = X.columns[sel.get_support(indices=True)] if hasattr(X, 'columns') else [f"Gene_{i}" for i in range(X_selected.shape[1])]
            feature_importance = pd.Series(clf.feature_importances_, index=selected_gene_names)
            top_genes = feature_importance.sort_values(ascending=False).head(20)
            self.log("\n[INFO] Top 20 Genes by Importance:\n\n" + str(top_genes))

            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_genes.values, y=top_genes.index)
            plt.title("Top 20 Cancer-Associated Genes")
            plt.xlabel("Feature Importance")
            plt.ylabel("Gene")
            plt.tight_layout()
            plt.savefig("top_cancer_genes.png")
            plt.close()
            self.log("[INFO] Saved top_cancer_genes.png")

            self.log("\n✅ COMPLETE: Classification and gene ranking done.")

        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))

        if self.smoking_df is not None:
            smoking_col = self.find_smoking_column(self.smoking_df)
            if smoking_col:
                merged = pd.merge(merged, self.smoking_df, on='SampleID', how='left')
                cancer_cases = merged[merged['Label'] == 'Cancer']
                smoker_counts = cancer_cases[smoking_col].value_counts()
                self.log("\n[INFO] Smoking status distribution among Cancer patients:\n")
                self.log(str(smoker_counts))

                # Add percent breakdown
                total_cancer_cases = cancer_cases.shape[0]
                smoker_percentages = (smoker_counts / total_cancer_cases) * 100
                self.log("\n[INFO] Smoking status percentage among Cancer patients:\n")
                self.log(str(smoker_percentages.round(2)))

            else:
                self.log("\n[WARNING] Could not find a column related to smoking status in the uploaded data.")
        else:
            self.log("\n[WARNING] Smoking status file not uploaded. Skipping smoking analysis.")

if __name__ == "__main__":
    root = tk.Tk()
    app = GeneExpressionApp(root)
    root.mainloop()

