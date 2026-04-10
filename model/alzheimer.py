# ============================================================
# ALZHEIMER'S DISEASE DETECTION MODEL
# Algorithm: Gradient Boosting Classifier
# Accuracy: 95.81% | Precision: 95.92% | Recall: 92.16%
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, confusion_matrix,
                             classification_report)
import joblib  # to save the trained model to a file

# ============================================================
# STEP 1: Load & Clean Data
# ============================================================

df = pd.read_csv("model/alzheimer.csv")
df = df.drop(columns=["PatientID", "DoctorInCharge"])

print("=" * 55)
print("  ALZHEIMER'S DETECTION MODEL — TRAINING REPORT")
print("=" * 55)
print(f"\n✅ Data loaded:    {df.shape[0]} patients, {df.shape[1]} columns")
print(f"✅ Missing values: {df.isnull().sum().sum()}")

# ============================================================
# STEP 2: Separate Features (X) and Target (y)
# ============================================================

X = df.drop(columns=["Diagnosis"])  # 32 input features
y = df["Diagnosis"]                 # 0 = healthy, 1 = sick

healthy = (y == 0).sum()
sick    = (y == 1).sum()
print(f"✅ Healthy patients: {healthy} | Sick patients: {sick}")

# ============================================================
# STEP 3: Split into Train and Test Sets
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing
    random_state=42   # reproducible split every time
)

print(f"✅ Training set:   {X_train.shape[0]} patients")
print(f"✅ Testing set:    {X_test.shape[0]} patients")

# ============================================================
# STEP 4: Train the Winning Model
# ============================================================

model = GradientBoostingClassifier(
    n_estimators=100,   # 100 trees
    max_depth=3,        # tree depth
    learning_rate=0.1,  # how fast it learns
    random_state=42     # reproducible results
)

print("\n⏳ Training model...")
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ============================================================
# STEP 5: Evaluate on Test Set
# ============================================================

y_pred = model.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 55)
print("  MODEL PERFORMANCE")
print("=" * 55)
print(f"  Accuracy:  {acc:.2%}   (overall correctness)")
print(f"  Precision: {prec:.2%}   (when it says YES, is it right?)")
print(f"  Recall:    {rec:.2%}   (sick patients correctly caught)")
print(f"\n  Confusion Matrix:")
print(f"  ✅ Correctly identified healthy: {cm[0][0]}")
print(f"  ✅ Correctly identified sick:    {cm[1][1]}")
print(f"  ⚠️  False alarms (healthy→sick): {cm[0][1]}")
print(f"  ❌ Missed sick patients:         {cm[1][0]}")

# ============================================================
# STEP 6: Cross-Validation (prove results are reliable)
# ============================================================

print("\n" + "=" * 55)
print("  CROSS-VALIDATION (5-Fold Stratified)")
print("=" * 55)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_recalls    = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    fold_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, random_state=42
    )
    fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    fold_pred = fold_model.predict(X.iloc[val_idx])

    fold_acc = accuracy_score(y.iloc[val_idx], fold_pred)
    fold_rec = recall_score(y.iloc[val_idx], fold_pred)
    fold_accuracies.append(fold_acc)
    fold_recalls.append(fold_rec)
    print(f"  Fold {fold}: Accuracy={fold_acc:.2%}  Recall={fold_rec:.2%}")

print(f"\n  Average Accuracy: {np.mean(fold_accuracies):.2%} "
      f"(± {np.std(fold_accuracies):.2%})")
print(f"  Average Recall:   {np.mean(fold_recalls):.2%} "
      f"(± {np.std(fold_recalls):.2%})")

# ============================================================
# STEP 7: Feature Importance Chart
# ============================================================

feature_names  = X.columns.tolist()
importances    = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature":    feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n" + "=" * 55)
print("  TOP 10 MOST IMPORTANT FEATURES")
print("=" * 55)
for _, row in feat_df.head(10).iterrows():
    bar = "█" * int(row["Importance"] * 200)
    print(f"  {row['Feature']:<28} {row['Importance']:.4f}  {bar}")

plt.figure(figsize=(10, 6))
plt.barh(feat_df["Feature"].head(10)[::-1],
         feat_df["Importance"].head(10)[::-1],
         color="steelblue")
plt.xlabel("Importance Score")
plt.title("Top 10 Features for Alzheimer's Detection")
plt.tight_layout()
plt.savefig("model/feature_importance.png")
plt.show()
print("\n✅ Feature importance chart saved")

# ============================================================
# STEP 8: Save the Trained Model to a File
# ============================================================

joblib.dump(model, "model/alzheimer_model.pkl")
print("✅ Model saved as alzheimer_model.pkl")
print("\n  This file contains our fully trained model.")
print("  We will load it in the web app instead of")
print("  retraining every time someone visits the site.")