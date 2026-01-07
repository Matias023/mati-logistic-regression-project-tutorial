import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)


RAW_PATH = os.path.join("data", "raw", "bank-marketing-campaign-data.csv")
PROCESSED_DIR = os.path.join("data", "processed")


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError("No existe el archivo en: " + path)

    # lectura robusta (por si el separador cambia)
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    if "y" not in df.columns:
        raise KeyError("No encuentro la columna target 'y'. Columnas: " + ", ".join(df.columns))

    return df


def quick_eda(df):
    print("== Dataset ==")
    print("Shape:", df.shape)
    print("Columnas:", len(df.columns))
    print("\n== Target y (counts) ==")
    print(df["y"].value_counts(dropna=False))
    print("\n== Target y (ratio) ==")
    print(df["y"].value_counts(normalize=True))

    dup = df.duplicated().sum()
    print("\nDuplicados:", dup)

    missing = df.isna().sum()
    total_missing = int(missing.sum())
    print("Nulos totales:", total_missing)
    if total_missing > 0:
        print("\nTop missing:")
        print(missing.sort_values(ascending=False).head(10))


def build_preprocess(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocess


def evaluate_model(name, y_true, y_pred, y_proba):
    print("\n" + "=" * 70)
    print("MODEL:", name)
    print("ROC AUC:", roc_auc_score(y_true, y_proba))
    print("\nClassification report:\n", classification_report(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


def pick_threshold_for_recall(y_true, y_proba, target_recall=0.70):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)

    idx = np.where(rec >= target_recall)[0]
    if len(idx) == 0:
        return 0.5

    best = idx[np.argmax(prec[idx])]
    # thr tiene largo n-1 vs prec/rec largo n
    best_thr = thr[best - 1] if best > 0 else 0.5
    return float(best_thr)


def save_splits(X_train, X_test, y_train, y_test):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df = X_train.copy()
    train_df["y"] = y_train

    test_df = X_test.copy()
    test_df["y"] = y_test

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n== Guardado ==")
    print("Train:", train_path)
    print("Test :", test_path)


def main():
    # Ajustes principales
    test_size = 0.2
    random_state = 42
    target_recall = 0.70

    df = load_data(RAW_PATH)
    quick_eda(df)

    # target binario
    y = df["y"].map({"yes": 1, "no": 0})
    X = df.drop(columns=["y"])

    if y.isna().any():
        raise ValueError("El target y tiene valores inesperados (no yes/no). Revisa df['y'].unique().")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    preprocess = build_preprocess(X_train)

    # -------------------------
    # Modelo base
    # -------------------------
    model_base = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    model_base.fit(X_train, y_train)
    y_pred_base = model_base.predict(X_test)
    y_proba_base = model_base.predict_proba(X_test)[:, 1]
    evaluate_model("LogReg BASE", y_test, y_pred_base, y_proba_base)

    # -------------------------
    # Modelo balanced
    # -------------------------
    model_bal = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    model_bal.fit(X_train, y_train)
    y_pred_bal = model_bal.predict(X_test)
    y_proba_bal = model_bal.predict_proba(X_test)[:, 1]
    evaluate_model("LogReg BALANCED", y_test, y_pred_bal, y_proba_bal)

    # -------------------------
    # Threshold tuning (sobre balanced)
    # -------------------------
    best_thr = pick_threshold_for_recall(y_test, y_proba_bal, target_recall=target_recall)
    y_pred_thr = (y_proba_bal >= best_thr).astype(int)

    print("\n== Threshold tuning ==")
    print("Target recall:", target_recall)
    print("Threshold elegido:", best_thr)
    evaluate_model("LogReg BALANCED + THRESHOLD", y_test, y_pred_thr, y_proba_bal)

    # Guardar splits
    save_splits(X_train, X_test, y_train, y_test)

    print("\nFIN ✅")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:", str(e))
        sys.exit(1)