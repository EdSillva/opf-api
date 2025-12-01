"""
Treina um modelo scikit-learn a partir do CSV de features e salva como joblib.
Uso:
  OUT_MODEL=app/model/sk_model.joblib python scripts/train_sklearn.py
"""
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

INPUT_CSV = os.environ.get('INPUT_CSV', 'data/features.csv')
OUT_MODEL = os.environ.get('OUT_MODEL', 'app/model/sk_model.joblib')
TARGET = os.environ.get('TARGET', 'Adesao_ao_OPF')

def preprocess(df: pd.DataFrame):
    df = df.copy()
    # Simple preprocessing: drop rows with nulls in target
    df = df.dropna(subset=[TARGET])

    # Convert categorical columns to dummies
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != TARGET]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Fill numeric NaNs with median
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    return df

def main():
    df = pd.read_csv(INPUT_CSV)
    if TARGET not in df.columns:
        raise RuntimeError(f'Target {TARGET} not in {INPUT_CSV}')

    df2 = preprocess(df)
    X = df2.drop(columns=[TARGET])
    y = df2[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump({'model': model, 'feature_columns': X.columns.tolist()}, OUT_MODEL)
    print(f'Model salvo em {OUT_MODEL}')

if __name__ == '__main__':
    main()
