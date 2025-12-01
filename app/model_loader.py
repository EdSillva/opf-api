import json
from pyspark.ml import PipelineModel
from app.config import MODEL_PATH, METADATA_PATH
from typing import Any, Dict, List
from pathlib import Path


def load_metadata():
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_model():
    return PipelineModel.load(MODEL_PATH)


metadata = load_metadata()
model = load_model()


def load_model_from(path: str):
    """Carrega um PipelineModel de um caminho arbitrário.

    Exemplo de uso:
        m = load_model_from('/content/modelo_openfinance_rf')
        pred = predict_with_model(m, df_novo)
    """
    return PipelineModel.load(path)


def predict_with_model(model_obj, df):
    """Executa a transformação usando um PipelineModel já carregado.

    `df` deve ser um pyspark DataFrame com as colunas esperadas pelo pipeline.
    Retorna o DataFrame transformado.
    """
    return model_obj.transform(df)


def load_sklearn_model(path: str) -> Any:
    """Load a scikit-learn model saved with joblib.

    Expected object is a dict with keys:
      - 'model': sklearn estimator
      - 'feature_columns': list[str]
    """
    import joblib
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Sklearn model not found at {path}')
    obj = joblib.load(path)
    return obj


def predict_sklearn(model_obj: Any, client: Dict[str, Any]) -> Dict[str, Any]:
    """Predict single client dict using sklearn model object produced by load_sklearn_model.

    `client` is a mapping with keys for raw features (human-readable). We build a feature vector
    matching `model_obj['feature_columns']` by selecting values or using 0 for missing dummies.
    Returns a dict with 'prediction' and optional 'probability'.
    """
    import pandas as pd
    feature_cols: List[str] = model_obj.get('feature_columns') or []
    model_skl = model_obj.get('model')
    if model_skl is None or not feature_cols:
        raise RuntimeError('Invalid sklearn model object')

    row = pd.DataFrame([client])
    row_enc = pd.get_dummies(row)

    for c in feature_cols:
        if c not in row_enc.columns:
            row_enc[c] = 0

    X = row_enc[feature_cols].astype(float)
    proba = None
    if hasattr(model_skl, 'predict_proba'):
        proba = model_skl.predict_proba(X)[0].tolist()
    pred = model_skl.predict(X)[0].item()
    return {'prediction': pred, 'probability': proba}
