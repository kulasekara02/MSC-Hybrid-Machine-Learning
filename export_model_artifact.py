import os
import json
import time
import zipfile
from pathlib import Path

def _safe_dump_pickle(obj, path):
    import joblib
    joblib.dump(obj, path)

def _maybe_export_keras(model, folder):
    """If model is a Keras/TF model, export SavedModel format."""
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            saved_dir = folder / "keras_savedmodel"
            model.save(saved_dir, include_optimizer=False)
            return True
    except Exception:
        pass
    return False

def export_model_artifact(
    model,
    preprocessors=None,
    feature_list=None,
    label_name=None,
    threshold=None,
    model_name="hybrid_model",
    version="1.0.0",
    extra_files=None,
    requirements=None,
):
    """Create a versioned ZIP bundle with:
      - model.pkl
      - preprocessors.pkl (if provided)
      - features.json
      - metadata.json
      - requirements.txt (if provided)
      - model_card.md
      - keras_savedmodel/ (if TF model)
    """
    preprocessors = preprocessors or {}
    requirements = requirements or []
    extra_files = extra_files or {}

    ts = time.strftime("%Y%m%d-%H%M%S")
    zip_name = f"{model_name}-{version}-{ts}.zip"
    out_zip = Path.cwd() / zip_name

    bundle_root = Path.cwd() / f".tmp_bundle_{ts}"
    bundle_root.mkdir(parents=True, exist_ok=True)

    # Save core artifacts
    _safe_dump_pickle(model, bundle_root / "model.pkl")
    if preprocessors:
        _safe_dump_pickle(preprocessors, bundle_root / "preprocessors.pkl")
    if feature_list is not None:
        (bundle_root / "features.json").write_text(json.dumps(feature_list, indent=2), encoding="utf-8")

    has_keras = _maybe_export_keras(model, bundle_root)

    # Metadata
    metadata = {
        "model_name": model_name,
        "version": version,
        "exported_at": ts,
        "label_name": label_name,
        "threshold": threshold,
        "has_keras_dir": has_keras,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
    }
    (bundle_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Model Card (build lines with plain ASCII)
    lines = [
        f"# {model_name} (v{version})",
        "",
        f"Exported: {ts}",
        "",
        "## Contents",
        "- model.pkl - main trained model (joblib)",
        "- preprocessors.pkl - scalers/encoders/pipelines (joblib)",
        "- features.json - training/inference feature order",
        "- metadata.json - label/threshold and export info",
    ]
    if has_keras:
        lines.append("- keras_savedmodel/ - TensorFlow SavedModel")
    lines.append("- requirements.txt - pip packages for reproduction")
    lines += [
        "",
        "## Inference snippet (Python)",
        "```python",
        "import joblib, json",
        "from pathlib import Path",
        "",
        "root = Path('UNZIPPED_FOLDER')",
        "model = joblib.load(root/'model.pkl')",
        "pp = joblib.load(root/'preprocessors.pkl')",
        "features = json.loads((root/'features.json').read_text())",
        "",
        "# X should be a pandas DataFrame with columns = features",
        "# X = ...",
        "",
        "X_t = X.copy()",
        "# apply pp steps if not inside model",
        "y_prob = model.predict_proba(X_t)[:, 1]  # if classifier",
        "```",
    ]
    (bundle_root / "model_card.md").write_text("\n".join(lines), encoding="utf-8")

    # Requirements
    if requirements:
        (bundle_root / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")

    # Extra text files
    for fname, content in extra_files.items():
        (bundle_root / fname).write_text(content, encoding="utf-8")

    # Zip
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in bundle_root.rglob("*"):
            zf.write(p, p.relative_to(bundle_root))

    # Cleanup temp
    def _rm_tree(p):
        for c in p.iterdir():
            if c.is_dir():
                _rm_tree(c)
            else:
                try:
                    c.unlink()
                except FileNotFoundError:
                    pass
        p.rmdir()

    _rm_tree(bundle_root)
    return out_zip
