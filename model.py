import os
from pathlib import Path
import joblib
import pandas as pd
from collections import Counter
from sklearn.ensemble import (GradientBoostingClassifier,
                               RandomForestClassifier, VotingClassifier)
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      train_test_split)
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# ── Path file model ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_pln_ensemble.pkl"
ENCODER_PATH = BASE_DIR / "encoders_pln.pkl"
USE_SAVED_MODEL = os.environ.get("USE_SAVED_MODEL") == "1"


def _build_model():
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15,
        min_samples_leaf=2, class_weight='balanced', random_state=42)
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.05, subsample=0.8, random_state=42)
    return VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)], voting='soft')


def _train_model(X_tr, y_tr, encoders):
    vc = _build_model()
    vc.fit(X_tr, y_tr)
    if USE_SAVED_MODEL:
        try:
            joblib.dump(vc, MODEL_PATH)
            joblib.dump(encoders, ENCODER_PATH)
        except OSError:
            # Streamlit Cloud can run from a read-only checkout; keep using memory.
            pass
    return vc

@st.cache_resource
def train_ml_pipeline(df: pd.DataFrame, data_hash: int):
    """
    Alur:
      1. Fit encoder selalu (agar konsisten dengan data terkini)
      2. Jika model .pkl ada → load dari disk (cepat)
      3. Jika tidak → training → simpan ke disk
      4. Evaluasi & kembalikan semua artefak
    """
    ml = df.copy()

    # ── Encode fitur kategorikal ──────────────────────────────────────────────
    encoders = {}
    for col in ['GI', 'HALAMAN TOWER', 'POLUTAN ISOLATOR']:
        le = LabelEncoder()
        ml[col + '_ENC'] = le.fit_transform(ml[col].astype(str))
        encoders[col] = le

    # ── Buat label target ─────────────────────────────────────────────────────
    ml['LABEL'] = ml['STATUS_RISIKO'].map(
        {'Aman': 0, 'Waspada': 1, 'Kritis': 2}
    ).fillna(0).astype(int)

    # Jika Waspada terlalu sedikit (< 5), merge ke Aman agar model stabil
    class_counts = ml['LABEL'].value_counts().to_dict()
    if class_counts.get(1, 0) < 5:
        ml['LABEL'] = ml['LABEL'].replace(1, 0)

    FEATS = ['UMUR_ASET', 'GI_ENC', 'HALAMAN TOWER_ENC', 'POLUTAN ISOLATOR_ENC']
    X, y  = ml[FEATS], ml['LABEL']

    # Stratify aman jika tiap kelas ≥ 2 sampel
    min_c = min(Counter(y).values())
    strat = y if min_c >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat)

    # ── Load atau Training model ──────────────────────────────────────────────
    if not USE_SAVED_MODEL:
        vc = _train_model(X_tr, y_tr, encoders)
    elif MODEL_PATH.exists() and ENCODER_PATH.exists():
        try:
            vc = joblib.load(MODEL_PATH)
        except (ModuleNotFoundError, ImportError, AttributeError, ValueError, OSError):
            st.warning(
                "Model tersimpan tidak kompatibel dengan environment saat ini. "
                "Aplikasi melatih ulang model otomatis."
            )
            vc = _train_model(X_tr, y_tr, encoders)
    else:
        vc = _train_model(X_tr, y_tr, encoders)

    # ── Evaluasi ──────────────────────────────────────────────────────────────
    y_pred = vc.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)

    present_labels = sorted(set(y_te) | set(y_pred))
    name_map       = {0: 'Aman', 1: 'Waspada', 2: 'Kritis'}
    present_names  = [name_map[l] for l in present_labels]

    report = classification_report(
        y_te, y_pred,
        labels=present_labels,
        target_names=present_names,
        output_dict=True,
        zero_division=0
    )
    for k in ['Aman', 'Waspada', 'Kritis']:   # pastikan semua key ada
        if k not in report:
            report[k] = {'precision': 0, 'recall': 0,
                          'f1-score': 0, 'support': 0}

    cm = confusion_matrix(y_te, y_pred, labels=present_labels)

    # ── Cross-Validation ──────────────────────────────────────────────────────
    n_splits = min(5, min_c) if min_c >= 2 else 2
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cvs = cross_val_score(vc, X, y, cv=cv, scoring='accuracy')

    # ── Feature Importance dari RF ────────────────────────────────────────────
    rf_fitted = vc.estimators_[0]
    fi = dict(zip(
        ['Umur Aset', 'Gardu Induk', 'Halaman Tower', 'Polutan Isolator'],
        rf_fitted.feature_importances_
    ))

    return (vc, encoders, acc, report, cm,
            cvs, fi, X_te, y_te, present_labels, present_names)
