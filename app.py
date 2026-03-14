from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import os
from collections import Counter

torch.set_num_threads(1)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILES = {
    "random_forest": "random_forest_model.pkl",
    "xgboost": "xgboost_model.pkl",
    "scaler": "scaler.pkl",
    "label_encoder": "label_encoder.pkl",
    "gat_primary": "gat_model.pth",
    "gat_secondary": "gat_ids_model.pth",
}

rf_model = joblib.load(os.path.join(MODEL_DIR, MODEL_FILES["random_forest"]))
xgb_model = joblib.load(os.path.join(MODEL_DIR, MODEL_FILES["xgboost"]))

scaler = joblib.load(os.path.join(MODEL_DIR, MODEL_FILES["scaler"]))
label_encoder = joblib.load(os.path.join(MODEL_DIR, MODEL_FILES["label_encoder"]))

gat_model1_state = torch.load(
    os.path.join(MODEL_DIR, MODEL_FILES["gat_primary"]),
    map_location="cpu"
)
gat_model2_state = torch.load(
    os.path.join(MODEL_DIR, MODEL_FILES["gat_secondary"]),
    map_location="cpu"
)

gat_scaler = scaler
gat_le = label_encoder

class GAT1(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT1, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, heads=8)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)

        return x

class GAT2(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT2, self).__init__()

        self.gat1 = GATConv(in_channels, hidden_channels, heads=8)
        self.gat2 = GATConv(hidden_channels * 8, out_channels, heads=1)

    def forward(self, x, edge_index):

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = self.gat2(x, edge_index)

        return x

input_dim = rf_model.n_features_in_
num_classes = len(gat_le.classes_)
EXPECTED_FEATURES = list(scaler.feature_names_in_)


def validate_model_bundle():
    errors = []
    expected_class_ids = np.arange(num_classes)

    if input_dim != len(EXPECTED_FEATURES):
        errors.append(
            "Random forest feature count does not match scaler feature count."
        )

    if getattr(xgb_model, "n_features_in_", None) != len(EXPECTED_FEATURES):
        errors.append("XGBoost feature count does not match scaler feature count.")

    if not np.array_equal(np.asarray(rf_model.classes_), expected_class_ids):
        errors.append("Random forest class IDs do not match the label encoder.")

    if not np.array_equal(np.asarray(xgb_model.classes_), expected_class_ids):
        errors.append("XGBoost class IDs do not match the label encoder.")

    if gat_model1_state["conv2.bias"].shape[0] != num_classes:
        errors.append("Primary GAT output size does not match the label encoder.")

    if gat_model2_state["gat2.bias"].shape[0] != num_classes:
        errors.append("Secondary GAT output size does not match the label encoder.")

    if errors:
        formatted_errors = "\n- ".join(errors)
        raise RuntimeError(f"Invalid model bundle:\n- {formatted_errors}")


validate_model_bundle()

gat_model1 = GAT1(input_dim, 8, num_classes)

gat_model1.load_state_dict(gat_model1_state)

gat_model1.eval()

gat_model2 = GAT2(input_dim, 64, num_classes)

gat_model2.load_state_dict(gat_model2_state)

gat_model2.eval()

SIMPLE_LABELS = {
    "benign": "Normal Traffic",
    "bot": "Bot Activity",
    "ddos": "Distributed Denial of Service",
    "doshulk": "Denial of Service (Hulk)",
    "dosgoldeneye": "Denial of Service (GoldenEye)",
    "dosslowloris": "Denial of Service (Slowloris)",
    "dosslowhttptest": "Denial of Service (Slow HTTP Test)",
    "heartbleed": "Heartbleed Exploit Attempt",
    "infiltration": "Network Infiltration Attempt",
    "portscan": "Port Scanning Activity",
    "ftppatator": "FTP Brute-Force Attempt",
    "sshpatator": "SSH Brute-Force Attempt",
    "bruteforce": "Brute-Force Attempt",
    "webattackbruteforce": "Web Login Brute-Force Attempt",
    "webattackxss": "Cross-Site Scripting Attempt",
    "webattacksqlinjection": "SQL Injection Attempt"
}


def normalize_label(label):
    return "".join(ch.lower() for ch in str(label) if ch.isalnum())


def simplify_label(label):
    normalized = normalize_label(label)
    if normalized in SIMPLE_LABELS:
        return SIMPLE_LABELS[normalized]

    pretty = str(label).replace("_", " ").replace("-", " ")
    pretty = " ".join(pretty.split())
    return pretty.title() if pretty else "Unknown Traffic"


def derive_risk_and_status(label):
    normalized = normalize_label(label)

    if normalized == "benign":
        return "Low Risk", "SAFE", "green"

    medium_risk = {
        "portscan",
        "bruteforce",
        "infiltration",
        "ftppatator",
        "sshpatator",
        "webattackbruteforce"
    }

    if normalized in medium_risk:
        return "Medium Risk", "WARNING", "orange"

    return "High Risk", "ATTACK DETECTED", "red"


def get_security_status_from_risk(risk_level):
    if risk_level == "Low Risk":
        return "Safe"
    if risk_level == "Medium Risk":
        return "Suspicious"
    return "Attack"


def get_simple_explanation(label):
    normalized = normalize_label(label)

    if normalized in {"benign", "normaltraffic"}:
        return "Looks normal. No immediate action is required."

    if normalized == "portscan":
        return "Possible port scanning. Review source IP behavior and block if needed."

    if normalized in {
        "bruteforce",
        "ftppatator",
        "sshpatator",
        "webattackbruteforce",
        "webloginbruteforceattempt"
    }:
        return (
            "Repeated login attempts detected. Review authentication logs and "
            "lockout policy."
        )

    if normalized == "infiltration":
        return (
            "Possible unauthorized access pattern. Investigate internal host "
            "activity."
        )

    if normalized in {
        "ddos",
        "doshulk",
        "dosgoldeneye",
        "dosslowloris",
        "dosslowhttptest",
        "bot"
    }:
        return (
            "Traffic pattern may disrupt services. Check spikes and apply traffic "
            "controls."
        )

    if normalized in {"webattacksqlinjection", "webattackxss", "heartbleed"}:
        return (
            "Possible exploit attempt. Review WAF, app logs, and patch status "
            "immediately."
        )

    return (
        "Potential malicious behavior detected. Investigate logs and isolate "
        "suspicious endpoints."
    )

def create_edge_index(num_nodes):

    edge_index = torch.arange(0, num_nodes, dtype=torch.long)

    return torch.stack([edge_index, edge_index], dim=0)

@app.route("/")
def home():
    if request.args.get("plain") == "1":
        return (
            "<h1>Cyber Intrusion Console</h1>"
            "<p>Server is running. Open <a href='/'>full dashboard</a>.</p>"
            f"<p>Required features: {len(EXPECTED_FEATURES)}</p>"
        )

    return render_template(
        "index.html",
        feature_count=len(EXPECTED_FEATURES),
        feature_names=EXPECTED_FEATURES
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "required_feature_count": len(EXPECTED_FEATURES),
        "class_count": num_classes,
        "model_files": MODEL_FILES
    }), 200

@app.route("/predict", methods=["POST"])
def predict():

    try:

        payload = request.get_json(silent=True)

        if not payload or "features" not in payload:
            return jsonify({"error": "Missing JSON body with 'features' key."}), 400

        raw_features = payload["features"]
        if not isinstance(raw_features, list):
            return jsonify({"error": "'features' must be a list of numeric values."}), 400

        if len(raw_features) != len(EXPECTED_FEATURES):
            return jsonify({
                "error": (
                    f"Expected {len(EXPECTED_FEATURES)} features, "
                    f"received {len(raw_features)}."
                )
            }), 400

        try:
            features = [float(v) for v in raw_features]
        except (TypeError, ValueError):
            return jsonify({"error": "All features must be numeric values."}), 400

        if not np.isfinite(features).all():
            return jsonify({"error": "Feature values must be finite numbers."}), 400

        data_df = pd.DataFrame([features], columns=EXPECTED_FEATURES)

        scaled = scaler.transform(data_df)

        rf_probs = rf_model.predict_proba(scaled)
        rf_pred = np.argmax(rf_probs, axis=1)

        rf_label = label_encoder.inverse_transform(rf_pred)[0]
        rf_conf = float(np.max(rf_probs) * 100)

        xgb_probs = xgb_model.predict_proba(scaled)
        xgb_pred = np.argmax(xgb_probs, axis=1)

        xgb_label = label_encoder.inverse_transform(xgb_pred)[0]
        xgb_conf = float(np.max(xgb_probs) * 100)

        gat_scaled = gat_scaler.transform(data_df)

        x_tensor = torch.tensor(gat_scaled, dtype=torch.float)

        edge_index = create_edge_index(x_tensor.size(0))

        with torch.no_grad():

            out1 = gat_model1(x_tensor, edge_index)

            probs1 = torch.softmax(out1, dim=1)

            gat1_pred = torch.argmax(probs1, dim=1).numpy()

            gat1_label = gat_le.inverse_transform(gat1_pred)[0]

            gat1_conf = float(torch.max(probs1).item() * 100)

        with torch.no_grad():

            out2 = gat_model2(x_tensor, edge_index)

            probs2 = torch.softmax(out2, dim=1)

            gat2_pred = torch.argmax(probs2, dim=1).numpy()

            gat2_label = gat_le.inverse_transform(gat2_pred)[0]

            gat2_conf = float(torch.max(probs2).item() * 100)

        votes = [rf_label, xgb_label, gat1_label, gat2_label]
        final_prediction_raw = Counter(votes).most_common(1)[0][0]
        final_prediction = simplify_label(final_prediction_raw)

        avg_conf = (rf_conf + xgb_conf + gat1_conf + gat2_conf) / 4
        risk_level, system_status, status_color = derive_risk_and_status(
            final_prediction_raw
        )
        security_status = get_security_status_from_risk(risk_level)
        what_this_means = get_simple_explanation(final_prediction_raw)

        return jsonify({

            "RandomForest": simplify_label(rf_label),
            "RandomForestRaw": rf_label,
            "RF_Confidence": round(rf_conf, 2),

            "XGBoost": simplify_label(xgb_label),
            "XGBoostRaw": xgb_label,
            "XGB_Confidence": round(xgb_conf, 2),

            "GAT_Model1": simplify_label(gat1_label),
            "GAT_Model1Raw": gat1_label,
            "GAT1_Confidence": round(gat1_conf, 2),

            "GAT_Model2": simplify_label(gat2_label),
            "GAT_Model2Raw": gat2_label,
            "GAT2_Confidence": round(gat2_conf, 2),

            "FinalDecision": final_prediction,
            "FinalDecisionRaw": final_prediction_raw,

            "SystemStatus": system_status,
            "StatusColor": status_color,

            "SecurityStatus": security_status,
            "RiskLevel": risk_level,
            "WhatThisMeans": what_this_means,
            "AverageConfidence": round(avg_conf, 2)

        }), 200

    except Exception as e:

        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():

    try:

        if "file" not in request.files:
            return jsonify({"error": "Missing file field 'file' in form data."}), 400

        file = request.files["file"]
        if not file or not file.filename:
            return jsonify({"error": "No file selected."}), 400

        df = pd.read_csv(file)
        if df.empty:
            return jsonify({"error": "Uploaded CSV is empty."}), 400

        # Add missing columns automatically
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0

        df = df[EXPECTED_FEATURES].apply(pd.to_numeric, errors="coerce")
        if df.isnull().values.any():
            return jsonify({
                "error": (
                    "CSV contains non-numeric or missing values in required features."
                )
            }), 400

        scaled = scaler.transform(df)

        rf_preds = rf_model.predict(scaled)
        xgb_preds = xgb_model.predict(scaled)

        rf_labels = label_encoder.inverse_transform(rf_preds)
        xgb_labels = label_encoder.inverse_transform(xgb_preds)

        gat_scaled = gat_scaler.transform(df)

        x_tensor = torch.tensor(gat_scaled, dtype=torch.float)

        edge_index = create_edge_index(x_tensor.size(0))

        with torch.no_grad():

            out1 = gat_model1(x_tensor, edge_index)
            gat_preds1 = torch.argmax(out1, dim=1).numpy()
            gat_labels1 = gat_le.inverse_transform(gat_preds1)

            out2 = gat_model2(x_tensor, edge_index)
            gat_preds2 = torch.argmax(out2, dim=1).numpy()
            gat_labels2 = gat_le.inverse_transform(gat_preds2)

        results = []

        for i in range(len(df)):

            rf_raw = rf_labels[i]
            xgb_raw = xgb_labels[i]
            gat1_raw = gat_labels1[i]
            gat2_raw = gat_labels2[i]

            votes = [rf_raw, xgb_raw, gat1_raw, gat2_raw]
            final_prediction_raw = Counter(votes).most_common(1)[0][0]
            final_prediction = simplify_label(final_prediction_raw)
            risk_level, _, _ = derive_risk_and_status(final_prediction_raw)
            security_status = get_security_status_from_risk(risk_level)
            what_this_means = get_simple_explanation(final_prediction_raw)

            results.append({

                "Row": i + 1,

                "RF": simplify_label(rf_raw),
                "RF_Raw": rf_raw,

                "XGB": simplify_label(xgb_raw),
                "XGB_Raw": xgb_raw,

                "GAT_Model1": simplify_label(gat1_raw),
                "GAT_Model1Raw": gat1_raw,

                "GAT_Model2": simplify_label(gat2_raw),
                "GAT_Model2Raw": gat2_raw,

                "FinalDecision": final_prediction,
                "FinalDecisionRaw": final_prediction_raw,
                "RiskLevel": risk_level,
                "SecurityStatus": security_status,
                "WhatThisMeans": what_this_means

            })

        return jsonify({

            "total_rows": len(results),
            "required_feature_count": len(EXPECTED_FEATURES),
            "required_features": EXPECTED_FEATURES,

            "results": results

        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)
