import io
import os
import sys
import time

import pandas as pd
from flask import Flask, jsonify, render_template, request

# Add project root to import path for direct script execution.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # pragma: no cover

PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from app.ml.predict import (
    predict_batch as ml_predict_batch,
    predict_batch_v3,
    predict_single,
    predict_single_v3,
)

MAX_TEXT_LENGTH = 10000


def first_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0] if paths else None


def models_dir_from_version(model_version):
    version = str(model_version or "v3").strip().lower()

    if version in {"v1", "phase1-v1"}:
        return first_existing_path(
            [
                os.path.join(PHASE1_DIR, "artifacts", "models", "v1"),
                os.path.join(PHASE1_DIR, "models_v1"),
            ]
        )

    if version in {"v2", "phase1-v2"}:
        return first_existing_path(
            [
                os.path.join(PHASE1_DIR, "artifacts", "models", "v2"),
                os.path.join(PHASE1_DIR, "models_v2"),
            ]
        )

    if version in {"v3", "phase1-v3"}:
        return first_existing_path(
            [
                os.path.join(PHASE1_DIR, "artifacts", "models", "v3"),
                os.path.join(PHASE1_DIR, "models_v3"),
            ]
        )

    return None


def parse_payload(payload):
    if payload is None:
        raise ValueError("Invalid JSON body")

    if "text" not in payload:
        raise ValueError("Missing required field: text")

    text = payload.get("text")
    if not isinstance(text, str):
        raise TypeError("Field text must be a string")

    if len(text.strip()) == 0:
        raise ValueError("Field text must be non-empty")

    if len(text) > MAX_TEXT_LENGTH:
        raise OverflowError(f"Field text exceeds max length {MAX_TEXT_LENGTH}")

    try:
        rating = float(payload.get("rating", 0.0))
        helpful_vote = float(payload.get("helpful_vote", 0.0))
        verified_purchase = int(payload.get("verified_purchase", 0))
    except Exception as ex:
        raise TypeError("rating/helpful_vote/verified_purchase must be numeric") from ex

    model_version = payload.get("model_version", "v3")
    return text, rating, helpful_vote, verified_purchase, model_version


def run_prediction_for_version(version, text, rating, helpful_vote, verified_purchase):
    models_dir = models_dir_from_version(version)
    if models_dir is None:
        raise ValueError("model_version must be 'v1', 'v2', or 'v3'")

    version_str = str(version).strip().lower()

    if version_str in {"v3", "phase1-v3"}:
        result = predict_single_v3(
            text=text,
            rating=rating,
            helpful_vote=helpful_vote,
            verified_purchase=verified_purchase,
            models_dir=models_dir,
        )
    else:
        result = predict_single(
            text=text,
            rating=rating,
            helpful_vote=helpful_vote,
            verified_purchase=verified_purchase,
            models_dir=models_dir,
        )

    result["fake_probability"] = round(float(result["fake_probability"]) * 100.0, 2)
    result["threshold_percent"] = round(float(result["threshold"]) * 100.0, 2)
    return result


def create_app():
    template_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "frontend", "templates")
    )
    static_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "frontend", "static")
    )

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir,
        static_url_path="/static",
    )

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.post("/batch")
    def batch_predict():
        if "file" not in request.files:
            return jsonify({"error": "Missing CSV file upload"}), 400

        file = request.files["file"]
        if not file or not file.filename:
            return jsonify({"error": "Please choose a CSV file"}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported"}), 400

        try:
            df = pd.read_csv(file)
        except Exception as ex:
            return jsonify({"error": f"Could not read CSV: {str(ex)}"}), 400

        if "text" not in df.columns:
            return jsonify({"error": "CSV must contain a text column"}), 400

        try:
            texts = df["text"].fillna("").astype(str).tolist()
            ratings = (
                pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).tolist()
                if "rating" in df.columns
                else [0.0] * len(df)
            )
            helpful_votes = (
                pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0.0).tolist()
                if "helpful_vote" in df.columns
                else [0.0] * len(df)
            )
            verified_purchase = (
                pd.to_numeric(df["verified_purchase"], errors="coerce")
                .fillna(0)
                .astype(int)
                .tolist()
                if "verified_purchase" in df.columns
                else [0] * len(df)
            )

            model_version = str(request.form.get("model_version", "v3")).strip().lower() or "v3"
            models_dir = models_dir_from_version(model_version)
            if models_dir is None:
                return jsonify({"error": "model_version must be 'v1', 'v2', or 'v3'"}), 400

            if model_version == "v3":
                results = predict_batch_v3(
                    texts=texts,
                    ratings=ratings,
                    helpful_votes=helpful_votes,
                    verified_purchase=verified_purchase,
                    models_dir=models_dir,
                )
            else:
                results = ml_predict_batch(
                    texts=texts,
                    ratings=ratings,
                    helpful_votes=helpful_votes,
                    verified_purchase=verified_purchase,
                    models_dir=models_dir,
                )

            formatted_results = []
            fake_count = 0
            genuine_count = 0

            for item in results:
                label = str(item.get("label", "genuine")).lower()
                fake_probability = round(float(item.get("fake_probability", 0.0)) * 100.0, 2)

                if label == "fake":
                    fake_count += 1
                else:
                    genuine_count += 1

                formatted_results.append(
                    {
                        "text": item.get("text", ""),
                        "label": label,
                        "fake_probability": fake_probability,
                        "fake_percent": fake_probability,
                        "threshold": round(float(item.get("threshold", 0.5)) * 100.0, 2),
                        "model_version": item.get("model_version", model_version),
                    }
                )

            csv_output = io.StringIO()
            pd.DataFrame(formatted_results).to_csv(csv_output, index=False)

            return jsonify(
                {
                    "summary": {
                        "total_reviews": len(formatted_results),
                        "fake_count": fake_count,
                        "genuine_count": genuine_count,
                        "manual_review_count": 0,
                    },
                    "results": formatted_results,
                    "csv_data": csv_output.getvalue(),
                    "count": len(formatted_results),
                }
            ), 200

        except FileNotFoundError:
            return jsonify({"error": "Model artifacts not found. Train the model first."}), 500
        except Exception as ex:
            return jsonify({"error": f"Batch prediction failed: {str(ex)}"}), 500

    @app.post("/predict")
    def predict():
        started = time.time()

        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        try:
            payload = request.get_json(silent=True)
            text, rating, helpful_vote, verified_purchase, model_version = parse_payload(payload)
            result = run_prediction_for_version(
                version=model_version,
                text=text,
                rating=rating,
                helpful_vote=helpful_vote,
                verified_purchase=verified_purchase,
            )
        except OverflowError as ex:
            return jsonify({"error": str(ex)}), 413
        except (ValueError, TypeError) as ex:
            return jsonify({"error": str(ex)}), 400
        except FileNotFoundError:
            return jsonify({"error": "Model artifacts not found. Train the model first."}), 500
        except Exception as ex:
            return jsonify({"error": f"Prediction failed: {str(ex)}"}), 500

        elapsed_ms = (time.time() - started) * 1000.0
        result["latency_ms"] = round(elapsed_ms, 2)
        return jsonify(result), 200

    @app.post("/predict_all")
    def predict_all():
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        try:
            payload = request.get_json(silent=True)
            text, rating, helpful_vote, verified_purchase, _ = parse_payload(payload)
        except OverflowError as ex:
            return jsonify({"error": str(ex)}), 413
        except (ValueError, TypeError) as ex:
            return jsonify({"error": str(ex)}), 400

        try:
            predictions = {}
            labels = []

            for version in ["v1", "v2", "v3"]:
                pred = run_prediction_for_version(
                    version=version,
                    text=text,
                    rating=rating,
                    helpful_vote=helpful_vote,
                    verified_purchase=verified_purchase,
                )
                predictions[version] = pred
                labels.append(str(pred.get("label", "")).lower())

            disagreement = len(set(labels)) > 1
            majority_label = max(set(labels), key=labels.count) if labels else "unknown"
            recommendation = "manual_review" if disagreement else majority_label

            return jsonify(
                {
                    "predictions": predictions,
                    "disagreement": disagreement,
                    "majority_label": majority_label,
                    "recommendation": recommendation,
                }
            ), 200
        except FileNotFoundError:
            return jsonify({"error": "Model artifacts not found. Train the model first."}), 500
        except Exception as ex:
            return jsonify({"error": f"Prediction failed: {str(ex)}"}), 500

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)