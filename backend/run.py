import os
import sys
import csv
import json
import io

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to path so 'phase1' is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.predict import predict_one, predict_batch, model_loaded, _load_artifacts, ARTIFACTS_DIR
import time
import os
import sys

from flask import Flask, jsonify, render_template, request

# Add project root to import path for direct script execution.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from phase1.src.predict import predict_single, predict_single_v3


MAX_TEXT_LENGTH = 10000


def models_dir_from_version(model_version):
    # Map API model version selector to local artifact directory.
    version = str(model_version or "v2").strip().lower()
    if version in {"v1", "phase1-v1"}:
        return "phase1/models_v1"
    if version in {"v2", "phase1-v2"}:
        return "phase1/models_v2"
    if version in {"v3", "phase1-v3"}:
        return "phase1/models_v3"
    return None


def create_app():
    # Create Flask app and define API routes.
    app = Flask(__name__, template_folder="templates")

    @app.get("/")
    def home():
        # Render the web tester UI for direct prediction checks.
        return render_template("index.html")

    @app.get("/health")
    def health():
        # Return simple health status for monitoring.
        return jsonify({"status": "ok"}), 200

    @app.post("/predict")
    def predict():
        # Validate request payload and return fake probability prediction.
        started = time.time()

        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Invalid JSON body"}), 400

        if "text" not in payload:
            return jsonify({"error": "Missing required field: text"}), 400

        text = payload.get("text")
        if not isinstance(text, str):
            return jsonify({"error": "Field text must be a string"}), 400
        if len(text.strip()) == 0:
            return jsonify({"error": "Field text must be non-empty"}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"error": f"Field text exceeds max length {MAX_TEXT_LENGTH}"}), 413

        rating = payload.get("rating", 0.0)
        helpful_vote = payload.get("helpful_vote", 0.0)
        verified_purchase = payload.get("verified_purchase", 0)
        model_version = payload.get("model_version", "v2")

        try:
            rating = float(rating)
            helpful_vote = float(helpful_vote)
            verified_purchase = int(verified_purchase)
        except Exception:
            return jsonify({"error": "rating/helpful_vote/verified_purchase must be numeric"}), 400

        models_dir = models_dir_from_version(model_version)
        if models_dir is None:
            return jsonify({"error": "model_version must be 'v1' or 'v2'"}), 400

        try:
            if str(model_version).strip().lower() in {"v3", "phase1-v3"}:
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
        except FileNotFoundError:
            return jsonify({"error": "Model artifacts not found. Train the model first."}), 500
        except Exception as ex:
            return jsonify({"error": f"Prediction failed: {str(ex)}"}), 500

        # Add lightweight latency info to simplify local acceptance checks.
        elapsed_ms = (time.time() - started) * 1000.0
        # Convert probability outputs to percentage for user-facing responses.
        result["fake_probability"] = round(float(result["fake_probability"]) * 100.0, 2)
        result["threshold_percent"] = round(float(result["threshold"]) * 100.0, 2)
        result["latency_ms"] = round(elapsed_ms, 2)
        return jsonify(result), 200

    return app


app = create_app()


if __name__ == "__main__":
    # Run local development server.
    app.run(host="0.0.0.0", port=8000, debug=False)

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static'),
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates')
)
CORS(app)