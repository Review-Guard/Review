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

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static'),
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates')
)
CORS(app)