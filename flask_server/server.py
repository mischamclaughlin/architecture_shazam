# flask_server/server.py
import os
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

from flask import request, jsonify, send_from_directory, abort
from flask_cors import CORS
from flask_login import login_user, login_required, current_user
from werkzeug.utils import secure_filename

from flask_server import app
from flask_server.models import db, User, GeneratedImage
from flask_server.modules.song_info import ExtractSongInfo
from flask_server.modules.llm_description import GenerateLLMDescription
from flask_server.modules.image_generation import GenerateImage

CORS(app)

# In-memory store for intermediate results
# In production swap for Redis or a real DB
_store = {}


@app.route("/static/images/<path:filename>")
def serve_generated_image(filename):
    image_dir = os.path.join(os.path.dirname(__file__), "static", "images")
    full_path = os.path.join(image_dir, filename)
    if not os.path.isfile(full_path):
        return abort(404)
    return send_from_directory(image_dir, filename)


@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not username or not email or not password:
        return jsonify(error="Missing fields"), 400
    if User.query.filter_by(username=username).first():
        return jsonify(error="Username already exists"), 400
    if User.query.filter_by(email=email).first():
        return jsonify(error="Email already registered"), 400

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify(status="ok"), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify(error="Username and Password are required"), 400

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify(error="Invalid Username or Password"), 401

    login_user(user)
    return jsonify(status="ok", user={"id": user.id, "username": user.username}), 200


@login_required
@app.route("/api/analyse", methods=["POST"])
def analyse():
    try:
        f = request.files.get("file")
        if not f:
            return jsonify(error="No file"), 400

        # save to temp
        safe = secure_filename(f.filename)
        tmp = NamedTemporaryFile(suffix=Path(safe).suffix, delete=False)
        f.save(tmp.name)
        tmp.flush()

        # extract features
        info = ExtractSongInfo(file_path=tmp.name)
        librosa_info = info.analyse_features()
        genre_info = info.get_genre()
        inst_info = info.get_instruments()

        # store under an ID
        aid = tmp.name  # use the temp path as a simple key
        _store[aid] = {
            "librosa": librosa_info,
            "genre": genre_info,
            "instrument": inst_info,
            "filename": safe,
        }

        return jsonify(analysisId=aid)
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@login_required
@app.route("/api/describe", methods=["POST"])
def describe():
    try:
        data = request.get_json() or {}
        aid = data.get("analysisId")
        entry = _store.get(aid)
        if not entry:
            return jsonify(error="Invalid analysisId"), 400

        llm = GenerateLLMDescription(
            llm="deepseek-r1:14b", temperature=0.7, max_tokens=10_000
        )
        raw = llm.generate_description(
            librosa_tags=entry["librosa"],
            genre_tags=entry["genre"],
            instrument_tags=entry["instrument"],
            building_type="house",
        )
        # store prompt
        pid = aid + "_p"
        _store[pid] = {"prompt": raw, "filename": entry["filename"]}

        return jsonify(promptId=pid)
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@login_required
@app.route("/api/render", methods=["POST"])
def render_image():
    try:
        data = request.get_json() or {}
        pid = data.get("promptId")
        prompt_entry = _store.get(pid)
        if not prompt_entry:
            return jsonify(error="Invalid promptId"), 400

        # build image prompt
        llm_sum = GenerateLLMDescription(
            llm="deepseek-r1:14b", temperature=0, max_tokens=75
        )
        img_prompt = llm_sum.summarise_for_image(raw_description=prompt_entry["prompt"])

        # generate & save
        gen = GenerateImage(
            llm="deepseek-r1:14b",
            song_name=prompt_entry["filename"],
            img_prompt=img_prompt,
            num_inference_steps=25,
        )
        file_path = gen.save_image()
        gen.save_metadata(
            audio_tags=_store[pid.replace("_p", "")]["librosa"],
            genre_tags=_store[pid.replace("_p", "")]["genre"],
            instrument_tags=_store[pid.replace("_p", "")]["instrument"],
            llm_description=prompt_entry["prompt"],
            sdxl_prompt=img_prompt,
        )

        with open(file_path, "rb") as f:
            blob = f.read()

        img_record = GeneratedImage(
            user_id=current_user.id,
            filename=file_path.name,
            mime_type="image/png",
            data=blob,
        )
        db.session.add(img_record)
        db.session.commit()

        relative = file_path.relative_to(Path(__file__).parent / "static" / "images")

        return jsonify(
            imageUrl=f"/static/images/{relative.as_posix()}", imageId=img_record.id
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    return send_from_directory(app.static_folder, "index.html")
