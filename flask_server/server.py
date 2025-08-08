# ./flask_server/server.py
import os
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

from flask import request, jsonify, send_from_directory, abort, url_for
from flask_cors import CORS
from flask_login import login_user, login_required, current_user, logout_user
from werkzeug.utils import secure_filename
import librosa

from flask_server import app
from flask_server.models import db, User, GeneratedImage, GeneratedModel

from flask_server.modules import (
    GenerateLLMDescription,
    GenerateImage,
    Generate3d,
    get_features,
    get_song,
    get_origin,
    get_genres,
    get_instruments,
    get_spotify_search,
    get_itunes_search,
)


CORS(app)


# In-memory store for intermediate results
# In production swap for Redis or a real DB
_store = {}


@app.route("/static/images/<path:filename>")
def serve_generated_image(filename):
    folder = Path(__file__).parent / "static" / "images"
    full_path = folder / filename
    if not os.path.isfile(full_path):
        return abort(404)
    return send_from_directory(str(folder), filename)


@app.route("/static/models/<path:filename>")
@login_required
def serve_generated_model(filename):
    folder = Path(__file__).parent / "static" / "models"
    full_path = folder / filename
    if not full_path.is_file():
        return abort(404)
    return send_from_directory(str(folder), filename)


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


@app.route("/api/me")
def whoami():
    if not current_user.is_authenticated:
        return jsonify(user=None), 200
    return jsonify(user={"id": current_user.id, "username": current_user.username}), 200


@app.route("/api/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify(status="ok"), 200


@app.route("/api/track_snippet", methods=["POST"])
@login_required
def track_snippet():
    data = request.get_json() or {}
    song_name = data.get("songName", "").strip()
    if not song_name:
        return jsonify(error="Must provide a song name"), 400

    info_spotify = get_spotify_search(song_name)
    if not info_spotify.get("error") and info_spotify.get("preview_url"):
        return jsonify(info_spotify), 200

    info_itunes = get_itunes_search(song_name)
    if info_itunes:
        return jsonify(info_itunes), 200

    return jsonify(error="No snippet found on Spotify or iTunes"), 404


@app.route("/api/analyse", methods=["POST"])
@login_required
def analyse():
    try:
        f = request.files.get("file")
        if not f:
            return jsonify(error="No file"), 400

        title = request.form.get("title", "").strip()
        ext = Path(f.filename).suffix or ".mp3"
        if title:
            safe = secure_filename(title) + ext
        else:
            safe = secure_filename(f.filename)

        tmp = NamedTemporaryFile(suffix=Path(safe).suffix, delete=False)
        f.save(tmp.name)
        tmp.flush()

        y, sr = librosa.load(tmp.name, sr=16_000, mono=True)
        features = get_features(y, sr)

        artist = request.form.get("artist", "").strip()

        if title and artist:
            song_info = {
                "title": title,
                "artists": [artist],
                "album": request.form.get("album", "—"),
                "release": request.form.get("release", "—"),
                "genres": [],
            }
        else:
            song_info = get_song(tmp.name)

        genres = get_genres(tmp.name, song_info)
        instruments = get_instruments(y, sr)
        origin = (
            get_origin(song_info["artists"][0])
            if song_info.get("artists")
            else ("—", "—")
        )

        aid = tmp.name
        _store[aid] = {
            "features": features,
            "song_info": song_info,
            "genres": genres,
            "instruments": instruments,
            "origin": origin,
            "filename": song_info["title"],
        }

        return jsonify(analysisId=aid)
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/api/describe", methods=["POST"])
@login_required
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
            features=entry["features"],
            genre_tags=entry["genres"],
            instrument_tags=entry["instruments"],
            song_info=entry.get("song_info") or None,
            origin=entry["origin"],
            building_type="house",
        )
        pid = aid + "_p"
        _store[pid] = {"prompt": raw, "filename": entry["filename"]}

        return jsonify(promptId=pid)
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/api/render", methods=["POST"])
@login_required
def render_image_or_model():
    try:
        data = request.get_json() or {}
        pid = data.get("promptId")
        action = data.get("action", "image")

        prompt_entry = _store.get(pid)
        if not prompt_entry:
            return jsonify(error="Invalid promptId"), 400

        llm_sum = GenerateLLMDescription(
            llm="deepseek-r1:14b", temperature=0, max_tokens=75
        )

        filename = None
        record = None
        serve_fn = None

        if action == "image":
            img_prompt = llm_sum.summarise_for_image(
                raw_description=prompt_entry["prompt"]
            )

            gen = GenerateImage(
                llm="deepseek-r1:14b",
                song_name=prompt_entry["filename"],
                img_prompt=img_prompt,
                num_inference_steps=25,
            )
            filename = gen.save_image()
            gen.save_metadata(
                audio_tags=_store[pid.replace("_p", "")]["features"],
                genre_tags=_store[pid.replace("_p", "")]["genres"],
                instrument_tags=_store[pid.replace("_p", "")]["instruments"],
                llm_description=prompt_entry["prompt"],
                sdxl_prompt=img_prompt,
            )

            with open(filename, "rb") as f:
                blob = f.read()

            record = GeneratedImage(
                user_id=current_user.id,
                filename=Path(filename).name,
                mime_type="image/png",
                data=blob,
            )
            serve_fn = "serve_generated_image"

        elif action == "model":
            model_prompt = llm_sum.summarise_for_3d(
                raw_description=prompt_entry["prompt"]
            )
            gen = Generate3d(prompt=model_prompt)

            mesh_path = gen.save_meshes(prompt_entry["filename"])
            if not mesh_path:
                return jsonify(error="Mesh generation failed"), 500

            abs_path = Path(__file__).parent / "static" / mesh_path

            with open(abs_path, "rb") as f:
                blob = f.read()

            record = GeneratedModel(
                user_id=current_user.id,
                filename=Path(mesh_path).name,
                mime_type="model/obj",
                data=blob,
            )
            serve_fn = "serve_generated_model"
        else:
            return jsonify(error="Invalid action"), 400

        db.session.add(record)
        db.session.commit()

        url = url_for(serve_fn, filename=record.filename, _external=True)

        return (
            jsonify(
                {
                    "type": action,
                    "id": record.id,
                    "filename": record.filename,
                    "url": url,
                }
            ),
            200,
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/api/my_images", methods=["GET"])
@login_required
def list_images():
    imgs = GeneratedImage.query.filter_by(user_id=current_user.id).all()
    load_imgs = []
    for img in imgs:
        url = url_for("serve_generated_image", filename=img.filename, _external=False)
        load_imgs.append({"id": img.id, "filename": img.filename, "url": url})
    load_imgs.reverse()

    return jsonify(images=load_imgs), 200


@app.route("/api/image", methods=["GET"])
@login_required
def view_image():
    last_img = (
        GeneratedImage.query.filter_by(user_id=current_user.id)
        .order_by(GeneratedImage.id.desc())
        .first()
    )

    if not last_img:
        return jsonify(error="No images found"), 404

    url = url_for("serve_generated_image", filename=last_img.filename, _external=False)
    return jsonify({"id": last_img.id, "filename": last_img.filename, "url": url}), 200


@app.route("/api/models", methods=["GET"])
@login_required
def view_model():
    last_model = (
        GeneratedModel.query.filter_by(user_id=current_user.id)
        .order_by(GeneratedModel.id.desc())
        .first()
    )

    if not last_model:
        return jsonify(error="No model found"), 404

    url = url_for(
        "serve_generated_model", filename=last_model.filename, _external=False
    )
    return (
        jsonify({"id": last_model.id, "filename": last_model.filename, "url": url}),
        200,
    )


@app.route("/api/images/<int:image_id>", methods=["DELETE"])
@login_required
def delete_image(image_id):
    img = GeneratedImage.query.filter_by(user_id=current_user.id, id=image_id).first()

    if not img:
        return jsonify(error="Image not found"), 404

    db.session.delete(img)
    db.session.commit()
    return jsonify(status="ok"), 200


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    return send_from_directory(app.static_folder, "index.html")
