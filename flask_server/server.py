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
from flask_server.models import (
    db,
    User,
    GeneratedImage,
    GeneratedModel,
    Analysis,
    Prompt,
    RenderTask,
)

from flask_server.modules.helpers import (
    get_features,
    get_song,
    get_origin,
    get_genres,
    get_instruments,
    get_spotify_search,
    get_itunes_search,
)
from flask_server.modules.generators.description_generation import (
    GenerateLLMDescription,
)
from flask_server.modules.generators.image_generation import GenerateImage
from flask_server.modules.generators.model_generation import Generate3d
from flask_server.modules.services.meshy_service import MeshyService

# Initialise CORS
CORS(app)


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

        # Optional metadata (used when you search + download preview)
        title = (request.form.get("title") or "").strip()
        artist = (request.form.get("artist") or "").strip()
        album = (request.form.get("album") or "").strip() or None
        release = (request.form.get("release") or "").strip() or None

        # Save to a temp file
        ext = Path(f.filename).suffix or ".mp3"
        safe_name = (
            secure_filename(title) + ext if title else secure_filename(f.filename)
        )
        tmp = NamedTemporaryFile(suffix=Path(safe_name).suffix, delete=False)
        f.save(tmp.name)
        tmp.flush()

        # Audio features
        y, sr = librosa.load(tmp.name, sr=16_000, mono=True)
        features = get_features(y, sr)

        # If caller did not supply song info, try auto
        if not title or not artist:
            song_info = get_song(tmp.name)
            title = song_info.get("title") or title or Path(safe_name).stem
            artist = (song_info.get("artists") or ["—"])[0]
            album = song_info.get("album") or album
            release = song_info.get("release") or release

        genres = get_genres(tmp.name, {"title": title, "artists": [artist]})
        instruments = get_instruments(y, sr)
        origin = get_origin(artist) if artist else ("—", "—")

        # Save the analysis
        analysis = Analysis(
            user_id=current_user.id,
            title=title,
            artist=artist,
            album=album,
            release=release,
            audio_path=tmp.name,
            features=features,
            genres=genres,
            instruments=instruments,
            origin=origin,
        )
        db.session.add(analysis)
        db.session.commit()

        return jsonify(analysisId=analysis.id), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/api/describe", methods=["POST"])
@login_required
def describe():
    try:
        data = request.get_json(silent=True) or {}
        analysis_id = data.get("analysisId")
        if not analysis_id:
            return jsonify(error="Missing analysisId"), 400

        # Ensure row belongs to user
        analysis = (
            db.session.query(Analysis)
            .filter(Analysis.id == analysis_id, Analysis.user_id == current_user.id)
            .first()
        )
        if not analysis:
            return jsonify(error="Invalid analysisId"), 400

        building_type = (data.get("building_type") or "house").strip().lower()
        ALLOWED = {"house", "skyscraper", "apartments"}
        if building_type not in ALLOWED:
            return (
                jsonify(
                    error=f"Invalid building_type. Allowed: {', '.join(sorted(ALLOWED))}"
                ),
                400,
            )

        # LLM settings
        llm_name = "deepseek-r1:14b"
        temperature = 1.0
        max_tokens = 10_000

        # Generate concept text
        llm = GenerateLLMDescription(
            llm=llm_name, temperature=temperature, max_tokens=max_tokens
        )
        raw = llm.generate_description(
            features=analysis.features,
            genre_tags=analysis.genres,
            instrument_tags=analysis.instruments,
            song_info=(
                {"title": analysis.title, "artists": [analysis.artist]}
                if analysis.title and analysis.artist
                else None
            ),
            origin=analysis.origin,
            building_type=building_type,
        )

        # Save Prompt
        prompt = Prompt(
            analysis_id=analysis.id,
            building_type=building_type,
            llm=llm_name,
            temperature=temperature,
            max_tokens=max_tokens,
            raw_prompt=raw,
        )
        db.session.add(prompt)
        db.session.commit()

        return jsonify(promptId=prompt.id, building_type=building_type), 200

    except Exception as e:
        app.logger.exception("describe failed")
        return jsonify(error=str(e)), 500


@app.route("/api/render", methods=["POST"])
@login_required
def render_image_or_model():
    try:
        data = request.get_json() or {}
        prompt_id = data.get("promptId")
        action = (data.get("action") or "image").strip().lower()
        seed = data.get("seed")
        steps = data.get("steps")

        # Join prompt -> analysis (validate owner)
        prompt = (
            db.session.query(Prompt)
            .join(Analysis, Prompt.analysis_id == Analysis.id)
            .filter(Prompt.id == prompt_id, Analysis.user_id == current_user.id)
            .first()
        )
        if not prompt:
            return jsonify(error="Invalid promptId"), 400

        # Create a RenderTask record (RUNNING)
        task = RenderTask(
            prompt_id=prompt.id,
            action=action,
            status="RUNNING",
            seed=seed,
        )
        db.session.add(task)
        db.session.commit()

        # Shared LLM for summarisation
        llm_sum = GenerateLLMDescription(
            llm=prompt.llm, temperature=0.1, max_tokens=10_000
        )

        # ----- IMAGE PATH -----
        if action == "image":
            # Summarise into a 75-token SDXL prompt
            sdxl_prompt = llm_sum.summarise_for_image(raw_description=prompt.raw_prompt)
            task.sdxl_prompt = sdxl_prompt
            task.inference_steps = int(steps or 25)
            db.session.commit()

            # Generate image (Stable Diffusion XL)
            gen = GenerateImage(
                llm=prompt.llm,
                song_name=(prompt.analysis.title or "Untitled"),
                img_prompt=sdxl_prompt,
                num_inference_steps=task.inference_steps,
            )
            image_path = gen.save_image()
            gen.save_metadata(
                audio_tags=prompt.analysis.features,
                genre_tags=prompt.analysis.genres,
                instrument_tags=prompt.analysis.instruments,
                llm_description=prompt.raw_prompt,
                sdxl_prompt=sdxl_prompt,
            )

            # Persist image blob
            with open(image_path, "rb") as f:
                blob = f.read()

            image_row = GeneratedImage(
                user_id=current_user.id,
                filename=Path(image_path).name,
                mime_type="image/png",
                data=blob,
            )
            db.session.add(image_row)

            # Finalise task
            task.status = "SUCCEEDED"
            task.output_filename = image_row.filename
            db.session.commit()

            url = url_for(
                "serve_generated_image", filename=image_row.filename, _external=True
            )
            return (
                jsonify(
                    {
                        "type": action,
                        "id": image_row.id,
                        "filename": image_row.filename,
                        "url": url,
                    }
                ),
                200,
            )

        # ----- MODEL PATH -----
        elif action == "model":
            # Summarise into low-poly modeling prompt
            model_prompt = llm_sum.summarise_for_3d(raw_description=prompt.raw_prompt)
            task.sdxl_prompt = model_prompt  # name is sdxl_prompt in DB, but 3D model prompt is stored here as well
            db.session.commit()

            # Prefer Meshy; fall back to Shap-E path
            try:
                meshy = MeshyService()
                saved = meshy.preview_then_texture(
                    prompt=model_prompt,
                    name_hint=prompt.analysis.title or "Model",
                )
                mesh_rel = saved.get("relative")
                if not mesh_rel:
                    raise RuntimeError("Mesh generation failed (no file)")

                abs_path = Path(__file__).parent / "static" / mesh_rel
                with open(abs_path, "rb") as f:
                    blob = f.read()

                model_row = GeneratedModel(
                    user_id=current_user.id,
                    filename=Path(mesh_rel).name,
                    mime_type="model/gltf-binary",
                    data=blob,
                )
                db.session.add(model_row)

                task.status = "SUCCEEDED"
                task.output_filename = model_row.filename
                db.session.commit()

                url = url_for(
                    "serve_generated_model", filename=model_row.filename, _external=True
                )
                return (
                    jsonify(
                        {
                            "type": action,
                            "id": model_row.id,
                            "filename": model_row.filename,
                            "url": url,
                        }
                    ),
                    200,
                )

            except Exception:
                traceback.print_exc()

                # Fallback: your local Generate3d (OBJ)
                gen3d = Generate3d(prompt=model_prompt)
                mesh_rel = gen3d.save_meshes(prompt.analysis.title or "Model")
                if not mesh_rel:
                    task.status = "FAILED"
                    task.error = "Mesh generation failed"
                    db.session.commit()
                    return jsonify(error="Mesh generation failed"), 500

                abs_path = Path(__file__).parent / "static" / mesh_rel
                with open(abs_path, "rb") as f:
                    blob = f.read()

                model_row = GeneratedModel(
                    user_id=current_user.id,
                    filename=Path(mesh_rel).name,
                    mime_type="model/obj",
                    data=blob,
                )
                db.session.add(model_row)

                task.status = "SUCCEEDED"
                task.output_filename = model_row.filename
                db.session.commit()

                url = url_for(
                    "serve_generated_model", filename=model_row.filename, _external=True
                )
                return (
                    jsonify(
                        {
                            "type": action,
                            "id": model_row.id,
                            "filename": model_row.filename,
                            "url": url,
                        }
                    ),
                    200,
                )

        else:
            task.status = "FAILED"
            task.error = f"Invalid action: {action}"
            db.session.commit()
            return jsonify(error="Invalid action"), 400

    except Exception as e:
        traceback.print_exc()
        # Try to mark task failed if it exists
        try:
            if "task" in locals():
                task.status = "FAILED"
                task.error = str(e)
                db.session.commit()
        except Exception:
            pass
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


@app.route("/api/my_models", methods=["GET"])
@login_required
def list_models():
    imgs = GeneratedModel.query.filter_by(user_id=current_user.id).all()
    load_models = []
    for model in imgs:
        url = url_for("serve_generated_model", filename=model.filename, _external=False)
        load_models.append({"id": model.id, "filename": model.filename, "url": url})
    load_models.reverse()

    return jsonify(models=load_models), 200


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


@app.route("/api/models/<int:model_id>", methods=["DELETE"])
@login_required
def delete_model(model_id):
    m = GeneratedModel.query.filter_by(user_id=current_user.id, id=model_id).first()
    if not m:
        return jsonify(error="Model not found"), 404
    db.session.delete(m)
    db.session.commit()
    return jsonify(status="ok"), 200


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    return send_from_directory(app.static_folder, "index.html")
