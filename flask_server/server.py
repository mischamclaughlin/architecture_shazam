# ./flask_server/server.py
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/api/upload", methods=["POST"])
def upload():
    uploaded_file = request.files.get("file")
    if not uploaded_file:
        return {"error": "No file"}, 400

    filename = uploaded_file.filename or ""
    if not filename:
        return jsonify(error="Filename not provided"), 400

    if not filename.lower().endswith(".mp3"):
        return jsonify(error="Invalid file extension, must be .mp3"), 400

    if uploaded_file.mimetype not in ("audio/mpeg", "audio/mp3"):
        return jsonify(error="Invalid MIME type, must be audio.mpeg"), 400

    header = uploaded_file.stream.read(3)
    uploaded_file.stream.seek(0)
    if header != b"ID3" and (header[0] & 0xFF) != 0xFF:
        return jsonify(error="File does not appear to be a valid MP3"), 400

    data = uploaded_file.read()

    return jsonify(status="received", filename=uploaded_file.filename)
