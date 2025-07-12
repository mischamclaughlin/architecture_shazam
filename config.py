# ./config.py
import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or b"WR#&f&+%78er0we=%799eww+#7^90-;s"

    UPLOAD_FOLDER = os.path.join(basedir, "instance", "data", "uploads")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    REMEMBER_COOKIE_DURATION = 60 * 60 * 24 * 7

    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(
        basedir, "instance", "data", "data.sqlite"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
