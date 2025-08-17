# ./flask_server/__init__.py
import os
from flask import Flask
from flask_migrate import Migrate
from jinja2 import StrictUndefined
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_cors import CORS
from werkzeug.security import generate_password_hash

from config import Config
from .modules.settings import Settings

app = Flask(__name__, static_folder="../static/dist", static_url_path="")
app.jinja_env.undefined = StrictUndefined
app.config.from_object(Config)

settings = Settings()
app.config["APP_SETTINGS"] = settings

CORS(app, supports_credentials=True)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

login = LoginManager(app)
login.login_message = "Please log in to access this page."
login.login_message_category = "danger"
login.login_view = "login"  # type: ignore[attr-defined]

# Import ONLY models
from flask_server.models import (
    User,
    GeneratedImage,
    GeneratedModel,
    Analysis,
    Prompt,
    RenderTask,
)


@login.user_loader
def load_user(user_id: str) -> User | None:
    return User.query.get(int(user_id))


@app.shell_context_processor
def make_shell_context():
    return {
        "db": db,
        "User": User,
        "generate_password_hash": generate_password_hash,
        "GeneratedImage": GeneratedImage,
        "GeneratedModel": GeneratedModel,
        "Analysis": Analysis,
        "Prompt": Prompt,
        "RenderTask": RenderTask,
    }


# Skip registering routes when doing migrations
if not os.getenv("FLASK_SKIP_ROUTES"):
    from flask_server import server
