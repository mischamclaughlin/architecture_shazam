from flask import Flask
from config import Config
from jinja2 import StrictUndefined
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_cors import CORS
from werkzeug.security import generate_password_hash


app = Flask(__name__, static_folder="../static/dist", static_url_path="/static")
app.jinja_env.undefined = StrictUndefined
app.config.from_object(Config)

CORS(app)

db = SQLAlchemy(app)

login = LoginManager(app)
login.login_message = "Please log in to access this page."
login.login_message_category = "danger"
login.login_view = "login"  # type: ignore[attr-defined]

from flask_server.models import User, GeneratedImage


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
    }


from flask_server import server
