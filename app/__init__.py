from flask import Flask

app = Flask(__name__)

from app import run
from utilities.tokenize import tokenize
