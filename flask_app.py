"""Runs flask app server that listens to user interactions with MatchMentor site."""

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

@app.route("/")
def render_page():
    """Renders the test page."""
    return render_template("test_flask_ajax.html")

@app.route("/button_press", methods=["POST"])
def button_press():
    """Listens for button press and prints to Flask app."""
    print("Button was pressed!")  # This will print to the Flask app terminal
    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(debug=True)
