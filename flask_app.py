import cv2
from flask import Flask, render_template, request, url_for, jsonify, session, redirect, send_file
import mysql.connector
#from mysql.connector import Error
#from werkzeug.utils import secure_filename
#import io
import hashlib
import os

app = Flask(__name__, static_folder='static')
app.secret_key = "crabitat"

db_config = {
    'user': 'root',
    'password': 'B@ba7200',
    'host': 'localhost',
    'database': 'dev',
}

def connect():
    conn = mysql.connector.connect(host='localhost', user='root', password='B@ba7200', database='dev')
    return conn

@app.route("/")
def render_page():
    return render_template("landing.html")

def hashing_algorithm(password):
    hash_pass = (hashlib.sha256(password.encode('utf-8'))).hexdigest()

    return hash_pass

@app.route("/register_submitpress", methods=["POST"])
def register_submit_press():
    conn = connect()
    cursor = conn.cursor(dictionary=True)

    username = request.form.get('username')
    password = request.form.get('password')
    print(f"Username: {username}")
    print(f"Password: {password}")

    if username and password:
        select_query = "SELECT username FROM users"
        cursor.execute(select_query)
        results = cursor.fetchall()
        for result in results:
            if username == result['username']:
                cursor.close()
                conn.close()
                return jsonify({"error": "Username already taken"}), 400

        hashed_password = hashing_algorithm(password)
        print(f"Hashed Password: {hashed_password}")

        select_query = "SELECT userID FROM users ORDER BY userID DESC LIMIT 1"
        cursor.execute(select_query)
        result = cursor.fetchone()
        last_user_id = result['userID'] if result else 0
        new_user_id = int(last_user_id) + 1

        insert_query = "INSERT INTO users (userID, username, password) VALUES (%s, %s, %s)"
        user_data = (new_user_id, username, hashed_password)
        cursor.execute(insert_query, user_data)
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"redirect": url_for('login')})

    else:
        cursor.close()
        conn.close()
        return jsonify({"error": "No data received"}), 400

@app.route("/login_submitpress", methods=["POST"])
def login_submit_press():
    conn = connect()
    cursor = conn.cursor(dictionary=True)
    
    username = request.form.get('username')
    password = request.form.get('password')
    print(f"Received data: Username: {username}, Password: {password}")
    if username and password:
        selectQ = "SELECT userID, username, password FROM users"
        cursor.execute(selectQ)
        results = cursor.fetchall()
        for result in results:
            if username == result['username'] and hashing_algorithm(password) == result['password']:
                print("Login successful")
                session['username'] = username
                session['user_id'] = result['userID']
                cursor.close()
                conn.close()
                return jsonify({"redirect": url_for('homepage')})

        print("Login failed")
        cursor.close()
        conn.close()
        return jsonify({"error": "Username or password incorrect"}), 400
    else:
        cursor.close()
        conn.close()
        return jsonify({"error": "Please enter both fields"}), 400

@app.route('/add_game', methods=['POST'])
def add_game():
    conn = connect()
    cursor = conn.cursor(dictionary=True)

    userID = session.get('user_id')
    gameID = session.get('gameID')
    
    game_date = request.form.get('game_date')
    team1 = request.form.get('team1')
    team2 = request.form.get('team2')

    if gameID and game_date and team1 and team2:
        update_query = """
            UPDATE games
            SET GameDate = %s, Team1 = %s, Team2 = %s, userID = %s
            WHERE GamesID = %s
        """
        cursor.execute(update_query, (game_date, team1, team2, userID, gameID))
        conn.commit()

        cursor.close()
        conn.close()
        
        return redirect(url_for('homepage'))
    
    cursor.close()
    conn.close()
    return 'Error adding game'

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/homepage')
def homepage():
    user_id = session.get('user_id')
    username = session.get('username', 'Guest')
    capital_username = username.capitalize()

    conn = connect()
    cursor = conn.cursor(dictionary=True)
    select_query = """
    SELECT 
        GamesID,
        GameDate,
        Team1,
        Team2,
        ROW_NUMBER() OVER (PARTITION BY userID ORDER BY GameDate) as game_number
    FROM 
        games
    WHERE 
        userID = %s
    ORDER BY 
        game_number;
    """

    cursor.execute(select_query, (user_id,))
    games = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('homepage.html', username=capital_username, games=games)


@app.route('/logout')
def logout():
    session.pop('username',None)
    return redirect(url_for('login'))

UPLOAD_FOLDER = 'static/uploads/videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No video file provided', 400
    
    video = request.files['video']
    if video.filename == '':
        return 'No selected file', 400
    
    if video:
        #save the uploaded video to the uploads folder
        filename = video.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(filepath)

        #path for storing frames in the 'static/images' folder
        images_folder = os.path.join('static', 'images')
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        #extract the first frame from the video
        video_capture = cv2.VideoCapture(filepath)
        success, frame = video_capture.read()
        frame_filename = None
        if success:
            frame_filename = "frame_0.jpg"
            frame_path = os.path.join(images_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
        video_capture.release()

        conn = connect()
        cursor = conn.cursor(dictionary=True)
        select_query = "SELECT GamesID FROM games ORDER BY GamesID DESC LIMIT 1"
        cursor.execute(select_query)
        result = cursor.fetchone()
        lGameID = result['GamesID'] if result else 0
        gameID = int(lGameID) + 1

        insert_game_query = "INSERT INTO games (GamesID, userID, GameDate, Team1, Team2) VALUES (%s, NULL, NULL, NULL, NULL)"
        cursor.execute(insert_game_query, (gameID,))
        conn.commit()

        session['gameID'] = gameID

        cursor.close()
        conn.close()

        #redirect to labelling page, passing the video path and first frame details
        return redirect(url_for('labelling', frame_filename=frame_filename, frame_number=1, video_path=filepath))

    return 'File upload failed', 500

@app.route('/labelling', methods=['GET'])
def labelling():
    conn = connect()
    cursor = conn.cursor(dictionary=True)

    frame_filename = request.args.get('frame_filename')
    frame_number = int(request.args.get('frame_number', 0))
    video_path = request.args.get('video_path')

    gameID = session.get('gameID')

    if not gameID:
        cursor.close()
        conn.close()
        return 'Game ID not found in session', 400

    print("GamesID: "+str(gameID))
    if frame_filename:
        frame_path = f'images/{frame_filename}'
    else:
        frame_path = None

    cursor.close()
    conn.close()

    return render_template('labelling.html', frame_path=frame_path, frame_number=frame_number, video_path=video_path, gameID = gameID)

@app.route('/serve_frame')
def serve_frame():
    frame_filename = request.args.get('frame_filename')

    if not frame_filename:
        return 'Frame filename not provided', 400
    
    # Build the full path
    frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
    
    if not os.path.exists(frame_path):
        return 'Frame not found', 404
    
    return send_file(frame_path, mimetype='image/jpeg')

@app.route('/next_frame', methods=['POST'])
def next_frame():

    conn = connect()
    cursor = conn.cursor(dictionary=True)
    
    frame_number = int(request.form.get('frame_number', 0))
    video_path = request.form.get('video_path')
    gameID = session.get('gameID')

    selected_team = request.form.get('selected_team')
    in_play_value = request.form.get('in_play_value')

    session['selected_team'] = selected_team
    session['in_play_value'] = in_play_value

    if selected_team == ('your_team'):
        selected_team = 1
    else:
        selected_team = 0

    if in_play_value == ('true'):
        in_play_value = 1
    else:
        in_play_value = 0

    print(f"Received data: GameID: {gameID}, Frame Number: {frame_number}, Team: {selected_team}, InPlay: {in_play_value}")

    insertQ = """
    INSERT INTO labelstats (GamesID, Frame, Posession, InPlay)
    VALUES (%s,%s,%s,%s)
    """
    cursor.execute(insertQ,(gameID,frame_number,selected_team,in_play_value))
    conn.commit()
    cursor.close()
    conn.close()

    if not video_path or not os.path.exists(video_path):
        return 'Video not found', 404

    
    images_folder = os.path.join('static', 'images')
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Move to the desired frame
    success, frame = video_capture.read()

    frame_filename = None

    if success:
        # Save the frame as an image
        frame_filename = f"frame_{frame_number}.jpg"
        frame_path = os.path.join(images_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_number += 10  # Increment the frame number for the next request

    video_capture.release()

    # Redirect back to labelling with the updated frame
    return redirect(url_for('labelling', frame_filename=frame_filename, frame_number=frame_number, video_path=video_path, gameID = gameID))

@app.route('/stats')
def stats():
    return render_template('stats.html')


def show_frame(frame_path):
    print(frame_path)

    return 


if __name__ == "__main__":
    app.run(debug=True, port=5500)

