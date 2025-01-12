import cv2
from flask import Flask, render_template, request, url_for, jsonify, session, redirect, send_file
import mysql.connector
import hashlib
import os
import csv 
import re
import base64


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

    username_pattern = r"^[a-zA-Z0-9]{5,15}$"
    if not re.match(username_pattern, username):
        cursor.close()
        conn.close()
        return jsonify({"error": "Invalid username. Must be 5-15 alphanumeric characters."}), 400

    # Password validation
    password_pattern = r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d).{8,20}$"
    if not re.match(password_pattern, password):
        cursor.close()
        conn.close()
        return jsonify({"error": "Invalid password. Must be 8-20 characters, include one uppercase letter, one lowercase letter, and one digit."}), 400

    if username and password:
        # Check if the username already exists
        select_query = "SELECT username FROM users"
        cursor.execute(select_query)
        results = cursor.fetchall()
        for result in results:
            if username == result['username']:
                cursor.close()
                conn.close()
                return jsonify({"error": "Username already taken"}), 400

        # Generate a 32-byte salt
        salt = os.urandom(32)
        # Hash the password with the salt
        salted_password = salt + password.encode('utf-8')
        hashed_password = hashlib.sha256(salted_password).hexdigest()
        # Encode the salt for storage (base64)
        encoded_salt = base64.b64encode(salt).decode('utf-8')

        print(f"Salt: {encoded_salt}")
        print(f'Salted Password: {salted_password}')
        print(f"Hashed Password: {hashed_password}")
        

        # Get the next userID
        select_query = "SELECT userID FROM users ORDER BY userID DESC LIMIT 1"
        cursor.execute(select_query)
        result = cursor.fetchone()
        last_user_id = result['userID'] if result else 0
        new_user_id = int(last_user_id) + 1

        # Insert new user into the database
        insert_query = "INSERT INTO users (userID, username, password, salt) VALUES (%s, %s, %s, %s)"
        user_data = (new_user_id, username, hashed_password, encoded_salt)
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
    print(f"Username: {username}")
    print(f"Password: {password}")


    # Fetch the stored hashed password and salt for the given username
    query = "SELECT userID, password, salt FROM users WHERE username = %s"
    cursor.execute(query, (username,))
    result = cursor.fetchone()

    print(f"Result: {result}")

    if not result:
        # Username not found
        cursor.close()
        conn.close()
        return jsonify({"error": "Invalid username or password"}), 400

    # Extract the stored hashed password and salt
    stored_hashed_password = result['password']
    stored_salt = result['salt']

    # Decode the stored salt from base64
    decoded_salt = base64.b64decode(stored_salt)

    # Hash the provided password with the stored salt
    salted_password = decoded_salt + password.encode('utf-8')
    hashed_password = hashlib.sha256(salted_password).hexdigest()

    # Compare the provided password's hash with the stored hashed password
    if hashed_password == stored_hashed_password:
        # Login successful
        print("Login successful")
        session['username'] = username
        print(f'username: {username}')
        userID = result['userID']
        print(f'userid: {userID}')
        session['user_id'] = userID
        cursor.close()
        conn.close()

        return jsonify({"redirect": url_for('homepage')})
    else:
        # Invalid password
        cursor.close()
        conn.close()
        return jsonify({"error": "Invalid username or password"}), 400
    
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
    session.pop('gameID',None)
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

@app.route('/addgame', methods=['POST'])
def add_game():
    conn = connect()
    cursor = conn.cursor(dictionary=True)

    gameID = session.get('gameID')
    userID = session.get('user_id')
    game_date = request.form.get('game_date')
    team1 = request.form.get('team1')
    team2 = request.form.get('team2')
    video_path = session.get('video_path')

    if game_date and team1 and team2:
        update_game_query = """
            UPDATE games 
            SET GameDate = %s, Team1 = %s, Team2 = %s, userID = %s, videoPath = %s
            WHERE GamesID = %s
        """
        
        cursor.execute(update_game_query, (game_date, team1, team2, userID, video_path, gameID))
        conn.commit()


        
        print(f"Used GameID 2: {gameID}")

        labelQ = """
            SELECT *
            FROM labelstats
            WHERE GamesID = %s
        """
        cursor.execute(labelQ,(gameID,))
        labels = cursor.fetchall()
        print("Labels fetched:", labels)


        csvPath = 'static/labels2.csv'
        images_folder = 'C:/Users/karin/MatchMentor/static/images/'
        videos_path = 'C:/Users/karin/MatchMentor/'
        text_path = 'C:/Users/karin/MatchMentor/static/run_training.txt'
        with open(csvPath, mode ='w') as file:
            writer = csv.DictWriter(file,fieldnames=labels[0].keys())
            writer.writeheader()
            writer.writerows(labels)

        scp_command = f"scp {csvPath} hpc-gw1:~/MatchMentor/data"
        os.system(scp_command)

        scp_images_command = f"scp {images_folder}* hpc-gw1:~/MatchMentor/data"
        os.system(scp_images_command)
        print("All images transmitted")

        print(video_path)
        video_path = video_path.replace("\\", "/")
        print(video_path)
        scp_video_command = f"scp {videos_path}{video_path} hpc-gw1:~/MatchMentor/data"
        os.system(scp_video_command)
        print("Video sent")


        scp_text_command = f"scp {text_path} hpc-gw1:~/MatchMentor/data"
        os.system(scp_text_command)
        print("Sent text file")

        cursor.close()
        conn.close()


        
        session.pop('video_path', None)

    return redirect(url_for('homepage'))

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
        filename = video.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(filepath)

        images_folder = os.path.join('static', 'images')
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        video_capture = cv2.VideoCapture(filepath)
        success, frame = video_capture.read()
        frame_filename = None
        if success:
            frame_filename = "frame_0.jpg"
            frame_path = os.path.join(images_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
        video_capture.release()

        session['video_path'] = filepath
        
        return redirect(url_for('labelling', frame_filename=frame_filename, frame_number=1, video_path=filepath))

    return 'File upload failed', 500

@app.route('/serve_frame')
def serve_frame():
    frame_filename = request.args.get('frame_filename')

    if not frame_filename:
        return 'Frame filename not provided', 400
    
    frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
    
    if not os.path.exists(frame_path):
        return 'Frame not found', 404
    
    return send_file(frame_path, mimetype='image/jpeg')

@app.route('/labelling', methods=['GET'])
def labelling():
    gameID = session.get('gameID')
    print("GamesID before: " + str(gameID))
    conn = connect()
    cursor = conn.cursor(dictionary=True)

    frame_filename = request.args.get('frame_filename')
    frame_number = int(request.args.get('frame_number', 0))
    video_path = request.args.get('video_path')
    

    if not gameID:

        select_query = "SELECT GamesID FROM games ORDER BY GamesID DESC LIMIT 1"
        cursor.execute(select_query)
        result = cursor.fetchone()
        lgameID = result['GamesID'] if result else 0
        print(f"lgameID: {gameID}")
        gameID = lgameID + 1
        print(f"lgameID+1: {gameID}")
        session['gameID'] = gameID 

        insert_game_query = "INSERT INTO games (GamesID, userID, GameDate, Team1, Team2) VALUES (%s, NULL, NULL, NULL, NULL)"
        cursor.execute(insert_game_query, (gameID,))
        conn.commit()

    print("GamesID: " + str(gameID))
    
    if frame_filename:
        frame_path = f'images/{frame_filename}'
    else:
        frame_path = None

    cursor.close()
    conn.close()

    return render_template('labelling.html', frame_path=frame_path, frame_number=frame_number, video_path=video_path, gameID=gameID)

@app.route('/next_frame', methods=['POST'])
def next_frame():


    conn = connect()
    cursor = conn.cursor(dictionary=True)
    
    frame_number = int(request.form.get('frame_number', 0))
    video_path = request.form.get('video_path')
    gameID = (session.get('gameID')) 
    print(f"GameID being used for labeling: {gameID}")

    selected_team = request.form.get('selected_team')
    in_play_value = request.form.get('in_play_value')
    passing_value = request.form.get('passing_value')
    goal_value = request.form.get('goal_value')

    session['selected_team'] = selected_team
    session['in_play_value'] = in_play_value
    session['passing_value'] = passing_value
    session['goal_value'] = goal_value

    if selected_team == ('true'):
        selected_team = 1
    else:
        selected_team = 0

    if in_play_value == ('true'):
        in_play_value = 1
    else:
        in_play_value = 0

    if passing_value == ('true'):
        passing_value = 1
    else:
        passing_value = 0

    if goal_value == ('true'):
        goal_value = 1
    else:
        goal_value = 0

    print(f"Received data: GameID: {gameID}, Frame Number: {frame_number}, Team: {selected_team}, InPlay: {in_play_value}, Passing: {passing_value}, Goal: {goal_value}")

    insertQ = """
    INSERT INTO labelstats (GamesID, Frame, Posession, InPlay, Passing, Goal)
    VALUES (%s,%s,%s,%s,%s,%s)
    """
    cursor.execute(insertQ,(gameID,frame_number,selected_team,in_play_value,passing_value,goal_value))
    conn.commit()
    cursor.close()
    conn.close()

    if not video_path or not os.path.exists(video_path):
        return 'Video not found', 404

    
    images_folder = os.path.join('static', 'images')
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    video_capture = cv2.VideoCapture(video_path)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number) 
    success, frame = video_capture.read()

    frame_filename = None

    if success:
        frame_filename = f"frame_{frame_number}.jpg"
        frame_path = os.path.join(images_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_number += 10 

    video_capture.release()

    return redirect(url_for('labelling', frame_filename=frame_filename, frame_number=frame_number, video_path=video_path, gameID = gameID))

@app.route('/stats/<int:gameID>')
def stats(gameID):
    conn = connect()
    cursor = conn.cursor(dictionary=True)

    select_query = """
    SELECT GamesID, GameDate, Team1, Team2, videoPath, ROW_NUMBER() OVER (ORDER BY GameDate) as game_number
    FROM games
    WHERE GamesID = %s;
    """
    
    cursor.execute(select_query, (gameID,))
    game = cursor.fetchone()
    print(game)  

    rangeID = 0

    stats_query = """
    SELECT PossessionPercentage, SuccessfulPasses, FailedPasses, GoalsScored, PassSuccessRate, TimePerPossession, TotalInPlayTime, TotalPossessionTime 
    FROM individual_stats
    WHERE GamesID = %s AND RangeID = %s;
    """
    cursor.execute(stats_query, (gameID, rangeID))
    statGame_data = cursor.fetchall()
    team1_possession = statGame_data[0]['PossessionPercentage']
    successful_passes = statGame_data[0]['SuccessfulPasses']
    failed_passes = statGame_data[0]['FailedPasses']
    goals = statGame_data[0]['GoalsScored']
    pass_success_rate = statGame_data[0]['PassSuccessRate']
    time_per_possession = statGame_data[0]['TimePerPossession']
    total_inPlay_time = statGame_data[0]['TotalInPlayTime']
    total_Possession_time = statGame_data[0]['TotalPossessionTime']

    print(f'Successful Passes: {successful_passes}')
    print(f'Failed Passes: {failed_passes}')
    print(f'possession_data: {team1_possession }')
    print(f'Goals scored: {goals}')

    context = {
    'team1_possession': team1_possession,
    'successful_passes': successful_passes,
    'failed_passes': failed_passes,
    'goals': goals,
    'pass_success_rate': pass_success_rate,
    'time_per_possession': time_per_possession,
    'total_inPlay_time': total_inPlay_time,
    'total_Possession_time': total_Possession_time
    }

    cursor.close()
    conn.close()

    if game:
        video_path = game.get('videoPath', None)
        if video_path:
            print(f"Original video path: {video_path}")
            video_path = video_path.replace('\\', '/')  
            
            if not os.path.isabs(video_path):
                video_path = f"uploads/videos/{os.path.basename(video_path)}"

            video_path = video_path.replace(' ', '%20')

            print(f"Processed video path: {video_path}")
        else:
            print("No video path available for this game.")
            video_path = None  
        
        return render_template('stats.html', game=game, video_path=video_path, **context)
    else:
        return "Game not found", 404


def show_frame(frame_path):
    print(frame_path)

    return 




if __name__ == "__main__":
    app.run(debug=True, port=5000)