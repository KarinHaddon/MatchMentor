from flask import Flask, render_template, request, url_for, jsonify, session, redirect
import mysql.connector

app = Flask(__name__)
app.secret_key = "crabitat"

db_config = {
    'user': 'root',
    'password': 'B@ba7200',
    'host': 'localhost',
    'database': 'dev',
}

def connect():
    conn = mysql.connector.connect(**db_config)
    return conn

@app.route("/")
def render_page():
    return render_template("landing.html")

def hashing_algorithm(password):
    hash_val = int(0)
    for char in password:
        hash_val = hash_val ^ ord(char)
        hash_val = (hash_val << 5) | (hash_val >>27)
        hash_val = (hash_val + ord(char))
    hash_val = hex(hash_val & 0xFFFFFFFFFFFFFFFF)
    return hash_val

@app.route("/register_submitpress", methods=["POST"])
def register_submit_press():
    conn = connect()
    cursor = conn.cursor(dictionary=True)

    username1 = request.form.get('username')
    password1 = request.form.get('password')
    print(f"Username: {username1}")
    print(f"Password: {password1}")

    if username1 and password1:
        select_query = "SELECT username FROM users"
        cursor.execute(select_query)
        results = cursor.fetchall()
        for result in results:
            if username1 == result['username']:
                cursor.close()
                conn.close()
                return jsonify({"error": "Username already taken"}), 400

        hashed_password = hashing_algorithm(password1)
        print(f"Hashed Password: {hashed_password}")

        select_query = "SELECT userID FROM users ORDER BY userID DESC LIMIT 1"
        cursor.execute(select_query)
        result = cursor.fetchone()
        last_user_id = result['userID'] if result else 0
        new_user_id = int(last_user_id) + 1

        insert_query = "INSERT INTO users (userID, username, password) VALUES (%s, %s, %s)"
        user_data = (new_user_id, username1, hashed_password)
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

    print("login submit button was pressed!")
    username2 = request.form.get('username')
    password2 = request.form.get('password')
    print(f"Received data - Username: {username2}, Password: {password2}")

    if username2 and password2:
        select_query = "SELECT userID, username, password FROM users"
        cursor.execute(select_query)
        results = cursor.fetchall()
        for result in results:
            if username2 == result['username'] and hashing_algorithm(password2) == result['password']:
                print("Login successful")
                session['username'] = username2
                session['user_id'] = result['userID']
                cursor.close()
                conn.close()
                return jsonify({"redirect": url_for('homepage')})

        print("Login failed")
        cursor.close()
        conn.close()
        return jsonify({"error": "Username or password incorrect"}), 400
    else:
        print("No data received")
        cursor.close()
        conn.close()
        return jsonify({"error": "No data received"}), 400

@app.route('/add_game', methods=['POST'])
def add_game():
    conn = connect()
    cursor = conn.cursor(dictionary=True)

    user_id = session.get('user_id')
    
    select_query = "SELECT GamesID FROM games ORDER BY userID DESC LIMIT 1"
    cursor.execute(select_query)
    result = cursor.fetchone()
    last_games_id = result['GamesID'] if result else 0
    new_games_id = int(last_games_id) + 1
    

    game_date = request.form.get('game_date')
    team1 = request.form.get('team1')
    team2 = request.form.get('team2')

    if game_date and team1 and team2:
        cursor.execute("SELECT IFNULL(MAX(GamesID), 0) + 1 AS new_games_id FROM games")
        new_games_id = cursor.fetchone()['new_games_id']

    conn = connect()
    cursor = conn.cursor()
    insert_query = "INSERT INTO games (GamesID,userID, GameDate, Team1, Team2) VALUES (%s, %s, %s, %s,%s)"
    cursor.execute(insert_query, (new_games_id,user_id, game_date, team1, team2))
    conn.commit()
    cursor.close()
    conn.close()

    return redirect(url_for('homepage'))

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

@app.route('/labelling')
def labelling():
    return render_template('labelling.html')



if __name__ == "__main__":
    app.run(debug=True, port=5500)