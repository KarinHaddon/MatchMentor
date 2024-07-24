from flask import Flask, render_template, request, url_for, jsonify
import mysql.connector

app = Flask(__name__)

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
    return render_template("register.html")

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
        select_query = "SELECT username, password FROM users"
        cursor.execute(select_query)
        results = cursor.fetchall()
        for result in results:
            if username2 == result['username'] and hashing_algorithm(password2) == result['password']:
                print("Login successful")
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
    return render_template('homepage.html')

if __name__ == "__main__":
    app.run(debug=True, port=5500)