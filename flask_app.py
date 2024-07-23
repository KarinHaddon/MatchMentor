"""Runs flask app server that listens to user interactions with MatchMentor site."""
import mysql.connector
from flask import Flask, render_template, request, url_for, redirect
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
    """Renders the test page."""
    return render_template("register.html")


def hashing_algorithm(password):
    hash_val = int(0)
    for char in password:
        hash_val = hash_val ^ ord(char)
        hash_val = (hash_val << 5) | (hash_val >>27)
        hash_val = (hash_val + ord(char))
    hash_val =  hex(hash_val & 0xFFFFFFFFFFFFFFFF)
    return hash_val
   

@app.route("/register_submitpress", methods=["POST"])
def register_submit_press():
    conn = connect()
    cursor = conn.cursor(dictionary=True)

    print("register submit button was pressed!")  # This will print to the Flask app terminal
    username1 = request.form.get('username')
    password1 = request.form.get('password')
    print(f"Username: {username1}")
    print(f"Password: {password1}")
    hashed_password = hashing_algorithm(password1)
    print(f"Hashed Password: {hashed_password}")


    select_query = "SELECT userID FROM users ORDER BY userID DESC LIMIT 1"
    cursor.execute(select_query)
    result = cursor.fetchone()
    last_user_id = result['userID']
    print(f"Last user ID: {last_user_id}")
    new_user_id = int(last_user_id) + 1
    
    insert_query = "INSERT INTO users (userID, username, password) VALUES (%s,%s, %s)"
    user_data = (new_user_id,username1, hashed_password)
    cursor = conn.cursor()
    cursor.execute(insert_query, user_data)
    conn.commit()

    cursor.close()
    conn.close()
    
    return redirect(url_for('login'))



@app.route("/login_submitpress", methods=["POST"])
def login_submit_press():
    conn = connect()
    cursor = conn.cursor(dictionary=True)

    print("login submit button was pressed!")
    username2 = request.form.get('username')
    password2 = request.form.get('password')
    print(f"Username: {username2}")
    print(f"Password: {password2}")
    
    #hashed_password = hashing_algorithm(password)
    select_query = "SELECT username,password FROM users"
    cursor.execute(select_query)
    results = cursor.fetchall()
    for result in results:
        if username2 == result['username']:
            if hashing_algorithm(password2)== result['password']:
                print("successful")
    

    cursor.close()
    conn.close()
    #landing()
    #return redirect('http://127.0.0.1:5500/login')
    return render_template('homepage.html')

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
    app.run(debug=True,port=5500)
