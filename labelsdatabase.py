import mysql.connector
import pandas as pd


conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='B@ba7200',
    database='dev'
)
print("Connection successful!")

games_id = 63
csv_path = r"C:\Users\karin\MatchMentor\static\TestLabelsFullCSV.csv"
# Step 1: Load the CSV file (skip the header row while reading)
df = pd.read_csv(csv_path, skiprows=1, names=["Frame", "Possession", "Passing", "InPlay", "Goal"])
    
# Step 2: Process columns (convert > 0.5 to 1, others to 0)
for col in ["Possession", "Passing", "InPlay", "Goal"]:
    df[col] = df[col].apply(lambda x: 1 if x > 0.5 else 0)
    
insert_query = """
    INSERT INTO labelStats (
        GamesID, 
        Frame, 
        Posession, 
        InPlay, 
        Passing, 
        Goal
    ) VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        Posession = VALUES(Posession),
        InPlay = VALUES(InPlay),
        Passing = VALUES(Passing),
        Goal = VALUES(Goal);
""" 
# Step 4: Insert each row into the database
cursor = conn.cursor()
for _, row in df.iterrows():
    data = (games_id, int(row["Frame"]), int(row["Possession"]), int(row["InPlay"]), int(row["Passing"]), int(row["Goal"]))
    cursor.execute(insert_query, data)
    
# Commit the changes and close the cursor
conn.commit()
cursor.close()
print(f"Processed labels inserted successfully for GamesID {games_id}.")

    
print(f"Processed labels stored successfully in labels' table.")

# Example usage:



