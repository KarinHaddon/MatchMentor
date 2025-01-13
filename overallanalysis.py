import mysql.connector

def connect():
    """Establish database connection."""
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='B@ba7200',
        database='dev'
    )


userID = 1274
RangeID = 0

def calculate_totalGames(userID, RangeID):
    conn = connect()
    cursor = conn.cursor()

    gameCount_query = """
        SELECT COUNT(GamesID) AS games
        FROM individual_stats
        WHERE userID = %s AND RangeID = %s
    """
    cursor.execute(gameCount_query, (userID, RangeID))
    result = cursor.fetchone()
    totalGames = result[0]

    cursor.close()
    conn.close()
    return totalGames

def calculate_avgPossessionPercent(userID,RangeID, totalGames):
    conn = connect()
    cursor = conn.cursor()

    possessionPercentCount_query = """
        SELECT SUM(PossessionPercentage) AS avgPossessionPercent
        FROM individual_stats
        WHERE userID = %s AND RangeID = %s
    """
    cursor.execute(possessionPercentCount_query, (userID, RangeID))
    result = cursor.fetchone()
    avgPossessionPercent = result[0]
    avgPossessionPercent = avgPossessionPercent/totalGames


    cursor.close()
    conn.close()
    return avgPossessionPercent

def calculate_totalGoals(userID,RangeID):
    conn = connect()
    cursor = conn.cursor()

    goalCount_query = """
        SELECT SUM(GoalsScored) AS goals
        FROM individual_stats
        WHERE userID = %s AND RangeID = %s
    """
    cursor.execute(goalCount_query, (userID, RangeID))
    result = cursor.fetchone()
    totalgoals = result[0]

    cursor.close()
    conn.close()
    return totalgoals

def calculate_goalsPerGame(totalGoals, totalGames):
    goalsPerGame = round(totalGoals/totalGames,2)
    return goalsPerGame

def calculate_passSuccessRate(userID, RangeID):
    conn = connect()
    cursor = conn.cursor()

    passSuccessRateCount_query = """
        SELECT SUM(PassSuccessRate) AS passSuccessRate
        FROM individual_stats
        WHERE userID = %s AND RangeID = %s
    """
    cursor.execute(passSuccessRateCount_query, (userID, RangeID))
    result = cursor.fetchone()
    passSuccessRate = result[0]
    passSuccessRate = passSuccessRate/totalGames


    cursor.close()
    conn.close()

    return passSuccessRate

def calculate_totalPossessionTime(userID, RangeID):
    conn = connect()
    cursor = conn.cursor()

    PossessionTimeCount_query = """
        SELECT SUM(TotalPossessionTime) AS possessionTime
        FROM individual_stats
        WHERE userID = %s AND RangeID = %s
    """
    cursor.execute(PossessionTimeCount_query, (userID, RangeID))
    result = cursor.fetchone()
    totalPossessionTime = result[0]

    cursor.close()
    conn.close()
    return totalPossessionTime

def calculate_totalPasses(userID,RangeID):
    conn = connect()
    cursor = conn.cursor()

    passCount_query = """
        SELECT SUM(TotalPasses) AS passes
        FROM individual_stats
        WHERE userID = %s AND RangeID = %s
    """
    cursor.execute(passCount_query, (userID, RangeID))
    result = cursor.fetchone()
    totalpasses = result[0]

    cursor.close()
    conn.close()
    return totalpasses

def calculate_SuccessPasses(userID,RangeID):
    conn = connect()
    cursor = conn.cursor()

    passSCount_query = """
        SELECT SUM(SuccessfulPasses) AS SuccessfulPasses
        FROM individual_stats
        WHERE userID = %s AND RangeID = %s
    """
    cursor.execute(passSCount_query, (userID, RangeID))
    result = cursor.fetchone()
    successfulPasses = result[0]

    cursor.close()
    conn.close()
    return successfulPasses

def calculate_FailedPasses(userID,RangeID):
    conn = connect()
    cursor = conn.cursor()

    passFCount_query = """
        SELECT SUM(FailedPasses) AS FailedPasses
        FROM individual_stats
        WHERE userID = %s AND RangeID = %s
    """
    cursor.execute(passFCount_query, (userID, RangeID))
    result = cursor.fetchone()
    failedPasses = result[0]

    cursor.close()
    conn.close()
    return failedPasses

def calculate_totalInPlayTime(userID, RangeID):
    conn = connect()
    cursor = conn.cursor()

    inPlayTimeCount_query = """
        SELECT SUM(TotalInPlayTime) AS totalInPlay
        FROM individual_stats
        WHERE userID = %s AND RangeID = %s
    """
    cursor.execute(inPlayTimeCount_query, (userID, RangeID))
    result = cursor.fetchone()
    totalInPlay = result[0]

    cursor.close()
    conn.close()
    return totalInPlay

totalGames = calculate_totalGames(userID, RangeID)
avgPossessionPercent = round(calculate_avgPossessionPercent(userID, RangeID, totalGames),2)
totalGoals = calculate_totalGoals(userID, RangeID)
avgGoals = calculate_goalsPerGame(totalGoals,totalGames)
passSuccessRate = round(calculate_passSuccessRate(userID, RangeID),2)
totalPossessionTime = round(calculate_totalPossessionTime(userID, RangeID),2)
totalPasses = calculate_totalPasses(userID, RangeID)
successfulPasses = calculate_SuccessPasses(userID, RangeID)
failedPasses = calculate_FailedPasses(userID, RangeID)
totalInPlay = calculate_totalInPlayTime(userID, RangeID)

print(f"Total Games: {totalGames}")
print(f"Average Possession%: {avgPossessionPercent}")
print(f"Total Goals: {totalGoals}")
print(f"Average Goals: {avgGoals}")
print(f"Pass Success%: {passSuccessRate}")
print(f"Total Possession Time (s): {totalPossessionTime}")
print(f"Total Passes: {totalPasses}")
print(f"Total Successful Passes: {successfulPasses}")
print(f"Total Failed Passes: {failedPasses}")
print(f"Total In Play Time (s): {totalInPlay}")

insertStats_query = """
    INSERT INTO overall_stats  (
        StatsID,
        userID, 
        TotalGamesPlayed, 
        TotalPossessionTime, 
        TotalInPlayTime, 
        TotalPasses, 
        SuccessfulPasses, 
        FailedPasses, 
        PassSuccessRate, 
        TotalGoalsScored, 
        AvgPossessionPercentage, 
        GoalsPerGame
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        TotalGamesPlayed = VALUES(TotalGamesPlayed),
        TotalPossessionTime = VALUES(TotalPossessionTime),
        TotalInPlayTime = VALUES(TotalInPlayTime),
        TotalPasses = VALUES(TotalPasses),
        SuccessfulPasses = VALUES(SuccessfulPasses),
        FailedPasses = VALUES(FailedPasses),
        PassSuccessRate = VALUES(PassSuccessRate),
        TotalGoalsScored = VALUES(TotalGoalsScored),
        AvgPossessionPercentage = VALUES(AvgPossessionPercentage),
        GoalsPerGame= VALUES(GoalsPerGame);
    """

# Connect to the database
conn = connect()
cursor = conn.cursor()

StatsID = 1

# Execute the query
cursor.execute(insertStats_query, (
    StatsID,
    userID, 
    totalGames, 
    totalPossessionTime, 
    totalInPlay, 
    totalPasses, 
    successfulPasses, 
    failedPasses, 
    passSuccessRate, 
    totalGoals, 
    avgPossessionPercent, 
    avgGoals
))

# Commit the transaction
conn.commit()
print("Data inserted/updated successfully")

# Close the connection
cursor.close()
conn.close()
