import mysql.connector

def connect():
    """Establish database connection."""
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='B@ba7200',
        database='dev'
    )

# Connect to the database
conn = connect()
cursor = conn.cursor()




def calculate_possession(gameID, frame_range):
    conn = connect()
    cursor = conn.cursor()

    # Query to get possession data
    possession_query = """
        SELECT COUNT(Posession) AS total_frames, 
               SUM(Posession) AS possession_frames
        FROM labelstats
        WHERE GamesID = %s
    """
    if frame_range:
        possession_query += " AND Frame BETWEEN %s AND %s"
        cursor.execute(possession_query, (gameID, *frame_range))
    else:
        cursor.execute(possession_query, (gameID,))
    
    result = cursor.fetchone()

    # Extract total frames and possession frames
    total_frames = result[0] if result[0] else 0
    possession_frames = result[1] if result[1] else 0

    cursor.close()
    conn.close()
    return possession_frames, total_frames




def calculate_in_play_time(gameID, frame_range):
    conn = connect()
    cursor = conn.cursor()
    inPlay_query = """
        SELECT COUNT(InPlay) AS total_inPlay
        FROM labelstats
        WHERE GamesID = %s
    """
    if frame_range:
        inPlay_query += " AND Frame BETWEEN %s AND %s"
        cursor.execute(inPlay_query, (gameID, *frame_range))
    else:
        cursor.execute(inPlay_query, (gameID,))
    
    result = cursor.fetchone()

    # Extract total frames and possession frames
    total_inPlay = result[0]


    if total_inPlay > 0:
        total_inPlayTime = (total_inPlay / 30)
    else:
        total_inPlayTime = 0
    cursor.close()
    conn.close()
    return total_inPlay, total_inPlayTime

def calculate_possessionPercentage(total_frames, possession_frames,total_inPlay):
    # Calculate the possession percentage
    if total_frames > 0:
        possession_percentage = (possession_frames / total_inPlay) * 100
    else:
        possession_percentage = 0
    return possession_percentage

def calculate_goals(gameID, frame_range):
    conn = connect()
    cursor = conn.cursor()

    # Query to get goal data
    goal_query = """
        SELECT Goal
        FROM labelstats
        WHERE GamesID = %s
    """
    if frame_range:
        goal_query += " AND Frame BETWEEN %s AND %s"
        goal_query += " ORDER BY Frame"  
        cursor.execute(goal_query, (gameID, *frame_range))
    else:
        goal_query += " ORDER BY Frame"
        cursor.execute(goal_query, (gameID,))
    
    goals = [row[0] for row in cursor.fetchall()]

    # Process goals into sequences
    goal_sequences = 0
    in_sequence = False
    for goal in goals:
        if goal == 1:
            if not in_sequence:  # Start of a new sequence
                goal_sequences += 1
                in_sequence = True
        else:
            in_sequence = False  # End of the current sequence

    cursor.close()
    conn.close()
    return goal_sequences


def calculate_passSequences(gameID, frame_range):
    conn = connect()
    cursor = conn.cursor()
    passes_query = """
        SELECT Passing
        FROM labelstats
        WHERE GamesID = %s
        """
    if frame_range:
        passes_query += " AND Frame BETWEEN %s AND %s"
        passes_query += " ORDER BY Frame" 
        cursor.execute(passes_query, (gameID, *frame_range))
    else:
        cursor.execute(passes_query, (gameID,))
    
    passes = [row[0] for row in cursor.fetchall()]

    pass_sequence = 0
    in_sequence = False

    for passing  in passes:
        if passing  == 1:
            if not in_sequence:  
                    pass_sequence += 1
                    in_sequence = True
        else:
                in_sequence = False  

    cursor.close()
    conn.close()
    return pass_sequence

margin = 1

def calculate_SuccessFailPasses(gameID, frame_range, pass_sequence):
    conn = connect()
    cursor = conn.cursor()
    successPasses_query = """
        SELECT Frame, Passing, Posession
        FROM labelstats
        WHERE GamesID = %s
        """

    if frame_range:
        successPasses_query += " AND Frame BETWEEN %s and %s"
        successPasses_query += " ORDER BY Frame" 
        cursor.execute(successPasses_query, (gameID, *frame_range))
    else:
        cursor.execute(successPasses_query, (gameID,))
    
    rows = cursor.fetchall()  # Fetch all rows

    # Extract data
    frames = [row[0] for row in rows]
    passing = [row[1] for row in rows]
    possession = [row[2] for row in rows]

    # Identify pass sequences
    pass_sequences = []
    in_sequence = False
    start_frame = None

    for i, pass_value in enumerate(passing):
        if pass_value == 1:
            if not in_sequence:
                in_sequence = True
                start_frame = frames[i]
        else:
            if in_sequence:
                in_sequence = False
                end_frame = frames[i - 1]
                pass_sequences.append((start_frame, end_frame))

    # Function to perform majority voting
    def majority_voting(frame_index, margin):
        start_index = max(0, frame_index - margin)
        end_index = min(len(frames) - 1, frame_index + margin)
        possession_window = possession[start_index:end_index + 1]
        return sum(possession_window) > len(possession_window) / 2

    # Check possession at start and end of each pass sequence
    successful_passes = 0
    for start_frame, end_frame in pass_sequences:
        # Find indices of start and end frames
        start_index = frames.index(start_frame)
        end_index = frames.index(end_frame)

        # Check possession at start and end using majority voting
        start_success = majority_voting(start_index, margin)
        end_success = majority_voting(end_index, margin)

        if start_success and end_success:
            successful_passes += 1

    failed_passes = pass_sequence - successful_passes
    pass_success_rate = (successful_passes/pass_sequence)*100

    cursor.close()
    conn.close()
    return successful_passes, failed_passes, pass_success_rate


def calculate_possessionSequenceLength(gameID, frame_range):
    conn = connect()
    cursor = conn.cursor()
    possession_sequence_query = """
        SELECT Posession
        FROM labelstats
        WHERE GamesID = %s
        """

    if frame_range:
        possession_sequence_query += " AND Frame BETWEEN %s AND %s"
        possession_sequence_query += " ORDER BY Frame" 
        cursor.execute(possession_sequence_query, (gameID, *frame_range))
    else:
        cursor.execute(possession_sequence_query, (gameID,))
    
    teampossessions = [row[0] for row in cursor.fetchall()]

    possession_sequence = 0
    in_sequence = False

    possession_lengths = []
    current_length = 0

    for pos in teampossessions:
        if pos  == 1:
            if not in_sequence:  
                    possession_sequence += 1
                    in_sequence = True
        else:
                in_sequence = False  

    for pos in teampossessions:
        if pos == 1: 
            current_length += 1
        else:
            if current_length > 0:  
                possession_lengths.append(current_length)
                current_length = 0

    # Handle case where the last possession continues to the end of the data
    if current_length > 0:
        possession_lengths.append(current_length)

    # Calculate average length
    average_length = sum(possession_lengths) / len(possession_lengths) if possession_lengths else 0

    cursor.close()
    conn.close()
    return possession_sequence, average_length

frame_range = 0
gameID = 63
userID = 1274

frameNumber_query = """
    SELECT COUNT(*)
    FROM labelstats
    WHERE GamesID = %s
    """
cursor.execute(frameNumber_query, (gameID,))
result = cursor.fetchone()

total_frames = result[0]
print(f"Total Frames: {total_frames}")

frame_ranges = [
    (1, total_frames),
    (1, total_frames // 4),
    (total_frames // 4 + 1, total_frames // 2),
    (total_frames // 2 + 1, 3 * total_frames // 4),
    (3 * total_frames // 4 + 1, total_frames)
]
# Iterate through each range and calculate statistics
for range_id, frame_range in enumerate(frame_ranges):
    print(f"\nCalculating statistics for RangeID {range_id} ({frame_range[0]} to {frame_range[1]}):")

    # Possession
    possession_frames, total_frames_segment = calculate_possession(gameID, frame_range)
    total_possession_time = possession_frames / 30  # Convert frames to seconds
    print(f"Total Frames: {total_frames_segment}")
    print(f"Possession Frames: {possession_frames}")
    print(f"Possession Time (s): {total_possession_time:.2f}")

    # In-Play Time
    total_inPlay, total_inPlayTime = calculate_in_play_time(gameID, frame_range)
    print(f"In Play Time (s): {total_inPlayTime:.2f}")

    # Possession Percentage
    possession_percentage = calculate_possessionPercentage(total_frames_segment, possession_frames, total_inPlay)
    print(f"Possession Percentage: {possession_percentage:.2f}%")

    # Goals
    goal_sequences = calculate_goals(gameID, frame_range)
    print(f"Goals Scored: {goal_sequences}")

    # Pass Sequences
    pass_sequence = calculate_passSequences(gameID, frame_range)
    print(f"Total Passes: {pass_sequence}")

    # Successful and Failed Passes
    successful_passes, failed_passes, pass_success_rate = calculate_SuccessFailPasses(gameID, frame_range, pass_sequence)
    print(f"Successful Passes: {successful_passes}")
    print(f"Failed Passes: {failed_passes}")
    print(f"Pass Success Rate: {pass_success_rate:.2f}%")

    # Possession Sequences
    possession_sequence, average_length = calculate_possessionSequenceLength(gameID, frame_range)
    average_length_seconds = average_length / 30  # Convert frames to seconds
    print(f"Possession Sequences: {possession_sequence}")
    print(f"Average Length (s): {average_length_seconds:.2f}")

    # Insert or update the statistics in the database
    insertStats_query = """
        INSERT INTO individual_stats (
            GamesID, 
            userID, 
            RangeID,
            TotalPossessionTime, 
            TotalInPlayTime, 
            TotalPasses, 
            SuccessfulPasses, 
            FailedPasses, 
            PassSuccessRate, 
            GoalsScored, 
            PossessionPercentage, 
            TimePerPossession
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            TotalPossessionTime = VALUES(TotalPossessionTime),
            TotalInPlayTime = VALUES(TotalInPlayTime),
            TotalPasses = VALUES(TotalPasses),
            SuccessfulPasses = VALUES(SuccessfulPasses),
            FailedPasses = VALUES(FailedPasses),
            PassSuccessRate = VALUES(PassSuccessRate),
            GoalsScored = VALUES(GoalsScored),
            PossessionPercentage = VALUES(PossessionPercentage),
            TimePerPossession = VALUES(TimePerPossession);
    """

    # Connect to the database
    conn = connect()
    cursor = conn.cursor()

    # Execute the query
    cursor.execute(insertStats_query, (
        gameID, 
        userID, 
        range_id,  # Current RangeID
        round(total_possession_time, 2), 
        round(total_inPlayTime, 2), 
        pass_sequence, 
        successful_passes, 
        failed_passes, 
        round(pass_success_rate, 2), 
        goal_sequences, 
        round(possession_percentage, 2), 
        round(average_length_seconds, 2)
    ))

    # Commit the transaction
    conn.commit()
    print("Data inserted/updated successfully for RangeID:", range_id)

    # Close the connection
    cursor.close()
    conn.close()
