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

# Define the GameID
gameID = 63

# Query to get the possession data for the specified game
possession_query = """
    SELECT COUNT(Posession) AS total_frames, 
           SUM(Posession) AS possession_frames
    FROM labelstats
    WHERE GamesID = %s;
"""
cursor.execute(possession_query, (gameID,))
result = cursor.fetchone()

# Extract total frames and possession frames
total_frames = result[0]
possession_frames = result[1]



inPlay_query = """
    SELECT COUNT(InPlay) AS total_inPlay
    FROM labelstats
    WHERE GamesID = %s;
"""
cursor.execute(inPlay_query, (gameID,))
result = cursor.fetchone()

# Extract total frames and possession frames
total_inPlay = result[0]


if total_inPlay > 0:
    total_inPlayTime = (total_inPlay / 30)
else:
    total_inPlayTime = 0

# Calculate the possession percentage
if total_frames > 0:
    possession_percentage = (possession_frames / total_inPlay) * 100
else:
    possession_percentage = 0

goal_query = """
    SELECT Goal
    FROM labelstats
    WHERE GamesID = %s
    ORDER BY Frame;
    """
cursor.execute(goal_query, (gameID,))
goals = [row[0] for row in cursor.fetchall()]

goal_sequences = 0
in_sequence = False

for goal in goals:
    if goal == 1:
        if not in_sequence:  # Start of a new sequence
                goal_sequences += 1
                in_sequence = True
    else:
            in_sequence = False  # End of the current sequence

passes_query = """
    SELECT Passing
    FROM labelstats
    WHERE GamesID = %s
    ORDER BY Frame;
    """
cursor.execute(passes_query, (gameID,))
passes = [row[0] for row in cursor.fetchall()]

pass_sequence = 0
in_sequence = False

for passing  in passes:
    if passing  == 1:
        if not in_sequence:  # Start of a new sequence
                pass_sequence += 1
                in_sequence = True
    else:
            in_sequence = False  # End of the current sequence

margin = 1

successPasses_query = """
    SELECT Frame, Passing, Posession
    FROM labelstats
    WHERE GamesID = %s
    ORDER BY Frame;
    """
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
    """Perform majority voting on possession within a margin around the given frame index."""
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

possession_sequence_query = """
    SELECT Posession
    FROM labelstats
    WHERE GamesID = %s
    ORDER BY Frame;
    """
cursor.execute(possession_sequence_query, (gameID,))
teampossessions = [row[0] for row in cursor.fetchall()]

possession_sequence = 0
in_sequence = False

possession_lengths = []
current_length = 0

for pos in teampossessions:
    if pos  == 1:
        if not in_sequence:  # Start of a new sequence
                possession_sequence += 1
                in_sequence = True
    else:
            in_sequence = False  # End of the current sequence

for pos in teampossessions:
    if pos == 1:  # In possession
        current_length += 1
    else:
        if current_length > 0:  # End of a sequence
            possession_lengths.append(current_length)
            current_length = 0

# Handle case where the last possession continues to the end of the data
if current_length > 0:
    possession_lengths.append(current_length)

# Calculate average length
average_length = sum(possession_lengths) / len(possession_lengths) if possession_lengths else 0


print(f"GameID: {gameID}")
print(f"Total Frames: {total_frames}")
print(f"Possession Frames: {possession_frames}")
print(f"Possession Time (1s): {int(possession_frames/30)}")
print(f"Possession Percentage: {possession_percentage:.2f}%")
print(f"In Play Time (1s): {int(total_inPlayTime)}")
print(f"Goals: {goal_sequences}")
print(f"Passes: {pass_sequence}")
print(f"Total successful passes: {successful_passes}")
print(f"Total failed passes: {failed_passes}")
print(f"Successful pass rate: {pass_success_rate:.2f}%")
print(f"Possession sequences: {possession_sequence}")
print(f"Average Length in Seconds: {average_length / 30:.2f} seconds")


# Close the connection
cursor.close()
conn.close()
