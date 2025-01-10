

CREATE TABLE users (
userID INT PRIMARY KEY,
username VARCHAR(255),
password VARCHAR(255),
salt VARCHAR(255)
);

CREATE TABLE Games (
GamesID INT PRIMARY KEY,
userID INT,
GameDate DATE,
Team1 VARCHAR(255),
Team2 VARCHAR(255),
videoPath VARCHAR(255),
FOREIGN KEY (userID) REFERENCES users(userID)

);

CREATE TABLE labelStats (
GamesID INT,
Frame INT,
Posession INT,
InPlay INT,
Passing INT,
Goal INT,
PRIMARY KEY (GamesID,Frame),
FOREIGN KEY (GamesID) REFERENCES games(GamesID)

);

CREATE TABLE individual_stats (
    GamesID INT,               
    userID INT,                
    TotalPossessionTime FLOAT,  
    TotalInPlayTime FLOAT,      
    TotalPasses INT,            
    SuccessfulPasses INT,       
    FailedPasses INT,           
    PassSuccessRate FLOAT,      
    GoalsScored INT,            
    PossessionPercentage FLOAT, 
    TimePerPossession FLOAT,
    PRIMARY KEY (userID, GamesID),
    FOREIGN KEY (GamesID) REFERENCES games(GamesID)
);

CREATE TABLE overall_stats (
    StatsID INT AUTO_INCREMENT PRIMARY KEY,
    userID INT,                
    TotalGamesPlayed INT,       
    TotalPossessionTime FLOAT,  
    TotalInPlayTime FLOAT,      
    TotalPasses INT,            
    SuccessfulPasses INT,       
    FailedPasses INT,           
    PassSuccessRate FLOAT,     
    TotalGoalsScored INT,       
    AvgPossessionPercentage FLOAT, 
    GoalsPerGame FLOAT,
    FOREIGN KEY (userID) REFERENCES users(userID)
);




