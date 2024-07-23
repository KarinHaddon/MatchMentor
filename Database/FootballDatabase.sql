

CREATE TABLE Users (
userID INT PRIMARY KEY,
username VARCHAR(255),
password VARCHAR(255)
);

CREATE TABLE Games (
GamesID INT PRIMARY KEY,
userID INT,
GameDate DATE,
Team1 VARCHAR(255),
Team2 VARCHAR(255),
FOREIGN KEY (userID) REFERENCES users(userID)

);

CREATE TABLE labelStats (
GamesID INT,
Frame INT,
Posession INT,
InPlay INT,
PRIMARY KEY (GamesID,Frame),
FOREIGN KEY (GamesID) REFERENCES games(GamesID)

);

CREATE TABLE stats (
GamesID INT,
userID INT,
posession INT,
goals INT,
PRIMARY KEY (GamesID),
FOREIGN KEY (GamesID) REFERENCES games(GamesID),
FOREIGN KEY (userID) REFERENCES games(userID)

);

CREATE TABLE overallStats (
userID INT,
posession INT,
goals INT,
inPlay INT,
PRIMARY KEY (userID),
FOREIGN KEY (userID) REFERENCES stats(userID)

);




