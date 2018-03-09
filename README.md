# BigData-Recommender Project
Section 1 - Team 4 Big Data Project

Repository contains the sample code for the Gold section 1, team 4 Big Data project building a collaborative filtering recommendation system using Spark MLLib. 

The following files are included: 

- ProjectFullData.py - Full code which was implemented on AWS
- RSpark_Recom_sample.ipynb - Sample code to be run from this folder.
- Trends Handout.png : Handout to be distributed

# Data Files: 
- song_data_sample.csv
- triplets_file.txt

## Data Dictionary:

### Triplets File

|Attribute    | Description     | 
|-------------|-----------------|
| UserID      |users who listen to the songs, there are totally 1,019,318 unique users|
|SongID       |songs listened by users, 384,546 unique MSD songs|
|ListenCount  |Number of times that a user has listened to an individual song|

### Song Data

|Attribute    | Description     | 
|-------------|-----------------|
| SongID      |songs listened by users, 384,546 unique MSD songs|
| SongName    |Name of the song|
|ArtistName   |Artist Name|
|Year         | Year of song|




# Selecting a Listening Profile
From the tabs which display, select a listening profile which matches your taste preferences then click "Generate Recommendation"

Wait 15-30 seconds and your recommendations will appear. 

