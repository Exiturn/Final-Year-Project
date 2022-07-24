import sys
import trueskill
import trueskill as ts
import matplotlib as mpl
import pandas as pd
import numpy as np
import pprint
import random
from random import randrange
from trueskill import Rating, quality_1vs1, rate_1vs1

#read the cod.csv file and display in data from the first 5 and the last 5 cells of the database
df = pd.read_csv('C:/Users/44745/Documents/Final Year/FYP/input/cod.csv')
df.head(10).append(df.tail(10))

#create a new dataframe where the contents of the first column are taken,
#all unique names are checked then displayed are the number of cells in the new
#database and the first 10
unique_ids = pd.DataFrame(df.iloc[:,0].unique(), columns=['name'])
print('Number of unique accounts: ', unique_ids.shape[0], '\n')
print(unique_ids.head(10))

#initialise a trueskill object and changing the draw probablity to 0 for 1v1 scenarios
ts = trueskill.TrueSkill(draw_probability=0)
print(ts)

#creating a dictionary with a default trueskill rating for each unique account
ratings = dict()
for i in unique_ids.values.ravel():
    ratings[i] = ts.create_rating()
#returns the length of the dictionary, it should match the number of unique accounts
print("Length of default ratings dict: ", len(ratings.keys()))
unique_ids_num = unique_ids.shape[0]

#each unique account will have a rating value and the key is their name
pprint.pprint(str(ratings)[:126]+'}')

#find two random players between ones available
def match_random(ratings):
    first_player = random.sample(range(0, unique_ids_num), 1)
    second_player = random.sample(range(0, unique_ids_num), 1)

    if first_player == second_player:
        second_player = random.sample(range(0, unique_ids_num), 1)

    return first_player, second_player

print(match_random(ratings))

updated_ratings = dict()

def update_player_ratings(updated_ratings, ratings):
    for key in ratings.keys():
        updated_ratings[key] = ratings[key]
    
    return updated_ratings

update_player_ratings(updated_ratings, ratings)

matches = df.groupby('wins')
for e, group in enumerate(matches):
    break
print(matches)
# update_player_ratings(updated_ratings, ratings)

# df.columns
# df.sort_values(by=['wins'], ascending=False).head(10)

# alice, bob = Rating(25), Rating(35)
# if quality_1vs1(alice, bob) < 0.50:
#     print("This match seems to not be fair")
# else:
#     print("This match is fair,")
# alice, bob = rate_1vs1(alice, bob)
# ts = trueskill.TrueSkill()
# ts