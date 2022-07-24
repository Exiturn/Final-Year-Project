import sys
import trueskill
import trueskill as ts
import matplotlib.pyplot as mpl
import pandas as pd
import numpy as np
from scipy import stats
from random import randrange
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def calc_rating(mu, sigma):
    return mu-2*sigma

#read the results.csv file and display the data for for the first 10 and last
#10 indicies of the database
match_results = pd.read_csv('C:/Users/44745/Documents/Final Year/FYP/input/results.csv')
# match_results.head(10).append(match_results.tail(10))

#find all data that is from the 2017/2018 season using column value verification. A
#truncated version of the df will be stored while dropping noisy columns
latest_season = pd.DataFrame(match_results.loc[match_results['season'] == '2017-2018'])
latest_season_tr = latest_season[['home_team', 'away_team', 'result']].copy()
# latest_season_tr.head(10).append(latest_season_tr.tail(10))

teams = pd.DataFrame(latest_season.iloc[:,0].unique(), columns=['Team'])
# print("\n Number of participating teams: ", teams.shape[0], "\n")

#adding new columns into the dataframe for binary results
latest_season_tr['HomeTeamResult'] = latest_season_tr['result']
latest_season_tr['AwayTeamResult'] = latest_season_tr['result']

# #Add binary results for both home and away teams, where Win and Draw = 0 & Loss = 1
latest_season_tr.replace({'HomeTeamResult' : {'H':0, 'D':0, 'A':1}}, inplace=True)
latest_season_tr.replace({'AwayTeamResult' : {'H':1, 'D':0, 'A':0}}, inplace=True)

#checks for any NaN values in the data
# print(latest_season_tr.isna().sum())

ts = trueskill.TrueSkill()
# print(ts)

#creates a dictionary for all of default rating values for each team
ratings = dict()
for i in teams.values.ravel():
    ratings[i] = ts.create_rating()

#stacking arrays of home and away results into one array
home_result = latest_season_tr['HomeTeamResult'].values
away_result = latest_season_tr['AwayTeamResult'].values
ts_results = np.stack((home_result, away_result), axis=-1)

#creates the same stacked array for both teams, the two stacked arrays
#are then stacked into one array with all relevant information about matches
home_teams = latest_season_tr['home_team'].values
away_teams = latest_season_tr['away_team'].values
match_outcome = latest_season_tr['result'].values
matches_array = np.stack((home_teams, away_teams, match_outcome, home_result, away_result), axis=-1)
matches_array

#create empty lists for current and updated rankings
curr_rankings_list = []
new_rankings_list = []
for i in range(len(matches_array)):

    #current ranks of teams in this match
    home_rank = ratings[matches_array[i][0]]
    away_rank = ratings[matches_array[i][1]]

    #the rankings are added into the current ranking list
    curr_rankings_list.append([calc_rating(home_rank.mu,home_rank.sigma), 
                                    calc_rating(away_rank.mu, away_rank.sigma)])

    #new ranks of teams in this match
    new_rankings = ts.rate([(home_rank,),(away_rank,)], ranks= ts_results[i])
    new_home_team_ranking = new_rankings[0][0]
    new_away_team_ranking = new_rankings[1][0]

    #the updated ranks are then added into the updated ranking list
    new_rankings_list.append([calc_rating(new_away_team_ranking.mu, new_home_team_ranking.sigma),
                                calc_rating(new_away_team_ranking.mu, new_away_team_ranking.sigma)])

    #update dict with each teams new rankings
    ratings[matches_array[i][0]] = new_home_team_ranking
    ratings[matches_array[i][1]] = new_away_team_ranking

#dataframes for both ranks before and after are created, they are combined with the matches array
#to create a dataframe
ranks_before_df = pd.DataFrame(curr_rankings_list, columns=['HomeRanksBefore', 'AwayRanksBefore'])
ranks_after_df =  pd.DataFrame(new_rankings_list, columns=['HomeRankAfter', 'AwayRankAfter'])
ranks_together_df = pd.concat([ranks_before_df, ranks_after_df], axis="columns")
matches_df = pd.DataFrame(matches_array, columns=['HomeTeam', 'AwayTeam', 'result', 'HR', 'AR'])
#bring together all of the data
matches_df = pd.concat([matches_df,
                    ranks_together_df],
                    axis="columns")
matches_df.drop(columns=['HR', 'AR'], inplace=True)


#Get the updated TrueSkill Ranks
updated = pd.DataFrame(ratings).transpose()
updated.columns = ['mu', 'sigma']
updated['TrueSkillRating']=calc_rating(updated['mu'], updated['sigma'])
updated_sort = updated.sort_values(by='TrueSkillRating', ascending=False)
updated_sort_teamorder_list = updated_sort.index
updated_sort_ratings_list = updated_sort['TrueSkillRating'].values

#reversed lists for teams and trueskill ratings
reverse_updated_sort = updated.sort_values(by='TrueSkillRating', ascending=True)
reverse_updated_teamorder_list = reverse_updated_sort.index
reverse_updated_ratings_list = reverse_updated_sort['TrueSkillRating'].values


#DEF scatter_plot_analysis() 
#set parameters for the graph for Teams vs Points
actual_standings = pd.read_csv('C:/Users/44745/Documents/Final Year/FYP/input/1718LeagueTable.csv')
as_list = actual_standings['points'].values
as_teams = actual_standings['team'].values

#Lin regression function between points and rating
slope, intercept, r, p, std_err = stats.linregress(as_list, updated_sort_ratings_list)

def linregfunc(as_list):
    return slope * as_list + intercept

def predict_value(x):
    return slope * x + intercept

rating_prediction = predict_value(92)
print(rating_prediction)

# mymodel = list(map(linregfunc, as_list))
# mpl.plot(as_list, mymodel)
# mpl.xlabel("Points")
# mpl.ylabel("TrueSkill Rating")
# mpl.title("Linear Progression Between Points and Rating")
# mpl.scatter(as_list, updated_sort_ratings_list)
# mpl.show
#

#Train and test the resulting data
x_train, x_test, y_train, y_test = train_test_split(as_list, updated_sort_ratings_list, test_size=0.7)

mpl.scatter(x_train, y_train)
# mpl.show()
mpl.scatter(x_test, y_test)
# mpl.show()

print("Linear Regression on Training Data: ")
training_model = list(map(linregfunc, x_train))
mpl.scatter(x_train, y_train)
mpl.plot(x_train, training_model)
mpl.show()

print("Linear Regression on Testing Data")
testing_model = list(map(linregfunc, x_test))
mpl.scatter(x_test, y_test)
mpl.plot(x_test, testing_model)
mpl.show()

r2_train = r2_score(y_train, linregfunc(x_train))
print("Relationship metric for points scored and team rating before train/test methods: ", r)
print("Relationship metric after train/test methods on training set: ", r2_train)

r2_test = r2_score(y_test, linregfunc(x_test))
print("Relationship metric for points scored and team rating before train/test methods: ", r)
print("Relationship metric after train/test methods on test set: ", r2_test)

#FOR TEAMS AGAINST POINTS SCORED
mpl.figure(figsize=(15,5))
mpl.figure(dpi=1200)
mpl.plot(as_teams, as_list)
mpl.scatter(as_teams, as_list)
mpl.title(f"PL Teams against Points")
mpl.xlabel("Team")
mpl.ylabel("Points Scored")
mpl.xticks(as_teams, [str(i) for i in as_teams], rotation=90)
#set parameters for tick labels
mpl.tick_params(axis='x', which='minor')
mpl.tight_layout()

#FOR POINTS SCORED AGAINST TRUESKILL RATING
# mpl.figure(figsize=(15,5))
# mpl.figure(dpi=1200)
# mpl.plot(as_list, updated_sort_ratings_list)
# mpl.scatter(as_list, updated_sort_ratings_list)
# mpl.title(f"PL Points against TrueSkill Rating")
# mpl.xlabel("Team Points")
# mpl.ylabel("Rating")
# mpl.xticks(as_teams, [str(i) for i in as_teams])
# #set parameters for tick labels
# mpl.tick_params(axis='x', which='minor')
# mpl.tight_layout()
#DEF END

#DEF
#DEF END

# def Main():
#     # scatter_rating_()

