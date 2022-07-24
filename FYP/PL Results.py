import sys
import trueskill
import trueskill as ts
import matplotlib.pyplot as mpl
import pandas as pd
import numpy as np
import pprint
import random
from random import randrange
from trueskill import Rating, quality_1vs1, rate_1vs1

match_results = pd.read_csv('C:/Users/44745/Documents/Final Year/FYP/input/results.csv')
match_results.head(10).append(match_results.tail(10))

latest_season = pd.DataFrame(match_results.loc[match_results['season'] == '2017-2018'])
latest_season_tr = latest_season[['home_team', 'away_team', 'result']].copy()
# print("1: \n", latest_season_tr.head(10).append(latest_season_tr.tail(10)))

teams = pd.DataFrame(latest_season.iloc[:,0].unique(), columns=['Team'])
# print("\n Number of participating teams: ", teams.shape[0], "\n")

latest_season_tr['HomeTeamResult'] = latest_season_tr['result']
latest_season_tr['AwayTeamResult'] = latest_season_tr['result']

#Add binary results for both home and away teams, where Win and Draw = 0 & Loss = 1
latest_season_tr.replace({'HomeTeamResult' : {'H':0, 'D':0, 'A':1}}, inplace=True)
latest_season_tr.replace({'AwayTeamResult' : {'H':1, 'D':0, 'A':0}}, inplace=True)
print("\n ", latest_season_tr.head())

#checks for any NaN values in the data
# print(latest_season_tr.isna().sum())

ts = trueskill.TrueSkill()
# print(ts)

ratings = dict()
for i in teams.values.ravel():
    ratings[i] = ts.create_rating()
print("\n Length of default ratings dict: ", len(ratings.keys()))

home_result = latest_season_tr['HomeTeamResult'].values
away_result = latest_season_tr['AwayTeamResult'].values
ts_results = np.stack((home_result, away_result), axis=-1)

home_teams = latest_season_tr['home_team'].values
away_teams = latest_season_tr['away_team'].values
match_outcome = latest_season_tr['result'].values
matches_array = np.stack((home_teams, away_teams, match_outcome, home_result, away_result), axis=-1)

def calc_rating(mu, sigma):
    return mu-2*sigma

curr_rankings_list = []
new_rankings_list = []
for i in range(len(matches_array)):

    #current ranks of teams
    home_rank = ratings[matches_array[i][0]]
    away_rank = ratings[matches_array[i][1]]

    curr_rankings_list.append([calc_rating(home_rank.mu,home_rank.sigma), 
                                    calc_rating(away_rank.mu, away_rank.sigma)])

    #new ranks of teams
    new_rankings = ts.rate([(home_rank,),(away_rank,)], ranks= ts_results[i])

    new_home_team_ranking = new_rankings[0][0]
    new_away_team_ranking = new_rankings[1][0]

    new_rankings_list.append([calc_rating(new_away_team_ranking.mu, new_home_team_ranking.sigma),
                                calc_rating(new_away_team_ranking.mu, new_away_team_ranking.sigma)])

    #update dict with each teams new rankings
    ratings[matches_array[i][0]] = new_home_team_ranking
    ratings[matches_array[i][1]] = new_away_team_ranking


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

teams
test1 = teams.head(10)
test2 = updated.head(10)
teams_list = test1['Team'].values
rating_list = test2['TrueSkillRating'].values
print(teams_list)
updated
rating_list



fig1 = mpl.scatter(teams_list, rating_list)
mpl.
mpl.show(fig1)
# print(calc_rating(home_rank.mu,home_rank.sigma))
# print(calc_rating(away_rank.mu,home_rank.sigma))
# print(ts_ranks)
# print(ratings)

