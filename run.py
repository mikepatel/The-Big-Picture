"""
Michael Patel
December 2020

Project description:
    Perform data analysis and visualization of viewing habits

"""
################################################################################
# Imports
import os
import pandas as pd


################################################################################
# constants
DATA_CSV_FILEPATH = os.path.join(os.getcwd(), "data.csv")


################################################################################
# Main
if __name__ == "__main__":
    df = pd.read_csv(DATA_CSV_FILEPATH)

    # ----- NUMBER OF MOVIES ----- #
    num_movies = len(df)
    #print(f'Number of movies: {num_movies}')

    # ----- GENRES ----- #
    genres = df["Genre"].unique()
    genre_counts = df["Genre"].value_counts()
    #print(genre_counts)

    # plot number per genre

    # ----- COMPLETION PERCENTAGE ----- #
    num_finished_yes = len(df.loc[df["Finished"] == "Yes"])
    completion_percentage = num_finished_yes / num_movies
    #print(completion_percentage)

    # ----- SCORE ----- #
    # average score per genre
    genre_score_averages = pd.Series()
    for g in genres:
        x = df.loc[df["Genre"] == g]
        z = pd.Series(x["Score"])
        #print(f'{g}: {z.mean()}')
        genre_score_averages[g] = z.mean()

    print(genre_score_averages)

    # ----- STREAMING SERVICE ----- #
    # number per streaming service

    # pie chart streaming service

    # ----- FIRST LETTER ----- #
    # number per first letter in title
    quit()
