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
import string


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

    #print(genre_score_averages)

    # plot

    # ----- STREAMING SERVICE ----- #
    # number per streaming service
    platforms = [
        "Hulu",
        "Netflix",
        "Peacock"
    ]

    platform_counts = {
        "Hulu": 0,
        "Netflix": 0,
        "Peacock": 0
    }

    temp_sum = 0
    for p in platforms:
        platform_count = len(df.loc[df["Platform"] == p])
        platform_counts[p] = platform_count
        temp_sum += platform_count

    # Other
    platform_counts["Other"] = num_movies - temp_sum

    #print(platform_counts)

    # pie chart streaming service

    # ----- FIRST LETTER ----- #
    # number per first letter in title
    letters = list(string.ascii_uppercase)
    digits = list(string.digits)
    alphanumeric = letters + digits

    letter_counts = {}
    # initialize
    for c in alphanumeric:
        letter_counts[c] = 0

    for index, row in df.iterrows():
        letter = row["Title"][0]
        letter_counts[letter] += 1

    print(letter_counts)

    quit()
