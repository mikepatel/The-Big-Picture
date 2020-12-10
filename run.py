"""
Michael Patel
December 2020

Project description:
    Perform data analysis and visualization of viewing habits

"""
################################################################################
# Imports
import os
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.animation as animation


################################################################################
# constants
DATA_CSV_FILEPATH = os.path.join(os.getcwd(), "data.csv")


################################################################################
# Main
if __name__ == "__main__":
    df = pd.read_csv(DATA_CSV_FILEPATH)

    data = {}

    # ----- NUMBER OF MOVIES ----- #
    num_movies = len(df)
    #print(f'Number of movies: {num_movies}')

    # ----- GENRES ----- #
    genres = df["Genre"].unique()
    genres = sorted(genres)  # alphabetize
    #print(genres)
    genre_counts = {}

    # initialize
    for g in genres:
        genre_counts[g] = 0

    for index, row in df.iterrows():
        genre_counts[row["Genre"]] += 1

    #genre_counts = df["Genre"].value_counts()
    genre_counts = pd.Series(genre_counts)
    #print(genre_counts)

    # plot number per genre
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 10))
    colours = cm.rainbow(np.linspace(0, 1, len(genres)))
    plt.bar(genres, genre_counts[genres], color=colours)
    plt.show()
    """

    # racing bar
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 10))

    #print(counts)

    def draw_chart(frame):
        df = pd.read_csv(DATA_CSV_FILEPATH)
        df = df.head(frame)

        counts = {}
        for g in genres:
            counts[g] = 0

        counts_df = pd.DataFrame(counts.items(), columns=["Genre", "Count"])

        for index, row in df.iterrows():
            g = row["Genre"]
            #print(g)
            counts_df.loc[counts_df["Genre"] == g, "Count"] += 1

        counts_df["Rank"] = counts_df["Count"].rank(method="first")
        ax.clear()
        ax.bar(counts_df["Rank"], counts_df["Count"])

    gif_filepath = os.path.join(os.getcwd(), "genre_race.gif")
    animator = animation.FuncAnimation(fig, draw_chart, frames=range(len(df)), interval=800)
    animator.save(gif_filepath, writer="imagemagick")

    quit()

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

    # plot

    # racing bar

    quit()
