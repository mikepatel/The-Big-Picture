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
# read in CSV data file
def get_dataset():
    data_file = "data.csv"
    dataset = pd.read_csv(os.path.join(os.getcwd(), data_file))
    return dataset


# genres
def get_genres(dataframe):
    g = dataframe["Genre"].unique()
    g = sorted(g)
    return g


# count by genre
def get_genre_counts(dataframe):
    counts = {}
    for index, row in dataframe.iterrows():
        try:
            counts[row["Genre"]] += 1

        except KeyError:
            counts[row["Genre"]] = 1

    return counts


def get_genre_score_averages(dataframe, genres):
    averages = pd.Series(dtype="float32")
    for g in genres:
        genre = dataframe.loc[df["Genre"] == g]
        scores = pd.Series(genre["Score"])
        averages[g] = scores.mean()

    return averages


# platforms
def get_platform_counts(platforms):
    counts = {}
    for p in platforms:
        counts[p] = len(df.loc[df["Platform"] == p])

    return counts


# alphanumeric character counts
def get_char_counts(dataframe):
    letters = list(string.ascii_uppercase)
    digits = list(string.digits)
    alphanumeric = letters + digits

    counts = {}
    for c in alphanumeric:
        counts[c] = 0

    for index, row in dataframe.iterrows():
        c = row["Title"][0]
        counts[c] += 1

    return counts


# plot bar chart
def plot_bar(keys, values, title, ylabel, filename, save_dir, feature):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = cm.rainbow(np.linspace(0, 1, len(keys)))
    ax.clear()
    ax.bar(keys, values, color=colours)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    for i in range(len(values)):
        if feature == "genre":
            ax.annotate(f'{values[i]:.04f}', (i-0.25, values[i]+0.005))  # avg score per genre

        if feature == "letter":
            ax.annotate(f'{values[i]}', (i-0.4, values[i] + 0.1))  # first letter
        #ax.annotate(f'{genre_score_averages[i]:.04f}', (i - 0.1, genre_score_averages[i] + 0.001))

    plt.savefig(os.path.join(save_dir, filename))


# plot racing bar chart
def plot_racing_bar(column, keys, save_dir):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = cm.rainbow(np.linspace(0, 1, len(keys)))

    def draw_chart(frame):
        df = get_dataset()
        df = df.head(frame)

        counts = {}
        for k in keys:
            counts[k] = 0

        counts_df = pd.DataFrame(counts.items(), columns=[column, "Count"])

        for index, row in df.iterrows():
            g = row[column]
            counts_df.loc[counts_df[column] == g, "Count"] += 1

        counts_df["Rank"] = counts_df["Count"].rank(method="first")

        # plot
        ax.clear()
        ax.barh(counts_df["Rank"], counts_df["Count"], color=colours)
        ax.set_title("Count by " + column)
        [spine.set_visible(False) for spine in ax.spines.values()]  # remove border around figure
        ax.get_xaxis().set_visible(False)  # hide x-axis
        ax.get_yaxis().set_visible(False)  # hide y-axis

        for index, row in counts_df.iterrows():
            ax.text(x=0, y=row["Rank"], s=str(row[column]), ha="right", va="center")  # base axis
            ax.text(x=row["Count"] + 0.01, y=row["Rank"] + 0.01, s=row["Count"], ha="left", va="center")  # bar

    gif_filepath = os.path.join(save_dir, "racing_bar_" + column + ".gif")
    animator = animation.FuncAnimation(fig, draw_chart, frames=range(len(df)), interval=300)
    animator.save(gif_filepath, writer="imagemagick")


# plot pie chart
def plot_pie(fractions, labels, colours, autopct, filename, save_dir):
    plt.style.use("dark_background")
    plt.figure(figsize=(8, 8))
    plt.pie(fractions, colors=colours, labels=platforms, autopct=fraction_str, pctdistance=1.1, labeldistance=None)
    plt.legend()
    plt.savefig(os.path.join(save_dir, filename))


################################################################################
# Main
if __name__ == "__main__":
    # read in data from file
    df = get_dataset()

    # save plot outputs to a directory
    VISUALS_DIR = os.path.join(os.getcwd(), "visuals")
    if not os.path.exists(VISUALS_DIR):
        os.makedirs(VISUALS_DIR)

    # number of titles
    num_titles = len(df)
    #print(f'Number of titles: {num_titles}')

    # ----- GENRES ----- #
    genres = get_genres(df)
    #print(f'Genres: {genres}')

    # counts
    genre_counts = get_genre_counts(df)
    genre_counts = pd.Series(genre_counts)
    print(f'Genre counts: {genre_counts}')

    # genre racing bar chart
    #plot_racing_bar(column="Genre", keys=genres, save_dir=VISUALS_DIR)

    # ----- COMPLETION PERCENTAGE ----- #
    num_finished_yes = len(df.loc[df["Finished"] == "Yes"])
    completion_percentage = num_finished_yes / num_titles
    print(f'Completion percentage: {completion_percentage}')

    # ----- SCORE ----- #
    genre_score_averages = get_genre_score_averages(df, genres)
    #print(f'Avg Score by Genre: {genre_score_averages}')
    #"""
    plot_bar(
        keys=genres,
        values=genre_score_averages,
        title="Average score per genre",
        ylabel="Average user score",
        filename="bar_avg_score",
        save_dir=VISUALS_DIR,
        feature="genre"
    )
    #"""

    # ----- STREAMING SERVICE ----- #
    platforms = [
        "Hulu",
        "Netflix",
        "Peacock",
        "Theatre",
        "Other"
    ]
    platform_counts = get_platform_counts(platforms)

    # racing bar chart
    #print(f'Platform counts: {platform_counts}')
    """
    plot_racing_bar(
        column="Platform",
        keys=platforms,
        save_dir=VISUALS_DIR
    )
    """

    # pie chart
    # fractions
    fractions = {}
    for p in platform_counts:
        fractions[p] = platform_counts[p] / num_titles

    fractions = [v for k, v in fractions.items()]

    def fraction_str(value):
        value = f'{value:.1f}'
        return value

    colours = ["green", "red", "purple", "blue", "orange"]

    """
    plot_pie(
        fractions=fractions,
        labels=platforms,
        colours=colours,
        autopct=fraction_str,
        filename="pie_platform",
        save_dir=VISUALS_DIR
    )
    """

    # ----- FIRST LETTER ----- #
    char_counts = get_char_counts(dataframe=df)
    #print(f'Character counts: {char_counts}')
    #"""
    plot_bar(
        keys=char_counts.keys(),
        values=list(char_counts.values()),
        title="Count by 'first character'",
        ylabel="",
        filename="bar_firstChar",
        save_dir=VISUALS_DIR,
        feature="letter"
    )
    #"""
    quit()
