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


# racing bar
def plot_racing_bar(column, keys, save_dir):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 10))
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
    print(f'Number of titles: {num_titles}')

    # ----- GENRES ----- #
    genres = get_genres(df)
    print(f'Genres: {genres}')

    # counts
    genre_counts = get_genre_counts(df)
    genre_counts = pd.Series(genre_counts)
    print(f'Genre counts: {genre_counts}')

    # genre racing bar chart
    plot_racing_bar(column="Genre", keys=genres, save_dir=VISUALS_DIR)

    quit()

    plot_bar()  # feed in data source, saves plot figure
    plot_racing_bar()
    plot_pie


    # plot number per genre
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 10))
    colours = cm.rainbow(np.linspace(0, 1, len(genres)))
    plt.bar(genres, genre_counts[genres], color=colours)
    plt.show()
    """

    """
    # racing bar
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 10))
    colours = plt.cm.Dark2(range(len(genres)))
    colours = cm.rainbow(np.linspace(0, 1, len(genres)))

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
            counts_df.loc[counts_df["Genre"] == g, "Count"] += 1

        #counts_df["Rank"] = counts_df["Count"].rank(method="first", ascending=True).values
        counts_df["Rank"] = counts_df["Count"].rank(method="first")
        #counts_df = counts_df.sort_values(by="Rank", ascending=False)
        #counts_df = counts_df.reset_index(drop=True)

        # plot
        ax.clear()
        ax.barh(counts_df["Rank"], counts_df["Count"], color=colours)
        ax.set_title("Count by Genre")
        [spine.set_visible(False) for spine in ax.spines.values()]  # remove border around figure
        ax.get_xaxis().set_visible(False)  # hide x-axis
        ax.get_yaxis().set_visible(False)  # hide y-axis

        for index, row in counts_df.iterrows():
            ax.text(x=0, y=row["Rank"], s=str(row["Genre"]), ha="right", va="center")  # base axis
            ax.text(x=row["Count"]+0.01, y=row["Rank"]+0.01, s=row["Count"], ha="left", va="center")  # bar

    gif_filepath = os.path.join(os.getcwd(), "racing_bar_genre.gif")
    animator = animation.FuncAnimation(fig, draw_chart, frames=range(len(df)), interval=300)
    animator.save(gif_filepath, writer="imagemagick")
    """

    # ----- COMPLETION PERCENTAGE ----- #
    """
    num_finished_yes = len(df.loc[df["Finished"] == "Yes"])
    completion_percentage = num_finished_yes / num_movies
    #print(completion_percentage)
    """

    # ----- SCORE ----- #
    """
    # average score per genre
    genre_score_averages = pd.Series(dtype="float32")
    for g in genres:
        x = df.loc[df["Genre"] == g]
        z = pd.Series(x["Score"])
        #print(f'{g}: {z.mean()}')
        genre_score_averages[g] = z.mean()

    #print(genre_score_averages)

    # plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 10))
    colours = cm.rainbow(np.linspace(0, 1, len(genres)))
    ax.clear()
    ax.bar(genres, genre_score_averages, color=colours)
    ax.set_title("Average score per genre")
    #ax.set_xlabel("Genre")
    ax.set_ylabel("Average user score")

    for i in range(len(genre_score_averages)):
        ax.annotate(f'{genre_score_averages[i]:.04f}', (i-0.1, genre_score_averages[i]+0.001))

    plt.savefig("bar_avg_score")
    """

    # ----- STREAMING SERVICE ----- #
    # number per streaming service
    platforms = [
        "Hulu",
        "Netflix",
        "Peacock",
        "Theatre",
        "Other"
    ]

    platform_counts = {
        "Hulu": 0,
        "Netflix": 0,
        "Peacock": 0,
        "Theatre": 0,
        "Other": 0
    }

    temp_sum = 0
    for p in platforms:
        platform_count = len(df.loc[df["Platform"] == p])
        platform_counts[p] = platform_count
        temp_sum += platform_count

    # Other
    #platform_counts["Other"] = num_movies - temp_sum

    print(platform_counts)

    # racing bar
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 10))
    #colours = cm.rainbow(np.linspace(0, 1, len(genres)))
    colours = ["green", "red", "purple", "blue", "orange"]

    def draw_chart(frame):
        df = pd.read_csv(DATA_CSV_FILEPATH)
        df = df.head(frame)

        platform_counts = {
            "Hulu": 0,
            "Netflix": 0,
            "Peacock": 0,
            "Theatre": 0,
            "Other": 0
        }

        platform_df = pd.DataFrame(platform_counts.items(), columns=["Platform", "Count"])

        for index, row in df.iterrows():
            p = row["Platform"]
            platform_df.loc[platform_df["Platform"] == p, "Count"] += 1

        platform_df["Rank"] = platform_df["Count"].rank(method="first")

        # plot
        ax.clear()
        ax.barh(platform_df["Rank"], platform_df["Count"], color=colours)
        ax.set_title("Count by platform service")

        [spine.set_visible(False) for spine in ax.spines.values()]  # remove border around figure
        ax.get_xaxis().set_visible(False)  # hide x-axis
        ax.get_yaxis().set_visible(False)  # hide y-axis

        for index, row in platform_df.iterrows():
            ax.text(x=0, y=row["Rank"], s=str(row["Platform"]), ha="right", va="center")  # base axis
            ax.text(x=row["Count"] + 0.01, y=row["Rank"] + 0.01, s=row["Count"], ha="left", va="center")  # bar


    gif_filepath = os.path.join(os.getcwd(), "racing_bar_platform.gif")
    animator = animation.FuncAnimation(fig, draw_chart, frames=range(len(df)), interval=300)
    animator.save(gif_filepath, writer="imagemagick")
    """

    # pie chart streaming service
    """
    # fractions
    fractions = {}
    for p in platform_counts:
        fractions[p] = platform_counts[p] / num_movies

    #print(platform_counts)
    print(fractions)
    fractions = [v for k, v in fractions.items()]

    def fraction_str(value):
        value = f'{value:.1f}'
        return value

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 10))
    colours = ["green", "red", "purple", "blue", "orange"]
    plt.pie(fractions, colors=colours, labels=platforms, autopct=fraction_str, pctdistance=1.1, labeldistance=None)
    plt.legend()
    plt.savefig("pie_platform")
    """

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

    # plot static bar
    l_df = pd.Series(letter_counts)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 10))
    colours = cm.rainbow(np.linspace(0, 1, len(letter_counts)))
    ax.clear()
    ax.bar(letter_counts.keys(), letter_counts.values(), color=colours)
    ax.set_title("Count by 'first letter'")

    for i in range(len(l_df)):
        #print(l_df[i])
        ax.annotate(f'{l_df[i]}', (i-0.2, l_df[i]+0.01))

    plt.savefig("bar_letter")

    # racing bar
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 10))
    colours = cm.rainbow(np.linspace(0, 1, len(letter_counts)))

    def draw_chart(frame):
        df = pd.read_csv(DATA_CSV_FILEPATH)
        df = df.head(frame)

        counts = {}
        for c in letter_counts:
            counts[c] = 0

        count_df = pd.DataFrame(counts.items(), columns=["AN", "Count"])

        for index, row in df.iterrows():
            c = row["Title"][0]
            count_df.loc[count_df["AN"] == c, "Count"] += 1

        count_df["Rank"] = count_df["Count"].rank(method="first")
        count_df = count_df.head(min(len(df), 10))

        ax.clear()
        ax.barh(count_df["Rank"], count_df["Count"], color=colours)
        ax.set_title("Count by 'first letter'")

        [spine.set_visible(False) for spine in ax.spines.values()]  # remove border around figure
        ax.get_xaxis().set_visible(False)  # hide x-axis
        ax.get_yaxis().set_visible(False)  # hide y-axis

        for index, row in count_df.iterrows():
            ax.text(x=0, y=row["Rank"], s=str(row["AN"]), ha="right", va="center")  # base axis
            ax.text(x=row["Count"] + 0.01, y=row["Rank"] + 0.01, s=row["Count"], ha="left", va="center")  # bar

    gif_filepath = os.path.join(os.getcwd(), "racing_bar_letter.gif")
    animator = animation.FuncAnimation(fig, draw_chart, frames=range(len(df)), interval=300)
    animator.save(gif_filepath, writer="imagemagick")
    """

    quit()
