# merhaba

# Importing Libraries
import pandas as pd
pd.set_option('display.max_columns', 20)

# Data Preprocessing
# Dataset : https://www.kaggle.com/grouplens/movielens-20m-dataset
movie = pd.read_csv('data/movie_lens_dataset/movie.csv')
rating = pd.read_csv('data/movie_lens_dataset/rating.csv')

# We merge two dataframes with "movieId" Unique variable.
df = movie.merge(rating, how="left", on="movieId")
comment_counts = pd.DataFrame(df["title"].value_counts())

# We leave out movies with very few reviews (less than 1000). We define it as rare_movies.
rare_movies = comment_counts[comment_counts["title"] <= 1000].index

# We assign the ones that are not included in rare_movies as common_movies.
common_movies = df[~df["title"].isin(rare_movies)]
user_movie_df = common_movies.groupby(['userId', "title"])['rating'].sum().unstack()
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

# Determining the Movies Watched by the User 9990

user = 9990  # Defined user

user_df = user_movie_df[user_movie_df.index == user]
movies_watched = user_df.columns[user_df.notna().any()].tolist()
print(len(movies_watched))  # Defined user watched 62 different movies
#['American Movie (1999)',
# 'Badlands (1973)',
# 'Blue Velvet (1986)',
# 'Breaking the Waves (1996)',
# 'Chungking Express (Chung Hing sam lam) (1994)',
# 'Dancer in the Dark (2000)', ...

# Accessing Data and Ids of other users watching the same movies with our defined user 9990.

movies_watched_df = user_movie_df[movies_watched]
print(movies_watched_df.head())
print(movies_watched_df.shape)  # 138,493 people watched at least one movie like our defined user

user_movie_count = movies_watched_df.T.notnull().sum()  # Sum of True quantities
user_movie_count = user_movie_count.reset_index()  # How many movies watched in common with the defined user, based on userId
user_movie_count.columns = ["userId", "movie_count"]

# We got the data of other users who watched all the 60% same movies as the defined User.
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

# Identify the users who are most similar to the user to be suggested.
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      user_df[movies_watched]])
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['userId_1', 'userId_2']
corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)
# Not set a correlation constraint.

# We do not have information about which movie they rate how many points. Let's bring it.
# We will bring it from the ratings table and combine it.
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"userId_2": "userId"}, inplace=True)

# The Id's of the defined user and similar users are below.
# print(f"{[i for i in top_users['userId'][1:]]}")

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != user]

# Calculating Weighted Average Recommendation Score and Keeping Top 5 Movies

# We determine a weighted score as we cannot make suggestions based on the number of movies watched or likes.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

recommendation_df[recommendation_df["weighted_rating"] > 4]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 4].sort_values("weighted_rating",
                                                                                                 ascending=False)[0:5]

movies_to_be_recommend.merge(movie[["movieId", "title"]])
# movieId	weighted_rating	    title
# 5992	4.342995	Hours, The (2002)
# 38499	4.342995	Angels in America (2003)
# 555	4.342995	True Romance (1993)
# 7460	4.342995	Coffee and Cigarettes (2003)
# 26163	4.342995	Don't Look Back (1967)