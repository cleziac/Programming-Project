# IMPORTS 
from importlib.metadata import metadata
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from PIL import Image

# Setting up Streamlit

st.set_page_config(
    page_title="Movies EDA", 
    page_icon="🎬")

@st.cache(allow_output_mutation=True)
def get_data(url):
    movies_df = pd.read_csv(url_movies_dataset)
    return movies_df

@st.cache
def get_downloadable_data(df):
    return df.to_csv().encode('utf-8')

url_movies_dataset = 'https://raw.githubusercontent.com/cleziac/Programming-Project/main/movies_metadata.csv?token=GHSAT0AAAAAABRE66MWFWQ2PQ3R2DKTKSQ4YQEDYFA'

movies_static = get_data(url_movies_dataset)
movies_df = movies_static.copy()

st.header('📺🍿🎥 Movies EDA and Modeling 🎥🍿📺')
st.write("Exploration Data Analysis 📈")
st.write('Dataset source: [click link](' + url_movies_dataset + ')')
st.download_button('DOWNLOAD RAW DATA', get_downloadable_data(movies_static), file_name='movies_metadata.csv')

st.sidebar.subheader('Controls')
show_raw_data = st.sidebar.checkbox('Show raw data')

if show_raw_data:
    st.subheader('Raw data')
    st.write(movies_df)

st.sidebar.code('''
@st.cache(allow_output_mutation=True)
def get_data(url):
    movies_df = pd.read_csv(url)
    return movies_df
'''
)

st.sidebar.download_button('DOWNLOAD', get_downloadable_data(movies_df), file_name='movies.csv')

# Data Exploration

st.write('How many rows and columns in the dataset?')
st.write(movies_df.shape[0], movies_df.shape[1])

st.write('What types of data does the dataset contain?')
d_types = movies_df.dtypes.astype(str)
st.write(d_types)

st.write('Let''s display the details of the dataset:')
st.write(movies_df.describe())

st.write('How many NaN values does each column have?')
st.write(movies_df.isnull().sum())

st.write('Deletion of columns containing too many NaN values or that are not needed to perform EDA')
movies_df = movies_df.drop(['belongs_to_collection', 'homepage', 'tagline', 'poster_path',
                'status', 'video', 'overview', 'popularity'], axis = 1)
st.write(movies_df.columns)


# Let's clean each column separately to perform EDA

# Budget column

pd.to_numeric(movies_df.budget)
budget = movies_df.budget[movies_df.budget != 0]

# Genres column

genres = movies_df.genres
genres_col = movies_df[['id', 'genres']]
genlist = []

for i in range(genres_col.shape[0]):
    gen = eval(genres_col['genres'][i])
    for each in gen:
        each['id'] = genres_col['id'][i]
    genlist.extend(gen)
genre = pd.DataFrame(genlist)

# Create a DF that contains genres and respective count
how_many = genre.groupby('name').count()
names = genre.name.unique()
names = np.sort(names)

genres_df = pd.DataFrame({
    'genres' : names,
    'quantities' : how_many.id
})

# Original language column

lang = pd.DataFrame(data = movies_df.original_language)
lang = lang.dropna()
lang['count'] = lang.value_counts().sum()
lang1 = lang.groupby('original_language').count()
idx = pd.Index.to_list(lang1.index)
content = lang1['count'].to_list()
data = {'languages' : idx, 'counts' : content}
lang_df = pd.DataFrame(data = data, index = np.arange(0, 89))

lang_df_mask = lang_df[lang_df['counts'] >= 100]

# Revenue column

revenue = movies_df.revenue.dropna()
drev = pd.DataFrame(data = revenue)
drev = drev[drev.revenue != 0.0]

# Merge together genre, revenue and budget:

genre_revenue = pd.merge(genre, revenue, left_index= True, right_index= True)
genre_revenue_budget = pd.merge(genre_revenue, budget, left_index=True, right_index=True)
genre_revenue_budget = genre_revenue_budget[genre_revenue_budget.budget != 0.0]
genre_revenue_budget = genre_revenue_budget[genre_revenue_budget.revenue != 0.0]

# Release dates column

rd = pd.DataFrame(data = movies_df.release_date)
rd = rd.dropna()
rd[['year', 'month', 'day']] = rd.release_date.str.split('-', expand = True)
movies_per_year = rd.groupby('year').count()


data_cleaning = st.checkbox('Display data cleaning process')

if data_cleaning:
    # Let's clean each column separately to perform EDA

    # Budget column

    st.write('Budget column:')
    st.write('Data type: ', movies_df.budget.dtype)
    st.write('Minimum value: ', movies_df.budget.min())
    st.write('Maximum value: ', movies_df.budget.max())
    st.write('Average value: ', movies_df.budget.mean())

    st.write('Convert the type to numeric and delete all the rows with value 0:')
    pd.to_numeric(movies_df.budget)
    budget = movies_df.budget[movies_df.budget != 0]
    st.write(budget.describe())

    # Genres column

    st.write('Genres column:')
    genres = movies_df.genres
    st.write(genres.head())
    st.write('Given the fact that the data is stored in nested dictionaries, they need to split it.')
    genres_col = movies_df[['id', 'genres']]
    genlist = []

    for i in range(genres_col.shape[0]):
        gen = eval(genres_col['genres'][i])
        for each in gen:
            each['id'] = genres_col['id'][i]
        genlist.extend(gen)
    genre = pd.DataFrame(genlist)

    st.write(genre.head())

    # Original language column

    st.write('How many original languages are found in the dataset?')
    st.write(movies_df.original_language.unique())

    st.write('Creation of a new dataset which has a column for the language and a column for its respective count in the dataset.')
    lang = pd.DataFrame(data = movies_df.original_language)
    lang = lang.dropna()
    lang['count'] = lang.value_counts().sum()
    lang1 = lang.groupby('original_language').count()
    idx = pd.Index.to_list(lang1.index)
    content = lang1['count'].to_list()
    data = {'languages' : idx, 'counts' : content}
    lang_df = pd.DataFrame(data = data, index = np.arange(0, 89))
    st.write(lang_df)

    # Revenue column

    revenue = movies_df.revenue.dropna()
    drev = pd.DataFrame(data = revenue)
    st.write('Revenue column:')
    st.write(drev.head())
    st.write('Many rows have value 0.0, so they will be removed and interpreted as NaN.')
    drev = drev[drev.revenue != 0.0]
    st.write(drev.head())
    st.write('With the rows removed, the DataFrame has ', drev.shape[0], 'rows.')

    # Release rates column

    st.write('Release date column:')
    rd = pd.DataFrame(data = movies_df.release_date)
    rd = rd.dropna()
    st.write(rd.head())
    st.write('Let''s split the date into three different columns and group them by year:')
    rd[['year', 'month', 'day']] = rd.release_date.str.split('-', expand = True)
    movies_per_year = rd.groupby('year').count()
    st.write(movies_per_year.head())

# Graphs 

st.header('Graphs display')

# Graph 1

st.subheader('Distribution rate of movies every year')
fig, ax = plt.subplots(figsize=(10,6))
ax.xaxis.set_major_locator(plt.MaxNLocator(45))
plt.xticks(rotation = 45)
plt.plot(movies_per_year, color = 'red')
st.write(fig)

# Graph 2

st.subheader('Genre popularity')
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x = genres_df.quantities, y = genres_df.genres, label = genres_df.genres)
st.write(fig)

# Graph 3

st.subheader('Languages popularity (Only languages that are spoken in more than 100 movies are displayed)')
fig, ax = plt.subplots(figsize=(10,6))
hist(lang_df_mask.languages, weights=lang_df_mask.counts,
    orientation= 'vertical', label= lang_df_mask.languages, bins = 23,
    color = 'indianred')
st.write(fig)

# Graph 4

st.subheader('Is there a correlation between budget and revenue?')
fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(x = genre_revenue_budget.budget, y = genre_revenue_budget.revenue)
st.write(fig)

# Graph 5

st.subheader('Is there a correlation between budget and genre?')
fig, ax = plt.subplots(figsize=(10,6))
sns.boxplot(y = genre_revenue_budget['budget'], x = genre_revenue_budget['name'])
plt.xticks(rotation = 45)
st.write(fig)

# MODELING 

st.header('Modeling')

st.write('In order to perform predictions on the data, the ratings column will be added to the dataset.')
ratings = pd.read_csv('https://raw.githubusercontent.com/cleziac/Programming-Project/main/ratings_small.csv?token=GHSAT0AAAAAABRE66MXNW672ROOC4VETPLOYQED3VQ')
st.write('Ratings dataset sample')
st.write(ratings.head())
ratings = ratings.dropna()
st.write('In order to consider the genres separately, they will be pivoted and merged into the movies dataset.')

genre = genre.rename(columns={'name' : 'genre'})
genre = genre[['id', 'genre']]
genre['tmp'] = 1
pivot = genre.pivot_table('tmp', 'id', 'genre', fill_value=0)
genre_pivoted = pd.DataFrame(pivot.to_records())

st.write(genre_pivoted.head())

st.write('Adding the separate genres to the movies dataframe yields:')
metadata = pd.merge(movies_df, genre_pivoted, on = 'id', how='left')
st.write(metadata.head())

st.write('Removing all the columns that contain non numerical values yields:')
metadata = metadata.drop(['id', 'title', 'original_language', 'genres', 'imdb_id', 'production_companies', 'adult',
                          'production_countries', 'release_date', 'spoken_languages', 'original_title'], axis = 1)
st.write(metadata.head())

st.subheader('Correlation among all data')
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(metadata.corr())
st.write(fig)

st.write('Given the fact that the genres have no correlation with the data, the columns will be removed.')
metadata_no_genres = metadata.drop(['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama',
                                    'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 'Music', 'Mystery', 'Romance',
                                    'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'], axis = 1)
st.write(metadata_no_genres.head())

st.write('To keep consistency in the revenue and budget columns, all 0s will be replaced with random values similar to the others.')
metadata_no_genres.budget = metadata_no_genres.budget.replace(0, np.random.randint(1000000, 380000000))
metadata_no_genres.revenue = metadata_no_genres.revenue.replace(0, np.random.randint(29321695, 380000000))
model_df = pd.merge(metadata_no_genres, ratings.rating, left_index = True, right_index = True)
st.write(model_df.head())

model_df = model_df.dropna()
st.write('How big is the modeling dataset? ', model_df.shape)

st.subheader('Correlation among the remaining data')
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(model_df.corr(), annot = True)
st.write(fig)

with st.expander('Show model'):

    st.subheader('A model to predict movie ratings')

    y = model_df.rating

    select_model = st.selectbox('Select model:', ['LinearRegression','RandomForestRegressor'])

    model = LinearRegression()

    if select_model == 'RandomForestRegressor':
        model = RandomForestRegressor()

    choices = st.multiselect('Select features', ['budget', 'revenue','vote_count'])

    test_size = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1)

    if len(choices) > 0 and st.button('RUN MODEL'):
        with st.spinner('Training...'):
            x = model_df[choices]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=2)

            x_train = x_train.to_numpy().reshape(-1, len(choices))
            model.fit(x_train, y_train)

            x_test = x_test.to_numpy().reshape(-1, len(choices))
            y_pred = model.predict(x_test)

            st.write('How accurate is the model?')
            st.write('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
            st.write('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
            st.write('Mean Squared Log Error: ', metrics.mean_squared_log_error(y_test, y_pred))


with st.expander('Show model'):

    st.subheader('A model to classify movie ratings (binary classification)')

    ratings['binary'] = [0 if x < 3.5 else 1 for x in ratings.rating]

    y = ratings.binary

    select_model = st.selectbox('Select model:', ['LogisticRegression','KNeighborsClassifier'])

    model = LogisticRegression()

    if select_model == 'KNearestNeighbors':
        model = KNeighborsClassifier()

    choices = st.multiselect('Select features', ['rating'])

    test_size1 = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1, key = '1')

    if len(choices) > 0 and st.button('RUN MODEL', key = '1'):
        with st.spinner('Training...'):
            x = ratings[choices]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size1, random_state=2)

            x_train = x_train.to_numpy().reshape(-1, len(choices))
            model.fit(x_train, y_train)

            x_test = x_test.to_numpy().reshape(-1, len(choices))
            y_pred = model.predict(x_test)

            st.write('How accurate is the model?')
            result = metrics.confusion_matrix(y_test, y_pred)
            st.write(result)
            result1 = metrics.classification_report(y_test, y_pred)
            st.write(result1)
            