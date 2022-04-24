#!/usr/bin/env python
# coding: utf-8

# # Understanding the data

import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")



# pip install spotipy



#  pip install ruamel_yaml==0.11.14


data = pd.read_csv("data_music.csv")
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')



data.info()



data.drop(columns = ['year'], inplace = True)



genre_data.info()



len(genre_data['genres'].unique())



year_data.info()



data.describe()


# # Describing the data set
# 
# 1. Valence - Describing the musical positiveness conveyed by a track.
# 2. Year - The year the song was released in
# 3. Acousticness - Represents whether the track is acoustic
# 4. Danceability - Represents whether the track is dancable
# 5. Duration_ms - The running time of the song
# 6. Energy - Represents a perceptual measure of intensity and activity
# 7. Explicit - Whether the song has words which are considered sexual, offensive or violent in nature.
# 8. Instrumentalness - Tells whether a track contains vocals or not. OOH AAH are considered instrumental.
# 9. Key - The key the track is in. Integers map to pitches using standard Pitch Class notation .
# 10. Liveness - Detects the presence of audio in the song.
# 11. Loudness - Loudness of a track in decibels.
# 12. Mode - Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
# 13. Popularity - Depicts the popularity of the song
# 14. Speechiness - Detects the presence of spoken words in the song. 
# 15. Tempo - Overall tempo of the song in beats per minute.

# # Visualization and preprocessing


from yellowbrick.target import FeatureCorrelation
from sklearn.preprocessing import StandardScaler

scaler_object = StandardScaler()

feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode']



X, y = data[feature_names], data['popularity']
X = scaler_object.fit_transform(X)
                                
features = np.array(feature_names)

visualizer = FeatureCorrelation(labels=features)

#plt.rcParams['figure.figsize']=(8,8)
#visualizer.fit(X,y)     
#visualizer.show()



data.corr()



data.drop(columns = ['popularity']).corr().style.background_gradient(cmap="Blues")



# sns.set(rc={'figure.figsize':(12.7,9.27)})



def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

#print("Top Absolute Correlations")
#print(get_top_abs_correlations(data.drop(columns = ["artists", "id", "name", "release_date"]), 5))


# sns.heatmap(X.corr(), cmap="YlGnBu", annot=True)



# type(X)



# X.drop(columns=['loudness'], inplace = True)


# X.info()



# sns.set(style="ticks", color_codes=True)
# g = sns.pairplot(X, diag_kind = "kde")



# X.describe()





# # Working with the model


#genre_data



genre_data.drop(columns = ['mode']).corr().style.background_gradient(cmap="Blues")



def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

#######print("Top Absolute Correlations")
#######print(get_top_abs_correlations(genre_data.drop(columns = ["genres"]), 5))


genre_data.drop(columns = ["loudness", "acousticness"], inplace = True)



from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
X = genre_data.select_dtypes(np.number) # working only with number datatypes
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)



#######print(genre_data['cluster'])


# Visualizing the dataset using TSME
from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

# projection['title'] = data['name']
# projection['cluster'] = data['cluster_label']


#fig = px.scatter(
#    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
#fig.show()


# Visualizing the dataset using TSME
# from sklearn.manifold import TSNE

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
genre_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

# projection['title'] = data['name']
# projection['cluster'] = data['cluster_label']


#fig = px.scatter(
#    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
#fig.show()


song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=10, 
                                   verbose=False))
                                 ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels


from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

#fig = px.scatter(
#    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
#fig.show()



# tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('TSNE', TSNE(n_components=2, verbose = 1))])
# song_embedding = tsne_pipeline.fit_transform(X)
# projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
# projection['title'] = data['name']
# projection['cluster'] = data['cluster_label']

# fig = px.scatter(
#     projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
# fig.show()


# # Working with spotify

# !pip install spotipy



import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

# sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ["9125ef86e1a342788c6a13104218c6d9"],
#                                                            client_secret=os.environ["3fa046e1a94047fcb289047952a7bdee"]))

cid = '9125ef86e1a342788c6a13104218c6d9'
secret = '3fa046e1a94047fcb289047952a7bdee'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


def find_song(name):
    song_data = defaultdict()
    results = sp.search(q= 'track: {}'.format(name), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)



from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

number_cols = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            #######print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data=data, n_songs=10):
    
    metadata_cols = ['name', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

