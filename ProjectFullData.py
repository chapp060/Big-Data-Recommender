
# coding: utf-8

from pyspark.mllib.recommendation import ALS
from numpy import array
import math
import boto3
import pandas as pd
import sys

#Check system for stringIO
if sys.version_info[0] < 3: 
    from StringIO import StringIO # Python 2.x
else:
    from io import StringIO # Python 3.x

#Load the rating data
songs_rating = sc.textFile('s3://awsinitial/BigDataProject/train_triplets.txt')
songs_rating=songs_rating.map(lambda line: line.split("\t")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()


#Load the 
songs_metadata = sc.textFile('s3://awsinitial/BigDataProject/song_data.csv')
songs_metadata_header = songs_metadata.take(1)[0]
songs_metadata = songs_metadata.filter(lambda line: line!=songs_metadata_header)    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2],tokens[3],tokens[4])).cache()

#Replace with Indexed number
songs_rating_test=songs_rating.map(lambda x: (x[0],(x[1],x[2]))).cache()
user_mapping=songs_rating.map(lambda x: x[0]).distinct().zipWithIndex().cache()

#Data Cleaning and processing
songs_rating_1=songs_rating_test.join(user_mapping)
song_rating_user_encoded=songs_rating_1.map(lambda x: (x[1][1],x[1][0][0],x[1][0][1])).cache()
song_rating_user_encoded.take(1)

song_mapping_1=songs_metadata.map(lambda x: x[0]).distinct().zipWithIndex().cache()
song_mapping=songs_metadata.map(lambda x: (x[0],(x[1],x[3],x[4]))).cache()
songs_mapping_final=song_mapping.join(song_mapping_1).cache()
song_mapping_2=songs_mapping_final.map(lambda x: (x[1][1],x[1][0][0],x[1][0][1],x[1][0][2])).cache()
song_rating_user_encoded=song_rating_user_encoded.map(lambda x: (x[1],(x[0],x[2]))).cache()

songs_rating_test_2=song_rating_user_encoded.map(lambda x: (x[1],(x[0],x[2])))
song_data_final=song_rating_user_encoded.join(song_mapping_1).map(lambda x: (x[1][0][0],x[1][1],x[1][0][1])).cache()
song_data_final.take(1)


#Test Training split
training_RDD, validation_RDD, test_RDD = song_data_final.randomSplit([6, 2, 2], seed=0L)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

#Build model
from pyspark.mllib.recommendation import ALS
import math

seed = 5L
iterations = 10
regularization_parameter = 0.1
alpha = 0.05
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.trainImplicit(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter,alpha=alpha)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    if error < min_error:
        min_error = error
        best_rank = rank

print 'The best model was trained with rank %s' % best_rank


#Retrain with best parameters
model = ALS.trainImplicit(training_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter,alpha=alpha)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print 'For testing data the RMSE is %s' % (error)


from pyspark.mllib.recommendation import MatrixFactorizationModel
model_path = os.path.join('s3://awsinitial/BigDataProject', 'model_fin', 'songs')

# Save and load model
model.save(sc, model_path)
same_model = MatrixFactorizationModel.load(sc, model_path)

#Generate Test Recommendations
songs_rating_counts_RDD = song_data_final.map(lambda x: (x[1], int(x[2]))).reduceByKey(lambda x,y:x+y)
user_id = user_mapping.filter(lambda x: x[0] == 'ff4322e94814d3c7895d07e6f94139b092862611').map(lambda x: x[1]).take(1)[0]

user_id_songs_listened = song_data_final.filter(lambda x: x[0] == user_id)

songs_listened_by_user = user_id_songs_listened.map(lambda x: x[1])
songs_listened_by_user_1=[x for x in songs_listened_by_user.toLocalIterator()]


user_not_listened_songs_RDD = song_mapping_2.filter(lambda x: x[0] not in songs_listened_by_user_1).map(lambda x: (user_id, x[0]))

user_recommendations_RDD = model.predictAll(user_not_listened_songs_RDD)
user_recommendations_song_RDD = user_recommendations_RDD.map(lambda x: (x.product, x.rating))

song_titles = song_mapping_2.map(lambda x: (int(x[0]),(x[1] + ': ' + x[2] + '-' + x[3])))

user_recommendations_rating_title_and_count_RDD =     user_recommendations_song_RDD.join(song_titles).join(songs_rating_counts_RDD)
user_recommendations_rating_title_and_count_RDD =     user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

#Generate top songs
top_songs = user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=50).distinct().takeOrdered(5, key=lambda x: -x[1])

print ('TOP recommended songs (with more than 50 listen count):\n%s' %
        '\n'.join(map(str, top_songs)))


#Load libraries for Buttons
import difflib
import random
import requests
import ipywidgets as widgets
from ipywidgets import Button, Layout
from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider, Text
from IPython.display import clear_output, display
import time
import datetime
import pandas as pd

#Load SongProfile Data
s3 = boto3.client('s3')
obj = s3.get_object(Bucket='awsinitial', Key='BigDataProject/songProfiles.csv')
body = obj['Body']
csv_string = body.read().decode('utf-8')
user_list = pd.read_csv(StringIO(csv_string))
user_list['songs'] = user_list['songs'].str.split('|')
# Button for generating the recommendations

#Generate Recommendations button
def on_rec_clicked(rec):
    clear_output()
    print('Loading Recommendations: ............')
    st_time = datetime.datetime.now()
    user=user_list['user_id'][tab.selected_index]
    user_id = user_mapping.filter(lambda x: x[0] == user ).map(lambda x: x[1]).take(1)[0]
    user_id_songs_listened = song_data_final.filter(lambda x: x[0] == user_id)
    songs_listened_by_user = user_id_songs_listened.map(lambda x: x[1])
    songs_listened_by_user_1=[x for x in songs_listened_by_user.toLocalIterator()]
    user_not_listened_songs_RDD = song_mapping_2.filter(lambda x: x[0] not in songs_listened_by_user_1)    .map(lambda x: (user_id, x[0]))
    
    user_recommendations_RDD = model.predictAll(user_not_listened_songs_RDD)
    user_recommendations_song_RDD = user_recommendations_RDD.map(lambda x: (x.product, x.rating))
    song_titles = song_mapping_2.map(lambda x: (int(x[0]),x[1] + ': ' + x[3]))
    top_songs =         user_recommendations_song_RDD.join(song_titles).join(songs_rating_counts_RDD)        .map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))        .filter(lambda r: r[2]>=50).distinct().takeOrdered(10, key=lambda x: -x[1])
    print('Recommendation took {} '.format(datetime.datetime.now() - st_time))
    display(pd.DataFrame({'songs':[row[0] for row in top_songs]}))
    #results = is_model.get_user_items(child.description)
    #clear_output()
    #display(results[['rank','song']])

rec = Button(description='Generate  Recommendations',
       layout=Layout(width='100%', height='80px'), button_style = 'info')
#Event capture
rec.on_click(on_rec_clicked)
    
#Clear Button
clear = Button(description='Clear',
           layout=Layout(width='100%', height='80px'),button_style = 'primary')

#Clear button action
def on_clear(clear):
    clear_output()
    return(tab)
#Clear button event
clear.on_click(on_clear)

#Generate full tab list/UI
tab_list = user_list['user_name']
children = []
tab = widgets.Tab()
for i in range(len(tab_list)):
    #tab.children = (dropdown,button_form)
    tab.set_title(i, str(tab_list[i]))
    songs = user_list['songs'][i]
    user_song_labels= [widgets.Label(str(song)) for song in songs]
    child = widgets.VBox(user_song_labels,description = user_list['user_id'][i],
            layout=Layout(
               display='flex',
             flex_flow='column',
               border='solid 2px',
               width='75%'))

    button_form = Box(children=(rec,clear),
                    layout=Layout(display='flex',
                     flex_flow='column',
                    border='solid 2px',
                    fill='grey',          
           align_items='center',
           width='50%'))
    full_form = widgets.HBox([child,button_form])
    children.append(full_form)
    
#Display Tabs
tab.children = tuple(children)
tab
