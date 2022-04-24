from flask import Flask, request, jsonify, render_template,send_from_directory
from ray import method
import json
import Music_recommendation_colab_Without_Year
import Music_recommendation_colab

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/getTableInput")
def getTableInput():
    return render_template('tableInput.html')

@app.route('/dataGetting', methods = ['GET', 'POST'])
def home():
    if(request.method == 'GET'):
        data = "hello world"
        return jsonify({'data': data})
    elif(request.method=='POST'):
        withoutYear=json.loads(request.form['withoutYear'])
        withYear = json.loads(request.form['withYear'])
        print(withYear)
        print(withoutYear)
        #print(type(withoutYear))
        #print(type(withYear))
        dataWithoutYear = []
        dataWithYear = []
        if len(withoutYear)>0:
            dataWithoutYear = Music_recommendation_colab_Without_Year.recommend_songs(withoutYear)
        if len(withYear)>0:
            dataWithYear = Music_recommendation_colab.recommend_songs(withYear)
        for i in range(0,len(dataWithoutYear)):
            dataWithoutYear[i] = {
                'name':dataWithoutYear[i]['name'],
                'artists': dataWithoutYear[i]['artists'].replace('[','').replace(']','').replace('"',"").replace("'","").split(','),
                'year':'NA'
            }     
        for i in range(0,len(dataWithYear)):
            dataWithYear[i] = {
                'name':dataWithYear[i]['name'],
                'artists': dataWithYear[i]['artists'].replace('[','').replace(']','').replace('"',"").replace("'","").split(','),
                'year':dataWithYear[i]['year']
            } 
        return jsonify({'withYear':dataWithYear,'withoutYear':dataWithoutYear})

if __name__ == "__main__":
    app.run(debug=False)







'''dataWithYear = [{'name': 'Leave Them All Behind - 2001 Remaster',
    'year': 1992,
    'artists': "['Ride']"},
    {'name': 'Hummer - Remastered',
    'year': 1993,
    'artists': "['The Smashing Pumpkins']"},
    {'name': 'Fury Of The Storm', 'year': 2004, 'artists': "['DragonForce']"},
    {'name': 'Spanish Air', 'year': 1991, 'artists': "['Slowdive']"},
    {'name': "'Cross the Breeze (Album Version)",
    'year': 1988,
    'artists': "['Sonic Youth']"},
    {'name': 'Seer', 'year': 2006, 'artists': "['Witch']"},
    {'name': 'Holiday on the Moon',
    'year': 2002,
    'artists': "['Love and Rockets']"},
    {'name': "'Cross the Breeze (Album Version)",
    'year': 1988,
    'artists': "['Sonic Youth']"},
    {'name': 'Hocus Pocus - Extended Version',
    'year': 1971,
    'artists': "['Focus']"},
    {'name': 'Hocus Pocus', 'year': 2001, 'artists': "['Focus']"}]'''


'''  dataWithoutYear = [{'name': 'Seize the Day', 'artists': "['Avenged Sevenfold']"},
 {'name': 'Thunderstruck', 'artists': "['AC/DC']"},
 {'name': "Knockin' On Heaven's Door", 'artists': '["Guns N\' Roses","Jap Purohit"]'},
 {'name': 'Ode to Sleep', 'artists': "['Twenty One Pilots']"},
 {'name': 'Zombie (Live from the NIVA Save Our Stages Festival)',
  'artists': "['Miley Cyrus']"},
 {'name': 'Crazy', 'artists': "['Aerosmith']"},
 {'name': 'This Is War', 'artists': "['Thirty Seconds To Mars']"},
 {'name': 'Beast and the Harlot', 'artists': "['Avenged Sevenfold']"},
 {'name': 'Welcome To The Jungle', 'artists': '["Guns N\' Roses"]'},
 {'name': 'Fire Woman', 'artists': "['The Cult']"}]'''
