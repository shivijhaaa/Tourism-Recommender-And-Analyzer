from flask import Flask , render_template,request
import streamlit as st
import pickle
from Trial import recommend as rec
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

new_tour = pickle.load(open("new_tour.pkl","rb"))
similarity = pickle.load(open("similarity.pkl","rb"))
tour = pickle.load(open("tour.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route("/recommend_tour")
def recommend():
    user_input = request.args.get('user_input')
    result = rec(user_input)
    return render_template("show.html", recommendations=result)

@app.route("/streamlit")
def home():
    df1 = pd.read_csv("1. nationality wise fourign tourist.csv")
    arrival_f = df1[['Country ', '2017', '2018', '2019', '2020', '2021']]
    sorted_df = arrival_f.sort_values('2021', ascending=False)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(800, 300), gridspec_kw={'height_ratios': [1, 1, 1]})

    top_10 = sorted_df.head(10)
    top_10.set_index('Country ', inplace=True)
    top_10.plot(kind='bar', ax=ax1, figsize=(10, 6))
    ax1.set_xlabel('Country')
    ax1.set_ylabel('Number of Nationals')
    ax1.set_title('NATIONALITY-WISE FOREIGN TOURIST ARRIVALS IN INDIA, 2017-2021')
    #ax1.set_xticks(rotation='vertical')

    df1 = pd.read_csv("2fta mode of transfer.csv")
    df1.dropna(inplace=True)
    df1.isnull().sum()
    labels = ['Land', 'Air', 'Sea']
    sizes = [df1['Land'].sum(), df1['Air'].sum(), df1['Sea'].sum()]
    explode = (0.1, 0.1, 0.1)
    colors = ['#90EE90', '#ADD8E6', '#00008B']
    ax2.pie(sizes, labels=labels, autopct='%1.2f%%', explode=explode, shadow=True, startangle=30, colors=colors)
    ax2.set_title('DISTRIBUTION OF NATIONALITY-WISE FTAs IN INDIA BY MODE OF TRAVEL, 2021')

    df2 = pd.read_csv("3..Nationality wise FTA according to age group 2021.csv")
    df2.set_index('Year', inplace=True)
    labels = ['0-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65 and Above', 'Not Reported']
    explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c2c2f0', '#ffff99']
    last_row = df2.iloc[-1]
    labels = last_row.index[2:]
    values = last_row.values[2:]
    if len(explode) != len(values):
        explode = (0,) * len(values) 
    ax3.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode, shadow=True, colors=colors)
    ax3.set_title("Distribution of Age Groups in 2021")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return render_template('analyzer.html', image_base64=image_base64)
if __name__ == '__main__' :
    app.run(debug = True)




