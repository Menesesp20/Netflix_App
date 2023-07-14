import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Netflix Dashboard')

netflix = pd.read_csv('Data/Netflix.csv')

st.dataframe(netflix)

netflix['country'] = netflix['country'].fillna(netflix['country'].mode()[0])

netflix['day_added'] = netflix['date_added'].apply(lambda x: x.split(' ')[1].replace(',', '') if isinstance(x, str) else '')

netflix['month_added'] = netflix['date_added'].apply(lambda x: x.split(' ')[0].replace(',', '') if isinstance(x, str) else '')

netflix['year_added'] = netflix['date_added'].apply(lambda x: x.split(' ')[2].replace(',', '') if isinstance(x, str) else '')

netflix['isMovie'] = np.where(netflix['type'] == 'Movie', 'Yes', 'No')

# Drop all rows that contain NaN values
netflix.dropna(inplace=True)

# Drop Duplicates Values
netflix.drop_duplicates(inplace= True)

# Reset the index of the dataset
netflix.reset_index(drop=True, inplace=True)

netflix['year_added'] = netflix['year_added'].astype(int)

st.header('visualizations 📊')

# Helper column for various plots
netflix['count'] = 1

# Many productions have several countries listed - this will skew our results , we'll grab the first one mentioned

# Lets retrieve just the first country
netflix['first_country'] = netflix['country'].apply(lambda x: x.split(",")[0])
netflix['first_country'].head()

# Rating ages from this notebook: https://www.kaggle.com/andreshg/eda-beginner-to-expert-plotly (thank you!)

ratings_ages = {
    'TV-PG': 'Older Kids',
    'TV-MA': 'Adults',
    'TV-Y7-FV': 'Older Kids',
    'TV-Y7': 'Older Kids',
    'TV-14': 'Teens',
    'R': 'Adults',
    'TV-Y': 'Kids',
    'NR': 'Adults',
    'PG-13': 'Teens',
    'TV-G': 'Kids',
    'PG': 'Older Kids',
    'G': 'Kids',
    'UR': 'Adults',
    'NC-17': 'Adults'
}

netflix['target_ages'] = netflix['rating'].replace(ratings_ages)
netflix['target_ages'].unique()

# Genre

netflix['genre'] = netflix['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 

# Reducing name length

netflix['first_country'].replace('United States', 'USA', inplace=True)
netflix['first_country'].replace('United Kingdom', 'UK',inplace=True)
netflix['first_country'].replace('South Korea', 'S. Korea',inplace=True)

visualization_Type = st.selectbox(
    'What visualization you would like to analyze?',
    ['Distribution of content', 'Who creates more content?', 'Content by country', 'How content is added?', 'Target ages by country'])

def distribution_Content():
    count = netflix.groupby(['isMovie']).count()['type']
    labels = ["Tv Show", "Movie"]
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(6,6))
    plt.pie(count,labels=labels,autopct='%.2f %%',textprops={'fontsize': 14, 'color' : 'white'}, colors=['#221f1f', '#b20710'])
    plt.title("Distribution of TV Show & Movie",fontdict={'fontsize': 19})
    plt.legend(fontsize=10)

def countries_Production():
    data = netflix.groupby('first_country')['count'].sum().sort_values(ascending=False)[:10]

    # Plot

    color_map = ['#f5f5f1' for _ in range(10)]
    color_map[0] = color_map[1] = color_map[2] =  '#b20710' # color highlight

    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.bar(data.index, data, width=0.5, edgecolor='darkgray',
        linewidth=0.6,color=color_map)

    #annotations
    for i in data.index:
        ax.annotate(f"{data[i]}", 
                    xy=(i, data[i] + 80), #i like to change this to roughly 5% of the highest cat
                    va = 'center', ha='center', fontweight='light', fontfamily='serif')

    # Remove border from plot
    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)

def typeOfShow_byCountr():
    country_order = netflix['first_country'].value_counts()[:11].index
    data_q2q3 = netflix[['type', 'first_country']].groupby('first_country')['type'].value_counts().unstack().loc[country_order]
    data_q2q3['sum'] = data_q2q3.sum(axis=1)
    data_q2q3.fillna(0, inplace=True)
    data_q2q3_ratio = (data_q2q3.T / data_q2q3['sum']).T[['Movie', 'TV Show']].sort_values(by='TV Show',ascending=False)[::-1]

    ###
    fig, ax = plt.subplots(1,1,figsize=(15, 8),)

    ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['Movie'], 
            color='#b20710', alpha=0.8, label='Movie')
    ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['TV Show'], left=data_q2q3_ratio['Movie'], 
            color='#221f1f', alpha=0.8, label='TV Show')

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticklabels(data_q2q3_ratio.index, fontfamily='serif', fontsize=11)

    # male percentage
    for i in data_q2q3_ratio.index:
        ax.annotate(f"{round(data_q2q3_ratio['Movie'][i]*100, 2)}%", 
                    xy=(data_q2q3_ratio['Movie'][i]/2, i),
                    va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='serif',
                    color='white')

    for i in data_q2q3_ratio.index:
        ax.annotate(f"{round(data_q2q3_ratio['TV Show'][i]*100, 2)}%", 
                    xy=(data_q2q3_ratio['Movie'][i]+data_q2q3_ratio['TV Show'][i]/2, i),
                    va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='serif',
                    color='white')
        

    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)
        
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis=u'both', which=u'both',length=0)

def howContentIsAdded():
    month_order = ['January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December']

    netflix['month_name_added'] = pd.Categorical(netflix['month_added'], categories=month_order, ordered=True)

    data_sub = netflix.groupby('type')['month_name_added'].value_counts().unstack().fillna(0).loc[['TV Show','Movie']].cumsum(axis=0).T

    data_sub2 = data_sub

    data_sub2['Value'] = data_sub2['Movie'] + data_sub2['TV Show']
    data_sub2 = data_sub2.reset_index()

    df_polar = data_sub2.sort_values(by='month_name_added',ascending=False)


    color_map = ['#221f1f' for _ in range(12)]
    color_map[0] = color_map[11] =  '#b20710' # color highlight


    # initialize the figure
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    plt.axis('off')

    # Constants = parameters controling the plot layout:
    upperLimit = 30
    lowerLimit = 1
    labelPadding = 30

    # Compute max and min in the dataset
    max = df_polar['Value'].max()

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (max - lowerLimit) / max
    heights = slope * df_polar.Value + lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(df_polar.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df_polar.index)+1))
    angles = [element * width for element in indexes]

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth=2, 
        edgecolor="white",
        color=color_map,alpha=0.8
    )

    # Add labels
    for bar, angle, height, label in zip(bars,angles, heights, df_polar["month_name_added"]):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle, 
            y=lowerLimit + bar.get_height() + labelPadding, 
            s=label, 
            ha=alignment, fontsize=10,fontfamily='serif',
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor") 

def targetAges():
    data = netflix.groupby('first_country')[['count']].sum().sort_values(by='count',ascending=False).reset_index()[:10]
    data = data['first_country']

    # Custom colour map based on Netflix palette
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#221f1f', '#b20710','#f5f5f1'])

    df_heatmap = netflix.loc[netflix['first_country'].isin(data)]

    df_heatmap = pd.crosstab(df_heatmap['first_country'],df_heatmap['target_ages'],normalize = "index").T

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    country_order2 = ['USA', 'India', 'UK', 'Canada', 'Japan', 'France', 'Spain', 'Turkey']

    age_order = ['Kids','Older Kids','Teens','Adults']

    sns.heatmap(df_heatmap.loc[age_order,country_order2],cmap=cmap,square=True, linewidth=2.5,cbar=False,
                annot=True,fmt='1.0%',vmax=.6,vmin=0.05,ax=ax,annot_kws={"fontsize":12})

    ax.spines['top'].set_visible(True)


    fig.text(.99, .80, 'Target ages proportion of total content by country', fontweight='bold', fontfamily='serif', fontsize=15,ha='right')   
    fig.text(0.99, 0.78, 'Here we see interesting differences between countries. Most shows in India are targeted to teens, for instance.',ha='right', fontsize=12,fontfamily='serif') 

    ax.set_yticklabels(ax.get_yticklabels(), fontfamily='serif', rotation = 0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), fontfamily='serif', rotation=90, fontsize=11)

    ax.set_ylabel('')    
    ax.set_xlabel('')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.tight_layout()

if visualization_Type == 'Distribution of content':
    viz = distribution_Content()

elif visualization_Type == 'Who creates more content?':
    viz = countries_Production()

elif visualization_Type == 'Content by country':
    viz = typeOfShow_byCountr()

elif visualization_Type == 'How content is added?':
    viz = howContentIsAdded()

elif visualization_Type == 'Target ages by country':
    viz = targetAges()

st.pyplot(viz)