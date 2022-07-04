import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numpy.polynomial.polynomial import polyfit
import seaborn as sns
import random as rd

graphs = 50
fig = plt.figure()
fig.tight_layout()
fig.set_size_inches(5,5)

#-----------------------------------------------------------------------------------------------------------------------------#

    # SECTION build dataframe unbalance #

def generate_spectrum():

    amount_of_data = 1000

    # SECTION main spectrum #

    # NOTE create two dicts (frequency and velocity) -> turn to pandas dataframe -> concatenate them

    frequency_dict = {}
    velocity_dict = {}

    # NOTE dict[key] = val
    frequency_amount = 10
    velocity_amount = frequency_amount

    # NOTE create frequency columns
    for index in range(frequency_amount+1):
        name = f'frequency{index}'
        frequency_dict[name] = []

    # NOTE create velocity columns
    for index in range(velocity_amount+1):
        name = f'velocity{index}'
        velocity_dict[name] = []

    # NOTE generate values for frequency columns
    for column_index, column in enumerate(frequency_dict):
        for i in range(amount_of_data):
            frequency_dict[column].append(column_index)

    # NOTE generate values for velocity columns
    for column_index, column in enumerate(velocity_dict):
        random_range = rd.uniform(0,1)
        if column_index == 1:
            for i in range(amount_of_data):
                velocity_dict[column].append(rd.uniform(0,10))
        else:
            for i in range(amount_of_data):
                velocity_dict[column].append(rd.uniform(0,random_range))

    # SECTION noise #

    # NOTE generate noise frequency
    name = f'frequencynoise'
    frequency_dict[name] = []    

    # NOTE generate noise velocity
    name = f'velocitynoise'
    velocity_dict[name] = []        

    # NOTE generate values for frequency columns
    for i in range(amount_of_data):
        random_frequency = rd.uniform(0,10)
        frequency_dict['frequencynoise'].append(random_frequency)

    # NOTE generate values for velocity columns
    for i in range(amount_of_data):
        random_velocity = rd.uniform(0,0.2)
        velocity_dict['velocitynoise'].append(random_velocity)

    frequency_df = pd.DataFrame.from_dict(frequency_dict)
    velocity_df = pd.DataFrame.from_dict(velocity_dict)
    df_list = [frequency_df,velocity_df]

    return pd.concat(df_list,axis='columns')

for i in range(graphs):

        # SECTION generate and graph #

    generated_df = generate_spectrum()

        # SECTION randomize data #

    for columns in generated_df:
        if 'noise' not in columns:
            if 'frequency' in columns:
                random = rd.choice([1,2,3,4])
                generated_df[columns] = generated_df[columns].apply(lambda x: x/random)
            else:
                generated_df[columns] = generated_df[columns].apply(lambda x: x + (x * rd.choice([-0.3,-0.2,0,0.2,0.3])))
        else:
            random_delete = np.random.choice([0,1],p=[0.2,0.8])
            if columns == 'velocitynoise':
                if random_delete == 0:
                    generated_df['velocitynoise'] = generated_df['velocitynoise'].apply(lambda x: x * 0)
                    generated_df['frequencynoise'] = generated_df['frequencynoise'].apply(lambda x: x * 0)

    frequency_list = []
    velocity_list = []

    for columns in generated_df:
        if 'frequency' in columns:
            frequency_list.append(generated_df[columns])
        else:
            velocity_list.append(generated_df[columns])

    freq_df = pd.concat(frequency_list,axis='rows')
    vel_df = pd.concat(velocity_list,axis='rows')

    ax = fig.add_subplot(1,1,1)
    ax.scatter(frequency_list,velocity_list,s=0.5,color='black')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax.grid(False)
    ax.axis('off')

    print(i)
    fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Generated\\Unbalance\\unbalance{i}.png")
    fig.clf()

#-----------------------------------------------------------------------------------------------------------------------------#

    # SECTION build dataframe misalign #

def generate_spectrum():

    amount_of_data = 1000

    # SECTION main spectrum #

    # NOTE create two dicts (frequency and velocity) -> turn to pandas dataframe -> concatenate them

    frequency_dict = {}
    velocity_dict = {}

    # NOTE dict[key] = val
    frequency_amount = 10
    velocity_amount = frequency_amount

    # NOTE create frequency columns
    for index in range(frequency_amount+1):
        name = f'frequency{index}'
        frequency_dict[name] = []

    # NOTE create velocity columns
    for index in range(velocity_amount+1):
        name = f'velocity{index}'
        velocity_dict[name] = []

    # NOTE generate values for frequency columns
    for column_index, column in enumerate(frequency_dict):
        for i in range(amount_of_data):
            frequency_dict[column].append(column_index)

    # NOTE generate values for velocity columns
    for column_index, column in enumerate(velocity_dict):
        random_range0 = rd.uniform(0,1)
        random_range1 = rd.uniform(5,12)        
        if column_index == 0:
            for i in range(amount_of_data):
                velocity_dict[column].append(rd.uniform(0,10))
        elif column_index == 1:
            for i in range(amount_of_data):
                velocity_dict[column].append(rd.uniform(0,random_range1))
        else:
            for i in range(amount_of_data):
                velocity_dict[column].append(rd.uniform(0,random_range0))

    # SECTION noise #

    # NOTE generate noise frequency
    name = f'frequencynoise'
    frequency_dict[name] = []    

    # NOTE generate noise velocity
    name = f'velocitynoise'
    velocity_dict[name] = []        

    # NOTE generate values for frequency columns
    for i in range(amount_of_data):
        random_frequency = rd.uniform(0,10)
        frequency_dict['frequencynoise'].append(random_frequency)

    # NOTE generate values for velocity columns
    for i in range(amount_of_data):
        random_velocity = rd.uniform(0,0.2)
        velocity_dict['velocitynoise'].append(random_velocity)

    frequency_df = pd.DataFrame.from_dict(frequency_dict)
    velocity_df = pd.DataFrame.from_dict(velocity_dict)
    df_list = [frequency_df,velocity_df]

    return pd.concat(df_list,axis='columns')

for i in range(graphs):

        # SECTION generate and graph #

    generated_df = generate_spectrum()

        # SECTION randomize data #

    for columns in generated_df:
        if 'noise' not in columns:
            if 'frequency' in columns:
                random = rd.choice([1,2,3,4])
                generated_df[columns] = generated_df[columns].apply(lambda x: x/random)
            else:
                generated_df[columns] = generated_df[columns].apply(lambda x: x + (x * rd.choice([-0.3,-0.2,0,0.2,0.3])))
        else:
            random_delete = np.random.choice([0,1],p=[0.2,0.8])
            if columns == 'velocitynoise':
                if random_delete == 0:
                    generated_df['velocitynoise'] = generated_df['velocitynoise'].apply(lambda x: x * 0)
                    generated_df['frequencynoise'] = generated_df['frequencynoise'].apply(lambda x: x * 0)

    frequency_list = []
    velocity_list = []

    for columns in generated_df:
        if 'frequency' in columns:
            frequency_list.append(generated_df[columns])
        else:
            velocity_list.append(generated_df[columns])

    freq_df = pd.concat(frequency_list,axis='rows')
    vel_df = pd.concat(velocity_list,axis='rows')

    ax = fig.add_subplot(1,1,1)
    ax.scatter(frequency_list,velocity_list,s=0.5,color='black')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax.grid(False)
    ax.axis('off')

    print(i)
    fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Generated\\Misalign\\Misalign{i}.png")
    fig.clf()

#-----------------------------------------------------------------------------------------------------------------------------#

    # SECTION build dataframe loose #

def generate_spectrum():

    amount_of_data = 1000

    # SECTION main spectrum #

    # NOTE create two dicts (frequency and velocity) -> turn to pandas dataframe -> concatenate them

    frequency_dict = {}
    velocity_dict = {}

    # NOTE dict[key] = val
    frequency_amount = 10
    velocity_amount = frequency_amount

    # NOTE create frequency columns
    for index in range(frequency_amount+1):
        name = f'frequency{index}'
        frequency_dict[name] = []

    # NOTE create velocity columns
    for index in range(velocity_amount+1):
        name = f'velocity{index}'
        velocity_dict[name] = []

    # NOTE generate values for frequency columns
    for column_index, column in enumerate(frequency_dict):
        for i in range(amount_of_data):
            frequency_dict[column].append(column_index)

    # NOTE generate values for velocity columns
    for column_index, column in enumerate(velocity_dict):
        random_range0 = rd.uniform(0,7)      
        if column_index == 0:
            for i in range(amount_of_data):
                velocity_dict[column].append(rd.uniform(0,10))
        else:
            for i in range(amount_of_data):
                velocity_dict[column].append(rd.uniform(0,random_range0))

    # SECTION noise #

    # NOTE generate noise frequency
    name = f'frequencynoise'
    frequency_dict[name] = []    

    # NOTE generate noise velocity
    name = f'velocitynoise'
    velocity_dict[name] = []        

    # NOTE generate values for frequency columns
    for i in range(amount_of_data):
        random_frequency = rd.uniform(0,10)
        frequency_dict['frequencynoise'].append(random_frequency)

    # NOTE generate values for velocity columns
    for i in range(amount_of_data):
        random_velocity = rd.uniform(0,0.2)
        velocity_dict['velocitynoise'].append(random_velocity)

    frequency_df = pd.DataFrame.from_dict(frequency_dict)
    velocity_df = pd.DataFrame.from_dict(velocity_dict)
    df_list = [frequency_df,velocity_df]

    return pd.concat(df_list,axis='columns')

for i in range(graphs):

        # SECTION generate and graph #

    generated_df = generate_spectrum()

        # SECTION randomize data #

    for columns in generated_df:
        if 'noise' not in columns:
            if 'frequency' in columns:
                random = rd.choice([1,2,3,4])
                generated_df[columns] = generated_df[columns].apply(lambda x: x/random)
            else:
                generated_df[columns] = generated_df[columns].apply(lambda x: x + (x * rd.choice([-0.3,-0.2,0,0.2,0.3])))
        else:
            random_delete = np.random.choice([0,1],p=[0.2,0.8])
            if columns == 'velocitynoise':
                if random_delete == 0:
                    generated_df['velocitynoise'] = generated_df['velocitynoise'].apply(lambda x: x * 0)
                    generated_df['frequencynoise'] = generated_df['frequencynoise'].apply(lambda x: x * 0)

    frequency_list = []
    velocity_list = []

    for columns in generated_df:
        if 'frequency' in columns:
            frequency_list.append(generated_df[columns])
        else:
            velocity_list.append(generated_df[columns])

    freq_df = pd.concat(frequency_list,axis='rows')
    vel_df = pd.concat(velocity_list,axis='rows')

    ax = fig.add_subplot(1,1,1)
    ax.scatter(frequency_list,velocity_list,s=0.5,color='black')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax.grid(False)
    ax.axis('off')

    print(i)
    fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Generated\\Loose\\Loose{i}.png")
    fig.clf()