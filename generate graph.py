import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numpy.polynomial.polynomial import polyfit
import seaborn as sns
import random as rd

    # generate broken graph

os.chdir("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generate Dataset")

name_list = [
    'Unbalance',
    'Misalign',
    'Loose'
]

fig = plt.figure()
fig.tight_layout()
fig.set_size_inches(5,5)

count = 0
graphs = 200

    # type 1 # < 3% variation

for name in name_list:

    newpath = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\First Half\\{name}\\"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    newpath = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\Second Half\\{name}\\"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    for i in range(graphs):
        
        count += 1

        df = pd.read_csv(f'{name}.csv')

            # randomization method #

        if name == 'Loose':
            for column in df:
                if column.find('freq_noise') == -1:
                    if column.find('freq') == 0:
                        df[column] = df[column].transform(lambda x: x + (1 + rd.choice([-0.03,-0.02,0,0.02,0.03])))
                        if column.find('freq_main_spectrum') == -1:
                            dice = rd.choice([1,2])
                            if dice == 1:
                                pass
                            else:
                                df[column] = df[column].transform(lambda x: np.round(x))
                        else:
                            dice = rd.choice([1,2])
                            if dice == 1:
                                df[column] = df[column].transform(lambda x: np.round(x * rd.choice([1,2])))
                            else:
                                pass                 
                    else:
                        pass
                else:
                    df[column] = df[column].apply(lambda x: x*rd.choice([3,4,5]))

        if name == 'Unbalance':
            for column in df:
                if column.find('freq_noise') == -1:
                    if column.find('freq') == 0:
                        df[column] = df[column].transform(lambda x: x + (1 + rd.choice([-0.03,-0.02,0,0.02,0.03])))
                        if column.find('freq_main_spectrum') == -1:
                            dice = rd.choice([1,2,3])
                            if dice == 1:
                                pass
                            elif dice == 2:
                                df[column] = df[column].transform(lambda x: np.round(x * rd.uniform(1,3)))
                            else:
                                df[column] = df[column].transform(lambda x: np.round(x / rd.choice([1,1.5,2,2.5,3])))
                        else:
                            dice = rd.choice([1,2])
                            if dice == 1:
                                df[column] = df[column].transform(lambda x: x / rd.choice([2,3]))
                            else:
                                pass                 
                    else:
                        df[column] = df[column].transform(lambda x: x * rd.uniform(1,3))
                else:
                    df[column] = df[column].apply(lambda x: x*rd.choice([1,2]))

        if name == 'Misalign':
            misalign_main_freq_dice = rd.choice([1,2])
            for column in df:
                if column.find('freq_noise') == -1:
                    if column.find('freq') == 0:
                        df[column] = df[column].transform(lambda x: x + (1 + rd.choice([-0.03,-0.02,0,0.02,0.03])))
                        if column.find('freq_main_spectrum') == -1:
                            dice = rd.choice([1,2,3])
                            if dice == 1:
                                pass
                            elif dice == 2:
                                df[column] = df[column].transform(lambda x: np.round(x * rd.uniform(2,3)))
                            else:
                                df[column] = df[column].transform(lambda x: x * 3)
                        else:
                            if misalign_main_freq_dice == 1:
                                df[column] = df[column].transform(lambda x: np.round(x / 2))
                            else:
                                pass
                    else:
                        if column.find('vel_noise') == 0:
                            df[column] = df[column].transform(lambda x: x + (x * rd.uniform(-1,1)))
                        else:
                            df[column] = df[column].transform(lambda x: x * rd.uniform(1,3))
                else:
                    dice = rd.choice([1,2])
                    if dice == 1:
                        pass
                    else:
                        df[column] = df[column].apply(lambda x: x*0)

        freq_list = [df['freq_main_spectrum'],df['freq_noise_spectrum'],df['freq_noise'],df['freq_sub_spectrum']]
        vel_list = [df['vel_main_spectrum'],df['vel_noise_spectrum'],df['vel_noise'],df['vel_sub_spectrum']]

        freq_df = pd.concat(freq_list,axis='rows')
        vel_df = pd.concat(vel_list,axis='rows')    

            # plot #

        ax = fig.add_subplot(1,1,1)
        ax.scatter(freq_df,vel_df,s=0.5,color='black')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        ax.grid(False)
        ax.axis('off')

        if i < graphs/2:
            fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\First Half\\{name}\\{name}{i}3per.png")
        else:
            fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\Second Half\\{name}\\{name}{i}3per.png")            
        fig.clf()

        print(count)

    # type 2 # > 10% variation

for name in name_list:

    newpath = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\First Half\\{name}\\"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    newpath = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\Second Half\\{name}\\"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    for i in range(graphs):
        
        count += 1

        df = pd.read_csv(f'{name}.csv')

            # randomization method #

        if name == 'Loose':
            for column in df:
                if column.find('freq_noise') == -1:
                    if column.find('freq') == 0:
                        df[column] = df[column].transform(lambda x: x + (1 + rd.uniform(-0.1,0.1)))
                        if column.find('freq_main_spectrum') == -1:
                            dice = rd.choice([1,2])
                            if dice == 1:
                                pass
                            else:
                                df[column] = df[column].transform(lambda x: np.round(x))
                        else:
                            dice = rd.choice([1,2])
                            if dice == 1:
                                df[column] = df[column].transform(lambda x: np.round(x * rd.choice([1,2])))
                            else:
                                pass                 
                    else:
                        pass
                else:
                    df[column] = df[column].apply(lambda x: x*rd.choice([3,4,5]))

        if name == 'Unbalance':
            for column in df:
                if column.find('freq_noise') == -1:
                    if column.find('freq') == 0:
                        df[column] = df[column].transform(lambda x: x + (1 + rd.uniform(-0.1,0.1)))
                        if column.find('freq_main_spectrum') == -1:
                            dice = rd.choice([1,2,3])
                            if dice == 1:
                                pass
                            elif dice == 2:
                                df[column] = df[column].transform(lambda x: np.round(x * rd.uniform(1,3)))
                            else:
                                df[column] = df[column].transform(lambda x: np.round(x / rd.choice([1,1.5,2,2.5,3])))
                        else:
                            dice = rd.choice([1,2])
                            if dice == 1:
                                df[column] = df[column].transform(lambda x: x / rd.choice([2,3]))
                            else:
                                pass                 
                    else:
                        df[column] = df[column].transform(lambda x: x * rd.uniform(1,3))
                else:
                    df[column] = df[column].apply(lambda x: x*rd.choice([1,2]))

        if name == 'Misalign':
            misalign_main_freq_dice = rd.choice([1,2])
            for column in df:
                if column.find('freq_noise') == -1:
                    if column.find('freq') == 0:
                        df[column] = df[column].transform(lambda x: x + (1 + rd.uniform(-0.1,0.1)))
                        if column.find('freq_main_spectrum') == -1:
                            dice = rd.choice([1,2,3])
                            if dice == 1:
                                pass
                            elif dice == 2:
                                df[column] = df[column].transform(lambda x: np.round(x * rd.uniform(2,3)))
                            else:
                                df[column] = df[column].transform(lambda x: x * 3)
                        else:
                            if misalign_main_freq_dice == 1:
                                df[column] = df[column].transform(lambda x: np.round(x / 2))
                            else:
                                pass
                    else:
                        if column.find('vel_noise') == 0:
                            df[column] = df[column].transform(lambda x: x + (x * rd.uniform(-1,1)))
                        else:
                            df[column] = df[column].transform(lambda x: x * rd.uniform(1,3))
                else:
                    dice = rd.choice([1,2])
                    if dice == 1:
                        pass
                    else:
                        df[column] = df[column].apply(lambda x: x*0)

        freq_list = [df['freq_main_spectrum'],df['freq_noise_spectrum'],df['freq_noise'],df['freq_sub_spectrum']]
        vel_list = [df['vel_main_spectrum'],df['vel_noise_spectrum'],df['vel_noise'],df['vel_sub_spectrum']]

        freq_df = pd.concat(freq_list,axis='rows')
        vel_df = pd.concat(vel_list,axis='rows')    

            # plot #

        ax = fig.add_subplot(1,1,1)
        ax.scatter(freq_df,vel_df,s=0.5,color='black')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        ax.grid(False)
        ax.axis('off')

        if i < graphs/2:
            fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\First Half\\{name}\\{name}{i}10per.png")
        else:
            fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\Second Half\\{name}\\{name}{i}10per.png")
        fig.clf()

        print(count)