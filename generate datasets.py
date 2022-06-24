import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numpy.polynomial.polynomial import polyfit
import seaborn as sns
import random as rd

# freq_val_max = generated_df['freq_noise'].loc[generated_df['freq_noise'].idxmax()] find max value

#-----------------------------------------------------------------------------------------------------------------------------#

#     # build dataframe misalignment #

# dict_keys = ['freqvel','vel']
# header_keys = {}

# for index, key in enumerate(dict_keys):
#     header_keys[key] = []

# generated_df = pd.DataFrame(header_keys)

# def round_base(x, base=1.5):
#     return int(base * round(float(x)/base))

# def generate_spectrum(keys):
    
#         # main spectrum #

#     freq_data_main_spectrum = []
#     vel_data_main_spectrum = []

#     for i in range(1500):
#         freq_data_main_spectrum.append(np.random.choice([1,2]))
#         vel_data_main_spectrum.append(np.random.choice([1,2,3,4,5,6,7,8,9,10], p=[0.0625, 0.0625, 0.0625, 0.0625, 0.25, 0.25, 0.0625, 0.0625, 0.0625, 0.0625]) * rd.random())    

#         # noise spectrum #

#     freq_data_noise_spectrum = []
#     vel_data_noise_spectrum = []

#     for i in range(1000):
#         freq_data_noise_spectrum.append(np.random.choice([3,4,5,6,7,8,9,10]))
#         random_freq_noise_spectrum = rd.randint(10,20)
#         vel_data_noise_spectrum.append(np.random.choice([1,2,3,4,5,6,7,8,9,10]) * rd.random() / random_freq_noise_spectrum)        

#         # noise #

#     freq_data_noise = []
#     vel_data_noise = []

#     for i in range(1500):
#         freq_noise = np.random.choice([1,2,3,4,5,6,7,8,9,10])
#         random_freq_noise = rd.choice([-1,1]) * rd.uniform(0,0.3) # move by +- 20%
#         freq_noise += freq_noise * random_freq_noise 
#         freq_data_noise.append(freq_noise)  
#         dice = rd.random()  
#         if -0.1 < random_freq_noise < 0.1:
#             vel_data_noise.append(np.random.choice([1,2,3,4,5,6,7,8,9,10]) * rd.random() / 10)  
#         elif (-0.2 <= random_freq_noise <= -0.1) or (0.1 <= random_freq_noise <= 0.2):
#             vel_data_noise.append(np.random.choice([1,2,3,4,5,6,7,8,9,10]) * rd.random() / 30)  
#         else:
#             vel_data_noise.append(np.random.choice([1,2,3,4,5,6,7,8,9,10]) * rd.random() / 60)                

#         # combine #

#     freq_data_combine = freq_data_main_spectrum + freq_data_noise_spectrum + freq_data_noise
#     vel_data_combine = vel_data_main_spectrum + vel_data_noise_spectrum + vel_data_noise

#     generated_df[keys[0]] = freq_data_combine
#     generated_df[keys[0]] = generated_df[keys[0]]  

#     generated_df[keys[1]] = vel_data_combine
#     generated_df[keys[1]] = generated_df[keys[1]]

# generate_spectrum(dict_keys)

# print(generated_df)

# # generated_df.to_csv("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generate Dataset\\misalignment.csv")

# fig = plt.figure()
# fig.tight_layout()
# fig.canvas.manager.full_screen_toggle()

# ax = fig.add_subplot(1,1,1,title='generate')
# ax.scatter(generated_df['freqvel'],generated_df['vel'],s=0.5,color='black')

# plt.show()

#-----------------------------------------------------------------------------------------------------------------------------#

#     # build dataframe unbalance #

# dict_keys = ['freqvel','vel']
# header_keys = {}

# for index, key in enumerate(dict_keys):
#     header_keys[key] = []

# generated_df = pd.DataFrame(header_keys)

# def round_base(x, base=1.5):
#     return int(base * round(float(x)/base))

# def generate_spectrum(keys):
    
#         # main spectrum #

#     freq_data_main_spectrum = []
#     vel_data_main_spectrum = []

#     for i in range(5000):
#         freq = np.random.choice([1,2,3,4], p=[0.6,0.2,0.1,0.1])
#         freq_data_main_spectrum.append(freq)
#         if freq == 1:
#             vel_data_main_spectrum.append(np.random.choice([1,2,3,4,5,6,7,15,30,100], p=[0.0625, 0.0625, 0.0625, 0.0625, 0.25, 0.25, 0.0625, 0.0625, 0.0625, 0.0625]) * rd.random())    
#         elif freq == 2:
#             vel_data_main_spectrum.append(np.random.choice([1,2,3,4,5,6,7,13,15,40], p=[0.0625, 0.0625, 0.0625, 0.0625, 0.25, 0.25, 0.0625, 0.0625, 0.0625, 0.0625]) * rd.random())    
#         elif freq == 3:
#             vel_data_main_spectrum.append(np.random.choice([1,2,3,4,5,6,7,10,12,20], p=[0.0625, 0.0625, 0.0625, 0.0625, 0.25, 0.25, 0.0625, 0.0625, 0.0625, 0.0625]) * rd.random())    
#         else:
#             vel_data_main_spectrum.append(np.random.choice([1,2,3,4,5,6,7,8,9,10], p=[0.0625, 0.0625, 0.0625, 0.0625, 0.25, 0.25, 0.0625, 0.0625, 0.0625, 0.0625]) * rd.random())    

#         # noise spectrum #

#     freq_data_noise_spectrum = []
#     vel_data_noise_spectrum = []

#     for i in range(1000):
#         freq_data_noise_spectrum.append(np.random.choice([3,4,5,6,7,8,9,10]))
#         random_freq_noise_spectrum = rd.randint(10,20)
#         vel_data_noise_spectrum.append(np.random.choice([1,2,3,4,5,6,7,8,9,10]) * rd.random() / random_freq_noise_spectrum)        

#         # noise #

#     freq_data_noise = []
#     vel_data_noise = []

#     for i in range(1500):
#         freq_noise = np.random.choice([1,2,3,4,5,6,7,8,9,10])
#         random_freq_noise = rd.choice([-1,1]) * rd.uniform(0,0.3) # move by +- 20%
#         freq_noise += freq_noise * random_freq_noise 
#         freq_data_noise.append(freq_noise)  
#         dice = rd.random()  
#         if -0.1 < random_freq_noise < 0.1:
#             vel_data_noise.append(np.random.choice([1,2,3,4,5,6,7,8,9,10]) * rd.random() / 10)  
#         elif (-0.2 <= random_freq_noise <= -0.1) or (0.1 <= random_freq_noise <= 0.2):
#             vel_data_noise.append(np.random.choice([1,2,3,4,5,6,7,8,9,10]) * rd.random() / 30)  
#         else:
#             vel_data_noise.append(np.random.choice([1,2,3,4,5,6,7,8,9,10]) * rd.random() / 60)                

#         # combine #

#     freq_data_combine = freq_data_main_spectrum + freq_data_noise_spectrum + freq_data_noise
#     vel_data_combine = vel_data_main_spectrum + vel_data_noise_spectrum + vel_data_noise

#     generated_df[keys[0]] = freq_data_combine
#     generated_df[keys[0]] = generated_df[keys[0]]  

#     generated_df[keys[1]] = vel_data_combine
#     generated_df[keys[1]] = generated_df[keys[1]]

# generate_spectrum(dict_keys)

# print(generated_df)

# # generated_df.to_csv("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generate Dataset\\unbalance.csv")

# fig = plt.figure()
# fig.tight_layout()
# fig.canvas.manager.full_screen_toggle()

# ax = fig.add_subplot(1,1,1,title='generate')
# ax.scatter(generated_df['freqvel'],generated_df['vel'],s=0.5,color='black')

# plt.show()

#-----------------------------------------------------------------------------------------------------------------------------#

    # build dataframe loose #

def generate_spectrum():
    
        # main spectrum #

    freq_data_main_spectrum = []
    vel_data_main_spectrum = []

    for i in range(4000):
        freq = np.random.choice(np.arange(1,6,1))
        freq_data_main_spectrum.append(freq)
        if freq == 1:
            vel_data_main_spectrum.append(np.random.choice(np.arange(1,6,1)) * rd.random())         
        elif freq == 2:
            vel_data_main_spectrum.append(np.random.choice(np.arange(0,80,10)) * rd.random())    
        elif freq == 3:
            vel_data_main_spectrum.append(np.random.choice(np.arange(0,40,10)) * rd.random())    
        elif freq == 4:
            vel_data_main_spectrum.append(np.random.choice(np.arange(0,50,10)) * rd.random())    
        else:
            vel_data_main_spectrum.append(np.random.choice(np.arange(0,10,1)) * rd.random())    

        # sub spectrum #

    freq_data_sub_spectrum = []
    vel_data_sub_spectrum = []

    for i in range(4000):
        freq = np.random.choice(np.arange(7,11,1), p=[10/100,40/100,10/100,40/100])
        freq_data_sub_spectrum.append(freq)
        if freq == 7:
            vel_data_sub_spectrum.append(np.random.choice(np.arange(1,6,1)) * rd.random())    
        elif freq == 8:
            vel_data_sub_spectrum.append(np.random.choice(np.arange(1,11,2)) * rd.random())    
        elif freq == 9:
            vel_data_sub_spectrum.append(np.random.choice(np.arange(1,6,1)) * rd.random())    
        else:
            vel_data_sub_spectrum.append(np.random.choice(np.arange(1,11,2)) * rd.random())    

        # noise spectrum #

    freq_data_noise_spectrum = []
    vel_data_noise_spectrum = []

    for i in range(4000):
        freq_data_noise_spectrum.append(np.random.choice(np.arange(3,11,1)))
        random_freq_noise_spectrum = rd.randint(10,20)
        vel_data_noise_spectrum.append(np.random.choice(np.arange(1,11,1)) * rd.random() / random_freq_noise_spectrum)        

        # noise #

    freq_data_noise = []
    vel_data_noise = []

    for i in range(4000):
        freq_noise = np.random.choice(np.arange(1,11,1))
        random_freq_noise = rd.choice([-1,1]) * rd.uniform(0,0.3) # move by +- 20%
        freq_noise += freq_noise * random_freq_noise 
        freq_data_noise.append(freq_noise)  
        if -0.1 < random_freq_noise < 0.1:
            vel_data_noise.append(np.random.choice(np.arange(1,11,1)) * rd.random() / 10)  
        elif (-0.2 <= random_freq_noise <= -0.1) or (0.1 <= random_freq_noise <= 0.2):
            vel_data_noise.append(np.random.choice(np.arange(1,11,1)) * rd.random() / 20)  
        else:
            vel_data_noise.append(np.random.choice(np.arange(1,11,1)) * rd.random() / 40)                

        # combine #

    val_dict = {
        'freq_main_spectrum':freq_data_main_spectrum,
        'freq_noise_spectrum':freq_data_noise_spectrum,
        'freq_noise':freq_data_noise,
        'freq_sub_spectrum':freq_data_sub_spectrum,
        'vel_main_spectrum':vel_data_main_spectrum,
        'vel_noise_spectrum':vel_data_noise_spectrum,
        'vel_noise':vel_data_noise,
        'vel_sub_spectrum':vel_data_sub_spectrum,
        }

    df = pd.DataFrame.from_dict(val_dict)

    return df

#     randomization method #

# for column in generated_df:
#     dice = rd.choice([0,1,2,3])
#     print(dice)
#     if column.find('freq') == 0:
#         generated_df[column] = generated_df[column].transform(lambda x: x + (1 + rd.choice([-0.03,-0.02,0,0.02,0.03])))
#         if dice == 0:
#             pass
#         elif dice == 1:
#             generated_df[column] = generated_df[column].transform(lambda x: x**rd.choice([2,3]))
#         elif dice == 2:
#             generated_df[column] = generated_df[column].transform(lambda x: np.log(x))
#         else:
#             generated_df[column] = generated_df[column].transform(lambda x: np.round(x))
#     else:
#         generated_df[column] = generated_df[column].transform(lambda x: x * rd.choice([1,2,3]))

# freq_list = [generated_df['freq_main_spectrum'],generated_df['freq_noise_spectrum'],generated_df['freq_noise'],generated_df['freq_sub_spectrum']]
# vel_list = [generated_df['vel_main_spectrum'],generated_df['vel_noise_spectrum'],generated_df['vel_noise'],generated_df['vel_sub_spectrum']]

# freq_df = pd.concat(freq_list,axis='rows')
# vel_df = pd.concat(vel_list,axis='rows')

#-----------------------------------------------------------------------------------------------------------------------------#

#     # build dataframe misalign #

# def generate_spectrum():
    
#         # main spectrum #

#     freq_data_main_spectrum = []
#     vel_data_main_spectrum = []

#     for i in range(4000):
#         freq = np.random.choice([0,1])
#         freq_data_main_spectrum.append(freq)
#         vel_data_main_spectrum.append(np.random.choice(np.arange(10,110,10)) * rd.random())         

#         # sub spectrum #

#     freq_data_sub_spectrum = []
#     vel_data_sub_spectrum = []

#     for i in range(4000):
#         freq = np.random.choice(np.arange(2,8,1))
#         freq_data_sub_spectrum.append(freq)
#         dice = rd.choice([1,2])
#         if dice == 1:
#             vel_data_sub_spectrum.append(np.random.choice(np.arange(1,6,1)) * rd.random())    
#         else:
#             vel_data_sub_spectrum.append(np.random.choice(np.arange(1,11,2)) * rd.random())      

#         # noise spectrum #

#     freq_data_noise_spectrum = []
#     vel_data_noise_spectrum = []

#     for i in range(4000):
#         freq_data_noise_spectrum.append(np.random.choice(np.arange(3,11,1)))
#         random_freq_noise_spectrum = rd.randint(10,20)
#         vel_data_noise_spectrum.append(np.random.choice(np.arange(1,11,1)) * rd.random() / random_freq_noise_spectrum)        

#         # noise #

#     freq_data_noise = []
#     vel_data_noise = []

#     for i in range(4000):
#         freq_noise = np.random.choice(np.arange(1,11,1))
#         random_freq_noise = rd.choice([-1,1]) * rd.uniform(0,0.3) # move by +- 20%
#         freq_noise += freq_noise * random_freq_noise 
#         freq_data_noise.append(freq_noise)  
#         if -0.1 < random_freq_noise < 0.1:
#             vel_data_noise.append(np.random.choice(np.arange(1,11,1)) * rd.random() / 10)  
#         elif (-0.2 <= random_freq_noise <= -0.1) or (0.1 <= random_freq_noise <= 0.2):
#             vel_data_noise.append(np.random.choice(np.arange(1,11,1)) * rd.random() / 20)  
#         else:
#             vel_data_noise.append(np.random.choice(np.arange(1,11,1)) * rd.random() / 40)                

#         # combine #

#     val_dict = {
#         'freq_main_spectrum':freq_data_main_spectrum,
#         'freq_noise_spectrum':freq_data_noise_spectrum,
#         'freq_noise':freq_data_noise,
#         'freq_sub_spectrum':freq_data_sub_spectrum,
#         'vel_main_spectrum':vel_data_main_spectrum,
#         'vel_noise_spectrum':vel_data_noise_spectrum,
#         'vel_noise':vel_data_noise,
#         'vel_sub_spectrum':vel_data_sub_spectrum,
#         }

#     df = pd.DataFrame.from_dict(val_dict)

#     return df

# #     # randomization method #

# # combined_freq = []
# # combined_vel = []

# # for column in generated_df:
# #     if column.find('freq_noise') == -1:
# #         if column.find('freq') == 0:
# #             generated_df[column] = generated_df[column].transform(lambda x: x + (1 + rd.choice([-0.03,-0.02,0,0.02,0.03])))
# #             if column.find('freq_main_spectrum') == -1:
# #                 dice = rd.choice([1,2])
# #                 if dice == 1:
# #                     generated_df[column] = generated_df[column].transform(lambda x: x**rd.choice([1,1.5,2]))
# #                 else:
# #                     generated_df[column] = generated_df[column].transform(lambda x: np.sqrt(x))
# #             else:
# #                 pass
# #         else:
# #             generated_df[column] = generated_df[column].transform(lambda x: x * rd.choice([1,2,3]))
# #     else:
# #         generated_df[column] = generated_df[column].transform(lambda x: x * rd.choice(np.arange(1,4,1)))

# # freq_list = [generated_df['freq_main_spectrum'],generated_df['freq_noise_spectrum'],generated_df['freq_noise'],generated_df['freq_sub_spectrum']]
# # vel_list = [generated_df['vel_main_spectrum'],generated_df['vel_noise_spectrum'],generated_df['vel_noise'],generated_df['vel_sub_spectrum']]

# # freq_df = pd.concat(freq_list,axis='rows')
# # vel_df = pd.concat(vel_list,axis='rows')

#-----------------------------------------------------------------------------------------------------------------------------#

#     # build dataframe unbalance #

# def generate_spectrum():
    
#         # main spectrum #

#     freq_data_main_spectrum = []
#     vel_data_main_spectrum = []

#     for i in range(4000):
#         freq = np.random.choice(np.arange(1,5,1))
#         freq_data_main_spectrum.append(freq)
#         if freq == 1:
#             vel_data_main_spectrum.append(np.random.choice(np.arange(10,110,10)) * rd.random())         
#         elif freq == 2:
#             vel_data_main_spectrum.append(np.random.choice(np.arange(10,50,10)) * rd.random())   
#         elif freq == 3:
#             vel_data_main_spectrum.append(np.random.choice(np.arange(10,30,10)) * rd.random())   
#         else:
#             vel_data_main_spectrum.append(np.random.choice(np.arange(5,15,5)) * rd.random())   

#         # sub spectrum #

#     freq_data_sub_spectrum = []
#     vel_data_sub_spectrum = []

#     for i in range(4000):
#         freq = np.random.choice(np.arange(2,8,1))
#         freq_data_sub_spectrum.append(freq)
#         dice = rd.choice([1,2])
#         if dice == 1:
#             vel_data_sub_spectrum.append(np.random.choice(np.arange(1,6,1)) * rd.random())    
#         else:
#             vel_data_sub_spectrum.append(np.random.choice(np.arange(1,9,2)) * rd.random())      

#         # noise spectrum #

#     freq_data_noise_spectrum = []
#     vel_data_noise_spectrum = []

#     for i in range(4000):
#         freq_data_noise_spectrum.append(np.random.choice(np.arange(3,11,1)))
#         random_freq_noise_spectrum = rd.randint(10,20)
#         vel_data_noise_spectrum.append(np.random.choice(np.arange(1,11,1)) * rd.random() / random_freq_noise_spectrum)        

#         # noise #

#     freq_data_noise = []
#     vel_data_noise = []

#     for i in range(4000):
#         freq_noise = np.random.choice(np.arange(1,11,1))
#         random_freq_noise = rd.choice([-1,1]) * rd.uniform(0,0.3) # move by +- 20%
#         freq_noise += freq_noise * random_freq_noise 
#         freq_data_noise.append(freq_noise)  
#         if -0.1 < random_freq_noise < 0.1:
#             vel_data_noise.append(np.random.choice(np.arange(1,11,1)) * rd.random() / 10)  
#         elif (-0.2 <= random_freq_noise <= -0.1) or (0.1 <= random_freq_noise <= 0.2):
#             vel_data_noise.append(np.random.choice(np.arange(1,11,1)) * rd.random() / 20)  
#         else:
#             vel_data_noise.append(np.random.choice(np.arange(1,11,1)) * rd.random() / 40)                

#         # combine #

#     val_dict = {
#         'freq_main_spectrum':freq_data_main_spectrum,
#         'freq_noise_spectrum':freq_data_noise_spectrum,
#         'freq_noise':freq_data_noise,
#         'freq_sub_spectrum':freq_data_sub_spectrum,
#         'vel_main_spectrum':vel_data_main_spectrum,
#         'vel_noise_spectrum':vel_data_noise_spectrum,
#         'vel_noise':vel_data_noise,
#         'vel_sub_spectrum':vel_data_sub_spectrum,
#         }

#     df = pd.DataFrame.from_dict(val_dict)

#     return df

# #     # randomization method #

# # for column in generated_df:
# #     if column.find('freq_noise') == -1:
# #         if column.find('freq') == 0:
# #             generated_df[column] = generated_df[column].transform(lambda x: x + (1 + rd.choice([-0.03,-0.02,0,0.02,0.03])))
# #             if column.find('freq_main_spectrum') == -1:
# #                 dice = rd.choice([1,2])
# #                 if dice == 1:
# #                     generated_df[column] = generated_df[column].transform(lambda x: x**rd.choice([1,1.5,2]))
# #                 else:
# #                     generated_df[column] = generated_df[column].transform(lambda x: np.sqrt(x))
# #             else:
# #                 pass
# #         else:
# #             generated_df[column] = generated_df[column].transform(lambda x: x * rd.choice([1,2,3]))
# #     else:
# #         generated_df[column] = generated_df[column].transform(lambda x: x * rd.choice(np.arange(1,4,1)))

# # freq_list = [generated_df['freq_main_spectrum'],generated_df['freq_noise_spectrum'],generated_df['freq_noise'],generated_df['freq_sub_spectrum']]
# # vel_list = [generated_df['vel_main_spectrum'],generated_df['vel_noise_spectrum'],generated_df['vel_noise'],generated_df['vel_sub_spectrum']]

# # freq_df = pd.concat(freq_list,axis='rows')
# # vel_df = pd.concat(vel_list,axis='rows')      

#-----------------------------------------------------------------------------------------------------------------------------#

    # generate #

generated_df = generate_spectrum()

fig = plt.figure()
fig.tight_layout()
fig.canvas.manager.full_screen_toggle()

ax = fig.add_subplot(1,1,1,title='generate')

#     # randomization method #

# for column in generated_df:
#     if column.find('freq_noise') == -1:
#         if column.find('freq') == 0:
#             generated_df[column] = generated_df[column].transform(lambda x: x + (1 + rd.choice([-0.03,-0.02,0,0.02,0.03])))
#             if column.find('freq_main_spectrum') == -1:
#                 dice = rd.choice([1,2])
#                 if dice == 1:
#                     generated_df[column] = generated_df[column].transform(lambda x: x**rd.choice([1,1.5,2]))
#                 else:
#                     generated_df[column] = generated_df[column].transform(lambda x: np.sqrt(x))
#             else:
#                 pass
#         else:
#             generated_df[column] = generated_df[column].transform(lambda x: x * rd.choice([1,2,3]))
#     else:
#         generated_df[column] = generated_df[column].transform(lambda x: x * rd.choice(np.arange(1,4,1)))

freq_list = [generated_df['freq_main_spectrum'],generated_df['freq_noise_spectrum'],generated_df['freq_noise'],generated_df['freq_sub_spectrum']]
vel_list = [generated_df['vel_main_spectrum'],generated_df['vel_noise_spectrum'],generated_df['vel_noise'],generated_df['vel_sub_spectrum']]

freq_df = pd.concat(freq_list,axis='rows')
vel_df = pd.concat(vel_list,axis='rows')

ax.scatter(freq_df,vel_df,s=0.5,color='black')

plt.show()

generated_df.to_csv("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generate Dataset\\Loose.csv")

#-----------------------------------------------------------------------------------------------------------------------------#