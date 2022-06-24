import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#-------------------------------------------------------------------#
    
    #code graveyard

# df = pd.read_csv('SensorNode_20220411.csv')
# df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
# df.to_csv('temp.csv', index=False)
# print(len(df.columns))
# if len(df.columns) == 43:
#     temp_df = temp_df.drop(list(temp_df)[2:4], axis='columns')
# temp_df.drop(temp_df.columns[temp_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
# if len(temp_df.columns) == 43:
#     temp_df = temp_df.drop(list(temp_df)[2:4], axis='columns')

#-------------------------------------------------------------------#

    #clean data

# directory = os.fsencode("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Log Vibration")
    
# df_list = []
# problem_list = []
# for count, file in enumerate(os.listdir(directory)):
#     filename = os.fsdecode(file)
#     temp_df = pd.read_csv(filename, header=None)
#     df_list.append(temp_df)

# all_df = pd.concat(df_list)
# all_df.to_csv("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\all_data.csv", index=False)

# df = pd.read_csv('all_data.csv', header=None)
# df = df.drop(list(df)[2:4], axis='columns')
# df.to_csv("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\all_data_clean.csv", index=False)

# os.chdir("E:\Gun's stuff\Machine Learning and Deep Learning\Machine Learning\Ajinomoto Project")
    
# df = pd.read_csv('all_data_clean.csv', header=None)
# complete_df = df.iloc[:, [0,5,7,9,11,13,15,17,19,21,23,25,27,31,35]]
# complete_df_renamed = complete_df.rename({0:'name', 5:'freq1acc', 7:'acc1', 9:'freq2acc', 11:'acc2', 13:'accRMS', 15:'freq1vel', 17:'vel1', 19:'freq2vel', 21:'vel2', 23:'freq3vel', 25:'vel3', 27:'velRMS', 31:'temp',35:'time'}, axis='columns')

#-------------------------------------------------------------------#

    #plot

# fig = plt.figure()
# fig.tight_layout()
# fig.canvas.manager.full_screen_toggle()

    #freq acc vel

# ax0 = fig.add_subplot(3,4,1,title='freq1acc(x) vs acc1(y)')
# ax0.scatter(complete_df_renamed.iloc[:,1],complete_df_renamed.iloc[:,2],s=0.5)
    
# ax1 = fig.add_subplot(3,4,2,title='freq2acc(x) vs acc2(y)')
# ax1.scatter(complete_df_renamed.iloc[:,3],complete_df_renamed.iloc[:,4],s=0.5)

# ax2 = fig.add_subplot(3,4,3,title='freq1vel(x) vs vel1(y)')
# ax2.scatter(complete_df_renamed.iloc[:,6],complete_df_renamed.iloc[:,7],s=0.5)

# ax3 = fig.add_subplot(3,4,4,title='freq2vel(x) vs vel2(y)')
# ax3.scatter(complete_df_renamed.iloc[:,8],complete_df_renamed.iloc[:,9],s=0.5)

# ax4 = fig.add_subplot(3,4,5,title='freq3vel(x) vs vel3(y)')
# ax4.scatter(complete_df_renamed.iloc[:,10],complete_df_renamed.iloc[:,11],s=0.5)

# ax5 = fig.add_subplot(3,4,6,title='freqvel(x) vs vel(y)')
# ax5.scatter(complete_df_renamed.iloc[:,6],complete_df_renamed.iloc[:,7],s=0.5)
# ax5.scatter(complete_df_renamed.iloc[:,8],complete_df_renamed.iloc[:,9],s=0.5)
# ax5.scatter(complete_df_renamed.iloc[:,10],complete_df_renamed.iloc[:,11],s=0.5)

# ax6 = fig.add_subplot(3,4,7,title='temp(x) vs velRMS(y)')
# ax6.scatter(complete_df_renamed.iloc[:,13],complete_df_renamed.iloc[:,12],s=0.5)

# ax7 = fig.add_subplot(3,4,8,title='temp(x) vs accRMS(y)')
# ax7.scatter(complete_df_renamed.iloc[:,13],complete_df_renamed.iloc[:,5],s=0.5)

    #time accRMS

# ax = fig.add_subplot(1,1,1,title='time(x) vs accRMS(y)')
# ax.plot(complete_df_renamed.iloc[:,4], linewidth=0.05)

    #time velRMS

# ax = fig.add_subplot(1,1,1,title='time(x) vs velRMS(y)')
# ax.plot(complete_df_renamed.iloc[:,11], linewidth=0.05)   

# plt.show()

# index = complete_df_renamed['name'].str.contains('E3F7')
# E3F7_df = complete_df_renamed[index==True].reset_index(drop=True)

# unique_name = complete_df_renamed['name'].unique()
# print(unique_name)

# for name in unique_name:
#     index = complete_df_renamed['name'].str.contains(name)
#     temp_df = complete_df_renamed[index==True].reset_index(drop=True)
#     temp_df.to_csv(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\{name}.csv", index=False)

# print(temp_df)

    #clean data

# directory = os.fsencode("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Log Vibration")
    
# df_list = []
# problem_list = []
# for count, file in enumerate(os.listdir(directory)):
#     filename = os.fsdecode(file)
#     temp_df = pd.read_csv(filename, header=None)
#     df_list.append(temp_df)

# all_df = pd.concat(df_list)
# all_df.to_csv("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\all_data.csv", index=False)

# df = pd.read_csv('all_data.csv', header=None)
# os.chdir("E:\Gun's stuff\Machine Learning and Deep Learning\Machine Learning\Ajinomoto Project\E412BROKEN(DECON2)")

# df = pd.read_csv('DECON2MOTOR.csv', header=None)

# df = df.drop(list(df)[2:5], axis='columns')
# # print(df)
    
# complete_df = df.iloc[:, [0,5,7,9,11,13,15,17,19,21,23,25,27,31,35]]
# complete_df_renamed = complete_df.rename({0:'name', 8:'freq1acc', 10:'acc1', 12:'freq2acc', 14:'acc2', 16:'accRMS', 18:'freq1vel', 20:'vel1', 22:'freq2vel', 24:'vel2', 26:'freq3vel', 28:'vel3', 30:'velRMS', 34:'temp',38:'time'}, axis='columns')
# print(complete_df_renamed)
# complete_df_renamed.to_csv("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\E412BROKEN(DECON2)\\MOTORDATACLEANED.csv", index=False)

df = pd.read_csv("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\all_data_clean.csv")
unique_name = df['name'].unique()
print(unique_name)