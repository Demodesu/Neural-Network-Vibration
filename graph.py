import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir("E:\Gun's stuff\Machine Learning and Deep Learning\Machine Learning\Ajinomoto Project\Machine")

# name  freq1acc  acc1  freq2acc  acc2  accRMS  freq1vel  vel1  freq2vel  vel2  freq3vel  vel3  velRMS  temp  time
# 0 name 1 freq1acc 2 acc1 3 freq2acc 4 acc2 5 accRMS 6 freq1vel 7 vel1 8 freq2vel 9 vel2 10 freq3vel 11 vel3 12 velRMS 13 time

# name_list = [
#     'E3CE', 'E3F7', 'E484', 'E3D1', 'E407', 'E40E', 'E3FC', 'E3DF', 'E3DC', 'E410',
#     'E3EC', 'E3E1', 'E3D5', 'E3E6', 'E413', 'E470', 'E401', 'E47E', 'E3D4', 'E466',
#     'E412', 'E408', 'E402', 'E3F8', 'E480', 'E3F0', 'E473', 'E3D8', 'E417', 'E3FA',
#     'E416', 'E3F3', 'E476', 'E47B', 'E3F1', 'E414', 'E3FB', 'E3FD'
# ]

#     #plot

# fig = plt.figure()
# fig.tight_layout()
# fig.canvas.manager.full_screen_toggle()
# fig.set_size_inches(18, 10)

# for name in name_list:

#     df = pd.read_csv(f'{name}.csv')

#     fig.suptitle(f'{name}', fontsize=25)

#         #graph

#     ax0 = fig.add_subplot(3,4,1,title='freq1acc(x) vs acc1(y)')
#     ax0.scatter(df.iloc[:,1],df.iloc[:,2],s=0.5)
        
#     ax1 = fig.add_subplot(3,4,2,title='freq2acc(x) vs acc2(y)')
#     ax1.scatter(df.iloc[:,3],df.iloc[:,4],s=0.5)

#     ax2 = fig.add_subplot(3,4,3,title='freq1vel(x) vs vel1(y)')
#     ax2.scatter(df.iloc[:,6],df.iloc[:,7],s=0.5)

#     ax3 = fig.add_subplot(3,4,4,title='freq2vel(x) vs vel2(y)')
#     ax3.scatter(df.iloc[:,8],df.iloc[:,9],s=0.5)

#     ax4 = fig.add_subplot(3,4,5,title='freq3vel(x) vs vel3(y)')
#     ax4.scatter(df.iloc[:,10],df.iloc[:,11],s=0.5)

#     ax5 = fig.add_subplot(3,4,6,title='freqvel(x) vs vel(y)')
#     ax5.scatter(df.iloc[:,6],df.iloc[:,7],s=0.5)
#     ax5.scatter(df.iloc[:,8],df.iloc[:,9],s=0.5)
#     ax5.scatter(df.iloc[:,10],df.iloc[:,11],s=0.5)

#     ax6 = fig.add_subplot(3,4,7,title='temp(x) vs velRMS(y)')
#     ax6.scatter(df.iloc[:,13],df.iloc[:,12],s=0.5)

#     ax7 = fig.add_subplot(3,4,8,title='temp(x) vs accRMS(y)')
#     ax7.scatter(df.iloc[:,13],df.iloc[:,5],s=0.5)

#     ax8 = fig.add_subplot(3,4,9,title='temp(x) vs vel(y)')
#     ax8.scatter(df.iloc[:,13],df.iloc[:,7],s=0.5)
#     ax8.scatter(df.iloc[:,13],df.iloc[:,9],s=0.5)
#     ax8.scatter(df.iloc[:,13],df.iloc[:,11],s=0.5)

#     ax9 = fig.add_subplot(3,4,10,title='temp(x) vs acc(y)')
#     ax9.scatter(df.iloc[:,13],df.iloc[:,2],s=0.5)
#     ax9.scatter(df.iloc[:,13],df.iloc[:,4],s=0.5)

#     ax10 = fig.add_subplot(3,4,11,title='velRMS(y)')
#     ax10.plot(df.iloc[:,12],linewidth=0.2)

#     ax11 = fig.add_subplot(3,4,12,title='accRMS(y)')
#     ax11.plot(df.iloc[:,5],linewidth=0.2)

#     # ax12 = fig.add_subplot(1,1,1,title='velRMS(y)')
#     # ax12.plot(df.iloc[:,12],linewidth=0.5)

#     # ax13 = fig.add_subplot(1,1,1,title='accRMS(y)')
#     # ax13.plot(df.iloc[:,5],linewidth=0.5)

#     fig.savefig(name)
#     fig.clf()

# plt.show()

#-------------------------------------------------------------------#

    #time

# for name in name_list:

#     df = pd.read_csv(f'{name}.csv')

#     fig.suptitle(f'{name}accRMSavg', fontsize=25)

#     unique_date = df['time'].apply(lambda x: x[0:9]).unique().tolist()
#     velRMS_list = []

#     for date in unique_date:
#         index = df['time'].str.contains(date)
#         temp_df = df[index==True].reset_index(drop=True)
#         velRMS_avg = temp_df['accRMS'].mean()
#         velRMS_list.append(velRMS_avg)

#         #graph

#     ax = fig.add_subplot(1,1,1,title='time(x) vs accRMSavg(y)')
#     ax.plot(unique_date,velRMS_list)

#     plt.xticks(rotation=90)

#     fig.savefig(name+'accRMSavg')
#     fig.clf()

# plt.show()

#-------------------------------------------------------------------#

# for name in name_list:

#     df = pd.read_csv(f'{name}.csv')

#     fig.suptitle(f'{name}', fontsize=25)

#         #graph

#     ax = fig.add_subplot(1,1,1,title='freq(x) vs vel(y)')
#     ax.scatter(df.iloc[:,6],df.iloc[:,7],s=0.5)
#     ax.scatter(df.iloc[:,8],df.iloc[:,9],s=0.5)
#     ax.scatter(df.iloc[:,10],df.iloc[:,11],s=0.5)

#     plt.locator_params(axis='x', nbins=30)

#     fig.savefig(name+'freqvelEXPAND')
#     fig.clf()

#-------------------------------------------------------------------#

# for name in name_list:

#     df = pd.read_csv(f'{name}.csv')

#     fig.suptitle(f'{name}', fontsize=25)

#     val_dict = {'1': df.iloc[:,6], '2': df.iloc[:,8], '3': df.iloc[:,10]}

#         #graph

#     ax = fig.add_subplot(1,1,1,title='freqvel')
#     ax.boxplot(val_dict.values())
#     ax.set_xticklabels(val_dict.keys())

#     fig.savefig(name+'freqvelEXPAND')
#     fig.clf()

#-------------------------------------------------------------------#

# for name in name_list:

#     df = pd.read_csv(f'{name}.csv')

#     fig.suptitle(f'{name}', fontsize=25)

#         #graph

#     ax = fig.add_subplot(1,1,1,title='freqvel')
#     ax.boxplot(df.iloc[:,6]+df.iloc[:,8]+df.iloc[:,10])

#     fig.savefig(name+'freqvelEXPAND')
#     fig.clf()

#-----------------------------------------------------------------------------------------------------------------------------#

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numpy.polynomial.polynomial import polyfit
import seaborn as sns
import random as rd

fig = plt.figure()
fig.tight_layout()
fig.set_size_inches(5,5)

    # generate normal graph

os.chdir("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Machine")

name_list = [
    'E3CE', 'E3F7', 'E484', 'E3D1', 'E407', 'E40E', 'E3FC', 'E3DF', 'E3DC', 'E410',
    'E3EC', 'E3E1', 'E3D5', 'E3E6', 'E413', 'E470', 'E401', 'E47E', 'E3D4', 'E466',
    'E412', 'E408', 'E402', 'E3F8', 'E480', 'E3F0', 'E473', 'E3D8', 'E417', 'E3FA',
    'E416', 'E3F3', 'E476', 'E47B', 'E3F1', 'E414', 'E3FB', 'E3FD'
]

for name in name_list:

    print(name)

    df = pd.read_csv(f'{name}.csv')

    newpath = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generated Images\\Test Images"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # combine vel freq #
    freq_vel_list = [df['freq1vel'],df['freq2vel'],df['freq3vel']]
    combined_freq_vel = pd.concat(freq_vel_list, axis='rows', ignore_index=True)

    vel_list = [df['vel1'],df['vel2'],df['vel3']]
    combined_vel = pd.concat(vel_list, axis='rows', ignore_index=True)

    # plot1 #
    ax = fig.add_subplot(1,1,1)
    ax.scatter(combined_freq_vel,combined_vel,s=0.5,color='black')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plt.grid(False)
    plt.axis('off')

    fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generated Images\\Test Images\\{name}.png")
    fig.clf()

#-----------------------------------------------------------------------------------------------------------------------------#