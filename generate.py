import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numpy.polynomial.polynomial import polyfit
import seaborn as sns
import random as rd

os.chdir("E:\Gun's stuff\Machine Learning and Deep Learning\Machine Learning\Ajinomoto Project\E412(DECON2)")

df = pd.read_csv(f'E412.csv')
df_gear = pd.read_csv(f'GEARDATACLEANED.csv')
df_motor = pd.read_csv(f'MOTORDATACLEANED.csv')

fig = plt.figure()
fig.tight_layout()
fig.set_size_inches(5,5)

#-----------------------------------------------------------------------------------------------------------------------------#

#     # generate normal graph

# for i in range(300):
#     print(i)
#     random_df = df[['freq1acc','acc1','freq2acc','acc2','accRMS','freq1vel','vel1','freq2vel','vel2','freq3vel','vel3','velRMS']].apply(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.5)))

#     # combine vel freq #
#     freq_vel_list = [random_df['freq1vel'],random_df['freq2vel'],random_df['freq3vel']]
#     combined_freq_vel = pd.concat(freq_vel_list, axis='rows', ignore_index=True)

#     vel_list = [random_df['vel1'],random_df['vel2'],random_df['vel3']]
#     combined_vel = pd.concat(vel_list, axis='rows', ignore_index=True)

#     # plot1 #
#     ax1 = fig.add_subplot(1,1,1)
#     ax1.scatter(combined_freq_vel,combined_vel,s=0.5,color='black')
#     ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

#     fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input\\Normal\\normal{i}.png")
#     fig.clf()

#-----------------------------------------------------------------------------------------------------------------------------#

#     # generate broken bearing coupling graph

# for i in range(300):
#     print(i)
#     random_df_motor = df_motor[['freq1acc','acc1','freq2acc','acc2','accRMS','freq1vel','vel1','freq2vel','vel2','freq3vel','vel3','velRMS']].apply(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.5)))

#     # combine vel freq #
#     freq_vel_list = [random_df_motor['freq1vel'],random_df_motor['freq2vel'],random_df_motor['freq3vel']]
#     combined_freq_vel = pd.concat(freq_vel_list, axis='rows', ignore_index=True)

#     vel_list = [random_df_motor['vel1'],random_df_motor['vel2'],random_df_motor['vel3']]
#     combined_vel = pd.concat(vel_list, axis='rows', ignore_index=True)

#     # plot1 #
#     ax1 = fig.add_subplot(1,1,1)
#     ax1.scatter(combined_freq_vel,combined_vel,s=0.5,color='black')
#     ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

#     fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input\\Broken (Coupling Bearing)\\brokenCB{i}.png")
#     fig.clf()

#-----------------------------------------------------------------------------------------------------------------------------#

#     # generate broken bearing coupling with bigger range graph

# for i in range(50):
#     print(i)
#     random_df_motor = df_motor[['freq1acc','acc1','freq2acc','acc2','accRMS','freq1vel','vel1','freq2vel','vel2','freq3vel','vel3','velRMS']].apply(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.5)))

#     # combine vel freq #
#     freq_vel_list = [random_df_motor['freq1vel'],random_df_motor['freq2vel'],random_df_motor['freq3vel']]
#     combined_freq_vel = pd.concat(freq_vel_list, axis='rows', ignore_index=True)

#     vel_list = [random_df_motor['vel1'],random_df_motor['vel2'],random_df_motor['vel3']]
#     combined_vel = pd.concat(vel_list, axis='rows', ignore_index=True)

#     # plot1 #
#     ax1 = fig.add_subplot(1,1,1)
#     ax1.scatter(combined_freq_vel,combined_vel,s=0.5,color='black')
#     ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

#     fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Classification Test 2\\Broken Coupling Bearing\\brokenCB{i}.png")
#     fig.clf()

#     # generate normal with bigger range graph

# for i in range(50):
#     print(i)
#     random_df = df[['freq1acc','acc1','freq2acc','acc2','accRMS','freq1vel','vel1','freq2vel','vel2','freq3vel','vel3','velRMS']].apply(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.5)))

#     # combine vel freq #
#     freq_vel_list = [random_df['freq1vel'],random_df['freq2vel'],random_df['freq3vel']]
#     combined_freq_vel = pd.concat(freq_vel_list, axis='rows', ignore_index=True)

#     vel_list = [random_df['vel1'],random_df['vel2'],random_df['vel3']]
#     combined_vel = pd.concat(vel_list, axis='rows', ignore_index=True)

#     # plot1 #
#     ax1 = fig.add_subplot(1,1,1)
#     ax1.scatter(combined_freq_vel,combined_vel,s=0.5,color='black')
#     ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

#     fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Classification Test 2\\Normal\\normal{i}.png")
#     fig.clf()

#-----------------------------------------------------------------------------------------------------------------------------#

# random_df = df[['freq1acc','acc1','freq2acc','acc2','accRMS','freq1vel','vel1','freq2vel','vel2','freq3vel','vel3','velRMS']].apply(lambda x: x + (x*rd.randint(-1,1)*0.15))
# random_df_gear = df_gear[['freq1acc','acc1','freq2acc','acc2','accRMS','freq1vel','vel1','freq2vel','vel2','freq3vel','vel3','velRMS']].apply(lambda x: x + (x*rd.randint(-1,1)*0.15))
# random_df_motor = df_motor[['freq1acc','acc1','freq2acc','acc2','accRMS','freq1vel','vel1','freq2vel','vel2','freq3vel','vel3','velRMS']].apply(lambda x: x + (x*rd.randint(-1,1)*0.15))

# fig = plt.figure()
# fig.tight_layout()
# fig.canvas.manager.full_screen_toggle()
# fig.set_size_inches(18, 10)
# fig.suptitle(f'E412(DECON2)', fontsize=25)

# # combine vel freq #
# freq_vel_list = [df['freq1vel'],df['freq2vel'],df['freq3vel']]
# combined_freq_vel = pd.concat(freq_vel_list, axis='rows', ignore_index=True)

# gear_freq_vel_list = [df_gear['freq1vel'],df_gear['freq2vel'],df_gear['freq3vel']]
# gear_combined_freq_vel = pd.concat(gear_freq_vel_list, axis='rows', ignore_index=True)

# motor_freq_vel_list = [df_motor['freq1vel'],df_motor['freq2vel'],df_motor['freq3vel']]
# motor_combined_freq_vel = pd.concat(motor_freq_vel_list, axis='rows', ignore_index=True)

# vel_list = [df['vel1'],df['vel2'],df['vel3']]
# combined_vel = pd.concat(vel_list, axis='rows', ignore_index=True)

# gear_vel_list = [df_gear['vel1'],df_gear['vel2'],df_gear['vel3']]
# gear_combined_vel = pd.concat(gear_vel_list, axis='rows', ignore_index=True)

# motor_vel_list = [df_motor['vel1'],df_motor['vel2'],df_motor['vel3']]
# motor_combined_vel = pd.concat(motor_vel_list, axis='rows', ignore_index=True)

# # plot1 #
# ax1 = fig.add_subplot(2,3,1,title='normal freq vs vel')
# ax1.scatter(combined_freq_vel,combined_vel,s=0.5,color='black')
# ax1.set_xticks(np.arange(min(combined_freq_vel), max(combined_freq_vel)+max(combined_freq_vel)/15, max(combined_freq_vel)/30))
# ax1.set_xticklabels(ax1.get_xticks(), rotation = 90)
# ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# rpm = 1460
# Hz = rpm/60
# ax1.axvline(0, alpha=0.2, color='red', label=f'machine freq({Hz:.2f})')
# for i in range(round(max(combined_freq_vel)/Hz)):
#     ax1.axvline(Hz*(i+1), alpha=0.2, color='red')
# ax1.legend()

# # plot2 #
# ax2 = fig.add_subplot(2,3,2,title='gear freq vs vel')
# ax2.scatter(gear_combined_freq_vel,gear_combined_vel,s=0.5,color='black')
# ax2.set_xticks(np.arange(min(gear_combined_freq_vel), max(gear_combined_freq_vel)+max(gear_combined_freq_vel)/15, max(gear_combined_freq_vel)/30))
# ax2.set_xticklabels(ax2.get_xticks(), rotation = 90)
# ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# rpm = 1460
# Hz = rpm/60
# ax2.axvline(0, alpha=0.2, color='red', label=f'machine freq({Hz:.2f})')
# for i in range(round(max(gear_combined_freq_vel)/Hz)):
#     ax2.axvline(Hz*(i+1), alpha=0.2, color='red')
# ax2.legend()

# # plot3 #
# ax3 = fig.add_subplot(2,3,3,title='motor freq vs vel')
# ax3.scatter(motor_combined_freq_vel,motor_combined_vel,s=0.5,color='black')
# ax3.set_xticks(np.arange(min(motor_combined_freq_vel), max(motor_combined_freq_vel)+max(motor_combined_freq_vel)/15, max(motor_combined_freq_vel)/30))
# ax3.set_xticklabels(ax3.get_xticks(), rotation = 90)
# ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# rpm = 1460
# Hz = rpm/60
# ax3.axvline(0, alpha=0.2, color='red', label=f'machine freq({Hz:.2f})')
# for i in range(round(max(motor_combined_freq_vel)/Hz)):
#     ax3.axvline(Hz*(i+1), alpha=0.2, color='red')
# ax3.legend()

# # combine vel freq #
# random_freq_vel_list = [random_df['freq1vel'],random_df['freq2vel'],random_df['freq3vel']]
# random_combined_freq_vel = pd.concat(random_freq_vel_list, axis='rows', ignore_index=True)

# random_gear_freq_vel_list = [random_df_gear['freq1vel'],random_df_gear['freq2vel'],random_df_gear['freq3vel']]
# random_gear_combined_freq_vel = pd.concat(random_gear_freq_vel_list, axis='rows', ignore_index=True)

# random_motor_freq_vel_list = [random_df_motor['freq1vel'],random_df_motor['freq2vel'],random_df_motor['freq3vel']]
# random_motor_combined_freq_vel = pd.concat(random_motor_freq_vel_list, axis='rows', ignore_index=True)

# random_vel_list = [random_df['vel1'],random_df['vel2'],random_df['vel3']]
# random_combined_vel = pd.concat(random_vel_list, axis='rows', ignore_index=True)

# random_gear_vel_list = [random_df_gear['vel1'],random_df_gear['vel2'],random_df_gear['vel3']]
# random_gear_combined_vel = pd.concat(random_gear_vel_list, axis='rows', ignore_index=True)

# random_motor_vel_list = [random_df_motor['vel1'],random_df_motor['vel2'],random_df_motor['vel3']]
# random_motor_combined_vel = pd.concat(random_motor_vel_list, axis='rows', ignore_index=True)

# # plot4 #
# ax4 = fig.add_subplot(2,3,4,title='random normal freq vs vel')
# ax4.scatter(random_combined_freq_vel,random_combined_vel,s=0.5,color='black')
# ax4.set_xticks(np.arange(min(random_combined_freq_vel), max(random_combined_freq_vel)+max(random_combined_freq_vel)/15, max(random_combined_freq_vel)/30))
# ax4.set_xticklabels(ax4.get_xticks(), rotation = 90)
# ax4.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# rpm = 1460
# Hz = rpm/60
# ax4.axvline(0, alpha=0.2, color='red', label=f'machine freq({Hz:.2f})')
# for i in range(round(max(random_combined_freq_vel)/Hz)):
#     ax4.axvline(Hz*(i+1), alpha=0.2, color='red')
# ax4.legend()

# # plot5 #
# ax5 = fig.add_subplot(2,3,5,title='random gear freq vs vel')
# ax5.scatter(random_gear_combined_freq_vel,random_gear_combined_vel,s=0.5,color='black')
# ax5.set_xticks(np.arange(min(random_gear_combined_freq_vel), max(random_gear_combined_freq_vel)+max(random_gear_combined_freq_vel)/15, max(random_gear_combined_freq_vel)/30))
# ax5.set_xticklabels(ax5.get_xticks(), rotation = 90)
# ax5.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# rpm = 1460
# Hz = rpm/60
# ax5.axvline(0, alpha=0.2, color='red', label=f'machine freq({Hz:.2f})')
# for i in range(round(max(random_gear_combined_freq_vel)/Hz)):
#     ax5.axvline(Hz*(i+1), alpha=0.2, color='red')
# ax5.legend()

# # plot6 #
# ax6 = fig.add_subplot(2,3,6,title='random motor freq vs vel')
# ax6.scatter(random_motor_combined_freq_vel,random_motor_combined_vel,s=0.5,color='black')
# ax6.set_xticks(np.arange(min(random_motor_combined_freq_vel), max(random_motor_combined_freq_vel)+max(random_motor_combined_freq_vel)/15, max(random_motor_combined_freq_vel)/30))
# ax6.set_xticklabels(ax6.get_xticks(), rotation = 90)
# ax6.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# rpm = 1460
# Hz = rpm/60
# ax6.axvline(0, alpha=0.2, color='red', label=f'machine freq({Hz:.2f})')
# for i in range(round(max(random_motor_combined_freq_vel)/Hz)):
#     ax6.axvline(Hz*(i+1), alpha=0.2, color='red')
# ax6.legend()

# plt.show()

#-----------------------------------------------------------------------------------------------------------------------------#

#     # generate normal graph

# os.chdir("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Machine")

# name_list = [
#     'E3CE', 'E3F7', 'E484', 'E3D1', 'E407', 'E40E', 'E3FC', 'E3DF', 'E3DC', 'E410',
#     'E3EC', 'E3E1', 'E3D5', 'E3E6', 'E413', 'E470', 'E401', 'E47E', 'E3D4', 'E466',
#     'E412', 'E408', 'E402', 'E3F8', 'E480', 'E3F0', 'E473', 'E3D8', 'E417', 'E3FA',
#     'E416', 'E3F3', 'E476', 'E47B', 'E3F1', 'E414', 'E3FB', 'E3FD'
# ]

# for name in name_list:

#     for i in range(300):

#         print(i)

#         df = pd.read_csv(f'{name}.csv')

#         newpath = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generated Images\\{name}\\"
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)

#         df['freq1vel'] = df['freq1vel'].transform(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.05)))
#         df['vel1'] = df['vel1'].transform(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.05)))
#         df['freq2vel'] = df['freq2vel'].transform(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.05)))
#         df['vel2'] = df['vel2'].transform(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.05)))
#         df['freq3vel'] = df['freq3vel'].transform(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.05)))
#         df['vel3'] = df['vel3'].transform(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.05)))

#         # combine vel freq #
#         freq_vel_list = [df['freq1vel'],df['freq2vel'],df['freq3vel']]
#         combined_freq_vel = pd.concat(freq_vel_list, axis='rows', ignore_index=True)

#         vel_list = [df['vel1'],df['vel2'],df['vel3']]
#         combined_vel = pd.concat(vel_list, axis='rows', ignore_index=True)

#         # plot1 #
#         ax1 = fig.add_subplot(1,1,1)
#         ax1.scatter(combined_freq_vel,combined_vel,s=0.5,color='black')
#         ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

#         fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generated Images\\{name}\\normal{i}.png")
#         fig.clf()

#-----------------------------------------------------------------------------------------------------------------------------#

#     # generate normal graph

# os.chdir("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\E412BROKEN(DECON2)")

# for i in range(1000):

#     print(i)

#     df = pd.read_csv(f'MOTORDATACLEANED.csv')

#     newpath = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generated Images\\armature\\"
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)

#     df['freq1vel'] = df['freq1vel'].transform(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.05)))
#     df['vel1'] = df['vel1'].transform(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.05)))
#     df['freq2vel'] = df['freq2vel'].transform(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.05)))
#     df['vel2'] = df['vel2'].transform(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.05)))
#     df['freq3vel'] = df['freq3vel'].transform(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.05)))
#     df['vel3'] = df['vel3'].transform(lambda x: x + (x*rd.randint(-1,1)*rd.uniform(0,0.05)))

#     # combine vel freq #
#     freq_vel_list = [df['freq1vel'],df['freq2vel'],df['freq3vel']]
#     combined_freq_vel = pd.concat(freq_vel_list, axis='rows', ignore_index=True)

#     vel_list = [df['vel1'],df['vel2'],df['vel3']]
#     combined_vel = pd.concat(vel_list, axis='rows', ignore_index=True)

#     # plot1 #
#     ax1 = fig.add_subplot(1,1,1)
#     ax1.scatter(combined_freq_vel,combined_vel,s=0.5,color='black')
#     ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

#     fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generated Images\\Armature\\Armature{i}.png")
#     fig.clf()

#-----------------------------------------------------------------------------------------------------------------------------#

#     # generate normal graph

# os.chdir("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\E412BROKEN(DECON2)")

# column_list = ['freq1vel','freq2vel','freq3vel','vel1','vel2','vel3']

# for i in range(10):

#     print(i)

#     df = pd.read_csv(f'MOTORDATACLEANED.csv')

#     newpath = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generated Images\\Armature\\"
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)

#     for column in column_list:
#         dice = np.random.choice([1,2],p=[0.9,0.1])
#         if dice == 1:
#             if column.find('freq1vel') == 0:
#                 dice = rd.choice([0,1])
#                 if dice == 0:
#                     pass
#                 else:
#                     df[column] = df[column].transform(lambda x: x * rd.uniform(0,0.05))
#             elif column.find('freq2vel') == 0:
#                 dice = rd.choice([0,1])
#                 if dice == 0:
#                     pass
#                 else:
#                     df[column] = df[column].transform(lambda x: x * rd.uniform(0,0.05))
#             elif column.find('freq3vel') == 0:
#                 dice = rd.choice([0,1])
#                 if dice == 0:
#                     pass
#                 else:
#                     df[column] = df[column].transform(lambda x: x * rd.uniform(0,0.05))
#             elif column.find('vel1') == 0:
#                 dice = rd.choice([0,1,2])
#                 if dice == 0:
#                     pass
#                 elif dice == 1:
#                     df[column] = df[column].transform(lambda x: np.floor(x) * rd.uniform(0,0.5))
#                 else:
#                     df[column] = df[column].transform(lambda x: np.ceil(x) * rd.uniform(0,0.5))
#             elif column.find('vel2') == 0:
#                 dice = rd.choice([0,1,2])
#                 if dice == 0:
#                     pass
#                 elif dice == 1:
#                     df[column] = df[column].transform(lambda x: np.floor(x) * rd.uniform(0,0.5))
#                 else:
#                     df[column] = df[column].transform(lambda x: np.ceil(x) * rd.uniform(0,0.5))
#             elif column.find('vel3') == 0:
#                 dice = rd.choice([0,1,2])
#                 if dice == 0:
#                     pass
#                 elif dice == 1:
#                     df[column] = df[column].transform(lambda x: np.floor(x) * rd.uniform(0,0.5))
#                 else:
#                     df[column] = df[column].transform(lambda x: np.ceil(x) * rd.uniform(0,0.5))
#             else:
#                 pass
#         else:
#             df[column] = df[column].transform(lambda x: x * rd.uniform(0,0.2))

#     # combine vel freq #
#     freq_vel_list = [df['freq1vel'],df['freq2vel'],df['freq3vel']]
#     combined_freq_vel = pd.concat(freq_vel_list, axis='rows', ignore_index=True)

#     vel_list = [df['vel1'],df['vel2'],df['vel3']]
#     combined_vel = pd.concat(vel_list, axis='rows', ignore_index=True)

#     # plot1 #
#     ax = fig.add_subplot(1,1,1)
#     ax.scatter(combined_freq_vel,combined_vel,s=0.5,color='black')
#     ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

#     fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generated Images\\Armature\\Armature{i}.png")
#     fig.clf()

#-----------------------------------------------------------------------------------------------------------------------------#
