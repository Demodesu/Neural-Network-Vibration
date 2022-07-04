import tkinter
import PIL
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import random as rd
import sys
import ctypes

class MainApplication(tkinter.Frame):
    def __init__(self, master):
        tkinter.Frame.__init__(self, master)
        self.master = master

        self.status_text = tkinter.StringVar()
        self.status_text.set('None')

        self.status_label = tkinter.Label(self.master,text=self.status_text.get())
        self.status_label.grid(row=0,column=0,padx=10,pady=10)

        self.status_text.set('Select File And Enter Parameters')
        self.status_label.config(text=self.status_text.get())

        self.exit_button = tkinter.Button(
            self.master,
            text='Exit',
            command=exit
        )
        self.exit_button.grid(row=0,column=1,padx=10,pady=10)

        self.restart_button = tkinter.Button(
            self.master,
            text='Restart',
            command=restart
        )
        self.restart_button.grid(row=0,column=2,padx=10,pady=10)

        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        self.machine_names_dict = {
        'E3CE':'E3CE (MAIN A)',
        'E3D1':'E3D1 (DECON A)',
        'E3D4':'E3D4 (VACUUM PCF A)',
        'E3D5':'E3D5 (NEUT B)',
        'E3D8':'E3D8 (VACUUM PCF)',
        'E3DC':'E3DC (NEUT A)',
        'E3DF':'E3DF (NEUT C)',
        'E3E1':'E3E1 (NEUT F)',
        'E3E6':'E3E6 (MVR A)',
        'E3EC':'E3EC (NEUT D)',
        'E3F1':'E3F1 (VACUUM PCF B)',
        'E3F3':'E3F3 (SEED C)',
        'E3F7':'E3F7 (MAIN C)',
        'E3F8':'E3F8 (SDC D)',
        'E3FA':'E3FA (RENSHO RECYCLE B)',
        'E3FB':'E3FB (FEED PUMP TO BOILER)',
        'E3FD':'E3FD (BROTH)',
        'E401':'E401 (MVR B)',
        'E402':'E402 (SDC B)',
        'E407':'E407 (DECON B)',
        'E408':'E408 (SDC A)',
        'E410':'E410 (NEUT E)',
        'E412':'E412 (DECON 2)',
        'E413':'E413 (MVR A)',
        'E414':'E414 (MAIN B)',
        'E416':'E416 (RENSHO 2 RECYCLE)',
        'E417':'E417 (RENSHO RECYCLE A)',
        'E470':'E470 (VACUUM PCF A)',
        'E473':'E473 (FEED PUMP B TO BOILER)',
        'E476':'E476 (SEED B)',
        'E47B':'E47B (SEED A)',
        'E47E':'E47E (MVR B)',
        'E480':'E480 (SDC C)',
        }

        self.threshold = 200
        self.barrier = lambda x: 0 if x > self.threshold else 255

        self.enter_kW = tkinter.Entry(self.master,width=50,borderwidth=2)
        self.enter_kW.grid(row=1,column=0,columnspan=4,padx=10,pady=10)
        self.enter_kW.insert(0, 'Enter Machine kW (Kilowatts)')

        self.enter_RPM = tkinter.Entry(self.master,width=50,borderwidth=2)
        self.enter_RPM.grid(row=2,column=0,columnspan=4,padx=10,pady=10)
        self.enter_RPM.insert(0, 'Enter Machine RPM (Round Per Minute)')

        self.foundation = tkinter.StringVar()
        self.foundation.set('Foundation Type')

        self.foundation_list = ['Soft (Soil, Sand, Etc.)','Heavy (Concrete, Steel, Etc.)']

        self.select_foundation_drop_down = tkinter.OptionMenu(
            self.master,
            self.foundation,
            *self.foundation_list,
        )
        self.select_foundation_drop_down.grid(row=3,column=0,columnspan=4,padx=10,pady=10)

        # SECTION start program #

        self.get_file()

    def get_file(self):

        # SECTION button #

        self.get_file_button = tkinter.Button(
            self.master,
            text='Select File',
            command=self.select_file
        )
        self.get_file_button.grid(row=0,column=3,padx=10,pady=10)
        
    def select_file(self):

        self.machine_kW = int(self.enter_kW.get())
        self.machine_RPM = int(self.enter_RPM.get())
        self.machine_foundation = self.foundation.get()
        if self.machine_foundation == 'Soft (Soil, Sand, Etc.)':
            self.machine_foundation = 'Soft'
        elif self.machine_foundation == 'Heavy (Concrete, Steel, Etc.)':
            self.machine_foundation = 'Heavy'
        else:
            restart()

        if self.machine_kW <= 15:
            self.machine_class = 'I'
            self.class_velocity_limit = [0,0.71,1.8,4.5]
        elif 15 < self.machine_kW <= 75:
            self.machine_class = 'II'
            self.class_velocity_limit = [0,1.12,2.8,7.1]
        else:
            if self.machine_foundation == 'Soft':
                self.machine_class = 'III'
                self.class_velocity_limit = [0,1.8,4.5,11.2]
            else:
                self.machine_class = 'IV'
                self.class_velocity_limit = [0,2.8,7.1,18]

        # SECTION destroy #

        self.get_file_button.destroy()
        self.enter_kW.destroy()
        self.enter_RPM.destroy()
        self.select_foundation_drop_down.destroy()

        # SECTION dataframe #

        filename = tkinter.filedialog.askopenfilenames(
            initialdir=self.dir_path,
            filetypes=((('CSV Files'),('*.csv')),(('All Files'),('*.*')))
            )        

        file_df_list = []
        for files in filename:
            df = pd.read_csv(files,header=None)
            df = df.drop(list(df)[2:5],axis='columns')    
            df = df.iloc[:, [0,5,7,9,11,13,15,17,19,21,23,25,27,31,35]]
            cleaned_df = df.rename({
                0:'name',
                8:'freq1acc',
                10:'acc1', 
                12:'freq2acc', 
                14:'acc2', 
                16:'accRMS', 
                18:'freq1vel', 
                20:'vel1', 
                22:'freq2vel', 
                24:'vel2', 
                26:'freq3vel', 
                28:'vel3', 
                30:'velRMS', 
                34:'temp',
                38:'time'},
                axis='columns'
                )
            file_df_list.append(cleaned_df)
        combined_file_df = pd.concat(file_df_list,ignore_index=True)

        cleaned_df_list = []
        for key in self.machine_names_dict:
            location_of_index = list(combined_file_df['name'].loc[lambda x: x==key].index)
            machine_data = combined_file_df.iloc[location_of_index]
            cleaned_df_list.append(machine_data)

        self.machine_df = pd.concat(cleaned_df_list,ignore_index=True)
        self.machine_df['name'] = self.machine_df['name'].replace(self.machine_names_dict)

        self.machine_names = self.machine_df['name'].unique()

        # SECTION status text #

        self.status_text.set(f'{len(filename)} file(s)')
        self.status_label.config(text=self.status_text.get())

        # SECTION widget #

        self.select_machine_button = tkinter.Button(
            self.master,
            text='Continue',
            command=self.select_machine
        )
        self.select_machine_button.grid(row=0,column=3,padx=10,pady=10)

    def select_machine(self):

        # SECTION destroy #
        
        self.select_machine_button.destroy()

        # SECTION status text #

        self.status_text.set('Select Machine')
        self.status_label.config(text=self.status_text.get())

        # SECTION dropdown menu #

        self.selection = tkinter.StringVar()
        self.selection.set('None')

        self.select_machine_drop_down = tkinter.OptionMenu(
            self.master,
            self.selection,
            *self.machine_names,
        )
        self.select_machine_drop_down.grid(row=0,column=3,padx=10,pady=10)

        # SECTION dropdown button #

        self.select_machine_drop_down_button = tkinter.Button(
            self.master,
            text='Confirm Selected Machine',
            command=self.get_selected_machine_dataframe
        )
        self.select_machine_drop_down_button.grid(row=0,column=4,padx=10,pady=10)

    def generate_condition_graphs(self):

        self.graphs = 9
        fig = plt.figure()
        fig.tight_layout()
        fig.set_size_inches(5,5)

        newpath = self.dir_path + f'\\Condition'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

            # SECTION build dataframe unbalance #

        def generate_spectrum_unbalance():

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

        for i in range(self.graphs):

                # SECTION generate and graph #

            generated_df = generate_spectrum_unbalance()

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
            ax.scatter(freq_df,vel_df,s=0.5,color='black')
            ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            ax.grid(False)
            ax.axis('off')

            print(i)
            fig.savefig(self.dir_path + f"\\Condition\\Unbalance{i}.png")
            fig.clf()

            # SECTION build dataframe misalign #

        def generate_spectrum_misalign():

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

        for i in range(self.graphs):

                # SECTION generate and graph #

            generated_df = generate_spectrum_misalign()

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
            ax.scatter(freq_df,vel_df,s=0.5,color='black')
            ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            ax.grid(False)
            ax.axis('off')

            print(i)
            fig.savefig(self.dir_path + f"\\Condition\\Misalign{i}.png")
            fig.clf()

            # SECTION build dataframe loose #

        def generate_spectrum_loose():

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

        for i in range(self.graphs):

                # SECTION generate and graph #

            generated_df = generate_spectrum_loose()

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
            ax.scatter(freq_df,vel_df,s=0.5,color='black')
            ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            ax.grid(False)
            ax.axis('off')

            print(i)
            fig.savefig(self.dir_path + f"\\Condition\\Loose{i}.png")
            fig.clf()

    def get_selected_machine_dataframe(self):
        self.selected_machine = self.selection.get()

        self.selected_machine_df_index = list(self.machine_df['name'].loc[lambda x: x==self.selected_machine].index)
        self.selected_machine_df = self.machine_df.iloc[self.selected_machine_df_index]

        self.status_text.set(self.selected_machine)
        self.status_label.config(text=self.status_text.get())

        self.select_machine_drop_down.destroy()
        self.select_machine_drop_down_button.destroy()

        self.make_graph()
        self.generate_condition_graphs()

        self.display_graph_button = tkinter.Button(
            self.master,
            text='Display Graph',
            command=self.display_graph
        )
        self.display_graph_button.grid(row=0,column=3,padx=10,pady=10)

    def make_graph(self):
        fig = plt.figure()
        fig.tight_layout()
        fig.set_size_inches(5,5)

        newpath = self.dir_path + f'\\Machine'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        freq_vel_list = [self.selected_machine_df['freq1vel'],self.selected_machine_df['freq2vel'],self.selected_machine_df['freq3vel']]
        combined_freq_vel = pd.concat(freq_vel_list, axis='rows', ignore_index=True)

        vel_list = [self.selected_machine_df['vel1'],self.selected_machine_df['vel2'],self.selected_machine_df['vel3']]
        combined_vel = pd.concat(vel_list, axis='rows', ignore_index=True)

        ax = fig.add_subplot(1,1,1)
        ax.scatter(combined_freq_vel,combined_vel,s=0.5,color='black')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.grid(False)
        plt.axis('off')

        fig.savefig(self.dir_path + f"\\Machine\\Machine_Graph.png")

        self.time_domain_graph()

    def display_graph(self):
        self.display_graph_button.destroy()

        import_image = PIL.Image.open(self.dir_path + f"\\Machine\\Machine_Graph.png")
        import_image = import_image.convert('L').point(self.barrier,mode='1')
        import_image = import_image.resize((250,250))
        self.import_image = PIL.ImageTk.PhotoImage(import_image)
        self.display_import_image_label = tkinter.Label(self.master,image=self.import_image)
        self.display_import_image_label.grid(row=1,column=0,padx=10,pady=10,columnspan=6)

        self.predict_condition_button = tkinter.Button(
            self.master,
            text=f'Predict Condition For {self.selected_machine}',
            command=self.predict_condition
        )
        self.predict_condition_button.grid(row=0,column=4,padx=10,pady=10)

    def predict_condition(self):
        self.predict_condition_button.destroy()
        self.display_import_image_label.destroy()

        self.load_prediction_data()
        self.pair_data()
        self.model()
        self.predictions = self.siamese_model.predict([self.x1_test, self.x2_test])

        self.overall_prediction_mean = []
        self.predictions_mean = []
        for index, value in enumerate(self.predictions):
            self.predictions_mean.append(value)
            if len(self.predictions_mean) == self.graphs:
                mean = sum(self.predictions_mean)/len(self.predictions_mean)
                self.overall_prediction_mean.append(mean)
                self.predictions_mean.clear()

        self.exit_button.grid(row=0,column=1,padx=10,pady=10)
        self.restart_button.grid(row=0,column=2,padx=10,pady=10)

        label = ['Loose','Misalign','Unbalance']
        for index in range(len(self.overall_prediction_mean)):
            percentage = tkinter.Label(self.master,text=f'{label[index]} | {self.overall_prediction_mean[index][0]:.5f}')
            percentage.grid(row=0,column=3+index,padx=10,pady=10)

        self.visualize(self.pair_array,predictions=self.predictions)

        self.import_prediction_image = PIL.Image.open(self.dir_path + f"\\Machine\\Predict_Graph.png")
        self.prediction_image = PIL.ImageTk.PhotoImage(self.import_prediction_image)
        self.display_prediction_image_label = tkinter.Label(self.master,image=self.prediction_image)
        self.display_prediction_image_label.grid(row=1,column=0,padx=10,pady=10,columnspan=5)

        self.import_velocity_limit_image = PIL.Image.open(self.dir_path + f"\\Machine\\VelRMS_graph.png")
        self.velocity_limit_image = PIL.ImageTk.PhotoImage(self.import_velocity_limit_image)
        self.display_velocity_limit_image_label = tkinter.Label(self.master,image=self.velocity_limit_image)
        self.display_velocity_limit_image_label.grid(row=1,column=6,padx=10,pady=10,columnspan=5)

    def load_prediction_data(self):
        condition_img_list = []

        machine_img_list = []

        input_path = self.dir_path

        machine_img_path = f"{input_path}\\Machine\\Machine_Graph.png"
        machine_img = PIL.Image.open(machine_img_path) # load as grayscale
        machine_img = machine_img.convert('L').point(self.barrier,mode='1') # convert to greyscale 
        machine_img = np.asarray(machine_img)
        machine_img = machine_img[::2,::2] #downsampling by 2
        machine_img_list.append(machine_img)

        condition_path = f"{input_path}\\Condition"
        self.condition_names = [f for f in os.listdir(condition_path) if os.path.isfile(os.path.join(condition_path, f))]
        for condition in self.condition_names:
            condition_img_path = f"{condition_path}\\{condition}"
            condition_img = PIL.Image.open(condition_img_path) # load as grayscale
            condition_img = condition_img.convert('L').point(self.barrier,mode='1') # convert to greyscale 
            condition_img = np.asarray(condition_img)
            condition_img = condition_img[::2,::2] #downsampling by 2
            condition_img_list.append(condition_img)            

        self.machine_image_list = machine_img_list
        self.conditon_image_list = condition_img_list

    def pair_data(self):
        pairs = []

        for condition_image in self.conditon_image_list:

            pairs.append([self.machine_image_list[0],condition_image])

        pair_array = np.array(pairs)

        self.pair_array = pair_array

        self.x1_test = self.pair_array[:,0] 
        self.x2_test = self.pair_array[:,1]

    def model(self):

        def euclidean_distance(vects):

            vect1, vect2 = vects
            sum_square = tf.math.reduce_sum(tf.math.square(vect1 - vect2), axis=1, keepdims=True)
            return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

        def loss(margin=1):

            def contrastive_loss(y_true, y_pred):

                square_pred = tf.math.square(y_pred)
                margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
                return tf.math.reduce_mean(
                    (y_true * margin_square) + ((1 - y_true) * (square_pred))
                )

            return contrastive_loss

        input_embedding = tf.keras.layers.Input((250,250,1),name='embedding')

        image_input = tf.keras.layers.BatchNormalization()(input_embedding)

            # SECTION first stack #

        x = tf.keras.layers.Conv2D(filters=2,kernel_size=(1,1),strides=(1,1),activation='tanh')(image_input)
        x = tf.keras.layers.Conv2D(filters=4,kernel_size=(3,3),strides=(1,1),activation='tanh')(x)
        x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)

            # SECTION second stack #

        y = tf.keras.layers.Conv2D(filters=2,kernel_size=(1,1),strides=(1,1),activation='tanh')(image_input)
        y = tf.keras.layers.Conv2D(filters=4,kernel_size=(5,5),strides=(1,1),activation='tanh')(y)
        y = tf.keras.layers.ZeroPadding2D(padding=(2,2))(y)

            # SECTION third stack #

        z = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(image_input)
        z = tf.keras.layers.Conv2D(filters=4,kernel_size=(1,1),strides=(1,1),activation='tanh')(z)

            # SECTION fourth stack #

        w = tf.keras.layers.Conv2D(filters=4,kernel_size=(1,1),strides=(1,1),activation='tanh')(image_input)

            # SECTION concatenate #

        concatted = tf.keras.layers.Concatenate()([x,y,z,w])

            # SECTION dense #

        dense = tf.keras.layers.BatchNormalization()(concatted)
        dense = tf.keras.layers.Flatten()(dense)
        dense = tf.keras.layers.Dense(6,activation='tanh')(dense) # generic features
        dense = tf.keras.layers.BatchNormalization()(dense)
        dense = tf.keras.layers.Dense(3,activation='tanh')(dense) # down to 3 classes 

        self.embedding_network = tf.keras.Model(inputs=input_embedding,outputs=dense,name='embedding')

        input_siamese_1 = tf.keras.layers.Input((250,250,1)) 
        input_siamese_2 = tf.keras.layers.Input((250,250,1))

        tower_1 = self.embedding_network(input_siamese_1)
        tower_2 = self.embedding_network(input_siamese_2)

        merge_layer = tf.keras.layers.Lambda(euclidean_distance)([tower_1,tower_2])
        normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
        output_layer = tf.keras.layers.Dense(1,activation='sigmoid')(normal_layer)

        self.siamese_model = tf.keras.Model(inputs=[input_siamese_1,input_siamese_2],outputs=output_layer,name='siamese')

        margin = 1

        self.siamese_model.compile(
            loss=loss(margin=margin),
            optimizer=tf.keras.optimizers.Adadelta(0.001),
            metrics=['accuracy']
        )

        self.siamese_model.load_weights(self.dir_path + f"\\Similarity Model Weight\\Siamese_Weights.h5")

    def visualize(self,pairs,predictions):

        num_col = 3
        num_row = 9

        fig, axes = plt.subplots(num_row, num_col, figsize=(9,9))
        for index in range(num_col*num_row):

            ax = axes[index // num_col, index % num_col]

            ax.imshow(tf.concat([pairs[index][0], pairs[index][1]], axis=1), cmap="gray")
            ax.set_axis_off()

            ax.set_title("{:.5f} | {}".format(predictions[index][0],self.condition_names[index].replace('.png','')),fontsize=5)        
                
        plt.tight_layout(rect=(0, 0, 1, 1),w_pad=0.0)
        fig.savefig(self.dir_path + f"\\Machine\\Predict_Graph.png")

    def time_domain_graph(self):
        velRMS_df = self.selected_machine_df['velRMS']

        fig = plt.figure()
        fig.tight_layout()
        fig.set_size_inches(5,5)

        ax = fig.add_subplot(1,1,1)
        ax.set_title(f'{self.selected_machine} Machine Class {self.machine_class} Time Domain',fontsize=10)            
        ax.plot(velRMS_df,color='black')

        color_list = ['blue','green','yellow','red']

        for index, limit in enumerate(self.class_velocity_limit):
            ax.axhline(y=limit, color=color_list[index], linestyle='-')

        fig.savefig(self.dir_path + f"\\Machine\\VelRMS_graph.png")

user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def start():
    global root
    root = tkinter.Tk()
    root.attributes('-fullscreen',True)
    MainApplication(root)
    root.mainloop()

def exit():
    root.destroy()
    sys.exit()

if __name__ == '__main__':
    def restart():
        root.destroy()
        start()
    start()
sys.exit()