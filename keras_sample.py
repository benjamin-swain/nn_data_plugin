from keras.layers import Input, Dense
from keras.models import Model, model_from_json
from keras.optimizers import Adam
import numpy as np
from pathlib import Path
import copy
import pandas as pd

load_from_file = 0
enable_normalize_inputs = 0

def stub_nans_data(x):
    # NaNs will appear in the data when either you or the opponent are demolished
    if np.isnan(x[20]):
        # Stub opponent vars
        x[14:16] = 0
        x[16:17] = -800
        x[17:27] = 0
    if np.isnan(x[31]):
        # Stub ball vars
        x[27:36] = 0
    return x

def get_pregame_rows(data):
    pregame_rows = []
    cnt = 0
    for i, row in enumerate(data):
        cnt += 1
        pregame_rows.append(i)
        if abs(row[7]) > 1.0:
            break
    # Delete all pregame rows
    return pregame_rows

if load_from_file:
    all_data = np.loadtxt('all_data_save.txt', dtype=int)
    print('Loaded data')
    all_labels_float = np.loadtxt('all_labels_float_save.txt', dtype=int)
    print('Loaded label floats')
    all_labels_bool = np.loadtxt('all_labels_bool_save.txt', dtype=int)
    print('Loaded label bools')
else:
    pathlist = Path('sample_data').glob('**/*.txt')
    files = []
    for path in pathlist:
         path_in_str = str(path)
         print(path_in_str)
         # Make a list of all the txt files in ...\sample_data\
         files.append(path_in_str)


    # Data used to train the neural network in this keras sample:

    # Inputs:

    # 0 my_x
    # 1 my_y
    # 2 my_z
    # 3 my_rotx
    # 4 my_roty
    # 5 my_rotz
    # 6 my_vx
    # 7 my_vy
    # 8 my_vz
    # 9 my_avx
    # 10 my_avy
    # 11 my_avz
    # 12 my_supersonic
    # 13 my_boostamount
    # 14 opponent0_x
    # 15 opponent0_y
    # 16 opponent0_z
    # 17 opponent0_rotx
    # 18 opponent0_roty
    # 19 opponent0_rotz
    # 20 opponent0_vx
    # 21 opponent0_vy
    # 22 opponent0_vz
    # 23 opponent0_avx
    # 24 opponent0_avy
    # 25 opponent0_avz
    # 26 opponent0_supersonic
    # 27 ball_x
    # 28 ball_y
    # 29 ball_z
    # 30 ball_vx
    # 31 ball_vy
    # 32 ball_vz
    # 33 ball_avx
    # 34 ball_avy
    # 35 ball_avz

    # Outputs:

    # 0 my_throttle (float)
    # 1 my_steer (float)
    # 2 my_pitch (float)
    # 3 my_roll (float)
    # 4 my_jump (bool)
    # 5 my_activateboost (bool)
    # 6 my_handbrake (bool)


    for file in files:
        df = pd.read_csv(file, sep=' ')
        text = np.array(df)
        print('appending ' + file)
        # Extract the inputs/data from txt files
        data = text[:, 4:17]
        data = np.append(data, text[:, 26:27], 1)
        data = np.append(data, text[:, 30:43], 1)
        data = np.append(data, text[:, 53:62], 1)

        # Extract the float outputs from txt files
        labels_float = text[:, 17:20]
        labels_float = np.append(labels_float, text[:, 21:22], 1)

        # Extract the bool outputs from txt files
        labels_bool = text[:, 22:25]

        team = text[:, 0:1]

        # Remove rows from the data before the match started
        # These rows were recorded because the first kickoff in a match has inconsistent times occasionally
        # The rows are determined to be pre-game until y velocity is above 0.
        pregame_rows = get_pregame_rows(data)
        print('file')
        print(file)
        print('pregame rows deleted')
        print(len(pregame_rows))
        data = np.delete(data, pregame_rows, 0)
        labels_float = np.delete(labels_float, pregame_rows, 0)
        labels_bool = np.delete(labels_bool, pregame_rows, 0)
        team = np.delete(team, pregame_rows, 0)

        if not 'all_data' in dir():
            all_data = data
            all_labels_float = labels_float
            all_labels_bool = labels_bool
            all_teams = team
        else:
            all_data = np.append(all_data, data, 0)
            all_labels_float = np.append(all_labels_float, labels_float, 0)
            all_labels_bool = np.append(all_labels_bool, labels_bool, 0)
            all_teams = np.append(all_teams, team, 0)


    # Data from all txt files in ...\sample_data\ have been appended to one numpy array, all_data
    print(all_data.shape)


    # Get a version of all_data with all NaN rows removed - to be used for estimating means and standard deviations
    non_nan_data = all_data[~np.isnan(all_data).any(axis=1)]

    # Calculate mean & std
    data_mean = non_nan_data.mean(axis=0)
    data_std = non_nan_data.std(axis=0)
    # Save mean and std in case they need to be used to normalize inputs while running the neural network in-game
    np.savetxt('data_means.txt', data_mean, fmt='%-10.15f')
    np.savetxt('data_stds.txt', data_std, fmt='%-10.15f')
    print('mean')
    print(data_mean)
    print('std')
    print(data_std)

    # Handle NaN values in the data
    # Your vars will be NaN when you are demolished. Opponent vars will be NaN when they are demolished.
    im_demolished = []
    for i,row in enumerate(all_data):
        if np.isnan(row[0]):
            # store rows to be deleted (when I'm demolished)
            im_demolished.append(i)
        row_copy = copy.deepcopy(row)
        # Stub opponent position (under map) when they are demolished
        fixed_row = stub_nans_data(row_copy)
        all_data[i] = fixed_row

    # Delete all rows where I'm demolished
    all_data = np.delete(all_data, im_demolished, 0)
    all_labels_float = np.delete(all_labels_float, im_demolished, 0)
    all_labels_bool = np.delete(all_labels_bool, im_demolished, 0)

    # Normalize inputs
    if enable_normalize_inputs:
        for i, row in enumerate(all_data):
            # x = (x - mean(x)) / std(x)
            all_data[i] = (all_data[i] - data_mean) / data_std

    print('any nans?')
    any_nans = np.isnan(all_data).any()
    print(any_nans)
    if any_nans:
        print(np.argwhere(np.isnan(all_data)))

    # Save the pre-processed data to a file in case user wants to skip this process later using load_from_file = 1
    np.savetxt('all_data_save.txt', all_data, fmt='%-10.15f')
    np.savetxt('all_labels_float_save.txt', all_labels_float, fmt='%-10.15f')
    np.savetxt('all_labels_bool_save.txt', all_labels_bool, fmt='%-10.15f')


print(all_data.shape)


num_inputs = len(all_data[0])

num_output_floats = 4
num_output_bools = 3

input_vector = Input(shape=(num_inputs,))

# There are 2 output layers - one for floats and one for booleans
input_layer = Dense(32, activation='tanh')(input_vector)
hidden_layer = Dense(32, activation='tanh')(input_layer)
output_layer = Dense(num_output_floats, activation='tanh')(hidden_layer)
output_layer2 = Dense(num_output_bools, activation='softmax')(hidden_layer)

model = Model(inputs=[input_vector], outputs=[output_layer, output_layer2])
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

model.fit(all_data, [all_labels_float, all_labels_bool], epochs=12, batch_size=200, validation_split=0.25)

output = model.predict(np.array([all_data[0]]))

# evaluate the model
# scores = model.evaluate(all_data, labels)

print('Test output')
print(output)

# Save the model and weights to a file to be used by another app which runs in-game to control the vehicle
model.save('keras_model.h5', include_optimizer=False)
print("Saved model to disk")



