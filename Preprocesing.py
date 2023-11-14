# !curl -o dataset.zip https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip
# !unzip '/content/drive/MyDrive/Copy of BCICIV_2a_gdf.zip' -d ./dataset
# !pip install -q mne
import mne
from mne.preprocessing import ICA
from mne.datasets import eegbci
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder




dataset_directory = './dataset/'
subject_files = [f'A0{i}T.gdf' for i in range(1, 3)]
raws = []


for file in subject_files:
    file_path = dataset_directory + file
    raw = mne.io.read_raw_gdf(file_path, preload=True, stim_channel='auto')
    raws.append(raw)

concatenated_raw = mne.concatenate_raws(raws)


channel_names_mapping = {
    'EEG-0': 'FC3',
    'EEG-1': 'FC1',
    'EEG-2': 'FCz',
    'EEG-3': 'FC2',
    'EEG-4': 'FC4',
    'EEG-5': 'C5',
    'EEG-6': 'C1',
    'EEG-7': 'C2',  # Often corresponds to T7 in the extended 10-20 system
    'EEG-8': 'C6',
    'EEG-9': 'CP3',
    'EEG-10': 'CP1',
    'EEG-11': 'CPz',  # Often corresponds to T8 in the extended 10-20 system
    'EEG-12': 'CP2',  # Often corresponds to P7 in the extended 10-20 system
    'EEG-13': 'CP4',
    'EEG-14': 'P1',
    'EEG-15': 'P2',
    'EEG-16': 'POz',  # Often corresponds to P8 in the extended 10-20 system
    'EEG-C3': 'C3',  # C3 does not need to be remapped
    'EEG-Cz': 'Cz',  # Cz does not need to be remapped
    'EEG-C4': 'C4',  # C4 does not need to be remapped
    'EEG-Pz': 'Pz',  # Pz does not need to be remapped
    'EEG-Fz': 'Fz',  # Fz does not need to be remapped

}
# The following electrodes have overlapping positions, which causes problems during visualization:
# Fc3, CPZ, TOz, EOG-0, EOG-1, EOG-2
# Apply the renaming
concatenated_raw .rename_channels(channel_names_mapping)
concatenated_raw .set_channel_types({
    'EOG-left': 'eog',
    'EOG-central': 'eog',
    'EOG-right': 'eog'
})

# Set the standard montage
montage = mne.channels.make_standard_montage('standard_1020')
concatenated_raw .set_montage(montage, on_missing='ignore')

# Verify the montage application
concatenated_raw .plot_sensors(show_names=True)
concatenated_raw .get_data


# Plot the raw data for a duration of 5 seconds starting from 100 seconds
concatenated_raw.plot(duration=5, start=250*5)

# Compute and plot the PSD of the raw data
concatenated_raw.plot_psd(fmin=0.5, fmax=120, spatial_colors=True)

# # Filter the data between 7 and 32 Hz
raw_fil = concatenated_raw.copy().filter(7., 32., fir_design='firwin', skip_by_annotation='edge')

# Visualize the filtered data for a duration of 5 second|s starting from 100 seconds
raw_fil.plot(duration=5, start=100)

# # Compute and plot the PSD of the filtered data
raw_fil.plot_psd(fmin=0.5, fmax=120, spatial_colors=True)



ica = ICA(n_components=22, random_state=97, max_iter=800)
ica.fit(raw_fil )
ica.plot_components()

ica.plot_sources(raw_fil)
eog_indices, eog_scores = ica.find_bads_eog(raw_fil, ch_name=['EOG-left', 'EOG-central', 'EOG-right'])
ica.exclude = eog_indices

ica.plot_scores(eog_scores)
ica.plot_properties(raw_fil, picks=eog_indices)
# Define the event_id dictionary only for task-related events
event_id = {
    'Cue onset left': 7,
    'Cue onset right': 8
}

# Load events
events, event_id_map = mne.events_from_annotations(raw_fil)
# Filter out the events you're interested in
task_events = events[(events[:, 2] == event_id['Cue onset left']) | (events[:, 2] == event_id['Cue onset right'])]
print(task_events.shape)

epochs = mne.Epochs(raw_fil, task_events, event_id=event_id, tmin=-0.5, tmax=4,
                    proj=True, baseline=None, preload=True,
                    event_repeated='merge')
# Define your baseline period
baseline = (None, 0)  # This will take the period from the beginning of your epoch to time 0

# Apply baseline correction
epochs.apply_baseline(baseline)

# Visualize the epochs to confirm the baseline correction
epochs.plot_image(combine='mean')
epochs[1].plot(n_epochs=1, block=True)
# Epoch = epochs.copy().set_eeg_reference('average', projection=True)

# Plot the re-referenced data
Epoch[1].plot(n_epochs=1, block=True)

# Extracting data and converting to (samples, features) format
X = Epoch.get_data()
y = Epoch.events[:, -1]  # The column containing the event cod
X_new = X[:,:22,:]
print(np.shape(X_new))



label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
