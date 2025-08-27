import pandas as pd
import numpy as np
import wfdb
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from scipy.signal import resample

data = 'dataset/physionet.org/files/mitdb/1.0.0/'

patients = ['100','101','102','103','104','105','106','107',
            '108','109','111','112','113','114','115','116',
            '117','118','119','121','122','123','124','200',
            '201','202','203','205','207','208','209','210',
            '212','213','214','215','217','219','220','221',
            '222','223','228','230','231','232','233','234']

AAMI = ['N','L','R','B','A','a','j','S','V','r','E','F','f','/','Q','?']

dic = {
    'N': ['N','L','R','B'],
    'S': ['A','a','j','S','e','j','n'],
    'V': ['V','r','E'],
    'F': ['F'],
    'Q': ['Q','?','f','/'],
}

old_fs = 360
new_fs = 128
insize_sec = 5
insize = new_fs * insize_sec

y = []
beat_l2 = []

for num in patients:
    print('Processing record number', num)
    record = wfdb.rdrecord(data + num, smooth_frames=True)
    l2 = preprocessing.scale(np.nan_to_num(record.p_signal[:,0]))
    
    num_samples_new = int(len(l2) * (new_fs / old_fs))
    l2_resampled = resample(l2, num_samples_new)

    ann = wfdb.rdann(data + num, extension='atr')
    
    for symbol, peak in zip(ann.symbol, ann.sample):
        if symbol in AAMI:
            peak_resampled = int(peak * (new_fs / old_fs))
            start, end = peak_resampled - insize // 2, peak_resampled + insize // 2
            start = max(start, 0)
            end = min(end, len(l2_resampled))
            
            if (end - start) > 0:
                for cl, an in dic.items():
                    if symbol in an:
                        beat = l2_resampled[start:end]
                        if len(beat) < insize:
                            beat = np.pad(beat, (0, insize - len(beat)), 'constant')
                        y.append(cl)
                        beat_l2.append(beat)
    print(f"Finished record {num}. Beats so far: {len(y)}")

z = [i for i, beat in enumerate(beat_l2) if len(beat) != insize]
if z:
    print(f"Indices of samples with incorrect length: {z}")

if y and beat_l2:
    features = np.array(beat_l2, dtype=np.float32)
    le = LabelEncoder()
    labels = le.fit_transform(y)

    print(f"\nFinal Class Counts:")
    print(Counter(y))
    print(Counter(labels))

    np.save("MIT-BIH_features.npy", features)
    np.save("MIT-BIH_labels.npy", labels)
    print("\nSuccessfully created MIT-BIH_features.npy and MIT-BIH_labels.npy")
else:
    print("\nNo valid beats were found to create .npy files.")
