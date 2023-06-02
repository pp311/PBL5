import sounddevice as sd
import pyrebase
import numpy as np
from scipy.io.wavfile import write, read
config = {
        "apiKey":' AIzaSyCcd-PSDyR3qWVSp2iPf5SQ9-XYAyQfqas',
        "authDomain":'pbldemo-51145.firebaseapp.com',
        "databaseURL":'https://pbldemo-51145-default-rtdb.asia-southeast1.firebasedatabase.app/',
        "storageBucket":'pbldemo-51145.appspot.com'
        }
# config = {
#         "apiKey":'AIzaSyDG_m77TMMnIzcQnlAm3GIDttK91lXqHdw',
#         "authDomain":'pbl5-c5b07.firebaseapp.com',
#         "databaseURL":'https://pbl5-c5b07-default-rtdb.firebaseio.com/',
#         "storageBucket":'pbl5-c5b07.appspot.com'
#         }

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
record_path = "wakeword.wav"

window = None

def sd_callback(rec, count, time, status):
    global window
    rec = np.squeeze(rec)
    if window is None:
        window = np.copy(rec)
    window2 = np.copy(rec)
    rec = np.concatenate((window, rec))
    window = np.copy(window2)
    write(record_path, RATE, rec)
    storage.child("audio/wakeword" + count + ".wav").put(record_path)


RECORD_SECONDS = 0.5
RATE = 16000
count = 0
while True:
    print("------------" + str(count % 3))
    myrecording = sd.rec(int(RECORD_SECONDS * RATE), channels=1, samplerate=RATE)
    sd.wait()
    count = count + 1
    # write(record_path, RATE, myrecording)
    sd_callback(myrecording, str(count % 3), None, None)

