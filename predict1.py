import pickle
import wave
import librosa
import math
import numpy as np
import pyaudio
import pyrebase
from pydub import AudioSegment, effects
import sounddevice as sd
from scipy.io.wavfile import write, read
import noisereduce as nr
import webrtcvad
import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
vad = webrtcvad.Vad()
vad.set_mode(1)

# config = {
#         "apiKey":' AIzaSyCcd-PSDyR3qWVSp2iPf5SQ9-XYAyQfqas',
#         "authDomain":'pbldemo-51145.firebaseapp.com',
#         "databaseURL":'https://pbldemo-51145-default-rtdb.asia-southeast1.firebasedatabase.app/',
#         "storageBucket":'pbldemo-51145.appspot.com'
#         }
config = {
        "apiKey":'AIzaSyDG_m77TMMnIzcQnlAm3GIDttK91lXqHdw',
        "authDomain":'pbl5-c5b07.firebaseapp.com',
        "databaseURL":'https://pbl5-c5b07-default-rtdb.firebaseio.com/',
        "storageBucket":'pbl5-c5b07.appspot.com'
        }
firebase = pyrebase.initialize_app(config)
db = firebase.database()
storage = firebase.storage()
model = {}
class_names = ['bat', 'tat', 'mot', 'hai', 'ba', 'sheila', 'den', 'quat']
audio_format = 'wav'

record_path = 'temp/record.wav'
trimmed_path = 'temp/trimmed.wav'
model_path = 'models_train_time'

RATE = 16000
DURATION = 5
FRAME_LEN = 0.02

def framing(speech_signal, Fs):
    frame_size = round(FRAME_LEN * Fs)
    frame_count = math.floor(len(speech_signal) / frame_size) 
    
    temp = 0
    frame = list()
    for i in range(frame_count):
        frame.append(speech_signal[temp : temp + frame_size]) 
        temp = temp + frame_size + 1

    return frame, frame_size, frame_count


def vad_collector(vad, data):
    triggered = False
    voiced_frames = []
    buf = []
    frames, frame_size, frame_count = framing(data, RATE)
    empty_frame = [0 for _ in range(frame_size)]

    for i in range(frame_count - 1):
        curr_frame = np.array(frames[i])
        is_speech = vad.is_speech(curr_frame, RATE)
        if is_speech and not triggered:
            buf.append(curr_frame)
            triggered = not triggered 
        
        elif not is_speech and triggered:
            voiced_frames.append(buf)
            buf = []
            triggered = not triggered 
        elif len(buf) != 0 and triggered:
            buf.append(curr_frame)

    return voiced_frames


def get_mfcc(file_path):
    y, sr = librosa.load(file_path)  # read .wav file
       
    hop_length = math.floor(sr * 0.010)  # 10ms hop
    win_length = math.floor(sr * 0.025)  # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # subtract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1, 1))
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, mode='nearest', order=1)
    delta2 = librosa.feature.delta(mfcc, mode='nearest', order=2)
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0)  # O^r
    # return T x 36 (transpose of X)
    return X.T  # hmmlearn use T x N matrix


def detect_leading_silence(sound, silence_threshold=-42.0, chunk_size=10):
        trim_ms = 0  # ms
        assert chunk_size > 0  # to avoid infinite loop
        while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
            trim_ms += chunk_size
        return trim_ms


for key in class_names:
    name = f"{model_path}/model_{key}.pkl"
    with open(name, 'rb') as file:
        model[key] = pickle.load(file)


def predict(file_name=None):
    if not file_name:
        file_name = record_path

    trimmed_path = 'temp/trimmed.wav'

    rate, data = read(file_name)
    # reduced_noise = nr.reduce_noise(y=data, sr=rate)
    # reduced_noise = nr.reduce_noise(y=data, sr=rate)
    write(trimmed_path, rate, data)
        
    sound = AudioSegment.from_file(file_name, format='wav')
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())

    duration = len(sound)
    # print(str(start_trim) + " - " + str(duration - end_trim))

    if start_trim >= 800:
        print("XXX")
        return
    # if duration - end_trim - start_trim > 700:
    #     print("XXX")
    #     return
    # if duration - end_trim - start_trim < 100:
    #     print("XXX")
    #     return
    normalizedsound = effects.normalize(sound)
    normalizedsound.export(trimmed_path, format='wav')
    
    rate, data = read(trimmed_path)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    write(trimmed_path, rate, reduced_noise)
    
    sound = AudioSegment.from_file(trimmed_path, format='wav')
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())

    duration = len(sound)
    print(str(start_trim) + " - " + str(duration - end_trim))
    # if start_trim > 200:
    #     start_trim -= 200
    # else: 
    #     start_trim = 0
    trimmed_sound = sound[start_trim:duration - end_trim]
    trimmed_sound.export(trimmed_path, format='wav')

    if start_trim >= 800 or start_trim == 0:
        print("XXX")
        return
    # if duration - end_trim - start_trim > 700:
    #     print("XXX")
    #     return
    # if duration - end_trim - start_trim < 200:
    #     print("XXX")
    #     return
    # Predict
    record_mfcc = get_mfcc(trimmed_path)
    scores = [model[cname].score(record_mfcc) for cname in class_names]
    predict_word = np.argmax(scores)
    # print(class_names[predict_word])
    return class_names[predict_word]


window = None
def sd_callback(rec, frames, time, status):
    # global window
    # rec = np.squeeze(rec)
    # if window is None:
    #     window = np.copy(rec)
    # window2 = np.copy(rec)
    # rec = np.concatenate((window, rec))
    # window = np.copy(window2)

    # write(record_path, RATE, rec)
    # res = predict(record_path)
    # if res == 'sheila':
    print('Sheila listening')
    db.child("devices").child("tdt@gmail,com").update({"sheila": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")})
    myrecording = sd.rec(int(5.5 * RATE), channels=1, samplerate=RATE, dtype='int16')
    sd.wait()
    print('Sheila stop listening')
    # write(record_path, RATE, myrecording)
    rate, data = read('/home/pp311/mountfolder/Downloads/PBL/5sec.wav')
    # reduced_noise = nr.reduce_noise(y=data, sr=RATE)
    write("temp/5secreduced.wav", rate, data)

    y = vad_collector(vad, data) 
    res = []
    for i,f in enumerate(y):
        length = 50 - len(f)
        empty = np.zeros(320, dtype=np.int16)
        
        for _ in range(length//2):
            f.insert(0, empty)
            f.append(empty)
            
        wav = []
        for k in range(len(f)):
            for j in range(320):
                wav.append(f[k][j])
        
        write(trimmed_path, RATE, np.array(wav))
        res.append(predict(trimmed_path))
        
    # print(res)

    action = ""
    device_name = ""
    device_num = ""
    for word in res:
        if word == "bat":
            action = "on"
        elif word == "tat":
            action = "off"
        elif word == "den":
            device_name = "Den"
        elif word == "quat":
            device_name = "Quat"
        elif word == "mot":
            device_num = 1
        elif word == "hai":
            device_num = 2
        elif word == "ba":
            device_num = 3
        

    if device_name != "" and device_num != "":
        device_name = device_name + " " + str(device_num)

        name_list = db.child("devices").child("tdt@gmail,com").child("name").get()
        for name in name_list.each():
            if device_name == name.val():
                if action == "":
                    curr_state = db.child("devices").child("tdt@gmail,com").child("state").child(name.key()).get()
                    action = "off" if curr_state.val() == "on" else "on"
                db.child("devices").child("tdt@gmail,com").child("state").update({name.key(): action })
    hd = "BAT" if action == "on" else "TAT"
    print(hd + " " + device_name.upper())
   
RECORD_SECONDS = 0.5
count = 0
# while True:
#     count = (count + 1) % 3
#     start = time.time()
#     storage.child("audio/wakeword" + str(count) + ".wav").download("wakeword" + str(count) + ".wav", filename="temp/wakeword" + str(count) + ".wav")
#     print(time.time() - start)
#     print(predict("temp/wakeword" + str(count) + ".wav"))    
# with sd.InputStream(channels=1, callback=sd_callback, blocksize=int(16000*0.5), samplerate=RATE, dtype='int16'):
#     while True:
#         pass


while True:
    print("------------")
    a = sd.rec(int(0.5 * RATE), channels=1, samplerate=RATE, dtype='int16')
    sd.wait()
    try:
        pre = predict('/home/pp311/mountfolder/Downloads/PBL/wakeword.wav')
        # print(pre)
        if pre == 'sheila':
            sd_callback(a, 0, 0, 0) 
    except Exception as e:
        print(e)
        pass

