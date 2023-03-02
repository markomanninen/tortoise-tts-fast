# Imports used through the rest of the notebook.
import torchaudio, IPython
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
from pydub import AudioSegment
from zipfile import ZipFile
from glob import glob
from numba import cuda
import os, shutil
import locale

# can we use a mounted gdrive in colab environment for back ups?
is_colab = False

# pydub requires UTF-8
# this is initialized by init function
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"

def list_voices():
    directory_list = list()
    for root, dirs, files in os.walk("tortoise/voices/", topdown=False):
        for name in dirs:
            directory_list.append(name)
    print(directory_list)

# if directory does not exist, new ones will be created
# directory sub structure will be: colab_tts_files/{voiceid}/voices/
drive_colab_tts_dir = f"/content/drive/MyDrive/colab_tts_files"

def load_gdrive():
    global is_colab
    try:
        from google.colab import drive
        from google.colab import files

        # for back-upping files, this will pop up a google access grant window!
        drive.mount('/content/drive')

        is_colab = True
    except Exception as e:
        from ipywidgets import FileUpload
        from IPython.display import display
        print("Could not mount GDrive", e)

tts = None
# This will download all the models used by Tortoise from the HuggingFace hub.
def init():
    global tts
    if locale.getpreferredencoding() != 'UTF-8':
        print(locale.getpreferredencoding(), '->')
        locale.getpreferredencoding = getpreferredencoding

    print(locale.getpreferredencoding())
    
    check_gpu()
    
    tts = TextToSpeech()

def check_gpu():
    cc_cores_per_SM_dict = {(2,0) : 32, (2,1) : 48, (3,0) : 192, (3,5) : 192, (3,7) : 192, (5,0) : 128, (5,2) : 128, (6,0) : 64, (6,1) : 128, (7,0) : 64, (7,5) : 64, (8,0) : 64, (8,6) : 128, (8,9) : 128, (9,0) : 128 }
    # the above dictionary should result in a value of "None" if a cc match 
    # is not found.  The dictionary needs to be extended as new devices become
    # available, and currently does not account for all Jetson devices
    try:
      device = cuda.get_current_device()
      my_sms = getattr(device, 'MULTIPROCESSOR_COUNT')
      my_cc = device.compute_capability
      cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
      total_cores = cores_per_sm*my_sms
      print("GPU compute capability: " , my_cc)
      print("GPU total number of SMs: " , my_sms)
      print("total cores: " , total_cores)
    except:
      print("GPU not available!")

def remove_old_files(path):
    for f in glob(path):
        try:
            os.remove(f)
        except:
            print("Error while deleting file : ", f)

# If you want to use your own voice call load_voices. With reload = True,
# you can replace the files you may have already loaded to the voice directory.
# Upload at least 2 audio clips, which must be a WAV file, 6-10 seconds long.

def load_my_voice(voice, reload = False):
    global drive_colab_tts_dir, is_colab

    custom_voice_folder = f"tortoise/voices/{voice}"
    drive_colab_tts_voice_dir = f"{drive_colab_tts_dir}/{voice}"
    drive_voice_files_dir = f"{drive_colab_tts_voice_dir}/voices"

    # create a new voice directory if not exists
    try:
        # local
        os.makedirs(custom_voice_folder)
    except Exception as e:
        pass

    # use backup directories in colab / google drive
    if is_colab:
        ok = True
        for directory in (drive_colab_tts_dir, drive_colab_tts_voice_dir, drive_voice_files_dir):
            if True:
                try:
                    os.mkdir(directory)
                except:
                    ok = False
                    pass

        try:
            # get voices from back up
            print("Back up voices:", os.listdir(drive_voice_files_dir), "Tortoise voices", os.listdir(custom_voice_folder))

            if not reload and os.listdir(drive_voice_files_dir) == [] and os.listdir(custom_voice_folder) != []:
                print("Copying files from %s to %s" % (custom_voice_folder, drive_voice_files_dir))
                for f in os.listdir(custom_voice_folder):
                    shutil.copy(f"{custom_voice_folder}/{f}", drive_voice_files_dir)

            if not reload and os.listdir(drive_voice_files_dir) != [] and os.listdir(custom_voice_folder) == []:
                print("Copying files from %s to %s" % (drive_voice_files_dir, custom_voice_folder))
                for f in os.listdir(drive_voice_files_dir):
                    shutil.copy(f"{drive_voice_files_dir}/{f}", custom_voice_folder)
  
        except Exception as e:
            print(e)
    else:
        print("GDrive not available. Working with local Tortoise voices only", os.listdir(custom_voice_folder))

    # if force load new voices or the given voice has no files
    if reload or os.listdir(custom_voice_folder) == []:
        # loop over all file input files
        if os.listdir(custom_voice_folder) != []:
            print("Removing old voice files...")
            # remove old files from local
            remove_old_files(f'{custom_voice_folder}/*.wav')
            if is_colab:
                # remove old files directory from google drive
                remove_old_files(f'{drive_voice_files_dir}/*.wav')

        
        if is_colab:
            # loop over all file input files
            for i, file_data in enumerate(files.upload().values()):
                file_path = os.path.join(custom_voice_folder, f'{i}.wav')
                with open(file_path, 'wb') as f:
                    f.write(file_data)
                # back up to google drive
                if is_colab:
                    shutil.copy(file_path, f'{drive_voice_files_dir}/{i}.wav')
        else:
            def button_click_func(*change):
                print(change)
                for i, file_data in enumerate(change[0]['owner'].data):
                    file_path = os.path.join(custom_voice_folder, f'{i}.wav')
                    with open(file_path, 'wb') as f:
                        f.write(file_data)
                    # back up to google drive
                    if is_colab:
                        shutil.copy(file_path, drive_voice_files_dir)
            
            upload = FileUpload(accept='.wav', multiple=True)
            
            upload.observe(button_click_func, 'value')
            
            display(upload)
            

def text2speech(text, section_name, voice = "marko", preset = "fast", append_next = False, append_prev = False):
    global drive_colab_tts_dir, is_colab, tts

    drive_colab_tts_voice_dir = f"{drive_colab_tts_dir}/{voice}"

    # create a new backup voice directory if not exists
    try:
        os.mkdir(drive_colab_tts_dir)
    except:
        pass

    if is_colab:
        try:
            os.makedirs(drive_colab_tts_voice_dir)
        except:
            pass

    # remove all old generated files from root
    remove_old_files(f'generated-{voice}*.wav')
    # and from google drive back-up dir
    if is_colab:
        remove_old_files(f'{drive_colab_tts_voice_dir}/generated-{voice}*.wav')
    # load voice neural data
    voice_samples, conditioning_latents = load_voice(voice)

    # The first sentence seems to give a gradient warning,
    # thus this line is just to get over it before actual speech generation
    #tts.tts_with_preset("Dummy.", voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)
    #print()

    # keep a count of skipped empty or error lines
    r = 0
    # split text to lines list
    splt = text.split("\n")

    for i, t in enumerate(splt):
        # remove trailing spaces
        t = t.strip()
        i -= r

        # skip empty lines
        if t == "":
            r += 1
            continue

        # weather to add some silence to the end of the line
        add_silence = False
        l = len(t)
        # replace silence tag and compare the length of the string
        t = t.replace("[silence]", "")
        if l > len(t):
            add_silence = True

        try:
            # append the next three words from the next row sentence to give a feeling of
            # continuation of the bigger context for the speech model
            # note that this requires post editing for the generated audio files
            # because there will be extra three words at the end of the current line

            if append_prev and i+r > 0 and splt[i+r-1].strip() != "":
                y = splt[i+r-1]
                x = y.replace("[silence]", "")
                # if silence was in the previous sentence, then we want a
                # fresh start for the next line
                if len(x) == len(y):
                    t = (("%s" % ' '.join(x.split(" ")[-3:])).rstrip('.').lower().capitalize() + '. ') + t

            if append_next and i+r < len(splt) and splt[i+r+1].strip() != "":
                t += " %s..." % ' '.join(splt[i+r+1].split(" ")[:3])

            print(f"SENTENCE #{i}: ", t)

            # text to speech generation
            gen = tts.tts_with_preset(t, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)

            # save temp audio file
            file_wav = f"generated-{voice}-{i}.wav"
            torchaudio.save(file_wav, gen.squeeze(0).cpu(), 24000)

            if add_silence:
                seq = AudioSegment.from_file(file_wav) + AudioSegment.silent(duration=1000)
                seq.export(file_wav, format="wav")

            # backup temp audio file
            if is_colab:
                shutil.copy(file_wav, drive_colab_tts_voice_dir)
            print()
        except Exception as e:
            print(e)
            r += 1

    # zip all generated wav voice files
    file_zip = f"generated-{voice}-{section_name}-{preset}.zip"
    with ZipFile(file_zip, 'w') as zo:
        for j in range(i):
            zo.write(f"generated-{voice}-{j}.wav")
        zo.close()

    # backup zip
    if is_colab:
        shutil.copy(file_zip, drive_colab_tts_voice_dir)

    # combine wavs to a single file
    audio_files = [AudioSegment.from_file(f"generated-{voice}-{j}.wav") for j in range(i)]
    # add a second silence at the end
    audio_files.append(AudioSegment.silent(duration=1000))
    combined = sum(audio_files)

    # save combined file as mp3
    file_mp3 = f"generated-{voice}-{section_name}-{preset}.mp3"
    combined.export(file_mp3, format="mp3")

    # backup mp3
    if is_colab:
        shutil.copy(file_mp3, drive_colab_tts_voice_dir)

    # return a playback widget
    return AudioSegment.from_file(file_mp3)
