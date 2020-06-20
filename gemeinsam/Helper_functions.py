#######################################################################
# This file contains the helper functions and tools used in the Jupyter Notebook
# for the Speech Analysis Part of Speech Communication Laboratory.
#
# P.A. Bereuter, C. Frischmann, F. Kraxberger
# SPSC, TU Graz -- 2020
#######################################################################

import matplotlib.pyplot as plt
import ipywidgets as widgets
import librosa
import IPython.display as ipd
import numpy as np
import parselmouth
import soundfile as sf
import bokeh
import sounddevice as sd
import time as clock

from pathlib import Path
from scipy import signal, fft, ifft



def import_sound_file(fileName, soundFolder=Path('../sounds/')):
    # This function imports a sound file for further usage.
    # @param: fileName The name of the sound file including the file extension; e.g. 'soundfile1.wav'
    # @param: soundFolder The path of the folder in which the sound files are located. Must be a Path-object.
    #                     Defaults to '../sounds/'
    #
    # @return: snd The sound file as a praat-parselmouth sound object
    # @return: audio1 The sound file as a 'soundfile' object
    # @return: fs Detected sampling frequency
    # @return: str(filePath) The file path as a string (dependent on the OS this script is running on)
    #
    filePath = soundFolder / fileName

    audio1, fs = sf.read(filePath)
    print(fileName+" loaded with sampling frequency f_s = {}".format(fs))

    snd = parselmouth.Sound(str(filePath))
    return snd, audio1, fs, str(filePath)


def deleteNan(indata):
    # helper function to delete Nans in record-array (indata). Trims data-vector to all values which are NOT Nan.
    indata_no_Nan = np.zeros([np.sum(~np.isnan(indata[:,0])),indata.shape[1]])
    indata_no_Nan[:,0]=indata[~np.isnan(indata[:,0]),0]
    indata_no_Nan[:,1]=indata[~np.isnan(indata[:,0]),1]
    return indata_no_Nan


def get_rec_and_play_button(fs, num_channels, max_duration = 30):
    # This function provides the RECORD, PLAY, and DELETE buttons to record audio from the local soundcard.
    # @param: fs
    # @param: num_channels
    # @param: max_duration
    #
    # @return: toggleRec
    # @return: togglePlay
    # @return: clearButton
    # @return: out
    # @return: box_layout
    # @return: indata
    #
    def on_click_Rec(button):
    #callback function for Rec-Toggle button. Determines what happens if Rec-Button is switched On/Off
        value = button.new
        if (button.new == True):
            button.owner.button_style='danger'
            with out:
                print('Recording started!')
            sd.rec(out=indata,samplerate=fs)

        else:
            button.owner.button_style=''
            sd.stop()
            with out:
                print('Recording stopped!')

    def on_click_Play(button):
        #callback function for Play-Toggle button. Determines what happens if Play-Button is switched On/Off
        if (button.new == True):
            button.owner.button_style='Success'
            with out:
                print('Playback started!')
            indata_no_Nan = deleteNan(indata)
            sd.play(indata_no_Nan)
            button.owner.description = 'Stop'
            button.owner.icon = 'stop-circle'
        if (button.new == False):
            sd.stop()
            with out:
                print('Playback stoped!')
            button.owner.button_style = ''
            button.owner.description = 'Play'
            button.owner.icon = 'play-circle'

    def on_click_Clear(button):
        indata[:] = np.NaN
        with out:
            ipd.clear_output()
    #        print('Audio-Array has been cleared!')
        return indata

    #maximum-duration for recorded samples ==> recorded sample shouldn't be so long (nothing is recorded after max_duration)
    #max_duration = 30 # seconds
    indata = np.empty([max_duration*fs,num_channels])
    indata[:] = np.NaN


    toggleRec =widgets.ToggleButton(
        value=False,
        description='Recording',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Record_Button',
        icon='circle' # (FontAwesome names without the `fa-` prefix)
    )

    togglePlay = widgets.ToggleButton(
        value=False,
        description='Play',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Play_Button',
        icon='play-circle' # (FontAwesome names without the `fa-` prefix)
    )

    clearButton = widgets.Button(
        description='Clear Audio-Array',
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Clear_Button',
        icon='trash' # (FontAwesome names without the `fa-` prefix)
    )

    toggleRec.observe(on_click_Rec, 'value')

    togglePlay.observe(on_click_Play, 'value')

    clearButton.on_click(on_click_Clear)

    out = widgets.Output()
    #display(out)

    # object_methods = [method_name for method_name in dir(clearButton)
    #                  if callable(getattr(clearButton, method_name))]
    # print(object_methods)

    box_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='center',
                    width='100%')

    return toggleRec,togglePlay,clearButton,out,box_layout,indata