#######################################################################
# This file contains the functions which produce the plots used in the Jupyter Notebook
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

from scipy import signal, fft, ifft
from ipywidgets import interact, interact_manual, Layout
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.io import push_notebook, output_notebook
from bokeh.layouts import gridplot, column, row
from bokeh.models import ColorBar, LogColorMapper, LogTicker, CustomJS, Slider, LinearAxis, Range1d, ColumnDataSource, Label, LabelSet, ImageURL
from bokeh.palettes import Greys256

Greys256 = list(Greys256)
Greys256.reverse() # reverse the grey color palette such that black is the maximum and white is the minimum
output_notebook(hide_banner=True) # suppress Bokeh banner when loading plots

#######################################################################
# PLOT FUNCTIONS EXERCISE 1
#######################################################################
def get_plot_window(window, dt_win, plottitle, showPlot=False):
	# Plots the window function
    #
	# @param: window The window which sould be plotted
	# @param: dt_win Time axis, same length as window
	# @param: plottitle Title of the plot
	# @param: showPlot Should the plot be shown? Default false
    #
	# @return: p Plot handle (only if showPlot=False)
    #
	p = figure(title=plottitle, plot_width=600, plot_height=200)
	p.line(dt_win, window, line_width = 2, color = 'blue')
	p.xaxis.axis_label = 't in s'
	p.yaxis.axis_label = 'lin. amplitude'
	if showPlot:
		show(p, notebook_handle=False)
		pass
	else:
		return p
		
		
def get_plot_intensity(snd, dt_snd, intensity, dt_intensity, plottitle, showPlot=False):
    # Plot intwnsity curve (use in conjuction with plot_in_subplots())
    #
    # @param: snd Values of audio signal
    # @param: dt_snd Time axis of audio signal
    # @param: intensity Values of intensity curve
    # @param: dt_intensity Time axis of intensity curve
    # @param: plottitle Title of plot
    # @param: showPlot Should the plot be shown? Default false
    #
    # @return: p Return plot handle (only if showPlot=False)
    #

    TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]
    p = figure(title=plottitle,plot_width=600, plot_height=400, x_range=(dt_snd[0], dt_snd[-1]), y_range=(np.floor(np.min(snd)*10)/10, np.ceil(np.max(snd)*10)/10),tools=TOOLS,tooltips=TOOLTIPS)
    p.line(dt_snd, snd, line_width = 0.5, color = '#BEBEBE')
    p.xaxis.axis_label = 't in s'
    p.yaxis.axis_label = 'lin. amplitude'

    p.extra_y_ranges = {"intensity": Range1d(start=np.floor(np.min(intensity)/10)*10, end=np.ceil(np.max(intensity)/10)*10)}
    p.line(dt_intensity, intensity, line_width = 2, color = 'red', y_range_name="intensity")
    p.add_layout(LinearAxis(y_range_name="intensity", axis_label='Intensity in dB'), 'right')
    if showPlot:
        show(p, notebook_handle=False)
        pass
    else:
        return p
		
def plot_in_subplots(p1, p2):
    # Plot two plot as subplots (use in conjuction with get_plot_intensity())
    #
    # @param: p1 Plot 1
    # @param: p2 Plot 2
    #
    show(column(p1, p2), notebook_handle=False)
	
def plot_two_intensity_curves(dt_PM_intensity, PM_intensity_val, dt_SC_intensity, SC_intensity, plottitle):
    # Plot two intensity curves together
    #
    # @param: dt_PM_intensity Time axis of first intensity curve
    # @param: PM_intensity_val Values of first intensity curve
    # @param: dt_SC_intensity Time axis of second intensity curve
    # @param: SC_intensity Values of second intensity curve
    # @param: plottitle Title of plot
    #
    TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]

    p = figure(title=plottitle, plot_width=600, plot_height=400, y_range=(np.floor(np.min(PM_intensity_val)*10)/10, np.ceil(np.max(PM_intensity_val)*10)/10),tools=TOOLS,tooltips=TOOLTIPS)
    p.line(dt_PM_intensity, PM_intensity_val, line_width = 2, color = 'blue', legend_label="parselmouth intensity curve")
    p.xaxis.axis_label = 't in s'
    p.yaxis.axis_label = 'Intensity in dB'

    p.line(dt_SC_intensity, SC_intensity, line_width = 2, color = 'red', legend_label="custom intensity curve")

    p.legend.location = "bottom_center"
    show(p, notebook_handle=False)
	

#######################################################################
# PLOT FUNCTIONS EXERCISE 2
#######################################################################
def plot_interactive_spectrogram(spectroData, tVec, fVec, plottitle, exportValue=0, dynamicRange=50):
    # This function plots an interactive spectrogram displaying the spectrogram and the spectrum of the selected slice
    #
    # @param: spectroData Spectrogram data in dB
    # @param: tVec Time axis of spectrogram
    # @param: fVec Frequency axis of spectrogram
    # @param: plottitle Title of plot
    # @param: defaultValue, defaultValue for the time-slider
    # @param: dynamicRange The dynamic range that sould be plotted, default 50dB
    #
    # @return: timeWidget Handle to time selector widget
    #
    layout=Layout(width='650px')
    timeWidget = widgets.FloatSlider(min=tVec[0], max=tVec[-1], step=tVec[1]-tVec[0], value=exportValue,
                                     description="Time in s")
    timeWidget.layout = layout
    timeInStft = timeWidget.value
    stftFrame = (np.abs(tVec - timeInStft)).argmin()

    TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]
    p1 = figure(title=plottitle,plot_width=650, plot_height=450, x_range=(tVec[0],tVec[-1]),
                y_range=(fVec[0],fVec[-1]), tools=TOOLS, tooltips=TOOLTIPS)
    p1.xaxis.axis_label = 'Time in s'
    p1.yaxis.axis_label = 'Frequency in Hz'
    
    color_mapper = bokeh.models.LinearColorMapper(palette=Greys256, low=spectroData.max()-dynamicRange, high=spectroData.max())
    color_bar = ColorBar(color_mapper=color_mapper, title='dB', title_text_align='left',
                     label_standoff=12, border_line_color=None, location=(0,0))

    p1.add_layout(color_bar, 'right')
       
    p1.image(image=[spectroData], x=tVec[0], y=fVec[0], dw=tVec[-1], dh=fVec[-1], color_mapper=color_mapper)
    p1.grid.visible=False
    spectrumLine = p1.line([timeInStft, timeInStft], [fVec[0], fVec[-1]], line_width = 2, color = 'red')

    p2 = figure(title="Spectrum of Selected STFT Frame",plot_width=650, plot_height=250, x_range=(fVec[0],fVec[-1]),
                y_range=(np.min(spectroData[:,stftFrame]),0), tools=TOOLS, tooltips=TOOLTIPS)
    p2.xaxis.axis_label = 'Frequency in Hz'
    p2.yaxis.axis_label = 'relative Magnitude in dB'
    spectrumPlot = p2.line(fVec, spectroData[:,stftFrame], line_width = 2, color = 'red', legend_label='slice of spectrogram')
    p2.line(fVec, spectroData.max()-dynamicRange, line_width=1, color='grey', legend_label='dynamic range')
    p2.legend.location = "top_right" #"bottom_center"
    p2.legend.orientation = 'horizontal'
    
    pAll = gridplot([[p2], [p1]])
    show(pAll,notebook_handle=True)

    def update_plot(timeInStft, stftFrame):
        spectrumLine.data_source.data['x'] = [timeInStft, timeInStft]
        spectrumPlot.data_source.data['y'] = spectroData[:,stftFrame]
        push_notebook()

    def on_value_change(change):
        timeInStft = timeWidget.value
        stftFrame = (np.abs(tVec - timeInStft)).argmin()
        update_plot(timeInStft, stftFrame)

    timeWidget.observe(on_value_change, names='value')
    return timeWidget
	
def plot_spectrogram_with_formants(spectroData, tVec, fVec, formantValues, formant_tVec, plottitle,
                                          pitchValues=np.array([np.nan, np.nan]), pitch_tVec=np.array([np.nan, np.nan]),
                                          dynamicRange=50):
    # This function plot a spectrogram with formants and f0
    #
    # @param: spectroData The spectrogram data in dB
    # @param: tVec Time axis of spectrogram
    # @param: fVec Frequency axis of spectrogram
    # @param: formantValues Values of all detected formants
    # @param: formant_tVec Time axis of formants
    # @param: plottitle Title of plot
    # @param: pitchValues Values of f0-contour, default nan -> no f0 coutour is plotted: set if f0 contour should be plotted
    # @param: pitch_tVec Time axis of f0-contour, default nan -> no f0 coutour is plotted: set if f0 contour should be plotted
    # @param: dynamicRange Dynamic range, default 50dB
    #


    TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]
    
    p1 = figure(title=plottitle,plot_width=650, plot_height=450, x_range=(tVec[0],tVec[-1]),
                y_range=(fVec[0],fVec[-1]), tools=TOOLS, tooltips=TOOLTIPS)
    p1.xaxis.axis_label = 'Time in s'
    p1.yaxis.axis_label = 'Frequency in Hz'

    color_mapper = bokeh.models.LinearColorMapper(palette=Greys256, low=spectroData.max()-dynamicRange, high=spectroData.max())
    color_bar = ColorBar(color_mapper=color_mapper, title='dB', title_text_align='left',
                     label_standoff=12, border_line_color=None, location=(0,0))

    p1.add_layout(color_bar, 'right')
       
    p1.image(image=[spectroData], x=tVec[0], y=fVec[0], dw=tVec[-1], dh=fVec[-1], color_mapper=color_mapper)
    p1.grid.visible=False
    
    if ~np.isnan(pitchValues.all()):
        p1.scatter(pitch_tVec, pitchValues, size=6, line_color=None, fill_alpha=0.8,
                       fill_color='white')
        p1.scatter(pitch_tVec, pitchValues, size=4, line_color=None, fill_alpha=1,
                       fill_color='red', legend_label='f0')
    
    
    maxNumberFormants = formantValues.shape[0]
    for formantIdx in range(maxNumberFormants):
        p1.scatter(formant_tVec, formantValues[formantIdx,:], size=6, line_color=None, fill_alpha=0.8,
                   fill_color='white')
        p1.scatter(formant_tVec, formantValues[formantIdx,:], size=4, line_color=None, fill_alpha=1,
                   fill_color=bokeh.palettes.viridis(maxNumberFormants)[formantIdx], legend_label='F{}'.format(formantIdx+1))

    p1.legend.location = "center_right"
    p1.legend.orientation = 'vertical'
    show(p1,notebook_handle=True)
    pass


def plot_F1_F2_in_vowel_chart(audioVal, dt_snd, F1, F2, formant_tVec, plottitle,exportRange,lineSwitch=True):
    # This function plot the formants F1 & F2 in a vowel chart with selectable time region from the corresponding audio file
    #
    # @param: audioVal Values of the audio signal in a numpy array
    # @param: dt_snd Time axis of audio file
    # @param: F1 Values of Formant F1
    # @param: F2 Values of Formant F2
    # @param: formant_tVec Time axis of Formants
    # @param: plottitle Title of Plot
    # @param: exportRange Time Range for export
    # @param: lineSwitch True: connects the vowel dots with lines, False: no lines between vowel dots, default True
    #
    # @return: timeWidget Handle to time selector widget
    #
    layout = Layout(width='650px')

    timeWidget = widgets.FloatRangeSlider(
        value=[exportRange[0], exportRange[1]],
        min=formant_tVec[0],
        max=formant_tVec[-1],
        step=formant_tVec[1] - formant_tVec[0],
        description='Time range',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.4f',
    )

    timeWidget.layout = layout
    timeRangeStart = timeWidget.value[0]
    timeRangeEnd = timeWidget.value[1]
    timeRangeStartFrame = (np.abs(formant_tVec - timeRangeStart)).argmin()
    timeRangeEndFrame = (np.abs(formant_tVec - timeRangeEnd)).argmin()

    TOOLS = "hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]

    p1 = figure(title="Region of Audio Signal", plot_width=650, plot_height=150, x_range=(dt_snd[0], dt_snd[-1]),
                y_range=(-1, 1), tools=TOOLS, tooltips=TOOLTIPS)
    p1.xaxis.axis_label = 'Time in s'
    p1.yaxis.axis_label = 'lin. Amplidude'
    p1.line(dt_snd, audioVal, line_width=2)

    timeRangeStartFill = p1.patch([0, timeRangeStart, timeRangeStart,0], [-1, -1, 1,1], alpha=0.5, line_width=0, color='grey')
    timeRangeEndFill = p1.patch([timeRangeEnd, dt_snd[-1], dt_snd[-1], timeRangeEnd], [-1, -1, 1, 1], alpha=0.5, line_width=0, color='grey')

    timeRangeStartLine = p1.line([timeRangeStart, timeRangeStart], [-1, 1], line_width=2, color='red')
    timeRangeEndLine = p1.line([timeRangeEnd, timeRangeEnd], [-1, 1], line_width=2, color='red')

    p3 = figure(title=plottitle, plot_width=653, plot_height=418, x_range=(3000,100),
                y_range=(1100,100), tools=TOOLS, tooltips=TOOLTIPS)

    # DISPLAY GROUND TRUTH FROM SENDLMEIER
    # image from Sendlmeier paper as background
    p3.image_url(url=["sendlmeier_formants.png"], x=100, y=100, w=2900, h=1000, anchor="top_right",level='image') #switch on/off

    p3.xaxis.axis_label = 'F2 in Hz'
    p3.yaxis.axis_label = 'F1 in Hz'

    formantScatterPlot = p3.scatter(F2[timeRangeStartFrame:timeRangeEndFrame],
                                    F1[timeRangeStartFrame:timeRangeEndFrame], legend_label='detected formants')

    if lineSwitch:
        formantLinePlot = p3.line(F2[timeRangeStartFrame:timeRangeEndFrame],F1[timeRangeStartFrame:timeRangeEndFrame], line_width=1.5)

    # calculated means from Sendlmeier paper as Backgound
    sendlmeierVowels = ['a','a:','e:','e2','e2:','i','i2:','o','o:','u','u2:','y','y:','oe','o/:','e3']
    sendlmeierF1 =    [ 765, 817, 391, 549, 533, 401, 283, 571,412, 417, 328, 400, 311, 519, 406, 545]
    sendlmeierF2 =    [1479,1396,2294,1929,2257,1999,1902,1137,865,1046, 905,1607,1766,1566,1553,1605]
    sendlmeierData = ColumnDataSource(data=dict(trueF1=sendlmeierF1,
                                        trueF2=sendlmeierF2,
                                        trueVow=sendlmeierVowels))
    labels = LabelSet(x='trueF2', y='trueF1', text='trueVow', level='annotation',
                      x_offset=0.0, y_offset=0.0, source=sendlmeierData, render_mode='canvas', text_align='center',text_baseline='middle')
    p3.legend.location = "bottom_left"
    p3.legend.click_policy = "hide"

    # FINISH PLOT
    pAll = gridplot([[p3], [p1]])
    show(pAll, notebook_handle=True)

    def update_plot(timeRangeStart, timeRangeEnd,timeRangeStartFrame,timeRangeEndFrame):
        # This funtion updates the plot after a change in the time selector widget
        #
        # @param: timeRangeStart Begin of time range in seconds
        # @param: timeRangeEnd End of time range in seconds
        # @param: timeRangeStartFrame Begin of time range in frames
        # @param: timeRangeEndFrame End of time range in frames
        #
        timeRangeStartLine.data_source.data['x'] = [timeRangeStart, timeRangeStart]
        timeRangeEndLine.data_source.data['x'] = [timeRangeEnd, timeRangeEnd]
        timeRangeStartFill.data_source.data['x'] = [0, timeRangeStart, timeRangeStart,0]
        timeRangeEndFill.data_source.data['x'] = [timeRangeEnd, dt_snd[-1], dt_snd[-1], timeRangeEnd]
        formantScatterPlot.data_source.data = {'x': F2[timeRangeStartFrame:timeRangeEndFrame],
                                               'y': F1[timeRangeStartFrame:timeRangeEndFrame]}
        if lineSwitch:
            formantLinePlot.data_source.data = {'x': F2[timeRangeStartFrame:timeRangeEndFrame],
                                                'y': F1[timeRangeStartFrame:timeRangeEndFrame]}
        push_notebook()

    def on_value_change(change):
        # This function listens for value changes in the time selector widget.
        # See ipython widgets documentation for more information.
        #
        # @param: change
        #
        timeRangeStart = timeWidget.value[0]
        timeRangeEnd = timeWidget.value[1]
        timeRangeStartFrame = (np.abs(formant_tVec - timeRangeStart)).argmin()
        timeRangeEndFrame = (np.abs(formant_tVec - timeRangeEnd)).argmin()
        update_plot(timeRangeStart, timeRangeEnd,timeRangeStartFrame,timeRangeEndFrame)

    timeWidget.observe(on_value_change, names='value')
    return timeWidget


#######################################################################
# PLOT FUNCTIONS EXERCISE 3
#######################################################################
def plot_autocorr(R, plottitle):
    TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]
    p = figure(title=plottitle,plot_width=650, plot_height=350,
                y_range=(min(R),max(R)), tools=TOOLS, tooltips=TOOLTIPS)
    p.xaxis.axis_label = 'Lag in Samples'
    p.yaxis.axis_label = 'Auto-Correlation'
    
    p.line(range(len(R)), R, color = 'blue')
    show(p)
    pass

def plot_filter(coeffs, fs, plottitle):
    TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]
    
    w,h = signal.freqz([1], a, fs=fs)
    
    p = figure(title=plottitle,plot_width=650, plot_height=450,
                y_range=(min(20*np.log10(abs(h))),max(20*np.log10(abs(h)))), tools=TOOLS, tooltips=TOOLTIPS)
    p.xaxis.axis_label = 'Frequenzy in Hz'
    p.yaxis.axis_label = 'Magnitude in dB'
    
    p.line(w, 20*np.log10(abs(h)), color = 'blue')
    show(p)
    pass

def plot_zplane(zerosPolynom, polesPolynom, plottitle):
    TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]
    
    functionZeros = np.roots(zerosPolynom)
    functionPoles = np.roots(polesPolynom)
    
    p = figure(title=plottitle,plot_width=650, plot_height=600,
                x_range=(-1.2, 1.2), y_range=(-1.2, 1.2), tools=TOOLS, tooltips=TOOLTIPS)
    p.xaxis.axis_label = 'Real Part'
    p.yaxis.axis_label = 'Imaginary Part'
    
    p.line(np.cos(np.linspace(0, 2*np.pi, 1024)), np.sin(np.linspace(0, 2*np.pi, 1024)), color = 'blue', line_dash='dotted')
    p.scatter(np.real(functionZeros), np.imag(functionZeros), marker = 'o', color = 'red')
    p.scatter(np.real(functionPoles), np.imag(functionPoles), marker = 'x', color = 'red')
    show(p)
    pass


def plot_interactive_filter_zplane(coeffs, coeffBound, fs, powerDensity, plottitle, exportValue):
    layout=Layout(width='650px')
    iterationWidget = widgets.IntSlider(min=0, max=coeffs.shape[0]-1, step=1, value=exportValue,
                                     description="Iterations")
    iterationWidget.layout = layout
    iterationIndex = iterationWidget.value
    
    TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]
    
    w,h = signal.freqz([1], coeffs[-1, 0:coeffBound], fs=fs)
    wMatrix = np.zeros([coeffs.shape[0], len(w)])
    hMatrix = np.zeros([coeffs.shape[0], len(h)])
    hMin = min(abs(h))
    hMax = max(abs(h))
    
    for i in range(0, coeffs.shape[0]):
        w,h = signal.freqz([1], coeffs[i, 0:coeffBound], fs=fs)
        wMatrix[i, 0:len(w)] = w
        hMatrix[i, 0:len(h)] = abs(h)
        if hMin > min(abs(h)):
            hMin = min(abs(h))
        if hMax < max(abs(h)):
            hMax = max(abs(h))
        
    #hMin = 20*np.log10(hMin) - 10
    #hMax = 20*np.log10(hMax) + 10
    hMin = min(powerDensity)
    hMax = max(powerDensity)
    
    p1 = figure(title=plottitle,plot_width=650, plot_height=350,
                y_range=(hMin,hMax), tools=TOOLS, tooltips=TOOLTIPS)
    p1.xaxis.axis_label = 'Frequenzy in Hz'
    p1.yaxis.axis_label = 'Magnitude in dB'
    
    p1.line(np.linspace(0, fs/2, len(powerDensity)), powerDensity, line_width = 0.2, color = 'grey')

    spectrumLine = p1.line(wMatrix[iterationIndex], 20*np.log10(hMatrix[iterationIndex]), color = 'blue')
    
    functionPoles = np.roots(coeffs[iterationIndex,0:coeffBound])
    
    p2 = figure(title='Z Plane',plot_width=300, plot_height=300,
                x_range=(-1.2, 1.2), y_range=(-1.2, 1.2), tools=TOOLS, tooltips=TOOLTIPS)
    p2.xaxis.axis_label = 'Real Part'
    p2.yaxis.axis_label = 'Imaginary Part'
    
    p2.line(np.cos(np.linspace(0, 2*np.pi, 1024)), np.sin(np.linspace(0, 2*np.pi, 1024)), color = 'blue', line_dash='dotted')
    polesPlot = p2.scatter(np.real(functionPoles), np.imag(functionPoles), marker = 'x', color = 'red')
    
    pAll = gridplot([[p1], [p2]])
    show(pAll,notebook_handle=True)

    def update_plot(iterationIndex):
        spectrumLine.data_source.data['x'] = wMatrix[iterationIndex]
        spectrumLine.data_source.data['y'] = 20*np.log10(abs(hMatrix[iterationIndex]))
        functionPoles = np.roots(coeffs[iterationIndex, 0:coeffBound])
        polesPlot.data_source.data['x'] = np.real(functionPoles)
        polesPlot.data_source.data['y'] = np.imag(functionPoles)
        push_notebook()

    def on_value_change(change):
        iterationIndex = iterationWidget.value
        update_plot(iterationIndex)

    iterationWidget.observe(on_value_change, names='value')
    return iterationWidget


def plot_time_signal(timeSignal, fs, plottitle):
    TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]
    p = figure(title=plottitle,plot_width=650, plot_height=250, x_range=(0,np.divide(len(timeSignal),fs)),
                y_range=(min(timeSignal),max(timeSignal)), tools=TOOLS, tooltips=TOOLTIPS)
    p.xaxis.axis_label = 'Time in s'
    p.yaxis.axis_label = 'Amplitude'
    
    p.line(np.divide(range(len(timeSignal)), fs), timeSignal)
    show(p)
    pass


def plot_spectrum(spectData, fVec, plottitle):
    TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]
    
    p1 = figure(title=plottitle,plot_width=650, plot_height=250, x_range=(fVec[0],fVec[-1]),
                y_range=(min(spectData),max(spectData)), tools=TOOLS, tooltips=TOOLTIPS)#, x_axis_type="log")
    p1.xaxis.axis_label = 'Frequency in Hz'
    p1.yaxis.axis_label = 'Magnitude in dB'
    p1.line(fVec, spectData)
    show(p1)
    pass


def plot_cepstrum(cepstData, qVec, lifterLength, plottitle, lifterLP = False):
    TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]
    if lifterLP:
        p = figure(title=plottitle,plot_width=650, plot_height=350, x_range=(0,20*lifterLength),
                y_range=(min(cepstData),0), tools=TOOLS, tooltips=TOOLTIPS)
    else:
        p = figure(title=plottitle,plot_width=650, plot_height=450, x_range=(qVec[0],qVec[-1]),
                y_range=(min(cepstData),max(cepstData)), tools=TOOLS, tooltips=TOOLTIPS)
    p.xaxis.axis_label = 'Quefrency in s'
    p.yaxis.axis_label = 'Magnitude'
    p.line(qVec, cepstData)
    if lifterLP:
        p.line([0, lifterLength], [0, 0], color='red')
        p.line([lifterLength, lifterLength], [0, -1000], color='red')
    show(p)
    pass


#######################################################################
# PLOT FUNCTIONS EXERCISE 4
#######################################################################
def get_plot_time_domain_sig(snd, dt_snd, plottitle, showPlot):
    #
    # get_plot_time_domain_sig(): plots the time-domain signal of a given 1-D audio-file
    #
    # Input: snd        ... 1D np.array containing audio-file which is to be plotted
    #       dt_snd      ... 1D np.array containing time-vector of given audio-file
    #       plottitile  ... String containing tile of Plot
    #       showPlot    ... Bool in order to surpress plot display.
    #                      If set to 'true' plot is displayed.

    TOOLS = "hover,crosshair,pan,wheel_zoom,box_zoom,save,reset"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]
    p = figure(title=plottitle, plot_width=600, plot_height=250, x_range=(dt_snd[0], dt_snd[-1]),
               y_range=(np.floor(np.min(snd) * 10) / 10, np.ceil(np.max(snd) * 10) / 10), tools=TOOLS,
               tooltips=TOOLTIPS)
    p.line(dt_snd, snd, line_width=0.5, color='#BEBEBE')
    p.xaxis.axis_label = 't in s'
    p.yaxis.axis_label = 'lin. amplitude'
    if showPlot:
        show(p, notebook_handle=False)
        pass
    else:
        return p