#!/usr/bin/env python

##########################
#
#Code to extract Pitch and save it to a MIDI file.
#Needs improvement in onset detection, and pitch detection could be improved as well
#
##########################

import essentia.standard as es
import numpy as np
import matplotlib.pyplot as plt
from essentia import array as arrayCast
from midiutil.MidiFile import MIDIFile

#Load audio and define parameters
audio = es.MonoLoader(filename = 'melodies/melodies/melody7.wav')()
winSize = 2048  
hopSize = 128
Fs = 44100
pend = np.size(audio) - winSize #ultimo valor de ventana para calcular
pin =0  #Apuntador
totalTime = np.size(audio)/float(Fs)

#Instanciar funciones de essentia
w = es.Windowing(type = 'hann', size = winSize)
yin = es.PitchYinFFT(frameSize = winSize, minFrequency = 20, maxFrequency = 2000)
spectrum = es.Spectrum()    #FUncion para calcular espectro por frame
calcEnergy = es.Energy()
predMel = es.PredominantMelody(numberHarmonics = 15, filterIterations = 3, frameSize = 2048, hopSize = 128, minFrequency = 20, maxFrequency = 2000, minDuration = 50)
onsetDet = es.OnsetDetection(method = 'flux')

#Definir Funciones
def freq2cent (freq):
#Convertir de frecuencia a cents, entrega int redondeado
    freq += .00000001    
    cents = 1200*np.log2(freq/440) + 6900
    for i in range(0, cents.size, 1):         
        if cents[i] < 0: cents[i] = -1
        cents[i] = int(np.round(cents[i]))
    return cents

def cambioNota(cents):
    return np.diff(cents)

def medfilt (x, kms):
    """Apply a length-k (ms) median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    k = int(np.ceil(kms*44.1)) #Convertir de ms a samples
    if k % 2 == 0 : k-=1
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)

def meanNote(cents, timeRes = .3):
#Promediar notas en la resolucion de tiempo indicada
    res = np.size(cents)/totalTime #cuantas muestras por ms hay
    deltaT = int(np.round(timeRes*res))   #Cuantas muestras hay que promediar
    print res, deltaT    
    for i in range(np.size(cents)-deltaT):
        cents[i] = int(np.mean(cents[i:i+deltaT]))
    return cents
    
def quantizeNote(centsV):
#Receive a vector in cents, output it quantized in MIDI note
    cents = centsV.copy()    
    for i in range(np.size(cents)):
        cents[i] = np.round(cents[i]/100.0)
    return cents
#Quantizar un vector de cents a notas
    
def segmentByFreq(notes, minSeparation = 100):
#Devuelve el indice donde hay cambio de nota ignorando separaciones muy juntas
#MinSeparation dado en ms
    cambios = cambioNota(quantizeNote(notes))   #Vector con cambios de nota quantizados
    noteChange = []
    res = np.size(notes)/float(totalTime*1000)  #cuantos bins por ms
    minSep = int(np.round(minSeparation*res))   #Minima separacion en bins
    lastChange = -1*minSep                      #Inicializar el ultimo cambio
    for i in range(np.size(cambios)):
        if cambios[i] > 0:
            if (i-lastChange) >= minSep:
                noteChange.append(i)
                lastChange = i
    return noteChange
    
def saveMIDI(filename, noteOnsets, melody, tempo):
    barOnsets = (noteOnsets*hopSize/float(Fs))*(tempo/60)    #Onsets dado en barra
    notes = quantizeNote(melody)    
    track = 0
    time = 0       
    MIDI = MIDIFile(1)
    
    # Add track name and tempo.
    MIDI.addTrackName(track,time,"MIDI TRACK")
    MIDI.addTempo(track,time,tempo)
    
    channel = 0
    volume = 100
    
    for i in range(np.size(barOnsets)):
        pitch = notes[noteOnsets[i]+1]  #leer el pitch en el siguiente frame al onset
        if pitch > 0:
            time = barOnsets[i]
            if i == np.size(barOnsets)-1:
                duration = 1
            else:
                duration = barOnsets[i+1]-barOnsets[i] 
            
            MIDI.addNote(track,channel,pitch,time,duration,volume)

    # And write it to disk.
    binfile = open(filename, 'wb')
    MIDI.writeFile(binfile)
    binfile.close()    
    
#Inicializar vectores
vectSize = np.ceil(np.size(audio))/hopSize  
peakDet = es.PeakDetection(threshold = 0.15, maxPosition = vectSize, range = vectSize, maxPeaks = 300)    #has to be normalized

melodyV = []
confV = []
energy = []
freqBands = []

for frame in es.FrameGenerator(audio, winSize, hopSize):
    spec = spectrum(w(frame))   #Calcular el espectro  
    melody, conf = yin(spec)    #Calcular melodia con algoritmo yin
    melody = 0 if conf < 0.5 else melody #0 para valores con poca confidence
    melodyV.append(melody)    
    ener = calcEnergy(frame)   #Calcular energia por frame
    energy.append(ener)    
    freqBands.append(es.FrequencyBands()(spec))
    #onsets[i] = onsetDet(spec, spec)    

#Convertir valores a float32 compatible con Essentia
melodyV = arrayCast(melodyV)
confV = arrayCast(confV)
energy = arrayCast(energy)

predMelody, confid = predMel(audio) #Calcular melodia con otro algoritmo

melodyV = freq2cent(melodyV)
predMelody = freq2cent(predMelody)

energy = np.log10(energy+.0000001)
energy = energy/float(max(energy)) #Normalize Energy
smoothEnergy = medfilt(arrayCast(energy), .7)   #Usando filtro para alisar curva, valor en ms
onsets, amplitudes = peakDet(smoothEnergy)  #detectar picos

noteOnsets = arrayCast(segmentByFreq(melodyV.copy()))

bpm, bpmAmpl = es.NoveltyCurveFixedBpmEstimator()(es.NoveltyCurve()(arrayCast(freqBands)))

saveMIDI('MIDIfile7.mid', noteOnsets, predMelody, bpm[0])

#Graficar
plt.subplot(3,1,1)  
plt.plot(np.linspace(0, totalTime, np.size(melodyV)), melodyV, 'b')
plt.plot(np.linspace(0, totalTime, np.size(predMelody)), predMelody, 'r')
plt.vlines(noteOnsets*hopSize/Fs, min(melodyV), max(melodyV))

plt.subplot(3,1,2)
plt.plot(np.linspace(0, totalTime, np.size(energy)),energy)
plt.vlines(onsets*hopSize/Fs, min(energy), max(energy))

plt.subplot(3,1,3)
plt.plot(np.linspace(0, totalTime, np.size(quantizeNote(cambioNota(melodyV)))),quantizeNote(cambioNota(melodyV)))
plt.show()
