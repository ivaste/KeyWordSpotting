#Imports
#import librosa
import numpy as np
import matplotlib.pyplot as plt
#import sounddevice as sd
import Models #Our models
import LoadAndPreprocessDataset
#.......................

import tkinter as tk


from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
sr=16000


#Window settings
window=tk.Tk()
window.geometry("600x400")
window.title("Key Word Spotting")
window.grid_columnconfigure(0,weight=1)
window.resizable(False,False)

def Record():
    lbl=tk.Label(window,text="ciaoo",font=("Helvetica",15))
    lbl.grid(row=0,column=1,sticky="W")

def Stop():
    lbl=tk.Label(window,text="ciaoo2")
    lbl.grid(row=1,column=1,padx=20,sticky="W")

def load():
    #myrec = np.load("LiveDemo/ONaudioclip.npy")
    myrec = np.load("LiveDemo/myrec2.npy")
    fig = Figure(figsize = (15, 6), 
                 dpi = 100) 
  
    
    # adding the subplot 
    plot1 = fig.add_subplot(111) 
  
    # plotting the graph 
    plot1.plot(myrec)
    plot1.xlabel("Samples")
    plot1.ylabel("Intensity")
    plot1.xlim(0, len(myrec))
  
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, 
                               master = window)   
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack() 
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
                                   window) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack() 

'''btn=tk.Button(text="Record",command=Record)
btn.grid(row=0, column=0,sticky="W")

btn2=tk.Button(text="Stop",command=Stop)
btn2.grid(row=1, column=0,pady=20,sticky="W")

btn3=tk.Button(text="load",command=load)
btn3.grid(row=3, column=0,pady=20,sticky="W")'''

# button that displays the plot 
plot_button = tk.Button(master = window,  
                     command = load, 
                     height = 2,  
                     width = 10, 
                     text = "Plot") 
  
# place the button  
# in main window 
plot_button.pack() 



from tkinter import *
from tkinter.ttk import *

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = Tk()

figure = Figure(figsize=(5, 4), dpi=100)
plot = figure.add_subplot(1, 1, 1)

plot.plot(0.5, 0.3, color="red", marker="o", linestyle="")

x = [ 0.1, 0.2, 0.3 ]
y = [ -0.1, -0.2, -0.3 ]
plot.plot(x, y, color="blue", marker="x", linestyle="")

canvas = FigureCanvasTkAgg(figure, root)
canvas.get_tk_widget().grid(row=0, column=0)

root.mainloop()

if __name__ =="__main__":
    window.mainloop()

