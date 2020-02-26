"""Demonstration of deep symbolic regression."""
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from cycler import cycler

import tkinter as tk
# from tkinter import ttk

# sleep 


class Model:
    """Class for the DSR backend."""

    def __init__(self):
        pass

    def step():
        """Perform one iteration of DSR"""
        pass

    # control <-> view


class View(tk.Tk):
# class View(tk.Toplevel):
    """Class for plots and diagnostics."""

    # controller, visulization, diagnostic

    def __init__(self, root):
        # tk.Toplevel.__init__(self, master)
        self.root = root
        # self.protocol('WM_DELETE_WINDOW', self.master.destroy)
        self.init_window()

    def init_window(self):
        self.content_left = tk.Frame(self.root)
        self.content_right = tk.Frame(self.root)

        self.content_left.pack(side=tk.LEFT)
        self.content_right.pack(side=tk.RIGHT)

        ### LEFT ###
        frame_vis = tk.Frame(self.content_left, borderwidth=20, width=720, height=540)
        frame_config = tk.Frame(self.content_left)
        frame_control = tk.Frame(self.content_left)

        frame_vis.pack(side=tk.TOP)
        frame_config.pack(side=tk.LEFT, padx=50)
        frame_control.pack(side=tk.RIGHT, padx=100)

        ### LEFT 1: vis ###
        # self.visualization = Trace(frame_vis, colors=['brown'],dpi=100)
        self.visualization = Trace(frame_vis, colors=['brown'], figsize=(6,6), dpi=100)
        self._init_visualization()

        ### LEFT 2-1: config ###
        onevar = tk.BooleanVar()
        twovar = tk.BooleanVar()
        threevar = tk.BooleanVar()
        onevar.set(True)
        twovar.set(False)
        threevar.set(True)

        fram_name = tk.Frame(frame_config)
        fram_name.pack(side=tk.TOP)
        # namelabel = tk.Label(fram_name, text="csv")
        # name = tk.Entry(fram_name)
        tk.Label(fram_name, text="csv").pack(side=tk.LEFT)
        tk.Entry(fram_name).pack(fill=tk.X)

        one = tk.Checkbutton(frame_config, text="add", variable=onevar, onvalue=True)
        two = tk.Checkbutton(frame_config, text="mul", variable=twovar, onvalue=True)
        three = tk.Checkbutton(frame_config, text="exp", variable=threevar, onvalue=True)

        one.pack(side=tk.LEFT)
        two.pack(side=tk.LEFT)
        three.pack(side=tk.LEFT)

        ### LEFT 2-2: control ###
        start = tk.Button(frame_control, text="Start")
        stop = tk.Button(frame_control, text="Stop")
        start.pack(fill=tk.X,side=tk.LEFT, ipadx=10,ipady=3)
        stop.pack(side=tk.LEFT, ipadx=10,ipady=3) 

        ### RIGHT ### self.content_right
        self.equation = tk.Frame(self.content_right, borderwidth=20, width=720, height=100)

        self.training_nmse = Trace(self.content_right, colors=['brown'], figsize=(7,2), dpi=100)
        self.training = Trace(self.content_right, colors=['brown'], figsize=(7,2), dpi=100)
        self.distribution = Trace(self.content_right, colors=['brown'], figsize=(7,2), dpi=100)

        self.equation.pack()
        self.training_nmse.pack()
        self.training.pack()
        self.distribution.pack()

    def update_plots(self, best_p, rewards):
        self.visualization.plot_vis
        self.equation

        self.training_nmse.plot
        self.training.plot


    def update_distribution(self, data):
        self.distribution
        
    def _init_visualization(self, min=-100, max=100):
        self.visualization.ax.set_xlim(min, max)
        self.visualization.ax.set_ylim(min, max)
        self.visualization.min = -100
        self.visualization.max = 100

        # include data points in range min,max
        self.visualization.plot_vis()
    


##### PLOT TIME STEPS #####
# Prints time series information
# If speedup is needed, can refactor so that a single Trace object has multiple subplots.
class Trace(FigureCanvasTkAgg):
    def __init__(self, parent, colors=None, *args, **kwargs):
        self.length = 411 # Length (in time steps) of the plot at any given time (411 maps to 48 hr)
        self.shift = self.length/4 # How far (in time steps) to shift the plot when it jumps

        # HACK FOR NOW. It should find parent's time.
        self.time = 0
        
        self.parent = parent
        self.lines = None
        self.history = None        
        self.figure = Figure(*args, **kwargs)
        #self.figure.set_tight_layout(True)
        self.ax = self.figure.add_subplot(111)        
        FigureCanvasTkAgg.__init__(self, self.figure, self.parent)
        #self.show()        
        
        self.ax.set_xlim(0, self.length)
        self.ax.set_ylim(0, 100)
        
        # Will rotate colors after resetting.
        if colors is not None:
            self.ax.set_prop_cycle(cycler('color', colors))
    
    '''def animate(self, dummy):
        #self.ax.clear()
        max_yval = 0
        #max_xval = self.history[:,0]
        
        for i,line in enumerate(self.lines):
            line.set_xdata(range(len(self.history)))
            line.set_ydata(self.history[:,i])
            
            if max(self.history[:,i]) > max_yval:
                max_yval = max(self.history[:,i])
            
        self.ax.set_ylim(0, max_yval)
        #self.ax.set_xlim(max(0, max_xval-1000), max(1000, max_xval))
        
        return self.lines #self.ax.plot(self.history)'''
    
    def grid(self, row, column):
        self.get_tk_widget().grid(row=row, column=column)

    def pack(self, **kwargs):
        self.get_tk_widget().pack(**kwargs)
        
    def plot(self, data, log_scale=False, ymin_input=0):
        if log_scale:
            data = np.log10(data)

        if self.history is None:
            self.history = np.reshape(data, (1, data.shape[0]))
        else:
            self.history = np.vstack((self.history, data))
        
        updateFrequency = 1
        if self.time % updateFrequency == 0:
            
            # Animation
            '''
            if self.time == 0:
                self.ax.set_xlim(0,500)
                self.ax.set_ylim(0,2000)
                if self.colors is not None:
                    self.ax.set_prop_cycle(cycler('color', self.colors))
                self.lines = self.ax.plot(self.history)
                self.animation = animation.FuncAnimation(self.figure, self.animate, blit=True, interval=1000)
            '''                
            
            # Normal: 4-5 FPS
            '''self.ax.clear()
            if self.colors is not None:
                self.ax.set_prop_cycle(cycler('color', self.colors))
            self.ax.plot(self.history)
            self.ax.set_xlim(max(0, self.time-1000), max(1000, self.time))
            self.draw()'''
            
            # Artists
            blit = True
            if self.time == 0:                
                self.lines = self.ax.plot(self.history)
                self.background = self.figure.canvas.copy_from_bbox(self.ax.bbox) # Store the background
                self.xmax = -1   
                self.ymax = 1
                self.ymin = ymin_input
            else:
                self.figure.canvas.restore_region(self.background) # Restore the background                
                new_xmax = max(self.length, self.shift*(1 + self.time/self.shift) - 1)
                new_ymax = np.nanmax(self.history)
                #self.ax.draw_artist(self.ax.patch)
                # Turn off blitting if axes need redrawing

                if new_ymax > self.ymax:
                    if log_scale:
                        self.ymax = np.nanmax([2, 1 + new_ymax])
                    else:
                        self.ymax = 100*(1 + ceil(new_ymax/100)) # Update to the next 100
                    # print self.ymax
                    self.ax.set_ylim(self.ymin, self.ymax)
                    blit = False
                if new_xmax > self.xmax:
                    self.xmax = new_xmax
                    self.ax.set_xlim(self.xmax-self.length, self.xmax)                                           
                    blit = False              
                
                for i,line in enumerate(self.lines):
                    line.set_xdata(range(len(self.history)))
                    line.set_ydata(self.history[:,i])
                    self.ax.draw_artist(line)      

                if blit:
                    self.figure.canvas.blit(self.ax.bbox) # 20-21 FPS (19-20 FPS without argument)
                else:
                    self.figure.canvas.draw_idle()
                    self.figure.canvas.flush_events()
        
        self.time += 1
        
    def plot_vis(self, equation=None):
        """ visualization frame for equation plot """
        # plot 
        xs=np.arange(self.min,self.max,0.3)
        ys = 2*np.sin(xs)
        self.ax.set_ylim(auto=True)
        self.ax.plot(xs,ys)

        self.pack(fill=tk.BOTH)
    

    def reset(self):
        self.time = 0     
        self.history = None        
        if self.lines is not None:
            [line.remove() for line in self.lines] # Prevents old lines from still appearing on plots after reset.
        self.ax.set_xlim(0, self.length)
        self.ax.set_ylim(0, 100)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()



class Controller:
    """Class for uploading data and configuring runs."""

    def __init__(self, root):
        self.model = Model()
        self.view = View(root)

    # interact w/ buttons
    # config: upload csv, set library, noise
    # control: start, stop, step, reset
    # self.view.start.config(command=self.startDSR)
    # def startDSR(self) ~~


def main():
    root = tk.Tk()
    root.title("DSR VIS")
    # root.withdraw()
    # root.attributes('-fullscreen', True)
    root.geometry("1440x900")
    app = Controller(root)
    root.mainloop()


if __name__ == '__main__':
    main()