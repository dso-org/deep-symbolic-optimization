"""Demonstration of deep symbolic regression."""
import numpy as np
from sympy.parsing.latex import parse_latex
# from sympy import init_printing
# from sympy import preview
# init_printing(use_latex='mathjax')


import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from cycler import cycler

import tkinter as tk

from dsr.program import Program
import utils as U

# from tkinter import ttk

# sleep 

""" test static data """
test_eq = parse_latex(r"\frac {1 + \sqrt {\a}} {\b}")

# Configure the Program class from config file
U.configure_program("./data/demo.json")


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
        content_left = tk.Frame(self.root)
        # content_left = tk.Frame(self.root,width=450)
        content_mid = tk.Frame(self.root)
        content_right = tk.Frame(self.root)

        content_left.pack(side=tk.LEFT)
        content_mid.pack(side=tk.LEFT)
        content_right.pack(side=tk.LEFT)

        # content_left.grid(column=0)
        # content_mid.grid(column=1)
        # content_right.grid(column=2)

        ###########
        ### MID ###
        frame_vis = tk.Frame(content_mid, borderwidth=20, width=720, height=810)
        frame_vis_info = tk.Frame(content_mid, borderwidth=20, width=720, height=100)

        frame_vis.pack(side=tk.TOP)
        frame_vis_info.pack()

        """ visualization,vis_zoom_in/out/reset """
        self._init_frame_vis(frame_vis) # missing: plot label
        """ best_equation """
        self._init_frame_vis_info(frame_vis_info, equation= test_eq)

        ############
        ### LEFT ###
        frame_control = tk.Frame(content_left)
        frame_config = tk.Frame(content_left)

        frame_control.pack(ipady=10, pady=50)
        frame_config.pack()

        """ start_button/.. """
        self._init_control(frame_control)
        """ token libs, """
        self._init_config(frame_config)

        #############
        ### RIGHT ###
        self.training_nmse = Trace(content_right, colors=['brown'], figsize=(7,2), dpi=100)
        self.training = Trace(content_right, colors=['brown'], figsize=(7,2), dpi=100)
        self.distribution = Trace(content_right, colors=['brown'], figsize=(7,2), dpi=100)

        self.training_nmse.pack()
        self.training.pack()
        self.distribution.pack()

    def update_plots(self, best_p, rewards): 
        """ each iteration """
        best_equation=None
        self.visualization.plot_vis(best_equation)

        self.equation.pack()

        self.training_nmse.plot
        self.training.plot

    def update_distribution(self, data):
        """ over several iterations """
        self.distribution
        
    def _init_frame_vis(self, frame, min=-100, max=100):
        buttons = tk.Frame(frame)
        self.vis_zoom_in = tk.Button(buttons, text="+")
        self.vis_zoom_reset = tk.Button(buttons, text="zoom")
        self.vis_zoom_out = tk.Button(buttons, text="-")

        self.visualization = Trace(frame, colors=['brown'], figsize=(6,6), dpi=100)

        self.visualization.ax.set_xlim(min, max)
        self.visualization.ax.set_ylim(min, max)
        self.visualization.min = -100
        self.visualization.max = 100
        self.visualization.data_points = None

        # include data points in range min,max
        self.visualization.plot_vis()

        """ pack vis"""
        buttons.pack()
        self.vis_zoom_out.pack(side=tk.RIGHT, fill=tk.X, expand=1)
        self.vis_zoom_reset.pack(side=tk.RIGHT, fill=tk.X, expand=1)
        self.vis_zoom_in.pack(side=tk.RIGHT, fill=tk.X, expand=1)

        self.visualization.pack(fill=tk.BOTH)
        # frame.pack(expand=1)

    def _init_frame_vis_info(self, frame, equation):
        self.best_equation = tk.Label(frame, text=equation)
        # tk.Label(frame, image=tk.PhotoImage())
        
        tk.Label(frame, text="Best Equation:").pack(side=tk.LEFT)
        self.best_equation.pack(side=tk.LEFT)
    
    def _init_control(self, frame):
        self.start_button = tk.Button(frame, text="Start")
        self.stop = tk.Button(frame, text="Stop")

        self.start_button.pack(ipadx=50, ipady=20, side=tk.LEFT, fill=tk.X)
        self.stop.pack(ipadx=50, ipady=20, side=tk.LEFT, fill=tk.X) 

    def _init_config(self, frame):
        fr_upload = tk.Frame(frame)
        fr_library = tk.Frame(frame)
        fr_sliders = tk.Frame(frame)

        fr_upload.pack()
        fr_library.pack()
        fr_sliders.pack()

        ### upload ### 
        tk.Label(fr_upload, text="Import Data").pack(side=tk.LEFT)
        self.data_input = tk.Entry(fr_upload)
        self.data_input_button = tk.Button(fr_upload, text="UPLOAD")

        self.data_input.pack(side=tk.LEFT)
        self.data_input_button.pack(side=tk.LEFT)

        ### library ###
        token_library = [["add","sub","mul","div"],["sin","cos","exp","log"]]
        self.buttons_lib = [[tk.Checkbutton(fr_library, text=token, onvalue=True) for token in token_set] for token_set in token_library]

        # onevar = tk.BooleanVar()
        # onevar.set(True)
        # one = tk.Checkbutton(fr_library, text="add", variable=onevar, onvalue=True)

        """ pack""" 
        for row, button_set in enumerate(self.buttons_lib):
            for col, button in enumerate(button_set):
                button.grid(column=col, row=row)

        ### sliders ###
        tk.Label(fr_sliders, text="Number of Variables").grid(row=0, column=0)
        self.slide_num_var = tk.Scale(fr_sliders, from_=1, to=5, orient=tk.HORIZONTAL)
        # self.slide_num_var = tk.Scale(fr_sliders, from_=1, to=5, orient=tk.HORIZONTAL,label="hi")
        tk.Label(fr_sliders, text="Noise Level").grid(row=1, column=0)
        self.slide_noise = tk.Scale(fr_sliders, from_=0, to=1, orient=tk.HORIZONTAL)

        self.slide_num_var.grid(row=0, column=1)
        self.slide_noise.grid(row=1, column=1)


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
        xs = Program.X_train
        ys = Program.y_train

        # xs=np.arange(self.min,self.max,0.3)
        # ys = 2*np.sin(xs)
        self.ax.set_xlim(auto=True)
        self.ax.set_ylim(auto=True)
        self.ax.scatter(xs,ys)
    

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
    root.geometry("1600x900")
    app = Controller(root)
    root.mainloop()


if __name__ == '__main__':
    main()