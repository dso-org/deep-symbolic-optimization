"""Demonstration of deep symbolic regression."""
import os

import numpy as np
import pandas as pd
from sympy.parsing.latex import parse_latex

import time
# import pyautogui

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from cycler import cycler

import tkinter as tk

from dsr.program import Program
import utils as U

""" test static data """
Total_timesteps=2000

PATH = "./data"

# Configure the Program class from config file
U.configure_program(os.path.join(PATH, "demo.json"))


""" color set """
FONT_CONFIG=("arial 13 bold")
COLOR_VISINFO="red"

""" window pixel """
WIN_W = 1500
WIN_WH = 700

class Model:
    """Class for the DSR backend."""

    def __init__(self):

        self.callbacks = {}

        # Data
        self.batch_rewards = [] # List of np.ndarrays of size (batch_size,)
        self.best_programs = [] # List of best Programs
        self.training_info = [] # List of training information
        self.iteration = 0

        # Load offline data files
        with open(os.path.join(PATH, "traversals.txt"), "r") as f:
            self.traversal_text = f.readlines()

        self.all_rewards = np.load(os.path.join(PATH, "dsr_Nguyen-5_0_all_r.npy"))

        pd_all_tinfo = pd.read_csv(os.path.join(PATH, "dsr_Nguyen-5_0.csv"))
        self.all_training_info = pd_all_tinfo.to_numpy()


    def addCallback(self, func):
        self.callbacks[func] = 1


    def delCallback(self, func):
        del self.callbacks[func]


    def _docallbacks(self):
        for func in self.callbacks:
             func(self.batch_rewards, self.best_programs[-1], self.training_info[-1])


    def step(self):
        """Perform one iteration of DSR"""

        # Read rewards from file
        r = self.all_rewards[self.iteration]
        self.batch_rewards.append(r)

        # Read Program from file
        p = U.make_program(self.traversal_text, self.iteration)
        self.best_programs.append(p)

        # Read training info from file
        ti = self.all_training_info[self.iteration]
        self.training_info.append(ti)

        self.iteration += 1

        self._docallbacks()

    # control <-> view


class View(tk.Tk):
# class View(tk.Toplevel):
    """Class for plots and diagnostics."""

    # controller, visulization, diagnostic

    def __init__(self, root):
        self.root = root
        self.init_window()

    def init_window(self):
        content_left = tk.Frame(self.root, width=800)
        content_mid = tk.Frame(self.root, width=1000)
        content_right = tk.Frame(self.root, width=800)

        content_left.pack(side=tk.LEFT)
        content_mid.pack(side=tk.LEFT)
        content_right.pack(side=tk.LEFT)

        ###########
        ### MID ###
        frame_vis = tk.Frame(content_mid)
        frame_vis_info = tk.Frame(content_mid)
        frame_vis.pack(fill=tk.X, expand=1)
        frame_vis_info.pack(fill=tk.Y, expand=1)

        """ visualization,vis_zoom_in/out/reset """
        self._init_frame_vis(frame_vis) # missing: plot label
        """ best_equation """
        self._init_frame_vis_info(frame_vis_info)

        ############
        ### LEFT ###
        frame_control = tk.Frame(content_left)
        frame_config = tk.Frame(content_left)

        frame_control.pack(ipady=10, pady=10)
        frame_config.pack()

        """ start_button/.. """
        self._init_control(frame_control)
        """ token libs, """
        self._init_config(frame_config)

        #############
        ### RIGHT ###
        self.training_nmse = Trace(content_right,  xlabel='timestep', ylabel='Best NMSE',  colors=['brown'], figsize=(4,1.7), dpi=100)
        self.training_nmse.ax.set_ylim(0,0.6)

        self.training_best_reward = Trace(content_right, xlabel='timestep', ylabel='Best Reward',colors=['brown'], figsize=(4,1.7), dpi=100)
        self.training_best_reward.ax.set_ylim(0,1.1)

        self.distribution = Trace(content_right, xlabel='timestep', ylabel='Reward Dist',colors=['brown'], figsize=(4,1.7), dpi=100)

        self.training_nmse.pack()
        self.training_best_reward.pack()
        self.distribution.pack()

        top_choices = ["Choose Top Plot","Training curve"]
        bot_choices = ["Choose Bottom Plot","Reward distribution"]
        self.dropdown_top_var = tk.StringVar(content_right)
        self.dropdown_bot_var = tk.StringVar(content_right)

        self.dropdown_top_var.set(top_choices[0])
        self.dropdown_bot_var.set(bot_choices[0])

        menu_top = tk.OptionMenu(content_right, self.dropdown_top_var, *top_choices)
        menu_bot = tk.OptionMenu(content_right, self.dropdown_bot_var, *bot_choices)
        menu_top.config(width=15)
        menu_bot.config(width=15)
        menu_top.pack(side=tk.TOP, pady=30) 
        menu_bot.pack(side=tk.TOP)


    # def update_plots(self, best_p, rewards): 
    #     """ each iteration """
    #     best_equation=None
    #     self.visualization.plot_vis(best_equation)

    #     self.equation.pack()

    #     self.training_nmse.plot
    #     self.training_best_reward.plot

    # def update_distribution(self, data):
    #     """ over several iterations """
    #     self.distribution
        
    def _init_frame_vis(self, frame, min=-100, max=100):

        self.visualization = Trace(frame, xlabel='X', ylabel='Y', title='DSR results', colors=['brown'], figsize=(3.9,3.9), dpi=100)

        buttons = tk.Frame(frame)
        self.vis_zoom_in = tk.Button(buttons, text="+", bg="green")
        self.vis_zoom_reset = tk.Button(buttons, text="zoom")
        self.vis_zoom_out = tk.Button(buttons, text="-")


        self.visualization.ax.set_xlim(min, max)
        self.visualization.ax.set_ylim(min, max)
        self.visualization.min = -100
        self.visualization.max = 100
        self.visualization.data_points = None

        # include data points in range min,max
        # self.visualization.plot_vis()

        """ pack vis"""
        buttons.pack()
        self.vis_zoom_out.pack(side=tk.RIGHT, fill=tk.X, expand=1)
        self.vis_zoom_reset.pack(side=tk.RIGHT, fill=tk.X, expand=1)
        self.vis_zoom_in.pack(side=tk.RIGHT, fill=tk.X, expand=1)

        self.visualization.pack(fill=tk.BOTH)
        # frame.pack(expand=1)

    def _init_frame_vis_info(self, frame):
        self.best_equation_var = tk.StringVar()
        self.best_equation_var.set("N/A")
        self.time_step = tk.StringVar()
        self.time_step.set("2000")
        self.best_reward_var = tk.StringVar()
        self.best_reward_var.set("1.00E+00")
        self.best_nmse = tk.StringVar()
        self.best_nmse.set("0.00E+00")

        # tk.Label(frame, image=tk.PhotoImage(eq_file))
        tk.Label(frame, text="Training timesteps:", fg=COLOR_VISINFO).grid(column=0, row=0)
        tk.Label(frame, textvariable=self.time_step).grid(column=1, row=0)
        tk.Label(frame, text="Best Equation:", fg=COLOR_VISINFO).grid(column=0, row=1)
        tk.Label(frame, textvariable=self.best_equation_var).grid(column=1, row=1)
        tk.Label(frame, text="Best Reward:", fg=COLOR_VISINFO).grid(column=0, row=2)
        tk.Label(frame, textvariable=self.best_reward_var).grid(column=1, row=2)
        tk.Label(frame, text="Best NMSE:", fg=COLOR_VISINFO).grid(column=0, row=3)
        tk.Label(frame, textvariable=self.best_nmse).grid(column=1, row=3)

    def _init_control(self, frame):
        self.start_button = tk.Button(frame, text="Start", bg='green')
        self.step_button = tk.Button(frame, text="Step", bg='green')
        self.stop_button = tk.Button(frame, text="Stop", bg='green', state="disabled")

        self.start_button.pack(ipadx=25, ipady=10, side=tk.LEFT, fill=tk.X)
        self.step_button.pack(ipadx=25, ipady=10, side=tk.LEFT, fill=tk.X)
        self.stop_button.pack(ipadx=25, ipady=10, side=tk.LEFT, fill=tk.X) 

    def _init_config(self, frame):
        fr_upload = tk.Frame(frame)
        fr_library = tk.Frame(frame)
        fr_sliders = tk.Frame(frame)

        fr_upload.pack(pady=10)
        fr_library.pack(pady=20)
        fr_sliders.pack()

        ### upload ### 
        tk.Label(fr_upload, text="Data", font=FONT_CONFIG).pack(side=tk.LEFT)
        self.data_input = tk.Entry(fr_upload)
        self.data_input_button = tk.Button(fr_upload, text="Upload...")

        self.data_input.pack(side=tk.LEFT)
        self.data_input_button.pack(side=tk.LEFT)

        ### library ###
        token_library = [["add","sub","mul","div"],["sin","cos","tan","exp","log"]]
        self.buttons_lib = [[tk.Checkbutton(fr_library, text=token, onvalue=True) for token in token_set] for token_set in token_library]

        # onevar = tk.BooleanVar()
        # onevar.set(True)
        # one = tk.Checkbutton(fr_library, text="add", variable=onevar, onvalue=True)

        """ pack""" 
        tk.Label(fr_library, text="Library", font=FONT_CONFIG).grid(column=0, row=0, rowspan=len(token_library))
        for row, button_set in enumerate(self.buttons_lib):
            for col, button in enumerate(button_set):
                button.grid(column=col+1, row=row, sticky=tk.W)

        ### sliders ###
        self.slide_explore = tk.Scale(fr_sliders, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL,  length=200)
        self.slide_risk = tk.Scale(fr_sliders, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=200) 
        self.slide_len_eq = tk.Scale(fr_sliders, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL, length=200)
        self.slide_batch = tk.Scale(fr_sliders, from_=50, to=1000, resolution=50, orient=tk.HORIZONTAL, length=200)
        self.slide_lr = tk.Scale(fr_sliders, from_=0.0001, to=0.01, resolution=0.0001, orient=tk.HORIZONTAL, length=200)

        tk.Label(fr_sliders, text="Batch size",  font=FONT_CONFIG).grid(row=0, column=0, rowspan=2, sticky=tk.E+tk.S)
        tk.Label(fr_sliders, text="Learning rate",  font=FONT_CONFIG).grid(row=2, column=0, rowspan=2, sticky=tk.E+tk.S)
        tk.Label(fr_sliders, text="Exploration",  font=FONT_CONFIG).grid(row=4, column=0, rowspan=2, sticky=tk.E+tk.S)
        tk.Label(fr_sliders, text="Risk Factor",  font=FONT_CONFIG).grid(row=6, column=0, rowspan=2, sticky=tk.E+tk.S)
        tk.Label(fr_sliders, text="Max length", font=FONT_CONFIG).grid(row=8, column=0, rowspan=2, sticky=tk.E+tk.S)
        self.slide_batch.grid(row=0, column=1)
        self.slide_lr.grid(row=2, column=1)
        self.slide_explore.grid(row=4, column=1)
        self.slide_risk.grid(row=6, column=1)
        self.slide_len_eq.grid(row=8, column=1)

##### PLOT TIME STEPS #####
# Prints time series information
# If speedup is needed, can refactor so that a single Trace object has multiple subplots.
class Trace(FigureCanvasTkAgg):
    def __init__(self, parent, xlabel=None, ylabel=None, title=None, colors=None, *args, **kwargs):
        self.length = 400 # Length (in time steps) of the plot at any given time (411 maps to 48 hr)
        self.shift = self.length/4 # How far (in time steps) to shift the plot when it jumps

        # HACK FOR NOW. It should find parent's time.
        self.time = 0
        
        self.parent = parent
        self.lines = None
        self.history = None        
        self.figure = Figure(*args, **kwargs)
        self.figure.set_tight_layout(True)
        self.ax = self.figure.add_subplot(111)        
        FigureCanvasTkAgg.__init__(self, self.figure, self.parent)
        #self.show()        
        
        self.ax.set_xlim(0, self.length)
        self.ax.set_ylim(0, 100)

        if xlabel != None:
                self.ax.set_xlabel(xlabel)
        if ylabel != None:
                self.ax.set_ylabel(ylabel)
        if title != None:
                self.ax.set_title(title)
 
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
                        self.ymax = 100*(1 + np.ceil(new_ymax/100)) # Update to the next 100
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
        
    def plot_vis(self, p):
        """ visualization frame for equation plot """

        self.ax.clear()
        
        # Plot real data
        xs = Program.X_train
        ys = Program.y_train
        self.ax.scatter(xs,ys)
 
        # Plot expression       
        n = 1000
        xs = np.linspace(-2, 2, num=n).reshape(n, -1) # TBD: GENERATE FOR ALL INPUT VARIABLES
        ys = p.execute(xs)
        self.ax.plot(xs, ys)

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-2, 2)

        self.figure.canvas.draw()
    

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
        self.root = root
        self.model = Model()
        self.view = View(root)

        self.model.addCallback(self.update_views)

        self.stopped = True

        self.view.step_button.config(command=self.step)
        self.view.start_button.config(command=self.init_start)
        self.view.stop_button.config(command=self.stop)


    def init_start(self):
        """Called once when Start is clicked."""

        if self.stopped:
            self.stopped = False
            self.view.start_button["state"] = "disabled"
            self.view.step_button["state"] = "disabled"
            self.view.stop_button["state"] = "normal"
            self.start()

        else:
            return


    def start(self):
        """Called repeatedly after Start is clicked."""

        self.step()

        if not self.stopped:
            self.root.after(100, self.start)


    def step(self):
        self.model.step()

    # def stop_all(self):
    #     screenshot=pyautogui.screenshot()
    #     screenshot.save("screenshot.png")


    def stop(self):
        self.view.start_button["state"] = "normal"
        self.view.step_button["state"] = "normal"
        self.view.stop_button["state"] = "disabled"
        self.stopped = True


    def update_views(self, batch_rewards, p, training_info):
        # vis
        self.view.visualization.plot_vis(p)
        # vis_info
        expression = p.sympy_expr
        self.view.best_equation_var.set("TBD")

        # training plots (nmse, best_reward)
        self.view.training_nmse.plot(np.atleast_1d(training_info[0]))
        self.view.training_best_reward.plot(np.atleast_1d(p.r))


def main():
    root = tk.Tk()
    root.title("Deep symbolic regression")
    # root.withdraw()
    # root.attributes('-fullscreen', True)
    root.geometry("1500x700")
    app = Controller(root)
    root.mainloop()


if __name__ == '__main__':
    main()
