# import matplotlib

# matplotlib.use("TkAgg")

from tkinter import *


# class Window(Frame):

#     def __init__(self, master=None):
#         Frame.__init__(self, master)               
#         self.master = master

# root = Tk()
# app = Window(root)
"""
root = Tk()
frame = Frame(root)
frame.pack()

bottomframe = Frame(root)
bottomframe.pack( side = BOTTOM )

# bottomframe2 = Frame(root)
# bottomframe2.pack( side = BOTTOM )

redbutton = Button(frame, text="Red", fg="red")
redbutton.pack( side = LEFT)

greenbutton = Button(frame, text="green", fg="green")
greenbutton.pack( side = LEFT )

bluebutton = Button(frame, text="Blue", fg="blue")
bluebutton.pack( side = LEFT )

blackbutton = Button(bottomframe, text="Black", fg="black")
blackbutton.pack()

root.mainloop()
"""
"""Demonstration of deep symbolic regression."""

import tkinter as tk
from tkinter import ttk


class Model:
    """Class for the DSR backend."""

    def __init__(self):
        pass

    def step():
        """Perform one iteration of DSR"""
        pass


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
        self.content_top = ttk.Frame(self.root)
        self.content_bottom = ttk.Frame(self.root)
        # self.content_bottom.pack()
        
        ### TOP ###
        frame = ttk.Frame(self.content_top, borderwidth=20, width=300, height=300)
        namelbl = tk.Label(self.content_top, text="Deep symbolic regression")
        # tk.Label(content, text="Deep symbolic regression").pack(side="left")
        name = ttk.Entry(self.content_top)

        ### BOTTOM ###
        onevar = tk.BooleanVar()
        twovar = tk.BooleanVar()
        threevar = tk.BooleanVar()
        onevar.set(True)
        twovar.set(False)
        threevar.set(True)

        one = ttk.Checkbutton(self.content_bottom, text="One", variable=onevar, onvalue=True)
        two = ttk.Checkbutton(self.content_bottom, text="Two", variable=twovar, onvalue=True)
        three = ttk.Checkbutton(self.content_bottom, text="Three", variable=threevar, onvalue=True)
        ok = ttk.Button(self.content_bottom, text="Okay")
        cancel = ttk.Button(self.content_bottom, text="Cancel")

        ### GRIDDDDD ###
        self.content_top.pack(side=tk.TOP)
        frame.grid(column=0, row=0, columnspan=2)
        namelbl.grid(column=2, row=0)
        name.grid(column=2, row=1)
        
        self.content_bottom.pack(side=tk.BOTTOM)       
        one.grid(column=0, row=3)
        two.grid(column=1, row=3)
        three.grid(column=2, row=3)
        ok.grid(column=3, row=3)
        cancel.grid(column=4, row=2)   



class Controller:
    """Class for uploading data and configuring runs."""

    def __init__(self, root):
        self.model = Model()
        self.view = View(root)


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