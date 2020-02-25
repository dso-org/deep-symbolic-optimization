"""Demonstration of deep symbolic regression."""

import tkinter as tk


class Model:
    """Class for the DSR backend."""

    def __init__(self):
        pass

    def step():
        """Perform one iteration of DSR"""
        pass


class View(tk.Toplevel):
    """Class for plots and diagnostics."""

    def __init__(self, master):
        tk.Toplevel.__init__(self, master)
        self.protocol('WM_DELETE_WINDOW', self.master.destroy)
        tk.Label(self, text="Deep symbolic regression").pack(side="left")


class Controller:
    """Class for uploading data and configuring runs."""

    def __init__(self, root):
        self.model = Model()
        self.view = View(root)


def main():
    root = tk.Tk()
    root.withdraw()
    app = Controller(root)
    root.mainloop()


if __name__ == '__main__':
    main()