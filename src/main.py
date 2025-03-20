import tkinter as tk
from src.gui.interface import BiasDetectorApp

def main():
    root = tk.Tk()
    app = BiasDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()