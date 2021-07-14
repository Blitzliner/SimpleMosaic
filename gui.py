import tkinter as tk
from tkinter.filedialog import askdirectory, asksaveasfilename, askopenfilename
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from Mosaic import Mosaic
from ImageView import ImageView
import os
import shutil
import logging
import queue
import json
import threading

logger = logging.getLogger(__name__)


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)


class ViewDatabase:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.ratio_list = {'1:1 (1.0)': 1, '5:4 (1.25)': 5 / 4, '4:3 (1.333)': 4 / 3, '3:2 (1.5)': 3 / 2,
                           '16:10 (1.6)': 16 / 10, '5:3 (1.667)': 5 / 3, '16:9 (1.778)': 16 / 9}
        self.__view_database_settings()

    def __view_database_settings(self):
        frame = tk.LabelFrame(self.root, text="Database Settings")
        frame.grid(row=0, column=0, pady=5, padx=5, sticky='ewn')
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        # select image directory
        tk.Label(frame, text="Image directory", anchor='w').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.txtInputImageDir = tk.Label(frame, width=25, text=self.config['image_dir'], anchor='e')
        self.txtInputImageDir.grid(row=0, column=1, sticky='ew')
        tk.Button(frame, text="...", padx=5, command=self.__select_image_dir).grid(row=0, column=2, sticky='w', padx=5)
        # tile ratio
        tk.Label(frame, text="Tile aspect ratio", anchor='w').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        value_list = list(self.ratio_list.keys())
        self.txtTileRatio = ttk.Combobox(frame, values=value_list)
        self.txtTileRatio.current(value_list.index(self.config['tile_size_ratio']))
        self.txtTileRatio.grid(row=2, column=1, sticky='ew')
        # tile width
        tk.Label(frame, text="Tile size", anchor='w').grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.txtTileSize = tk.Entry(frame)
        self.txtTileSize.insert(0, self.config['tile_max_width'])
        tk.Label(frame, text="px").grid(row=3, column=2, sticky='w', padx=5)
        self.txtTileSize.grid(row=3, column=1, sticky='ew')
        # database path
        tk.Label(frame, text="Database file", anchor='w').grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.txtDatabaseFile = tk.Label(frame, width=25, text=self.config['database_file'], anchor='e')
        self.txtDatabaseFile.grid(row=4, column=1, sticky='ew')
        tk.Button(frame, text="...", padx=5, command=self.__select_database).grid(row=4, column=2, sticky='w', padx=5)
        # create database button
        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=3)
        tk.Button(btn_frame, text="Create/Update Database", padx=10, command=self.__create_database).grid(row=0,
                                                                                                          column=0,
                                                                                                          pady=5)

    def get_settings(self):
        return self.config

    def __select_image_dir(self):
        selected_path = askdirectory(initialdir=self.config['image_dir'])
        if len(selected_path):
            self.txtInputImageDir['text'] = selected_path

    def __select_database(self):
        selected_path = asksaveasfilename(initialdir=self.config['database_file'],
                                          filetypes=(("Database files", "*.p"), ("All files", "*.*")))
        if len(selected_path):
            self.txtDatabaseFile['text'] = selected_path

    def __create_database(self):
        image_dir = self.txtInputImageDir['text']
        tile_max_width = int(self.txtTileSize.get())
        tile_size_ratio_key = self.txtTileRatio.get()
        database_file = self.txtDatabaseFile['text']
        tdb = Mosaic.TileDatabase(image_dir=image_dir, database_file=database_file,
                                  tile_size_ratio=self.ratio_list[tile_size_ratio_key],
                                  tile_max_width=tile_max_width)
        thread = threading.Thread(target=tdb.create)
        thread.start()

        self.config['width'] = self.root.winfo_width()
        self.config['height'] = self.root.winfo_height()
        self.config['image_dir'] = image_dir
        self.config['tile_max_width'] = tile_max_width
        self.config['tile_size_ratio'] = tile_size_ratio_key
        self.config['database_file'] = database_file


class ViewTileFitter:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.__view_tile_fitter_settings()
        self.__view_image_preview()

    def get_settings(self):
        return self.config

    def __view_tile_fitter_settings(self):
        frame = tk.LabelFrame(self.root, text="Mosaic Settings")
        frame.grid(row=1, column=0, pady=5, padx=5, sticky='ewn')
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        # select database
        tk.Label(frame, text="Database", anchor='w').grid(row=0, sticky='w', padx=5, pady=5)
        self.txtDatabasePath = tk.Label(frame, width=25, text=self.config['database_file'], anchor='e')
        self.txtDatabasePath.grid(row=0, column=1, sticky='ew')
        tk.Button(frame, text="...", padx=5, command=self.__select_database).grid(row=0, column=2, sticky='w',
                                                                                       padx=5)
        # select overlay image
        tk.Label(frame, text="Overlay image", anchor='w').grid(row=1, sticky='w', padx=5, pady=5)
        self.txtOverlayImagePath = tk.Label(frame, width=25, text=self.config['overlay_image_path'], anchor='e')
        self.txtOverlayImagePath.grid(row=1, column=1, sticky='ew')
        tk.Button(frame, text="...", padx=5, command=self.__select_overlay_image).grid(row=1, column=2, sticky='w',
                                                                                       padx=5)
        # tile multiplier
        tk.Label(frame, text="Tile multiplier", anchor='w').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.txtTileMultiplier = tk.Entry(frame)
        self.txtTileMultiplier.insert(0, self.config['tile_multiplier'])
        self.txtTileMultiplier.grid(row=2, column=1, sticky='ew')
        # dpi
        tk.Label(frame, text="Image Resolution", anchor='w').grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.txtDpi = tk.Entry(frame)
        self.txtDpi.insert(0, self.config['dpi'])
        tk.Label(frame, text="dpi").grid(row=3, column=2, sticky='w', padx=5)
        self.txtDpi.grid(row=3, column=1, sticky='ew')
        # overlay
        tk.Label(frame, text="Overlay", anchor='w').grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.txtOverlayAlpha = tk.Entry(frame)
        self.txtOverlayAlpha.insert(0, self.config['overlay_alpha'] * 100)
        tk.Label(frame, text='%').grid(row=4, column=2, sticky='w', padx=5)
        self.txtOverlayAlpha.grid(row=4, column=1, sticky='ew')
        # button to export/display image
        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=3)
        tk.Button(btn_frame, text="Run", padx=10, command=self.__create_mosaic).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="Show", padx=10, command=self.__show_mosaic).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text="Save as", padx=10, command=self.__save_as_mosaic).grid(row=0, column=2, padx=5,
                                                                                          pady=5)

    def __create_mosaic(self):
        overlay_image_path = self.txtOverlayImagePath["text"]
        database_file = self.txtDatabasePath['text']
        tile_multiplier = int(self.txtTileMultiplier.get())
        dpi = int(self.txtDpi.get())
        overlay_alpha = float(self.txtOverlayAlpha.get()) / 100
        output_file_path = self.config['output_file_path']

        self.config['width'] = self.root.winfo_width()
        self.config['height'] = self.root.winfo_height()
        self.config['database_file'] = database_file
        self.config['overlay_image_path'] = overlay_image_path
        self.config['tile_multiplier'] = tile_multiplier
        self.config['overlay_alpha'] = overlay_alpha
        self.config['dpi'] = dpi
        self.config['output_file_path'] = output_file_path

        tf = Mosaic.TileFitter(overlay_image_path=overlay_image_path, database_file=database_file,
                               output_file_path=output_file_path, tile_multiplier=tile_multiplier,
                               overlay_alpha=overlay_alpha, dpi=dpi)
        thread = threading.Thread(target=tf.run)
        thread.start()

    def __show_mosaic(self):
        self.canvas.init(self.config['output_file_path'])

    def __save_as_mosaic(self):
        filepath = asksaveasfilename(filetypes=(("Image files", "*.jpg"), ("All files", "*.*")))
        if 0 == len(filepath):
            logging.error('Please select a valid filename')
            return
        elif not (filepath.endswith('.jpg') or filepath.endswith('.jepg')):
            logging.warning('No valid image extension found. I add ".jpg" for you.')
            filepath += '.jpg'

        temp_file = self.config['output_file_path']
        if os.path.isfile(temp_file):
            shutil.copy2(temp_file, filepath)
            logger.info(f'File is saved under {filepath}')
        else:
            logging.error('Please run first the mosaic fitter')

    def __select_database(self):
        selected_path = askopenfilename(initialdir=self.config['database_file'],
                                        filetypes=(("Database files", "*.p"), ("All files", "*.*")))
        if len(selected_path):
            self.txtDatabasePath['text'] = selected_path

    def __select_overlay_image(self):
        selected_path = askopenfilename(initialdir=self.config['overlay_image_path'],
                                        filetypes=(("Image files", "*.jpg;*.png"), ("All files", "*.*")))
        if len(selected_path):
            self.txtOverlayImagePath['text'] = selected_path

    def __view_image_preview(self):
        self.image_frame = tk.LabelFrame(self.root, text="Mosaic Preview")
        self.image_frame.grid(row=0, column=1, rowspan=2, pady=5, padx=5, sticky='ewns')
        self.image_frame.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.image_frame.columnconfigure(0, weight=1)
        self.canvas = ImageView.CanvasImage(self.image_frame)  # create widget
        self.canvas.grid(row=0, column=0)  # show widget


class ViewConsole:
    def __init__(self, root):
        self.root = root
        self.__view_console()

    def __poll_log_queue(self):
        while True:
            try:
                record = self.log_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self.__display(record)
        self.root.after(100,
                        self.__poll_log_queue)  # Check every 100ms if there is a new message in the queue to display

    def __display(self, record):
        msg = self.queue_handler.format(record)
        self.console.configure(state='normal')
        self.console.insert(tk.END, msg + '\n', record.levelname)
        self.console.configure(state='disabled')
        self.console.yview(tk.END)  # Autoscroll to the bottom

    def __view_console(self):
        self.console = ScrolledText(self.root, state='disabled')
        self.console.grid(row=2, column=0, columnspan=2, sticky='nesw', padx=5, pady=5)
        self.console.configure(font='TkFixedFont')
        self.console.tag_config('INFO', foreground='black')
        self.console.tag_config('DEBUG', foreground='gray')
        self.console.tag_config('WARNING', foreground='orange')
        self.console.tag_config('ERROR', foreground='red')
        self.console.tag_config('CRITICAL', foreground='red', underline=1)
        # Create a logging handler using a queue
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        self.queue_handler.setFormatter(
            logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', "%H:%M:%S"))
        logging.getLogger().addHandler(self.queue_handler)
        self.root.after(100, self.__poll_log_queue)  # Start polling messages from the queue


class App:
    def __init__(self, root):
        self.root = root
        self.default_config_path = 'config.cfg'
        self.config = self.__load_settings()
        self.__init_main_window()
        self.view_database = ViewDatabase(self.root, self.config)
        self.view_tile_fitter = ViewTileFitter(self.root, self.config)
        self.view_console = ViewConsole(self.root)
        # self.__view_image_preview(self.root)

    def save_settings(self):
        self.config = {**self.config, **self.view_database.get_settings()}
        self.config = {**self.config, **self.view_tile_fitter.get_settings()}
        with open(self.default_config_path, 'w') as file:
            return json.dump(self.config, file, indent=4, sort_keys=True)

    def __load_settings(self):
        if os.path.isfile(self.default_config_path):
            with open(self.default_config_path, 'r') as file:
                return json.load(file)
        else:
            # load default config
            return {'width': 800, 'height': 500, 'image_dir': 'Please select directory',
                    'tile_size_ratio': '4:3 (1.333)',
                    'tile_multiplie': 50, 'dpi': 150, 'tile_max_width': 250, 'output_file_path': '_temp.jpg',
                    'database_file': 'database.p', 'overlay_alpha': 0.03, 'overlay_image_path': 'Please select image'}

    def __init_main_window(self):
        self.root.title("MosaicMaker v0.1")
        width = self.config['width']
        height = self.config['height']
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        pos_x = int((screenwidth - width) / 2)
        pos_y = int((screenheight - height) / 2)
        self.root.geometry(f'{width}x{height}+{pos_x}+{pos_y}')
        self.root.resizable(width=True, height=True)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    app.save_settings()