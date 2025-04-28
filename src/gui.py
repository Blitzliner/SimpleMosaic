import tkinter as tk
from tkinter.filedialog import askdirectory, asksaveasfilename, askopenfilename
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from tile.database import TileDatabase
from tile.fitter import TileFitter
from gui import ImageView
from gui import ScrolledWindow
import os
import shutil
import logging
import queue
import json
import threading
from PIL import Image
from config import ImageAspectRatio

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
        self.tread = None
        # self.aspect_ratio = ImageAspectRatio.list()
        # self.ratio_list = {
        #     '1:1 (1.0)': 1,
        #     '5:4 (1.25)': 5 / 4,
        #     '4:3 (1.33)': 4 / 3,
        #     '3:2 (1.5)': 3 / 2,
        #     '16:10 (1.6)': 16 / 10,
        #     '5:3 (1.66)': 5 / 3,
        #     '16:9 (1.78)': 16 / 9,
        # }
        self._view_database_settings()

    def _view_database_settings(self):
        frame = tk.LabelFrame(self.root, text='Database Settings')
        frame.grid(row=0, column=0, pady=5, padx=5, sticky='enw')
        frame.columnconfigure(0, weight=0, minsize=100)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        # select image directory
        tk.Label(frame, text='Image directory', anchor='w').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.txtInputImageDir = tk.Label(frame, width=25, text=self.config['image_dir'], anchor='e')
        self.txtInputImageDir.grid(row=0, column=1, sticky='ew')
        tk.Button(frame, text='...', padx=5, command=self._btn_select_image_dir).grid(
            row=0, column=2, sticky='w', padx=5
        )
        # tile ratio
        tk.Label(frame, text='Tile aspect ratio', anchor='w').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        value_list = ImageAspectRatio.list() # list(self.ratio_list.keys())
        self.txtTileRatio = ttk.Combobox(frame, values=value_list)
        self.txtTileRatio.current(value_list.index(self.config['tile_aspect_ratio']))
        self.txtTileRatio.grid(row=2, column=1, sticky='ew')
        # tile width
        tk.Label(frame, text='Tile width', anchor='w').grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.txtTileSize = tk.Entry(frame)
        self.txtTileSize.insert(0, self.config['tile_max_width'])
        tk.Label(frame, text='px').grid(row=3, column=2, sticky='w', padx=5)
        self.txtTileSize.grid(row=3, column=1, sticky='ew')
        # database path
        tk.Label(frame, text='Database file', anchor='w').grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.txtDatabaseFile = tk.Label(frame, width=25, text=self.config['database_file'], anchor='e')
        self.txtDatabaseFile.grid(row=4, column=1, sticky='ew')
        tk.Button(frame, text='...', padx=5, command=self._btn_select_database).grid(
            row=4, column=2, sticky='w', padx=5
        )
        # create database button
        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=3)
        tk.Button(btn_frame, text='Create', padx=10, command=self._btn_create_database).grid(row=0, column=0, pady=5)

    def get_settings(self):
        return self.config

    def _btn_select_image_dir(self):
        selected_path = askdirectory(initialdir=self.config['image_dir'])
        if len(selected_path):
            self.txtInputImageDir['text'] = selected_path

    def _btn_select_database(self):
        selected_path = asksaveasfilename(
            initialdir=self.config['database_file'],
            filetypes=(('Database files', '*.p'), ('All files', '*.*')),
        )
        if len(selected_path):
            self.txtDatabaseFile['text'] = selected_path

    def _btn_create_database(self):
        tile_dir = self.txtInputImageDir['text']
        tile_width = int(self.txtTileSize.get())
        tile_aspect_ratio_key = self.txtTileRatio.get()
        database_path = self.txtDatabaseFile['text']
        
        tdb = TileDatabase(tile_dir=tile_dir,
                           database_path=database_path,
                           tile_ratio=ImageAspectRatio.from_string(tile_aspect_ratio_key).to_float(),
                           tile_width=tile_width)

        if self.tread and self.tread.is_alive():
            logger.warning('Please wait until database has been created')
        else:
            self.thread = threading.Thread(target=tdb.create)
            self.thread.start()

        self.config['image_dir'] = tile_dir
        self.config['tile_max_width'] = tile_width
        self.config['tile_aspect_ratio'] = tile_aspect_ratio_key
        self.config['database_file'] = database_path


class ViewTileFitter:
    def __init__(self, root, settings_frame, config):
        self.config = config
        self.thread = None
        self.tf = None
        self._view_fitter_settings(settings_frame)
        self._view_image_settings(settings_frame)
        self._view_image_preview(root)

    def get_settings(self):
        return self.config

    def _view_fitter_settings(self, root):
        frame = tk.LabelFrame(root, text='Mosaic Fitter')
        frame.grid(row=1, column=0, pady=5, padx=5, sticky='enw')
        frame.columnconfigure(0, weight=0, minsize=100)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        frame.columnconfigure(3, weight=0)
        # select database
        tk.Label(frame, text='Database', anchor='w').grid(row=0, sticky='w', padx=5, pady=5)
        self.txtDatabasePath = tk.Label(frame, width=25, text=self.config['database_file'], anchor='e')
        self.txtDatabasePath.grid(row=0, column=1, sticky='ew')
        tk.Button(frame, text='...', padx=5, command=self._btn_select_database).grid(
            row=0, column=2, sticky='w', padx=5
        )
        # select overlay image
        tk.Label(frame, text='Overlay image', anchor='w').grid(row=1, sticky='w', padx=5, pady=5)
        self.txtOverlayImagePath = tk.Label(frame, width=25, text=self.config['overlay_image_path'], anchor='e')
        self.txtOverlayImagePath.grid(row=1, column=1, sticky='ew')
        tk.Button(frame, text='...', padx=5, command=self._btn_select_overlay).grid(
            row=1, column=2, sticky='w', padx=5
        )
        # tile multiplier
        tk.Label(frame, text='Grid size', anchor='w').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.txtGridSize = tk.Entry(frame)
        self.txtGridSize.insert(0, self.config['grid_size'])
        self.txtGridSize.grid(row=2, column=1, sticky='ew')
        self.txtGridSize.bind("<KeyRelease>", self._update_resolution)

        # Add label to display image resolution
        self.lblResolution = tk.Label(frame, text="Multiplier: 0, Resolution: 0 px")
        self.lblResolution.grid(row=3, column=1, columnspan=2, sticky='w')
        self._update_resolution()  # call it once to set the initial value

        # button to export/display image
        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=3)
        tk.Button(btn_frame, text='Run Fitter', padx=10, command=self._btn_fitter).grid(
            row=0, column=0, padx=5, pady=5
        )
        tk.Button(btn_frame, text='Show', padx=10, command=self._btn_show).grid(row=0, column=1, padx=5, pady=5)

    def _update_resolution(self, *args):
        try:
            gridSize = int(self.txtGridSize.get())
            width = int(gridSize * self.config['tile_max_width'])
            tile_height = int(self.config['tile_max_width'] / ImageAspectRatio.from_string(self.config['tile_aspect_ratio']).to_float())
            height = int(gridSize * tile_height)
            self.lblResolution.config(text=f"Resolution: {width} x {height} px")
        except ValueError:
            self.lblResolution.config(text="Invalid grid size.")

    def _view_image_settings(self, root):
        frame = tk.LabelFrame(root, text='Image Settings')
        frame.grid(row=2, column=0, pady=5, padx=5, sticky='enw')
        frame.columnconfigure(0, weight=0, minsize=100)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        # dpi
        tk.Label(frame, text='Resolution', anchor='w').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.txtDpi = tk.Entry(frame)
        self.txtDpi.insert(0, self.config['dpi'])
        tk.Label(frame, text='dpi').grid(row=0, column=2, sticky='w', padx=5)
        self.txtDpi.grid(row=0, column=1, sticky='ew')
        # overlay
        tk.Label(frame, text='Overlay', anchor='w').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.txtOverlayAlpha = tk.Entry(frame)
        self.txtOverlayAlpha.insert(0, self.config['overlay_alpha'] * 100)
        tk.Label(frame, text='%').grid(row=1, column=2, sticky='w', padx=5)
        self.txtOverlayAlpha.grid(row=1, column=1, sticky='ew')
        # checkbox for advanced settings
        cb_frame = tk.Frame(frame)
        cb_frame.grid(row=2, column=0, columnspan=3)

        self.cbGrayScale = ttk.Checkbutton(cb_frame, text='Grayscale')
        self.cbGrayScale.state(['!alternate'])
        if self.config['grayscale_active']:
            self.cbGrayScale.state(['selected'])
        else:
            self.cbGrayScale.state(['!selected'])
        self.cbGrayScale.grid(row=0, column=0, sticky='w', padx=5, pady=5)

        # button to export/display image
        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=3, column=0, columnspan=3)
        tk.Button(btn_frame, text='Apply', padx=10, command=self.__btn_update).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text='Save as', padx=10, command=self._btn_save_as).grid(row=0, column=2, padx=5, pady=5)

    def _btn_fitter(self):
        if self.thread:
            if self.thread.is_alive():
                logger.warning('Please wait until mosaic has been created')
                return
        self.config['overlay_image_path'] = self.txtOverlayImagePath['text']
        self.config['database_file'] = self.txtDatabasePath['text']
        self.config['grid_size'] = int(self.txtGridSize.get())

        self.tf = TileFitter(
            overlay_image_path=self.config['overlay_image_path'],
            database_file=self.config['database_file'],
            output_file_path=self.config['fitter_out_file'],
            grid_size=self.config['grid_size'],
            overlay_alpha=0.0,
            dpi=300,
        )
        self.thread = threading.Thread(target=self.tf.run)
        self.thread.start()

    def _show(self, path):
        if self.thread:
            if self.thread.is_alive():
                logging.warning('Please wait until mosaic has been created')
            else:
                logging.info('Prepare image preview')
                t = threading.Thread(target=self.canvas.init, args=(path,))
                t.start()
        elif os.path.isfile(path):
            logging.info('Prepare image preview')
            t = threading.Thread(target=self.canvas.init, args=(path,))
            t.start()
        else:
            logging.warning('No image exist. Please create mosaic first.')

    def _btn_show(self):
        self._show(self.config['fitter_out_file'])

    def __btn_update(self):
        try:
            self.config['dpi'] = int(self.txtDpi.get())
        except:
            logging.error(f'Wrong data format only integer are allowed: {self.txtDpi.get()}')
            return
        try:
            self.config['overlay_alpha'] = float(self.txtOverlayAlpha.get()) / 100
        except:
            logging.error(f'Wrong data format only float are allowed: {self.txtDpi.get()}')
            return
        if self.config['overlay_alpha'] < 0 or self.config['overlay_alpha'] > 1:
            logging.error(f'Provided alpha "{self.config["overlay_alpha"]}" but allowed range is between 0.0 and 1.0')
            return False
        if self.config['dpi'] < 30 or self.config['dpi'] > 600:
            logging.error(f'Provided dpi "{self.config["dpi"]}" but allowed range is between 30 and 600')
            return False

        self.config['grayscale_active'] = self.cbGrayScale.instate(['selected'])

        if os.path.isfile(self.config['fitter_out_file']):
            if self.tf:
                overlay = self.tf.get_overlay()
                img = Image.open(self.config['fitter_out_file'])
                if self.config['grayscale_active']:
                    overlay = overlay.convert('L')
                    img = img.convert('L')
                logger.info(f'Create overlay with {self.config["overlay_alpha"]}')
                w, h = img.size
                w_cm = int(w / self.config['dpi'] * 2.54)
                h_cm = int(h / self.config['dpi'] * 2.54)
                logger.info(f'Image can be printed with {self.config["dpi"]} dpi in size of {w_cm}x{h_cm} cm')
                new_image = Image.blend(img, overlay, self.config['overlay_alpha'])
                new_image.save(
                    self.config['overlay_out_file'],
                    dpi=(self.config['dpi'], self.config['dpi']),
                )
                self._show(self.config['overlay_out_file'])
            else:
                logger.warning('Please run fitter first')
        else:
            logger.error(f'Image does not exist under {self.config["fitter_out_file"]}. Please run fitter first')

    def _btn_save_as(self):
        filepath = asksaveasfilename(filetypes=(('Image files', '*.jpg'), ('All files', '*.*')))
        if 0 == len(filepath):
            logging.error('Please select a valid filename')
            return
        elif not (filepath.endswith('.jpg') or filepath.endswith('.jepg')):
            logging.warning('No valid image extension found. I add ".jpg" for you.')
            filepath += '.jpg'

        temp_file = self.config['fitter_out_file']
        if os.path.isfile(temp_file):
            shutil.copy2(temp_file, filepath)
            logger.info(f'File is saved under {filepath}')
        else:
            logging.error('Please run first the mosaic fitter')

    def _btn_select_database(self):
        selected_path = askopenfilename(
            initialdir=self.config['database_file'],
            filetypes=(('Database files', '*.p'), ('All files', '*.*')),
        )
        if len(selected_path):
            self.txtDatabasePath['text'] = selected_path

    def _btn_select_overlay(self):
        selected_path = askopenfilename(
            initialdir=self.config['overlay_image_path'],
            filetypes=(('Image files', '*.jpg;*.png'), ('All files', '*.*')),
        )
        if len(selected_path):
            self.txtOverlayImagePath['text'] = selected_path

    def _view_image_preview(self, root):
        self.image_frame = tk.LabelFrame(root, text='Mosaic Preview')
        self.image_frame.grid(row=0, column=1, pady=5, padx=5, sticky='ewns')
        self.image_frame.rowconfigure(0, weight=1)
        self.image_frame.columnconfigure(0, weight=1)
        self.canvas = ImageView.CanvasImage(self.image_frame)
        self.canvas.grid(row=0, column=0)


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
        self.root.after(100, self.__poll_log_queue)  # Check every 100ms if there is a new message in the queue

    def __display(self, record):
        msg = self.queue_handler.format(record)
        self.console.configure(state='normal')
        self.console.insert(tk.END, msg + '\n', record.levelname)
        self.console.configure(state='disabled')
        self.console.yview(tk.END)  # Autoscroll to the bottom

    def __view_console(self):
        self.console = ScrolledText(self.root, state='disabled', height=10)
        self.console.grid(row=1, column=0, columnspan=2, sticky='nesw', padx=5, pady=5)
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
            logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', '%H:%M:%S')
        )
        logging.getLogger().addHandler(self.queue_handler)
        self.root.after(100, self.__poll_log_queue)  # Start polling messages from the queue


class App:
    def __init__(self, root):
        self.root = root
        self.default_config_path = 'config.cfg'
        self.config = self._load_settings()
        self._init_main_window()
        settings_view = ScrolledWindow.ScrolledWindow(self.root)
        self.view_database = ViewDatabase(settings_view.scrollwindow, self.config)
        self.view_tile_fitter = ViewTileFitter(self.root, settings_view.scrollwindow, self.config)
        self.view_console = ViewConsole(self.root)

    def save_settings(self):
        self.config = {**self.config, **self.view_database.get_settings()}
        self.config = {**self.config, **self.view_tile_fitter.get_settings()}

        with open(self.default_config_path, 'w') as file:
            return json.dump(self.config, file, indent=4, sort_keys=True)

    def _load_settings(self):
        if os.path.isfile(self.default_config_path):
            with open(self.default_config_path, 'r') as file:
                return json.load(file)
        else:
            logger.warning('No settings file found. Empty one with default values is created')
            # load default config
            return {
                'width': 800,
                'height': 600,
                'image_dir': 'Please select directory',
                'tile_aspect_ratio': ImageAspectRatio.FOUR_THREE.value,
                'grid_size': 50,
                'dpi': 150,
                'tile_max_width': 250,
                'fitter_out_file': '_temp.jpg',
                'database_file': 'database.p',
                'overlay_alpha': 0.03,
                'overlay_image_path': 'Please select image',
                'grayscale_active': False,
                'overlay_out_file': '_tempOverlay.jpg',
            }

    def _init_main_window(self):
        self.root.title('MosaicMaker v0.1')
        width = self.config['width']
        height = self.config['height']
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        pos_x = int((screenwidth - width) / 2)
        pos_y = int((screenheight - height) / 2)
        self.root.geometry(f'{width}x{height}+{pos_x}+{pos_y}')
        self.root.resizable(width=True, height=True)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)

    def on_closing():
        app.config['width'] = root.winfo_width()
        app.config['height'] = root.winfo_height()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', on_closing)
    root.mainloop()
    app.save_settings()
