# -*- coding: utf-8 -*-
# Advanced zoom for images of various types from small to huge up to several GB
import math
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import logging

logger = logging.getLogger(__name__)


class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)


class CanvasImage:
    """ Display and zoom image """

    def __init__(self, placeholder, filter=Image.ANTIALIAS):
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        self.placeholder = placeholder
        self.__filter = filter  # Image.ANTIALIAS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for the big image
        self.__band_width = 1024  # width of the tile band
        self.__huge_size = 14000  # define size of the huge image
        # Create ImageFrame in placeholder widget
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B1-Motion>', self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.__wheel)  # zoom for Linux, wheel scroll up

        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        # self.init()
        self.__delta = 1.3  # zoom magnitude
        self.__previous_state = 0  # previous state of the keyboard
        self.path = None  # path to the image, should be public for outer classes
        self.__image = None
        self.imwidth = None
        self.imheight = None
        self.imscale = 1.0
        self.__pyramid = []
        self.__ratio = 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = 1.0  # image pyramide scale
        self.__reduction = 2
        self.container = self.canvas.create_rectangle((0, 0, 10, 10), width=0)

    def init(self, path):
        logger.info(f'Init image preview for "{path}"')
        self.__delta = 1.3  # zoom magnitude
        self.__previous_state = 0  # previous state of the keyboard
        self.path = path  # path to the image, should be public for outer classes
        self.__image = Image.open(self.path)  # open image, but down't load it
        self.imwidth, self.imheight = self.__image.size  # public for outer class
        c_dist = 24  # correction distance to remove scrollbar
        scale_x = (self.placeholder.winfo_width() - c_dist) / self.imwidth
        scale_y = (self.placeholder.winfo_height() - c_dist) / self.imheight
        self.imscale = min(scale_x, scale_y)  # instead of 1

        # Create image pyramid
        self.__pyramid = [self.__image]
        # Set ratio coefficient for image pyramid
        self.__ratio = 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w = w // self.__reduction  # divide on reduction degree
            h = h // self.__reduction  # divide on reduction degree
            logger.info(f'Image pyramid: Reduce to {w}x{h}')
            self.__pyramid.append(self.__pyramid[-1].resize((w, h), self.__filter))

        # Put image into container rectangle and use it to set proper coordinates to the image
        fitted_width = self.imwidth * self.__scale
        fitted_height = self.imheight * self.__scale
        off_x = (self.placeholder.winfo_width() - fitted_width) / 2
        off_y = (self.placeholder.winfo_height() - fitted_height) / 2
        self.container = self.canvas.create_rectangle((off_x, off_y, fitted_width + off_x, fitted_height + off_y),
                                                      width=0)
        logger.info('Show image')
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        if self.path is None:
            return
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        # Vertical part of the image is in the visible area
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                (int(x1 / self.__scale), int(y1 / self.__scale),
                 int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        if self.path is None:
            return
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        __min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        if self.outside(x, y):
            return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down, smaller
            if round(__min_side * self.imscale) < 30:
                return  # image is less than 30 pixels
            self.imscale /= self.__delta
            scale /= self.__delta
        if event.num == 4 or event.delta == 120:  # scroll up, bigger
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.__delta
            scale *= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right: keys 'D', 'Right' or 'Numpad-6'
                self.__scroll_x('scroll', 1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left: keys 'A', 'Left' or 'Numpad-4'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up: keys 'W', 'Up' or 'Numpad-8'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down: keys 'S', 'Down' or 'Numpad-2'
                self.__scroll_y('scroll', 1, 'unit', event=event)

    def crop(self, bbox):
        """ Crop rectangle from the image and return it """
        return self.__pyramid[0].crop(bbox)

    def destroy(self):
        """ ImageFrame destructor """
        self.__image.close()
        map(lambda i: i.close, self.__pyramid)  # close all pyramid images
        del self.__pyramid[:]  # delete pyramid list
        del self.__pyramid  # delete pyramid variable
        self.canvas.destroy()
        self.__imframe.destroy()


if __name__ == "__main__":
    class MainWindow(ttk.Frame):
        """ Main window class """

        def __init__(self, mainframe):
            """ Initialize the main Frame """
            ttk.Frame.__init__(self, master=mainframe)
            self.master.title('Advanced Zoom v3.0')
            self.master.geometry('800x300')  # size of the main window
            self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
            self.master.columnconfigure(0, weight=1)
            self.canvas = CanvasImage(self.master)  # create widget
            self.canvas.grid(row=0, column=0)  # show widget


    filename = '../_temp.jpg'
    app = MainWindow(tk.Tk())
    app.canvas.init(filename)
    app.mainloop()
