import os
import numpy as np
import logging
import math
from PIL import Image, ImageOps
from tile.database import TileDatabase, DatabaseConfig
from scipy.optimize import linear_sum_assignment
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class TileFitter:
    def __init__(self, overlay_image_path, database_file, output_file_path, grid_size, overlay_alpha=0.03, dpi=300):
        self.overlay_image_path = overlay_image_path
        self.database_file = database_file
        self.output_file_path = output_file_path
        self.grid_size = grid_size
        self.overlay_alpha = overlay_alpha
        self.dpi = dpi
        self.db = None
        self.tile_size = None

    def _load_data(self):
        self.db: DatabaseConfig = TileDatabase.load(self.database_file)
        h = int(self.db.tile_width / self.db.tile_ratio)
        w = self.db.tile_width
        self.tile_size = (w, h)

    def _validate_inputs(self):
        if len(self.overlay_image_path) == 0:
            logging.error('Overlay image path is empty.')
            return False
        if not os.path.isfile(self.overlay_image_path):
            logging.error(f'Overlay image path does not exist "{self.overlay_image_path}"')
            return False
        if not isinstance(self.grid_size, int):
            logging.error(f'Not valid integer provided for tile multiplier "{self.grid_size}"')
            return False
        if self.grid_size < 1 or self.grid_size > 2000:
            logging.error(f'Provided tile multiplier "{self.grid_size}" but allowed range is between 1 and 2000')
            return False
        if not isinstance(self.overlay_alpha, float):
            logging.error(f'Not valid float provided for overlay alpha "{self.overlay_alpha}"')
            return False
        if self.overlay_alpha < 0 or self.overlay_alpha > 1:
            logging.error(f'Provided alpha "{self.overlay_alpha}" but allowed range is between 0.0 and 1.0')
            return False
        if not isinstance(self.dpi, int):
            logging.error(f'Not valid integer provided for dpi "{self.dpi}"')
            return False
        if self.dpi < 30 or self.dpi > 600:
            logging.error(f'Provided dpi "{self.dpi}" but allowed range is between 30 and 600')
            return False
        return True

    def run(self):
        logging.info("Run TileFitter")
        if not self._validate_inputs():
            return
        self._load_data()
        overlay_img = self.__prepare_overlay_size()
        if overlay_img is not None and self.db is not None:
            db_rgb = TileDatabase.db_to_rgb_array(self.db)
            template = self._get_tile_template(overlay_img)
            # split up into different jobs
            len_temp = template.shape[0]
            len_db = db_rgb.shape[0]
            multiplier = math.ceil(len_temp / len_db)
            # method 1: increase database to same size of needed tile images
            if self.grid_size > 60:
                logging.warning('Processing could take a while or reduce tile multiplier!')
            if multiplier > 1:
                logging.warning(f'Not enough images in database. {len_db} exist but {len_temp} needed')
                logging.info(f'Images are {multiplier} times repeated')
                db_rgb_resized = np.stack([db_rgb for _ in range(multiplier)], axis=0).reshape(
                    (db_rgb.shape[0] * multiplier, db_rgb.shape[1]))
            cost = self.__calc_error_matrix(template, db_rgb_resized)
            logging.info('Run linear sum assignment problem')
            row_ind, col_ind = linear_sum_assignment(cost)

            logging.info(f'Create image with best tiles {(self.grid_size * self.tile_size[0], self.grid_size * self.tile_size[1])}')
            out = Image.new('RGB', (self.grid_size * self.tile_size[0], self.grid_size * self.tile_size[1]))
            for idx, img_idx in zip(row_ind, col_ind):
                col = idx % self.grid_size
                row = idx // self.grid_size
                row_px = col * self.tile_size[0]
                col_px = row * self.tile_size[1]
                tile_img = self.db.tiles[img_idx % len(db_rgb)].image
                out.paste(tile_img, (row_px, col_px))
           
            logging.info(f'Overlay image with alpha of {self.overlay_alpha*100}%')
            out = Image.blend(out, overlay_img, self.overlay_alpha)
            logging.info(f'Save image under "{self.output_file_path}"')
            out.save(self.output_file_path, dpi=(self.dpi, self.dpi))
            logging.info('Finished')
        else:
            logging.error('Overlay Image can not be processed or database is corrupt')

    def __calc_error_matrix(self, template, db_rgb):
        shape_cost_matrix = (template.shape[0], db_rgb.shape[0])
        cost_matrix = np.zeros(shape_cost_matrix)
        logging.info(f'Create cost matrix with shape {shape_cost_matrix}')
        for idx in range(template.shape[0]):
            single_cost = np.sum((template[idx] - db_rgb) ** 2, axis=1)
            cost_matrix[idx] = single_cost
        return cost_matrix

    def _get_tile_template(self, image):
        small = image.resize((self.grid_size, self.grid_size))
        tile_template = np.asarray(small) / 255
        s = tile_template.shape
        tile_template = tile_template.reshape((s[0] * s[1], s[2]))
        return tile_template

    def __prepare_overlay_size(self):
        if os.path.isfile(self.overlay_image_path):
            short_name = '..' + self.overlay_image_path[-50:] if len(self.overlay_image_path) > 50 else self.overlay_image_path
            logging.info(f'Prepare overlay "{short_name}"')
            img = Image.open(self.overlay_image_path)
            img = ImageOps.exif_transpose(img)
            # calc target least size in pixel
            w_target = int(self.tile_size[0] * self.grid_size)
            h_target = int(self.tile_size[1] * self.grid_size)
            w_current, h_current = img.size
            logging.info(f'Input image size: {img.size}')
            logging.info(f'Image target size: ({w_target}, {h_target})')
            w_k = w_target / w_current
            h_k = h_target / h_current
            if w_k > h_k:
                img = img.resize((w_target, int(h_current * w_k)))
            else:
                img = img.resize((int(w_current * h_k), h_target))
            logging.info(f'Input image resized: {img.size}')
            # crop to fit target size
            w, h = img.size
            if w > w_target:
                w_crop = (w - w_target) // 2
                img = img.crop((w_crop, 0, w - w_crop, h_target))
            else:
                h_crop = (h - h_target) // 2
                img = img.crop((0, h_crop, w_target, h - h_crop))
            logging.info(f'Input image cropped: {img.size}')
            return img.convert('RGB').crop((0, 0, w_target, h_target))
        else:
            logging.error(f'Invalid overlay image path given: {self.overlay_image_path}')

    def get_overlay(self):
        return self.__prepare_overlay_size()
