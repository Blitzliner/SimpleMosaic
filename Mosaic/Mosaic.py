import os
import numpy as np
import glob
import logging
import multiprocessing
import math
import pickle
from PIL import Image, ImageOps
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import linear_sum_assignment
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class TileDatabase:
    def __init__(self, image_dir: str, database_file: str, tile_size_ratio: float, tile_max_width: int):
        self.image_dir = image_dir
        self.database_file = database_file
        self.tile_size_ratio = tile_size_ratio
        self.tile_max_width = tile_max_width
        self.progress = 0
        self.database = None

    def _get_file_list(self, file_extensions: str = '*.jpg') -> list:
        search_pattern = os.path.join(self.image_dir, '**', file_extensions)
        logging.info(f'Search for images in "{search_pattern}"')
        all_paths = list(glob.iglob(search_pattern, recursive=True))
        logging.info(f'Found {len(all_paths)} images')
        return all_paths

    def _reduce_file_list(self, path_list: list) -> list:
        if self.database.get('files') and self.database.get('tile_size_ratio') == self.tile_size_ratio \
                and self.database.get('tile_max_width') == self.tile_max_width:
            logging.info('Database already exists with same attributes')
            path_list = [path for path in path_list if os.path.basename(path) not in self.database['files']]
        return path_list
    
    def _prepare_database(self) -> None:
        if os.path.isfile(self.database_file):
            self.database = TileDatabase.load(self.database_file)
            if self.database is None:
                self._init_database()
        else:
            logging.info(f'"{self.database_file}" does not exist. Creating empty database...')
            self._init_database()

    def _init_database(self) -> None:
        self.database = {
            'tile_size_ratio': self.tile_size_ratio,
            'tile_max_width': self.tile_max_width,
            'files': OrderedDict()
        }

    def _save_database(self) -> None:
        short_name = '..' + self.database_file[-50:] if len(self.database_file) > 50 else self.database_file
        logging.info(f'Saving database under "{short_name}"')
        with open(self.database_file, 'wb') as f:
            pickle.dump(self.database, f)
        logging.info('Database saved successfully')

    @staticmethod
    def _get_mean_rgb(image: Image.Image) -> list:
        data = np.asarray(image)
        return (np.mean(data, axis=(0, 1)) / 255.0).round(4).tolist()

    @staticmethod
    def _process_image(params: tuple) -> tuple | None:
        path, tile_size_ratio, tile_max_width = params
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            w, h = img.size
            h_new = min(w / tile_size_ratio, h)
            w_new = h_new * tile_size_ratio
            w_crop = (w - w_new) / 2
            h_crop = (h - h_new) / 2
            img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))
            prepared_img = img.resize(
                (int(tile_max_width), int(tile_max_width / tile_size_ratio)),
                Image.Resampling.LANCZOS  # updated for Pillow >=10.0
            ).convert('RGB')
            rgb = TileDatabase._get_mean_rgb(prepared_img)
            img_id = os.path.basename(path)
            return img_id, prepared_img, rgb
        except Exception as e:
            logging.warning(f'Failed processing {path}: {e}')
            return None

    def _validate_inputs(self) -> bool:
        if not self.database_file:
            logging.error('Please provide a valid database file.')
            return False
        if not self.image_dir or not os.path.isdir(self.image_dir):
            logging.error('Please provide a valid image directory.')
            return False
        if not (16 <= self.tile_max_width <= 512):
            logging.error('Tile size must be between 16 and 512 pixels.')
            return False
        return True

    def create(self):
        if not self._validate_inputs():
            return
        
        self._prepare_database()
        path_list = self._get_file_list()
        path_list = self._reduce_file_list(path_list)
        
        if not path_list:
            logging.info('No new images to add.')
            return
        
        logging.info(f'Adding {len(path_list)} images to database.')
        
        params_list = [(path, self.tile_size_ratio, self.tile_max_width) for path in path_list]
        num_workers = min(multiprocessing.cpu_count(), 8)  # Limit workers for stability

        data = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(TileDatabase._process_image, params): params for params in params_list}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                self.progress = (i + 1) / len(path_list)
                logging.info(f'Done {self.progress * 100:.1f}% [{i + 1}/{len(path_list)}]')
                if result:
                    data.append(result)

        for img_id, img_obj, rgb in data:
            self.database['files'][img_id] = {'image': img_obj, 'rgb': rgb}

        self._save_database()

    @staticmethod
    def load(database_file: str):
        try:
            short_name = '..' + database_file[-50:] if len(database_file) > 50 else database_file
            logging.info(f'Loading database "{short_name}"')
            with open(database_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f'Failed to load database: {e}')
            return None

    @staticmethod
    def db_to_rgb_array(db) -> np.ndarray:
        rgb = np.zeros((len(db['files']), 3))
        for idx, val in enumerate(db['files'].values()):
            rgb[idx] = val['rgb']
        return rgb

class TileFitter:
    def __init__(self, overlay_image_path, database_file, output_file_path, tile_multiplier, overlay_alpha=0.03, dpi=300):
        self.overlay_image_path = overlay_image_path
        self.database_file = database_file
        self.output_file_path = output_file_path
        self.tile_multiplier = tile_multiplier
        self.overlay_alpha = overlay_alpha
        self.dpi = dpi
        self.db = None
        self.tile_size = None

    def _load_data(self):
        self.db = TileDatabase.load(self.database_file)
        h = int(self.db.get('tile_max_width', 0) / self.db.get('tile_size_ratio', 1))
        w = int(self.db.get('tile_max_width', 0))
        self.tile_size = (w, h)

    def _validate_inputs(self):
        if len(self.overlay_image_path) == 0:
            logging.error('Overlay image path is empty.')
            return False
        if not os.path.isfile(self.overlay_image_path):
            logging.error(f'Overlay image path does not exist "{self.overlay_image_path}"')
            return False
        if not isinstance(self.tile_multiplier, int):
            logging.error(f'Not valid integer provided for tile multiplier "{self.tile_multiplier}"')
            return False
        if self.tile_multiplier < 1 or self.tile_multiplier > 2000:
            logging.error(f'Provided tile multiplier "{self.tile_multiplier}" but allowed range is between 1 and 2000')
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
        method = 1
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
            if method == 1:
                if self.tile_multiplier > 60:
                    logging.warning('Processing could take a while or reduce tile multiplier!')
                if multiplier > 1:
                    logging.warning(f'Not enough images in database. {len_db} exist but {len_temp} needed')
                    logging.info(f'Images are {multiplier} times repeated')
                    db_rgb_resized = np.stack([db_rgb for _ in range(multiplier)], axis=0).reshape(
                        (db_rgb.shape[0] * multiplier, db_rgb.shape[1]))
                cost = self.__calc_error_matrix(template, db_rgb_resized)
                logging.info('Run linear sum assignment problem')
                row_ind, col_ind = linear_sum_assignment(cost)

                logging.info(f'Create image with best tiles {(self.tile_multiplier * self.tile_size[0], self.tile_multiplier * self.tile_size[1])}')
                out = Image.new('RGB', (self.tile_multiplier * self.tile_size[0], self.tile_multiplier * self.tile_size[1]))
                files = list(self.db['files'].items())
                for idx, img_idx in zip(row_ind, col_ind):
                    col = idx % self.tile_multiplier
                    row = idx // self.tile_multiplier
                    row_px = col * self.tile_size[0]
                    col_px = row * self.tile_size[1]
                    tile_img = files[img_idx % len(db_rgb)][1]['image']
                    out.paste(tile_img, (row_px, col_px))
            else:
                # method 2
                out = Image.new('RGB', (self.tile_multiplier * self.tile_size[0], self.tile_multiplier * self.tile_size[1]))
                for m in range(multiplier):
                    start = m*len_db
                    end = min(start + len_db, len_temp)
                    sub_template = template[start:end]
                    print(sub_template.shape)
                    cost = self.__calc_error_matrix(sub_template, db_rgb)
                    logging.info('Run linear sum assignment problem')
                    row_ind, col_ind = linear_sum_assignment(cost)

                    files = list(self.db['files'].items())
                    for idx, img_idx in zip(row_ind, col_ind):
                        idx = m*len_db + idx
                        col = idx % self.tile_multiplier
                        row = idx // self.tile_multiplier
                        row_px = col * self.tile_size[0]
                        col_px = row * self.tile_size[1]
                        tile_img = files[img_idx][1]['image']
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
        small = image.resize((self.tile_multiplier, self.tile_multiplier))
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
            w_target = int(self.tile_size[0] * self.tile_multiplier)
            h_target = int(self.tile_size[1] * self.tile_multiplier)
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
