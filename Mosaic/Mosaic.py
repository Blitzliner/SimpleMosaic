import os
import numpy as np
import glob
import logging
import multiprocessing
import math
import pickle
from PIL import Image, ImageOps
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class TileDatabase:
    def __init__(self, image_dir, database_file, tile_size_ratio, tile_max_width):
        self.image_dir = image_dir
        self.database_file = database_file
        self.tile_size_ratio = tile_size_ratio  # 1 mean square options are: 1/1, 3/2, 4/3, 16/9
        self.tile_max_width = tile_max_width
        self.progress = 0
        self.database = None

    def __get_file_list(self, file_extensions='*.jpg'):
        search_pattern = os.path.join(self.image_dir, '**', file_extensions)
        logging.info(f'Search for images in "{search_pattern}"')
        all_paths = [path for path in glob.iglob(search_pattern, recursive=True)]
        logging.info(f'Found {len(all_paths)} images')
        return all_paths

    def __reduce_file_list(self, path_list):
        if self.database['files'] and self.database.get('tile_size_ratio') == self.tile_size_ratio \
                and self.database.get('tile_max_width') == self.tile_max_width:
            logging.info('Database exist already with same attributes')
            path_list = [path for path in path_list if os.path.basename(path) not in self.database['files']]
        return path_list

    def __prepare_database(self):
        if os.path.isfile(self.database_file):
            self.database = TileDatabase.load(self.database_file)
            if self.database is None:
                self.__create_empty_database()
        else:
            logging.info(f'"{self.database_file}" does not exist. Create empty database with tile ratio: {self.tile_size_ratio} and tile width {self.tile_max_width}.')
            self.__create_empty_database()

    def __create_empty_database(self):
        self.database = {'tile_size_ratio': self.tile_size_ratio, 'tile_max_width': self.tile_max_width,
                         'files': OrderedDict()}

    def __save_database(self):
        short_name = '..' + self.database_file[-50:] if len(self.database_file) > 50 else self.database_file
        logging.info(f'Save database under "{short_name}"')
        self.database['tile_size_ratio'] = self.tile_size_ratio
        self.database['tile_max_width'] = self.tile_max_width
        with open(self.database_file, 'wb') as file:
            pickle.dump(self.database, file)

    def __get_mean_rgb(self, image):
        data = np.asarray(image)
        return (np.mean(data, axis=(0, 1)) / 255.0).round(4).tolist()

    def __process_image(self, path):
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            w, h = img.size
            h_new = min(w / self.tile_size_ratio, h)
            w_new = h_new * self.tile_size_ratio
            w_crop = (w - w_new) / 2
            h_crop = (h - h_new) / 2
            img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))
            prepared_img = img.resize((int(self.tile_max_width), int(self.tile_max_width / self.tile_size_ratio)),
                                      Image.ANTIALIAS)
            prepared_img = prepared_img.convert('RGB')
            rgb = self.__get_mean_rgb(prepared_img)
            id = os.path.basename(path)
            return id, prepared_img, rgb
        except Exception as e:
            logging.warning(f'{path} could not be processed: {e}')
            return None

    def __call__(self, path):
        return self.__process_image(path)

    def __validate_inputs(self):
        if len(self.database_file) == 0:
            logging.error('Please provide valid database file')
            return False

        if len(self.image_dir) == 0:
            logging.error('Please select image directory')
            return False
        elif not os.path.isdir(self.image_dir):
            logging.error('Please select valid image directory')
            return False

        if self.tile_max_width < 16 or self.tile_max_width > 512:
            logging.error('Tile size shall be between 16 and 512 pixel')
            return False
        return True

    def create(self):
        if not self.__validate_inputs():
            return
        self.__prepare_database()
        path_list = self.__get_file_list()
        path_list = self.__reduce_file_list(path_list)
        logging.info(f'Add {len(path_list)} images to database')
        num_worker = multiprocessing.cpu_count()
        data = []
        with multiprocessing.Pool(num_worker) as p:
            for i, res in enumerate(p.imap_unordered(self, path_list)):
                self.progress = i / len(path_list)
                logging.info(f'Done {self.progress*100:3.1f} %')
                data.append(res)
            self.progress = 1.0
            logging.info(f'Done {self.progress*100:3.1f} %')
        for d in data:
            self.database['files'][d[0]] = {'image': d[1], 'rgb': d[2]}
        self.__save_database()

    @staticmethod
    def load(database_file):
        try:
            short_name = '..' + database_file[-50:] if len(database_file) > 50 else database_file
            logging.info(f'Load database "{short_name}"')
            with open(database_file, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            logging.error(f'Could not load database. Create empty one. {e}')
        return None

    @staticmethod
    def db_to_rgb_array(db):
        rgb = np.zeros((len(db['files']), 3))
        for idx, (key, val) in enumerate(list(db['files'].items())):
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

    def __load_data(self):
        self.db = TileDatabase.load(self.database_file)
        h = int(self.db.get('tile_max_width', 0) / self.db.get('tile_size_ratio', 1))
        w = int(self.db.get('tile_max_width', 0))
        self.tile_size = (w, h)

    def __validate_inputs(self):
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
        logging.info(f"Run TileFitter")
        if not self.__validate_inputs():
            return
        self.__load_data()
        overlay_img = self.__prepare_overlay_size()
        if overlay_img is not None and self.db is not None:
            db_rgb = TileDatabase.db_to_rgb_array(self.db)
            template = self.__get_tile_template(overlay_img)
            cost = self.__calc_error_matrix(template, db_rgb)
            logging.info(f"Run linear sum assignment problem")
            row_ind, col_ind = linear_sum_assignment(cost)

            logging.info(f"Create image with best tiles {(self.tile_multiplier * self.tile_size[0], self.tile_multiplier * self.tile_size[1])}")
            out = Image.new('RGB', (self.tile_multiplier * self.tile_size[0], self.tile_multiplier * self.tile_size[1]))
            files = list(self.db['files'].items())
            for idx, img_idx in zip(row_ind, col_ind):
                col = idx % self.tile_multiplier
                row = idx // self.tile_multiplier
                row_px = col * self.tile_size[0]
                col_px = row * self.tile_size[1]
                tile_img = files[img_idx % len(db_rgb)][1]['image']
                out.paste(tile_img, (row_px, col_px))

            logging.info(f'Overlay image with alpha of {self.overlay_alpha*100}%')
            out = Image.blend(out, overlay_img, self.overlay_alpha)
            logging.info(f'Save image under "{self.output_file_path}"')
            out.save(self.output_file_path, dpi=(self.dpi, self.dpi))
            logging.info('Finished')
        else:
            logging.error('Overlay Image can not be processed or database is corrupt')

    def __calc_error_matrix(self, template, db_rgb):
        len_temp = template.shape[0]
        len_db = db_rgb.shape[0]

        if len_temp > len_db:
            logging.warning(f'Not enough images in database. {len_db} exist but {len_temp} needed. Lets repeat some images.')
            # we have less images in db than tiles repeat database images to reach tile count
            multiplier = math.ceil(len_temp / len_db)
            db_rgb = np.stack([db_rgb for _ in range(multiplier)], axis=0).reshape(
                (db_rgb.shape[0] * multiplier, db_rgb.shape[1]))
        shape_cost_matrix = (template.shape[0], db_rgb.shape[0])
        cost_matrix = np.zeros(shape_cost_matrix)
        logging.info(f'Create cost matrix with shape {shape_cost_matrix}')
        for idx in range(template.shape[0]):
            single_cost = np.sum((template[idx] - db_rgb) ** 2, axis=1)
            cost_matrix[idx] = single_cost
        return cost_matrix

    def __get_tile_template(self, image):
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
