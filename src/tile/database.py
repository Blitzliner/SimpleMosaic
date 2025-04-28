import os
import numpy as np
import glob
import logging
import multiprocessing
import pickle
from PIL import Image, ImageOps
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


@dataclass
class ImageData:
    name: str
    image: Image
    mean_rgb: tuple[float, float, float]


@dataclass
class DatabaseConfig:
    tiles: list[ImageData]
    tile_ratio: float = 4.0 / 3.0
    tile_width: int = 250
    path: str = ''

    def __post_init__(self):
        if self.tile_ratio <= 0:
            raise ValueError('tile_ratio must be positive.')
        if not self.path:
            raise ValueError('Please provide a valid database file path.')
        if not (16 <= self.tile_width <= 512):
            raise ValueError('Tile size must be between 16 and 512 pixels.')


class TileDatabase:
    def __init__(self, tile_dir: str, database_path: str, tile_ratio: float, tile_width: int, file_extensions: str = '*.jpg'):
        if not os.path.isdir(tile_dir):
            raise ValueError(f'Invalid tile directory: <{tile_dir}>')
        self.tile_dir = tile_dir
        self.file_extensions = file_extensions

        if os.path.isfile(database_path):
            logging.info(f'Load existing database from path <{database_path}>')
            db = TileDatabase.load(database_path)
            
            if db.tile_ratio != tile_ratio or db.tile_width != tile_width:
                logging.warning('Database attributes do not match. Overwriting existing database.')
                self.database = DatabaseConfig(tiles=[], tile_ratio=tile_ratio, tile_width=tile_width, path=database_path)
            else:
                logging.info('Database attributes match. Using existing database.')
                self.database = db
        else:
            logging.info(f'Creating new database at <{database_path}>')
            self.database = DatabaseConfig(tiles=[], tile_ratio=tile_ratio, tile_width=tile_width, path=database_path)

    def _get_file_list(self, tile_dir, file_extensions: str = '*.jpg') -> list[str]:
        search_pattern = os.path.join(tile_dir, '**', file_extensions)
        logging.info(f'Search for images in <{search_pattern}>')
        paths = list(glob.iglob(search_pattern, recursive=True))
        tile_names = {tile.name for tile in self.database.tiles}
        new_paths  = [path for path in paths if os.path.basename(path) not in tile_names]
        logging.info(f'Found {len(paths)} images and {len(new_paths )} new images.')
        return new_paths 

    @staticmethod
    def _get_mean_rgb(image: Image.Image) -> list:
        arr = np.asarray(image)
        mean_rgb = np.mean(arr, axis=(0, 1)) / 255.0
        return tuple(np.round(mean_rgb, 4))

    @staticmethod
    def _process_image(path: str, tile_ratio: float, tile_width: int) -> ImageData | None:
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            w, h = img.size
            h_new = min(w / tile_ratio, h)
            w_new = h_new * tile_ratio
            w_crop = (w - w_new) / 2
            h_crop = (h - h_new) / 2
            img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))
            prepared_img = img.resize(
                (int(tile_width), int(tile_width / tile_ratio)),
                Image.Resampling.LANCZOS,
            ).convert('RGB')
            return ImageData(name=os.path.basename(path),
                             image=prepared_img,
                             mean_rgb=TileDatabase._get_mean_rgb(prepared_img))
        except Exception as e:
            logging.warning(f'Failed processing {path}: {e}')
            return None

    def create(self):
        files = self._get_file_list(self.tile_dir, self.file_extensions)
        if not files:
            logging.info('No new images to add.')
            return
        total = len(files)
        
        logging.info(f'Process {total} images.')
        num_workers = multiprocessing.cpu_count() - 1

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._process_image, path=path, tile_ratio=self.database.tile_ratio, tile_width=self.database.tile_width): path
                for path in files
            }
            completed = 0
            dividor = max(1, total // 30)
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if completed % dividor == 0:
                    progress = completed / total * 100
                    logging.info(f"Progress: {progress:.1f}% ({completed}/{total})")
                
                if result:
                    self.database.tiles.append(result)
        TileDatabase.save(self.database)

    @staticmethod
    def save(database: DatabaseConfig) -> None:
        logging.info(f'Saving database under <{database.path}>')
        os.makedirs(os.path.dirname(database.path), exist_ok=True)
        with open(database.path, 'wb') as f:
            pickle.dump(database, f)
        logging.info('Database saved successfully')

    @staticmethod
    def load(database_path: str) -> DatabaseConfig:
        logging.info(f'Loading database "{database_path}"')
        with open(database_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def db_to_rgb_array(db: DatabaseConfig) -> np.ndarray:
        rgb = np.zeros((len(db.tiles), 3))
        for idx, val in enumerate(db.tiles):
            rgb[idx] = val.mean_rgb
        return rgb
