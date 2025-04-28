from dataclasses import dataclass
import yaml
import os
from enum import Enum

class StrEnum(str, Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
    
class ImageAspectRatio(StrEnum):
    """Enum for image aspect ratios."""
    SQUARE = '1:1'
    FIVE_FOUR = '5:4'
    FOUR_THREE = '4:3'
    THREE_TWO = '3:2'
    SIXTEEN_TEN = '16:10'
    FIVE_THREE = '5:3'
    SIXTEEN_NINE = '16:9'

    def to_float(self) -> float:
        """Convert aspect ratio string to float."""
        num, denom = map(int, self.value.split(':'))
        return num / denom
    
    @classmethod
    def from_string(cls, value: str) -> "ImageAspectRatio":
        """Get the enum entry for a given string."""
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")

@dataclass
class Config:
    database_file: str
    dpi: int
    fitter_out_file: str
    grayscale_active: bool
    height: int
    image_dir: str
    overlay_alpha: float
    overlay_image_path: str
    overlay_out_file: str
    tile_max_width: int
    grid_size: int
    tile_aspect_ratio: str
    width: int

    @staticmethod
    def load(path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return Config(**data)

    def save(self, path: str) -> None:
        """Save the configuration back to a YAML file."""
        with open(path, 'w') as f:
            yaml.safe_dump(self.__dict__, f, sort_keys=False)

    def set_dpi(self, dpi: int):
        if dpi <= 0:
            raise ValueError("DPI must be a positive integer.")
        self.dpi = dpi

    def set_width(self, width: int):
        if width <= 0:
            raise ValueError("Width must be positive.")
        self.width = width

    def set_height(self, height: int):
        if height <= 0:
            raise ValueError("Height must be positive.")
        self.height = height

    def set_image_dir(self, image_dir: str):
        if not os.path.isdir(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")
        self.image_dir = image_dir

    def set_overlay_alpha(self, alpha: float):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Overlay alpha must be between 0.0 and 1.0.")
        self.overlay_alpha = alpha

    def set_grayscale_active(self, active: bool):
        if not isinstance(active, bool):
            raise ValueError("grayscale_active must be a boolean value.")
        self.grayscale_active = active
    