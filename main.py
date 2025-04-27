import argparse
from Mosaic import Mosaic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('overlay_image_path', help='Overlay image for which the mosaic shall be created')
    parser.add_argument('-d', '--database_file', default='database.p', help='Database file')
    parser.add_argument('-i', '--image_dir', help='Image directory for the mosaic')
    parser.add_argument(
        '-m',
        '--tile_multiplier',
        default=20,
        help='How many tiles should be in one row/column',
    )
    parser.add_argument('-c', '--create_database', action='store_true', default=False)
    parser.add_argument('--output_file_path', default='mosaic.jpg', help='Output filepath of mosaic')
    parser.add_argument('--gui', action='store_true', default=False, help='Show gui')
    parser.add_argument('--tile_ratio', default=4 / 3, help='Tile size ratio')
    parser.add_argument('--tile_width', default=250, help='Tile maximum width')
    parser.add_argument('--dpi', default=300, help='Dots per inch only relevant if picture is printed')
    parser.add_argument('--overlay_alpha', default=0.03, help='Overlay image')

    args = parser.parse_args()
    db = Mosaic.TileDatabase(
        image_dir=args.image_dir,
        database_file=args.database_file,
        tile_size_ratio=args.tile_ratio,
        tile_max_width=args.tile_width,
    )
    db.create()

    tf = Mosaic.TileFitter(
        overlay_image_path=args.overlay_image_path,
        database_file=args.database_file,
        output_file_path=args.output_file_path,
        tile_multiplier=args.tile_multiplier,
        overlay_alpha=args.overlay_alpha,
        dpi=args.dpi,
    )
    tf.run()
