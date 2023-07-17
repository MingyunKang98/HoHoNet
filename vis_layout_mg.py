import json
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.signal import correlate2d
from scipy.ndimage import shift

from lib.misc.post_proc import np_coor2xy, np_coorx2u, np_coory2v
from eval_layout import layout_2_depth


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', required=True,
                        help='Image texture in equirectangular format')
    parser.add_argument('--layout', required=True,
                        help='Txt or json file containing layout corners (cor_id)')
    parser.add_argument('--out')
    parser.add_argument('--no_vis', action='store_true')
    parser.add_argument('--show_ceiling', action='store_true',
                        help='Rendering ceiling (skip by default)')
    parser.add_argument('--ignore_floor', action='store_true',
                        help='Skip rendering floor')
    parser.add_argument('--ignore_wall', action='store_true',
                        help='Skip rendering wall')
    parser.add_argument('--ignore_wireframe', action='store_true',
                        help='Skip rendering wireframe')
    args = parser.parse_args(["--img", "./assets/pano_asmasuxybohhcj.png",
                              "--layout", "./assets/pano_asmasuxybohhcj.layout.txt", "--out", "./asset", "--ignore_floor"])


    if not args.out and args.no_vis:
        print('You may want to export (via --out) or visualize (without --vis)')
        import sys; sys.exit()

    # Reading source (texture img, cor_id txt)
    equirect_texture = np.array(Image.open(args.img))
    H, W = equirect_texture.shape[:2]
    if args.layout.endswith('json'):
        with open(args.layout) as f:
            inferenced_result = json.load(f)
        cor_id = np.array(inferenced_result['uv'], np.float32)
        cor_id[:, 0] *= W
        cor_id[:, 1] *= H
    else:
        cor_id = np.loadtxt(args.layout).astype(np.float32)

    # Convert corners to layout
    depth, floor_mask, ceil_mask, wall_mask = layout_2_depth(cor_id, H, W, return_mask=True)


import matplotlib.pyplot as plt

cor_id = cor_id.astype("int64")
cor_id_x = np.sort(cor_id[:,0])
for k in range(1, len(cor_id_x), 2):
    equirect_texture = np.array(Image.open(args.img))
    wall_mask[:, :cor_id_x[k]] = False
    wall_mask[:, cor_id_x[k+1]:] = False
    equirect_texture[~wall_mask] = [0,0,0]
    plt.imshow(equirect_texture)
    plt.show()


