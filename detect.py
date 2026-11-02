#!/usr/bin/env python3
"""Run the pillar 3D object detector (ONNX) on a lidar scan in PCD format, print the boxes and optionally show them.

The model takes the current scan and previous scans in the current lidar frame (x forward, y left, z up, metres,
origin at the lidar), raw sensor intensity (0-255; normalised in the graph), a time-lag column (0 for the current scan,
the age in seconds for previous ones) and the lidar height above the ground. Previous scans come from <name>_sweep_<k>.pcd (k = 1 newest) next to the cloud, with their lags in
sweeps.json ({"lag": [0.1, 0.2, ...]}); they must already be ego-motion compensated into the current lidar frame.

usage: python detect.py example.pcd [--sweeps 1] [--style nuscenes] [--lidar_height 2.11] [--threshold 0.5]
                                    [--show] [--plot out.png]
"""
import argparse, json, os, time
import numpy as np
import onnxruntime as ort
from pypcd4 import PointCloud

GROUP_COLOURS = {'vehicle': (0.12, 0.47, 0.71), 'pedestrian': (0.84, 0.15, 0.16), 'cyclist': (0.17, 0.63, 0.17),
                 'obstacle': (1.0, 0.5, 0.05), 'animal': (0.58, 0.4, 0.74), 'unknown': (0.5, 0.5, 0.5)}


def read_pcd(path):
    """PCD (ascii, binary or binary_compressed) -> (N, 4) float32 x, y, z, intensity (0 if the file has none)."""
    cloud = PointCloud.from_path(path)
    points = cloud.numpy(('x', 'y', 'z')).astype(np.float32)
    intensity = cloud.numpy(('intensity',)).astype(np.float32) if 'intensity' in cloud.fields else np.zeros((len(points), 1), np.float32)
    points = np.hstack((points, intensity))
    return points[np.isfinite(points).all(1)]


def read_scans(path, sweeps):
    """The cloud (lag 0) and its `sweeps` newest previous scans -> (N, 5) x, y, z, intensity, lag; current scan first."""
    lags = [0.0]
    if sweeps:
        lags += json.load(open(os.path.join(os.path.dirname(os.path.abspath(path)), 'sweeps.json')))['lag'][:sweeps]
    files = [path] + [f'{os.path.splitext(path)[0]}_sweep_{k}.pcd' for k in range(1, len(lags))]
    scans = [read_pcd(f) for f in files]
    return np.concatenate([np.column_stack((p, np.full(len(p), lag, np.float32))) for p, lag in zip(scans, lags)])


def load_model(path, threads, cpu=False):
    """ONNX Runtime session (CUDA if available, else CPU) and the model's metadata."""
    providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')[cpu:] if p in ort.get_available_providers()]
    if 'CUDAExecutionProvider' in providers and hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()   # CUDA and cuDNN from the pip wheels of onnxruntime-gpu[cuda,cudnn], if installed
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads   # explicit: the default can oversubscribe shared machines
    session = ort.InferenceSession(path, options, providers=providers)
    meta = {k: json.loads(v) for k, v in session.get_modelmeta().custom_metadata_map.items()}
    return session, meta


def model_input(points, meta):
    """Fixed-size input [max_input_points, 5]; padding rows lie below the pillar grid, which the model ignores. A larger
    merged cloud is truncated: the current scan comes first, so the oldest sweep points are dropped."""
    grid = meta['grid']
    rows = np.zeros((grid['max_input_points'], 5), np.float32)
    rows[:, 2] = grid['z_min'] - 100.0
    n = min(len(points), len(rows))
    rows[:n] = points[:n]
    return rows


def detect(session, meta, points, style, lidar_height, threshold):
    """-> (M, 11) boxes [x, y, z, l, w, h, yaw, score, class, vx, vy] in the lidar frame, score > 0."""
    styles = meta['styles']
    if style not in styles + ['unknown']:
        raise SystemExit(f'--style must be one of {styles} or unknown')
    feed = dict(points=model_input(points, meta),
                style=np.array([styles.index(style) if style in styles else len(styles)], np.int64),
                lidar_height=np.array([lidar_height], np.float32), score_thresh=np.array([threshold], np.float32))
    boxes = session.run(None, feed)[0][0]
    return boxes[boxes[:, 7] > 0]


def box_lines(b, velocity):
    """Box edges, heading line (centre -> front) and velocity arrow (1 s of motion) as (K, 2, 3) segments."""
    c, s = np.cos(b[6]), np.sin(b[6])
    corners = 0.5 * np.array([[x, y, z] for z in (-1, 1) for x, y in ((1, 1), (1, -1), (-1, -1), (-1, 1))]) * b[3:6]
    corners = corners @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + b[:3]
    edges = [(i, (i + 1) % 4) for i in range(4)] + [(i + 4, (i + 1) % 4 + 4) for i in range(4)] + [(i, i + 4) for i in range(4)]
    segs = [corners[[i, j]] for i, j in edges] + [[b[:3], b[:3] + 0.5 * b[3] * np.array([c, s, 0])]]
    v = np.array([b[9], b[10], 0.0])
    if velocity and np.linalg.norm(v) > 0.5:
        tip, back = b[:3] + v, -0.4 * v / np.linalg.norm(v)
        side = np.array([-back[1], back[0], 0]) * 0.6
        segs += [[b[:3], tip], [tip, tip + back + side], [tip, tip + back - side]]
    return np.array(segs)


def scene(points, boxes, meta, velocity):
    """Open3D geometries: points shaded by intensity, boxes as one line set coloured by class group."""
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3].astype(np.float64)))
    shade = 0.55 - 0.45 * np.clip(points[:, 3:4] / 60.0, 0, 1)   # brighter return, darker point
    cloud.colors = o3d.utility.Vector3dVector(np.repeat(shade, 3, 1).astype(np.float64))
    segs, colours = [np.zeros((0, 2, 3))], [np.zeros((0, 3))]
    for b in boxes:
        s = box_lines(b, velocity)
        segs.append(s)
        colours.append(np.tile(GROUP_COLOURS.get(meta['class_groups'][int(b[8])], (0, 0, 0)), (len(s), 1)))
    segs = np.concatenate(segs)
    lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(segs.reshape(-1, 3)),
                                 o3d.utility.Vector2iVector(np.arange(2 * len(segs)).reshape(-1, 2)))
    lines.colors = o3d.utility.Vector3dVector(np.concatenate(colours))
    return cloud, lines


VIEW = dict(lookat=[10.0, 1.0, -1.5], eye=[-10.0, -9.0, 6.0], up=[0.0, 0.0, 1.0])   # behind and above the car


def render(cloud, lines, path, size=(1600, 1000)):
    """The Open3D scene rendered offscreen to a PNG."""
    import open3d as o3d
    rendering = o3d.visualization.rendering
    r = rendering.OffscreenRenderer(*size)
    r.scene.set_background([1, 1, 1, 1])
    r.scene.view.set_post_processing(False)   # exact colours, no tone mapping
    for name, geometry, shader, attr, value in (('points', cloud, 'defaultUnlit', 'point_size', 3.0),
                                                ('boxes', lines, 'unlitLine', 'line_width', 5.0)):
        material = rendering.MaterialRecord()
        material.shader = shader
        setattr(material, attr, value)
        r.scene.add_geometry(name, geometry, material)
    r.setup_camera(60.0, VIEW['lookat'], VIEW['eye'], VIEW['up'])
    o3d.io.write_image(path, r.render_to_image())


def show(cloud, lines, title):
    import open3d as o3d
    front = np.subtract(VIEW['eye'], VIEW['lookat'])
    o3d.visualization.draw_geometries([cloud, lines], window_name=title, lookat=VIEW['lookat'],
                                      front=front / np.linalg.norm(front), up=VIEW['up'], zoom=0.3)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('pcd')
    ap.add_argument('--sweeps', type=int, default=1, choices=range(5), help='previous scans to add (0: single scan)')
    ap.add_argument('--style', default='nuscenes', help='annotation style (dataset embedding), see README')
    ap.add_argument('--lidar_height', type=float, default=2.11, help='lidar height above the ground (m); 2.11 for the example')
    ap.add_argument('--threshold', type=float, default=0.5, help='score threshold')
    ap.add_argument('--show', action='store_true', help='show the scene in an Open3D window (needs open3d)')
    ap.add_argument('--plot', help='render the scene offscreen to a PNG (needs open3d)')
    ap.add_argument('--model', default=os.path.join(here, 'pillar_film_v5_fp16.onnx'))
    ap.add_argument('--threads', type=int, default=min(8, os.cpu_count() or 1), help='CPU threads')
    ap.add_argument('--cpu', action='store_true', help='use the CPU even if CUDA is available')
    a = ap.parse_args()

    points = read_scans(a.pcd, a.sweeps)
    if len(points) > 131072:
        print(f'warning: {len(points)} points, the model input holds 131072; the rest (oldest scans) are dropped')
    session, meta = load_model(a.model, a.threads, a.cpu)
    detect(session, meta, points, a.style, a.lidar_height, a.threshold)   # warm-up
    t = time.perf_counter()
    boxes = detect(session, meta, points, a.style, a.lidar_height, a.threshold)
    ms = 1000 * (time.perf_counter() - t)
    print(f'{len(points)} points ({a.sweeps} previous scans), {len(boxes)} detections, {ms:.0f} ms on {session.get_providers()[0]}')
    print(f'{"class":12s} {"x":>6s} {"y":>6s} {"z":>5s} {"l":>5s} {"w":>4s} {"h":>4s} {"yaw":>5s} {"score":>5s} '
          f'{"speed":>5s} {"vx":>5s} {"vy":>5s}')
    for b in boxes[np.argsort(np.hypot(boxes[:, 0], boxes[:, 1]))]:
        v = f'{np.hypot(b[9], b[10]):5.1f} {b[9]:5.1f} {b[10]:5.1f}' if a.sweeps else f'{"-":>5s} {"-":>5s} {"-":>5s}'
        print(f'{meta["class_names"][int(b[8])]:12s} {b[0]:6.1f} {b[1]:6.1f} {b[2]:5.1f} {b[3]:5.1f} {b[4]:4.1f} {b[5]:4.1f} '
              f'{b[6]:5.2f} {b[7]:5.2f} {v}')
    if not a.sweeps:
        print('speed, vx, vy (m/s, over the ground) need previous scans (--sweeps 1 or more)')
    if a.show or a.plot:
        cloud, lines = scene(points[points[:, 4] == 0], boxes, meta, a.sweeps > 0)
        if a.plot:
            render(cloud, lines, a.plot)
            print(f'image written to {a.plot}')
        if a.show:
            show(cloud, lines, os.path.basename(a.pcd))


if __name__ == '__main__':
    main()
