#!/usr/bin/env python3
"""Extract PINHOLE-only model from a COLMAP reconstruction that contains mixed camera models."""
import struct, os, sys
from pathlib import Path

def read_colmap_text(sparse_txt):
    """Read COLMAP TXT format reconstruction."""
    cameras = {}
    with open(sparse_txt / 'cameras.txt') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            cameras[cam_id] = {
                'model': parts[1],
                'width': int(parts[2]), 'height': int(parts[3]),
                'params': [float(p) for p in parts[4:]]
            }

    images = {}
    with open(sparse_txt / 'images.txt') as f:
        lines = [l.strip() for l in f if not l.startswith('#')]
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            img_id = int(parts[0])
            images[img_id] = {
                'qw': float(parts[1]), 'qx': float(parts[2]),
                'qy': float(parts[3]), 'qz': float(parts[4]),
                'tx': float(parts[5]), 'ty': float(parts[6]), 'tz': float(parts[7]),
                'cam_id': int(parts[8]), 'name': parts[9],
                'points2d': []
            }
            if i+1 < len(lines) and lines[i+1].strip():
                vals = lines[i+1].strip().split()
                for j in range(0, len(vals), 3):
                    if j+2 < len(vals):
                        images[img_id]['points2d'].append(
                            (float(vals[j]), float(vals[j+1]), int(vals[j+2])))

    points3d = {}
    with open(sparse_txt / 'points3D.txt') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            pt_id = int(parts[0])
            points3d[pt_id] = {
                'x': float(parts[1]), 'y': float(parts[2]), 'z': float(parts[3]),
                'r': int(parts[4]), 'g': int(parts[5]), 'b': int(parts[6]),
                'error': float(parts[7]),
                'track': [int(p) for p in parts[8:]]
            }
    return cameras, images, points3d

def write_colmap_binary(output_dir, cameras, images, points3d):
    """Write COLMAP binary format."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_id_map = {
        'SIMPLE_PINHOLE': 1, 'PINHOLE': 2, 'SIMPLE_RADIAL': 3, 'RADIAL': 4,
        'OPENCV': 5, 'OPENCV_FISHEYE': 6, 'FULL_OPENCV': 7, 'FOV': 8
    }
    
    # cameras.bin
    with open(os.path.join(output_dir, 'cameras.bin'), 'wb') as f:
        f.write(struct.pack('<I', 634527751))
        f.write(struct.pack('<Q', len(cameras)))
        for cam_id in sorted(cameras):
            c = cameras[cam_id]
            f.write(struct.pack('<I', cam_id))
            f.write(struct.pack('<I', model_id_map.get(c['model'], 2)))
            f.write(struct.pack('<II', c['width'], c['height']))
            for p in c['params']:
                f.write(struct.pack('<d', p))

    # images.bin
    with open(os.path.join(output_dir, 'images.bin'), 'wb') as f:
        f.write(struct.pack('<I', 634527752))
        f.write(struct.pack('<Q', len(images)))
        for img_id in sorted(images):
            img = images[img_id]
            f.write(struct.pack('<I', img_id))
            f.write(struct.pack('<dddd', img['qw'], img['qx'], img['qy'], img['qz']))
            f.write(struct.pack('<ddd', img['tx'], img['ty'], img['tz']))
            f.write(struct.pack('<I', img['cam_id']))
            name_bytes = img['name'].encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            pts = img['points2d']
            f.write(struct.pack('<Q', len(pts)))
            for x, y, p3d in pts:
                f.write(struct.pack('<dd', x, y))
            for x, y, p3d in pts:
                f.write(struct.pack('<q', p3d))

    # points3D.bin
    with open(os.path.join(output_dir, 'points3D.bin'), 'wb') as f:
        f.write(struct.pack('<I', 634527753))
        f.write(struct.pack('<Q', len(points3d)))
        for pt_id in sorted(points3d):
            pt = points3d[pt_id]
            f.write(struct.pack('<Q', pt_id))
            f.write(struct.pack('<ddd', pt['x'], pt['y'], pt['z']))
            f.write(struct.pack('<BBB', pt['r'], pt['g'], pt['b']))
            f.write(struct.pack('<d', pt['error']))
            track = pt['track']
            n_track = len(track) // 2
            f.write(struct.pack('<Q', n_track))
            for val in track:
                f.write(struct.pack('<I', val))

def main():
    sparse_txt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/tmp/colmap_v3_txt_1')
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/home/wm1/jwang/pointcloud-nav/output/BD-GLKGQ_4cam_v2/sparse/1_pinhole_only'

    cameras, images, points3d = read_colmap_text(sparse_txt)
    
    # Filter: only PINHOLE cameras
    pinhole_cam_ids = {cid for cid, c in cameras.items() if c['model'] == 'PINHOLE'}
    pinhole_img_ids = {iid for iid, img in images.items() if img['cam_id'] in pinhole_cam_ids}
    
    # Filter cameras
    filtered_cameras = {cid: cameras[cid] for cid in pinhole_cam_ids}
    
    # Filter images
    filtered_images = {iid: images[iid] for iid in pinhole_img_ids}
    
    # Filter points3D: must be observed by PINHOLE images
    # Valid point IDs
    valid_point_ids = set()
    for pt_id, pt in points3d.items():
        track_img_ids = set(pt['track'][::2])
        if track_img_ids & pinhole_img_ids:
            valid_point_ids.add(pt_id)
    
    filtered_points = {}
    for pt_id in valid_point_ids:
        pt = points3d[pt_id]
        # Filter track to only PINHOLE images
        filtered_track = []
        for j in range(0, len(pt['track']), 2):
            if pt['track'][j] in pinhole_img_ids:
                filtered_track.extend([pt['track'][j], pt['track'][j+1]])
        
        # Filter points2d in images to only include valid points
        filtered_points[pt_id] = dict(pt)
        filtered_points[pt_id]['track'] = filtered_track
    
    # Update images: filter points2d
    for iid in filtered_images:
        img = filtered_images[iid]
        img['points2d'] = [(x, y, p3d) for x, y, p3d in img['points2d'] 
                          if p3d == -1 or p3d in valid_point_ids]
    
    print(f'PINHOLE cameras: {len(filtered_cameras)}, images: {len(filtered_images)}, points: {len(filtered_points)}')
    
    write_colmap_binary(output_dir, filtered_cameras, filtered_images, filtered_points)
    print(f'Written to: {output_dir}')

if __name__ == '__main__':
    main()
