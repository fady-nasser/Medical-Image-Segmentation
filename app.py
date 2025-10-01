from flask import Flask, render_template, jsonify, send_from_directory, send_file
import nibabel as nib
import numpy as np
import os
from pathlib import Path
import base64
from scipy import ndimage
from scipy.ndimage import zoom
import json
import io
import traceback

app = Flask(__name__)

# Configure paths - CHANGE THESE TO YOUR ACTUAL PATHS
BASE_PATH = "data"
RAW_FILE = f"{BASE_PATH}/s0010/ct.nii.gz"
GT_DIR = f"{BASE_PATH}/s0010/segmentations"

# AI Model output directories
AI_MODELS = {
    "TotalSegmentator": f"{BASE_PATH}/segmentations_total_out",
    "wholebody_ct_segmentation": f"{BASE_PATH}/Monai",
    "MedIm": f"{BASE_PATH}/MedIm"
}

# Organ groups
ORGAN_GROUPS = {
    "Lungs": [
        "lung_upper_lobe_left.nii.gz",
        "lung_lower_lobe_left.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_upper_lobe_right.nii.gz",
        "lung_lower_lobe_right.nii.gz",
    ],
    "Lower Abdominal": [
        "iliac_artery_left.nii.gz",
        "iliac_artery_right.nii.gz",
        "iliac_vena_left.nii.gz",
        "iliac_vena_right.nii.gz",
        "iliopsoas_left.nii.gz",
        "iliopsoas_right.nii.gz"
    ],
    "Gluteus": [
        "gluteus_maximus_left.nii.gz",
        "gluteus_maximus_right.nii.gz",
        "gluteus_medius_left.nii.gz",
        "gluteus_medius_right.nii.gz",
        "gluteus_minimus_left.nii.gz",
        "gluteus_minimus_right.nii.gz"
    ]
}

# Cache for loaded data
data_cache = {}

def load_nifti(filepath, target_shape=None):
    """Load NIfTI file and return data, with optional resampling at load time"""
    cache_key = f"{filepath}_shape_{target_shape}" if target_shape else filepath
    
    if cache_key in data_cache:
        return data_cache[cache_key]
    
    if not os.path.exists(filepath):
        return None
    
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        
        if target_shape and data.shape != target_shape:
            print(f"⚡ Resampling {os.path.basename(filepath)} once: {data.shape} → {target_shape}")
            zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
            data = zoom(data, zoom_factors, order=0)
            print(f"✓ Cached resampled volume for {os.path.basename(filepath)}")
        
        data_cache[cache_key] = data
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compute_metrics(gt_data, pred_data):
    """Compute Dice, IoU, HD95"""
    try:
        gt_bin = (gt_data > 0).astype(np.uint8)
        pred_bin = (pred_data > 0).astype(np.uint8)
        
        intersection = np.sum(gt_bin * pred_bin)
        
        dice = (2.0 * intersection) / (np.sum(gt_bin) + np.sum(pred_bin) + 1e-8)
        union = np.sum(gt_bin) + np.sum(pred_bin) - intersection
        iou = intersection / (union + 1e-8)
        
        try:
            from scipy.ndimage import distance_transform_edt
            if np.sum(gt_bin) > 0 and np.sum(pred_bin) > 0:
                dt_gt = distance_transform_edt(~gt_bin.astype(bool))
                dt_pred = distance_transform_edt(~pred_bin.astype(bool))
                hd95 = max(np.percentile(dt_gt[pred_bin > 0], 95), 
                          np.percentile(dt_pred[gt_bin > 0], 95))
            else:
                hd95 = 0.0
        except:
            hd95 = 0.0
        
        return {"dice": float(dice), "iou": float(iou), "hd95": float(hd95)}
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {"dice": 0.0, "iou": 0.0, "hd95": 0.0}

def extract_mesh_data(volume_data, spacing=(1, 1, 1)):
    """Extract mesh vertices and faces from 3D volume using marching cubes with smoothing"""
    from skimage import measure
    from scipy.ndimage import gaussian_filter
    
    if np.sum(volume_data) == 0:
        return None
    
    try:
        smoothed = gaussian_filter(volume_data.astype(float), sigma=1.0)
        verts, faces, normals, values = measure.marching_cubes(
            smoothed, 
            level=0.5, 
            spacing=spacing,
            step_size=1
        )
        
        max_verts = 20000
        if len(verts) > max_verts:
            step = len(verts) // max_verts
            keep_indices = np.arange(0, len(verts), step)
            verts = verts[keep_indices]
            
            face_mask = np.all(np.isin(faces, keep_indices), axis=1)
            faces = faces[face_mask]
            
            index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
            faces = np.array([[index_map.get(idx, 0) for idx in face] for face in faces])
        
        return {
            "vertices": verts.tolist(),
            "faces": faces.tolist(),
            "normals": normals.tolist() if len(normals) > 0 else []
        }
    except Exception as e:
        print(f"Mesh extraction error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html', organ_groups=ORGAN_GROUPS)

@app.route('/api/models')
def get_models():
    """Return available AI models"""
    return jsonify({"models": list(AI_MODELS.keys())})

@app.route('/api/organs')
def get_organs():
    """Return available organ groups"""
    return jsonify({"organs": list(ORGAN_GROUPS.keys())})

@app.route('/api/organ/<organ_name>/parts/<model_name>')
def get_organ_parts(organ_name, model_name):
    """Return parts for a specific organ and model"""
    try:
        if organ_name not in ORGAN_GROUPS:
            return jsonify({"error": "Organ not found"}), 404
        
        if model_name not in AI_MODELS:
            return jsonify({"error": "Model not found"}), 404
        
        pred_dir = AI_MODELS[model_name]
        
        parts = []
        for filename in ORGAN_GROUPS[organ_name]:
            part_name = filename.replace('.nii.gz', '').replace('_', ' ').title()
            pred_path = os.path.join(pred_dir, filename)
            gt_path = os.path.join(GT_DIR, filename)
            
            parts.append({
                "filename": filename,
                "name": part_name,
                "available_pred": os.path.exists(pred_path),
                "available_gt": os.path.exists(gt_path)
            })
        
        return jsonify({"parts": parts})
    except Exception as e:
        print(f"Error in get_organ_parts: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/organ/<organ_name>/mesh/<part_filename>/<model_name>')
def get_part_mesh(organ_name, part_filename, model_name):
    """Get 3D mesh data for a specific part"""
    try:
        if organ_name not in ORGAN_GROUPS:
            return jsonify({"error": "Organ not found"}), 404
        
        if model_name not in AI_MODELS:
            return jsonify({"error": "Model not found"}), 404
        
        pred_path = os.path.join(AI_MODELS[model_name], part_filename)
        
        if not os.path.exists(pred_path):
            return jsonify({"error": "File not found"}), 404
        
        data = load_nifti(pred_path)
        if data is None:
            return jsonify({"error": "Could not load NIfTI file"}), 500
            
        mesh_data = extract_mesh_data(data)
        
        if mesh_data is None:
            return jsonify({"error": "Could not generate mesh"}), 500
        
        return jsonify(mesh_data)
    except Exception as e:
        print(f"Error in get_part_mesh: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/download/obj/<organ_name>/<model_name>')
def download_complete_obj(organ_name, model_name):
    """Download complete organ model as single OBJ file"""
    try:
        if organ_name not in ORGAN_GROUPS or model_name not in AI_MODELS:
            return jsonify({"error": "Invalid parameters"}), 404
        
        pred_dir = AI_MODELS[model_name]
        
        obj_content = "# Medical Segmentation Complete Model\n"
        obj_content += f"# Organ: {organ_name}\n"
        obj_content += f"# Model: {model_name}\n\n"
        
        vertex_offset = 0
        
        for filename in ORGAN_GROUPS[organ_name]:
            pred_path = os.path.join(pred_dir, filename)
            
            if not os.path.exists(pred_path):
                continue
            
            data = load_nifti(pred_path)
            if data is None:
                continue
                
            mesh_data = extract_mesh_data(data)
            
            if mesh_data is None:
                continue
            
            part_name = filename.replace('.nii.gz', '').replace('_', ' ')
            obj_content += f"\n# Part: {part_name}\n"
            obj_content += f"g {part_name.replace(' ', '_')}\n"
            
            for vertex in mesh_data['vertices']:
                obj_content += f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"
            
            for face in mesh_data['faces']:
                obj_content += f"f {face[0]+1+vertex_offset} {face[1]+1+vertex_offset} {face[2]+1+vertex_offset}\n"
            
            vertex_offset += len(mesh_data['vertices'])
        
        safe_name = organ_name.replace(' ', '_')
        filename = f"{safe_name}_{model_name}_complete.obj"
        
        return send_file(
            io.BytesIO(obj_content.encode()),
            mimetype='text/plain',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f"Error in download_complete_obj: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/slice/<int:slice_idx>')
def get_slice(slice_idx):
    """Get CT slice with overlays"""
    try:
        raw_data = load_nifti(RAW_FILE)
        
        if raw_data is None:
            return jsonify({"error": "Raw CT not found"}), 404
        
        if slice_idx >= raw_data.shape[2]:
            return jsonify({"error": "Slice index out of range"}), 400
        
        raw_slice = raw_data[:, :, slice_idx]
        
        raw_min, raw_max = np.percentile(raw_slice, [1, 99])
        raw_normalized = np.clip((raw_slice - raw_min) / (raw_max - raw_min) * 255, 0, 255).astype(np.uint8)
        
        from PIL import Image
        img = Image.fromarray(raw_normalized.T, mode='L')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        raw_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            "raw": f"data:image/png;base64,{raw_b64}",
            "max_slices": int(raw_data.shape[2])
        })
    except Exception as e:
        print(f"Error in get_slice: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/organ/<organ_name>/metrics/<model_name>')
def get_organ_metrics(organ_name, model_name):
    """Get metrics for all parts of an organ"""
    try:
        if organ_name not in ORGAN_GROUPS:
            return jsonify({"error": "Organ not found"}), 404
        
        if model_name not in AI_MODELS:
            return jsonify({"error": "Model not found"}), 404
        
        pred_dir = AI_MODELS[model_name]
        
        raw_data = load_nifti(RAW_FILE)
        target_shape = raw_data.shape if raw_data is not None else None
        
        results = []
        for filename in ORGAN_GROUPS[organ_name]:
            gt_path = os.path.join(GT_DIR, filename)
            pred_path = os.path.join(pred_dir, filename)
            
            if os.path.exists(gt_path) and os.path.exists(pred_path):
                gt_data = load_nifti(gt_path, target_shape=target_shape)
                pred_data = load_nifti(pred_path, target_shape=target_shape)
                
                if gt_data is not None and pred_data is not None:
                    metrics = compute_metrics(gt_data, pred_data)
                    metrics["part"] = filename.replace('.nii.gz', '')
                    results.append(metrics)
        
        if results:
            avg_metrics = {
                "dice": np.mean([r["dice"] for r in results]),
                "iou": np.mean([r["iou"] for r in results]),
                "hd95": np.mean([r["hd95"] for r in results])
            }
        else:
            avg_metrics = {"dice": 0, "iou": 0, "hd95": 0}
        
        return jsonify({
            "parts": results,
            "average": avg_metrics
        })
    except Exception as e:
        print(f"Error in get_organ_metrics: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/slice/<organ_name>/<int:slice_idx>/<model_name>')
def get_organ_slice(organ_name, slice_idx, model_name):
    """Get slice with organ overlays"""
    try:
        if organ_name not in ORGAN_GROUPS:
            return jsonify({"error": "Organ not found"}), 404
        
        if model_name not in AI_MODELS:
            return jsonify({"error": "Model not found"}), 404
        
        raw_data = load_nifti(RAW_FILE)
        if raw_data is None:
            return jsonify({"error": "Raw CT not found"}), 404
        
        if slice_idx >= raw_data.shape[2]:
            return jsonify({"error": "Slice index out of range"}), 400
        
        pred_dir = AI_MODELS[model_name]
        target_shape = raw_data.shape
        
        gt_mask = np.zeros_like(raw_data[:, :, slice_idx], dtype=np.int16)
        pred_mask = np.zeros_like(raw_data[:, :, slice_idx], dtype=np.int16)
        
        for idx, filename in enumerate(ORGAN_GROUPS[organ_name], start=1):
            gt_path = os.path.join(GT_DIR, filename)
            pred_path = os.path.join(pred_dir, filename)
            
            if os.path.exists(gt_path):
                gt_data = load_nifti(gt_path, target_shape=target_shape)
                if gt_data is not None and slice_idx < gt_data.shape[2]:
                    gt_slice = gt_data[:, :, slice_idx]
                    gt_mask[gt_slice > 0] = idx
            
            if os.path.exists(pred_path):
                pred_data = load_nifti(pred_path, target_shape=target_shape)
                if pred_data is not None and slice_idx < pred_data.shape[2]:
                    pred_slice = pred_data[:, :, slice_idx]
                    pred_mask[pred_slice > 0] = idx
        
        from PIL import Image
        
        def mask_to_b64(mask):
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
            ]
            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            
            for i in range(1, 9):
                if np.any(mask == i):
                    color_idx = (i - 1) % len(colors)
                    rgba[mask == i] = (*colors[color_idx], 180)
            
            img = Image.fromarray(np.transpose(rgba, (1, 0, 2)), mode='RGBA')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            "gt_overlay": f"data:image/png;base64,{mask_to_b64(gt_mask)}",
            "pred_overlay": f"data:image/png;base64,{mask_to_b64(pred_mask)}"
        })
    except Exception as e:
        print(f"Error in get_organ_slice: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Medical Segmentation Viewer...")
    print(f"Please open your browser to: http://localhost:5000")
    print(f"Available AI Models: {', '.join(AI_MODELS.keys())}")
    app.run(debug=True, port=5000)