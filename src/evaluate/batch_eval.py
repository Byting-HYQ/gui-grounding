"""
批量评测：P5-1/P5-2/P5-3 在 ScreenSpot 上的 Click Accuracy
"""

import sys
sys.path.insert(0, "src")

import json
import os
from PIL import Image
from infer.self_correction import GUIGroundingPredictor


def bbox_xywh_to_xyxy(bbox):
    """[x, y, w, h] -> [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def is_point_in_box(point, bbox_xyxy):
    """判断点是否在框内"""
    px, py = point
    x1, y1, x2, y2 = bbox_xyxy
    return x1 <= px <= x2 and y1 <= py <= y2


def coord_1000_to_pixel(coord, img_w, img_h):
    """[0,1000] 坐标转像素坐标"""
    x, y = coord
    return int(x / 1000 * img_w), int(y / 1000 * img_h)


def convert_to_serializable(obj):
    """将 numpy 类型转为 Python 原生类型"""
    if hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return [convert_to_serializable(i) for i in obj]
    return obj


def evaluate_sample(predictor, image_path, instruction, gt_bbox_xywh, strategy, num_samples=3):
    """评测单个样本"""
    img = Image.open(image_path)
    img_w, img_h = img.size
    
    gt_bbox = bbox_xywh_to_xyxy(gt_bbox_xywh)
    
    if strategy == "single":
        result = predictor.predict_single(img, instruction)
        coord = result["coordinate"]
        consistency = None
    elif strategy == "retry":
        result = predictor.predict_with_retry(img, instruction, num_samples=num_samples, max_retries=1)
        coord = result["final_coordinate"]
        consistency = result["consistency"]
    else:
        result = predictor.predict_with_aggregate(img, instruction, num_samples=num_samples)
        coord = result["final_coordinate"]
        consistency = result["consistency"]
    
    if coord is None:
        return {
            "correct": False,
            "predicted": None,
            "consistency_score": None,
            "is_reliable": None,
            "raw_output": result.get("raw_output", ""),
            "gt_bbox": gt_bbox_xywh,
            "image_size": [img_w, img_h],
        }
    
    px, py = coord_1000_to_pixel(coord, img_w, img_h)
    correct = is_point_in_box((px, py), gt_bbox)
    
    return {
        "correct": bool(correct),
        "predicted": list(coord),
        "pixel_coord": [int(px), int(py)],
        "consistency_score": float(consistency["consistency_score"]) if consistency else None,
        "is_reliable": bool(consistency["is_reliable"]) if consistency else None,
        "raw_output": result.get("raw_output", ""),  # 保存模型原始输出
        "gt_bbox": gt_bbox_xywh,  # 保存真实边界框
        "image_size": [img_w, img_h],  # 保存图片尺寸
    }


def evaluate_dataset(predictor, annotation_file, image_dir, strategy, max_samples=None):
    """评测整个数据集"""
    with open(annotation_file) as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    results = []
    correct_count = 0
    
    print(f"\nEvaluating {len(data)} samples with strategy: {strategy}")
    
    for i, item in enumerate(data):
        img_path = os.path.join(image_dir, item["img_filename"])
        
        if not os.path.exists(img_path):
            print(f"  Warning: image not found: {img_path}")
            continue
        
        result = evaluate_sample(
            predictor, img_path, item["instruction"], 
            item["bbox"], strategy
        )
        result["img_filename"] = item["img_filename"]
        result["instruction"] = item["instruction"]
        result["data_type"] = item.get("data_type", "unknown")
        result["data_source"] = item.get("data_source", "unknown")
        
        results.append(result)
        if result["correct"]:
            correct_count += 1
        
        if (i + 1) % 10 == 0:
            acc = correct_count / (i + 1)
            print(f"  Progress: {i+1}/{len(data)}, Accuracy: {acc:.3f}")
    
    final_acc = correct_count / len(results) if results else 0
    print(f"Final Accuracy: {final_acc:.3f} ({correct_count}/{len(results)})")
    
    return {
        "strategy": strategy,
        "total": len(results),
        "correct": correct_count,
        "accuracy": float(final_acc),
        "results": results,
    }


if __name__ == "__main__":
    model_path = "./hf_cache/qwen/Qwen/Qwen3-VL-4B-Instruct"
    data_dir = "./data"
    
    print("Loading model...")
    predictor = GUIGroundingPredictor(model_path)
    
    # 先测试少量样本（10条），验证流程
    max_test = None
    
    for strategy in ["single", "retry", "aggregate"]:
        result = evaluate_dataset(
            predictor,
            os.path.join(data_dir, "screenspot_web.json"),
            data_dir,
            strategy,
            max_samples=max_test
        )
        
        # 保存结果
        output_file = f"experiments/person5/eval_{strategy}_web.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        serializable_result = convert_to_serializable(result)
        with open(output_file, "w") as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {output_file}")