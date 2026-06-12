"""
5号任务：预测一致性、自纠错与单步 Demo
核心功能：多次采样 + 一致性分数 + 重试/聚合
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict, Optional
import json
import re


class GUIGroundingPredictor:
    """GUI 定位预测器，支持多次采样和一致性评估"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        self.device = device
        
        print("Loading model...")
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        # 使用 FP16 加载完整模型（24GB 显存足够）
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        self.model.eval()
        print("Model loaded!")
    
    def _build_prompt(self, instruction: str) -> str:
        """构建提示词，要求输出 JSON 坐标"""
        return (
            f"请根据指令定位需要点击的元素。"
            f"指令：{instruction}。"
            f"只输出 JSON 格式：{{\"action\":\"click\",\"x\":整数,\"y\":整数}}。"
            f"坐标范围 [0,1000]。不要输出其他内容。"
        )
    
    def _parse_coordinate(self, text: str) -> Optional[Tuple[int, int]]:
        """从模型输出解析坐标"""
        if "assistant" in text:
            text = text.split("assistant")[-1]
        text = text.strip()
        
        # 方法1: 直接找 JSON 对象
        match = re.search(r'\{[^{}]+\}', text)
        if match:
            json_str = match.group()
            try:
                data = json.loads(json_str)
                x = data.get("x")
                y = data.get("y")
                if x is not None and y is not None:
                    x, y = int(x), int(y)
                    if 0 <= x <= 1000 and 0 <= y <= 1000:
                        return (x, y)
            except:
                pass
        
        # 方法2: 直接找 x 和 y 的数字
        x_match = re.search(r'"x"\s*:\s*(\d+)', text)
        y_match = re.search(r'"y"\s*:\s*(\d+)', text)
        if x_match and y_match:
            x, y = int(x_match.group(1)), int(y_match.group(1))
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                return (x, y)
        
        # 方法3: 找任意两个数字
        nums = re.findall(r'\b\d+\b', text)
        if len(nums) >= 2:
            x, y = int(nums[0]), int(nums[1])
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                return (x, y)
        
        return None
    
    def predict_single(self, image, instruction: str, temperature: float = 0.0) -> Dict:
        """单次预测（P5-1 基线）"""
        prompt = self._build_prompt(instruction)
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            if temperature > 0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )
        
        result_text = self.processor.batch_decode(
            outputs, skip_special_tokens=True
        )[0]
        
        coord = self._parse_coordinate(result_text)
        
        return {
            "raw_output": result_text,
            "coordinate": coord,
            "parse_success": coord is not None,
        }
    
    def predict_multiple(self, image, instruction: str, num_samples: int = 5, temperature: float = 0.8) -> List[Dict]:
        """多次采样预测（P5-2/P5-3 基础）"""
        predictions = []
        for i in range(num_samples):
            result = self.predict_single(image, instruction, temperature=temperature)
            predictions.append(result)
            coord_str = str(result['coordinate']) if result['coordinate'] else "FAIL"
            print(f"  Sample {i+1}/{num_samples}: {coord_str}")
        return predictions
    
    def compute_consistency(self, predictions: List[Dict]) -> Dict:
        """计算一致性分数"""
        coords = [p["coordinate"] for p in predictions if p["coordinate"] is not None]
        
        if len(coords) < 2:
            return {
                "consistency_score": 0.0,
                "num_valid": len(coords),
                "cluster_center": coords[0] if coords else None,
                "is_reliable": False,
            }
        
        points = cdist(coords, coords, metric='euclidean')
        mask = cdist(coords, coords, metric='euclidean')
        # 修正：上面写错了，重新写
        
        import numpy as np
        points = np.array(coords, dtype=np.float32)
        distances = cdist(points, points, metric='euclidean')
        mask = np.triu(np.ones_like(distances), k=1).astype(bool)
        avg_distance = distances[mask].mean()
        consistency_score = avg_distance / 1000.0
        cluster_center = tuple(np.median(points, axis=0).astype(int).tolist())
        is_reliable = consistency_score < 0.15
        
        return {
            "consistency_score": float(consistency_score),
            "avg_distance": float(avg_distance),
            "num_valid": len(coords),
            "cluster_center": cluster_center,
            "is_reliable": is_reliable,
            "all_coords": coords,
        }
    
    def predict_with_retry(self, image, instruction: str, num_samples: int = 5, temperature: float = 0.8, max_retries: int = 2) -> Dict:
        """P5-2: 低一致性时重试"""
        print(f"Initial sampling: {num_samples} samples")
        predictions = self.predict_multiple(image, instruction, num_samples, temperature)
        consistency = self.compute_consistency(predictions)
        
        retries = 0
        while not consistency["is_reliable"] and retries < max_retries:
            retries += 1
            print(f"Low consistency ({consistency['consistency_score']:.3f}), retry {retries}/{max_retries}")
            new_predictions = self.predict_multiple(image, instruction, num_samples, temperature)
            predictions.extend(new_predictions)
            consistency = self.compute_consistency(predictions)
        
        final_coord = consistency["cluster_center"] if consistency["is_reliable"] else None
        
        return {
            "strategy": "retry",
            "final_coordinate": final_coord,
            "consistency": consistency,
            "all_predictions": predictions,
            "num_retries": retries,
        }
    
    def predict_with_aggregate(self, image, instruction: str, num_samples: int = 5, temperature: float = 0.8) -> Dict:
        """P5-3: 多次预测后聚合"""
        print(f"Sampling: {num_samples} samples")
        predictions = self.predict_multiple(image, instruction, num_samples, temperature)
        consistency = self.compute_consistency(predictions)
        
        final_coord = consistency["cluster_center"]
        
        return {
            "strategy": "aggregate",
            "final_coordinate": final_coord,
            "consistency": consistency,
            "all_predictions": predictions,
        }


if __name__ == "__main__":
    from PIL import Image
    
    model_path = "./hf_cache/qwen/Qwen/Qwen2-VL-7B-Instruct"
    predictor = GUIGroundingPredictor(model_path)
    
    test_img = Image.new('RGB', (800, 600), color='white')
    
    print("\n=== P5-1: Single Prediction ===")
    result = predictor.predict_single(test_img, "点击右上角的搜索按钮")
    print(f"Parsed: {result['coordinate']}, Success: {result['parse_success']}")
    
    print("\n=== P5-2: Retry Strategy ===")
    result = predictor.predict_with_retry(test_img, "点击右上角的搜索按钮", num_samples=3, max_retries=1)
    print(f"Final: {result['final_coordinate']}, Consistency: {result['consistency']['consistency_score']:.3f}")
    
    print("\n=== P5-3: Aggregate Strategy ===")
    result = predictor.predict_with_aggregate(test_img, "点击右上角的搜索按钮", num_samples=3)
    print(f"Final: {result['final_coordinate']}, Consistency: {result['consistency']['consistency_score']:.3f}")