"""
Person5: GUI 定位 Demo
支持图片上传、指令输入、预测结果可视化
"""

import sys
sys.path.insert(0, "src")

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os
from infer.self_correction import GUIGroundingPredictor


# 模型路径
MODEL_PATH = "./hf_cache/qwen/Qwen/Qwen3-VL-4B-Instruct"

# 颜色配置
COLORS = {
    'predicted': '#00FF00',     # 绿色 - 最终预测点
    'samples': '#FF6600',       # 橙色 - 多次采样点
    'bbox': '#0066FF',          # 蓝色 - 边界框
    'text': '#FFFFFF',          # 白色 - 文字
    'bg': '#000000',            # 黑色 - 背景
}


def draw_point_with_confidence(img, point, confidence, label="预测点"):
    """在图片上绘制带置信度的点"""
    draw = ImageDraw.Draw(img)
    x, y = point

    # 绘制中心点
    radius = 8
    draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                 fill=COLORS['predicted'], outline='white', width=2)

    # 绘制置信度标签
    font = ImageFont.load_default()
    text = f"{label}: ({int(x)}, {int(y)}) - 置信度: {confidence:.2f}"
    text_bbox = draw.textbbox((x+15, y-15), text, font=font)
    draw.rectangle(text_bbox, fill=COLORS['bg'])
    draw.text((x+15, y-15), text, fill=COLORS['text'], font=font)

    return img


def draw_samples(img, samples):
    """在图片上绘制多次采样点"""
    draw = ImageDraw.Draw(img)

    for i, point in enumerate(samples):
        x, y = point
        # 绘制小点
        radius = 4
        draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                     fill=COLORS['samples'], outline='white', width=1)

    return img


def draw_ground_truth(img, bbox):
    """绘制真实边界框"""
    if bbox is None:
        return img

    draw = ImageDraw.Draw(img)
    x, y, w, h = bbox

    # 绘制边界框
    draw.rectangle([x, y, x+w, y+h], outline=COLORS['bbox'], width=3)

    # 添加标签
    font = ImageFont.load_default()
    text = "真实位置"
    text_bbox = draw.textbbox((x, y-25), text, font=font)
    draw.rectangle(text_bbox, fill=COLORS['bbox'])
    draw.text((x, y-25), text, fill=COLORS['text'], font=font)

    return img


def predict_single(image, instruction, strategy):
    """执行单次预测"""
    if image is None or not instruction.strip():
        return None, "请上传图片并输入指令", "", ""

    try:
        # 转换图片为 RGB
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = Image.fromarray(image).convert('RGB')

        # 执行预测
        if strategy == "Single (单次预测)":
            result = predictor.predict_single(img, instruction)
            coord = result['coordinate']
            consistency_score = 1.0
            all_coords = [coord] if coord else []
            is_reliable = True

        elif strategy == "Retry (低置信重试)":
            result = predictor.predict_with_retry(
                img, instruction, num_samples=3, max_retries=1
            )
            coord = result['final_coordinate']
            consistency = result['consistency']
            consistency_score = consistency['consistency_score']
            all_coords = consistency.get('all_coords', [])
            is_reliable = consistency['is_reliable']

        else:  # Aggregate
            result = predictor.predict_with_aggregate(
                img, instruction, num_samples=5
            )
            coord = result['final_coordinate']
            consistency = result['consistency']
            consistency_score = consistency['consistency_score']
            all_coords = consistency.get('all_coords', [])
            is_reliable = consistency['is_reliable']

        # 准备结果
        if coord:
            # 转换坐标到像素
            img_w, img_h = img.size
            px, py = int(coord[0] / 1000 * img_w), int(coord[1] / 1000 * img_h)

            # 绘制结果
            result_img = img.copy()

            # 绘制所有采样点
            if len(all_coords) > 1:
                pixel_coords = [(int(c[0]/1000*img_w), int(c[1]/1000*img_h))
                               for c in all_coords]
                result_img = draw_samples(result_img, pixel_coords)

            # 绘制最终预测点
            confidence_text = "高置信" if is_reliable else "低置信"
            result_img = draw_point_with_confidence(
                result_img, (px, py), consistency_score, confidence_text
            )

            # 准备文本输出
            summary = f"""
预测成功！📍

策略: {strategy}
预测坐标: ({coord[0]}, {coord[1]})
像素坐标: ({px}, {py})
一致性分数: {consistency_score:.3f}
置信状态: {'高置信' if is_reliable else '低置信'}
采样次数: {len(all_coords)}
"""

            raw_output = result.get('raw_output', 'N/A')[:500]

            return result_img, summary, raw_output, json.dumps({
                'coordinate': coord,
                'pixel_coord': [px, py],
                'consistency_score': consistency_score,
                'is_reliable': is_reliable,
                'num_samples': len(all_coords)
            }, indent=2, ensure_ascii=False)
        else:
            return img, "❌ 预测失败：无法解析坐标", "", ""

    except Exception as e:
        return None, f"❌ 错误: {str(e)}", "", ""


def predict_with_gt(image, instruction, gt_bbox, strategy):
    """带真实标注的预测"""
    if image is None or not instruction.strip():
        return None, "请上传图片并输入指令", ""

    try:
        # 解析真实边界框
        if gt_bbox and gt_bbox.strip():
            bbox = [int(x) for x in gt_bbox.split(',')]
            if len(bbox) != 4:
                bbox = None
        else:
            bbox = None

        # 转换图片
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = Image.fromarray(image).convert('RGB')

        # 执行预测
        if strategy == "Single (单次预测)":
            result = predictor.predict_single(img, instruction)
            coord = result['coordinate']
        elif strategy == "Retry (低置信重试)":
            result = predictor.predict_with_retry(
                img, instruction, num_samples=3, max_retries=1
            )
            coord = result['final_coordinate']
        else:  # Aggregate
            result = predictor.predict_with_aggregate(
                img, instruction, num_samples=5
            )
            coord = result['final_coordinate']

        # 判断是否正确
        correct = False
        if coord and bbox:
            img_w, img_h = img.size
            px, py = int(coord[0] / 1000 * img_w), int(coord[1] / 1000 * img_h)

            x, y, w, h = bbox
            correct = (x <= px <= x + w) and (y <= py <= y + h)

        # 绘制结果
        result_img = img.copy()
        if bbox:
            result_img = draw_ground_truth(result_img, bbox)

        if coord:
            img_w, img_h = img.size
            px, py = int(coord[0] / 1000 * img_w), int(coord[1] / 1000 * img_h)
            result_img = draw_point_with_confidence(
                result_img, (px, py), 0.9, "预测点" if correct else "错误预测"
            )

        status = "✅ 预测正确！" if correct else "❌ 预测错误"
        summary = f"{status}\n策略: {strategy}\n预测: {coord}\n真实: {bbox}"

        return result_img, summary

    except Exception as e:
        return None, f"❌ 错误: {str(e)}", ""


# 创建 Gradio 界面
def create_demo():
    with gr.Blocks(title="GUI Grounding Demo - Person5") as demo:
        gr.Markdown("# 🖱️ GUI 元素定位 Demo")
        gr.Markdown("上传 GUI 截图，输入操作指令，查看预测结果和置信度分析")

        with gr.Tabs():
            # Tab 1: 基础预测
            with gr.Tab("📍 基础预测"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="上传 GUI 截图")
                        instruction_input = gr.Textbox(
                            label="操作指令",
                            placeholder="例如：点击右上角的搜索按钮",
                            lines=2
                        )
                        strategy_input = gr.Radio(
                            choices=[
                                "Single (单次预测)",
                                "Retry (低置信重试)",
                                "Aggregate (多次聚合)"
                            ],
                            value="Single (单次预测)",
                            label="预测策略"
                        )
                        predict_btn = gr.Button("🚀 开始预测", variant="primary")

                    with gr.Column():
                        image_output = gr.Image(label="预测结果")
                        summary_output = gr.Textbox(label="结果摘要", lines=10)
                        raw_output = gr.Textbox(label="原始输出", lines=5)
                        json_output = gr.JSON(label="详细数据")

                predict_btn.click(
                    fn=predict_single,
                    inputs=[image_input, instruction_input, strategy_input],
                    outputs=[image_output, summary_output, raw_output, json_output]
                )

            # Tab 2: 带真实标注
            with gr.Tab("🎯 带真实标注验证"):
                with gr.Row():
                    with gr.Column():
                        image_input2 = gr.Image(label="上传 GUI 截图")
                        instruction_input2 = gr.Textbox(
                            label="操作指令",
                            placeholder="例如：点击右上角的搜索按钮",
                            lines=2
                        )
                        gt_bbox_input = gr.Textbox(
                            label="真实边界框 (x,y,w,h)",
                            placeholder="例如：2321,129,208,70",
                            lines=1
                        )
                        strategy_input2 = gr.Radio(
                            choices=[
                                "Single (单次预测)",
                                "Retry (低置信重试)",
                                "Aggregate (多次聚合)"
                            ],
                            value="Single (单次预测)",
                            label="预测策略"
                        )
                        predict_btn2 = gr.Button("🚀 开始预测", variant="primary")

                    with gr.Column():
                        image_output2 = gr.Image(label="预测结果")
                        summary_output2 = gr.Textbox(label="结果摘要", lines=8)

                predict_btn2.click(
                    fn=predict_with_gt,
                    inputs=[image_input2, instruction_input2, gt_bbox_input, strategy_input2],
                    outputs=[image_output2, summary_output2]
                )

        gr.Markdown("## 📊 说明")
        gr.Markdown("""
        - **绿色点**：最终预测坐标
        - **橙色点**：多次采样点（Retry/Aggregate 策略）
        - **蓝色框**：真实目标位置（仅在验证标签页显示）
        - **置信度**：基于多次采样的一致性分数
        """)

    return demo


if __name__ == "__main__":
    print("加载模型...")
    global predictor
    predictor = GUIGroundingPredictor(MODEL_PATH)
    print("模型加载完成！")

    print("启动 Demo...")
    demo = create_demo()

    # 启动服务器
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,        # 端口号
        share=False,             # 不使用公共链接（云服务器上设为 False）
        show_error=True
    )
