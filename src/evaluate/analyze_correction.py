"""
分析自纠错效果
- 统计纠正错误数
- 统计改错数量
- 分析成功和失败案例
"""

import json
import os
from collections import defaultdict


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


def load_results(results_dir):
    """加载所有策略的结果"""
    strategies = ['single', 'retry', 'aggregate']
    results = {}

    for strategy in strategies:
        file = os.path.join(results_dir, f'eval_{strategy}_web.json')
        if os.path.exists(file):
            with open(file) as f:
                data = json.load(f)
            results[strategy] = data['results']

    return results


def analyze_retry_effects(single_results, retry_results):
    """分析 Retry 策略的效果"""
    corrected = []      # 纠正错误：Single错 → Retry对
    corrupted = []      # 改错：Single对 → Retry错
    unchanged_wrong = [] # 仍然是错的
    unchanged_right = [] # 仍然是对的

    for s, r in zip(single_results, retry_results):
        s_correct = s['correct']
        r_correct = r['correct']

        if not s_correct and r_correct:
            corrected.append({
                'instruction': s['instruction'],
                'single_pred': s['predicted'],
                'retry_pred': r['predicted'],
                'img_filename': s['img_filename']
            })
        elif s_correct and not r_correct:
            corrupted.append({
                'instruction': s['instruction'],
                'single_pred': s['predicted'],
                'retry_pred': r['predicted'],
                'img_filename': s['img_filename']
            })
        elif not s_correct and not r_correct:
            unchanged_wrong.append({
                'instruction': s['instruction'],
                'single_pred': s['predicted'],
                'retry_pred': r['predicted'],
                'img_filename': s['img_filename']
            })
        else:
            unchanged_right.append({
                'instruction': s['instruction'],
                'single_pred': s['predicted'],
                'retry_pred': r['predicted'],
                'img_filename': s['img_filename']
            })

    return {
        'corrected': corrected,
        'corrupted': corrupted,
        'unchanged_wrong': unchanged_wrong,
        'unchanged_right': unchanged_right
    }


def analyze_aggregate_effects(single_results, aggregate_results):
    """分析 Aggregate 策略的效果"""
    corrected = []      # 纠正错误：Single错 → Aggregate对
    corrupted = []      # 改错：Single对 → Aggregate错
    unchanged_wrong = [] # 仍然是错的
    unchanged_right = [] # 仍然是对的

    for s, a in zip(single_results, aggregate_results):
        s_correct = s['correct']
        a_correct = a['correct']

        if not s_correct and a_correct:
            corrected.append({
                'instruction': s['instruction'],
                'single_pred': s['predicted'],
                'aggregate_pred': a['predicted'],
                'img_filename': s['img_filename']
            })
        elif s_correct and not a_correct:
            corrupted.append({
                'instruction': s['instruction'],
                'single_pred': s['predicted'],
                'aggregate_pred': a['predicted'],
                'img_filename': s['img_filename']
            })
        elif not s_correct and not a_correct:
            unchanged_wrong.append({
                'instruction': s['instruction'],
                'single_pred': s['predicted'],
                'aggregate_pred': a['predicted'],
                'img_filename': s['img_filename']
            })
        else:
            unchanged_right.append({
                'instruction': s['instruction'],
                'single_pred': s['predicted'],
                'aggregate_pred': a['predicted'],
                'img_filename': s['img_filename']
            })

    return {
        'corrected': corrected,
        'corrupted': corrupted,
        'unchanged_wrong': unchanged_wrong,
        'unchanged_right': unchanged_right
    }


def analyze_by_data_type(results):
    """按数据类型分析"""
    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for r in results:
        dtype = r.get('data_type', 'unknown')
        type_stats[dtype]['total'] += 1
        if r['correct']:
            type_stats[dtype]['correct'] += 1

    return {k: {'total': v['total'], 'correct': v['correct'],
                'accuracy': v['correct']/v['total'] if v['total'] > 0 else 0}
            for k, v in type_stats.items()}


def analyze_by_data_source(results):
    """按数据来源分析"""
    source_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for r in results:
        source = r.get('data_source', 'unknown')
        source_stats[source]['total'] += 1
        if r['correct']:
            source_stats[source]['correct'] += 1

    return {k: {'total': v['total'], 'correct': v['correct'],
                'accuracy': v['correct']/v['total'] if v['total'] > 0 else 0}
            for k, v in source_stats.items()}


def print_analysis(results_dir='/mnt/workspace/gui-project/experiments/person5'):
    """打印分析结果"""
    print('='*70)
    print('Person5 自纠错效果分析')
    print('='*70)
    print()

    # 加载结果
    results = load_results(results_dir)

    if 'single' not in results:
        print("错误：找不到 single 策略的结果")
        return

    print(f"样本总数: {len(results['single'])}")
    print()

    # 各策略准确率
    print("各策略准确率:")
    print("-"*40)
    for strategy in ['single', 'retry', 'aggregate']:
        if strategy in results:
            data = results[strategy]
            acc = sum(r['correct'] for r in data) / len(data)
            correct = sum(r['correct'] for r in data)
            print(f"  {strategy.upper():12} {acc:>6.1%} ({correct}/{len(data)})")
    print()

    # Retry 效果分析
    if 'retry' in results:
        print("RETRY 策略效果分析:")
        print("-"*40)
        retry_effects = analyze_retry_effects(results['single'], results['retry'])
        print(f"  ✅ 纠正错误: {len(retry_effects['corrected'])} 例")
        print(f"  ❌ 改错:     {len(retry_effects['corrupted'])} 例")
        print(f"  ➖ 仍错:     {len(retry_effects['unchanged_wrong'])} 例")
        print(f"  ✓ 仍对:     {len(retry_effects['unchanged_right'])} 例")
        print()

        if retry_effects['corrected']:
            print("  纠正成功案例 (前5个):")
            for i, case in enumerate(retry_effects['corrected'][:5]):
                print(f"    {i+1}. {case['instruction'][:40]}")
                print(f"       Single: {case['single_pred']} → Retry: {case['retry_pred']}")
            print()

        if retry_effects['corrupted']:
            print("  改错案例 (前5个):")
            for i, case in enumerate(retry_effects['corrupted'][:5]):
                print(f"    {i+1}. {case['instruction'][:40]}")
                print(f"       Single: {case['single_pred']} → Retry: {case['retry_pred']}")
            print()

    # Aggregate 效果分析
    if 'aggregate' in results:
        print("AGGREGATE 策略效果分析:")
        print("-"*40)
        agg_effects = analyze_aggregate_effects(results['single'], results['aggregate'])
        print(f"  ✅ 纠正错误: {len(agg_effects['corrected'])} 例")
        print(f"  ❌ 改错:     {len(agg_effects['corrupted'])} 例")
        print(f"  ➖ 仍错:     {len(agg_effects['unchanged_wrong'])} 例")
        print(f"  ✓ 仍对:     {len(agg_effects['unchanged_right'])} 例")
        print()

        if agg_effects['corrected']:
            print("  纠正成功案例 (前5个):")
            for i, case in enumerate(agg_effects['corrected'][:5]):
                print(f"    {i+1}. {case['instruction'][:40]}")
                print(f"       Single: {case['single_pred']} → Aggregate: {case['aggregate_pred']}")
            print()

        if agg_effects['corrupted']:
            print("  改错案例 (前5个):")
            for i, case in enumerate(agg_effects['corrupted'][:5]):
                print(f"    {i+1}. {case['instruction'][:40]}")
                print(f"       Single: {case['single_pred']} → Aggregate: {case['aggregate_pred']}")
            print()

    # 按数据类型分析
    print("按数据类型分析 (Single 策略):")
    print("-"*40)
    type_analysis = analyze_by_data_type(results['single'])
    for dtype, stats in type_analysis.items():
        print(f"  {dtype:12} {stats['accuracy']:>6.1%} ({stats['correct']}/{stats['total']})")
    print()

    # 按数据来源分析
    print("按数据来源分析 (Single 策略):")
    print("-"*40)
    source_analysis = analyze_by_data_source(results['single'])
    for source, stats in source_analysis.items():
        print(f"  {source:12} {stats['accuracy']:>6.1%} ({stats['correct']}/{stats['total']})")
    print()

    print('='*70)


def save_detailed_report(output_file='/mnt/workspace/gui-project/experiments/person5/correction_analysis.json'):
    """保存详细分析报告"""
    results_dir = '/mnt/workspace/gui-project/experiments/person5'
    results = load_results(results_dir)

    report = {
        'total_samples': len(results['single']),
        'strategies': {},
        'retry_effects': analyze_retry_effects(results['single'], results.get('retry', [])),
        'aggregate_effects': analyze_aggregate_effects(results['single'], results.get('aggregate', [])),
        'by_data_type': analyze_by_data_type(results['single']),
        'by_data_source': analyze_by_data_source(results['single'])
    }

    for strategy in ['single', 'retry', 'aggregate']:
        if strategy in results:
            data = results[strategy]
            report['strategies'][strategy] = {
                'total': len(data),
                'correct': sum(r['correct'] for r in data),
                'accuracy': sum(r['correct'] for r in data) / len(data)
            }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"详细报告已保存到: {output_file}")


if __name__ == "__main__":
    print_analysis()
    save_detailed_report()
