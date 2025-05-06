from collections import defaultdict
import numpy as np

from visualizer import SpamVisualizer


def parse_log_to_results(log_path: str, model_name='llm'):
    results = {
        model_name: {
            'overall': {'confusion_matrix': np.zeros((2, 2), dtype=int)},
            'enron': {'confusion_matrix': np.zeros((2, 2), dtype=int)},
            'spamassassin': {'confusion_matrix': np.zeros((2, 2), dtype=int)},
        }
    }

    stats = defaultdict(int)

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith("index="):
                continue
            parts = dict(p.split('=') for p in line.strip().split(',') if '=' in p and '=' in p)

            required_keys = {'true', 'pred', 'source'}
            if not required_keys.issubset(parts):
                continue  # 跳过非法行

            source = parts['source']
            true = parts['true']
            pred = parts['pred']

            y_true = 1 if true == 'spam' else 0
            y_pred = 1 if pred == 'spam' else 0

            source_key = 'enron' if 'enron' in source else 'spamassassin'
            results[model_name][source_key]['confusion_matrix'][y_true][y_pred] += 1
            results[model_name]['overall']['confusion_matrix'][y_true][y_pred] += 1

            stats[f'total_{true}'] += 1
            if 'val' in source or 'enron' in source or 'spamassassin' in source:
                stats[f'{source_key}_val'] += 1
                stats[f'val_{true}'] += 1

    # compute accuracy
    for split in ['overall', 'enron', 'spamassassin']:
        cm = results[model_name][split]['confusion_matrix']
        correct = cm[0][0] + cm[1][1]
        total = cm.sum()
        acc = correct / total if total > 0 else 0
        results[model_name][split]['accuracy'] = round(acc, 4)

    # convert defaultdict to normal dict
    stats = dict(stats)
    # 映射 overall 为 test
    results[model_name]['test'] = {
        'confusion_matrix': results[model_name]['overall']['confusion_matrix'].copy(),
        'accuracy': results[model_name]['overall']['accuracy']
    }

    stats['test_ham'] = stats.get('total_ham', 0)
    stats['test_spam'] = stats.get('total_spam', 0)
    return stats, results

logfile = 'log/classification_2025-05-05_18-20-16.log'
stats, results = parse_log_to_results(logfile)
viz = SpamVisualizer()

viz.plot_test_performance(results)
viz.plot_test_confusion_matrices(results)
viz.generate_test_report(stats, results)
