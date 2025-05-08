import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

class SpamVisualizer:
    def __init__(self, output_dir='reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.rcParams.update({
            'figure.figsize': [6, 4],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.dpi': 300
        })

        self.colors = ['#4392F1', '#DC493A', '#6BB392']
        self.pie_colors = ['#4392F1', '#DC493A']
        self.heatmap_cmap = 'Blues'  #

    def plot_data_distributions(self, train_stats):
        print("Generating total data distribution")
        plt.figure()
        total_enron = train_stats['enron_train'] + train_stats['enron_val']
        total_spamassassin = train_stats['spamassassin_train'] + train_stats['spamassassin_val']
        plt.pie([total_enron, total_spamassassin],
                labels=['Enron', 'SpamAssassin'],
                autopct='%1.1f%%',
                colors=self.pie_colors)
        plt.title('Total Dataset Source Distribution')
        plt.savefig(f'{self.output_dir}/total_source_distribution_{self.timestamp}.png',
                    bbox_inches='tight')
        plt.close()

        print("Generating total ham/spam distribution")
        plt.figure()
        plt.pie([train_stats['total_ham'], train_stats['total_spam']],
                labels=['Ham', 'Spam'],
                colors=self.pie_colors,
                autopct='%1.1f%%')
        plt.title('Total Ham/Spam distribution')
        plt.savefig(f'{self.output_dir}/total_label_distribution_{self.timestamp}.png',
                    bbox_inches='tight')
        plt.close()

        print("Generating data distribution in validation set")
        plt.figure()
        plt.pie([train_stats['enron_val'], train_stats['spamassassin_val']],
                labels=['Enron', 'SpamAssassin'],
                colors=self.pie_colors,
                autopct='%1.1f%%')
        plt.title('Validation Set Data Distribution')
        plt.savefig(f'{self.output_dir}/validation_source_distribution_{self.timestamp}.png',
                    bbox_inches='tight')
        plt.close()

        print("Generating ham/spam distribution in validation set")
        plt.figure()
        plt.pie([train_stats['val_ham'], train_stats['val_spam']],
                labels=['Ham', 'Spam'],
                colors=self.pie_colors,
                autopct='%1.1f%%')
        plt.title('Validation Set Ham/Spam Distribution')
        plt.savefig(f'{self.output_dir}/validation_label_distribution_{self.timestamp}.png',
                    bbox_inches='tight')
        plt.close()

    def plot_performance_comparison(self, results):
        print("Generating overall accuracy")
        plt.figure()
        models = list(results.keys())
        x = np.arange(3)
        width = 0.8 / len(models)
        for i, model in enumerate(models):
            results_model = results[model]
            plt.bar(x + i * width,
                    [results_model['overall']['accuracy'],
                     results_model['enron']['accuracy'],
                     results_model['spamassassin']['accuracy']],
                    width,
                    label=model.upper(),
                    color=self.colors[i],
                    edgecolor='black',
                    linewidth=1)
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.xticks(x + width * len(models) / 2, ['Overall', 'Enron', 'SpamAssassin'])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_performance_{self.timestamp}.png',
                    bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(self, results):
        for model_name, results_model in results.items():
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
            for ax, data, title in [
                (ax1, results_model['overall']['confusion_matrix'], 'Overall'),
                (ax2, results_model['enron']['confusion_matrix'], 'Enron'),
                (ax3, results_model['spamassassin']['confusion_matrix'], 'SpamAssassin')
            ]:
                sns.heatmap(data,
                            annot=True,
                            fmt='d',
                            cmap=self.heatmap_cmap,
                            xticklabels=['Ham', 'Spam'],
                            yticklabels=['Ham', 'Spam'],
                            ax=ax,
                            cbar=False)
                ax.set_title(f'{model_name.upper()} on {title}')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/{model_name}_confusion_matrices_{self.timestamp}.png',
                        bbox_inches='tight')
            plt.close()

    def plot_accuracy_table(self, results):
        plt.figure(figsize=(12, 4))
        plt.axis('off')
        models = list(results.keys())
        cell_data = []
        for model in models:
            row = [
                f"{results[model]['overall']['accuracy']:.4f}",
                f"{results[model]['enron']['accuracy']:.4f}",
                f"{results[model]['spamassassin']['accuracy']:.4f}"
            ]
            cell_data.append(row)
        table = plt.table(
            cellText=cell_data,
            rowLabels=models,
            colLabels=['Overall', 'Enron', 'SpamAssassin'],
            loc='center',
            cellLoc='center',
            colWidths=[0.2] * 3
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.5, 2)
        plt.title('Model Accuracy Comparison', pad=20)
        plt.savefig(f'{self.output_dir}/accuracy_table_{self.timestamp}.png',
                    bbox_inches='tight')
        plt.close()

    def generate_text_report(self, stats, results, is_drytest=False):
        with open(f'{self.output_dir}/detailed_report_{self.timestamp}.txt', 'w') as f:
            f.write("Spam Detection Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("Dataset Distribution:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Emails: {stats['total_ham'] + stats['total_spam']}\n")
            f.write(
                f"Ham Emails: {stats['total_ham']} ({stats['total_ham'] / (stats['total_ham'] + stats['total_spam']):.1%})\n")
            f.write(
                f"Spam Emails: {stats['total_spam']} ({stats['total_spam'] / (stats['total_ham'] + stats['total_spam']):.1%})\n\n")

            if 'val_ham' in stats:
                f.write("Validation Set Distribution:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Validation Emails: {stats['val_ham'] + stats['val_spam']}\n")
                f.write(f"From Enron: {stats['enron_val']}\n")
                f.write(f"From SpamAssassin: {stats['spamassassin_val']}\n")
                f.write(
                    f"Ham Emails: {stats['val_ham']} ({stats['val_ham'] / (stats['val_ham'] + stats['val_spam']):.1%})\n")
                f.write(
                    f"Spam Emails: {stats['val_spam']} ({stats['val_spam'] / (stats['val_ham'] + stats['val_spam']):.1%})\n\n")
            elif 'test_ham' in stats:
                f.write("Test Set Distribution:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Test Emails: {stats['test_ham'] + stats['test_spam']}\n")
                f.write(
                    f"Ham Emails: {stats['test_ham']} ({stats['test_ham'] / (stats['test_ham'] + stats['test_spam']):.1%})\n")
                f.write(
                    f"Spam Emails: {stats['test_spam']} ({stats['test_spam'] / (stats['test_ham'] + stats['test_spam']):.1%})\n\n")

            f.write("Model Performance:\n")
            f.write("-" * 30 + "\n")
            for model in results.keys():
                f.write(f"\n{model.upper()}:\n")
                f.write(f"Overall Accuracy: {results[model]['overall']['accuracy']:.4f}\n")
                f.write(f"Enron Accuracy: {results[model]['enron']['accuracy']:.4f}\n")
                f.write(f"SpamAssassin Accuracy: {results[model]['spamassassin']['accuracy']:.4f}\n")

    def plot_test_performance(self, results):
        print("Generating test set performance visualization")
        plt.figure()
        models = list(results.keys())
        accuracies = [results[model]['test']['accuracy'] for model in models]

        plt.bar(range(len(models)), accuracies,
                color=self.colors[:len(models)],
                edgecolor='black',
                linewidth=1)

        plt.ylabel('Accuracy')
        plt.title('Model Performance on Test Set')
        plt.xticks(range(len(models)), [model.upper() for model in models])
        plt.grid(True, linestyle='--', alpha=0.7)
        for i, acc in enumerate(accuracies):
            plt.text(i, acc, f'{acc:.4f}',
                     ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/test_performance_{self.timestamp}.png',
                    bbox_inches='tight')
        plt.close()

    def plot_test_confusion_matrices(self, results):
        for model_name, results_model in results.items():
            plt.figure(figsize=(5, 4))
            sns.heatmap(results_model['test']['confusion_matrix'],
                        annot=True,
                        fmt='d',
                        cmap=self.heatmap_cmap,
                        xticklabels=['Ham', 'Spam'],
                        yticklabels=['Ham', 'Spam'],
                        cbar=False)
            plt.title(f'{model_name.upper()} Test Set Confusion Matrix')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/{model_name}_test_confusion_matrix_{self.timestamp}.png',
                        bbox_inches='tight')
            plt.close()

    def generate_test_report(self, stats, results):
        with open(f'{self.output_dir}/test_report_{self.timestamp}.txt', 'w') as f:
            f.write("Final Test Set Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("Test Set Distribution:\n")
            f.write("-" * 30 + "\n")
            total_test = stats['test_ham'] + stats['test_spam']
            f.write(f"Total Test Emails: {total_test}\n")
            f.write(f"Ham Emails: {stats['test_ham']} ({stats['test_ham'] / total_test:.1%})\n")
            f.write(f"Spam Emails: {stats['test_spam']} ({stats['test_spam'] / total_test:.1%})\n\n")
            f.write("Model Performance on Test Set:\n")
            f.write("-" * 30 + "\n")
            for model in results.keys():
                f.write(f"\n{model.upper()}:\n")
                f.write(f"Accuracy: {results[model]['test']['accuracy']:.4f}\n")
                cm = results[model]['test']['confusion_matrix']
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n")
                f.write(f"True Negatives: {tn}\n")
                f.write(f"False Positives: {fp}\n")
                f.write(f"False Negatives: {fn}\n")
                f.write(f"True Positives: {tp}\n")
