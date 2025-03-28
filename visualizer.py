import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

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
        total_enron = train_stats['enron_train'] + train_stats['enron_test']
        total_spamassassin = train_stats['spamassassin_train'] + train_stats['spamassassin_test']
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

        print("Generating data distribution in test set")
        plt.figure()
        plt.pie([train_stats['enron_test'], train_stats['spamassassin_test']], 
                labels=['Enron', 'SpamAssassin'],
                colors=self.pie_colors,
                autopct='%1.1f%%')
        plt.title('Test Set Data Distribution')
        plt.savefig(f'{self.output_dir}/test_source_distribution_{self.timestamp}.png',
                   bbox_inches='tight')
        plt.close()

        print("Generating ham/sam distribution in test set")
        plt.figure()
        plt.pie([train_stats['test_ham'], train_stats['test_spam']], 
                labels=['Ham', 'Spam'],
                colors=self.pie_colors,
                autopct='%1.1f%%')
        plt.title('TestSet Ham/Spam Distribution')
        plt.savefig(f'{self.output_dir}/test_label_distribution_{self.timestamp}.png',
                   bbox_inches='tight')
        plt.close()

    def plot_performance_comparison(self, test_results):
        print("Generating overall accuracy")
        plt.figure()
        models = list(test_results.keys())
        x = np.arange(3)
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            results = test_results[model]
            plt.bar(x + i*width, 
                   [results['overall']['accuracy'],
                    results['enron']['accuracy'],
                    results['spamassassin']['accuracy']], 
                   width, 
                   label=model.upper(),
                   color=self.colors[i],
                   edgecolor='black',
                   linewidth=1)
        
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.xticks(x + width*len(models)/2, ['Overall', 'Enron', 'SpamAssassin'])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_performance_{self.timestamp}.png',
                   bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(self, test_results):
        for model_name, results in test_results.items():
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
            
            for ax, data, title in [
                (ax1, results['overall']['confusion_matrix'], 'Overall'),
                (ax2, results['enron']['confusion_matrix'], 'Enron'),
                (ax3, results['spamassassin']['confusion_matrix'], 'SpamAssassin')
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

    def plot_accuracy_table(self, test_results):
        plt.figure(figsize=(12, 4))
        plt.axis('off')
        
        models = list(test_results.keys())
        cell_data = []
        
        for model in models:
            row = [
                f"{test_results[model]['overall']['accuracy']:.4f}",
                f"{test_results[model]['enron']['accuracy']:.4f}",
                f"{test_results[model]['spamassassin']['accuracy']:.4f}"
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
    
    def generate_text_report(self, train_stats, test_results, is_drytest=False):
        with open(f'{self.output_dir}/detailed_report_{self.timestamp}.txt', 'w') as f:
            f.write("Spam Detection Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")

            f.write("Dataset Distribution:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Emails: {train_stats['total_ham'] + train_stats['total_spam']}\n")
            f.write(f"Ham Emails: {train_stats['total_ham']} ({train_stats['total_ham']/(train_stats['total_ham'] + train_stats['total_spam']):.1%})\n")
            f.write(f"Spam Emails: {train_stats['total_spam']} ({train_stats['total_spam']/(train_stats['total_ham'] + train_stats['total_spam']):.1%})\n\n")

            f.write("Test Set Distribution:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Test Emails: {train_stats['test_ham'] + train_stats['test_spam']}\n")
            f.write(f"From Enron: {train_stats['enron_test']}\n")
            f.write(f"From SpamAssassin: {train_stats['spamassassin_test']}\n")
            f.write(f"Ham Emails: {train_stats['test_ham']} ({train_stats['test_ham']/(train_stats['test_ham'] + train_stats['test_spam']):.1%})\n")
            f.write(f"Spam Emails: {train_stats['test_spam']} ({train_stats['test_spam']/(train_stats['test_ham'] + train_stats['test_spam']):.1%})\n\n")

            f.write("Model Performance:\n")
            f.write("-" * 30 + "\n")
            for model in test_results.keys():
                f.write(f"\n{model.upper()}:\n")
                f.write(f"Overall Accuracy: {test_results[model]['overall']['accuracy']:.4f}\n")
                f.write(f"Enron Accuracy: {test_results[model]['enron']['accuracy']:.4f}\n")
                f.write(f"SpamAssassin Accuracy: {test_results[model]['spamassassin']['accuracy']:.4f}\n") 