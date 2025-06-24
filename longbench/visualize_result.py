import json
import os
import matplotlib.pyplot as plt
import numpy as np
import math

def visualize_results():
    """
    Visualizes model performance results from JSON files.
    
    This function reads performance data of different models from specified
    JSON files, processes the data, and generates a single figure with
    subplots for each dataset to compare model performance across various 
    context lengths. It also includes a subplot for the average score
    across all datasets. The generated plot is saved as a PNG file.
    """
    models = [
        'llama2-7b-hf',
        'llama2-7b-hf-32k',
        'llama2-7b-hf-slimpajama-ntk-32k',
        'llama2-7b-hf-slimpajama-yarn-32k',
        'llama2-7b-hf-slimpajama-pi-32k',
        'llama2-7b-hf-slimpajama-yarn-16k-long0.6-short0.4',
        'llama2-7b-hf-slimpajama-yarn-16k-long0.8-short0.2'
    ]
    base_pred_path = 'pred'
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    all_data = {}
    for model in models:
        json_path = os.path.join(base_pred_path, model, 'result_by_length.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = content.replace("NaN", "null")
            all_data[model] = json.loads(content)
        except FileNotFoundError:
            print(f"Warning: {json_path} not found. Skipping this model.")
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {json_path}. Error: {e}. Skipping this model.")

    models = [m for m in models if m in all_data]
    if not models:
        print("No data found for any model. Exiting.")
        return

    first_model_with_data = models[0]
    datasets = sorted(list(all_data[first_model_with_data].keys()))
    lengths = ["0-4k", "4-8k", "8-16k", "16-32k"]
    model_labels = {
        'llama2-7b-hf': 'Llama2-7B',
        'llama2-7b-hf-32k': 'Llama2-32K',
        'llama2-7b-hf-slimpajama-yarn-32k': 'YARN',
        'llama2-7b-hf-slimpajama-pi-32k': 'PI',
        'llama2-7b-hf-slimpajama-ntk-32k': 'NTK',
        'llama2-7b-hf-slimpajama-yarn-16k-long0.6-short0.4': 'YARN-16K-Long0.6-Short0.4',
        'llama2-7b-hf-slimpajama-yarn-16k-long0.8-short0.2': 'YARN-16K-Long0.8-Short0.2',
    }

    avg_scores_sum = {model: {length: 0 for length in lengths} for model in models}
    avg_scores_count = {model: {length: 0 for length in lengths} for model in models}

    num_datasets = len(datasets)
    num_plots = num_datasets + 1
    ncols = 4
    nrows = math.ceil(num_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 2.5 * nrows), constrained_layout=True)
    axes = axes.flatten()

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        valid_models = [model for model in models if dataset in all_data[model]]
        
        n_models = len(valid_models)
        bar_width = 0.8 / n_models
        index = np.arange(len(lengths))

        for j, model in enumerate(valid_models):
            scores = [all_data[model][dataset].get(length) for length in lengths]
            
            for k, length in enumerate(lengths):
                if scores[k] is not None:
                    avg_scores_sum[model][length] += scores[k]
                    avg_scores_count[model][length] += 1
            
            scores_proc = [0 if s is None else s for s in scores]
            
            pos = index - (0.8 / 2) + (j + 0.5) * bar_width
            ax.bar(pos, scores_proc, bar_width, label=model_labels.get(model, model))

        ax.set_title(dataset, fontsize=14)
        ax.set_xticks(index)
        ax.set_xticklabels(lengths, rotation=45, ha="right")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        if i % ncols == 0:
            ax.set_ylabel('Score', fontsize=12)

    avg_scores = {model: {} for model in models}
    for model in models:
        for length in lengths:
            if avg_scores_count[model][length] > 0:
                avg_scores[model][length] = avg_scores_sum[model][length] / avg_scores_count[model][length]
            else:
                avg_scores[model][length] = 0

    ax = axes[num_datasets]
    n_models = len(models)
    bar_width = 0.8 / n_models
    index = np.arange(len(lengths))

    for i, model in enumerate(models):
        scores = [avg_scores[model].get(length, 0) for length in lengths]
        pos = index - (0.8 / 2) + (i + 0.5) * bar_width
        ax.bar(pos, scores, bar_width, label=model_labels.get(model, model))

    ax.set_title('Average Score', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(lengths, rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    if num_datasets % ncols == 0:
        ax.set_ylabel('Score', fontsize=12)

    for i in range(num_plots, nrows * ncols):
        fig.delaxes(axes[i])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(models), bbox_to_anchor=(0.5, 1.05), fontsize=12)
    fig.suptitle('Model Performance by Context Length', fontsize=20)
    
    plot_path = os.path.join(output_dir, 'all_results_by_length.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {plot_path}")

if __name__ == '__main__':
    visualize_results()
