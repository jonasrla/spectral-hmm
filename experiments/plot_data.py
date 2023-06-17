import json
from argparse import ArgumentParser
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

metrics = ['time', 'abs_prob_error', 'abs_norm_prob_error']
metric_label = {
    'time': 'Tempo de treinamento (Segundos)',
    'abs_prob_error': 'Média do Erro Absoluto',
    'abs_norm_prob_error': 'Média do Erro Absoluto Normalizado'
}
tamanho = {
    'small': 'Pequenos',
    'medium': 'Médios',
    'large': 'Grandes',
}

def load_data(model_size):
    path = f'data/*data_{model_size}*'
    list_files = glob(path)
    data = dict()
    for path in list_files:
        dataset_size = int(path.split('.')[0].split('_')[-1])
        if not dataset_size in data:
            data[dataset_size] = dict()
            data[dataset_size]['EM'] = dict()
            data[dataset_size]['EM']['time'] = list()
            data[dataset_size]['EM']['abs_prob_error'] = list()
            data[dataset_size]['EM']['abs_norm_prob_error'] = list()
            data[dataset_size]['SL'] = dict()
            data[dataset_size]['SL']['time'] = list()
            data[dataset_size]['SL']['abs_prob_error'] = list()
            data[dataset_size]['SL']['abs_norm_prob_error'] = list()

        with open(path, 'r') as f:
            raw_json = json.load(f)
        
        if 'cloud' in path:
            data[dataset_size]['EM']['time'] += sum([
                session['em_est_time']
                for session in raw_json
            ], [])
        data[dataset_size]['EM']['abs_prob_error'] += sum([
            session['abs_prob_error']['em_ests']
            for session in raw_json
        ],[])
        data[dataset_size]['EM']['abs_norm_prob_error'] += sum([
            session['abs_norm_prob_error']['em_ests']
            for session in raw_json
        ],[])
        data[dataset_size]['SL']['time'] += sum([
            session['sl_est_time']
            for session in raw_json
        ], [])
        data[dataset_size]['SL']['abs_prob_error'] += [
            session['abs_prob_error']['sl_est']
            for session in raw_json
        ]
        data[dataset_size]['SL']['abs_norm_prob_error'] += [
            session['abs_norm_prob_error']['sl_est']
            for session in raw_json
        ]
    
    for size_key in data.keys():
        for model_key in data[size_key].keys():
            for metric_key in data[size_key][model_key].keys():
                data[size_key][model_key][metric_key] = np.array(
                    data[size_key][model_key][metric_key]
                )

    return data

def save_plot(data, metric, name, size):
    dataset_sizes = sorted(data.keys())
    
    sl_mins = np.array([data[d_size]['SL'][metric].min() for d_size in dataset_sizes])
    sl_maxes = np.array([data[d_size]['SL'][metric].max() for d_size in dataset_sizes])
    sl_means = np.array([data[d_size]['SL'][metric].mean() for d_size in dataset_sizes])
    sl_stds = np.array([data[d_size]['SL'][metric].std() for d_size in dataset_sizes])

    em_mins = np.array([data[d_size]['EM'][metric].min() for d_size in dataset_sizes])
    em_maxes = np.array([data[d_size]['EM'][metric].max() for d_size in dataset_sizes])
    em_means = np.array([data[d_size]['EM'][metric].mean() for d_size in dataset_sizes])
    em_stds = np.array([data[d_size]['EM'][metric].std() for d_size in dataset_sizes])

    fig, ax = plt.subplots()
    ax.errorbar(dataset_sizes, em_means, [em_means - em_mins, em_maxes - em_means], marker='o', mfc='#97D077', ecolor='#c5e5b4', color='#97D077', label='EM', lw=4)
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc='upper right',numpoints=1)
    ax.errorbar(dataset_sizes, em_means, em_stds, ecolor='#97D077', elinewidth=5, lw=0)
    
    ax.errorbar(dataset_sizes, sl_means, [sl_means - sl_mins, sl_maxes - sl_means], marker='.', mfc='#B5739D', ecolor='#cfa3bf', color='#B5739D', label='SL', lw=1)
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc='upper right',numpoints=1)
    ax.errorbar(dataset_sizes, sl_means, sl_stds, ecolor='#B5739D', elinewidth=2, lw=0, zorder=20)
    
    ax.xaxis.set_ticks(dataset_sizes)
    ax.xaxis.set_label_text('Tamanho do Treinamento')
    ax.yaxis.set_label_text(metric_label[metric])
    ax.set_title(f'Comparação entre Modelos com Paramêtros {tamanho[size]}')

    plt.savefig(name)


if __name__ == '__main__':
    for s in tamanho.keys():
        data = load_data(s)
        for m in metrics:
            save_plot(data, m, f'images/{s}_{m}.png', s)