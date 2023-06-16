from time import time
import json
from argparse import ArgumentParser

from hmmlearn.hmm import CategoricalHMM
import numpy as np
from ext.HMM import SLHMM

def random_parameters(n_components, n_features):
    startprob = np.random.rand(n_components)
    startprob = startprob / startprob.sum()

    transmat = np.random.rand(n_components, n_components)
    transmat = (transmat.T / transmat.sum(1)).T

    emissionprob = np.random.rand(n_components, n_features)
    emissionprob = (emissionprob.T / emissionprob.sum(1)).T

    return (startprob, transmat, emissionprob)

class myCategoricalHMM(CategoricalHMM):
    def score(self, X, lengths=None):
        return np.exp(super().score(X, lengths))

def create_generators(**gen_param):
    if 'startprob' in gen_param:
        gen = myCategoricalHMM(
            n_components=gen_param['n_components'],
            n_features=gen_param['n_features']
        )
        gen.startprob_ = gen_param['startprob']
        gen.transmat_ = gen_param['transmat']
        gen.emissionprob_ = gen_param['emissionprob']
        gens = [gen]
    else:
        gens = [
            myCategoricalHMM(
                n_components=gen_param['n_components'],
                n_features=gen_param['n_features']
            ) for i in range(gen_param.get('n_models'))
        ]
        for i in range(gen_param.get('n_models')):
            gens[i].startprob_, gens[i].transmat_, gens[i].emissionprob_ = random_parameters(
                gen_param['n_components'],
                gen_param['n_features']
            )
    return gens

def create_em_estimators(samples, **em_est_param):
    n_gens = len(samples)
    est_fit_t = np.zeros((n_gens, em_est_param['n_models']))
    em_ests = \
    [
        [
            myCategoricalHMM(
            n_components=em_est_param['n_components'],
            n_features=em_est_param['n_features']
            ) for i in range(em_est_param['n_models'])
        ] for j in range(n_gens)
    ]
    for gen_index in range(n_gens):
        for em_est_index, em_est in enumerate(em_ests[gen_index]):
            sample = samples[gen_index]
            em_data = np.concatenate(sample)
            em_lenghts = list(map(len, sample))
            tic = time()
            em_est.fit(em_data, em_lenghts)
            tac = time()
            est_fit_t[gen_index][em_est_index] = (tac-tic)
    return em_ests, est_fit_t

def create_sl_estimators(samples, **sl_est_param):
    n_gens = len(samples)
    est_fit_t = np.zeros((n_gens))
    sl_ests = \
    [
        SLHMM(
            num_hidden=sl_est_param['n_components'],
            num_observ=sl_est_param['n_features'])
        for i in range(n_gens)
    ]
    for gen_index, sl_est in enumerate(sl_ests):
        sample = samples[gen_index]
        sl_data = [s.T[0] for s in sample]
        tic = time()
        sl_est.fit(sl_data)
        tac = time()
        est_fit_t[gen_index] = (tac-tic)
    return sl_ests, est_fit_t

def abs_prob_error(gen, est, sample):
    return sum([np.abs(gen.score(s) - est.score(s)) for s in sample]) / len(sample)

def abs_norm_prob_error(gen, est, sample):
    return sum([np.power(np.abs(gen.score(s) - est.score(s)), 1/len(s))
                for s in sample]) / len(sample)

def em_sl_comparison(gen_param, em_est_param, sl_est_param, n_samples, max_t=30, metrics=[]):
    gens = create_generators(**gen_param)
    
    samples = \
    [
        [
            gen.sample(np.random.randint(2,max_t))[0]
            for i in range(n_samples)
        ] for gen in gens
    ]

    em_ests, em_est_fit_t = create_em_estimators(samples, **em_est_param)
    sl_ests, sl_est_fit_t = create_sl_estimators(samples, **sl_est_param)

    test_samples = \
    [
        [
            gen.sample(np.random.randint(2,max_t))[0]
            for i in range(int(n_samples*0.3))
        ] for gen in gens
    ]
    exp_data = list()
    for gen_index, gen in enumerate(gens):
        exp_data.append(dict)
        exp_data[gen_index] = dict()
        for metric in metrics:
            metric_name = metric.__name__
            exp_data[gen_index][metric_name] = dict()
            exp_data[gen_index][metric_name]['em_ests'] = [
                metric(gen, em_est, test_samples[gen_index])
                for em_est in em_ests[gen_index]
            ]
            exp_data[gen_index][metric_name]['sl_est'] = \
                metric(gen, sl_ests[gen_index], test_samples[gen_index])
        exp_data[gen_index]['gen_startprob'] = gen.startprob_.tolist()
        exp_data[gen_index]['gen_transmat'] = gen.transmat_.tolist()
        exp_data[gen_index]['gen_emissionprob'] = gen.emissionprob_.tolist()
        exp_data[gen_index]['em_est_time'] = em_est_fit_t.tolist()
        exp_data[gen_index]['sl_est_time'] = sl_est_fit_t.tolist()
    
    return exp_data

if __name__ == '__main__':
    parser = ArgumentParser(prog='Spectral Learning Experiment')
    parser.add_argument('-o', '--output')
    parser.add_argument('-c', '--config')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    exp_data = em_sl_comparison(
        gen_param=config['gen_param'],
        em_est_param=config['em_est_param'],
        sl_est_param=config['sl_est_param'],
        n_samples=config['n_samples'],
        max_t=30,
        metrics=[abs_prob_error, abs_norm_prob_error]
    )
    with open(args.output, 'w') as f:
        json.dump(exp_data, f, indent=2)