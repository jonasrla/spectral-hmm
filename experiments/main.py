import cProfile

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

def em_sl_comparison(generator_parameters, em_estimator_parameters, sl_estimator_parameters, n_samples):
    gen = CategoricalHMM(
    n_components=generator_parameters['n_components'],
    n_features=generator_parameters['n_features']
    )
    if 'startprob' in generator_parameters:
        gen.startprob_ = generator_parameters['startprob']
        gen.transmat_ = generator_parameters['transmat']
        gen.emissionprob_ = generator_parameters['emissionprob']
    else:
        startprob, transmat, emissionprob = random_parameters(
            generator_parameters['n_components'],
            generator_parameters['n_features']
        )
        gen.startprob_ = startprob
        gen.transmat_ = transmat
        gen.emissionprob_ = emissionprob
    
    samples = [gen.sample(np.random.randint(2,30))[0] for i in range(n_samples)]

    em_est = CategoricalHMM(
        n_components=em_estimator_parameters['n_components'],
        n_features=em_estimator_parameters['n_features']
    )
    sl_est = SLHMM(
        num_hidden=sl_estimator_parameters['n_components'], num_observ=sl_estimator_parameters['n_features']
    )
    em_data = np.concatenate(samples)
    em_lenghts = list(map(len, samples))

    em_pr = cProfile.Profile()
    em_pr.bias = 2.37677932206823e-06
    em_pr.enable()
    em_est.fit(em_data, em_lenghts)
    em_pr.disable()

    sl_data = [s.T[0] for s in samples]
    sl_pr = cProfile.Profile()
    sl_pr.bias = 2.37677932206823e-06
    sl_pr.enable()
    sl_est.fit(sl_data)
    sl_pr.disable()
    return gen, em_est, sl_est


if __name__ == '__main__':
    gen, em_est, sl_est = em_sl_comparison(
        generator_parameters={
            'n_components': 3,
            'n_features': 5,
            'startprob': np.array([0.6,0.4,0.0]),
            'transmat': np.array(
                [
                    [0.7,0.3,0.0],
                    [0.3,0.5,0.2],
                    [0.5,0.4,0.1]
                ]
            ),
            'emissionprob': np.array(
                [
                    [0.2,0.3,0.0,0.0,0.5],
                    [0.5,0.0,0.0,0.3,0.2],
                    [0.5,0.0,0.3,0.0,0.2],
                ])

        },
        em_estimator_parameters={
            'n_components': 3,
            'n_features': 5
        },
        sl_estimator_parameters={
            'n_components': 3,
            'n_features': 5
        },
        n_samples=10000
    )