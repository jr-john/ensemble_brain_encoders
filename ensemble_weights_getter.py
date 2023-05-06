import numpy as np
import pandas as pd


# Linear weights
def linear_weights(accuracies):
    return accuracies / np.sum(accuracies)


# Normalized weights
def normalized_weights(accuracies):
    return accuracies / np.max(accuracies)


# Softmax weights
def softmax_weights(accuracies):
    exp_accuracies = np.exp(accuracies)
    return exp_accuracies / np.sum(exp_accuracies)


# Max weights
def max_weights(accuracies):
    max_index = np.argmax(accuracies)
    weights = np.zeros_like(accuracies)
    weights[max_index] = 1
    return weights


# Rank-based weights
def rank_weights(accuracies):
    ranks = len(accuracies) - np.argsort(np.argsort(accuracies))
    return ranks / np.sum(ranks)


# Square root weights
def sqrt_weights(accuracies):
    weights = np.sqrt(accuracies)
    return weights / np.sum(weights)


# Exponential weights
def exp_weights(accuracies, alpha=1):
    weights = np.exp(alpha * accuracies)
    return weights / np.sum(weights)


# Inverse weights
def inv_weights(accuracies):
    weights = 1 / accuracies
    return weights / np.sum(weights)


# power mean weights
def power_mean_weights(accuracies, p=1.1):
    weights = np.power(accuracies, p)
    weights = weights / np.sum(weights)
    return weights


# Calculate model weights based on their accuracies using different methods
def calculate_weights(accuracies, method="power_mean"):
    """
    Calculate model weights based on their accuracies using different methods.

    Parameters:
                    accuracies (array): An array of model accuracies.
                    method (str): The method used to calculate the weights.
                                    'linear' (default): Calculate linear weights.
                                    'normalized': Calculate normalized weights.
                                    'softmax': Calculate softmax weights.
                                    'max': Calculate max weights.
                                    'rank': Calculate rank-based weights.
                                    'sqrt': Calculate square root weights.
                                    'exp': Calculate exponential weights.
                                    'inv': Calculate inverse weights.
                                    'power_mean': Calculate power mean weights.

    Returns:
                    An array of model weights that add up to 1.

    """
    if method == "linear":
        weights = linear_weights(accuracies)
    elif method == "normalized":
        weights = normalized_weights(accuracies)
    elif method == "softmax":
        weights = softmax_weights(accuracies)
    elif method == "max":
        weights = max_weights(accuracies)
    elif method == "rank":
        weights = rank_weights(accuracies)
    elif method == "sqrt":
        weights = sqrt_weights(accuracies)
    elif method == "exp":
        weights = exp_weights(accuracies)
    elif method == "inv":
        weights = inv_weights(accuracies)
    elif method == "power_mean":
        weights = power_mean_weights(accuracies)
    else:
        raise ValueError("Invalid method: " + method)

    return weights


def get_weights(path_to_results="results", method="power_mean"):
    # read all the pickle files from results folder
    bert = pd.read_pickle(f"{path_to_results}/results_bert.pkl")
    coref = pd.read_pickle(f"{path_to_results}/results_coref.pkl")
    ensemble = pd.read_pickle(f"{path_to_results}/results_ensemble.pkl")
    ner = pd.read_pickle(f"{path_to_results}/results_ner.pkl")
    nli = pd.read_pickle(f"{path_to_results}/results_nli.pkl")
    paraphrase = pd.read_pickle(f"{path_to_results}/results_paraphrase.pkl")
    qa = pd.read_pickle(f"{path_to_results}/results_qa.pkl")
    sa = pd.read_pickle(f"{path_to_results}/results_sa.pkl")
    srl = pd.read_pickle(f"{path_to_results}/results_srl.pkl")
    ss = pd.read_pickle(f"{path_to_results}/results_ss.pkl")
    sums = pd.read_pickle(f"{path_to_results}/results_sum.pkl")
    wsd = pd.read_pickle(f"{path_to_results}/results_wsd.pkl")

    results = {
        "bert": bert,
        "coref": coref,
        "ner": ner,
        "nli": nli,
        "paraphrase": paraphrase,
        "qa": qa,
        "sa": sa,
        "srl": srl,
        "ss": ss,
        "sums": sums,
        "wsd": wsd,
        "ensemble": ensemble,
    }

    subjects = ["P01", "M02", "M04", "M07", "M15"]
    P01 = {}
    M02 = {}
    M04 = {}
    M07 = {}
    M15 = {}

    keys1 = list(results.keys())
    keys2 = list(results[keys1[0]].keys())
    keys3 = list(results[keys1[0]][keys2[0]].keys())

    for key1 in keys3:
        P01[key1] = {}
        M02[key1] = {}
        M04[key1] = {}
        M07[key1] = {}
        M15[key1] = {}
        for key2 in keys2:
            P01[key1][key2] = {}
            M02[key1][key2] = {}
            M04[key1][key2] = {}
            M07[key1][key2] = {}
            M15[key1][key2] = {}
            for key3 in keys1:
                M02[key1][key2][key3] = results[key3][key2][key1]["M02"]
                P01[key1][key2][key3] = results[key3][key2][key1]["P01"]
                M04[key1][key2][key3] = results[key3][key2][key1]["M04"]
                M07[key1][key2][key3] = results[key3][key2][key1]["M07"]
                M15[key1][key2][key3] = results[key3][key2][key1]["M15"]

    subjects = {
        "P01": P01,
        "M02": M02,
        "M04": M04,
        "M07": M07,
        "M15": M15,
    }

    ROIs = [
        "language_lh",
        "language_rh",
        "vision_body",
        "vision_face",
        "vision_object",
        "vision_scene",
        "vision",
        "dmn",
        "task",
    ]

    allcombined = {}
    for subject in subjects.keys():
        allcombined[subject] = {}
        for roi in ROIs:
            allcombined[subject][roi] = {}
    for subject in subjects.keys():
        for roi in ROIs:
            accuracies = [i for i in subjects[subject][roi]["pear"].values()]
            weights = calculate_weights(accuracies, method=method)
            allcombined[subject][roi] = weights
    allcombined["weightlist"] = [
        "bert",
        "coref",
        "ner",
        "nli",
        "paraphrase",
        "qa",
        "sa",
        "srl",
        "ss",
        "sums",
        "wsd",
        "ensemble",
    ]
    return allcombined
