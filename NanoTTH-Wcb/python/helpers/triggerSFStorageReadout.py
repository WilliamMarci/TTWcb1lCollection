import datetime
import json
import os.path

import hist
import correctionlib.convert
import correctionlib.schemav2 as cs
import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Literal


def store_scale_factor(bins: Dict[str, ArrayLike], values: ArrayLike,
                       channel: Literal['0L', '1L', '2L'],
                       year: Literal[2016, 2017, 2018],
                       mc_name: str, data_name: str,
                       outputfile: str) -> None:
    """
    store_scale_factor(bins, values, channel, year, mc_name, data_name, outputfile)

    Convert scale factors stored in histograms to format usable in analysis.

    This function takes input binning information and trigger scale factor values, and stores
    the resulting scale factor corrections in a correctionlib formatted JSON file under keys
    of the form "trigger_SF_[nom/down/up]"

    Parameters:
        bins (Dict[str, ArrayLike]): A dictionary mapping N variable names to arrays of numeric bin edges.
                                    Each variable should have a 1-dimensional array containing bin edges.
        values (ArrayLike): A (N+1)-dimensional array-like object containing trigger scale factor values.
                            The first axis represents nominal, lower, and upper bounds of the scale factors,
                            and the subsequent axes correspond to the variables defined in the 'bins' dictionary.
        channel (Literal['0L', '1L', '2L']): A string literal indicating the analysis channel ('0L', '1L', or '2L').
        year (Literal[2016, 2017, 2018]): A literal representing the data-taking year (2016, 2017, or 2018).
        mc_name (str): Name of the Monte Carlo (MC) dataset used to derive the scale factors.
        data_name (str): Name of the data dataset for which scale factors are derived.
        outputfile (str): Path to the output JSON file where the scale factor corrections will be stored. 
                          A suffix [CHANNEL]_[YEAR] will be auomatically appended.

    Raises:
        AssertionError: If the dimensionality of the 'values' array does not match the provided binning,
                       or if the length of the first axis of 'values' is not 3 (nominal, lower, upper).
                       If any variable in 'bins' is not properly defined as a 1-dimensional numeric array.
                       If the number of bins for a variable does not match the corresponding axis length in 'values'.

    Notes:
        - If the output file already exists, then the corrections will be added to it. Corrections with the same name
          as one of the new corrections will get a tag "archived on TODAY" added to their description.
        - Non-finite scale factor values are converted to 1
        - the extension .json is automatically appended to outputfile if not already present

    Example:
        bins = {'pt': [0, 20, 40, 60, 80, 100],
                'eta': [-2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5]}
        values = np.random.rand(3, len(bins['pt']) - 1, len(bins['eta']) - 1)
        store_scale_factor(bins, values, '0L', 2018, 'ttH_MC', 'data', 'scale_factors.json')
    """

    sf_configuration = dict(c=channel, y=year, mc=mc_name, dt=data_name)

    if outputfile.endswith('.json'):
        outputfile = outputfile[:-len('.json')]
    outputfile += '{c}_{y}.json'.format(**sf_configuration)

    values = np.array(values)
    bins = {x: np.array(y) for x, y in bins.items()}

    assert values.ndim == len(bins) + 1, "dimensionality of values array does not match the provided binning"
    assert values.shape[0] == 3, "first axis should be of length 3 with upper and lower bound on histogram"
    for k, v in enumerate(bins):
        assert bins[v].ndim == 1 and np.issubdtype(bins[v].dtype, np.number), \
            "each variable in 'bins' should be mapped to a 1d arraylike object containing numeric bin edges, " \
            "but {v} is mapped to {ndim}d array and dtype {dt}".format(v=v, ndim=bins[v].ndim, dt=bins[v].dtype)
        assert values.shape[k + 1] == len(bins[v]) - 1, \
            "number of bins (nb) for {v} does not match " \
            "length of corresponding axis in values (nv)".format(v=v, nb=len(bins[v]) - 1, nv=values.shape[k + 1])

    all_corrections = []

    histogram_axes = [hist.axis.Variable(bins[vr], name=vr) for vr in bins]
    scale_factor_histogram = hist.Hist(*histogram_axes)

    for k, variation in enumerate(['nom', 'down', 'up']):
        # create histogram understable by correctionlib
        valid_values = np.isfinite(values[k])
        scale_factor_histogram.view()[:] = np.where(valid_values, values[k], 1)

        # name of histogram is used by correctionlib as key
        name = "trigger_SF_{v}".format(v=variation)
        scale_factor_histogram.name = scale_factor_histogram.label = name

        # convert histogram with correctionlib and add proper description
        corr = correctionlib.convert.from_histogram(scale_factor_histogram)
        corr.description = "{c} trigger scale factors for {y} derived from {mc} MC and the {dt} dataset " \
                           "({v} variation)".format(**sf_configuration, v=variation)

        # I don't know what this means, correctionlib does not provide any explanation
        corr.data.flow = "clamp"
        corr.output.description = "event reweighting factor"
        all_corrections.append(corr)

    old = {}
    if os.path.exists(outputfile):
        with open(outputfile) as doc:
            old = json.load(doc)

        # archive pre-existing scale factors under same name as new one
        name_map = {c.name: c for c in all_corrections}
        for c in old['corrections']:
            if c['name'] in name_map:
                name_map[c['name']].version = max(c['version'] + 1, name_map[c['name']].version)
                if 'Archived on' not in c['description']:
                    c['description'] += ' (Archived on {time})'.format(time=datetime.datetime.today())

    final_correction_set = cs.CorrectionSet(
        schema_version=2,
        description="scale factors for ttH(cc) analysis, {c} channel, year {y}".format(**sf_configuration),
        corrections=all_corrections,
    )

    print(f'storing SFs to {outputfile}')
    with open(outputfile, "w") as fout:
        fout.write(final_correction_set.json(exclude_unset=True, indent=2))

    if old:
        with open(outputfile) as doc:
            new = json.load(doc)
        old['corrections'].extend(new['corrections'])

        # sort corrections such that newest appear last (correctionlib always takes last occurence)
        old['corrections'] = sorted(old['corrections'], key=lambda x: (x['version'], x['name']))
        with open(outputfile, 'w') as doc:
            json.dump(old, doc, indent=2)


def get_scale_factor(scalefactorfile: str) -> Dict[Literal['nom', 'down', 'up'], cs.Correction]:
    """
    get_scale_factor(scalefactorfile)

    Retrieve trigger scale factor corrections from a JSON file.

    This function reads a JSON file containing trigger scale factor corrections and extracts
    the latest nominal, down, and up corrections as correctionlib Correction objects.   

    Parameters:
        scalefactorfile (str): Path to the input JSON file containing the stored scale factor corrections.

    Returns:
        dict: A dictionary containing nominal, down and up trigger scale factors as correctionlib correction objects.

    Raises:
        KeyError: If the requested corrections are not found in the JSON file.

    Notes:
        - The function assumes that the scale factor corrections have been stored using the 'store_scale_factor' function.

    Example:
        scalefactorfile = 'scale_factors_1L_2017.json'
        corrections = get_scale_factor(scalefactorfile)
        print(corrections)  # Output: {'nom': <correctionlib.Correction object at ...>, ...}
    """

    if not scalefactorfile.endswith('.json'):
        scalefactorfile += '.json'

    all_corrections = correctionlib.CorrectionSet.from_file(scalefactorfile)
    corrector = {}
    for v in ['nom', 'down', 'up']:
        corrector[v] = all_corrections['trigger_SF_' + v]

    # newer versions of correctionlib have this inputs attribute which I used before. Add it manually if it is not there.
    if not hasattr(corrector['nom'], 'inputs'):
        with open(scalefactorfile) as doc:
            dict_fmt = json.load(doc)
            dict_fmt = {x['name']: x['inputs'] for x in dict_fmt['corrections']}

        class myvar:
            def __init__(self, name):
                self.name = name

        for v in corrector:
            corrector[v].inputs = [myvar(x['name']) for x in dict_fmt['trigger_SF_' + v]]

    return corrector


def evaluate_correction(corrector: Dict[Literal['nom', 'down', 'up'], cs.Correction], data: dict) -> Dict[str, float]:
    weights = {}
    for v in corrector:
        weights[v] = corrector[v].evaluate(*[float(data[x.name]) for x in corrector[v].inputs])
    return weights
