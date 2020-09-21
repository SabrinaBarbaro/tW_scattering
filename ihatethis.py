from coffea import hist, processor
import os
import time
import glob
import re
from functools import reduce
from klepto.archives import dir_archive

import numpy as np
from tqdm.auto import tqdm
import coffea.processor as processor
from coffea.processor.accumulator import AccumulatorABC
from coffea.analysis_objects import JaggedCandidateArray
from coffea import hist
import pandas as pd
import uproot_methods
import awkward

from memory_profiler import profile

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from Tools.helpers import *
matplotlib.use('Agg')


class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        b_pt_axis = hist.Bin("b_pt", r"$p_{T}$ (GeV)", 1000, 0, 1000) 
        self._accumulator = processor.dict_accumulator({
            "b_pt" : hist.Hist("Counts", dataset_axis,b_pt_axis)
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
        dataset = sf["dataset"]
        cfg = loadConfig()


        Jet = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt = df['Jet_pt'].content,
            eta = df['Jet_eta'].content,
            phi = df['Jet_phi'].content,
            mass = df['Jet_mass'].content,
            goodjet = df['Jet_isGoodJetAll'].content,
            bjet = df['Jet_isGoodBJet'].content,
            jetId = df['Jet_jetId'].content,
            puId = df['Jet_puId'].content,
        )

        b_jets = Jet[Jet['bjet']>=1]
        b_pt_values = b_jets.pt

        output['b_pt'].fill(dataset=dataset,b_pt=b_pt_values)        # ...

        return output

    def postprocess(self, accumulator):
        return accumulator
def main():

    overwrite = True
    small = True
    cfg = loadConfig()
    cacheName = 'singleLep_small' if small else 'singleLep'
    from samples import fileset, fileset_small, fileset_1l
    histograms = ["b_pt"] 

    ache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches'][cacheName]), serialized=True)
    if not overwrite:
        cache.load()

    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and cache.get('simple_output'):
        output = cache.get('simple_output')

    else:
        # Run the processor
        if small:
            fileset = fileset_small
            workers = 1
        else:
            fileset = fileset_1l
            workers = 6
        output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=exampleProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers': workers, 'function_args': {'flatten': False}},
                                      chunksize=50000,
                                     )
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()

    # Make a few plots
    outdir = "./tmp_plots"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for name in histograms:
        print (name)
        histogram = output[name]

        ax = hist.plot1d(histogram,overlay="dataset", stack=True) # make density plots because we don't care about x-sec differences
        ax.set_yscale('linear')
        ax.figure.savefig(os.path.join(outdir, "{}.pdf".format(name)))
        ax.clear()

    return output

if __name__ == "__main__":
    output = main()

p = MyProcessor()

