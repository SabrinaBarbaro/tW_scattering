'''
Simple processor using coffea.
[x] weights
[ ] Missing pieces: appropriate sample handling
[x] Accumulator caching

'''


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


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from Tools.helpers import *

# This just tells matplotlib not to open any
# interactive windows.
matplotlib.use('Agg')

class exampleProcessor(processor.ProcessorABC):
    """Dummy processor used to demonstrate the processor principle"""
    def __init__(self):

        # we can use a large number of bins and rebin later
        dataset_axis        = hist.Cat("dataset",   "Primary dataset")
        pt_axis             = hist.Bin("pt",        r"$p_{T}$ (GeV)", 600, 0, 1000)
        mass_axis           = hist.Bin("mass",      r"M (GeV)", 500, 0, 2000)
        eta_axis            = hist.Bin("eta",       r"$\eta$", 60, -5.5, 5.5)
        phi_axis            = hist.Bin("phi",       r"$\eta$", 60, -5.5, 5.5)
        multiplicity_axis   = hist.Bin("multiplicity",         r"N", 20, -0.5, 19.5)

        self._accumulator = processor.dict_accumulator({
            "MET_pt" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "pt_spec_max" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "MT" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "b_nonb_massmax" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "N_b" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_jet" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_spec" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            'cutflow_wjets':      processor.defaultdict_accumulator(int),
            'cutflow_ttbar':      processor.defaultdict_accumulator(int),
            'cutflow_TTW':      processor.defaultdict_accumulator(int),
            'cutflow_TTX':      processor.defaultdict_accumulator(int),
            'cutflow_signal':   processor.defaultdict_accumulator(int),

            'light_pair_mass':  hist.Hist("Counts", dataset_axis, mass_axis),
            #'lep_b_mass':       hist.Hist("Counts", dataset_axis, mass_axis),
            #'lep_light_mass':   hist.Hist("Counts", dataset_axis, mass_axis),
            'light_pair_pt':     hist.Hist("Counts", dataset_axis, pt_axis),
            'lep_b_pt':         hist.Hist("Counts", dataset_axis, pt_axis),
            'lep_light_pt':     hist.Hist("Counts", dataset_axis, pt_axis),

            'S_T':    hist.Hist("Counts", dataset_axis, pt_axis),
            'H_T':    hist.Hist("Counts", dataset_axis, pt_axis),

            'pt_b':   hist.Hist("Counts", dataset_axis, pt_axis),
            'eta_b':  hist.Hist("Counts", dataset_axis, eta_axis),
            'phi_b': hist.Hist("Counts", dataset_axis, phi_axis),

            'pt_lep':   hist.Hist("Counts", dataset_axis, pt_axis),
            'eta_lep':  hist.Hist("Counts", dataset_axis, eta_axis),
            'phi_lep': hist.Hist("Counts", dataset_axis, phi_axis),

            'pt_lead_light':  hist.Hist("Counts", dataset_axis, pt_axis),
            'eta_lead_light': hist.Hist("Counts", dataset_axis, eta_axis),
            'phi_lead_light': hist.Hist("Counts", dataset_axis, phi_axis),
            'ttbar':            processor.defaultdict_accumulator(int),
            'TTW':              processor.defaultdict_accumulator(int),
            'tW_scattering':    processor.defaultdict_accumulator(int),
            'TTX':              processor.defaultdict_accumulator(int),
            'wjets':            processor.defaultdict_accumulator(int),
            'diboson':          processor.defaultdict_accumulator(int),
            'totalEvents':      processor.defaultdict_accumulator(int),

        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        """
        Processing function. This is where the actual analysis happens.
        """
        output = self.accumulator.identity()
        dataset = df["dataset"]
        cfg = loadConfig()
        # We can access the data frame as usual
        # The dataset is written into the data frame
        # outside of this function

        output['totalEvents']['all'] += len(df['weight'])


        output['cutflow_wjets']['all events'] += sum(df['weight'][(df['dataset']=='wjets')].flatten())
        output['cutflow_ttbar']['all events'] += sum(df['weight'][(df['dataset']=='ttbar')].flatten())
        output['cutflow_TTW']['all events'] += sum(df['weight'][(df['dataset']=='TTW')].flatten())
        output['cutflow_TTX']['all events'] += sum(df['weight'][(df['dataset']=='TTX')].flatten())
        output['cutflow_signal']['all events'] += sum(df['weight'][(df['dataset']=='tW_scattering')].flatten())

        cutFlow = ((df['nLepton']==1) & (df['nVetoLepton']==1))

        output['cutflow_wjets']['singleLep']  += sum(df['weight'][(df['dataset']=='wjets')         & cutFlow].flatten())
        output['cutflow_ttbar']['singleLep']  += sum(df['weight'][(df['dataset']=='ttbar')         & cutFlow].flatten())
        output['cutflow_TTW']['singleLep']    += sum(df['weight'][(df['dataset']=='TTW')           & cutFlow].flatten())
        output['cutflow_TTX']['singleLep']    += sum(df['weight'][(df['dataset']=='TTX')           & cutFlow].flatten())
        output['cutflow_signal']['singleLep'] += sum(df['weight'][(df['dataset']=='tW_scattering') & cutFlow].flatten())

        cutFlow = ((df['nLepton']==1) & (df['nVetoLepton']==1) & (df['nGoodJet']>5))

        output['cutflow_wjets']['5jets']  += sum(df['weight'][(df['dataset']=='wjets')         & cutFlow].flatten())
        output['cutflow_ttbar']['5jets']  += sum(df['weight'][(df['dataset']=='ttbar')         & cutFlow].flatten())
        output['cutflow_TTW']['5jets']    += sum(df['weight'][(df['dataset']=='TTW')           & cutFlow].flatten())
        output['cutflow_TTX']['5jets']    += sum(df['weight'][(df['dataset']=='TTX')           & cutFlow].flatten())
        output['cutflow_signal']['5jets'] += sum(df['weight'][(df['dataset']=='tW_scattering') & cutFlow].flatten())

        cutFlow = ((df['nLepton']==1) & (df['nVetoLepton']==1) & (df['nGoodJet']>5) & (df['nGoodBTag']==2))

        output['cutflow_wjets']['btags']  += sum(df['weight'][(df['dataset']=='wjets')         & cutFlow].flatten())
        output['cutflow_ttbar']['btags']  += sum(df['weight'][(df['dataset']=='ttbar')         & cutFlow].flatten())
        output['cutflow_TTW']['btags']    += sum(df['weight'][(df['dataset']=='TTW')           & cutFlow].flatten())
        output['cutflow_TTX']['btags']    += sum(df['weight'][(df['dataset']=='TTX')           & cutFlow].flatten())
        output['cutflow_signal']['btags'] += sum(df['weight'][(df['dataset']=='tW_scattering') & cutFlow].flatten())

        # preselection of events
        selection = ((df['nLepton']==1) & (df['nVetoLepton']==1))
        #df = df[((df['nLepton']==1) & (df['nGoodJet']>5) & (df['nGoodBTag']==2))]

        # And fill the histograms
        output['MET_pt'].fill(dataset=dataset, pt=df["MET_pt"][selection].flatten(), weight=df['weight'][selection]*cfg['lumi'])
        output['MT'].fill(dataset=dataset, pt=df["MT"][selection].flatten(), weight=df['weight'][selection]*cfg['lumi'])
        output['N_b'].fill(dataset=dataset, multiplicity=df["nGoodBTag"][selection], weight=df['weight'][selection]*cfg['lumi'] )
        output['N_jet'].fill(dataset=dataset, multiplicity=df["nGoodJet"][selection], weight=df['weight'][selection]*cfg['lumi'] )

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

        Lepton = JaggedCandidateArray.candidatesfromcounts(
            df['nLepton'],
            pt = df['Lepton_pt'].content,
            eta = df['Lepton_eta'].content,
            phi = df['Lepton_phi'].content,
            mass = df['Lepton_mass'].content,
            pdgId = df['Lepton_pdgId'].content,
        )
        
        b = Jet[Jet['bjet']==1]
        nonb = Jet[(Jet['goodjet']==1) & (Jet['bjet']==0)]
        spectator = Jet[(abs(Jet.eta)>2.0) & (abs(Jet.eta)<4.7) & (Jet.pt>25) & (Jet['puId']>=7) & (Jet['jetId']>=6)] # 40 GeV seemed good. let's try going lower
        
        light_pair = nonb.choose(2)
        light_pair_selec = (Jet.counts>6)
        lep_light = Lepton.cross(nonb)
        lep_b = Lepton.cross(b)
        nonb_pair = nonb.choose(2)
        
        lep_b_selec = (Jet.counts>6) & (b.counts>=2) & (df['nLepton']==1) & (df['nVetoLepton']==1)
        lep_light_selec = (Jet.counts>6) & (df['nLepton']==1) & (df['nVetoLepton']==1)
        b_nonb_selection = (Jet.counts>5) & (b.counts>=2) & (nonb.counts>=4) & (df['nLepton']==1) & (df['nVetoLepton']==1)
        b_nonb_pair = b.cross(nonb)
        b_selec = (Jet.counts>6) & (b.counts>=2)

        lead_light = nonb[nonb.pt.argmax()]

        good_jet = Jet[Jet['goodjet'] ==1]
        good_jet_selec = (Jet.counts>6) 
        mega1_pt = Lepton.pt.sum() + good_jet.pt.sum()
        mega_pt = mega1_pt + df["MET_pt"]
        mega_selec = (Jet.counts>6) & (df['nLepton']==1) & (df['nVetoLepton']==1)
        
        singlelep = ((df['nLepton']==1) & (df['nVetoLepton']==1))
        fivejets = (Jet.counts > 5)
        sixjets = (Jet.counts > 6)
        sevenjets = (Jet.counts > 7)
        #eta_lead = ((lead_light.eta) > -1) & ((lead_light.eta) < 1)
        eta_lead =((lead_light.eta>-.5).counts>0) & ((lead_light.eta<0.5).counts>0)
        
        myProcesses = ['TTW','ttbar','wjets','tW_scattering']

        #Make CutFlow tables
        addRowToCutFlow(output, df, cfg, 'skim',      None, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'singlelep',     singlelep, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'fivejets',       singlelep & fivejets, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'sixjets',        singlelep & sixjets, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'sevenjets',      singlelep & sevenjets, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'b_selec',        singlelep & sevenjets & b_selec, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'eta_lead',     singlelep & sixjets & eta_lead, processes=myProcesses)


        #Fill out the histograms
        output['b_nonb_massmax'].fill(dataset=dataset, mass=b_nonb_pair[b_nonb_selection].mass.max().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['N_spec'].fill(dataset=dataset, multiplicity=spectator[b_nonb_selection].counts, weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['pt_spec_max'].fill(dataset=dataset, pt=spectator[b_nonb_selection & (spectator.counts>0)].pt.max().flatten(), weight=df['weight'][b_nonb_selection & (spectator.counts>0)]*cfg['lumi'])
        output['light_pair_mass'].fill(dataset=dataset, mass= light_pair[light_pair_selec].mass.flatten())
#        output['lep_b_mass'].fill(dataset=dataset, mass= lep_b[lep_b_selec].mass.flatten())
#        output['lep_light_mass'].fill(dataset=dataset, mass= lep_light.mass.flatten())
        output['light_pair_pt'].fill(dataset=dataset, pt= light_pair[light_pair_selec].pt.flatten())
        output['lep_b_pt'].fill(dataset=dataset, pt= lep_b[lep_b_selec].pt.flatten())
        output['lep_light_pt'].fill(dataset=dataset, pt= lep_light[lep_light_selec].pt.flatten())

        output['S_T'].fill(dataset=dataset, pt= mega_pt[mega_selec].flatten())
        output['H_T'].fill(dataset=dataset, pt= good_jet[good_jet_selec].pt.flatten())
        output['pt_b'].fill(dataset=dataset, pt= b[b_selec].pt.flatten())
        output['eta_b'].fill(dataset=dataset, eta= b[b_selec].eta.flatten())
        output['phi_b'].fill(dataset=dataset, phi= b[b_selec].phi.flatten())
        output['pt_lep'].fill(dataset=dataset, pt= Lepton[selection].pt.flatten())
        output['eta_lep'].fill(dataset=dataset, eta= Lepton[selection].eta.flatten())
        output['phi_lep'].fill(dataset=dataset, phi= Lepton[selection].phi.flatten())
        output['pt_lead_light'].fill(dataset=dataset, pt= lead_light[light_pair_selec].pt.flatten())
        output['eta_lead_light'].fill(dataset=dataset, eta= lead_light[light_pair_selec].eta.flatten())
        output['phi_lead_light'].fill(dataset=dataset, phi= lead_light[light_pair_selec].phi.flatten())

        return output

    def postprocess(self, accumulator):
        return accumulator


def main():

    overwrite = True

    # load the config and the cache
    cfg = loadConfig()

    # Inputs are defined in a dictionary
    # dataset : list of files
    from samples import fileset, fileset_small, fileset_1l

    # histograms
    histograms = ["MET_pt", "N_b", "N_jet", "MT", "b_nonb_massmax", "N_spec", "pt_spec_max", "light_pair_mass", "light_pair_pt", "lep_light_pt", "lep_b_pt","S_T", "H_T", "pt_b", "eta_b", "phi_b", "pt_lep", "eta_lep", "phi_lep", "pt_lead_light", "eta_lead_light", "phi_lead_light"] 


    # initialize cache
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['singleLep']), serialized=True)
    if not overwrite:
        cache.load()

    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and fileset == cache.get('fileset') and cache.get('simple_output'):
        output = cache.get('simple_output')

    else:
        # Run the processor
        output = processor.run_uproot_job(fileset_1l, #maybe the Event scale is for fileset_1l
                                      treename='Events',
                                      processor_instance=exampleProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers': 8, 'function_args': {'flatten': False}},
                                      chunksize=100000,
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

    return output

if __name__ == "__main__":
    output = main()

    df = getCutFlowTable(output, processes=['tW_scattering', 'TTW', 'ttbar', 'wjets'], lines=['skim', 'singlelep', 'fivejets','sixjets', 'eta_lead','sevenjets', 'b_selec'])


