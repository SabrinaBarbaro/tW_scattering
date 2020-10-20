'''simple processor using coffea.
[x] weights
[ ] Missing pieces: appropriate sample handling
[x] Accumulator caching
'''


import os
import time
import glob
import re
import pandas as pd
from functools import reduce
from klepto.archives import dir_archive
import math

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

from Tools.helpers import loadConfig, getCutFlowTable, mergeArray

from Tools.objects import Collections
from Tools.cutflow import Cutflow

# This just tells matplotlib not to open any
# interactive windows.
matplotlib.use('Agg')

class exampleProcessor(processor.ProcessorABC):
    """Dummy processor used to demonstrate the processor principle"""
    def __init__(self):

        # we can use a large number of bins and rebin later
        dataset_axis        = hist.Cat("dataset",   "Primary dataset")
        pt_axis             = hist.Bin("pt",        r"$p_{T}$ (GeV)", 1000, 0, 1000)
        ht_axis             = hist.Bin("ht",        r"$H_{T}$ (GeV)", 500, 0, 5000)
        mass_axis           = hist.Bin("mass",      r"M (GeV)", 1000, 0, 2000)
        eta_axis            = hist.Bin("eta",       r"$\eta$", 60, -5.5, 5.5)
        multiplicity_axis   = hist.Bin("multiplicity",         r"N", 20, -0.5, 19.5)
        norm_axis            = hist.Bin("norm",         r"N", 25, 0, 1)
        ugh_axis   = hist.Bin("ugh",         r"mu+mu+ mu+e+ e+e+ mu-mu- mu-e- e-e-", 20, -0.5, 19.5)
 
        self._accumulator = processor.dict_accumulator({
            "MET_pt" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "pt_spec_max" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "MT" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "HT" :          hist.Hist("Counts", dataset_axis, ht_axis),
            "ST" :          hist.Hist("Counts", dataset_axis, ht_axis),
            "mbj_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mjj_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlb_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlb_min" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlj_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlj_min" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "N_b" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_ele" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_diele" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_mu" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_dimu" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_jet" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_spec" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "FWMT1" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "FWMT2" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "FWMT3" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "FWMT4" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "FWMT5" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "S" :               hist.Hist("Counts", dataset_axis, norm_axis),
            "S_lep" :           hist.Hist("Counts", dataset_axis, norm_axis),
            'diboson':          processor.defaultdict_accumulator(int),
            'ttbar':            processor.defaultdict_accumulator(int),
            'TTW':              processor.defaultdict_accumulator(int),
            'TTX':              processor.defaultdict_accumulator(int),
            'tW_scattering':    processor.defaultdict_accumulator(int),
            'totalEvents':      processor.defaultdict_accumulator(int),
            'DY':               processor.defaultdict_accumulator(int),
            'AttemptOne':       hist.Hist("Counts", dataset_axis, ugh_axis),
            'trailing_lep_pt':  hist.Hist("Counts", dataset_axis, pt_axis),
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

        jet = JaggedCandidateArray.candidatesfromcounts(
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
        jet     = jet[(jet['goodjet']==1)]
        btag    = jet[jet['bjet']==1]
        light   = jet[(jet['goodjet']==1) & (jet['bjet']==0)]
        lightCentral = jet[(jet['goodjet']==1) & (jet['bjet']==0) & (abs(jet.eta)<2.4) & (jet.pt>30)]
        fw        = light[abs(light.eta).argmax()] # the most forward light jet

        ## Leptons
        lepton = JaggedCandidateArray.candidatesfromcounts(
            df['nLepton'],
            pt = df['Lepton_pt'].content,
            eta = df['Lepton_eta'].content,
            phi = df['Lepton_phi'].content,
            mass = df['Lepton_mass'].content,
            pdgId = df['Lepton_pdgId'].content,
        )

        

        ## Muons
        muon = Collections(df, "Muon", "tight").get()
        vetomuon = Collections(df, "Muon", "veto").get()
        dimuon = muon.choose(2)
        SSmuon = ( dimuon[(dimuon.i0.charge * dimuon.i1.charge)>0].counts>0 )
        
        
        ## Electrons
        electron = Collections(df, "Electron", "tight").get()
        vetoelectron = Collections(df, "Electron", "veto").get()
        dielectron = electron.choose(2)
        SSelectron = ( dielectron[(dielectron.i0.charge * dielectron.i1.charge)>0].counts>0 )

        ## E/Mu cross
        dilepton = electron.cross(muon)
        SSdilepton = ( dilepton[(dilepton.i0.charge * dilepton.i1.charge)>0].counts>0 )
        
        ## how to get leading lepton easily? Do I actually care?
        leading_muon = muon[muon.pt.argmax()]
        leading_electron = electron[electron.pt.argmax()]
        
                      
        ## MET
        met_pt  = df["MET_pt"]
        met_phi = df["MET_phi"]

        ## other variables
        st = df["MET_pt"] + jet.pt.sum() + muon.pt.sum() + electron.pt.sum()
        ht = jet.pt.sum()
        
        ## Event classifieres
        # We want to get the deltaEta between the most forward jet fw and the jet giving the largest invariant mass with fw, fw2
        jj = fw.cross(light)
        deltaEta = abs(fw.eta - jj[jj.mass.argmax()].i1.eta)
        deltaEtaJJMin = ((deltaEta>2).any())

        ## this is my personal forward jet definition, can be used later
        spectator = jet[(abs(jet.eta)>2.0) & (abs(jet.eta)<4.7) & (jet.pt>25) & (jet['puId']>=7) & (jet['jetId']>=6)] # 40 GeV seemed good. let's try going lower
        leading_spectator = spectator[spectator.pt.argmax()]        

        neg_SS_dimuon = ( dimuon[(dimuon.i0.charge<0 * dimuon.i1.charge)>0].counts>0 ) 
        neg_SS_diele = ( dielectron[(dielectron.i0.charge<0 * dielectron.i1.charge)>0].counts>0 )
        neg_SS_dilep = ( dilepton[(dilepton.i0.charge<0 * dilepton.i1.charge)>0].counts>0 )
        pos_SS_dimuon = ( dimuon[(dimuon.i0.charge>0 * dimuon.i1.charge)>0].counts>0 )
        pos_SS_diele = ( dielectron[(dielectron.i0.charge>0 * dielectron.i1.charge)>0].counts>0 )
        pos_SS_dilep = ( dilepton[(dilepton.i0.charge>0 * dilepton.i1.charge)>0].counts>0 )

        ## updated
        dilep      = ((electron.counts + muon.counts)==2)
        lepveto    = ((vetoelectron.counts + vetomuon.counts)==2)
        SS         = (SSelectron | SSmuon | SSdilepton)
        Mll_veto   = (~(dimuon.mass<125).any() & ~(dielectron.mass<125).any() & ~(dilepton.mass<125).any() )
        Mll_veto2  = (~(abs(dimuon.mass-91.2)<15).any() & ~(abs(dielectron.mass-91.2)<15).any() ) # not so strict Z veto
        nbtag      = (btag.counts>0)
        met        = (met_pt > 50)
        nfwd       = (spectator.counts>0)  
        fwdJet50   = ((leading_spectator.pt>50).any())
        ptl100     = (((leading_muon.pt>100).any()) | ((leading_electron.pt>100).any()))
        eta_fwd    = ((abs(light.eta)>1.75).any())
        
                
        
        ## work on the cutflow
        output['totalEvents']['all'] += len(df['weight'])

        processes = ['tW_scattering', 'TTW', 'TTX', 'diboson', 'ttbar', 'DY']
        cutflow = Cutflow(output, df, cfg, processes)
        
        cutflow.addRow( 'dilep',       dilep )
        cutflow.addRow( 'lepveto',     lepveto )
        cutflow.addRow( 'SS',          SS )
        cutflow.addRow( 'njet4',       (jet.counts>=4) )
        cutflow.addRow( 'njet5',       (jet.counts>=5) )
        cutflow.addRow( 'central3',    (lightCentral.counts>=3) ) 
        #cutflow.addRow( 'central4',    (lightCentral.counts>=4) ) # not very efficient for signal. lets look at the plot
        cutflow.addRow( 'nbtag',       nbtag )
        #cutflow.addRow( 'lep100',      ptl100 )
        cutflow.addRow( 'Mll',         Mll_veto2 ) # switched to not so strict Z veto
        cutflow.addRow( 'MET>50',      met )



        # pre selection of events
        event_selection = cutflow.selection
        
        ## And fill the histograms
        # just the number of electrons and muons
        output['N_ele'].fill(dataset=dataset, multiplicity=electron[event_selection].counts, weight=df['weight'][event_selection]*cfg['lumi'])
        output['N_mu'].fill(dataset=dataset, multiplicity=muon[event_selection].counts, weight=df['weight'][event_selection]*cfg['lumi'])

        # something a bit more tricky
        output['N_diele'].fill(dataset=dataset, multiplicity=dielectron[event_selection].counts, weight=df['weight'][event_selection]*cfg['lumi'])
        output['N_dimu'].fill(dataset=dataset, multiplicity=dimuon[event_selection].counts, weight=df['weight'][event_selection]*cfg['lumi'])
        
        #plots of spectator quark
        output['N_spec'].fill(dataset=dataset, multiplicity=spectator[event_selection].counts, weight=df['weight'][event_selection]*cfg['lumi'])
        output['pt_spec_max'].fill(dataset=dataset, pt=spectator[event_selection].pt.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        
        # MET_pt, MT, N_b, N_jets, HT, ST
        
        output['MET_pt'].fill(dataset=dataset, pt=df["MET_pt"][event_selection].flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['MT'].fill(dataset=dataset, pt=df["MT"][event_selection].flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['N_b'].fill(dataset=dataset, multiplicity=df["nGoodBTag"][event_selection], weight=df['weight'][event_selection]*cfg['lumi'] )
        output['N_jet'].fill(dataset=dataset, multiplicity=df["nGoodJet"][event_selection], weight=df['weight'][event_selection]*cfg['lumi'] )
        
        st = jet[jet['goodjet']==1].pt.sum() + lepton.pt.sum() + df['MET_pt']
        output['ST'].fill(dataset=dataset, ht=st[event_selection].flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['HT'].fill(dataset=dataset, ht=df['Jet_pt'][event_selection].sum().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        
        #mbj_max, mjj_max, mlb_max, mlb_min, mlj_max, mlj_min
        b_nonb_pair = btag.cross(light)
        jet_pair = light.choose(2)
        output['mbj_max'].fill(dataset=dataset, mass=b_nonb_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mjj_max'].fill(dataset=dataset, mass=jet_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        lepton_bjet_pair = lepton.cross(btag)
        output['mlb_max'].fill(dataset=dataset, mass=lepton_bjet_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlb_min'].fill(dataset=dataset, mass=lepton_bjet_pair[event_selection].mass.min().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        lepton_jet_pair = lepton.cross(jet)
        output['mlj_max'].fill(dataset=dataset, mass=lepton_jet_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlj_min'].fill(dataset=dataset, mass=lepton_jet_pair[event_selection].mass.min().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])

        output['AttemptOne'].fill(dataset=dataset, ugh=(pos_SS_dimuon[event_selection]), weight=df['weight'][event_selection]*cfg['lumi'])
        output['AttemptOne'].fill(dataset=dataset, ugh=2*(pos_SS_dilep[event_selection]), weight=df['weight'][event_selection]*cfg['lumi'])
        output['AttemptOne'].fill(dataset=dataset, ugh=3*(pos_SS_diele[event_selection]), weight=df['weight'][event_selection]*cfg['lumi'])
        output['AttemptOne'].fill(dataset=dataset, ugh=4*(neg_SS_dimuon[event_selection]), weight=df['weight'][event_selection]*cfg['lumi'])
        output['AttemptOne'].fill(dataset=dataset, ugh=5*(neg_SS_dilep[event_selection]), weight=df['weight'][event_selection]*cfg['lumi'])
        output['AttemptOne'].fill(dataset=dataset, ugh=6*(neg_SS_diele[event_selection]), weight=df['weight'][event_selection]*cfg['lumi'])

        output['trailing_lep_pt'].fill(dataset=dataset, pt=lepton[event_selection].pt.min().flatten(), weight=df['weight'][event_selection]*cfg['lumi']) 

        return output

    def postprocess(self, accumulator):
        return accumulator


def main():

    overwrite = True

    # load the config and the cache
    cfg = loadConfig()

    # Inputs are defined in a dictionary
    # dataset : list of files
    from processor.samples import fileset, fileset_small, fileset_2l

    # histograms
    histograms = []
    histograms += ['N_ele', 'N_mu', 'N_diele', 'N_dimu', 'MET_pt', 'pt_spec_max', 'MT', 'HT', 'ST', 'mbj_max', 'mjj_max', 'mlb_max', 'mlb_min', 'mlj_max', 'mlj_min', 'N_b', 'N_jet', 'N_spec']
    histograms += ['AttemptOne', 'trailing_lep_pt']    
    # initialize cache
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['singleLep']), serialized=True)
    if not overwrite:
        cache.load()

    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and fileset == cache.get('fileset') and cache.get('simple_output'):
        output = cache.get('simple_output')

    else:
        # Run the processor
        output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=exampleProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers': 18, 'function_args': {'flatten': False}},
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

df = getCutFlowTable(output, processes=['tW_scattering', 'ttbar', 'diboson', 'TTW', 'TTX', 'DY'], lines=['dilep','lepveto','SS','njet4','njet5','central3','nbtag','Mll','MET>50'])

#print percentage table
percentoutput = {}
for process in ['tW_scattering', 'ttbar', 'diboson', 'TTW', 'TTX', 'DY']:
    percentoutput[process] = {'dilep':0,'lepveto':0,'SS':0,'njet4':0,'njet5':0,'central3':0,'nbtag':0,'Mll':0,'MET>50':0}
    lastnum = output[process]['skim']
    for select in ['dilep','lepveto','SS','njet4','njet5','central3','nbtag','Mll','MET>50']:
        thisnum = output[process][select]
        thiser = output[process][select+'_w2']
        if lastnum==0:
            percent=0
            err=0
        else:
            percent = thisnum/lastnum
            err = math.sqrt(thiser)/lastnum
        percentoutput[process][select] = "%s +/- %s"%(round(percent,2), round(err, 2))
        lastnum = thisnum
df_p = pd.DataFrame(data=percentoutput)
df_p = df_p.reindex(['dilep','lepveto','SS','njet4','njet5','central3','nbtag','Mll','MET>50'])
print(df)
print(df_p)
