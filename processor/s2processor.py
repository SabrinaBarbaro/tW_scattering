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
        ht_axis             = hist.Bin("ht",        r"$H_{T}$ (GeV)", 500, 0, 5000)
        pt_axis             = hist.Bin("pt",        r"$p_{T}$ (GeV)", 600, 0, 1000)
        mass_axis           = hist.Bin("mass",      r"M (GeV)", 500, 0, 2000)
        eta_axis            = hist.Bin("eta",       r"$\eta$", 60, -5.5, 5.5)
        phi_axis            = hist.Bin("phi",       r"$\eta$", 60, -5.5, 5.5)
        multiplicity_axis   = hist.Bin("multiplicity",         r"N", 20, -0.5, 19.5)
#        lep_array = ['die', 'dimu', 'mue']
 #       lep_axis = hist.Bin("lep", "lep flavour?", 3, lep_array)

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

            "mbj_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mjj_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlb_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlb_min" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlj_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlj_min" :          hist.Hist("Counts", dataset_axis, mass_axis),
            #"muoeee":            hist.Hist("Counts", dataset_axis, lep_axis),

            'b_debug' :          hist.Hist("Counts", dataset_axis, multiplicity_axis),

            'light_pair_mass':  hist.Hist("Counts", dataset_axis, mass_axis),
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
            'mt_lep_MET': hist.Hist("Counts", dataset_axis, mass_axis),

            'HT': hist.Hist("Counts", dataset_axis, ht_axis),
            'ST': hist.Hist("Counts", dataset_axis, ht_axis),

            'central3':  hist.Hist("Counts", dataset_axis, eta_axis),
            'deltaEtaJJMin':  hist.Hist("Counts", dataset_axis, eta_axis),
            "lep_mass" :   hist.Hist("Counts", dataset_axis, mass_axis),
            'deltaEtalj':  hist.Hist("Counts", dataset_axis, eta_axis),
            'R':          hist.Hist("Counts", dataset_axis, multiplicity_axis), 

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

        met_pt = df["MET_pt"]
        met_phi = df["MET_phi"]
        bj_pair = b.cross(nonb)
        jj_pair = nonb.cross(nonb)
        lb_pair = Lepton.cross(b)
        lj_pair = Lepton.cross(nonb)
        ht = Jet[Jet['goodjet']==1].pt.sum()
        st = Jet[Jet['goodjet']==1].pt.sum() + Lepton.pt.sum() + met_pt    
        
        R= ((Lepton.eta.argmax()-Jet.eta.argmax())**2 + (Lepton.phi.argmax()-Jet.phi.argmax()**2))**0.5  #USE ARGMAX INSTEAD
        
        lead_light = nonb[nonb.pt.argmax()]
        b_debug_selec = ((Jet.counts>7) & (df['nLepton']==1) & (df['nVetoLepton']==1))

        good_jet = Jet[Jet['goodjet'] ==1]
        good_jet_selec = (Jet.counts>6) 
        mega1_pt = Lepton.pt.sum() + good_jet.pt.sum()
        mega_pt = mega1_pt + df["MET_pt"]
        mega_selec = (Jet.counts>6) & (df['nLepton']==1) & (df['nVetoLepton']==1)

        leading_lepton = Lepton[Lepton.pt.argmax()]
        mt_lep_met = mt(leading_lepton.pt, leading_lepton.phi, met_pt, met_phi)
        lightCentral = Jet[(Jet['goodjet']==1) & (Jet['bjet']==0) & (abs(Jet.eta)<2.4) & (Jet.pt>30)]
        fw = nonb[abs(nonb.eta).argmax()]
        jj = fw.cross(nonb)
        deltaEta = abs(fw.eta - jj[jj.mass.argmax()].i1.eta)
        deltaEtaJJMin = ((deltaEta>2).any())

        deltalj = Lepton.eta.sum() - Jet.eta.sum()

        
        b_selection = df["nGoodBTag"]>=1
        singlelep = ((df['nLepton']==1) & (df['nVetoLepton']==1))
        dilep =  ((df['nLepton']==2))
        sevenjets = (Jet.counts > 7)
        eta_lead =((lead_light.eta>-.5).counts>0) & ((lead_light.eta<0.5).counts>0)
        MET_cut = df["MET_pt"] >= 40
        MT_cut = df["MT"] >= 30
        HT_cut = (ht>500)
        ST_cut= (st>600)

        grand_selec = singlelep & b_selection & sevenjets & MET_cut & MT_cut & HT_cut & ST_cut
    

        event_selection = (Jet.counts>5) & (b.counts>=2) & (nonb.counts>=4) & (df['nLepton']==1) & (df['nVetoLepton']==1)
        tight_selection = (Jet.counts>5) & (b.counts>=2) & (nonb.counts>=4) & (df['nLepton']==1) & (df['nVetoLepton']==1) & (df['MET_pt']>50) & (ht>500) & (df['MT']>50) & (spectator.counts>=1) & (spectator.pt.max()>50) & (st>600) & (bj_pair.mass.max()>300) & (jj_pair.mass.max()>300)

        ### work on the cutflow
        myProcesses = ['TTW','TTX','ttbar','wjets','tW_scattering']

        #Make CutFlow tables
        addRowToCutFlow(output, df, cfg, 'skim',      None, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'singlelep',     singlelep, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'sevenjets',     singlelep & sevenjets, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'b_selection',    singlelep & sevenjets & b_selection, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'MET_cut',        singlelep & sevenjets & b_selection & MET_cut, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'MT_cut',        singlelep & sevenjets & b_selection & MET_cut & MT_cut, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'HT_cut',         singlelep & sevenjets & b_selection & MET_cut & MT_cut & HT_cut, processes=myProcesses)
        addRowToCutFlow(output, df, cfg, 'ST_cut',        singlelep & sevenjets & b_selection & MET_cut & MT_cut & HT_cut & ST_cut, processes=myProcesses)
        
        ### fill all the histograms
        
        output['MET_pt'].fill(dataset=dataset, pt=df["MET_pt"][singlelep].flatten(), weight=df['weight'][singlelep]*cfg['lumi'])
        output['MT'].fill(dataset=dataset, pt=df["MT"][singlelep].flatten(), weight=df['weight'][singlelep]*cfg['lumi'])
        output['N_b'].fill(dataset=dataset, multiplicity=df["nGoodBTag"][singlelep], weight=df['weight'][singlelep]*cfg['lumi'] )
        output['N_jet'].fill(dataset=dataset, multiplicity=df["nGoodJet"][singlelep], weight=df['weight'][singlelep]*cfg['lumi'] )
        
        output['mbj_max'].fill(dataset=dataset, mass=bj_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mjj_max'].fill(dataset=dataset, mass=jj_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlb_min'].fill(dataset=dataset, mass=lb_pair[event_selection].mass.min().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlb_max'].fill(dataset=dataset, mass=lb_pair[grand_selec].mass.max().flatten(), weight=df['weight'][grand_selec]*cfg['lumi'])
        output['mlj_min'].fill(dataset=dataset, mass=lj_pair[event_selection].mass.min().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlj_max'].fill(dataset=dataset, mass=lj_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])

        output['HT'].fill(dataset=dataset, ht=ht[grand_selec].flatten(), weight=df['weight'][grand_selec]*cfg['lumi'])
        output['ST'].fill(dataset=dataset, ht=st[event_selection].flatten(), weight=df['weight'][event_selection]*cfg['lumi'])

        output['b_debug'].fill(dataset=dataset, multiplicity = b[b_debug_selec].counts.flatten(), weight=df['weight'][b_debug_selec]*cfg['lumi'])

        # forward stuff
        output['N_spec'].fill(dataset=dataset, multiplicity=spectator[event_selection].counts, weight=df['weight'][event_selection]*cfg['lumi'])
        output['pt_spec_max'].fill(dataset=dataset, pt=spectator[grand_selec & (spectator.counts>0)].pt.max().flatten(), weight=df['weight'][grand_selec & (spectator.counts>0)]*cfg['lumi'])

        output['light_pair_mass'].fill(dataset=dataset, mass= light_pair[light_pair_selec].mass.flatten())
#        output['lep_b_mass'].fill(dataset=dataset, mass= lep_b[lep_b_selec].mass.flatten())
#        output['lep_light_mass'].fill(dataset=dataset, mass= lep_light.mass.flatten())
        output['light_pair_pt'].fill(dataset=dataset, pt= light_pair[grand_selec].pt.flatten())
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
        output['pt_lead_light'].fill(dataset=dataset, pt= lead_light[grand_selec].pt.flatten())
        output['eta_lead_light'].fill(dataset=dataset, eta= lead_light[grand_selec].eta.flatten())
        output['mt_lep_MET'].fill(dataset=dataset, mass = mt_lep_met[b_selec].flatten())
        output['central3'].fill(dataset=dataset, eta= lightCentral.eta.flatten())
        output['deltaEtaJJMin'].fill(dataset=dataset, eta= deltaEtaJJMin.flatten())
        output['deltaEtalj'].fill(dataset=dataset, eta= deltalj.flatten())
        output['R'].fill(dataset=dataset, multiplicity = R[dilep].flatten(), weight=df['weight'][dilep]*cfg['lumi'])
        output['lep_mass'].fill(dataset=dataset, mass= Lepton[selection].mass.flatten())

        ### event shape variables - neglect for now
        
        #output['FWMT1'].fill(dataset=dataset, norm=FWMT(leading_nonb)[1][tight_selection], weight=df['weight'][tight_selection]*cfg['lumi'])
        #output['FWMT2'].fill(dataset=dataset, norm=FWMT(leading_nonb)[2][tight_selection], weight=df['weight'][tight_selection]*cfg['lumi'])
        #output['FWMT3'].fill(dataset=dataset, norm=FWMT(leading_nonb)[3][tight_selection], weight=df['weight'][tight_selection]*cfg['lumi'])
        #output['FWMT4'].fill(dataset=dataset, norm=FWMT(leading_nonb)[4][tight_selection], weight=df['weight'][tight_selection]*cfg['lumi'])
        #output['FWMT5'].fill(dataset=dataset, norm=FWMT(leading_nonb)[5][tight_selection], weight=df['weight'][tight_selection]*cfg['lumi'])
        ##output['S'].fill(dataset=dataset, norm=sphericityBasic(alljet)[event_selection], weight=df['weight'][event_selection]*cfg['lumi'])

        #all_obj = mergeArray(alljet, lepton)
        #output['S_lep'].fill(dataset=dataset, norm=sphericityBasic(all_obj)[event_selection], weight=df['weight'][event_selection]*cfg['lumi'])
        
        
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
    histograms = ["MET_pt", "N_b", "N_jet", "MT", "b_nonb_massmax", "N_spec", "pt_spec_max", "light_pair_mass", "light_pair_pt", "lep_light_pt", "lep_b_pt","S_T", "H_T", "pt_b", "eta_b", "phi_b", "pt_lep", "eta_lep", "phi_lep", "pt_lead_light", "eta_lead_light", "phi_lead_light", "mt_lep_MET", "mbj_max", "mjj_max", "mlb_min", "mlb_max", "mlj_min", "mlj_max", "b_debug", "HT", "ST", "central3", "deltaEtaJJMin", "lep_mass","R", "deltaEtalj"] 


    # initialize cache
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['singleLep']), serialized=True)
    if not overwrite:
        cache.load()

    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and fileset == cache.get('fileset') and cache.get('simple_output'):
        output = cache.get('simple_output')

    else:
        # Run the processor
        output = processor.run_uproot_job(fileset_small, #maybe the Event scale is for fileset_1l
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

    df = getCutFlowTable(output, processes=['tW_scattering','TTX', 'TTW', 'ttbar', 'wjets'], lines=['skim', 'singlelep', 'sevenjets', 'b_selection', 'MET_cut', 'MT_cut', 'HT_cut', 'ST_cut'])
  

percentoutput = {}
for process in ['tW_scattering', 'TTX', 'TTW', 'ttbar', 'wjets']:
    percentoutput[process] = {'skim':0,'singlelep':0, 'sevenjets':0 , 'b_selection':0, 'MET_cut':0, 'MT_cut':0, 'HT_cut':0,'ST_cut':0}
    lastnum = output[process]['skim']
    for select in ['skim', 'singlelep', 'sevenjets', 'b_selection', 'MET_cut', 'MT_cut', 'HT_cut', 'ST_cut']:
        thisnum = output[process][select]
        percent = thisnum/lastnum
        percentoutput[process][select] = percent
        lastnum = thisnum
df_p = pd.DataFrame(data=percentoutput)
df_p = df_p.reindex(['skim', 'singlelep', 'sevenjets', 'b_selection', 'MET_cut', 'MT_cut', 'HT_cut','ST_cut'])
print(df_p) 
