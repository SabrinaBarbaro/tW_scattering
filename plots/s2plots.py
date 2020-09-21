'''
small script that reades histograms from an archive and saves figures in a public space

ToDo:
[x] Cosmetics (labels etc)
[x] ratio pad!
  [x] pseudo data
    [ ] -> move to processor to avoid drawing toys every time!
[x] uncertainty band
[ ] fix shapes
'''


from coffea import hist
import pandas as pd
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from klepto.archives import dir_archive

# import all the colors and tools for plotting
from Tools.helpers import loadConfig
from helpers import *

# load the configuration
cfg = loadConfig()

# load the results
cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['singleLep']), serialized=True)
cache.load()

histograms = cache.get('histograms')
output = cache.get('simple_output')
plotDir = os.path.expandvars(cfg['meta']['plots']) + '/plots1l/'
finalizePlotDir(plotDir)

if not histograms:
    print ("Couldn't find histograms in archive. Quitting.")
    exit()

print ("Plots will appear here:", plotDir )

bins = {\
    'MET_pt':   {'axis': 'pt',            'overflow':'over',  'bins': hist.Bin('pt', r'$E_T^{miss} \ (GeV)$', 20, 0, 200)},
    'MT':       {'axis': 'pt',            'overflow':'over',  'bins': hist.Bin('pt', r'$M_T \ (GeV)$', 20, 0, 200)},
    'N_jet':    {'axis': 'multiplicity',  'overflow':'over',  'bins': hist.Bin('multiplicity', r'$N_{jet}$', 15, -0.5, 14.5)},
    'N_spec':   {'axis': 'multiplicity',  'overflow':'over',  'bins': hist.Bin('multiplicity', r'$N_{jet, fwd}$', 6, -0.5, 5.5)},
    'N_b':      {'axis': 'multiplicity',  'overflow':'over',  'bins': hist.Bin('multiplicity', r'$N_{b-jet}$', 5, -0.5, 4.5)},
    'pt_spec_max': {'axis': 'pt',         'overflow':'over',  'bins': hist.Bin('pt', r'$p_{T, fwd jet}\ (GeV)$', 20, 0, 400)},
    'mbj_max':  {'axis': 'mass',          'overflow':'over',  'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 25, 0, 1500)},
    'mjj_max':  {'axis': 'mass',          'overflow':'over',  'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 25, 0, 1500)},
    'mlb_min':  {'axis': 'mass',          'overflow':'over',  'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 15, 0, 300)},
    'mlb_max':  {'axis': 'mass',          'overflow':'over',  'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 25, 0, 500)},
    'mlj_min':  {'axis': 'mass',          'overflow':'over',  'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 15, 0, 300)},
    'mlj_max':  {'axis': 'mass',          'overflow':'over',  'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 25, 0, 1000)},
    'HT':       {'axis': 'ht',            'overflow':'over',  'bins': hist.Bin('ht', r'$M(b, light) \ (GeV)$', 30, 0, 1500)},
    'ST':       {'axis': 'ht',            'overflow':'over',  'bins': hist.Bin('ht', r'$M(b, light) \ (GeV)$', 30, 0, 1500)},
    'mt_lep_MET': {'axis': 'mass',          'overflow':'over',  'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 15, 0, 300)},
    'FWMT1':    {'axis': 'norm',          'overflow':'none',  'bins': hist.Bin('norm', r'FWMT1', 25, 0, 1)},
    'FWMT2':    {'axis': 'norm',          'overflow':'none',  'bins': hist.Bin('norm', r'FWMT2', 20, 0, 0.8)},
    'FWMT3':    {'axis': 'norm',          'overflow':'none',  'bins': hist.Bin('norm', r'FWMT3', 25, 0, 1)},
    'FWMT4':    {'axis': 'norm',          'overflow':'none',  'bins': hist.Bin('norm', r'FWMT4', 25, 0, 1)},
    'FWMT5':    {'axis': 'norm',          'overflow':'none',  'bins': hist.Bin('norm', r'FWMT5', 25, 0, 1)},
    #'S':        {'axis': 'norm',            'bins': hist.Bin('norm', r'sphericity', 25, 0, 1)},
    #'S_lep':    {'axis': 'norm',            'bins': hist.Bin('norm', r'sphericity', 25, 0, 1)},
    }

for name in histograms:
    print (name)
    skip = False
    histogram = output[name]
    if name == 'MET_pt':
        # rebin
        axis = 'pt'
        new_met_bins = hist.Bin('pt', r'$E_T^{miss} \ (GeV)$', 20, 0, 200)
        histogram = histogram.rebin('pt', new_met_bins)
#    elif name == 'MT':
#        # rebin
#        axis = 'pt'
#        new_met_bins = hist.Bin('pt', r'$M_T \ (GeV)$', 20, 0, 200)
#        histogram = histogram.rebin('pt', new_met_bins)
#    elif name == 'N_jet':
#        # rebin
#        axis = 'multiplicity'
#        new_n_bins = hist.Bin('multiplicity', r'$N_{jet}$', 15, -0.5, 14.5)
#        histogram = histogram.rebin('multiplicity', new_n_bins)
#    elif name == 'N_spec':
#        # rebin
#        axis = 'multiplicity'
#        new_n_bins = hist.Bin('multiplicity', r'$N_{jet, fwd}$', 15, -0.5, 14.5)
#        histogram = histogram.rebin('multiplicity', new_n_bins)
#    elif name == 'pt_spec_max':
#        # rebin
#        axis = 'pt'
#        new_pt_bins = hist.Bin('pt', r'$p_{T, fwd jet}\ (GeV)$', 20, 0, 400)
#        histogram = histogram.rebin('pt', new_pt_bins)
#    elif name == 'N_b':
#        # rebin
#        axis = 'multiplicity'
#        new_n_bins = hist.Bin('multiplicity', r'$N_{b-jet}$', 5, -0.5, 4.5)
#        histogram = histogram.rebin('multiplicity', new_n_bins)
#    elif name == 'b_nonb_massmax':
#        # rebin
#        axis = 'mass'
#        new_mass_bins = hist.Bin('mass', r'$M(b, light) \ (GeV)$', 25, 0, 1500)
#        histogram = histogram.rebin('mass', new_mass_bins)
#    elif name == 'light_pair_mass':
#        # rebin
#        axis = 'mass'
#        new_mass_bins = hist.Bin('mass', r'$M(light, light) \ (GeV)$', 25, 0, 1500)
#        histogram = histogram.rebin('mass', new_mass_bins)
#    elif name == 'lep_light_mass':
#        # rebin
#        axis = 'mass'
#        new_mass_bins = hist.Bin('mass', r'$M(lepton, light) \ (GeV)$', 25, 0, 1500)
#        histogram = histogram.rebin('mass', new_mass_bins)
#    elif name == 'lep_b_mass':
#        # rebin
#        axis = 'mass'
#        new_mass_bins = hist.Bin('mass', r'$M(b, lepton) \ (GeV)$', 25, 0, 1500)
#        histogram = histogram.rebin('mass', new_mass_bins)
#    elif name == 'lep_b_pt':
#        # rebin
#        axis = 'pt'
#        new_pt_bins = hist.Bin('pt', r'$p_{b, lep}\ (GeV)$', 20, 0, 400)
#        histogram = histogram.rebin('pt', new_pt_bins)
#    elif name == 'light_pair_pt':
#        # rebin
#        axis = 'pt'
#        new_pt_bins = hist.Bin('pt', r'$p_{light, light}\ (GeV)$', 20, 0, 400)
#        histogram = histogram.rebin('pt', new_pt_bins)
#    elif name == 'lep_light_pt':
#        # rebin
#        axis = 'pt'
#        new_pt_bins = hist.Bin('pt', r'$p_{light, lep}\ (GeV)$', 20, 0, 400)
#        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'S_T':
        # rebin
        axis = 'pt'
        new_pt_bins = hist.Bin('pt', r'$p_{b, nonb}\ (GeV)$', 20, 0, 1000)
        histogram = histogram.rebin('pt', new_pt_bins)
#    elif name == 'H_T':
#        # rebin
#        axis = 'pt'
#        new_pt_bins = hist.Bin('pt', r'$p_{b, nonb}\ (GeV)$', 20, 0, 400)
#        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'pt_b':
        # rebin
        axis = 'pt'
        new_pt_bins = hist.Bin('pt', r'$p_{single b}\ (GeV)$', 20, 0, 400)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'eta_b':
        # rebin
        axis = 'eta'
        new_eta_bins = hist.Bin('eta', r'$\eta$', 30, -5.5, 5.5)
        histogram = histogram.rebin('eta', new_eta_bins)
    elif name == 'b_debug':
        # rebin
        axis = 'multiplicity'
        new_n_bins = hist.Bin('multiplicity', r'$N_{jet, fwd}$', 15, -0.5, 14.5)
        histogram = histogram.rebin('multiplicity', new_n_bins)
    elif name == 'HT':
        # rebin
        axis = 'ht'
        new_n_bins = hist.Bin('ht', r'$M(b, light) \ (GeV)$', 30, 0, 1500)
        histogram = histogram.rebin('ht', new_n_bins)
    elif name == 'ST':
        # rebin
        axis = 'ht'
        new_n_bins = hist.Bin('ht', r'$M(b, light) \ (GeV)$', 30, 0, 1500)
        histogram = histogram.rebin('ht', new_n_bins)
#    elif name == 'phi_b':
#        # rebin
#        axis = 'phi'
#        new_phi_bins = hist.Bin('phi', r'$phi(single b)$', 30, -5.5, 5.5)
#        histogram = histogram.rebin('phi', new_phi_bins)
#    elif name == 'pt_lep':
#        # rebin
#        axis = 'pt'
#        new_pt_bins = hist.Bin('pt', r'$p_{single lep}\ (GeV)$', 20, 0, 400)
#        histogram = histogram.rebin('pt', new_pt_bins)
#    elif name == 'eta_lep':
#        # rebin
#        axis = 'eta'
#        new_eta_bins = hist.Bin('eta', r'$eta(single lep)$', 30, -5.5, 5.5)
#        histogram = histogram.rebin('eta', new_eta_bins)
#    elif name == 'phi_lep':
#        # rebin
#        axis = 'phi'
#        new_phi_bins = hist.Bin('phi', r'$phi(single lep)$', 30, -5.5, 5.5)
#        histogram = histogram.rebin('phi', new_phi_bins)
#    elif name == 'pt_lead_light':
#        # rebin
#        axis = 'pt'
#        new_pt_bins = hist.Bin('pt', r'$p_{lead light jet}\ (GeV)$', 20, 0, 400)
#        histogram = histogram.rebin('pt', new_pt_bins)
#    elif name == 'eta_lead_light':
#        # rebin
#        axis = 'eta'
#        new_eta_bins = hist.Bin('eta', r'$eta(lead light jet)$', 30, -5.5, 5.5)
#        histogram = histogram.rebin('eta', new_eta_bins)
#    elif name == 'phi_lead_light':
#        # rebin
#        axis = 'phi'
#        new_phi_bins = hist.Bin('phi', r'$phi(lead light jet)$', 30, -5.5, 5.5)
#        histogram = histogram.rebin('phi', new_phi_bins)
#    elif name == 'mt_lep_MET':
#        # rebin
#        axis = 'mass'
#        new_mass_bins = hist.Bin('mass', r'$M(light, light) \ (GeV)$', 25, 0, 1500)
#        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'mbj_max':
        # rebin
        axis = 'mass'
        new_mass_bins = hist.Bin('mass', r'$M(light, light) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'mjj_max':
        # rebin
        axis = 'mass'
        new_mass_bins = hist.Bin('mass', r'$M(light, light) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
#    elif name == 'mlb_min':
#        # rebin
#        axis = 'mass'
#        new_mass_bins = hist.Bin('mass', r'$M(light, light) \ (GeV)$', 25, 0, 1500)
#        histogram = histogram.rebin('mass', new_mass_bins)
#    elif name == 'mlb_max':
#        # rebin
#        axis = 'mass'
#        new_mass_bins = hist.Bin('mass', r'$M(light, light) \ (GeV)$', 25, 0, 1500)
#        histogram = histogram.rebin('mass', new_mass_bins)
#    elif name == 'mlj_min':
#        # rebin
#        axis = 'mass'
#        new_mass_bins = hist.Bin('mass', r'$M(light, light) \ (GeV)$', 25, 0, 1500)
#        histogram = histogram.rebin('mass', new_mass_bins)
#    elif name == 'mlj_max':
#        # rebin
#        axis = 'mass'
#        new_mass_bins = hist.Bin('mass', r'$M(light, light) \ (GeV)$', 25, 0, 1500)
#        histogram = histogram.rebin('mass', new_mass_bins)
    else:
        skip = True

    if not skip:
        y_max = histogram.sum("dataset").values(overflow='over')[()].max()
        y_over = histogram.sum("dataset").values(overflow='over')[()][-1]

        # get pseudo data
        bin_values = histogram.axis(axis).centers(overflow='over')
        poisson_means = histogram.sum('dataset').values(overflow='over')[()]
        values = np.repeat(bin_values, np.random.poisson(np.maximum(np.zeros(len(poisson_means)), poisson_means)))
        if axis == 'pt':
            histogram.fill(dataset='pseudodata', pt=values)
        elif axis == 'ht':
            histogram.fill(dataset='pseudodata', ht=values)
        elif axis == 'mass':
            histogram.fill(dataset='pseudodata', mass=values)
        elif axis == 'multiplicity':
            histogram.fill(dataset='pseudodata', multiplicity=values)
        elif axis == 'eta':
            histogram.fill(dataset='pseudodata', eta=values)
        elif axis == 'phi':
            histogram.fill(dataset='pseudodata', phi=values)

        
        import re
        notdata = re.compile('(?!pseudodata)')

        fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

        # get axes 
        hist.plot1d(histogram[notdata],overlay="dataset", ax=ax, stack=True, overflow='over', clear=False, line_opts=None, fill_opts=fill_opts, error_opts=error_opts, order=['tW_scattering', 'TTX', 'TTW','ttbar','wjets']) #error_opts??
        hist.plot1d(histogram['pseudodata'], overlay="dataset", ax=ax, overflow='over', error_opts=data_err_opts, clear=False)

        # build ratio
        hist.plotratio(
            num=histogram['pseudodata'].sum("dataset"),
            denom=histogram[notdata].sum("dataset"),
            ax=rax,
            error_opts=data_err_opts,
            denom_fill_opts={},
            guide_opts={},
            unc='num',
            overflow='over'
        )


        for l in ['linear', 'log']:
            saveFig(fig, ax, rax, plotDir, name, scale=l, shape=False, y_max=y_max)
        fig.clear()
        rax.clear()
        #pd_ax.clear()
        ax.clear()

    
    if not name in bins.keys():
        continue

    axis = bins[name]['axis']
    print (name, axis)
    histogram = histogram.rebin(axis, bins[name]['bins'])

    y_max = histogram.sum("dataset").values(overflow='over')[()].max()
    y_over = histogram.sum("dataset").values(overflow='over')[()][-1]

    # get pseudo data
    bin_values = histogram.axis(axis).centers(overflow=bins[name]['overflow'])
    poisson_means = histogram.sum('dataset').values(overflow=bins[name]['overflow'])[()]
    values = np.repeat(bin_values, np.random.poisson(np.maximum(np.zeros(len(poisson_means)), poisson_means)))
    if axis == 'pt':
        histogram.fill(dataset='pseudodata', pt=values)
    elif axis == 'ht':
        histogram.fill(dataset='pseudodata', ht=values)
    elif axis == 'mass':
        histogram.fill(dataset='pseudodata', mass=values)
    elif axis == 'multiplicity':
        histogram.fill(dataset='pseudodata', multiplicity=values)
    elif axis == 'ht':
        histogram.fill(dataset='pseudodata', ht=values)
    elif axis == 'norm':
        histogram.fill(dataset='pseudodata', norm=values)

    
    import re
    notdata = re.compile('(?!pseudodata)')

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

    # get axes
    hist.plot1d(histogram[notdata],overlay="dataset", ax=ax, stack=True, overflow=bins[name]['overflow'], clear=False, line_opts=None, fill_opts=fill_opts, error_opts=error_opts, order=['tW_scattering', 'TTX', 'TTW','ttbar','wjets']) #error_opts??
    hist.plot1d(histogram['pseudodata'], overlay="dataset", ax=ax, overflow=bins[name]['overflow'], error_opts=data_err_opts, clear=False)

    # build ratio
    hist.plotratio(
        num=histogram['pseudodata'].sum("dataset"),
        denom=histogram[notdata].sum("dataset"),
        ax=rax,
        error_opts=data_err_opts,
        denom_fill_opts={},
        guide_opts={},
        unc='num',
        overflow=bins[name]['overflow']
    )


    for l in ['linear', 'log']:
        saveFig(fig, ax, rax, plotDir, name, scale=l, shape=False, y_max=y_max)
    fig.clear()
    rax.clear()
    ax.clear()

    
    try:
        #>>>>>>> 6fe1e83fa41b23b262513c90f05c4b5545e97540
        fig, ax = plt.subplots(1,1,figsize=(7,7))
        notdata = re.compile('(?!pseudodata|wjets|diboson)')
        hist.plot1d(histogram[notdata],overlay="dataset", density=True, stack=False, overflow=bins[name]['overflow'], ax=ax) # make density plots because we don't care about x-sec differences
        for l in ['linear', 'log']:
            saveFig(fig, ax, None, plotDir, name+'_shape', scale=l, shape=True)
        fig.clear()
        ax.clear()
    except ValueError:
        print ("Can't make shape plot for a weird reason")

    fig.clear()
    ax.clear()

    plt.close()


print ()
print ("Plots are here: http://uaf-10.t2.ucsd.edu/~%s/"%os.path.expandvars('$USER')+str(plotDir.split('public_html')[-1]) )
