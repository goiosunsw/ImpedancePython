#!/usr/bin/env python
"""
    vt_perturbations
    ----------------

    Calculate acoustic differences to a range of perturbations to the vocal
    tract geometry, and generate a summary of these

    Depends on pympedance: https://github.com/goiosunsw/ImpedancePython.git

    Author: Andre Almeida <a.almeida@unsw.edu.au> 
    Creation date: 2018/02/22
    Version: 0.0.1
"""

import sys
import pandas as pd
import scipy.signal as sig
import pympedance as pym

TYP_TO_FUN_DICT = {'notch': _pert_notch,
                   'const_vol': _pert_const_vol}

__parallel = True

def profile_to_duct(l=None, rad=None,
                    nsegments=None,
                    reverse=False, 
                    termination='piston', 
                    loss_multiplier=None):
    """
    Generate a pympedance.Synthesiser.Duct object 
    from a list of radii and segment lengths. 

    The object can be interpolated using nsegments (int) 
    so that segments are homogeneous

    Other arguments:
        * nsegments: number of segments of interpolation or None for raw
        * reverse: reverse the order of the segments
        * termination: type of acoustic termination:
            + closed (flow variation = 0 at end)
            + open (pressure variation = 0 at end)
            + piston (realistic plane-wave radiation)
        * loss_multiplier (incease loss factor, for instance to simulate
          flexible wall losses)
    """
    vt = pym.Synthesiser.Duct()

    if nsegments is not None:
        old_x = np.concatenate(([0], (np.cumsum(l))))
        new_x = np.arange(0, old_x[-1], nsegments)
        new_r = np.interp(new_x, old_x, rad)
        rad = new_r
        l = np.diff(new_x)

    if reverse:
        l = np.flipud(l)
        rad = np.flipud(rad)
    
    for ll, rr in zip(l,rad):
        if ll>0:
            vt.append_element(pym.Synthesiser.StraightDuct(length=ll,radius=rr,loss_multiplier=loss_multiplier))
    
    if termination=='piston':
        vt.set_termination(pym.Synthesiser.FlangedPiston(radius=rr))
    elif termination=='open':
        vt.set_termination(pym.Synthesiser.PerfectOpenEnd())
    elif termination=='closed':
        vt.set_termination(pym.Synthesiser.PerfectClosedEnd())
        
    return vt

def vocal_tract_reader(filename, columns=['lenghts','radii'], 
                       skiprows=0,
                       unit_multiplier=1.):
    """
    reads a vocat tract file with two columns:
        * By default first column has lengths and second column has radii in m
        * Valid column names are:
            + lengths
            + positions (segment limits: first segment starts at 0)
            + radii
            + area
        * Choose unit multiplier to convert units to m. for example, 1000 means
        the units of the file is mm, and .5 means the units are in m but the
        column represents diameter

    returns a pympedance.Synthesiser.Duct object
    """

    vtpd = pd.read_csv(fn, header=None, skiprows=skiprows)
    for ic, col in enumerate(columns):
        if col == 'lengths':
            l = np.array(vtpd[ic].tolist())
        elif col == 'positions':
            x = np.array(vtpd[ic].tolist())
            x = np.concatenate(([0], x))
            l = np.diff(x)
        elif col == 'radii':
            r = np.array(vtpd[ic].tolist())
        elif col == 'area':
            r = (np.array(vtpd[ic].tolist())/np.pi)**.5
        else:
            sys.stderr.write('Column {} ({}) skipped\n'.format(ic, col))
        
        # find 0 or negative lengths, warn and remove them
        nnp = np.flatnonzero(l<=0)
        if len(nnp>0):
            sys.stderr.write('Segments skipped:\n')
            for ii in nnp:
                sys.stderr.write('  {}: l={}, r={}\n'.format(ii,l[ii],r[ii]))
            r = r[np.logical_not(nnp)]
            l = l[np.logical_not(nnp)]
        
        vt = profile_to_duct(lengths=l, rad=r, nsegments=None)

def _pert_notch(vt, seg_idx, frac=0.01, scale_power=1.):
    """
    Returns perturbated vocal tract
    """
    vtp = vt.copy()
    vtp.elements[seg_idx] *= (1+frac)
    return vtp

def all_impedance_calculator(vt, tpy, seg_idx=0, frac=.01, scale_power=1.):
    pert_func = TYP_TO_FUN_DICT[typ]
    vtp = pert_func(vt, seg_idx, frac, scale_power)


def perturbation_analyser(vt, types='all', fracs=[0.1], scale_power=1):
    """
    Calculates perturbations to standard profile

    types: 
        * notch: a change in a single segment
        * const_vol: a change in two consecutive segments that maintains tract
        volume constant
    outputs:
        * 'zl': impedance at left end of the tract (x=0)
        * 'zr': impedance at right end of tract
        * 'transpedance': ratio of radiated pressure at right to input flow at x=0
    fracs:
        magnitude of change in radius
    scale_power:
        how to scale the magnitude change with radius
    """

    alltypes = TYP_TO_FUN_DICT.keys()
    if types == 'all':
        types = alltypes
    else:
        for tt in types:
            assert tt in alltypes

    # if outputs == 'all':
    #     outputs = ['zl', 'zr', 'transpedance']

    arg_iter = product([vt], types, range(len(vt.elements)), fracs)
    if __parallel:
        with Pool(processes = NPROC) as pool:
            results = pool.starmap(all_impedance_calculator, arg_iter)
    else:
        results = starmap(all_impedance_calculator, arg_iter)

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('-o', '--outfile', help="Output file",
                        default=sys.stdout, type=argparse.FileType('w'))

    args = parser.parse_args(arguments)

    vt = vocal_tract_reader(infile)
    
    perturbation_analyser(tract=tract, types='all', outputs='all')


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
