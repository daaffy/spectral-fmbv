'''
    load.py
Bundle of front-end functions that call FMBV related functions.
'''
import sys
sys.path.insert(1,'./src') # Add src directory to path.
sys.path.insert(1,'./src/legacy_src') # 
import pandas as pd
import SimpleITK as sitk
import numpy as np
import os
from fmbv_refactor import FMBV
from computeFMBV import computeFMBV
from computeFMBV_DD import computeFMBV_DD
        

'''
    run_paths
Run FMBV using data supplied from paths. Only pd_path is required; default segmentation will be applied and kretz-related methods
such as depth-correction will be inaccessible.
'''
def run_paths(pd_path='', seg_path='', kretz_path='', output_path='', mode=0, **fmbvkwargs):
    if not pd_path:
        raise Exception("pd_path required.")
    else:
        return run(kretz_path, pd_path, seg_path, output_path=output_path, mode=mode, **fmbvkwargs)


'''
    run
Handles running all FMBV routines given the input format:

Set output_path to save as .csv.

Outputs (in order):
- ...
'''
def run(*input, output_path='', mode=0, start = 0, **fmbvkwargs):

    print("Running batch...")

    if len(input) == 1:
        # .csv -> data frame

        # Perform string checks...

        # df = pd.read_csv(input[0], index_col=0)
        df = pd.read_csv(input[0])
        print(df)
    elif len(input) == 2:
        # no kretz supplied!
        df = pd.DataFrame(data={
            'index': [0],
            'kretzpath': [''],
            'dopplerpath': [input[0]],
            'segpath': [input[1]]
        })
    elif len(input) == 3:
        # file list -> data frame

        # Perform string checks...

        df = pd.DataFrame(data={
            'index': [0],
            'kretzpath': [input[0]],
            'dopplerpath': [input[1]],
            'segpath': [input[2]]
        })
    else:
        raise Exception("Invalid arguments. Supply either: [.csv], or [.vol], [.nii.gz], [nii.gz]")
    
    # Loop through data frame
    c = 0
    for ind, row in df.iterrows():
        if ind < start:
            continue

        print(str(c)+" of "+str(len(df)))
        print(str(ind)+" of "+str(len(df)))

        # Assume all file paths are formatted similarly. Therefore, we only check dopplerpath.
        if os.path.isfile(row['dopplerpath']) == False:
                data_root = os.getcwd()
        else:
            data_root = ''
        
        pd_path = data_root + row['dopplerpath']
        seg_path = data_root + row['segpath']

        print(pd_path)
        print(seg_path)

        # print(seg_path)

        f = None
        if mode == 1: # construct f once to save time!
            print("mode == 1; setting f_bypass...")
            f = FMBV(mode=0, **fmbvkwargs)
            f.load_pd(pd_path)
            f.load_seg(seg_path)

        # Save outputs to df
        df.at[ind, 'FMBV'], df.at[ind,'MPI'] = fmbv_basic(pd_path, seg_path, mode=mode, f_bypass=f,  **fmbvkwargs)

        print(df.at[ind, 'FMBV'])

        # df.at[ind, 'vol_ml'] = seg_vol(seg_path)
        if str(row['kretzpath']) and not row.isna().any():
        # if False:
            kretz_path = data_root + str(row['kretzpath'])

            df.at[ind, 'FMBV Depth Corrected'], df.at[ind,'volume_mm3'], df.at[ind,'start_mm'], df.at[ind,'end_mm'], df.at[ind,'slice_num'], df.at[ind,'success_pc'] = fmbv_depth_corrected(pd_path, seg_path, kretz_path, mode=mode, f_bypass=f, **fmbvkwargs)
        else:
            # print(row['kretzpath'])
            df.at[ind, 'FMBV Depth Corrected'], df.at[ind,'volume_mm3'], df.at[ind,'start_mm'], df.at[ind,'end_mm'], df.at[ind,'slice_num'], df.at[ind,'success_pc'] = 0, 0, 0, 0, 0, 0, 0
        
        if not output_path == '':
            df.to_csv(output_path, index=False)

        c = c + 1
        
    return df

def fmbv_basic(pd_path, seg_path, mode=0, f_bypass=None, **fmbvkwargs):
    fmbv = 0
    mpi = 0

    try:
        if mode == 0:
            if seg_path != '':
                _,  fmbv, _, _ , mpi = computeFMBV(pd_path, seg_path)
            else:
                raise Exception("mode=0 requires segmentation.")
        elif mode == 1: # Refactor
            if f_bypass == None:
                f = FMBV(mode=0, **fmbvkwargs)
                f.load_pd(pd_path)
                f.load_seg(seg_path)
            else:
                f = f_bypass
            
            f.global_method()

            try:
                f.naive_fmbvs()
                mpi = f.fmbvs_naive["mpi"]
            except:
                mpi = np.nan

            fmbv = f.global_fmbv_value_2
        elif mode == 2:
            f = FMBV(mode=1, **fmbvkwargs)
            f.load_pd(pd_path)
            f.load_seg(seg_path)
            f.global_method()
            fmbv = f.global_fmbv_value_2
            mpi = 0
        else:
            raise Exception("Mode not recognised by fmbv_basic.")
    except Exception as e:
        print(e)
    return fmbv, mpi

def fmbv_depth_corrected(pd_path, seg_path, kretz_path, mode=0, f_bypass=None, **fmbvkwargs):
    fmbv = 0
    # mpi = 0
    start = 0
    end = 0
    numslices = 0
    success_pc = 0
    seg_volume = 0

    try:
        if mode == 0:
            if seg_path != '':
                fmbv, mpi, start, end, numslices, success_pc = computeFMBV_DD(pd_path, seg_path, kretz_path)
            else:
                raise("mode=0 requires segmentation.")
        elif mode == 1:
            # print("I'm in the _DD refactor!")
            if f_bypass == None:
                f = FMBV(mode=0, **fmbvkwargs)
                f.load_pd(pd_path)
                f.load_seg(seg_path)
            else:
                f = f_bypass

            f.load_kretz(kretz_path)

            # print('Kretz supplied? '+str(f.kretz_supplied))
            try:
                f.depth_corrected_method()
                fmbv = f.dc_fmbv
            except:
                fmbv = np.nan


            start = f.start_depth
            end = f.end_depth
            numslices = f.numslices
            success_pc = f.success_num
            seg_volume = f.seg_volume
        elif mode == 2:
            f = FMBV(mode=1,**fmbvkwargs)
            f.load_pd(pd_path)
            f.load_seg(seg_path)
            f.load_kretz(kretz_path)

            # print('Kretz supplied? '+str(f.kretz_supplied))
            try:
                f.depth_corrected_method()
                fmbv = f.dc_fmbv
            except:
                fmbv = np.nan
            # mpi = 0
            start = 0
            end = 0
            numslices = 0
            success_pc = 0
            seg_volume = 0
        else:
            raise Exception("Mode not recognised by fmbv_depth_corrected")
    except Exception as e:
        print(e)
        
    return fmbv, seg_volume, start, end, numslices, success_pc