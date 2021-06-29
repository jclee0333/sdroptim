"""
  MPI role for Hyper Parameter Optimization by Jeongcheol lee
  -- jclee@kisti.re.kr
"""

from ast import literal_eval
import json, copy,os, time
import gc
import pandas as pd
import numpy as np
import optuna
from mpi4py import MPI
from multiprocessing import Pool
from itertools import chain
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from sdroptim.searching_strategies import stepwise_get_current_step_by_time, n_iter_calculation
from sdroptim.searching_strategies import _init_seed_fix_torch
from sdroptim.searching_strategies import find_linear
from sdroptim.searching_strategies import stepwise_guided_mpi_by_time, ModObjectiveFunctionWrapper, params_sorting_by_guided_list

guided_importance_order = ['lr', 'epoch','batch_size','decay', 'momentum']

#default_params = {
#                    'lr':0.0001,
#                    'decay':0.0002,
#                    'momentum':0.9,
#                    'batch_size':512,
#                    'epoch':5
#}

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

#####################################################
# threading rank 0 receiving
#####################################################

import threading
import time

class ThreadingforOptunaRank0(object):
    def __init__(self, study, comm, arg, tags, interval=0.5):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.study = study
        self.comm = comm
        self.arg = arg
        self.tags = tags
        self.interval = interval
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution
    def run(self):
        closed_workers = 0
        num_workers = self.comm.size
        begin_time = time.time()
        while closed_workers < num_workers:
            # resource allocation (processor acquisition @ READY status)
            status = MPI.Status()
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            elapsed_time = time.time() - begin_time
            if elapsed_time+self.interval > self.arg.max_sec: # do until max_sec
                self.comm.Abort() #MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD with errorcode 0.
                                  #NOTE: invoking MPI_ABORT causes Open MPI to kill all MPI processes.You may or may not see output from other processes,
                                  #depending on exactly when Open MPI kills them.
            if tag == self.tags.READY:
                print("TIME***",elapsed_time, self.arg.max_sec)
                if elapsed_time < self.arg.max_sec: # do until max_sec
                    self.comm.send(self.study, dest=source, tag=self.tags.START) # allow to train (1)
                else:
                    print("::::TIMEOVER::::", source)
                    self.comm.send(None, dest=source, tag=self.tags.EXIT) # do not allow to train
            elif tag == self.tags.DONE:
                # finished each trial -- log something
                study = data
            elif tag == self.tags.EXIT:
                # # of worker control (not yet)
                closed_workers += 1
                print("***CLOSEDWORKERS******************************* = ", closed_workers, num_workers)
            time.sleep(self.interval)

from mpi4py.futures import MPIPoolExecutor

class ThreadingforStepwiseRank0(object):
    def __init__(self, study, comm, arg, tags, n_inner_loop, original_stepwise_params, interval=0.5):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.study = study
        self.comm = comm
        self.arg = arg
        self.tags = tags
        self.n_inner_loop = n_inner_loop
        self.original_stepwise_params = original_stepwise_params
        self.controlled_stepwise_params = original_stepwise_params # initialize
        self.interval = interval
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution
    def run(self):
        closed_workers = 0
        num_workers = self.comm.size
        begin_time = time.time()
        while closed_workers < num_workers:
            # resource allocation (processor acquisition @ READY status)
            status = MPI.Status()
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            elapsed_time = time.time() - begin_time
            if elapsed_time+self.interval > self.arg.max_sec: # do until max_sec
                self.comm.Abort() #MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD with errorcode 0.
                                  #NOTE: invoking MPI_ABORT causes Open MPI to kill all MPI processes.You may or may not see output from other processes,
                                  #depending on exactly when Open MPI kills them.
            if tag == self.tags.READY:
                print("TIME***",elapsed_time, self.arg.max_sec)
                current_step = stepwise_get_current_step_by_time(self.arg.max_sec, elapsed_time, self.n_inner_loop)
                if elapsed_time < self.arg.max_sec: # do until max_sec
                    self.comm.send([self.study, self.controlled_stepwise_params, current_step], dest=source, tag=self.tags.START) # allow to train (1)
                else:
                    print("::::TIMEOVER::::", source)
                    #### kill all process
                    with MPIPoolExecutor() as executor:
                        executor.shutdown(wait=False)
                    self.comm.send(None, dest=source, tag=self.tags.EXIT) # do not allow to train
            elif tag == self.tags.DONE:
                # finished each trial -- log something
                study = data[0]
                # update only received_params are not None
                if data[1] != None:
                    self.controlled_stepwise_params = data[1]
            elif tag == self.tags.EXIT:
                # # of worker control (not yet)
                closed_workers += 1
                print("***CLOSEDWORKERS******************************* = ", closed_workers, num_workers)
            time.sleep(0.5)

########################################################
### 2021-05-04 
########################################################
def load_metadata(json_file_name='metadata.json'):
    with open(json_file_name) as data_file:
        gui_params = json.load(data_file)
    return gui_params

def get_dask_compatible_false_primitives_list(primitives):
    import featuretools as ft
    plist = ft.primitives.list_primitives()
    plist_dask_false_all_list = plist[plist['dask_compatible']==False]
    pfalse_agg = plist_dask_false_all_list[plist_dask_false_all_list['type']=='aggregation']
    pfalse_trans = plist_dask_false_all_list[plist_dask_false_all_list['type']=='transform']
    pfalse_agg_names = pfalse_agg['name'].to_list()
    pfalse_trans_names = pfalse_trans['name'].to_list()
    pfalse_all_list = pfalse_agg_names + pfalse_trans_names
    #pfalse_all_list = ['cum_sum','diff','cum_min','percentile','cum_max','cum_mean','haversine','latitude','longitude','cum_count','time_since_previous','first','last','trend','skew','median','n_most_common','mode','time_since_last','time_since_first','avg_time_between','entropy']
    #
    plist_dask_true_all_list = plist[plist['dask_compatible']==True]
    ptrue_agg = plist_dask_true_all_list[plist_dask_true_all_list['type']=='aggregation']
    ptrue_trans = plist_dask_true_all_list[plist_dask_true_all_list['type']=='transform']
    ptrue_agg_names = ptrue_agg['name'].to_list()
    ptrue_trans_names = ptrue_trans['name'].to_list()
    ptrue_all_list = ptrue_agg_names + ptrue_trans_names
    #
    by_cols_only_methods = [x for x in primitives if x in pfalse_all_list]
    if by_cols_only_methods:
        return by_cols_only_methods
    else:
        return None

def get_index_list_by_rows(n_rows, n_jobs, overlap=0.0): # bug fixed 21-04-21
    # REMIND: from_ <= index <= to_
    r = []
    ind = 0
    #n_rows=len(df)
    chunk_size = int(n_rows / n_jobs)
    #print("*",chunk_size)
    the_rest = n_rows-chunk_size*n_jobs
    overlap_size = int(chunk_size*overlap)
    for i in range(0,n_jobs):
        from_ = ind
        ind+=chunk_size
        to_ = ind if ind<n_rows-the_rest else n_rows
        if overlap>0.0:
            from_ = from_ if from_ == 0 else from_-overlap_size
            to_ = to_ if to_==n_rows else to_+overlap_size
        #r.append((from_,to_))
        r.append((from_,to_-1)) # fix as 'from_ <= index <= to_'
    return r

def check_csv_has_header(sample):
    import csv
    sniffer = csv.Sniffer()
    return sniffer.has_header(sample) 

def get_csv_shape(filepath, delimiter=','):
    # faster method comparing to pandas loading
    import gc
    n_sampleline=10
    try:
        with open(filepath) as f:
            data=f.readlines()
            #has_header = not any(cell.isdigit() for cell in data[0].split(delimiter)) ## old version that cannot find numeric column names
            has_header = check_csv_has_header(''.join(data[0:n_sampleline]))
            cols = len(data[0].split(delimiter))
            rows = len(data)-1 if has_header else len(data)
        del data
        gc.collect()
        return (rows, cols), has_header
    except:
        pass

def get_skiprows_for_partial_reading_csv(has_header, index_range, full_range):
    if len(index_range)!=2:
        return []
    else:
        from_i = index_range[0]
        to_i = index_range[1]
        from_f = full_range[0]
        to_f = full_range[1]
        #
        if has_header:
            from_i += 1
            to_i += 1
            from_f += 1
            to_f += 1
        return [x for x in range(from_f, to_f+1) if x not in range(from_i, to_i+1)]

def data_loader(specific_data_chunk_to_consume, processor):
    import pandas as pd
    data_list = []
    current_group_no = specific_data_chunk_to_consume['group_no'].values[0]
    for _index, row in specific_data_chunk_to_consume.iterrows():
        print("loading.. [G"+str(current_group_no) +"/P"+str(processor)+"] "+row['filepath']+" "+str(row['index_range'])+" / "+str(row['full_range'])+" on processor "+str(processor))
        each_df = pd.read_csv(
            row['filepath'], 
            skiprows=get_skiprows_for_partial_reading_csv(row['has_header'], row['index_range'], row['full_range']))
        each_df.index=range(row['index_range'][0],row['index_range'][1]+1)
        agg = row['agg']
        trans = row['trans']
        data_list.append((row, each_df))
    return data_list, (agg, trans), current_group_no

def get_a_data_chunk_per_group(data_chunk_df, group_number=0, random_pick=True):
    import pandas as pd
    df = data_chunk_df.copy()
    if random_pick:
        #group_number = df['group_no'].sample(1).values[0]
        group_number = df['group_no'].min()
    specific_data_chunk_to_consume = df[df['group_no']==group_number]
    the_rest = df[df['group_no']!=group_number]
    #
    return specific_data_chunk_to_consume, the_rest

######################################################

def get_data_chunk_by_metadata(gui_params, renew=False): # renew will be True after basic development
    import pandas as pd
    #group_no = gui_params['group_no'] if 'group_no' in gui_params else 1
    group_no = 1 # default group_no
    if 'autofe_system_attr' in gui_params:
        if 'group_no' in gui_params['autofe_system_attr']:
            group_no = gui_params['autofe_system_attr']['group_no'] # update group_no if exists in metadata.json
    group_no_for_save = group_no
    allocated_proc = 0
    chunk_list = [] # final chunk list according to the group_no
    base_has_header = False
    temp_has_header = False
    ######################
    ###### load if the chunk already exists
    ##
    if 'autofe_system_attr' in gui_params:
        if 'title' in gui_params['autofe_system_attr']:
            title = gui_params['autofe_system_attr']['title']
            outputpath = os.path.join("./", title+"__chunkfor"+str(group_no_for_save)+"subGroup.csv")
            if (os.path.exists(outputpath)) and (not renew):
                res = pd.read_csv(outputpath)
                from ast import literal_eval
                res['index_range'] = res['index_range'].apply(lambda x: literal_eval(str(x)))
                res['full_range'] = res['full_range'].apply(lambda x: literal_eval(str(x)))
                res['agg'] = res['agg'].apply(lambda x: literal_eval(str(x)))
                res['trans'] = res['trans'].apply(lambda x: literal_eval(str(x)))
                return res
    #######################################
    dataset = []
    # for a base csv
    if 'ml_file_path' in gui_params:
        if 'ml_file_name' in gui_params:
            base_filepath = os.path.join(gui_params['ml_file_path'], gui_params['ml_file_name'])
            if 'n_rows' in gui_params:
                if 'has_header' in gui_params:
                    base_n_rows = gui_params['n_rows']
                    base_has_header = bool(gui_params['has_header'])
            else:
                base_csv_shape, base_has_header = get_csv_shape(base_filepath) # (rows, cols)
                base_n_rows = base_csv_shape[0]
            dataset.append((base_filepath, base_n_rows, base_has_header)) # dataset[0] is the base csv
    ######################
    # for additional csvs (if exists)
    if 'additional_files' in gui_params:
        for i in range(len(gui_params['additional_files'])):
            if 'ml_file_path' in gui_params['additional_files'][i]:
                if 'ml_file_name' in gui_params['additional_files'][i]:
                    temp_filepath = os.path.join(gui_params['additional_files'][i]['ml_file_path'], gui_params['additional_files'][i]['ml_file_name'])
                    if 'n_rows' in gui_params['additional_files'][i]:
                        if 'has_header' in gui_params['additional_files'][i]:
                            temp_n_rows = gui_params['additional_files'][i]['n_rows']
                            temp_has_header = bool(gui_params['additional_files'][i]['has_header'])
                    else:
                        temp_csv_shape, temp_has_header = get_csv_shape(temp_filepath) # (rows, cols)
                        temp_n_rows = temp_csv_shape[0]
                    dataset.append((temp_filepath, temp_n_rows, temp_has_header))
    ######################
    # check primitives that CANNOT run under dask environments
    if "autofe_system_attr" in gui_params:
        agg   = gui_params['autofe_system_attr']['aggregation_primitives'] if 'aggregation_primitives' in gui_params['autofe_system_attr'] else []
        trans = gui_params['autofe_system_attr']['transformation_primitives'] if 'transformation_primitives' in gui_params['autofe_system_attr'] else []
        if group_no == 1:
            if dataset:
                for d in dataset:
                    chunk_list.append(
                        (d[0], d[2], (0,d[1]-1), (0,d[1]-1), 0, agg, trans, False)
                        )
        else:
            primitives = agg+trans
            dask_compatible_false_primitives_list = get_dask_compatible_false_primitives_list(primitives)
            #
            dask_false_agg = []
            dask_false_trans = []
            dask_true_agg = agg
            dask_true_trans = trans
            if dask_compatible_false_primitives_list:
                group_no = group_no - 1 # one process for dask false primitivies and other processes for the rest
                dask_false_agg = [x for x in agg if x in dask_compatible_false_primitives_list]
                dask_false_trans = [x for x in trans if x in dask_compatible_false_primitives_list]
                #
                dask_true_agg = [x for x in agg if x not in dask_compatible_false_primitives_list]
                dask_true_trans  = [x for x in trans if x not in dask_compatible_false_primitives_list]
            if dataset:
                for d in dataset:
                    ################################
                    # for dask_false
                    if dask_false_agg + dask_false_trans:
                        chunk_list.append(
                            (d[0], d[2], (0,d[1]-1), (0,d[1]-1), allocated_proc, dask_false_agg, dask_false_trans, False)
                            )
                        allocated_proc += 1
                    ################################
                    # for dask_true
                    if dask_true_agg + dask_true_trans:
                        index_list = get_index_list_by_rows(d[1],group_no)
                        for il in index_list:
                            chunk_list.append(
                                (d[0], d[2], il, (0,d[1]-1), allocated_proc, dask_true_agg, dask_true_trans, True)
                                )
                            allocated_proc += 1
                    #################################
                    allocated_proc = 0 # reset
    res = pd.DataFrame(chunk_list, columns=['filepath', 'has_header', 'index_range', 'full_range', 'group_no', 'agg', 'trans', 'parallelable']) # add parallelable Boolean
    outputpath = os.path.join("./", title+"__chunkfor"+str(group_no_for_save)+"subGroup.csv")
    res.to_csv(outputpath, index=False)
    os.chmod(outputpath, 0o776)
    return res

def get_data_chunk_by_metadata_old_forsave(gui_params):
    import pandas as pd
    #gui_params=load_metadata()
    #n_proc = gui_params['n_proc'] if 'n_proc' in gui_params else 1
    n_proc = 1 # default n_proc
    if 'autofe_system_attr' in gui_params:
        if 'n_proc' in gui_params['autofe_system_attr']:
            n_proc = gui_params['autofe_system_attr']['n_proc'] # update n_proc if exists in metadata.json
    allocated_proc = 0
    chunk_list = [] # final chunk list according to the n_proc
    base_has_header = False
    temp_has_header = False
    ######################
    dataset = []
    # for a base csv
    if 'ml_file_path' in gui_params:
        if 'ml_file_name' in gui_params:
            base_filepath = os.path.join(gui_params['ml_file_path'], gui_params['ml_file_name'])
            if 'n_rows' in gui_params:
                if 'has_header' in gui_params:
                    base_n_rows = gui_params['n_rows']
                    base_has_header = bool(gui_params['has_header'])
            else:
                base_csv_shape, base_has_header = get_csv_shape(base_filepath) # (rows, cols)
                base_n_rows = base_csv_shape[0]
            dataset.append((base_filepath, base_n_rows, base_has_header)) # dataset[0] is the base csv
    ######################
    # for additional csvs (if exists)
    if 'additional_files' in gui_params:
        for i in range(len(gui_params['additional_files'])):
            if 'ml_file_path' in gui_params['additional_files'][i]:
                if 'ml_file_name' in gui_params['additional_files'][i]:
                    temp_filepath = os.path.join(gui_params['additional_files'][i]['ml_file_path'], gui_params['additional_files'][i]['ml_file_name'])
                    if 'n_rows' in gui_params['additional_files'][i]:
                        if 'has_header' in gui_params['additional_files'][i]:
                            temp_n_rows = gui_params['additional_files'][i]['n_rows']
                            temp_has_header = bool(gui_params['additional_files'][i]['has_header'])
                    else:
                        temp_csv_shape, temp_has_header = get_csv_shape(temp_filepath) # (rows, cols)
                        temp_n_rows = temp_csv_shape[0]
                    dataset.append((temp_filepath, temp_n_rows, temp_has_header))
    ######################
    # check primitives that CANNOT run under dask environments
    if "autofe_system_attr" in gui_params:
        agg   = gui_params['autofe_system_attr']['aggregation_primitives'] if 'aggregation_primitives' in gui_params['autofe_system_attr'] else []
        trans = gui_params['autofe_system_attr']['transformation_primitives'] if 'transformation_primitives' in gui_params['autofe_system_attr'] else []
        if n_proc == 1:
            if dataset:
                for d in dataset:
                    chunk_list.append(
                        (d[0], d[2], (0,d[1]-1), (0,d[1]-1), 0, agg, trans)
                        )
        else:
            primitives = agg+trans
            dask_compatible_false_primitives_list = get_dask_compatible_false_primitives_list(primitives)
            #
            if dask_compatible_false_primitives_list:
                n_proc = n_proc - 1 # one process for dask false primitivies and other processes for the rest
                dask_false_agg = [x for x in agg if x in dask_compatible_false_primitives_list]
                dask_false_trans = [x for x in trans if x in dask_compatible_false_primitives_list]
                #
                dask_true_agg = [x for x in agg if x not in dask_compatible_false_primitives_list]
                dask_true_trans  = [x for x in trans if x not in dask_compatible_false_primitives_list]
                if dataset:
                    for d in dataset:
                        ################################
                        # for dask_false
                        if dask_false_agg + dask_false_trans:
                            chunk_list.append(
                                (d[0], d[2], (0,d[1]-1), (0,d[1]-1), allocated_proc, dask_false_agg, dask_false_trans)
                                )
                        ################################
                        # for dask_true
                        if dask_true_agg + dask_true_trans:
                            index_list = get_index_list_by_rows(d[1],n_proc)
                            for il in index_list:
                                allocated_proc += 1
                                chunk_list.append(
                                    (d[0], d[2], il, (0,d[1]-1), allocated_proc, dask_true_agg, dask_true_trans)
                                    )
                        #################################
                        allocated_proc = 0 # reset
    res = pd.DataFrame(chunk_list, columns=['filepath', 'has_header', 'index_range', 'full_range', 'processor', 'agg', 'trans'])
    return res

def get_subcollist_by_cols(df,n_jobs, methods):
    import featuretools as ft
    plist = ft.primitives.list_primitives()
    plist_dask_false_all_list = plist[plist['dask_compatible']==False]
    pfalse_agg = plist_dask_false_all_list[plist_dask_false_all_list['type']=='aggregation']
    pfalse_trans = plist_dask_false_all_list[plist_dask_false_all_list['type']=='transform']
    pfalse_agg_names = pfalse_agg['name'].to_list()
    pfalse_trans_names = pfalse_trans['name'].to_list()
    pfalse_all_list = pfalse_agg_names + pfalse_trans_names
    #pfalse_all_list = ['cum_sum','diff','cum_min','percentile','cum_max','cum_mean','haversine','latitude','longitude','cum_count','time_since_previous','first','last','trend','skew','median','n_most_common','mode','time_since_last','time_since_first','avg_time_between','entropy']
    #
    plist_dask_true_all_list = plist[plist['dask_compatible']==True]
    ptrue_agg = plist_dask_true_all_list[plist_dask_true_all_list['type']=='aggregation']
    ptrue_trans = plist_dask_true_all_list[plist_dask_true_all_list['type']=='transform']
    ptrue_agg_names = ptrue_agg['name'].to_list()
    ptrue_trans_names = ptrue_trans['name'].to_list()
    ptrue_all_list = ptrue_agg_names + ptrue_trans_names
    #
    by_cols_only_methods = [x for x in methods if x in pfalse_all_list]
    if not by_cols_only_methods:
        return None
    else:
        pass

def getInputOutputColumnNames(gui_params):
    output_col = ""
    input_cols = []
    input_index_list = []
    output_index = ""
    for index in gui_params['output_columns_index_and_name']:
        output_col = gui_params['output_columns_index_and_name'][index]
        output_index = int(index)
    for index in gui_params['input_columns_index_and_name']:
        input_cols.append(gui_params['input_columns_index_and_name'][index])
    return input_cols, output_col, input_index_list, output_index
    
##################################################

def getDatatypeDict():
    import pandas as pd
    gui_params=load_metadata()
    #n_proc = gui_params['n_proc'] if 'n_proc' in gui_params else 1
    n_proc = 1 # default n_proc
    if 'autofe_system_attr' in gui_params:
        if 'n_proc' in gui_params['autofe_system_attr']:
            n_proc = gui_params['autofe_system_attr']['n_proc'] # update n_proc if exists in metadata.json
    allocated_proc = 0
    chunk_list = [] # final chunk list according to the n_proc
    base_has_header = False
    temp_has_header = False
    ######################
    dataset = []
    # for a base csv
    if 'ml_file_path' in gui_params:
        if 'ml_file_name' in gui_params:
            base_filepath = os.path.join(gui_params['ml_file_path'], gui_params['ml_file_name'])
            if 'n_rows' in gui_params:
                if 'has_header' in gui_params:
                    base_n_rows = gui_params['n_rows']
                    base_has_header = bool(gui_params['has_header'])
            else:
                base_csv_shape, base_has_header = get_csv_shape(base_filepath) # (rows, cols)
                base_n_rows = base_csv_shape[0]
            dataset.append((base_filepath, base_n_rows, base_has_header)) # dataset[0] is the base csv
    ######################
    # for additional csvs (if exists)
    if 'additional_files' in gui_params:
        for i in range(len(gui_params['additional_files'])):
            if 'ml_file_path' in gui_params['additional_files'][i]:
                if 'ml_file_name' in gui_params['additional_files'][i]:
                    temp_filepath = os.path.join(gui_params['additional_files'][i]['ml_file_path'], gui_params['additional_files'][i]['ml_file_name'])
                    if 'n_rows' in gui_params['additional_files'][i]:
                        if 'has_header' in gui_params['additional_files'][i]:
                            temp_n_rows = gui_params['additional_files'][i]['n_rows']
                            temp_has_header = bool(gui_params['additional_files'][i]['has_header'])
                    else:
                        temp_csv_shape, temp_has_header = get_csv_shape(temp_filepath) # (rows, cols)
                        temp_n_rows = temp_csv_shape[0]
                    dataset.append((temp_filepath, temp_n_rows, temp_has_header))

def getColumnNamesandVariableTypes(gui_params, targetfilename=None):
    input_columns_index_and_name = {}
    output_columns_index_and_name = {}
    datatype_of_columns = {}
    if 'ml_file_path' in gui_params:
        if 'ml_file_name' in gui_params:
            filepath = os.path.join(gui_params['ml_file_path'], gui_params['ml_file_name'])
            if not targetfilename: 
                targetfilename = filepath
            if targetfilename == filepath:
                if "input_columns_index_and_name" in gui_params:
                    input_columns_index_and_name=gui_params["input_columns_index_and_name"]
                if "output_columns_index_and_name" in gui_params:
                    output_columns_index_and_name=gui_params["output_columns_index_and_name"]
                if "datatype_of_columns" in gui_params:
                    datatype_of_columns = gui_params['datatype_of_columns']
    return input_columns_index_and_name, output_columns_index_and_name, datatype_of_columns

def recursiveFindColumnNamesandVariableTypes(gui_params, targetfilename):
    ic = {}
    oc = {}
    vt = {}
    ic, oc, vt = getColumnNamesandVariableTypes(gui_params, targetfilename)
    if not ic:
        #if not vt:
            if 'additional_files' in gui_params:
                for each_file in gui_params['additional_files']:
                    ic, oc, vt = getColumnNamesandVariableTypes(each_file, targetfilename)
                    if ic:
                        #if vt:
                        break
    return ic, oc, vt

def getColumnNameforSpecificType(ic,vt,stype="Index", filepath=""):
    key_index = -1
    for key, values in vt.items():
        if values == stype:
            key_index = key
            break
    if key_index == -1:
        #raise ValueError("'"+stype+"' type cannot be founded.")
        print("'"+stype+"' type cannot be founded in "+str(filepath)+".")
        return None,None
    else:
        for key, values in ic.items():
            if key == key_index:
                return key, values    

def getFeaturetoolsVariableTypesDict(ic, vt, should_make_index, index_name):
    import featuretools as ft
    default_datatype_list = ['Index','DateIndex',
                             'Id','Numeric','Category', 'String', 'Date', 'Boolean']
    vtypes={}
    if len(ic) == len(vt):
        ## add index datatype
        if should_make_index:
            vtypes.update({index_name:ft.variable_types.variable.Index})
        #
        #for i in range(len(ic)):
        #    cur_column_name = ic[str(i)] # column_name
        #    cur_datatype = vt[str(i)] # datatype
        for i in ic:
            cur_column_name = ic[str(i)] # column_name
            cur_datatype = vt[str(i)] # datatype
            if cur_datatype not in default_datatype_list:
                raise ValueError("'"+cur_datatype+"' is not in the default datatype list ['Index','DateIndex','Id','Numeric','Category', 'String', 'Date', 'Boolean']. Please check metadata.json")
            if cur_datatype == 'Index':
                values = ft.variable_types.variable.Index
            elif cur_datatype == 'DateIndex':
                values = ft.variable_types.variable.DatetimeTimeIndex                
            elif cur_datatype == 'Id':
                values = ft.variable_types.variable.Id
            elif cur_datatype == 'Numeric':
                values = ft.variable_types.variable.Numeric
            elif cur_datatype == 'Category':
                values = ft.variable_types.variable.Categorical
            elif cur_datatype == 'String':
                values = ft.variable_types.variable.Text
            elif cur_datatype == 'Date':
                values = ft.variable_types.variable.Datetime
            elif cur_datatype == 'Boolean':
                values = ft.variable_types.variable.Boolean
            vtypes.update({cur_column_name:values})
        return vtypes
    else:
        raise ValueError("# of column names != # of datatypes, please check metadata.json")

def AutoFeatureGeneration(datasetlist, methods, gui_params, current_group_no):
    #print("*** DO GENERATION !! ", len(datasetlist), methods," *** in group " , str(current_group_no))
    import featuretools as ft
    import os
    import pandas as pd
    if "autofe_system_attr" in gui_params:
        if "title" in gui_params['autofe_system_attr']:
            title = gui_params['autofe_system_attr']['title']
        else:
            title = ""
    es = ft.EntitySet(id=title)
    #################################### 1. Entity load
    y = pd.Series()
    for each_df in datasetlist:
        df = each_df[1]
        ic, oc, vt=recursiveFindColumnNamesandVariableTypes(gui_params,each_df[0]['filepath'])
        if oc:
            #print(each_df[0]['filepath'], oc)
            for k, y_colname in oc.items():
                y = each_df[1][y_colname].copy()
                y_original_filepath = each_df[0]['filepath']
                df = each_df[1].drop(columns=[y_colname])         #### split target column if exists
        #
        index_key, index_name = getColumnNameforSpecificType(ic, vt, "Index", each_df[0]['filepath'])
        if index_key: # Entities with a unique index
            make_index = False
        else:
            make_index = True
            index_name = each_df[0]['filepath']+'_index'
        vtypes = getFeaturetoolsVariableTypesDict(ic, vt, make_index, index_name)  # get variable types dict by using datasetlist and gui_params
        ####
        es = es.entity_from_dataframe(entity_id=each_df[0]['filepath'], dataframe=df,
            make_index=make_index,
            index=index_name,
            variable_types=vtypes)
    #################################### 2. Add Relationships
    if "autofe_system_attr" in gui_params:
        if 'relationships' in gui_params['autofe_system_attr']:
            if len(gui_params['autofe_system_attr'])>0:
                relationships = []
                for each in gui_params['autofe_system_attr']['relationships']:
                    if 'parent' and 'child' in each:
                        relationships.append(ft.Relationship(es[each['parent'][0]][each['parent'][1]], es[each['child'][0]][each['child'][1]]))
                es = es.add_relationships(relationships)
    ##################################### 3. Do Deep Feature Synthesis
    fm, features = ft.dfs(entityset=es, target_entity=datasetlist[0][0]['filepath'],
                          agg_primitives=methods[0],
                          trans_primitives=methods[1],
                          where_primitives=[], seed_features=[],
                          max_depth=2, verbose=0)
    try:
        outputfilepath=os.path.join("./", "fm_"+title+"__G"+str(current_group_no)+".csv")
        fm.to_csv(outputfilepath, index=True)
        os.chmod(outputfilepath, 0o776)
        return True
    except:
        return False

def mergeAllSubgroupCSVs(gui_params):
    # find current exist groups
    # find all subgroups
    # merge and return df
    import pandas as pd
    import os
    if "autofe_system_attr" in gui_params:
        if "title" in gui_params['autofe_system_attr']:
            title = gui_params['autofe_system_attr']['title']
        else:
            title = ""
    targetdir='./'
    targetdir= targetdir+os.sep
    calculated_fms = [x for x in os.listdir(targetdir) if x.startswith('fm_'+title) if not x.endswith("__mergedAll.csv")]
    # divide into subgroups (same header)
    headers = {}
    for each in calculated_fms:
        with open(each, "r") as f:
            header = f.readline()
            headers.update({each:header})
    #
    flipped={}
    for key, value in headers.items():
        if value not in flipped:
            flipped[value] = [key]
        else:
            flipped[value].append(key)
    #
    each_sub_group_with_same_header = []
    for key,value in flipped.items():
        each_sub_group_with_same_header.append(value)
    #
    ic, oc, vt = getColumnNamesandVariableTypes(gui_params)
    idx_col_name = ""
    idx_col_num, idx_col_name = getColumnNameforSpecificType(ic,vt)
    #
    subgroup_df_list=[]
    for each_sub in each_sub_group_with_same_header:
        if len(each_sub)>1:
            combined_csv = pd.concat( [ pd.read_csv(f) for f in each_sub ] )
            if idx_col_name:
                combined_csv=combined_csv.sort_values(by=[idx_col_name])
                #combined_csv.index=pd.Index(range(len(combined_csv)))
                combined_csv=combined_csv.set_index(idx_col_name)
            subgroup_df_list.append(combined_csv)
        elif len(each_sub)==1:
            get_one_csv = pd.read_csv(each_sub[0])
            if idx_col_name:
                get_one_csv=get_one_csv.set_index(idx_col_name)
            subgroup_df_list.append(get_one_csv)    
    ##
    subgroup_df_rows = [len(x) for x in subgroup_df_list]
    max_rows_subg_index = subgroup_df_rows.index(max(subgroup_df_rows))
    base_df = subgroup_df_list[max_rows_subg_index]
    for i in range(0,len(subgroup_df_list)):
        if i != max_rows_subg_index:
            base_df=base_df.merge(subgroup_df_list[i],how='left').set_axis(base_df.index)
    #
    try:
        outputfilepath=os.path.join("./", "fm_"+title+"__mergedAll.csv")
        base_df.to_csv(outputfilepath, index=True)
        os.chmod(outputfilepath, 0o776)
        return True
    except:
        return False
############ 20210617
def getSubgroupNameListwithSameHeader(gui_params):
    # find current exist groups
    # find all subgroups
    # merge and return df
    import pandas as pd
    import os
    if "autofe_system_attr" in gui_params:
        if "title" in gui_params['autofe_system_attr']:
            title = gui_params['autofe_system_attr']['title']
        else:
            title = ""
    targetdir='./'
    targetdir= targetdir+os.sep
    calculated_fms = [x for x in os.listdir(targetdir) if x.startswith('fm_'+title) if not x.endswith("__mergedAll.csv") if not x.endswith("__output_list.csv")]
    # divide into subgroups (same header)
    headers = {}
    for each in calculated_fms:
        with open(each, "r") as f:
            header = f.readline()
            headers.update({each:header})
    #
    flipped={}
    for key, value in headers.items():
        if value not in flipped:
            flipped[value] = [key]
        else:
            flipped[value].append(key)
    #
    each_sub_group_with_same_header = []
    for key,value in flipped.items():
        each_sub_group_with_same_header.append(value)
    #
    return each_sub_group_with_same_header

##################################################################
##################################################################
class ThreadingforFeatureEngineeringRank0(object):
    def __init__(self, gui_params, data_chunk_df, comm, tags, max_sec, timeout_margin=20):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.gui_params = gui_params
        self.data_chunk_df = data_chunk_df
        self.remaining_data_chunk_df = data_chunk_df
        self.comm = comm
        self.tags = tags
        self.max_sec = max_sec
        self.timeout_margin = timeout_margin
        self.timeout = False
        self.elapsed_time = 0.0
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution
    def run(self):
        closed_workers = 0
        num_workers = self.comm.size
        begin_time = time.time()
        while closed_workers < num_workers -1 :
            # resource allocation (processor acquisition @ READY status)
            status = MPI.Status()
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            self.elapsed_time = time.time() - begin_time
            #if elapsed_time > self.max_sec: # do until max_sec considering merging time
            #    self.comm.Abort() #MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD with errorcode 0.
            #                      #NOTE: invoking MPI_ABORT causes Open MPI to kill all MPI processes.You may or may not see output from other processes,
            #                      #depending on exactly when Open MPI kills them.
            if tag == self.tags.READY:
                if self.elapsed_time < (self.max_sec - self.timeout_margin): # do until (max_sec-margin)
                    if len(self.remaining_data_chunk_df)>0:
                        specific_data_chunk_to_consume, the_rest = get_a_data_chunk_per_group(self.remaining_data_chunk_df)
                        self.remaining_data_chunk_df = the_rest
                        self.comm.send(specific_data_chunk_to_consume, dest=source, tag=self.tags.START) # allow to train (1)
                    else:
                        if source != 0:
                            self.comm.send(None, dest=source, tag=self.tags.EXIT) # allow to train (1)
                else:
                    #print(":::: TIMEOVER ::::  elapsed_time < (max_sec - timeout_margin)", elapsed_time, self.max_sec, self.timeout_margin)
                    #if source !=0:
                    for i in range(1, self.comm.size):
                        self.comm.send(None, dest=i, tag=self.tags.EXIT) # stop all except rank 0
                    ## merge csvs when work finished due to timeout
                    #if mergeAllSubgroupCSVs(self.gui_params):
                    self.timeout = True
                    ##
            elif tag == self.tags.DONE:
                print("[DONE] processor ",source," finished work!")
            elif tag == self.tags.EXIT:
                closed_workers += 1
                print("***CLOSEDWORKERS******************************* = ", closed_workers, num_workers)
            time.sleep(0.5)
        # finally rank 0 will be terminated
        #if mergeAllSubgroupCSVs(self.gui_params):
        #   if self.timeout:
        #       print("Subgroup CSVs generated before 'timeout' are merged successfully.")
        #   else:
        #       print("All Subgroup CSVs are merged successfully.")
        ### file save for elapsed time
        self.comm.send(None, dest=0, tag=self.tags.EXIT)

def autofe_mpi(metadata_filename):
    # Initializations and preliminaries
    allocated_fnc=0
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    tags = enum('READY', 'DONE', 'EXIT', 'START')
    #########################################################################
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    gui_params = load_metadata(metadata_filename)
    #max_sec = gui_params['time_deadline_sec'] if 'time_deadline_sec' in gui_params else 3600 # default: max 1 hour
    max_sec = 3600 # max_sec n_proc
    if 'autofe_system_attr' in gui_params:
        if 'time_deadline_sec' in gui_params['autofe_system_attr']:
            max_sec = gui_params['autofe_system_attr']['time_deadline_sec'] # update time_deadline_sec if exists in metadata.json
    if rank == 0:
        print("** Calculating partial index range (i.e., chunk) for each processor according to the type of primitives ... **")
        data_chunk_df = get_data_chunk_by_metadata(gui_params) # 이부분을 직접 돌리던지 파일 읽어와서 이어 작업하기 구현
        print("*** Index ranges calculated successfully. **")
        provider = ThreadingforFeatureEngineeringRank0(gui_params, data_chunk_df, comm, tags, max_sec)        
    name = MPI.Get_processor_name()
    #gpu_no = abs((rank-1)%2-1)
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)#str(rank - 1)
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        specific_data_chunk_to_consume = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == tags.START:
            # Do the work here
            print(">> Process (rank %d) on %s is running.." % (rank,name))
            datasetlist, methods, current_group_no = data_loader(specific_data_chunk_to_consume, rank)
            res = AutoFeatureGeneration(datasetlist, methods, gui_params, current_group_no)
            #
            if res:
                comm.send(None, dest=0, tag=tags.READY)
        elif tag == tags.EXIT:
            if rank != 0:
                print(">> Process (rank %d) on %s will waiting other process.." % (rank,name))
            else:
                print(">> All Process DONE controlled by the (rank 0) worker as well as the scheduler")
            break
        else:
            pass
    comm.send(None, dest=0, tag=tags.EXIT)
    if rank==0:
        return provider.elapsed_time
    else:
        return 0.0
####################################################

def merge_csvs(each_sub, rank):
    import pandas as pd
    import time
    while True:
        try:
            combined_csv = pd.concat( [ pd.read_csv(f) for f in each_sub ] )
            print("*** Successfully loaded on Process("+ str(rank) +") ... ***")
            break
        except:
            print("*** There might be an Memory Error! Try again after a few seconds ... ***")
            time.sleep(2)
    return combined_csv

def fast_flatten(input_list):
    return list(chain.from_iterable(input_list))

class ThreadingforMergeCSVsRank0_multiple(object):
    def __init__(self, gui_params, each_sub, comm, tags, max_sec):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.gui_params = gui_params
        self.each_sub = each_sub
        self.comm = comm
        self.tags = tags
        self.max_sec = max_sec
        self.each_sub_split = None
        self.merged_df = None
        self.finished = False
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution
    def run(self):
        closed_workers = 0
        num_workers = self.comm.size
        begin_time = time.time()
        #
        self.each_sub_split = np.array_split(self.each_sub, num_workers) # might be an memory issue 20210621 --> nested pool considering file sizes?
        assigned_each_sub_index = -1
        #
        loaded_df = []
        ic, oc, vt = getColumnNamesandVariableTypes(self.gui_params)
        idx_col_name = ""
        idx_col_num, idx_col_name = getColumnNameforSpecificType(ic,vt)
        while closed_workers < num_workers -1 :
            # resource allocation (processor acquisition @ READY status)
            status = MPI.Status()
            combined_csv = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            elapsed_time = time.time() - begin_time
            if tag == self.tags.READY:
                if elapsed_time < self.max_sec: # do until (max_sec)
                    assigned_each_sub_index += 1
                    if assigned_each_sub_index < num_workers:
                        self.comm.send(self.each_sub_split[assigned_each_sub_index], dest=source, tag=self.tags.START) # allow to train (1)
                    else:
                        if source != 0:
                            self.comm.send(None, dest=source, tag=self.tags.EXIT) # allow to train (1)
                else:
                    for i in range(1, self.comm.size):
                        self.comm.send(None, dest=i, tag=self.tags.EXIT) # stop all except rank 0
            elif tag == self.tags.DONE:
                print("[DONE] processor ",source," finished work!")
                if idx_col_name:
                    combined_csv=combined_csv.sort_values(by=[idx_col_name])
                    #combined_csv.index=pd.Index(range(len(combined_csv)))
                    combined_csv=combined_csv.set_index(idx_col_name)
                #print("combined_df!!!*********",combined_csv)
                loaded_df.append(combined_csv)
                #if sum([sum(x.memory_usage()) for x in loaded_df])>1000000000
            elif tag == self.tags.EXIT:
                closed_workers += 1
                print("***CLOSEDWORKERS******************************* = ", closed_workers, num_workers)
            time.sleep(0.5)
        # finally rank 0 will be terminated
        print("making merge on threading...")
        #########################################################
        #1) self.merged_df = pd.concat(loaded_df) ### 20210621 slow method -->
        self.merged_df = pd.concat(loaded_df)
        try:
            title = 'MultipleGroup'
            outputfilepath=os.path.join("./", "fm_"+title+"_mergedAll.csv")
            self.merged_df.to_csv(outputfilepath, index=True)
            os.chmod(outputfilepath, 0o776)
        except:
            print(">>"+title+") file cannot be generated. Please check!")
        #########################################################
        #2) new method belows --> more slow than just concat
        #COLUMN_NAMES = loaded_df[0].columns
        #df_dict = dict.fromkeys(COLUMN_NAMES, [])
        #for col in COLUMN_NAMES:
        #    extracted = (each_loaded_df[col] for each_loaded_df in loaded_df)
        #    df_dict[col] = fast_flatten(extracted)
        #self.merged_df = pd.DataFrame.from_dict(df_dict)[COLUMN_NAMES]
        #########################################################
        #
        self.finished = True
        self.comm.send(None, dest=0, tag=self.tags.EXIT)

def fast_df_append_with_same_cols(df_list):
    COLUMN_NAMES = df_list[0].columns
    df_dict = dict.fromkeys(COLUMN_NAMES, [])
    for col in COLUMN_NAMES:
        extracted = (each_loaded_df[col] for each_loaded_df in df_list)
        df_dict[col] = fast_flatten(extracted)
    return pd.DataFrame.from_dict(df_dict)[COLUMN_NAMES]

class ThreadingforMergeCSVsRank0_single(object):
    def __init__(self, gui_params, each_sub, max_sec):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.gui_params = gui_params
        self.each_sub = each_sub
        self.max_sec = max_sec
        self.merged_df = None
        self.finished = False
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution
    def run(self):
        begin_time = time.time()
        #
        if self.max_sec > 0: # do until (max_sec)
            ic, oc, vt = getColumnNamesandVariableTypes(self.gui_params)
            idx_col_name = ""
            idx_col_num, idx_col_name = getColumnNameforSpecificType(ic,vt)
            self.merged_df = pd.read_csv(self.each_sub)
            if idx_col_name:
                self.merged_df=self.merged_df.set_index(idx_col_name)
            self.finished = True

class ThreadingforMergeCSVsRank0(object):
    def __init__(self, gui_params, comm, tags, subgroup_df, single_group_each_sub, multiple_group_each_sub, max_sec):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.gui_params = gui_params
        self.comm = comm
        self.tags = tags
        self.subgroup_df = subgroup_df
        self.single_group_each_sub = single_group_each_sub
        self.multiple_group_each_sub = multiple_group_each_sub
        self.max_sec = max_sec
        self.parallelable_false_sugbroup_list = []
        self.parallelable_true_sugbroup_list = []
        self.df = None
        self.update_required = False
        self.title = ""
        self.idx_col_name = ""
        self.finished_job =[]
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution
    def run(self):
        #pd.read_csv(gui)
        ic, oc, vt = getColumnNamesandVariableTypes(self.gui_params)
        idx_col_name = ""
        idx_col_num, idx_col_name = getColumnNameforSpecificType(ic,vt)
        self.idx_col_name = idx_col_name
        begin_time = time.time()
        if "autofe_system_attr" in self.gui_params:
            if "title" in self.gui_params['autofe_system_attr']:
                self.title = self.gui_params['autofe_system_attr']['title']
        else:
            self.title = ""
        ###################################
        if self.single_group_each_sub:
            ## parallelable 한지 검사하고, 전체 subgroup이 1개만 있는 경우인지 조사
            self.unique_group_list = list(self.subgroup_df['group_no'].unique())
            if len(self.subgroup_df['group_no'].unique())>1: # row가 2개이상이고 (single job이 아니고..)
                self.parallelable_false_sugbroup_list = list(self.subgroup_df[self.subgroup_df['parallelable']==False]['group_no'].unique())
                self.parallelable_true_sugbroup_list = list(self.subgroup_df[self.subgroup_df['parallelable']==True]['group_no'].unique())
                if len(self.parallelable_false_sugbroup_list)>0: #병렬실행불가한것들이 있다면
                    self.update_required = True
            if self.update_required:
                print("# Thread 0 is loading for update...")
                self.df = pd.read_csv(self.single_group_each_sub)
                print("# Finished to load single group df for update to others...")
                #
                if self.idx_col_name:
                    self.df=self.df.set_index(idx_col_name)
                #self.finished = True
                # multiple group에 직접 assign해서 update해서 파일을 쓰도록 유도
                closed_workers = 0
                num_workers = self.comm.size
                begin_time = time.time()
                #
                for each in self.parallelable_false_sugbroup_list:
                    self.finished_job.append((each, "(no file) merged to other groups"))
                while closed_workers < num_workers -1 :
                    # resource allocation (processor acquisition @ READY status)
                    status = MPI.Status()
                    reveiced_data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    source = status.Get_source()
                    tag = status.Get_tag()
                    elapsed_time = time.time() - begin_time
                    if tag == self.tags.READY:
                        if elapsed_time < self.max_sec: # do until (max_sec)
                            if len(self.parallelable_true_sugbroup_list)>0:
                                success_poped = False
                                target_subgroup = -1
                                target_filename = ""
                                while not success_poped:
                                    target_subgroup = self.parallelable_true_sugbroup_list.pop(0)
                                    target_filename = 'fm_'+self.title+'__G'+str(target_subgroup)+'.csv'
                                    if target_filename in self.multiple_group_each_sub:
                                        success_poped = True
                                target_g = self.subgroup_df[(self.subgroup_df['group_no']==target_subgroup)]
                                origin_f = self.gui_params['ml_file_path']+self.gui_params['ml_file_name']
                                target_g_f = target_g[target_g['filepath']==origin_f]
                                index_range=literal_eval(target_g_f['index_range'].iloc[0])
                                data_csv = self.df.iloc[index_range[0]:index_range[1]+1]
                                ##
                                self.finished_job.append((target_subgroup ,target_filename))
                                self.comm.send([data_csv, target_subgroup, target_filename], dest=source, tag=self.tags.START) # allow to train (1)
                            else: # empty subgrouplist
                                if source != 0:
                                    self.comm.send(None, dest=source, tag=self.tags.EXIT) # allow to train (1)
                        else: # time out
                            for i in range(1, self.comm.size):
                                self.comm.send(None, dest=i, tag=self.tags.EXIT) # stop all except rank 0
                    elif tag == self.tags.DONE:
                        print("[DONE] processor ",source,"(update) finished work!")
                    elif tag == self.tags.EXIT:
                        closed_workers += 1
                        print("***CLOSEDWORKERS******************************* = ", closed_workers, num_workers)
                    time.sleep(0.5)
                # finally rank 0 will be terminated
                print(">>> (UPDATE) History generated as "+'fm_'+self.title+'__output_list.csv')
                res = pd.DataFrame(self.finished_job, columns = ['finished_subgroup','filename'])
                outputfilepath=os.path.join("./",'fm_'+self.title+'__output_list.csv')
                res.to_csv(outputfilepath, index=False)
                os.chmod(outputfilepath, 0o776)
                #self.comm.send(None, dest=0, tag=self.tags.EXIT)
                #########################################################
            else: 
                # 1. 단일 데이터 그룹의 파일이 있는데({G0}),
                # 2. 해당 데이터를 이용한 업데이트가 필요없다면, 타겟이 되는 컬럼이 같은 복수의 데이터 그룹(Multiple_group_each_sub)자체가 아예 존재하지 않는
                # 3. single-job 형태일 것이므로, 아래와 같이 직접 처리
                for i in range(1, self.comm.size):
                    self.comm.send(None, dest=i, tag=self.tags.EXIT) # stop all except rank 0
                unique_group_no=pd.Series(self.subgroup_df['group_no'].unique())
                final_output_file_names=['fm_'+self.title+'__G'+str(x)+'.csv'for x in unique_group_no]
                output_exists = []
                for x in final_output_file_names:
                    if x in self.single_group_each_sub: # 여기서 처리
                        output_exists.append(True)
                    else:
                        output_exists.append(False)
                res = pd.DataFrame()
                res['finished_subgroup'] = unique_group_no
                res['filename'] = final_output_file_names
                #res['output_exists'] = output_exists
                #res = res[res['output_exists']==True]
                print(">>> (SINGLE JOB) History generated as "+'fm_'+self.title+'__output_list.csv')
                outputfilepath=os.path.join("./",'fm_'+self.title+'__output_list.csv')
                res.to_csv(outputfilepath, index=False)
                os.chmod(outputfilepath, 0o776)
        else:
            # 1. 단일 데이터 그룹의 파일이 없다면({ }),
            # 2. 타겟이 되는 컬럼이 같은 복수의 데이터 그룹(Multiple_group_each_sub)자체 최종 결과물이므로 아래와 같이 처리
            for i in range(1, self.comm.size):
                self.comm.send(None, dest=i, tag=self.tags.EXIT) # stop all except rank 0
            unique_group_no=pd.Series(self.subgroup_df['group_no'].unique())
            final_output_file_names=['fm_'+self.title+'__G'+str(x)+'.csv'for x in unique_group_no]
            output_exists = []
            for x in final_output_file_names:
                if x in self.multiple_group_each_sub:
                    output_exists.append(True)
                else:
                    output_exists.append(False)
            res = pd.DataFrame()
            res['finished_subgroup'] = unique_group_no
            res['filename'] = final_output_file_names
            #res['output_exists'] = output_exists
            #res = res[res['output_exists']==True]
            print(">>> (NO UPDATE) History generated as "+'fm_'+self.title+'__output_list.csv')
            outputfilepath=os.path.join("./",'fm_'+self.title+'__output_list.csv')
            res.to_csv(outputfilepath, index=False)
            os.chmod(outputfilepath, 0o776)
        self.comm.send(None, dest=0, tag=self.tags.EXIT)


def merge_df_a_and_b(two_dfs):
    import pandas as pd
    base_df = two_dfs[0]
    target_df = two_dfs[1]
    res = base_df.merge(target_df,how='left').set_axis(base_df.index) # keep index
    return res

def parallelizize_join_by_pool(two_dfs, func, n_cores=16):
    import pandas as pd
    import gc
    base_df = two_dfs[0]
    target_df = two_dfs[1]
    base_only_cols = list(set(base_df.columns.tolist()+target_df.columns.tolist()) - set(target_df.columns.tolist()))
    base_df = base_df[base_only_cols]
    #gc.collect()
    #
    just_transfer_index = list(set(base_df.index.tolist()) - set(target_df.index.tolist()))
    overwapped_index = target_df.index.tolist()
    ''' base_df.loc[just_transfer_index] + base_df.loc[overwapped_index] => base_df '''
    #remaining_df = base_df.loc[just_transfer_index] # return overwapped samples only 20210621
    base_df_split = np.array_split(base_df.loc[overwapped_index], n_cores)
    del base_df
    gc.collect()
    base_df_split = [x for x in base_df_split if x.size > 0]
    splitted_pairs = [(x, target_df) for x in base_df_split]
    n_cores = min(len(base_df_split),n_cores)
    pool = Pool(n_cores)
    #df = pd.concat(pool.map(func, splitted_pairs))
    dfs = pool.map(func, splitted_pairs)
    dfs = pool.map(merge_df_a_and_b, splitted_pairs)
    dfs = [x for x in dfs if x is not None]
    dfs = pd.concat(dfs)
    if not dfs.index.is_monotonic_increasing:
        dfs = dfs.sort_index()
    pool.close()
    pool.join()
    return dfs # return overwapped samples only 20210621
    #
    ###### 20210621 ignoring samples that cannot be computed yet
    #res = pd.concat([remaining_df, dfs]) # very large .. too slow..
    #del remaining_df
    #del dfs
    #gc.collect()
    
    #return res

def mergecsv_mpi(metadata_filename, elapsed_time=0.0):
    # Initializations and preliminaries
    allocated_fnc=0
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    tags = enum('READY', 'DONE', 'EXIT', 'START')
    #########################################################################
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    gui_params = load_metadata(metadata_filename)
    #max_sec = gui_params['time_deadline_sec'] if 'time_deadline_sec' in gui_params else 3600 # default: max 1 hour
    max_sec = 3600 # max_sec n_proc
    if 'autofe_system_attr' in gui_params:
        if 'time_deadline_sec' in gui_params['autofe_system_attr']:
            max_sec = gui_params['autofe_system_attr']['time_deadline_sec '] # update time_deadline_sec if exists in metadata.json
    max_sec = max_sec - elapsed_time
    ic, oc, vt = getColumnNamesandVariableTypes(gui_params)
    idx_col_name = ""
    idx_col_num, idx_col_name = getColumnNameforSpecificType(ic,vt)
    if rank == 0:
        if "autofe_system_attr" in gui_params:
            if "title" in gui_params['autofe_system_attr']:
                title = gui_params['autofe_system_attr']['title']
        else:
            title = ""
        #group_no = gui_params['group_no'] if 'group_no' in gui_params else 1
        group_no = 1 # default group_no
        if 'autofe_system_attr' in gui_params:
            if 'group_no' in gui_params['autofe_system_attr']:
                group_no = gui_params['autofe_system_attr']['group_no '] # update group_no if exists in metadata.json
        group_no_for_save = group_no
        subgroup_df_name = os.path.join("./", title+"__chunkfor"+str(group_no_for_save)+"subGroup.csv")
        subgroup_df = pd.read_csv(subgroup_df_name)
        #
        each_sub_group_with_same_header = getSubgroupNameListwithSameHeader(gui_params)        
        single_group_each_sub = []
        multiple_group_each_sub = []
        for each_sub in each_sub_group_with_same_header:
            if len(each_sub)>1:
                #print("* Join the single group cols. into each of Multiple group files...")
                multiple_group_each_sub = each_sub
            elif len(each_sub)==1:
                #print("provider : each_sub == 1 is running...")
                print("* Single group file_name : ", each_sub[0])
                single_group_each_sub = each_sub[0]
        provider = ThreadingforMergeCSVsRank0(gui_params, comm, tags, subgroup_df, single_group_each_sub, multiple_group_each_sub, max_sec)
    name = MPI.Get_processor_name()
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        #[self.df.iloc[index_range[0]:index_range[1]+1], target_filename]
        tag = status.Get_tag()
        if tag == tags.START:
            # Do the work here
            print(">> Process (rank %d) on %s is running.." % (rank,name))
            df_merged = merge_df_a_and_b( (data[0], pd.read_csv(data[2]).set_index(idx_col_name) ) )
            if df_merged is not None:
                #try:
                outputfilepath=data[2]#+"_update"
                df_merged.to_csv(outputfilepath, index=True)
                os.chmod(outputfilepath, 0o776)
                print(str(data[1])+" -> "+str(data[2])+ "(merge done!)")
                comm.send([data[1], data[2]], dest=0, tag=tags.DONE) # merged, original + target
            comm.send(None, dest=0, tag=tags.READY)
        elif tag == tags.EXIT:
            if rank != 0:
                print(">> Process (rank %d) on %s will waiting other process.." % (rank,name))
                comm.send(None, dest=0, tag=tags.EXIT)
                break
            else:
                print(">> Merge CSVs almost DONE ! Please wait for other process..")
                break

def mergecsv_mpi_old(metadata_filename, elapsed_time=0.0):
    # Initializations and preliminaries
    allocated_fnc=0
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    tags = enum('READY', 'DONE', 'EXIT', 'START')
    #########################################################################
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    gui_params = load_metadata(metadata_filename)
    max_sec = gui_params['time_deadline_sec'] if 'time_deadline_sec' in gui_params else 3600 # default: max 1 hour
    max_sec = max_sec - elapsed_time
    if rank == 0:
        each_sub_group_with_same_header = getSubgroupNameListwithSameHeader(gui_params)
        if "autofe_system_attr" in gui_params:
            if "title" in gui_params['autofe_system_attr']:
                title = gui_params['autofe_system_attr']['title']
        else:
            title = ""
        m_provider_set = []
        s_provider_set = []
        privider_set = []
        for each_sub in each_sub_group_with_same_header:
            if len(each_sub)>1:
                print("provider : each_sub > 1 is running...")
                provider = ThreadingforMergeCSVsRank0_multiple(gui_params, each_sub, comm, tags, max_sec)
                m_provider_set.append(provider)
            elif len(each_sub)==1:
                print("provider : each_sub == 1 is running...")
                provider = ThreadingforMergeCSVsRank0_single(gui_params, each_sub[0], max_sec)
                s_provider_set.append(provider)
    name = MPI.Get_processor_name()
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        each_sub = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == tags.START:
            # Do the work here
            if each_sub.tolist() == []: # empty list ==>  n_groups < n_proc .. ==> (exists) idle process
                break
            print(">> Process (rank %d) on %s is running.." % (rank,name))
            df_merged = merge_csvs(each_sub, rank)
            #
            if df_merged is not None:
                comm.send(df_merged, dest=0, tag=tags.DONE)
            comm.send(None, dest=0, tag=tags.READY)
        elif tag == tags.EXIT:
            if rank != 0:
                print(">> Process (rank %d) on %s will waiting other process.." % (rank,name))
            else:
                print(">> Distributed subgroup loading had been finished.. please wait for merging those groups into a single dataframe (__mergedAll.csv)")
            break
        else:
            pass
    if rank==0:
        while True:
            if len(m_provider_set) == sum([provider.finished for provider in m_provider_set]): # when finished all jobs
                print(">> Almost finished.. so we merge them all ! >> ??")
                break
                #subgroup_df_list = [provider.merged_df for provider in provider_set]
                #subgroup_df_list = []
                #for provider in m_provider_set:
                #    subgroup_df_list.append(provider.merged_df)    
                #    del(provider)
                #del(m_provider_set)
                #gc.collect()
                # join m_provider_set among processors
                ########################
                #base_df = merged_df
                #subgroup_df_rows = [len(x) for x in subgroup_df_list]
                #max_rows_subg_index = subgroup_df_rows.index(max(subgroup_df_rows))
                #base_df = subgroup_df_list[max_rows_subg_index]
                #for i in range(0,len(subgroup_df_list)):
                #    if i != max_rows_subg_index:
                #        #base_df=base_df.merge(subgroup_df_list[i],how='left').set_axis(base_df.index)
                #        pairs = (base_df, subgroup_df_list[i])
                #        base_df = parallelizize_join_by_pool(pairs, merge_df_a_and_b, n_cores=16)
                #        #base_df = base_df.set_axis(base_df.index)
                #
                #try:
                #    outputfilepath=os.path.join("./", "fm_"+title+"__m_mergedAll.csv")
                #    base_df.to_csv(outputfilepath, index=True)
                #    os.chmod(outputfilepath, 0o776)
                #    return True
                #except:
                #    return False
    #    while True:
    #        if len(provider_set) == sum([provider.finished for provider in provider_set]): # when finished all jobs
    #            print(">> Almost finished.. so we merge them all !")
    #            #subgroup_df_list = [provider.merged_df for provider in provider_set]
    #            subgroup_df_list = []
    #            for provider in provider_set:
    #                subgroup_df_list.append(provider.merged_df)    
    #                del(provider)
    #            del(provider_set)
    #            gc.collect()
    #            #
    #            subgroup_df_rows = [len(x) for x in subgroup_df_list]
    #            max_rows_subg_index = subgroup_df_rows.index(max(subgroup_df_rows))
    #            base_df = subgroup_df_list[max_rows_subg_index]
    #            for i in range(0,len(subgroup_df_list)):
    #                if i != max_rows_subg_index:
    #                    base_df=base_df.merge(subgroup_df_list[i],how='left').set_axis(base_df.index)
    #            #
    #            try:
    #                outputfilepath=os.path.join("./", "fm_"+title+"__mergedAll.csv")
    #                base_df.to_csv(outputfilepath, index=True)
    #                os.chmod(outputfilepath, 0o776)
    #                return True
    #            except:
    #                return False                    
    else:
        comm.send(None, dest=0, tag=tags.EXIT)


########################################
def optuna_mpi(objective, arg):
    # Initializations and preliminaries
    allocated_fnc=0
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    tags = enum('READY', 'DONE', 'EXIT', 'START')
    #########################################################################
    seed = arg.seed + rank
    _init_seed_fix_torch(seed)#+arg.data_indice_num) # torch seed 7, 8, 9, 10, 11, 12 for each scaffold datasets
    sampler = optuna.samplers.TPESampler(seed=seed) # different sampler seed per rank 7~
    shpruner = optuna.pruners.SuccessiveHalvingPruner()
    noppruner = optuna.pruners.NopPruner()
    storage_url="postgresql://"+arg.db_id+":"+arg.db_pass+"@"+arg.db_ip+":"+arg.db_port+"/"+arg.user_name
    #########################################################################
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    if rank == 0:
        ## db check if not exists, generate db
        engine = create_engine(storage_url)
        if not database_exists(engine.url):
            create_database(engine.url)
            print(arg.user_name+" database has been generated.")
        ##
        study = optuna.create_study(sampler=sampler, pruner=shpruner,
                                    study_name=arg.study_name, storage=storage_url, load_if_exists=True,
                                    direction=arg.direction)
        provider = ThreadingforOptunaRank0(study, comm, arg, tags)        
    name = MPI.Get_processor_name()
    gpu_no = abs((rank-1)%2-1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)#str(rank - 1)
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        study = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        try:
            study.sampler = sampler
        except:
            pass
        tag = status.Get_tag()
        if tag == tags.START:
            # Do the work here
            print(">> Next Trial with (seed %d, rank %d, gpu %d) on %s." % (seed, rank,gpu_no,name))
            try: # 에러 실행 코드에 대한 예외처리
                study.optimize(objective,n_trials=1)#, timeout=2)
                comm.send(study, dest=0, tag=tags.DONE)
            except:
                pass
        elif tag == tags.EXIT:
            print(">> Process (rank %d, gpu %d) on %s will waiting other process.." % (rank,gpu_no,name))
            break
        else:
            pass#print("??????",tag)
    comm.send(None, dest=0, tag=tags.EXIT)

def stepwise_mpi_time(objective, arg, task_and_algorithm):
    # Initializations and preliminaries
    allocated_fnc=0
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    tags = enum('READY', 'DONE', 'EXIT', 'START')
    #########################################################################
    with open(arg.ss_json, "r") as __:
        task_name = task_and_algorithm[0]
        algorithm_name = task_and_algorithm[1]
        original_stepwise_params = json.load(__)[task_name][algorithm_name]
        controlled_stepwise_params = copy.deepcopy(original_stepwise_params)
        n_inner_loop = len(find_linear(original_stepwise_params))
    seed = arg.seed + rank
    _init_seed_fix_torch(seed)#+arg.data_indice_num) # torch seed 7, 8, 9, 10, 11, 12 for each scaffold datasets
    sampler = optuna.samplers.TPESampler(seed=seed) # different sampler seed per rank 7~
    shpruner = optuna.pruners.SuccessiveHalvingPruner()
    noppruner = optuna.pruners.NopPruner()
    storage_url="postgresql://"+arg.db_id+":"+arg.db_pass+"@"+arg.db_ip+":"+arg.db_port+"/"+arg.user_name # user_name == db_name
    #########################################################################
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    if rank == 0:
        ## db check if not exists, generate db
        engine = create_engine(storage_url)
        if not database_exists(engine.url):
            create_database(engine.url)
            print(arg.user_name+" database has been generated.")
        ##        
        study = optuna.create_study(sampler=sampler, pruner=shpruner,
                                    study_name=arg.study_name, storage=storage_url, load_if_exists=True,
                                    direction=arg.direction)
        provider = ThreadingforStepwiseRank0(study, comm, arg, tags, n_inner_loop, original_stepwise_params)
    name = MPI.Get_processor_name()
    gpu_no = abs((rank-1)%2-1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)#str(rank - 1)
    while True:
        controlled_stepwise_params={} # initialize
        controlled_stepwise_params_has_been_changed = False
        comm.send(None, dest=0, tag=tags.READY)
        recv_result = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if recv_result:
            study = recv_result[0]
            study.sampler = sampler
            controlled_stepwise_params = recv_result[1]
            current_step = recv_result[2]
        tag = status.Get_tag()
        if tag == tags.START:
            # Do the work here
            print(">> Next Trial with (seed %d, rank %d, gpu %d) on %s." % (seed, rank,gpu_no,name))
            try:# 에러 실행 코드에 대한 예외처리
                study, controlled_stepwise_params, controlled_stepwise_params_has_been_changed = stepwise_guided_mpi_by_time(study, objective,
                    original_stepwise_params, controlled_stepwise_params, arg.max_trials,
                        guided_importance_order, current_step, algorithm_name)#, default_params=default_params)
            except:
                pass
            if controlled_stepwise_params_has_been_changed:
                reporting_stepwise_params = controlled_stepwise_params
            else:
                reporting_stepwise_params = None
            comm.send([study,reporting_stepwise_params], dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            print(">> Process (rank %d, gpu %d) on %s will waiting other process.." % (rank,gpu_no,name))
            break
        else:
            pass
    comm.send(None, dest=0, tag=tags.EXIT)
########################################

def optuna_mpi_dobj(objective_cpu, objective_gpu, arg):
    # Initializations and preliminaries
    allocated_fnc=0
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    tags = enum('READY', 'DONE', 'EXIT', 'START')
    #########################################################################
    seed = arg.seed + rank
    _init_seed_fix_torch(seed)#+arg.data_indice_num) # torch seed 7, 8, 9, 10, 11, 12 for each scaffold datasets
    sampler = optuna.samplers.TPESampler(seed=seed) # different sampler seed per rank 7~
    shpruner = optuna.pruners.SuccessiveHalvingPruner()
    noppruner = optuna.pruners.NopPruner()
    storage_url="postgresql://"+arg.db_id+":"+arg.db_pass+"@"+arg.db_ip+":"+arg.db_port+"/"+arg.user_name # user_name == db_name
    #########################################################################
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    if rank == 0:
        ## db check if not exists, generate db
        engine = create_engine(storage_url)
        if not database_exists(engine.url):
            create_database(engine.url)
            print(arg.user_name+" database has been generated.")
        study = optuna.create_study(sampler=sampler, pruner=shpruner,
                                    study_name=arg.study_name, storage=storage_url, load_if_exists=True,
                                    direction=arg.direction)
        provider = ThreadingforOptunaRank0(study, comm, arg, tags)        
    name = MPI.Get_processor_name()
    gpu_no = abs((rank-1)%2-1)
    ###
    if rank%4 < 2: # when 0, 1 process --> cpu job
        objective = objective_cpu
        resource_type = 'CPU'
    else:          # when 2, 3 process --> gpu job
        objective = objective_gpu
        resource_type = 'GPU'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)#str(rank - 1)
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        study = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        try:
            study.sampler = sampler
        except:
            pass
        tag = status.Get_tag()
        if tag == tags.START:
            print(">> Next Trial with (seed %d, rank %d, [%d]"%(seed,rank, gpu_no), " by %s, @ %s)"%(resource_type, name))
            try: # 에러 실행 코드에 대한 예외처리
                study.optimize(objective,n_trials=1)#, timeout=2)
                comm.send(study, dest=0, tag=tags.DONE)
            except:
                pass
        elif tag == tags.EXIT:
            print(">> Process (rank %d, gpu %d) on %s will waiting other process.." % (rank,gpu_no,name))
            break
        else:
            pass#time.sleep(0.5)
    comm.send(None, dest=0, tag=tags.EXIT)