"""
  MPI role for Hyper Parameter Optimization by Jeongcheol lee (2020)
  MPI role for Automated Feature Engineering by Jeongcheol lee (2021)
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
from multiprocessing import cpu_count
from itertools import chain
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from sdroptim.searching_strategies import stepwise_get_current_step_by_time, n_iter_calculation
from sdroptim.searching_strategies import _init_seed_fix_torch
from sdroptim.searching_strategies import find_linear
from sdroptim.searching_strategies import stepwise_guided_mpi_by_time, ModObjectiveFunctionWrapper, params_sorting_by_guided_list
from gpuinfo import GPUInfo

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
        with open(filepath) as f: # UTF-8
            data=f.readlines()
    except:
        with open(filepath, encoding='ISO-8859-1') as f: # ISO-8859-1
            data=f.readlines()
    #has_header = not any(cell.isdigit() for cell in data[0].split(delimiter)) ## old version that cannot find numeric column names
    has_header = check_csv_has_header(''.join(data[0:n_sampleline]))
    cols = len(data[0].split(delimiter))
    rows = len(data)-1 if has_header else len(data)
    del data
    gc.collect()
    return (rows, cols), has_header

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

def preprocessing_dict(gui_params, target_file):
    # for a base csv
    if 'ml_file_path' in gui_params:
        if 'ml_file_name' in gui_params:
            base_filepath = os.path.join(gui_params['ml_file_path'], gui_params['ml_file_name'])
            if 'pre_processing' in gui_params:
                if base_filepath == target_file:
                    return gui_params['pre_processing']
    ######################
    # for additional csvs (if exists)
    if 'additional_files' in gui_params:
        for i in range(len(gui_params['additional_files'])):
            if 'ml_file_path' in gui_params['additional_files'][i]:
                if 'ml_file_name' in gui_params['additional_files'][i]:
                    temp_filepath = os.path.join(gui_params['additional_files'][i]['ml_file_path'], gui_params['additional_files'][i]['ml_file_name'])
                    if 'pre_processing' in gui_params['additional_files'][i]:
                        if temp_filepath == target_file:
                            return gui_params['additional_files'][i]['pre_processing']
    return None

def get_object_name(data): # deprecated
    return [x for x in globals() if globals()[x] is data][0]

def data_loader(specific_data_chunk_to_consume, processor, ordered_relationships, gui_params): # add gui_params for pre_processing 210708
    import gc
    import pandas as pd
    import numpy as np
    data_list = []
    loaded = []
    each_df_name = "each_df"
    current_group_no = specific_data_chunk_to_consume['group_no'].values[0]
    # 즉 관계가 있는데 base랑 무관하게 있다면..
    if ordered_relationships:
        # check base file relationships
        founded = False
        for each_relationship in ordered_relationships:
            if each_relationship['parent'][0] == os.path.join(gui_params['ml_file_path'],gui_params['ml_file_name']):
                founded = True
        if founded:
            for each_relationship in ordered_relationships:
                # load parent
                target_to_load = specific_data_chunk_to_consume[specific_data_chunk_to_consume['filepath']==each_relationship['parent'][0]].iloc[0]
                if target_to_load['filepath'] not in [x[0] for x in loaded]:
                    print("loading.. [G"+str(current_group_no) +"/P"+str(processor)+"] "+target_to_load['filepath']+" "+str(target_to_load['index_range'])+" / "+str(target_to_load['full_range'])+" on processor "+str(processor))
                    if target_to_load['index_range'] != (-1, -1):
                        each_df = pd.read_csv(
                            target_to_load['filepath'], 
                            skiprows=get_skiprows_for_partial_reading_csv(target_to_load['has_header'], target_to_load['index_range'], target_to_load['full_range']))
                        each_df.index=range(target_to_load['index_range'][0],target_to_load['index_range'][1]+1)
                    else:
                        each_df = pd.read_csv(
                            target_to_load['filepath'], 
                            #skiprows=get_skiprows_for_partial_reading_csv(row['has_header'], row['index_range'], row['full_range']))
                            )
                    #### filtering based on selected input columns
                    input_columns_index_and_name, output_columns_index_and_name, datatype_of_columns = getColumnNamesandVariableTypes(gui_params, target_to_load['filepath'])
                    filtered_cols = dict(input_columns_index_and_name, **output_columns_index_and_name)
                    each_df = each_df[filtered_cols.values()]
                    ####
                    ################
                    # pre-processing
                    ################
                    prep_res_dict = preprocessing_dict(gui_params, target_to_load['filepath'])
                    if prep_res_dict:
                        for k, v in prep_res_dict.items():
                            exec(each_df_name +'='+v.replace(k,each_df_name))
                    ################
                    loaded.append([target_to_load['filepath'], each_df])
                # load child
                target_to_load = specific_data_chunk_to_consume[specific_data_chunk_to_consume['filepath']==each_relationship['child'][0]].iloc[0]
                if target_to_load['filepath'] not in [x[0] for x in loaded]:
                    print("loading.. [G"+str(current_group_no) +"/P"+str(processor)+"] "+target_to_load['filepath']+" "+str(target_to_load['index_range'])+" / "+str(target_to_load['full_range'])+" on processor "+str(processor))
                    if target_to_load['index_range'] != (-1, -1):
                        each_df = pd.read_csv(
                            target_to_load['filepath'], 
                            skiprows=get_skiprows_for_partial_reading_csv(target_to_load['has_header'], target_to_load['index_range'], target_to_load['full_range']))
                        each_df.index=range(target_to_load['index_range'][0],target_to_load['index_range'][1]+1)
                    else:
                        each_df = pd.read_csv(
                            target_to_load['filepath'], 
                            #skiprows=get_skiprows_for_partial_reading_csv(row['has_header'], row['index_range'], row['full_range']))
                            )
                    #### filtering based on selected input columns
                    input_columns_index_and_name, output_columns_index_and_name, datatype_of_columns = getColumnNamesandVariableTypes(gui_params, target_to_load['filepath'])
                    filtered_cols = dict(input_columns_index_and_name, **output_columns_index_and_name)
                    each_df = each_df[filtered_cols.values()]
                    ################
                    # pre-processing
                    ################
                    prep_res_dict = preprocessing_dict(gui_params, target_to_load['filepath'])
                    if prep_res_dict:
                        for k, v in prep_res_dict.items():
                            exec(each_df_name +'='+v.replace(k,each_df_name))
                    ################
                    loaded.append([target_to_load['filepath'], each_df]) # use list for reassign instead of tuple
                # join via relation key
                #if each_relationship['parent'][1] == each_relationship['child'][1]: # do only same col_name
                #    for parent_index in range(len(loaded)):
                #        if loaded[parent_index][0]==each_relationship['parent'][0]:
                #            break
                #    for child_index in range(len(loaded)):
                #        if loaded[child_index][0]==each_relationship['child'][0]:
                #            break
                #    parent_index_values = loaded[parent_index][1][each_relationship['parent'][1]].values # all parent index
                #    child_index_values =  loaded[child_index][1][each_relationship['parent'][1]].values # all parent index
                #    shaped = list(set(parent_index_values) & set(child_index_values))
                #    child_rows_prev = loaded[child_index][1].shape[0] # rows
                #    loaded[child_index][1] = loaded[child_index][1][loaded[child_index][1][each_relationship['parent'][1]].isin(shaped)]
                #    child_rows_curr = loaded[child_index][1].shape[0] # rows
                #    print("reduced.. [G"+str(current_group_no) +"/P"+str(processor)+"] "+loaded[child_index][0]+" from "+str(child_rows_prev)+" rows to "+str(child_rows_curr)+" rows according to index relationships.")
                #    gc.collect()
                for parent_index in range(len(loaded)):
                    if loaded[parent_index][0]==each_relationship['parent'][0]:
                        break
                for child_index in range(len(loaded)):
                    if loaded[child_index][0]==each_relationship['child'][0]:
                        break
                parent_index_values = loaded[parent_index][1][each_relationship['parent'][1]].values # all parent index
                child_index_values =  loaded[child_index][1][each_relationship['parent'][1]].values # all parent index
                shaped = list(set(parent_index_values) & set(child_index_values))
                child_rows_prev = loaded[child_index][1].shape[0] # rows
                loaded[child_index][1] = loaded[child_index][1][loaded[child_index][1][each_relationship['parent'][1]].isin(shaped)]
                child_rows_curr = loaded[child_index][1].shape[0] # rows
                print("reduced.. [G"+str(current_group_no) +"/P"+str(processor)+"] "+loaded[child_index][0]+" from "+str(child_rows_prev)+" rows to "+str(child_rows_curr)+" rows according to index relationships.")
                gc.collect()                    
            #####################
            for _index, row in specific_data_chunk_to_consume.iterrows():
                founded = False
                for each_loaded in loaded:
                    if row['filepath'] == each_loaded[0]:
                        each_df = each_loaded[1]
                        founded = True
                if not founded:
                    #raise ValueError("Load failed!")
                    pass # ignore redundant csv files (not to load)
                else:
                    agg = row['agg']
                    trans = row['trans']
                    data_list.append((row, each_df))
            del loaded
            gc.collect()
        else: # retrieve basefile dataset only
            data_list = []
            current_group_no = specific_data_chunk_to_consume['group_no'].values[0]
            for _index, row in specific_data_chunk_to_consume.iterrows():
                if row['filepath'] == os.path.join(gui_params['ml_file_path'],gui_params['ml_file_name']):
                    print("loading.. [G"+str(current_group_no) +"/P"+str(processor)+"] "+row['filepath']+" "+str(row['index_range'])+" / "+str(row['full_range'])+" on processor "+str(processor))
                    each_df = pd.read_csv(
                        row['filepath'], 
                        skiprows=get_skiprows_for_partial_reading_csv(row['has_header'], row['index_range'], row['full_range']))
                    each_df.index=range(row['index_range'][0],row['index_range'][1]+1)
                    #### filtering based on selected input columns
                    input_columns_index_and_name, output_columns_index_and_name, datatype_of_columns = getColumnNamesandVariableTypes(gui_params, row['filepath'])
                    filtered_cols = dict(input_columns_index_and_name, **output_columns_index_and_name)
                    each_df = each_df[filtered_cols.values()]
                    ###
                    agg = row['agg']
                    trans = row['trans']
                    data_list.append((row, each_df))        
    else:
        data_list = []
        current_group_no = specific_data_chunk_to_consume['group_no'].values[0]
        for _index, row in specific_data_chunk_to_consume.iterrows():
            print("loading.. [G"+str(current_group_no) +"/P"+str(processor)+"] "+row['filepath']+" "+str(row['index_range'])+" / "+str(row['full_range'])+" on processor "+str(processor))
            each_df = pd.read_csv(
                row['filepath'], 
                skiprows=get_skiprows_for_partial_reading_csv(row['has_header'], row['index_range'], row['full_range']))
            each_df.index=range(row['index_range'][0],row['index_range'][1]+1)
            #### filtering based on selected input columns
            input_columns_index_and_name, output_columns_index_and_name, datatype_of_columns = getColumnNamesandVariableTypes(gui_params, row['filepath'])
            filtered_cols = dict(input_columns_index_and_name, **output_columns_index_and_name)
            each_df = each_df[filtered_cols.values()]
            ###
            agg = row['agg']
            trans = row['trans']
            data_list.append((row, each_df))        
    return data_list, (agg, trans), current_group_no

#datasetlist, methods, current_group_no = data_loader(specific_data_chunk_to_consume, rank, ordered_relationships, gui_params)


def data_loader_old20210706(specific_data_chunk_to_consume, processor):
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

def make_graph(relationships):
    graph={}
    for each in relationships:
        if each['parent'][0] not in graph.keys():
            graph.update({each['parent'][0]:[each['child'][0]]})
        else:
            bags = graph[each['parent'][0]]
            bags.append(each['child'][0])
            graph.update({each['parent'][0]:bags})
        #### pairwise-graph
        if each['child'][0] not in graph.keys():
            graph.update({each['child'][0]:[each['parent'][0]]})
        else:
            bags = graph[each['child'][0]]
            bags.append(each['parent'][0])
            graph.update({each['child'][0]:bags})
    return graph

def bfs(graph, start_node):
    visit = list()
    queue = list()
    queue.append(start_node)
    while queue:
        node = queue.pop(0)
        if node not in visit:
            visit.append(node)
            queue.extend(graph[node])
    return visit

def get_ordered_relationships(gui_params): # 20210901; selected range using relationships
    base_filepath = ""
    if 'ml_file_path' in gui_params:
        if 'ml_file_name' in gui_params:
            base_filepath = os.path.join(gui_params['ml_file_path'], gui_params['ml_file_name'])
    if 'autofe_system_attr' in gui_params:
        if 'relationships' in gui_params['autofe_system_attr']:
            relationships =  gui_params['autofe_system_attr']['relationships'].copy()
            ordered_relationships = []
            if relationships:
                graph = make_graph(relationships)
                bfs_order = bfs(graph, base_filepath)
                while bfs_order:
                    comp = bfs_order.pop(0)
                    for each in relationships:
                        if each['parent'][0] == comp:
                            ordered_relationships.append(each)
            return ordered_relationships

def get_data_chunk_by_metadata(gui_params, renew=False, probing=False): # renew will be True after basic development
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
    if probing:
        gui_params['autofe_system_attr']['aggregation_primitives'] = []
        if len(gui_params['input_columns_index_and_name'])<100:
            gui_params['autofe_system_attr']['transformation_primitives'] = ['add_numeric', 'divide_numeric', 'percentile', 'negate', 'absolute', 'divide_by_feature']
        else:
            gui_params['autofe_system_attr']['transformation_primitives'] = ['percentile', 'negate', 'absolute', 'divide_by_feature']
    ######################
    # check primitives that CANNOT run under dask environments
    if "autofe_system_attr" in gui_params:
        agg = gui_params['autofe_system_attr']['aggregation_primitives'] if 'aggregation_primitives' in gui_params['autofe_system_attr'] else []
        trans = gui_params['autofe_system_attr']['transformation_primitives'] if 'transformation_primitives' in gui_params['autofe_system_attr'] else []
        if agg is None:
            agg = []
        if trans is None:
            trans = []
        if agg+trans == []:
            group_no = 1 # no need to split groups for autofe
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
                        if d[0]== base_filepath:
                            index_list = get_index_list_by_rows(d[1],group_no)
                        else: ############################20210706 index_range update
                            index_list = []
                            for i in range(group_no):
                                index_list.append((-1, -1))
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

def get_fs_chunk_by_metadata(params, labels, use_original=True, use_converted=True, probing=False): # renew will be True after basic development
    gui_params=params
    chunk_list = []
    filter_based = {}
    wrapper_based = {}
    if 'autofe_system_attr' in gui_params:
        if 'feature_selection' in gui_params['autofe_system_attr']:
            fs = gui_params['autofe_system_attr']['feature_selection']
            for k, v in fs.items():
                if k.startswith("remove"):
                    filter_based.update({k:v})
                else:
                    wrapper_based.update({k:v})
            ##########
    if labels is None: # no labels
        if wrapper_based:
            print("ERROR: Wrapper-based Feature selection (GFS / GBDT) will run only with Y values (labels) in the original dataset.")
            return None, filter_based
    else:
        if probing:
            chunk_list.append(('original','GradientFeatureSelector','n_features',1))
            chunk_list.append(('converted','GradientFeatureSelector','n_features',0.25))
            chunk_list.append(('converted','GradientFeatureSelector','n_features',0.5))
            chunk_list.append(('converted','GradientFeatureSelector','n_features',0.75))
            chunk_list.append(('converted','GradientFeatureSelector','n_features',1))
            filter_based.update({'remove_low_information_features':None})
            filter_based.update({'remove_highly_null_features':{'pct_null_threshold': 0.95}})
            filter_based.update({'remove_single_value_features': {'count_nan_as_value': 0}})
            if len(gui_params['input_columns_index_and_name'])<100:
                filter_based.update({'remove_highly_correlated_features':None})
        else:
            if use_original:
                if wrapper_based:
                    for each_wrapper_algorithm, each_wrapper_algorithm_params in wrapper_based.items():
                        if each_wrapper_algorithm_params is not None: # 20211029 bug fix
                            for individual_params, its_values in each_wrapper_algorithm_params.items():
                                for value in its_values:
                                    chunk_list.append(('original',each_wrapper_algorithm,individual_params, value))
                chunk_list.append(('original',None,None,None))
            if use_converted:
                if wrapper_based:
                    for each_wrapper_algorithm, each_wrapper_algorithm_params in wrapper_based.items():
                        if each_wrapper_algorithm_params is not None: # 20211029 bug fix
                            for individual_params, its_values in each_wrapper_algorithm_params.items():
                                for value in its_values:
                                    chunk_list.append(('converted',each_wrapper_algorithm,individual_params, value))
                chunk_list.append(('converted',None,None,None))                                
        res = pd.DataFrame(chunk_list, columns=['base_df', 'wrapper','param_name','param_value']).reset_index().rename(columns={'index':'group_no'}) # add parallelable Boolean
        #outputpath = os.path.join("./", title+"__fs_chunk.csv")
        #res.to_csv(outputpath, index=False)
        #os.chmod(outputpath, 0o776)
    return res, filter_based
    
def get_data_chunk_by_metadata_old_forsave_20210706(gui_params, renew=False): # renew will be True after basic development
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

def reduce_mem_usage_df(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df    
    
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

def getColumnNamesandVariableTypes_old(gui_params, targetfilename=None):
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

def getColumnNamesandVariableTypes(gui_params, targetfilename=None): # modified 0817
    input_columns_index_and_name = {}
    output_columns_index_and_name = {}
    datatype_of_columns = {}
    if 'ml_file_path' in gui_params:
        if 'ml_file_name' in gui_params:
            base_filepath = os.path.join(gui_params['ml_file_path'], gui_params['ml_file_name'])
            if not targetfilename: 
                targetfilename = base_filepath
                base = True
            else:
                if base_filepath == targetfilename:
                    base = True
                else:
                    base = False
            ###
            if base:
                if "input_columns_index_and_name" in gui_params:
                    input_columns_index_and_name=gui_params["input_columns_index_and_name"]
                if "output_columns_index_and_name" in gui_params:
                    output_columns_index_and_name=gui_params["output_columns_index_and_name"]
                if "datatype_of_columns" in gui_params:
                    datatype_of_columns = gui_params['datatype_of_columns']
            else:
                if 'additional_files' in gui_params:
                    for i in range(len(gui_params['additional_files'])):
                        if 'ml_file_path' in gui_params['additional_files'][i]:
                            if 'ml_file_name' in gui_params['additional_files'][i]:
                                temp_filepath = os.path.join(gui_params['additional_files'][i]['ml_file_path'], gui_params['additional_files'][i]['ml_file_name'])
                                if temp_filepath == targetfilename:
                                    if "input_columns_index_and_name" in gui_params['additional_files'][i]:
                                        input_columns_index_and_name=gui_params['additional_files'][i]["input_columns_index_and_name"]
                                    if "output_columns_index_and_name" in gui_params['additional_files'][i]:
                                        output_columns_index_and_name=gui_params['additional_files'][i]["output_columns_index_and_name"]
                                    if "datatype_of_columns" in gui_params['additional_files'][i]:
                                        datatype_of_columns = gui_params['additional_files'][i]['datatype_of_columns']
                                    break
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
    base_index = None
    base_index_has_generated = False
    base_index_name = ""
    for each_df in datasetlist:
        df = each_df[1]
        ic, oc, vt=recursiveFindColumnNamesandVariableTypes(gui_params,each_df[0]['filepath'])
        if oc:
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
            index_name = os.path.basename(each_df[0]['filepath'])+'_index'
        # store base index
        if each_df[0]['filepath'] == os.path.join(gui_params['ml_file_path'],gui_params['ml_file_name']):
            base_index = each_df[1].index.copy()
            if make_index:
                # base does not have index
                base_index_has_generated = True
                base_index_name = index_name
        #
        vtypes = getFeaturetoolsVariableTypesDict(ic, vt, make_index, index_name)  # get variable types dict by using datasetlist and gui_params
        ####
        es = es.entity_from_dataframe(entity_id=os.path.basename(each_df[0]['filepath']), dataframe=df,
            make_index=make_index,
            index=index_name,
            variable_types=vtypes)
    #################################### 2. Add Relationships
    # check all relationships
    relationships_flag = False
    if "autofe_system_attr" in gui_params:
        if 'relationships' in gui_params['autofe_system_attr']:
            if len(gui_params['autofe_system_attr']['relationships'])>0:
                relationships = []
                for each in gui_params['autofe_system_attr']['relationships']:
                    if 'parent' and 'child' in each:
                        relationship_files = [each['parent'][0], each['child'][0]]
                        loaded_files = [x[0]['filepath'] for x in datasetlist]
                        if all(x in loaded_files for x in relationship_files):
                            relationships.append(ft.Relationship(es[os.path.basename(each['parent'][0])][os.path.basename(each['parent'][1])], es[os.path.basename(each['child'][0])][os.path.basename(each['child'][1])]))
                if relationships:
                    #print(relationships)
                    es = es.add_relationships(relationships)
                    relationships_flag = True
    ##################################### 3. Do Deep Feature Synthesis
    # element-wise calculation is too big to compute (make limitation on the automate process) # 2021-09-02
    #element_wise_methods = ['and','or','add_numeric','divide_numeric','substract_numeric', 'modulo_numeric','multiply_boolean']
    #n_element_wise = len([x for x in methods[0]+methods[1] if x in element_wise_methods])
    #max_depth = 2 if n_element_wise<2 else 1
    max_depth = 2 if relationships_flag else 1
    fm, features = ft.dfs(entityset=es, target_entity=os.path.basename(datasetlist[0][0]['filepath']),
                          agg_primitives=methods[0],
                          trans_primitives=methods[1],
                          where_primitives=[], seed_features=[],
                          max_depth=max_depth, verbose=0)
    ### fix index range
    fm.index = base_index
    if base_index_has_generated:
        fm.index.name = base_index_name
    fm = fm.reset_index()
    #else:
    #    fm.index.name = get_id_cols(gui_params)[0]# original base file index name instead of generated index_name
    try:
        outputfilepath=os.path.join("./", "fm_"+title+"__G"+str(current_group_no)+".csv")
        fm.to_csv(outputfilepath, index=False)
        os.chmod(outputfilepath, 0o776)
        ### save entity relationships
        return True
    except:
        print("[ERR] AutoFE did not work, so the feature matrix cannot be generated. Please check datatypes of dataframe and relationships among them.")
        return False

def AutoFeatureGeneration_old(datasetlist, methods, gui_params, current_group_no):
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
    base_index = None
    for each_df in datasetlist:
        # store base index
        if each_df[0]['filepath'] == os.path.join(gui_params['ml_file_path'],gui_params['ml_file_name']):
            base_index = each_df[1].index.copy()
        #
        df = each_df[1]
        ic, oc, vt=recursiveFindColumnNamesandVariableTypes(gui_params,each_df[0]['filepath'])
        if oc:
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
            index_name = os.path.basename(each_df[0]['filepath'])+'_index'
        vtypes = getFeaturetoolsVariableTypesDict(ic, vt, make_index, index_name)  # get variable types dict by using datasetlist and gui_params
        ####
        es = es.entity_from_dataframe(entity_id=os.path.basename(each_df[0]['filepath']), dataframe=df,
            make_index=make_index,
            index=index_name,
            variable_types=vtypes)
    #################################### 2. Add Relationships
    # check all relationships
    if "autofe_system_attr" in gui_params:
        if 'relationships' in gui_params['autofe_system_attr']:
            if len(gui_params['autofe_system_attr']['relationships'])>0:
                relationships = []
                for each in gui_params['autofe_system_attr']['relationships']:
                    if 'parent' and 'child' in each:
                        relationship_files = [each['parent'][0], each['child'][0]]
                        loaded_files = [x[0]['filepath'] for x in datasetlist]
                        if all(x in loaded_files for x in relationship_files):
                            relationships.append(ft.Relationship(es[os.path.basename(each['parent'][0])][os.path.basename(each['parent'][1])], es[os.path.basename(each['child'][0])][os.path.basename(each['child'][1])]))
                if relationships:
                    #print(relationships)
                    es = es.add_relationships(relationships)
    ##################################### 3. Do Deep Feature Synthesis
    fm, features = ft.dfs(entityset=es, target_entity=os.path.basename(datasetlist[0][0]['filepath']),
                          agg_primitives=methods[0],
                          trans_primitives=methods[1],
                          where_primitives=[], seed_features=[],
                          max_depth=2, verbose=0)
    ### fix index range
    fm.index = base_index
    if make_index:
        fm.index.name=index_name
    else:
        fm.index.name = get_id_cols(gui_params)[0]# original base file index name instead of generated index_name
    try:
        outputfilepath=os.path.join("./", "fm_"+title+"__G"+str(current_group_no)+".csv")
        fm.reset_index().to_csv(outputfilepath, index=False)
        os.chmod(outputfilepath, 0o776)
        ### save entity relationships
        return True
    except:
        print("[ERR] AutoFE did not work, so the feature matrix cannot be generated. Please check datatypes of dataframe and relationships among them.")
        return False

def data_loader(specific_data_chunk_to_consume, processor, ordered_relationships, gui_params): # add gui_params for pre_processing 210708
    import gc
    import pandas as pd
    import numpy as np
    data_list = []
    loaded = []
    each_df_name = "each_df"
    current_group_no = specific_data_chunk_to_consume['group_no'].values[0]
    if ordered_relationships:
        # check base file relationships
        founded = False
        for each_relationship in ordered_relationships:
            if each_relationship['parent'][0] == os.path.join(gui_params['ml_file_path'],gui_params['ml_file_name']):
                founded = True
        if founded:
            for each_relationship in ordered_relationships:
                # load parent
                target_to_load = specific_data_chunk_to_consume[specific_data_chunk_to_consume['filepath']==each_relationship['parent'][0]].iloc[0]
                if target_to_load['filepath'] not in [x[0] for x in loaded]:
                    print("loading.. [G"+str(current_group_no) +"/P"+str(processor)+"] "+target_to_load['filepath']+" "+str(target_to_load['index_range'])+" / "+str(target_to_load['full_range'])+" on processor "+str(processor))
                    if target_to_load['index_range'] != (-1, -1):
                        each_df = pd.read_csv(
                            target_to_load['filepath'], 
                            skiprows=get_skiprows_for_partial_reading_csv(target_to_load['has_header'], target_to_load['index_range'], target_to_load['full_range']))
                        each_df.index=range(target_to_load['index_range'][0],target_to_load['index_range'][1]+1)
                    else:
                        each_df = pd.read_csv(
                            target_to_load['filepath'], 
                            #skiprows=get_skiprows_for_partial_reading_csv(row['has_header'], row['index_range'], row['full_range']))
                            )
                    #### filtering based on selected input columns
                    input_columns_index_and_name, output_columns_index_and_name, datatype_of_columns = getColumnNamesandVariableTypes(gui_params, target_to_load['filepath'])
                    filtered_cols = dict(input_columns_index_and_name, **output_columns_index_and_name)
                    each_df = each_df[filtered_cols.values()]
                    ####
                    ################
                    # pre-processing
                    ################
                    prep_res_dict = preprocessing_dict(gui_params, target_to_load['filepath'])
                    if prep_res_dict:
                        for k, v in prep_res_dict.items():
                            exec(each_df_name +'='+v.replace(k,each_df_name))
                    ################
                    loaded.append([target_to_load['filepath'], each_df])
                # load child
                target_to_load = specific_data_chunk_to_consume[specific_data_chunk_to_consume['filepath']==each_relationship['child'][0]].iloc[0]
                if target_to_load['filepath'] not in [x[0] for x in loaded]:
                    print("loading.. [G"+str(current_group_no) +"/P"+str(processor)+"] "+target_to_load['filepath']+" "+str(target_to_load['index_range'])+" / "+str(target_to_load['full_range'])+" on processor "+str(processor))
                    if target_to_load['index_range'] != (-1, -1):
                        each_df = pd.read_csv(
                            target_to_load['filepath'], 
                            skiprows=get_skiprows_for_partial_reading_csv(target_to_load['has_header'], target_to_load['index_range'], target_to_load['full_range']))
                        each_df.index=range(target_to_load['index_range'][0],target_to_load['index_range'][1]+1)
                    else:
                        each_df = pd.read_csv(
                            target_to_load['filepath'], 
                            #skiprows=get_skiprows_for_partial_reading_csv(row['has_header'], row['index_range'], row['full_range']))
                            )
                    #### filtering based on selected input columns
                    input_columns_index_and_name, output_columns_index_and_name, datatype_of_columns = getColumnNamesandVariableTypes(gui_params, target_to_load['filepath'])
                    filtered_cols = dict(input_columns_index_and_name, **output_columns_index_and_name)
                    each_df = each_df[filtered_cols.values()]
                    ################
                    # pre-processing
                    ################
                    prep_res_dict = preprocessing_dict(gui_params, target_to_load['filepath'])
                    if prep_res_dict:
                        for k, v in prep_res_dict.items():
                            exec(each_df_name +'='+v.replace(k,each_df_name))
                    ################
                    loaded.append([target_to_load['filepath'], each_df]) # use list for reassign instead of tuple
                # join via relation key
                if each_relationship['parent'][1] == each_relationship['child'][1]: # do only same col_name
                    for parent_index in range(len(loaded)):
                        if loaded[parent_index][0]==each_relationship['parent'][0]:
                            break
                    for child_index in range(len(loaded)):
                        if loaded[child_index][0]==each_relationship['child'][0]:
                            break
                    parent_index_values = loaded[parent_index][1][each_relationship['parent'][1]].values # all parent index
                    child_index_values =  loaded[child_index][1][each_relationship['parent'][1]].values # all parent index
                    shaped = list(set(parent_index_values) & set(child_index_values))
                    child_rows_prev = loaded[child_index][1].shape[0] # rows
                    loaded[child_index][1] = loaded[child_index][1][loaded[child_index][1][each_relationship['parent'][1]].isin(shaped)]
                    child_rows_curr = loaded[child_index][1].shape[0] # rows
                    print("reduced.. [G"+str(current_group_no) +"/P"+str(processor)+"] "+loaded[child_index][0]+" from "+str(child_rows_prev)+" rows to "+str(child_rows_curr)+" rows according to index relationships.")
                    gc.collect()
            #####################
            for _index, row in specific_data_chunk_to_consume.iterrows():
                founded = False
                for each_loaded in loaded:
                    if row['filepath'] == each_loaded[0]:
                        each_df = each_loaded[1]
                        founded = True
                if not founded:
                    #raise ValueError("Load failed!")
                    pass # ignore redundant csv files (not to load)
                else:
                    agg = row['agg']
                    trans = row['trans']
                    data_list.append((row, each_df))
            del loaded
            gc.collect()
        else: # retrieve basefile dataset only
            data_list = []
            current_group_no = specific_data_chunk_to_consume['group_no'].values[0]
            for _index, row in specific_data_chunk_to_consume.iterrows():
                if row['filepath'] == os.path.join(gui_params['ml_file_path'],gui_params['ml_file_name']):
                    print("loading.. [G"+str(current_group_no) +"/P"+str(processor)+"] "+row['filepath']+" "+str(row['index_range'])+" / "+str(row['full_range'])+" on processor "+str(processor))
                    each_df = pd.read_csv(
                        row['filepath'], 
                        skiprows=get_skiprows_for_partial_reading_csv(row['has_header'], row['index_range'], row['full_range']))
                    each_df.index=range(row['index_range'][0],row['index_range'][1]+1)
                    #### filtering based on selected input columns
                    input_columns_index_and_name, output_columns_index_and_name, datatype_of_columns = getColumnNamesandVariableTypes(gui_params, row['filepath'])
                    filtered_cols = dict(input_columns_index_and_name, **output_columns_index_and_name)
                    each_df = each_df[filtered_cols.values()]
                    ###
                    agg = row['agg']
                    trans = row['trans']
                    data_list.append((row, each_df))        
    else:
        data_list = []
        current_group_no = specific_data_chunk_to_consume['group_no'].values[0]
        for _index, row in specific_data_chunk_to_consume.iterrows():
            print("loading.. [G"+str(current_group_no) +"/P"+str(processor)+"] "+row['filepath']+" "+str(row['index_range'])+" / "+str(row['full_range'])+" on processor "+str(processor))
            each_df = pd.read_csv(
                row['filepath'], 
                skiprows=get_skiprows_for_partial_reading_csv(row['has_header'], row['index_range'], row['full_range']))
            each_df.index=range(row['index_range'][0],row['index_range'][1]+1)
            #### filtering based on selected input columns
            input_columns_index_and_name, output_columns_index_and_name, datatype_of_columns = getColumnNamesandVariableTypes(gui_params, row['filepath'])
            filtered_cols = dict(input_columns_index_and_name, **output_columns_index_and_name)
            each_df = each_df[filtered_cols.values()]
            ###
            agg = row['agg']
            trans = row['trans']
            data_list.append((row, each_df))        
    return data_list, (agg, trans), current_group_no

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
        self.finished = False
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
                        self.comm.send(None, dest=source, tag=self.tags.EXIT_REQ) # allow to train (1)
                else:
                    #print(":::: TIMEOVER ::::  elapsed_time < (max_sec - timeout_margin)", elapsed_time, self.max_sec, self.timeout_margin)
                    for i in range(0, self.comm.size):
                        self.comm.send(None, dest=i, tag=self.tags.EXIT_REQ) # stop all except rank 0
                    self.timeout = True
            elif tag == self.tags.DONE:
                print("[DONE] processor ",source," finished work!")
            elif tag == self.tags.EXIT_RES:
                closed_workers += 1
                print("***CLOSEDWORKERS******************************* = ", closed_workers, num_workers)
            time.sleep(0.5)
        #################################
        print(":: AUTOFE_MPI scheduler thread terminated ::")
        self.finished=True

def autofe_mpi(metadata_filename):
    # Initializations and preliminaries
    allocated_fnc=0
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    tags = enum('READY', 'DONE', 'EXIT_REQ','EXIT_RES', 'START')
    #########################################################################
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    gui_params = load_metadata(metadata_filename)
    ordered_relationships = get_ordered_relationships(gui_params)
    #max_sec = gui_params['time_deadline_sec'] if 'time_deadline_sec' in gui_params else 3600 # default: max 1 hour
    max_sec = 28800 # max_sec n_proc
    if 'autofe_system_attr' in gui_params:
        if 'time_deadline_sec' in gui_params['autofe_system_attr']:
            max_sec = gui_params['autofe_system_attr']['time_deadline_sec'] # update time_deadline_sec if exists in metadata.json
    probing = False
    if "probing" in gui_params['autofe_system_attr']:
        if gui_params['autofe_system_attr']['probing']==True:
            probing=True
    if rank == 0:
        print("** Calculating partial index range (i.e., chunk) for each processor according to the type of primitives ... **")
        data_chunk_df = get_data_chunk_by_metadata(gui_params, renew=False, probing=probing) # 이부분을 직접 돌리던지 파일 읽어와서 이어 작업하기 구현
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
            datasetlist, methods, current_group_no = data_loader(specific_data_chunk_to_consume, rank, ordered_relationships, gui_params)
            res = AutoFeatureGeneration(datasetlist, methods, gui_params, current_group_no)
            if res:
                comm.send(None, dest=0, tag=tags.DONE)
        elif tag == tags.EXIT_REQ:
            print(">> Process (rank %d) on %s will waiting other process.." % (rank,name))
            comm.send(None, dest=0, tag=tags.EXIT_RES)
            break
    if rank == 0:
        while True:
            if provider.finished:
                print("ALL finishied")
                break
#    if rank==0:
#        return provider.elapsed_time
#    else:
#        return 0.0
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
        self.n_jobs = 0
        self.done_job = 0
        self.finished = False
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
                self.n_jobs = len(self.parallelable_true_sugbroup_list)
                if len(self.parallelable_false_sugbroup_list)>0: #병렬실행불가한것들이 있다면
                    self.update_required = True
            if self.update_required:
                print("# Thread 0 is loading for update...")
                #################################################
                if len(self.single_group_each_sub)>1:
                    to_merge_list = []
                    target_list = []
                    for i in range(len(self.single_group_each_sub)):
                        if [int(self.single_group_each_sub[i].split('__G')[1].split('.csv')[0])] == self.parallelable_false_sugbroup_list: # non-parallelable file should be merged to other file
                            to_merge_list.append(self.single_group_each_sub[i])
                        else:
                            target_list.append(self.single_group_each_sub[i])
                    if target_list:
                        target_df = pd.read_csv(target_list[0])
                        target_df_filename = target_list[0]
                        target_df_group = int(target_list[0].split('__G')[1].split('.csv')[0])
                    else:
                        target_df = pd.read_csv(to_merge_list[0])
                        target_df_filename = to_merge_list[0]
                        target_df_group = int(to_merge_list[0].split('__G')[1].split('.csv')[0])
                    for i in range(len(target_list)-1):
                        temp = pd.read_csv(target_list[i+1])
                        target_df = merge_df_a_and_b( (target_df, temp) )
                        self.finished_job.append( (int(target_list[i+1].split('__G')[1].split('.csv')[0]), "(no file) merged to other groups"))
                        if target_df is not None:
                            outputfilepath=os.path.join("./",target_df_filename)
                            target_df.to_csv(outputfilepath, index=False)
                            os.chmod(outputfilepath, 0o776)
                            print(str(target_list[i+1])+" -> "+str(target_df_filename)+ "(merge done!)")
                    for i in range(len(to_merge_list)):
                        temp = pd.read_csv(to_merge_list[i])
                        target_df = merge_df_a_and_b( (target_df, temp) )
                        self.finished_job.append( (int(to_merge_list[i].split('__G')[1].split('.csv')[0]), "(no file) merged to other groups"))
                        if target_df is not None:
                            outputfilepath=os.path.join("./",target_df_filename)
                            target_df.to_csv(outputfilepath, index=False)
                            os.chmod(outputfilepath, 0o776)
                            print(str(to_merge_list[i])+" -> "+str(target_df_filename)+ "(merge done!)")
                    if self.multiple_group_each_sub == []:
                        self.finished_job.append( (target_df_group, target_df_filename) )
                        # finally rank 0 will be terminated
                        for i in range(0, self.comm.size):
                            self.comm.send(None, dest=i, tag=self.tags.EXIT_REQ) # stop all except rank 0
                        print(">>> (UPDATE) History generated as "+'fm_'+self.title+'__output_list.csv')
                        res = pd.DataFrame(self.finished_job, columns = ['finished_subgroup','filename'])
                        outputfilepath=os.path.join("./",'fm_'+self.title+'__output_list.csv')
                        res.to_csv(outputfilepath, index=False)
                        os.chmod(outputfilepath, 0o776)
                        self.finished=True
                else:
                    self.df = pd.read_csv(self.single_group_each_sub[0])
                #################################################
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
                while closed_workers < num_workers:
                    # resource allocation (processor acquisition @ READY status)
                    status = MPI.Status()
                    reveiced_data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    source = status.Get_source()
                    tag = status.Get_tag()
                    elapsed_time = time.time() - begin_time
                    if tag == self.tags.READY:
                        if elapsed_time < self.max_sec: # do until (max_sec)
                            if (len(self.parallelable_true_sugbroup_list)>0) and (self.multiple_group_each_sub):
                                success_poped = False
                                target_subgroup = -1
                                target_filename = ""
                                while not success_poped:
                                    #print(self.parallelable_true_sugbroup_list, "****")
                                    target_subgroup = self.parallelable_true_sugbroup_list.pop(0)
                                    target_filename = 'fm_'+self.title+'__G'+str(target_subgroup)+'.csv'
                                    #print(target_filename, self.multiple_group_each_sub)
                                    if target_filename in self.multiple_group_each_sub:
                                        success_poped = True
                                target_g = self.subgroup_df[(self.subgroup_df['group_no']==target_subgroup)]
                                origin_f = os.path.join(self.gui_params['ml_file_path'],self.gui_params['ml_file_name'])
                                target_g_f = target_g[target_g['filepath']==origin_f]
                                index_range=literal_eval(target_g_f['index_range'].iloc[0])
                                data_csv = self.df.iloc[index_range[0]:index_range[1]+1]
                                ##
                                self.finished_job.append((target_subgroup ,target_filename))
                                self.comm.send([data_csv, self.single_group_each_sub[0], target_filename], dest=source, tag=self.tags.START) # allow to train (1)
                            else: # empty subgrouplist
                                self.comm.send(None,dest=source, tag=self.tags.EXIT_REQ)
                                #if source != 0:
                                #    self.comm.send(None, dest=source, tag=self.tags.EXIT) # allow to train (1)
                                #else:
                                #    if self.n_jobs == self.done_job:
                                #        print(">> Merging processes among non-parallelable csv and other csvs have done.")
                                #        closed_workers += 1
                        else: # time out
                            for i in range(0, self.comm.size):
                                self.comm.send(None, dest=i, tag=self.tags.EXIT_REQ) # stop all except rank 0
                    elif tag == self.tags.DONE:
                        self.done_job +=1
                        print("[DONE] processor ",source,"(update) finished work!")
                    elif tag == self.tags.EXIT_RES:
                        closed_workers += 1
                        print("***CLOSEDWORKERS******************************* = ", closed_workers, num_workers)
                    time.sleep(0.5)
                # finally rank 0 will be terminated
                print(">>> (UPDATE) History generated as "+'fm_'+self.title+'__output_list.csv')
                res = pd.DataFrame(self.finished_job, columns = ['finished_subgroup','filename'])
                outputfilepath=os.path.join("./",'fm_'+self.title+'__output_list.csv')
                res.to_csv(outputfilepath, index=False)
                os.chmod(outputfilepath, 0o776)
                self.finished=True
                #self.comm.send(None, dest=0, tag=self.tags.EXIT)
                #########################################################
            else: 
                # 1. 단일 데이터 그룹의 파일이 있는데({G0}),
                # 2. 해당 데이터를 이용한 업데이트가 필요없다면, 타겟이 되는 컬럼이 같은 복수의 데이터 그룹(Multiple_group_each_sub)자체가 아예 존재하지 않는
                # 3. single-job 형태일 것이므로, 아래와 같이 직접 처리
                for i in range(0, self.comm.size):
                    self.comm.send(None, dest=i, tag=self.tags.EXIT_REQ) # stop all except rank 0
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
                self.finished=True
        else:
            # 1. 단일 데이터 그룹의 파일이 없다면({ }),
            # 2. 타겟이 되는 컬럼이 같은 복수의 데이터 그룹(Multiple_group_each_sub)자체 최종 결과물이므로 아래와 같이 처리
            for i in range(0, self.comm.size):
                self.comm.send(None, dest=i, tag=self.tags.EXIT_REQ) # stop all except rank 0
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
            self.finished=True

class ThreadingforFeatureSelection(object):
    def __init__(self, gui_params, comm, tags, max_sec):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.gui_params = gui_params
        self.comm = comm
        self.tags = tags
        self.max_sec = max_sec
        self.elapsed_time = 0.0
        self.original_df = None
        self.labels = None
        self.generated_df = None
        self.fs_job_list = None
        self.remaining_fs_job_list = None
        self.filter_based_methods = None
        self.results_with_score = pd.DataFrame()
        self.n_job = 0
        self.done_job = 0
        self.title = ""
        self.finished=False
        self.timeout=False
        self.probing=False
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution
    def run(self):
        closed_workers = 0
        num_workers = self.comm.size
        begin_time = time.time()
        if "autofe_system_attr" in self.gui_params:
            if "title" in self.gui_params['autofe_system_attr']:
                self.title = self.gui_params['autofe_system_attr']['title']
            if "probing" in self.gui_params['autofe_system_attr']:
                if self.gui_params['autofe_system_attr']['probing']==True:
                    self.probing = True # probing attr. added 20211019
        else:
            self.title = ""        
        self.original_df, self.labels  = load_origin_dataset(params=self.gui_params)
        self.generated_df = load_entire_dataset(params=self.gui_params, reduce_mem_usage=False)
        use_original  = False if self.original_df is None else True
        use_converted = False if self.generated_df is None else True
        self.fs_job_list, self.filter_based_methods = get_fs_chunk_by_metadata(params=self.gui_params, labels=self.labels, use_original=use_original, use_converted=use_converted, probing=self.probing)
        if self.fs_job_list is None:
            self.comm.Abort() # nothing to do
        else:
            self.n_job = len(self.fs_job_list)
            self.remaining_fs_job_list = self.fs_job_list.copy()
        if self.filter_based_methods: # not null
            if use_original:
                self.original_df = apply_filters(self.original_df, self.filter_based_methods, "original")
            if use_converted:
                self.generated_df = apply_filters(self.generated_df, self.filter_based_methods, "converted")
        if use_converted:
            ### save generated_df (csv)
            outputfilepath=os.path.join(self.gui_params['ml_file_path'],'converted_'+self.gui_params['ml_file_name'])
            print(">> Saving converted csv .. "+str(outputfilepath))
            self.generated_df.to_csv(outputfilepath, index=False)
            os.chmod(outputfilepath, 0o776)
            print(">> Complete !")
        ###
        while closed_workers < num_workers :
            # resource allocation (processor acquisition @ READY status)
            status = MPI.Status()
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            self.elapsed_time = time.time() - begin_time
            ###
            #if elapsed_time > self.max_sec: # do until max_sec considering merging time
            #    self.comm.Abort() #MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD with errorcode 0.
            #                      #NOTE: invoking MPI_ABORT causes Open MPI to kill all MPI processes.You may or may not see output from other processes,
            #                      #depending on exactly when Open MPI kills them.
            if tag == self.tags.READY:
                if self.elapsed_time < self.max_sec: # do until (max_sec-margin)
                    if len(self.remaining_fs_job_list)>0:
                        specific_data_chunk_to_consume, the_rest = get_a_data_chunk_per_group(self.remaining_fs_job_list)
                        self.remaining_fs_job_list = the_rest
                        if specific_data_chunk_to_consume['base_df'].values[0] == 'original':
                            target_data = self.original_df
                        elif specific_data_chunk_to_consume['base_df'].values[0] == 'converted':
                            target_data = self.generated_df
                        self.comm.send([specific_data_chunk_to_consume, target_data, self.labels], dest=source, tag=self.tags.START) # allow to train (1)
                    else:
                        self.comm.send(None, dest=source, tag=self.tags.EXIT_REQ) # allow to train (1)
                else: # time out
                    for i in range(0, self.comm.size):
                        self.comm.send(None, dest=i, tag=self.tags.EXIT_REQ) # stop all except rank 0
                    self.timeout = True
            elif tag == self.tags.DONE:
                self.results_with_score = self.results_with_score.append(data)
                self.results_with_score = self.results_with_score.sort_values(by='group_no')
                print("[DONE] processor ",source," finished work!")
                print(">>> A partial csv score has been updated to "+'fs_'+self.title+'__output_scores.csv')
                outputfilepath=os.path.join("./",'fs_'+self.title+'__output_scores.csv')
                self.results_with_score.to_csv(outputfilepath, index=False)
                os.chmod(outputfilepath, 0o776)
                self.done_job += 1
            elif tag == self.tags.EXIT_RES:
                closed_workers += 1
                print("***CLOSEDWORKERS******************************* = ", closed_workers, num_workers)
            time.sleep(0.5)
        ### finalize
        print(">>> Score csv has generated as "+'fs_'+self.title+'__output_scores.csv')
        outputfilepath=os.path.join("./",'fs_'+self.title+'__output_scores.csv')
        self.results_with_score = self.results_with_score.sort_values(by='group_no')
        self.results_with_score.to_csv(outputfilepath, index=False)
        os.chmod(outputfilepath, 0o776)
        print(">>> Feature Selection Done !")
        ####
        # making plotchart
        from plotly.offline import plot as offplot
        self.results_with_score = self.results_with_score.replace(np.nan, "-")
        self.results_with_score = self.results_with_score.sort_values(by='n_cols')
        self.results_with_score.index = pd.Index(range(len(self.results_with_score)))
        score_figure = plot_model_scores(self.results_with_score, self.title)
        score_html_path = outputfilepath.split('.csv')[0]+'.html'
        offplot(score_figure, filename = score_html_path, auto_open=False)
        print(">>> Scores html has generated as "+'fs_'+self.title+'__output_scores.html')
        os.chmod(score_html_path, 0o776)
        print(":: FEATURESELECTION_MPI scheduler thread terminated ::")
        self.finished=True

###################################################################################
###################################################################################
###################################################################################

def read_csv_from_filelist(all_parts_split):
    res = pd.concat([pd.read_csv(f, low_memory=False) for f in all_parts_split])
    return res

def merge_df_a_and_b(two_dfs):
    import pandas as pd
    base_df = two_dfs[0]
    target_df = two_dfs[1]
    try:
        res = base_df.merge(target_df,how='left').set_axis(base_df.index) # keep index
    except:
        # when pandas bug
        mid = list(set(base_df.index) & set(target_df.index))
        #base_df = base_df.iloc[mid]
        #target_df = target_df.iloc[mid]
        base_df = base_df.loc[mid]
        target_df = target_df.loc[mid]
        res = base_df.transpose().append(target_df.transpose()).transpose()
    if 'Unnamed: 0' in res.columns:
        res.drop(['Unnamed: 0'], axis=1, inplace=True)
    return res

def parallelizize_concat_by_pool(all_parts, func, n_cores='auto'):
    import pandas as pd
    import gc
    if n_cores == 'auto':
        #import multiprocessing as mp
        #n_cores = min(int(mp.cpu_count() / 2), int(len(all_parts)/2))
        n_cores = min(int(cpu_count() / 2), int(len(all_parts)/2))
    all_parts_split = np.array_split(all_parts, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, all_parts_split))
    pool.close()
    pool.join()
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df

#df=parallelizize_concat_by_pool(all_parts, read_csv_from_filelist)

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
    tags = enum('READY', 'DONE', 'EXIT_REQ','EXIT_RES', 'START')
    #########################################################################
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    gui_params = load_metadata(metadata_filename)
    #max_sec = gui_params['time_deadline_sec'] if 'time_deadline_sec' in gui_params else 3600 # default: max 1 hour
    max_sec = 3600 # max_sec n_proc
    if 'autofe_system_attr' in gui_params:
        if 'time_deadline_sec' in gui_params['autofe_system_attr']:
            max_sec = gui_params['autofe_system_attr']['time_deadline_sec'] # update time_deadline_sec if exists in metadata.json
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
                group_no = gui_params['autofe_system_attr']['group_no'] # update group_no if exists in metadata.json
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
                single_group_each_sub.append(each_sub[0])
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
            data_csv = data[0]            
            target_csv = pd.read_csv(data[2])
            if idx_col_name: 
                target_csv=target_csv.set_index(idx_col_name)
            #
            df_merged = merge_df_a_and_b( (data_csv, target_csv) )
            if df_merged is not None:
                #try:
                outputfilepath=data[2]#+"_update"
                if idx_col_name: # 원래 index 항목이 있었다면
                    df_merged.to_csv(outputfilepath, index=True)
                else:            # 없었다면 동일하게 없게 출력
                    df_merged.to_csv(outputfilepath, index=False)
                os.chmod(outputfilepath, 0o776)
                print(str(data[1])+" -> "+str(data[2])+ "(merge done!!)")
                comm.send([data[1], data[2]], dest=0, tag=tags.DONE) # merged, original + target
            #comm.send(None, dest=0, tag=tags.READY)
        elif tag == tags.EXIT_REQ:
            print(">> Process (rank %d) on %s will waiting other process.." % (rank,name))
            comm.send(None, dest=0, tag=tags.EXIT_RES)
            #print(">> Merge CSVs almost DONE ! Please wait for other process..")
            break
    if rank == 0:
        while True:
            if provider.finished:
                print("ALL finishied")
                break
            #if rank != 0:
            #    print(">> Process (rank %d) on %s will waiting other process.." % (rank,name))
            #    comm.send(None, dest=0, tag=tags.EXIT)
            #    break
            #else:
            #    print(">> Merge CSVs almost DONE ! Please wait for other process..")
            #    break
        #comm.send(None, dest=0, tag=tags.EXIT)

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
###################################################################################
#### 2021-07-20 added featureselection_mpi
###################################################################################
def get_id_cols(gui_params, return_type='col_name'):
    base_filepath = os.path.join(gui_params['ml_file_path'],gui_params['ml_file_name'])
    ic, oc, vt=recursiveFindColumnNamesandVariableTypes(gui_params, base_filepath)
    for target_key, target_name in oc.items():
        pass
    index_key, index_name = getColumnNameforSpecificType(ic, vt, "Index", base_filepath)
    res_df=None
    if return_type == 'col_name':
        return (index_name, target_name)
        #return index_name, target_name
    elif return_type == 'dict':
        return ({index_key: index_name}, oc)
    elif return_type == 'key':
        return (int(index_key), int(target_key))

def load_origin_dataset(params):
    gui_params = params
    base_filepath = os.path.join(gui_params['ml_file_path'],gui_params['ml_file_name'])
    if os.path.exists(base_filepath):
        id_col, target_col = get_id_cols(gui_params)
        original_dataset = pd.read_csv(base_filepath)
        #### filtering based on selected input columns
        input_columns_index_and_name, output_columns_index_and_name, datatype_of_columns = getColumnNamesandVariableTypes(gui_params)
        filtered_cols = dict(input_columns_index_and_name, **output_columns_index_and_name)
        original_dataset = original_dataset[filtered_cols.values()]
        print(">> Original dataset has loaded successfully.")
        if id_col:
            original_dataset = original_dataset.set_index(id_col).sort_index()
        if target_col in original_dataset:
            labels = original_dataset[[target_col]]
            original_dataset = original_dataset.drop(columns = [target_col])
            return (original_dataset, labels)
        else:
            return (original_dataset, None)
    else:
        return (None, None)

def load_entire_dataset(params, reduce_mem_usage=False):
    gui_params = params
    if "autofe_system_attr" in gui_params:
            if "title" in gui_params['autofe_system_attr']:
                title = gui_params['autofe_system_attr']['title']
            else:
                title = ""
    if os.path.exists('fm_'+title+"__output_list.csv"):
        id_col, target_col = get_id_cols(gui_params)
        output_list = pd.read_csv('fm_'+title+"__output_list.csv")
        all_parts = [x for x in output_list['filename'].values if x.endswith('.csv')]
        print(">> Loading partial datasets in output_list.csv ... ***")
        if len(all_parts)>1:
            dataset=parallelizize_concat_by_pool(all_parts, read_csv_from_filelist)
        else:
            dataset = pd.concat( [ pd.read_csv(f) for f in all_parts ] )
        if id_col:
            dataset = dataset.set_index(id_col).sort_index()
        id_col_generated = gui_params['ml_file_name']+"_index"
        if id_col_generated in dataset.columns:
            dataset = dataset.set_index(id_col_generated).sort_index()
        dataset.index.name=id_col
        print(">> Converted Dataset has loaded successfully.")
        if reduce_mem_usage:
            dataset = reduce_mem_usage_df(dataset)
        return dataset
    else:
        return None

# 20210726 added parallel correlation calculation modified by Jclee
def remove_highly_correlated_features_parallel_NaNCorrMp(feature_matrix, features=None, pct_corr_threshold=0.95,
                                      features_to_check=None, features_to_keep=None):
    def _apply_feature_selection(keep, feature_matrix, features=None):
        new_matrix = feature_matrix[keep]
        new_feature_names = set(new_matrix.columns)
        if features is not None:
            new_features = []
            for f in features:
                if f.number_output_features > 1:
                    slices = [f[i] for i in range(f.number_output_features)
                              if f[i].get_name() in new_feature_names]
                    if len(slices) == f.number_output_features:
                        new_features.append(f)
                    else:
                        new_features.extend(slices)
                else:
                    if f.get_name() in new_feature_names:
                        new_features.append(f)
            return new_matrix, new_features
        return new_matrix
    from featuretools import variable_types as vtypes
    import random
    ##
    if pct_corr_threshold < 0 or pct_corr_threshold > 1:
        raise ValueError("pct_corr_threshold must be a float between 0 and 1, inclusive.")
    if features_to_check is None:
        features_to_check = feature_matrix.columns
    else:
        for f_name in features_to_check:
            assert f_name in feature_matrix.columns, "feature named {} is not in feature matrix".format(f_name)
    if features_to_keep is None:
        features_to_keep = []
    boolean = ['bool']
    numeric_and_boolean_dtypes = vtypes.PandasTypes._pandas_numerics + boolean
    fm_to_check = (feature_matrix[features_to_check]).select_dtypes(
        include=numeric_and_boolean_dtypes)
    columns_to_check = fm_to_check.columns
    from nancorrmp.nancorrmp import NaNCorrMp
    #
    corr_matrix=NaNCorrMp.calculate(fm_to_check, n_jobs=-1, chunks=3000) # using all threads available
    remove_negative = True
    #
    threshold = pct_corr_threshold
    corr_mat = corr_matrix.corr()
    if remove_negative:
        corr_mat = np.abs(corr_mat)
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][corr_mat[col] > threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    unique_selection = pd.Series(select_flat).unique()
    dropped = [x for x in corr_mat.columns if x in unique_selection]
    keep = [f_name for f_name in feature_matrix.columns if (f_name in features_to_keep or f_name not in dropped)]
    return _apply_feature_selection(keep, feature_matrix, features)

def remove_highly_correlated_features(feature_matrix, features=None, pct_corr_threshold=0.95,
                                      features_to_check=None, features_to_keep=None):
    ''' modified by Jeongcheol Lee 2021-07-27: automated parallelization '''
    """Removes columns in feature matrix that are highly correlated with another column.
        Note:
            We make the assumption that, for a pair of features, the feature that is further
            right in the feature matrix produced by ``dfs`` is the more complex one.
            The assumption does not hold if the order of columns in the feature
            matrix has changed from what ``dfs`` produces.
        Args:
            feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature
                        names and rows are instances.
            features (list[:class:`featuretools.FeatureBase`] or list[str], optional):
                        List of features to select.
            pct_corr_threshold (float): The correlation threshold to be considered highly
                        correlated. Defaults to 0.95.
            features_to_check (list[str], optional): List of column names to check
                        whether any pairs are highly correlated. Will not check any
                        other columns, meaning the only columns that can be removed
                        are in this list. If null, defaults to checking all columns.
            features_to_keep (list[str], optional): List of colum names to keep even
                        if correlated to another column. If null, all columns will be
                        candidates for removal.
        Returns:
            pd.DataFrame, list[:class:`.FeatureBase`]:
                The feature matrix and the list of generated feature definitions.
                Matches dfs output. If no feature list is provided as input,
                the feature list will not be returned. For consistent results,
                do not change the order of features outputted by dfs.
    """
    from featuretools import variable_types as vtypes
    import multiprocessing
    if pct_corr_threshold < 0 or pct_corr_threshold > 1:
        raise ValueError("pct_corr_threshold must be a float between 0 and 1, inclusive.")
    if features_to_check is None:
        features_to_check = feature_matrix.columns
    else:
        for f_name in features_to_check:
            assert f_name in feature_matrix.columns, "feature named {} is not in feature matrix".format(f_name)
    if features_to_keep is None:
        features_to_keep = []
    boolean = ['bool']
    numeric_and_boolean_dtypes = vtypes.PandasTypes._pandas_numerics + boolean
    global fm_to_check
    fm_to_check = (feature_matrix[features_to_check]).select_dtypes(
        include=numeric_and_boolean_dtypes)
    #s_t2=time.time()
    #print(s_t2-s_t1," *1")
    #dropped = set()
    #columns_to_check = fm_to_check.columns
    # When two features are found to be highly correlated,
    # we drop the more complex feature
    # Columns produced later in dfs are more complex
    def _apply_feature_selection(keep, feature_matrix, features=None):
        new_matrix = feature_matrix[keep]
        new_feature_names = set(new_matrix.columns)
        if features is not None:
            new_features = []
            for f in features:
                if f.number_output_features > 1:
                    slices = [f[i] for i in range(f.number_output_features)
                              if f[i].get_name() in new_feature_names]
                    if len(slices) == f.number_output_features:
                        new_features.append(f)
                    else:
                        new_features.extend(slices)
                else:
                    if f.get_name() in new_feature_names:
                        new_features.append(f)
            return new_matrix, new_features
        return new_matrix
    def _do_single(): #15.62it/s for generated_df
        from tqdm import tqdm
        dropped = set()
        columns_to_check = fm_to_check.columns
        #for i in range(len(columns_to_check) - 1, 0, -1):
        for i in tqdm(range(len(columns_to_check) - 1, 0, -1)):
            more_complex_name = columns_to_check[i]
            more_complex_col = fm_to_check[more_complex_name]
            target_j_all=[] ##
            for j in range(i - 1, -1, -1):
                target_j_all.append(j)##
                less_complex_name = columns_to_check[j]
                less_complex_col = fm_to_check[less_complex_name]
                if abs(more_complex_col.corr(less_complex_col)) >= pct_corr_threshold:
                    dropped.add(more_complex_name)
                    break
        return dropped
    # 1.0515291690826416 original_df
    def _do_parallel(): #1.62it/s for generated_df
        from tqdm import tqdm
        dropped = set()
        columns_to_check = fm_to_check.columns
        #for i in range(len(columns_to_check) - 1, 0, -1):
        for i in tqdm(range(len(columns_to_check) - 1, 0, -1)):
            more_complex_name = columns_to_check[i]
            more_complex_col = fm_to_check[more_complex_name]
            target_j_all=[]
            for j in range(i - 1, -1, -1):
                target_j_all.append(j)
            #print(target_j_all)
            bigger_than_pct_corr_threshold=False
            bigger_than_pct_corr_threshold=calculate_corr_by_mpipool(pct_corr_threshold= pct_corr_threshold,
                compare_target_a_name = more_complex_name, compare_targets_subset_b_names=columns_to_check[target_j_all].tolist(),
                func=compare_corr, n_cores='auto')
            if bigger_than_pct_corr_threshold:
                dropped.add(more_complex_name)
        return dropped
    # 18.323530673980713 original_df
    if fm_to_check.size/1000000 < 10000: # under 10GB
    #if fm_to_check.size/1000000 < 100: # under 10GB
        print("remove_highly_correlated_features by single processing.. (under 10GB dataframe)")
        dropped = _do_single()
    else:
        print("remove_highly_correlated_features by multiple processing.. (over 10GB dataframe)")
        dropped = _do_parallel() # more than 10GB        
    keep = [f_name for f_name in feature_matrix.columns if (f_name in features_to_keep or f_name not in dropped)]
    return _apply_feature_selection(keep, feature_matrix, features)

def calculate_corr_by_mpipool(pct_corr_threshold, compare_target_a_name, compare_targets_subset_b_names, func, n_cores='auto'):
    import gc
    if n_cores == 'auto':
        n_cores = max(min(int(cpu_count() / 2), int(len(compare_targets_subset_b_names)/2)),1)
    #print(n_cores)
    compare_targets_subset_b_names_split = np.array_split(compare_targets_subset_b_names, n_cores)
    mapping_list = []
    for compare_target_b_name in compare_targets_subset_b_names_split:
        mapping_list.append([compare_target_a_name, compare_target_b_name, pct_corr_threshold])
    res=False
    with Pool(n_cores) as pool:
        results = pool.imap_unordered(func, mapping_list)
        for result in results:
            if result:
                res = True
                break
    pool.close()
    pool.join()
    gc.collect()
    return res

def compare_corr(mapping_list):
    pct_corr_threshold = mapping_list[2]
    col_a_name = mapping_list[0]
    for col_b_name in mapping_list[1]:
        if abs(fm_to_check[col_a_name].corr(fm_to_check[col_b_name])) >= pct_corr_threshold:
            #print(col_a_name, col_b_name, mapping_list[1], abs(fm_to_check[col_a_name].corr(fm_to_check[col_b_name])))
            return True
    return False

def apply_filters(df, filter_based_methods, df_types):
    import featuretools as ft
    if filter_based_methods:
        for each_filter_algorithm, each_filter_algorithm_params in filter_based_methods.items():
            if each_filter_algorithm == 'remove_low_information_features':
                before = df.shape[1]
                df = ft.selection.remove_low_information_features(df)
                print("* "+df_types+" <- Remove Low Information Features (at least 2 unique values not null): "+str(before)+" columns -> "+str(df.shape[1])+" columns.")
            elif (each_filter_algorithm == 'remove_highly_null_features') and (each_filter_algorithm_params):
                before = df.shape[1]                
                df = ft.selection.remove_highly_null_features(df, **each_filter_algorithm_params)
                print("* "+df_types+" <- Remove Highly Null Features: "+str(before)+" columns -> "+str(df.shape[1])+" columns.")
            elif (each_filter_algorithm == 'remove_single_value_features') and (each_filter_algorithm_params):
                before = df.shape[1]                
                df = ft.selection.remove_single_value_features(df, **each_filter_algorithm_params)
                print("* "+df_types+" <- Remove Single Value Features (all the values are the same): "+str(before)+" columns -> "+str(df.shape[1])+" columns.")
            elif (each_filter_algorithm == 'remove_highly_correlated_features') and (each_filter_algorithm_params):
                ''' too slow when large number of columns --> should run under mp '''
                before = df.shape[1]
                df = ft.selection.remove_highly_correlated_features(df, **each_filter_algorithm_params)
                df = remove_highly_correlated_features(df, **each_filter_algorithm_params)
                #print("(donotusenow)* "+df_types+" <- Remove Highly Correlated Features: "+str(before)+" columns -> "+str(df.shape[1])+" columns.")
                print("* "+df_types+" <- Remove Highly Correlated Features: "+str(before)+" columns -> "+str(df.shape[1])+" columns.")
    return df

def plot_feature_importance(fi_df, fi_title):
    import plotly.io as pio
    import plotly.express as px
    pio.renderers.default = 'colab'
    fig = px.bar(fi_df,
                x='value',
                y='Feature',
                color='value',
                orientation='h',
                title='Feature Importances: '+fi_title,
                labels={"value":"LightGBM Features (avg over folds)", "Feature":"Feature"},
                color_continuous_scale='sunsetdark',
    )
    fig.update_yaxes(autorange='reversed')
    return fig


def plot_model_scores(results_with_score, fs_title):
    #import plotly.io as pio
    #import plotly.express as px
    #pio.renderers.default = 'colab'
    #fig = px.bar(results_with_score,
    #            x='group_no',
    #            y='score',
    #            text='n_cols',
    #            color='base_df',
    #            title='Scores: '+fs_title,
    #            hover_data=['score', 'base_df', 'group_no','wrapper', 'n_cols'],
    #            labels={"group_no":"Group Number", "score":"Performance Score",
    #            "base_df":"Dataframe", "n_cols":"Selected Columns", "wrapper":"Wrapper Type" },
    #)
    #fig.update_traces(textposition='outside')
    #minv = results_with_score.score.min()
    #maxv = results_with_score.score.max()
    #diffv = maxv-minv
    #fig.update_yaxes(range=[minv-diffv/2, maxv+diffv/2])
    ##########################
    import plotly.io as pio
    import plotly.express as px
    ## added 20211026
    res=[]
    for i, row in results_with_score.iterrows():
        res.append(str(row['n_cols'])+'/'+str(row['original_n_cols'])+" (top "+str(round(row['n_cols']/row['original_n_cols']*100))+"% cols)")
    results_with_score['percents_str']=res
    results_with_score['base_df']=results_with_score['base_df'].apply(lambda x: 'AutoSynthesized Dataframe' if x=='converted' else 'Raw Dataframe')
    ##
    
    fig = px.bar(results_with_score,
                x='group_no',
                y='score',
                text='percents_str',
                color='base_df',
                title='Scores: '+fs_title,
                hover_data=['score', 'group_no','wrapper', 'n_cols','original_n_cols','model_size'],
                labels={"group_no":"Group Number", "score":"Performance Score",
                "base_df":"Dataframe", "n_cols":"Selected Columns", "wrapper":"Wrapper Type",
                "model_size":"Model Size" ,'original_n_cols':"All Columns"},
    )
    fig.update_traces(textposition='outside')
    minv = results_with_score.score.min()
    maxv = results_with_score.score.max()
    diffv = maxv-minv
    fig.update_yaxes(range=[minv-diffv/2, maxv+diffv/2])
    fig.add_annotation(
            x=results_with_score[results_with_score['score']==results_with_score['score'].max()]['group_no'].values[0],
            y=results_with_score['score'].max(),
            xref="x",
            yref="y",
            text="Best Score = "+str(round(results_with_score['score'].max(),4)),
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
                ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=0,
            ay=-50,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8
            )
    ## add model size later.. 2021-10-27 disabled
    #for i, row in results_with_score.iterrows():
    #    fig.add_annotation(x=row['group_no'], y=minv-diffv/2+abs(((minv-diffv/2)-(maxv+diffv/2))/8),
    #        text=str(round(row['model_size']/1000000,1))+"M",
    #        align="center",
    #        showarrow=False,ax=0,ay=0,
    #        font=dict(
    #        family="Courier New, monospace",
    #        size=30,
    #        color="#ffffff"),)
    return fig

def plot_output_scores_html(metadata_json):
    gui_params=load_metadata(metadata_json)
    jsonpaths = os.path.split(metadata_json)
    FINISHED = False
    if len(jsonpaths)==2:
        status_file = os.path.join(jsonpaths[0],'status')
        if os.path.exists(status_file):
            with open(status_file) as f:
                s=f.read()
                FINISHED = s.startswith('FINISHED') # True if finished
    if not FINISHED:
        title=""
        res = False
        if "autofe_system_attr" in gui_params:
            if "title" in gui_params['autofe_system_attr']:
                title = gui_params['autofe_system_attr']['title']
            else:
                title = ""
        if 'autofe_portal_attr' in gui_params:
            if 'workspaceLocation' in gui_params['autofe_portal_attr']:
                scores_filepath = os.path.join(gui_params['autofe_portal_attr']['workspaceLocation'],"fs_"+title+"__output_scores.csv")
                scores_filepath2 = scores_filepath.replace("/science-data/", '/EDISON/SCIDATA/')
                scores_filepath3 = os.path.basename(scores_filepath)
                paths = [scores_filepath, scores_filepath2, scores_filepath3]
                for each_path in paths:
                    if os.path.exists(each_path):
                        res=make_chart(each_path,title)
        if res == False:
            print("Model should be prepared at least one to make html chart. Please wait for seconds...")

def make_chart(scores_filepath,title):
    results_with_score = pd.read_csv(scores_filepath)
    from plotly.offline import plot as offplot
    results_with_score = results_with_score.replace(np.nan, "-")
    results_with_score = results_with_score.sort_values(by='n_cols')
    results_with_score.index = pd.Index(range(len(results_with_score)))
    score_figure = plot_model_scores(results_with_score, title)
    score_html_path = scores_filepath.split('.csv')[0]+'.html'
    offplot(score_figure, filename = score_html_path, auto_open=False)
    print(">>> Scores html has generated as "+'fs_'+title+'__output_scores.html')
    os.chmod(score_html_path, 0o776)
    print(":: Model Score html Chart has been generated successfully ::")
    return True

## 2021-08-30 make chart before finishing all job (running status)
## singularity exec -H ${PWD}:/make_chart/  /EDISON/SCIDATA/singularity-images/userenv python -c "from sdroptim.mpi_role import *;plot_output_scores_html('/make_chart/metadata.json')"


def model_score(params, job_to_do, dataset, labels, hparams):
    import copy
    gui_params = copy.deepcopy(params)
    def_hparams = copy.deepcopy(hparams)
    #### dataset 처리
    if "autofe_system_attr" in gui_params:
        if "title" in gui_params['autofe_system_attr']:
            title = gui_params['autofe_system_attr']['title']
        else:
            title = ""
    wrapper = job_to_do['wrapper'].values[0]
    param_name = job_to_do['param_name'].values[0]
    param_value = job_to_do['param_value'].values[0]
    current_group_no = job_to_do['group_no'].values[0]
    target_col = labels.columns[0]
    if 'encoding' in def_hparams:
        encoding = def_hparams['encoding']
    else:
        encoding='ohe'
    if 'cv' in def_hparams:
        num_cv = def_hparams['cv']
    else:
        num_cv = 0
    #### 인코딩 하기 전이 오리지널 데이터셋이다
    if encoding == 'ohe':
        #ori_dataset = dataset.copy()
        dataset = pd.get_dummies(dataset)
        # Align the dataframes by the columns
        # No categorical indices to record
        cat_indices = 'auto'
    # Integer label encoding
    elif encoding == 'le':
        # Create a label encoder
        label_encoder = LabelEncoder()
        # List for storing categorical indices
        cat_indices = []
        # Iterate through each column
        for i, col in enumerate(dataset):
            if dataset[col].dtype == 'object':
                # Map the categorical features to integers
                dataset[col] = label_encoder.fit_transform(np.array(dataset[col].astype(str)).reshape((-1,)))
                # Record the categorical indices
                cat_indices.append(i)
    features = dataset.columns.tolist()
    n_features = len(features)
    if param_value<=0:
        print("[ERR] Number of columns requires more than 2.")
        return 0.0
    elif param_value < 1:
        n_cols = max(int(n_features*param_value), 1) #minimum 1
    elif param_value > 1:
        n_cols = min(int(param_value), n_features) # set maximum n_cols
    else: # if None or NaN
        n_cols = n_features
    ################################
    labels = labels.loc[sorted(list(set(dataset.index)&set(labels.index)))] ## 0811 이게 동작을 안하나??
    global label_names
    label_names = labels[target_col].unique()
    LightGBM_num_boost_round = def_hparams['num_boost_round']
    ################################
    if 'encoding' in def_hparams:
        def_hparams.pop('encoding',None)
    use_gpu=-1 # default. use_gpu (-1: cpu, otherwise: gpu_no)
    cpu_only = False
    global DEVICE
    DEVICE=0
    if 'gpu_no' in def_hparams:
        use_gpu=def_hparams.pop('gpu_no',None)
        try:
            DEVICE = int(use_gpu)
        except:
            DEVICE = 0
    else:
        if 'nthread' in def_hparams:
            def_hparams.pop('nthread')
    ################################
    ################################
    from sklearn.model_selection import train_test_split
    global X_train, X_test, y_train, y_test
    ##### cleaning for automatic modeling # add 20210811
    m_dataset = dataset.copy()
    m_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(m_dataset, labels[target_col], test_size=gui_params['testing_frame_rate'])
    features = X_train.columns.tolist()
    original_n_cols = len(features)
    target = target_col
    ################################################
    if (wrapper == "GradientFeatureSelector") and (n_cols >= 1):
        feature_selected = True
        from nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector
        gfs_params={}
        gfs_params['learning_rate']=def_hparams['learning_rate']
        #if use_gpu is not None:
        if use_gpu>=0:
            gfs_params['device']='cuda' # 'cuda' instead of 'gpu'
        gfs_params['classification']=True if gui_params['task']=='Classification' else False
        gfs_params['n_epochs']=10
        gfs_params['n_features']=n_cols
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit(X_train.append(X_test))
        try:
            fgs = FeatureGradientSelector(**gfs_params)
            fgs.fit(scaled.transform(X_train.replace(np.nan,0)), y_train.values) # torch 
        except:
            gfs_params['device']='cpu'
            fgs = FeatureGradientSelector(**gfs_params)
            cpu_only = True
            fgs.fit(scaled.transform(X_train.replace(np.nan,0)), y_train.values) # torch 
        # get improtant features
        # will return the index with important feature here.
        X_train = X_train.iloc[:,fgs.get_selected_features()].copy() # in order to avoid highly-defragmented frame
        X_test = X_test.iloc[:,fgs.get_selected_features()].copy() # in order to avoid highly-defragmented frame
        features = X_train.columns.tolist()
        target = target_col
        #### file save
        outputfilepath=os.path.join("./", "fs_GFS_n"+str(n_cols)+"_"+title+"__G"+str(current_group_no)+".csv")
        two_dfs =  (X_train.append(X_test).sort_index().reset_index(), labels.sort_index().reset_index())
        fs = merge_df_a_and_b(two_dfs)
        if 'index' in fs.columns:
            fs = fs.drop('index', axis=1)
        fs.to_csv(outputfilepath, index=False)
        os.chmod(outputfilepath, 0o776)
        ##############
    elif wrapper == "GBDTSelector":
        feature_selected = True
        from nni.algorithms.feature_engineering.gbdt_selector import GBDTSelector
        for_wrapper_param = def_hparams.copy()
        if gui_params['task'] == "Classification":
            for_wrapper_param['num_class']=len(label_names)
        if use_gpu >= 0:
            for_wrapper_param['device_type']='gpu'
            for_wrapper_param['gpu_device_id']=use_gpu
        if 'num_boost_round' in for_wrapper_param:
            for_wrapper_param.pop('num_boost_round')
        if 'silent' in for_wrapper_param:
            for_wrapper_param.pop('silent')
        fgs = GBDTSelector()
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit(X_train.append(X_test))
        try:
            fgs.fit(scaled.transform(X_train.values), y_train.values,lgb_params=for_wrapper_param, eval_ratio=0.2,early_stopping_rounds= max(int(LightGBM_num_boost_round/10),5),num_boost_round=LightGBM_num_boost_round,importance_type='split', verbose=-100)
        except:
            for_wrapper_param['device_type']='cpu'
            #for_wrapper_param['nthread']=-1 # sometimes too slow -- 20210830
            cpu_only = True
            fgs.fit(scaled.transform(X_train.values), y_train.values,lgb_params=for_wrapper_param, eval_ratio=0.2,early_stopping_rounds= max(int(LightGBM_num_boost_round/10),5),num_boost_round=LightGBM_num_boost_round,importance_type='split', verbose=100)
        '''/home/jclee/Feature_study/mpi_role.py:2079: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider using pd.concat instead.  To get a de-fragmented frame, use `newframe = frame.copy()`'''
        X_train = X_train.iloc[:,fgs.get_selected_features(n_cols)].copy() # in order to avoid highly-defragmented frame
        X_test = X_test.iloc[:,fgs.get_selected_features(n_cols)].copy() # in order to avoid highly-defragmented frame
        features = X_train.columns.tolist()
        target = target_col
        #### file save
        outputfilepath=os.path.join("./", "fs_GBDT_n"+str(n_cols)+"_"+title+"__G"+str(current_group_no)+".csv")
        two_dfs =  (X_train.append(X_test).sort_index().reset_index(), labels.sort_index().reset_index())
        fs = merge_df_a_and_b(two_dfs)
        if 'index' in fs.columns:
            fs = fs.drop('index', axis=1)
        fs.to_csv(outputfilepath, index=False)
        os.chmod(outputfilepath, 0o776)
        ##############
    else:
        feature_selected = False
        outputfilepath=os.path.join("./", "fs_GBDT_n"+str(n_cols)+"_"+title+"__G"+str(current_group_no)+".csv")
        two_dfs = (dataset.sort_index().reset_index(), labels.sort_index().reset_index())
        fs = merge_df_a_and_b(two_dfs)
        if 'index' in fs.columns:
            fs = fs.drop('index', axis=1)
        #fs = ori_dataset
        fs.to_csv(outputfilepath, index=False)
        os.chmod(outputfilepath, 0o776)
        ##############
    final_column_names = X_test.columns.tolist()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit(X_train.append(X_test))
    #
    X_train = scaled.transform(X_train.values)
    X_test = scaled.transform(X_test.values)
    y_train = y_train.values
    y_test = y_test.values    
    ################################
    gui_params['hparams'] = def_hparams
    gui_params['algorithm'] = 'LightGBM'
    ################################
    from sdroptim_client.PythonCodeModulator import getLightGBM
    global clfs
    clfs = []
    code=getLightGBM(gui_params, -1, False, False)
    code=code.replace("DEVICE = 0","")
    code=code.replace("X_train, y_train = train_data[features].values, train_data[target].values","")
    code=code.replace("X_test, y_test = test_data[features].values, test_data[target].values","")
    code=code.replace("confidence = metrics", "confidence = sklearn.metrics")
    ## cv configuration
    code=code.replace("clf = lgb.train(params = lgb_params, train_set = dtrain, num_boost_round = LightGBM_num_boost_round, early_stopping_rounds = max(int(LightGBM_num_boost_round/10),5))",\
                      "clf = lgb.train(params = lgb_params, train_set = dtrain, num_boost_round = LightGBM_num_boost_round, early_stopping_rounds = max(int(LightGBM_num_boost_round/10),5), valid_sets = ["+("dvalid" if num_cv>0 else "dtest")+"], categorical_feature = 'auto', verbose_eval=False)")
    code=code.replace("scores.append(confidence)","scores.append(confidence)\n    clfs.append(clf)\n")
    code="\n".join([x for x in code.split('\n') if not x.strip().startswith('print(')])
    ##########################
    global confidence
    code=code.replace("'gpu'", "'cpu'")
    code=code.replace("    'gpu_device_id': DEVICE,\n","")
    if cpu_only == False:
        if use_gpu >=0:
            code=code.replace("'cpu'", "'gpu'")
            code=code.replace("'device_type': 'gpu',\n", "'device_type': 'gpu',\n    'gpu_device_id': DEVICE,\n    'nthread':1,\n")
    try:
        if not feature_selected:
            print(code)
            print(X_train)
        exec(code, globals())
    except:
        code=code.replace("'gpu'", "'cpu'")
        code=code.replace("'gpu'", "'cpu'")
        code=code.replace("    'gpu_device_id': DEVICE,\n","")
        exec(code, globals())
    #return confidence
    ### 이아래부분도 class/reg 구분지어줘야한다...
    avg_clf_fi = None
    if num_cv>1:
        model_size = 0
        if gui_params['task']=='Classification':
            test_preds_proba = np.zeros((X_test.shape[0],len(label_names)))
            for each_clf in clfs:
                model_size+=len(each_clf.model_to_string())
                test_preds_proba += each_clf.predict(X_test)/num_cv
                if avg_clf_fi is None:
                    avg_clf_fi = each_clf.feature_importance()
                else:
                    avg_clf_fi = np.sum([avg_clf_fi, each_clf.feature_importance()], axis=0)
            avg_clf_fi = avg_clf_fi / num_cv
            model_size = int(model_size / num_cv)
            test_preds = np.argmax(test_preds_proba, axis=1)
            confidence = sklearn.metrics.f1_score(test_preds, y_test, average='macro')
        elif gui_params['task']=='Regression':
            confidence = scores.mean()
            for each_clf in clfs:
                model_size+=len(each_clf.model_to_string())
                if avg_clf_fi is None:
                    avg_clf_fi = each_clf.feature_importance()
                else:
                    avg_clf_fi = np.sum([avg_clf_fi, each_clf.feature_importance()], axis=0)
            avg_clf_fi = avg_clf_fi / num_cv
            model_size = int(model_size / num_cv)
    else:
        avg_clf_fi = clf.feature_importance()    #
        model_size = len(clf.model_to_string())
    ##############
    fi_df = pd.DataFrame({'value':avg_clf_fi, 'Feature': final_column_names}).sort_values(by="value",ascending=False)
    fi_csv_path = outputfilepath.split('.csv')[0]+'_FI.csv'
    fi_df.to_csv(fi_csv_path, index=False)
    os.chmod(fi_csv_path, 0o776)
    from plotly.offline import plot as offplot
    fi_title = outputfilepath + " ( "+("-" if wrapper is None else wrapper)+ " / " + str(n_cols) + " cols.)"
    fi_figure = plot_feature_importance(fi_df, fi_title)
    fi_html_path = outputfilepath.split('.csv')[0]+'.html'
    offplot(fi_figure, filename = fi_html_path, auto_open=False)
    print(">>> Feature Importance html has generated as "+fi_html_path)
    os.chmod(fi_html_path, 0o776)
    return (confidence, n_cols, original_n_cols, model_size)

def featureselection_mpi(metadata_filename, elapsed_time=0.0): # 20210720 add
    # Initializations and preliminaries
    allocated_fnc=0
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    tags = enum('READY', 'DONE', 'EXIT_REQ','EXIT_RES', 'START')
    #########################################################################
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    gui_params = load_metadata(metadata_filename)
    max_sec = gui_params['time_deadline_sec'] if 'time_deadline_sec' in gui_params else 3600 # default: max 1 hour
    max_sec = max_sec - elapsed_time
    if rank == 0:
        provider = ThreadingforFeatureSelection(gui_params,comm, tags, max_sec)
    ####################################################
    name = MPI.Get_processor_name()
    gpu_no = abs((rank-1)%2-1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)#str(rank - 1)
    ##############################################################
    def_hparams_gpu={ # gpu/normal
    "gpu_no":gpu_no,
    "cv":5,
    "encoding":"ohe",
    "num_boost_round":100,
    "nthread":-1,
    "objective":"regression" if gui_params['task'] == "Regression" else "multiclass",
    "metric":"rmse" if gui_params['task'] == "Regression" else "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate":0.01,
    "max_depth": 11,
    "num_leaves":58,
    "colsample_bytree":0.5,
    "subsample":0.5,
    "max_bin":255,
    "reg_alpha":0.0,
    "reg_lambda":0.0,
    "min_child_weight": 6,
    "min_child_samples":20,
    "verbose":-1,
    }
    def_hparams={ # cpu/normal
    #"gpu_no":0,
    "cv":5,
    "encoding":"ohe",
    "num_boost_round":100,
    "nthread":4, # large nthread may occur deadlock 20210830
    "objective":"regression" if gui_params['task'] == "Regression" else "multiclass",
    "metric":"rmse" if gui_params['task'] == "Regression" else "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate":0.1,
    "max_depth": -1,
    "num_leaves":31,
    "colsample_bytree":0.5,
    "subsample":0.5,
    "max_bin":63,
    "reg_alpha":0.0,
    "reg_lambda":0.0,
    "min_child_weight": 6,
    "min_child_samples":20,
    "verbose":-1,
    }
    def_hparams_small={ # cpu/normal
    #"gpu_no":0,
    "cv":5,
    "encoding":"ohe",
    "num_boost_round":100,
    "nthread":4, # large nthread may occur deadlock 20210830
    "objective":"regression" if gui_params['task'] == "Regression" else "multiclass",
    "metric":"rmse" if gui_params['task'] == "Regression" else "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate":0.1,
    "max_depth": -1,
    "num_leaves":31,
    "colsample_bytree":0.5,
    "subsample":0.5,
    "max_bin":63,
    "reg_alpha":0.0,
    "reg_lambda":0.0,
    "min_child_weight": 6,
    "min_child_samples":20,
    "verbose":-1,
    }
    ############################# default system conf. in slurm workers
    maximum_cores_per_a_node = 30
    maximum_gpus_per_a_node  =  2
    #GPUInfo.get_info()[0]
    #{'16771': ['0'], '16772': ['0'], '16774': ['0'], '16773': ['1'], '16775': ['1'], '16805': ['1']}
    #############################
    gpu_available = False
    for i in range(maximum_gpus_per_a_node):
        if rank%maximum_cores_per_a_node == i:
            curr_n_gpu_process = len(GPUInfo.get_info()[0])
            if curr_n_gpu_process < maximum_gpus_per_a_node:
                gpu_available = True
    if not gpu_available:
        if 'gpu_no' in def_hparams:
            tmp=def_hparams.pop('gpu_no',None)
    ##############################################################
    while True:
        ### 월요일 추가테스트 필요
        #if not gpu_available: # deprecated 2021-08-30 because of cpu only params
        #    comm.send(None,dest=0,tag=tags.EXIT_RES) # train only gpus
        ###
        comm.send(None, dest=0, tag=tags.READY)
        #each_sub = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        dataset = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if dataset is not None:
            job_to_do = dataset[0]
            df = dataset[1]
            labels = dataset[2]
        tag = status.Get_tag()
        if tag == tags.START:
            # Do the work here
            print(">> Process (rank %d) on %s is running.." % (rank,name))
            score, n_cols, original_n_cols, model_size = model_score(gui_params,job_to_do,df,labels,def_hparams_small) # lightgbm params for cpus..
            job_to_do['score'] = score
            job_to_do['n_cols'] = n_cols
            job_to_do['original_n_cols'] = original_n_cols
            job_to_do['model_size'] = model_size
            #
            if score is not None:
                comm.send(job_to_do, dest=0, tag=tags.DONE)
            #comm.send(None, dest=0, tag=tags.READY)
        elif tag == tags.EXIT_REQ:
            print(">> Process (rank %d) on %s will waiting other process.." % (rank,name))
            comm.send(None, dest=0, tag=tags.EXIT_RES)
            break
    if rank == 0:
        while True:
            if provider.finished:
                print("ALL finishied")
                break

##########################################################################
## model score
## model_score()
##########################################################################





##########################################################################


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