"""
  MPI role for Hyper Parameter Optimization by Jeongcheol lee
  -- jclee@kisti.re.kr
"""


import json, copy,os, time
import optuna
from mpi4py import MPI
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
    gpu_no = abs(rank%2-1)
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
            print("??????",tag)
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
    gpu_no = abs(rank%2-1)
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
    gpu_no = abs(rank%2-1)
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
