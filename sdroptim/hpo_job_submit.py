"""
  Job Submit API for Hyper Parameter Optimization by Jeongcheol lee
  (for Jupyter User)
  -- jclee@kisti.re.kr
"""
from sdroptim.PythonCodeModulator import get_jobpath_with_attr, get_batch_script, from_userpy_to_mpipy, get_user_id
import json, requests, base64
from subprocess import (Popen, PIPE)
import optuna

def get_params(objective):
    '''
    retrieve_params_range_in_optuna_style_objective_functions
    e.g)
    params = get_params(objective = custom_objective_function)
    params
    {'RF_cv': {'low': 5.0, 'high': 5.0},
     'RF_n_estimators': {'low': 203.0, 'high': 1909.0},
     'RF_criterion': {'choices': ['gini', 'entropy']},
     'RF_min_samples_split': {'low': 0.257, 'high': 0.971},
     'RF_max_features': {'low': 0.081, 'high': 0.867},
     'RF_min_samples_leaf': {'low': 0.009, 'high': 0.453}}
    '''
    import inspect, ast,astunparse
    objective_strings=inspect.getsource(objective)
    p = ast.parse(objective_strings)
    for node in p.body[:]:
        if type(node) not in [ast.FunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef]:
            p.body.remove(node)
    if len(p.body)<1:
        raise ValueError("Objective function should be the python def/class style.")
    pre = astunparse.unparse(p)
    #    
    d = {}
    lines=pre.split("\n")
    for i in range(0,len(lines)):
        if 'trial.suggest_' in lines[i]:
            from_index=lines[i].index('(')
            to_index=lines[i].rindex(')') #last index
            target = lines[i][from_index+1:to_index]
            target.replace("'","").replace('"',"")
            targets=[x.replace(' ',"").replace('(',"").replace(')',"") for x in target.split(',')]
            target_name = targets[0].replace("'","")
            if 'trial.suggest_categorical' in lines[i]:
                cate_from_index=target.index('[')
                cate_to_index=target.index(']')
                cate_items=target[cate_from_index:cate_to_index+1].replace('"',"").replace("'","")
                cate_items=[x.strip().replace("'","") for x in cate_items[1:-1].split(',')]
                d.update({target_name:{"choices":cate_items}})
                #print(cate_items)
            else:
                if 'suggest_int' in lines[i]:
                    d.update({target_name:{"low":int(targets[1]),"high":int(targets[2])}})
                else:
                    d.update({target_name:{"low":float(targets[1]),"high":float(targets[2])}})
    return d

####################################
####################################
def check_stepwisefunc(objective):
    import os
    import inspect, ast,astunparse
    objective_strings=inspect.getsource(objective)
    p = ast.parse(objective_strings)
    for node in p.body[:]:
        if type(node) not in [ast.FunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef]:
            p.body.remove(node)
    if len(p.body)<1:
        raise ValueError("Objective function should be the python def/class style.")
    pre = astunparse.unparse(p)
    #    
    lines=pre.split("\n")
    for i in range(0,len(lines)):
        if node.name in lines[i]:
            if ', params):' in lines[i]:
                return True
            else:
                return False


def override_objfunc_with_newparams(objective, params=None):
    '''
    override_objfunc_with_newparams
    e.g)
    params = override_objfunc_with_newparams(objective = custom_objective_function)
    this function exploits values of the input params to override the input function.
    otherwise, if params are None, override the obj-func. into the stepwise-style obj-func.
    custom_objective_function(trial) -> custom_objective_function(trial, params)
    '''
    import os
    stepwise=False
    if params==None:
        params={}
        stepwise=True
    #
    import inspect, ast,astunparse
    objective_strings=inspect.getsource(objective)
    p = ast.parse(objective_strings)
    for node in p.body[:]:
        if type(node) not in [ast.FunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef]:
            p.body.remove(node)
    if len(p.body)<1:
        raise ValueError("Objective function should be the python def/class style.")
    
    pre = astunparse.unparse(p)
    #    
    d = {}
    lines=pre.split("\n")
    for i in range(0,len(lines)):
        if stepwise:
            if node.name in lines[i]:
                from_index=lines[i].index('(')
                if 'params' in lines[i]:
                    print("("+node.name + ") function is already stepwise-style.")
                    return 0
                else:
                    lines[i]=lines[i][:from_index+1]+"trial, params):"
        if 'trial.suggest_' in lines[i]:
            from_index=lines[i].index('(')
            to_index=lines[i].rindex(')')
            target = lines[i][from_index+1:to_index]
            target.replace("'","").replace('"',"")
            #print(target)
            targets=[x.replace(' ',"").replace('(',"").replace(')',"") for x in target.split(',')]
            target_name = targets[0].replace("'","")
            if 'trial.suggest_categorical' in lines[i]:
                cate_from_index=target.index('[')
                cate_to_index=target.rindex(']')
                cate_items=target[cate_from_index:cate_to_index+1].replace('"',"")
                cate_items=[x.strip().replace("'","") for x in cate_items[1:-1].split(',')]
                d.update({target_name:{"choices":cate_items}})
                if stepwise:
                    targets[1] = "params['"+target_name+"']['choices']"
                    mod_target=', '.join(["'"+target_name+"'",str(targets[1])])
                    lines[i]=(lines[i][:from_index+1]+mod_target+")")
                if target_name in params:
                    targets[1] = params[target_name]['choices']
                    mod_target=', '.join(["'"+target_name+"'",str(targets[1])])
                    lines[i]=(lines[i][:from_index+1]+mod_target+")")
            else:
                d.update({target_name:{"low":float(targets[1]),"high":float(targets[2])}})
                if stepwise:
                    targets[1] = "params['"+target_name+"']['low']"
                    targets[2] = "params['"+target_name+"']['high']"
                    mod_target=', '.join(["'"+target_name+"'",str(targets[1]),str(targets[2])])
                    lines[i]=(lines[i][:from_index+1]+mod_target+")")
                if target_name in params:
                    targets[1]=params[target_name]['low']
                    targets[2]=params[target_name]['high']
                    mod_target=', '.join(["'"+target_name+"'",str(targets[1]),str(targets[2])])
                    lines[i]=(lines[i][:from_index+1]+mod_target+")")
                    # d is current params
    prefix="global "+node.name+'\n'
    new_string = '\n'.join([x for x in lines if x is not ''])
    results = prefix+new_string
    p2=ast.parse(results)
    #exec(compile(p2, filename="<ast>", mode="exec"))
    exec(compile(p2, filename="___temp_module___.py", mode="exec"))
    try:
        with open('___temp_module___.py', 'w') as f:
            f.write(results)
    except:
        raise ValueError("___temp_module___.py cannot be generated!")
    #os.remove("___temp_module___.py")
    return results
#####################################
#####################################
def create_hpojob(study_name=None, workspace_name=None, job_directory=None, debug=False):
    return Job(study_name=study_name, workspace_name=workspace_name, job_directory=job_directory, debug=debug)

def load_hpojob(workspace_name=None, job_directory=None, debug=False):
    uname, each = get_user_id(debug=debug)
    cwd = os.getcwd()
    if workspace_name is None:
        if debug:
            wsname="ws_default"
        else:
            wsname=cwd.split('/workspace/')[1].split('/')[0]
    else:
        wsname = workspace_name
    ###
    if job_directory is None:
        raise ValueError("load_hpojob() requires job_directory(directory name). Try again.")
    jobpath = each+uname+'/workspace/'+str(wsname)+'/job/'+str(job_directory)
    if debug:
        jobpath = each+'/workspace/'+str(wsname)+'/job/'+str(job_directory)
    ###
    with open(jobpath+os.sep+"metadata.json") as data_file:
        gui_params = json.load(data_file)
    return Job(gui_params=gui_params)

class Job(object):
    def __init__(self,
                 study_name=None,
                 workspace_name=None,
                 job_directory=None,
                 env_name="my-rdkit-env",#env_name=None,
                 task_name="unknown_task",
                 algorithm="unknown_algo",
                 gui_params=None,
                 debug=False):
        # default setting
        self.debug=True if debug else False
        #
        if not gui_params:
            gui_params = {'kernel':'Python','task':task_name, 'algorithm':[algorithm],'hpo_system_attr':{}} # set default 
            #self.task_name = task_name
            #self.algorithm = algorithm
            if study_name is not None:
                gui_params['hpo_system_attr'].update({"study_name":study_name})
            if workspace_name is not None:
                gui_params['hpo_system_attr'].update({"workspace_name":workspace_name})
            if job_directory is not None:
                gui_params['hpo_system_attr'].update({"job_directory":job_directory})
            if env_name is not None:
                gui_params['hpo_system_attr'].update({"env_name":env_name})
                self.env_name=env_name
            jobpath, (uname, study_name, job_title, workspace_name, job_directory) = get_jobpath_with_attr(gui_params=gui_params, debug=debug)
            gui_params['hpo_system_attr'].update({'user_name':uname})
            self.job_path = jobpath
            gui_params['hpo_system_attr'].update({"job_path":self.job_path})
            self.study_name = study_name
            gui_params['hpo_system_attr'].update({"study_name":self.study_name})
            self.workspace_name = workspace_name
            gui_params['hpo_system_attr'].update({"workspace_name":self.workspace_name})
            self.job_directory = job_directory
            gui_params['hpo_system_attr'].update({"job_directory":self.job_directory})
            self.job_title = job_title
            gui_params['hpo_system_attr'].update({"job_name":self.job_title})
            #
            print("Job path: ", self.job_path)
            print("Workspace name: ", self.workspace_name)
            print("Job directory: ", self.job_directory)
            print("Study name: ", self.study_name)
            print("\n")
            #
            self.gui_params=gui_params
            jsonfile = json.dumps(gui_params)
            with open(jobpath+os.sep+'metadata.json', 'w') as f:
                f.write(jsonfile)
        else:
            #self.task_name = gui_params['task'] 
            #self.algorithm = gui_params['algorithm']
            self.job_path = gui_params['hpo_system_attr']['job_path']
            self.study_name = gui_params['hpo_system_attr']['study_name']
            self.job_title = gui_params['hpo_system_attr']['job_name']
            self.workspace_name = gui_params['hpo_system_attr']['workspace_name']
            self.job_directory = gui_params['hpo_system_attr']['job_directory']
            #
            print("Job path: ", self.job_path)
            print("Workspace name: ", self.workspace_name)
            print("Job directory: ", self.job_directory)
            print("Study name: ", self.study_name)
            print("\n")
            #
            self.n_nodes = int(gui_params['hpo_system_attr']['n_nodes'])
            self.max_sec = int(gui_params['hpo_system_attr']['time_deadline_sec'])
            #self.greedy = True if gui_params['hpo_system_attr']['greedy']==1 else False
            self.stepwise = True if gui_params['hpo_system_attr']['stepwise']==1 else False
            self.gui_params = gui_params
            if 'env_name' in gui_params['hpo_system_attr']:
                self.env_name = gui_params['hpo_system_attr']['env_name']
            self.direction = gui_params['hpo_system_attr']['direction']
    def optimize(self,
        objective,
        n_nodes=1,
        n_tasks=None,
        max_sec=300,
        direction='maximize',
        #greedy=True,
        stepwise=False,
        searching_space="searching_space",
        #task_type="gpu"
        ):
        ## configurations
        max_nodes = 4
        #
        if n_nodes>max_nodes:
            raise ValueError("The maximum number of n_nodes is 4. Try again.")
        else:
            self.n_nodes = n_nodes
        self.max_sec = max_sec
        #self.greedy = greedy
        if 'direction' in self.gui_params['hpo_system_attr']: # previous direction cannot be modified
            self.direction = self.gui_params['hpo_system_attr']['direction']
        else:
            self.direction= direction
        print("Study direction: "+self.direction)
        self.stepwise = stepwise
        self.searching_space = searching_space
        if n_tasks == None:
            self.n_tasks = self.n_nodes * 2
        else:
            self.n_tasks = n_tasks
        if self.n_tasks/self.n_nodes > 32:
            raise ValueError("Current n_tasks is too big. Each node can hold 32 CPU tasks max.")
        self.gui_params['hpo_system_attr'].update({'n_nodes':int(self.n_nodes)})
        self.gui_params['hpo_system_attr'].update({'n_tasks':int(self.n_tasks)})
        print("Note that currently the maximum n_nodes is 4 and each node has 32 CPU core and 2 GPU (nvidia P100).")
        print("(For example, 8 gpu tasks might be preocessed via 4 nodes (4 nodes * each 2 Gpu tasks). In this case, n_tasks is 8.)")
        print(str(self.n_nodes)+" nodes are preparing for this job ...")
        print(str(self.n_tasks)+" tasks are evenly distributed to each node.")
        print("This job will be terminated within "+str(self.max_sec)+" (sec) after beginning the job.")
        # update gui_params       
        self.gui_params['hpo_system_attr'].update({'time_deadline_sec':int(self.max_sec)})
        self.gui_params['hpo_system_attr'].update({'direction':self.direction})
        #self.gui_params['hpo_system_attr'].update({'greedy':1 if self.greedy == True else 0}) # greedy cannot be used in the jupyter-hpo job
        self.gui_params['hpo_system_attr'].update({'stepwise':1 if self.stepwise == True else 0})
        self.gui_params['hpo_system_attr'].update({'searching_space':searching_space+".json"})
        self.gui_params['hpo_system_attr'].update({"n_tasks":self.n_tasks})
        #
        params = get_params(objective)
        params_to_update = {self.gui_params['task']:{self.gui_params['algorithm'][0]:params}}
        params_to_update_json = json.dumps(params_to_update)
        with open(self.job_path+os.sep+self.gui_params['hpo_system_attr']['searching_space'], 'w') as f:
            f.write(params_to_update_json)
        print("A searching space jsonfile has been generated.")
        #
        mod_func_stepwise=""
        if stepwise:
            func_stepwise = check_stepwisefunc(objective)
            if not func_stepwise:
                mod_func_stepwise=override_objfunc_with_newparams(objective)
                if mod_func_stepwise:
                    print("The objective function has been overrided for using the stepwise strategy.")
        if self.debug:
            copied = copy_all_files_to_jobpath(cur_dir=os.getcwd(), dest_dir=self.job_path, by='copy')
        else:
            copied = copy_all_files_to_jobpath(cur_dir=os.getcwd(), dest_dir=self.job_path, by='copy')
            ##copied = copy_all_files_to_jobpath(cur_dir=os.getcwd(), dest_dir=self.job_path, by='symlink')
            #user_id = get_user_id(debug=self.debug)
            #__ = self.job_path.split(user_id[0])
            #dest_in_singularity_image = "/home/"+user_id[0]+__[1]
            #copied = copy_all_files_to_jobpath(cur_dir=os.getcwd(), dest_dir=dest_in_singularity_image, by='symlink')
        if copied:
            #print("Symlinks are generated in "+str(dest_in_singularity_image))
            print("Files are copied to the current job directory")
        gen_py_pathname = save_this_nb_to_py(dest_dir=self.job_path)
        if gen_py_pathname:
            print("This notebook has been copied as a python file(.py) successively.")
        generated_code = generate_mpipy(objective_name=objective.__name__, userpy=gen_py_pathname, postfunc=mod_func_stepwise)
        with open(self.job_path+os.sep+self.job_title+'_generated.py', 'w') as f:
            f.write(generated_code)
        if generated_code:
            print("The Python Script for submit a job has been generated successively.")
        jsonfile = json.dumps(self.gui_params)
        with open(self.job_path+os.sep+'metadata.json', 'w') as f:
            f.write(jsonfile)
        if jsonfile:
            print("metadata.json has been updated successively.")
        else:
            raise ValueError("metadata.json cannnot be generated.")
        #
        # submit phase
        print("\n")
        if self._request_submit_job():
            print("Job has been registerd to the portal database.")
        # make jobscript(.sh) using dejob_id        
        jobscripts= get_batch_script(gui_params=self.gui_params, debug=self.debug, dejob_id=self.dejob_id)
        jobshfile_path= self.job_path+os.sep+'job.sh'
        with open(jobshfile_path, 'w') as f:
            f.write(jobscripts)
        # Set permission to run the script
        os.chmod(jobshfile_path, 0o777)
        if jobscripts:
            print("job.sh has been generated successively.")
        #
        #
        if self._run_slurm_script():
            print("Job has been submitted !")
        ## 이후과정은 sbatch job.sh 실행하는 내용
        #results=run_job_script(user_name = self.gui_params['hpo_system_attr']['user_name'], dest_dir=self.jobpath)
        ## 8/20 계획 : getStudy 개체를 붙여서 dataframe을 jupyter에서 불러오고 이를 분석할 수 있어야함
        #print(results)
        ####
    def _request_submit_job(self):
        user_id = get_user_id(debug=self.debug)
        data = {
          'screenName': user_id[0],
          'title': self.job_title,
          'targetType': '82', # 82= HPO Job
          'workspaceName': self.workspace_name,
          'location': self.job_path 
        }
        if self.debug:
            print(data)        
        response = requests.post('https://sdr.edison.re.kr:8443/api/jsonws/SDR_base-portlet.dejob/studio-submit-de-job', data=data)
        if response.status_code == 200:            
            self.dejob_id = response.json()
            if self.debug:
                print(self.job_id)            
        else:
            raise ValueError("A problem occured when generating the job.")
        return True

    def _run_slurm_script(self):
        if hasattr(self, 'dejob_id'):
            user_id = get_user_id(debug=self.debug)
            #print(user_id)
            data = {
              'screenName': user_id[0],
              'location': self.job_path
            }
            if self.debug:
                print(data)
            print("Running Slurm script...")
            response = requests.post('https://sdr.edison.re.kr:8443/api/jsonws/SDR_base-portlet.dejob/slurm-de-job-run', data=data)
            if response.status_code == 200:
                while True: # waiting for file io
                    try:
                        with open(self.job_path+os.sep+"job.id", "r") as f:
                            self.job_id = int(f.readline())
                        break
                    except:
                        pass
                print("The job_id is "+str(self.job_id))
                return True
        else:
            raise ValueError("Slurm Job Not Found.")            
        # waiting for slurm job id
        #time.sleep(3)
        #try:
        #    with open(self.this_job_path+'/job.id','r') as f:
        #        idstr = f.readline()
        #    self.slurm_job_directory = int(idstr)
        #except:
        #    print("Slurm Job Not Found.")

    def _request_to_portal_stop_job(self):
        if hasattr(self, 'job_id'):
            user_id = get_user_id(debug=self.debug)
            data = {
              'jobId': self.job_id,
              'screenName': user_id[0]
            }
            response = requests.post('https://sdr.edison.re.kr:8443/api/jsonws/SDR_base-portlet.dejob/slurm-de-job-cancel', data=data)    
            print("Stop a Requested Job on the Portal.")
        else:
            raise ValueError("Slurm Job Not Found. Job cancel failed.")
    
    def stop(self):
        self._request_to_portal_stop_job()

    def get_study(self):
        return self._get_study_object(types = "study")

    def get_study_dataframe(self):
        df = self._get_study_object(types = "dataframe")
        self.df = df
        print("Current study dataframe can be founded at '{job}.df'")
        return df

    def _get_study_object(self, types):
        if hasattr(self, 'job_id'):
            user_id = get_user_id(debug=self.debug)
        key = str(base64.b64decode('cG9zdGdyZXNxbDovL3Bvc3RncmVzOnBvc3RncmVzQDE1MC4xODMuMjQ3LjI0NDo1NDMyLw=='))[2:-1]
        url = key + self.user_id[0]
        try:
            rdbs = optuna.storages.RDBStorage(url)
            study_id = rdbs.get_study_id_from_name(self.study_name)
        except:
            raise ValueError(
                    "Cannot find the study_name {}".format(
                        self.study_name#, args.storage_url
                    ))
        try:
            s = optuna.create_study(load_if_exists=True, study_name=self.study_name, storage=url, direction='minimize')
        except:
            s = optuna.create_study(load_if_exists=True, study_name=self.study_name, storage=url, direction='maximize')
        
        if types == "study":
            return s
        elif types == "dataframe":
            return s.trials_dataframe()

    def show_logs(self, show_type='both'):
        if show_type == 'output':
            show_list = ['out']
        elif show_type == 'error':
            show_list = ['err']
        else:
            show_list = ['out','err']
        for each in show_list:
            print("** print std."+each)
            with open(self.job_path+os.sep+'std.'+each) as f:
                temp=f.readlines()
            print("".join(temp))
            print("\n")

    def show_job_status(self):
        if hasattr(self, 'job_id'):
            command = "scontrol show job "+str(self.job_id)
            stdout, stderr = self._execute_subprocess(command)
            print(stdout)
        else:
            raise ValueError("Slurm Job Not Found. Show job status failed.")

    def show_job_queue(self):
        stdout, stderr = self._execute_subprocess("squeue")
        print(stdout)

    def _execute_subprocess(self, command):
        process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        stdout, stderr = process.communicate()
        return stdout.decode("utf-8"), stderr.decode("utf-8")

#####################################
#####################################

def generate_mpipy(objective_name, userpy, postfunc=""):
    import ast, astunparse
    try:
        with open(userpy) as f:
            p = ast.parse(f.read())
        for node in p.body[:]:
            if type(node) not in [ast.FunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef]:
                p.body.remove(node)
        pre = astunparse.unparse(p)
        pre+="\n"+postfunc
        pre+="\n\n"
        body ='if __name__ == "__main__":\n'
        body+='    import optuna\n'
        body+='    import sdroptim\n'
        body+='    stepwise, task_and_algorithm = sdroptim.check_stepwise_available("metadata.json")\n'
        body+='    args = sdroptim.get_argparse(automl=True, json_file_name="metadata.json")\n'
        #
        post ='    if stepwise:\n'
        post+='        sdroptim.stepwise_mpi_time('+objective_name+', args, task_and_algorithm)\n'
        post+='    else:\n'
        post+='        sdroptim.optuna_mpi('+objective_name+', args)\n'
        return pre+body+post
    except:
        raise ValueError("The Python Script for submit a job cannot be generated.")

def SubmitHPOjob(objective_or_setofobjectives, args):
    ''' 1. file copying( symbolic link )
        2. generates metadata.json, search_space.json, job.sh
        3. run job.sh
    '''
    from inspect import isfunction
    if isfunction(objective_or_setofobjectives):
        n_obj = 1
    else:
        n_obj = len(objective_or_setofobjectives)
    if n_obj > 2:
        raise ValueError("The number of objectives cannot be exceed by two.")
    objective_name_list = []
    for each_obj in objective_or_setofobjectives:
        objective_name_list.append(each_obj.__name__)
    ##################################################
    if args.job_directory == "": ## generate job_directory(directory)
        jobpath, (uname, sname, job_title, wsname, job_directory) = get_jobpath_with_attr()
        args.update({'jobpath':jobpath})
        args.update({'uname':uname})
        args.update({'sname':sname})
        args.update({'job_title':job_title})
        args.update({'wsname':wsname})
        args.update({'job_directory':job_directory})
        copy_all_files_to_jobpath(cur_dir=os.getcwd(), dest_dir=jobpath, by='symlink')
    else:
        with open(args.metadata) as data_file:
            gui_params = json.load(data_file)
        jobpath = gui_params['hpo_system_attr']['job_directory']
    #######
    gen_py_pathname=save_this_nb_to_py(dest_dir=jobpath) # should run in jupyter only
    # 1. generates gui_params and its metadata.json
    if args.metadata_json == "": ## first try
        if args.task_name == "":
            args.task_name = "unknown_task"
        if args.algorithm_name == "":
            args.algorithm_name = "unknown_algorithm"
        generates_metadata_json(args=args, dest_dir=jobpath)
    else: ## update metadata.json with the new args
        #update_metadata_json(args=args, dest_dir=jobpath)
        generates_metadata_json(args=args, dest_dir=jobpath) # 항상 최신의 args로 update하면 될듯?
    ## 2. load gui_params
    with open(jobpath+os.sep+"metadata.json") as data_file:
        gui_params = json.load(data_file)
    ##
    generated_code = from_userpy_to_mpipy(objective_name_list=objective_name_list, args=args, userpy=gen_py_pathname)
    with open(jobpath+os.sep+args.job_title+'_generated.py', 'w') as f:
        f.write(generated_code)
    # 생성된 py에서 함수만 호출(class, def) -> 이전 함수 활용
    # 그리고 실행 함수 제작(mpirun 용)
    # 그리고나서 만들어진 metadata이용하여 batch script 생성
    jobscripts= get_batch_script(gui_params)
    with open(jobpath+os.sep+'job.sh', 'w') as f:
        f.write(jobscripts)
    ##
    ## 이후과정은 sbatch job.sh 실행하는 내용
    #results=run_job_script(user_name=gui_params['hpo_system_attr']['user_name'], dest_dir=jobpath)
    #print(results)

def run_job_script(user_name, dest_dir):
    import requests, shlex, subprocess
    curl_script = 'curl https://sdr.edison.re.kr:8443/api/jsonws/SDR_base-portlet.dejob/slurm-de-job-run \\ '   
    curl_script+= '-d location='+dest_dir+' \\ '
    curl_script+= '-d screenName='+user_name
    args = shlex.split(curl_script)
    process=subprocess.Popen(args,shell=False,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr=process.communicate()
    return stdout


def generates_metadata_json(args, dest_dir):
    if len(args.algorithm_name.split(','))==1:
        algorithms = '"'+args.algorithm_name+'"'
    elif len(args.algorithm_name.split(','))>1:
        each_algos = args.algorithm_name.split(',')
        each_algos = ['"'+x.strip()+'"' for x in each_algos]
        algorithms = ','.join(each_algos)
    #
    results = '{"kernel":"Python", "task":"'+args.task_name+'", "algorithm":['+algorithms+'],\n'
    results+= '\t"hpo_system_attr":{"user_name":"'+args.uname+'", "study_name":"'+(args.sname if args.study_name == "" else args.study_name)+'", '
    if args.env_name:
        env_name = '"env_name":"'+args.env_name+'", '
    else:
        env_name = ""
    results+= '"job_name":"'+args.job_title+'", '+env_name+'"workspace_name":"'+args.wsname+'", "job_directory":"'+args.job_directory+'", '
    results+= '"time_deadline_sec": '+str(args.max_sec)+', "n_nodes":'+str(args.n_nodes)+', '
    results+= '"greedy":'+('0' if not args.greedy else '1')+', "stepwise":'+('0' if not args.stepwise else '1') + ', '
    results+= '"top_n_all":'+str(args.top_n_all)+ ', "top_n_each_algo":'+str(args.top_n_each_algo)+'}\n'
    #
    results+= '\n}'
    try:
        with open(dest_dir+os.sep+"metadata.json", 'w') as f:
            f.write(results)
        print("Successively generated metadata jsonfile! -> metadata.json")
        token=True
    except:
        print("Cannot generate metadata jsonfile!")
        token=False
    return token

#def update_metadata_json(args, dest_dir):
#    # load previous metadata_json and update it
#    with open(dest_dir+os.sep+"metadata.json") as data_file:
#        gui_params = json.load(data_file)
    

############################################################################
#######
# file exists error need to be handled
def copy_all_files_to_jobpath(cur_dir, dest_dir, by='symlink'):
    if by == 'symlink':
        for item in os.listdir(cur_dir):
            try:
                os.symlink(cur_dir+os.sep+item, dest_dir+os.sep+item)
                return True
            except:
                raise ValueError("Symlinks cannot be generated.")
    elif by == 'copy':
        try:
            copytree(cur_dir, dest_dir)
            return True
        except:
            raise ValueError("Files cannot be copied.")

######################################
#
#def current_notebook_name():
#    import ipyparams
#    notebook_name = ipyparams.notebook_name
#    return notebook_name
#def save_this_nb_to_py(args, dest_dir="./"):
#    import subprocess
#    if args.nb_name=="":
#        name= current_notebook_name()
#        filepath = os.getcwd()+os.sep+name
#        ipynbfilename=name
#    else:
#        filepath = os.getcwd()+os.sep+args.nb_name
#        ipynbfilename=args.nb_name
#    try:
#        #!jupyter nbconvert --to script {filepath} --output-dir={dest_dir}
#        subprocess.check_output("jupyter nbconvert --to script "+filepath+" --output-dir="+dest_dir, shell=True)
#        return dest_dir+os.sep+ipynbfilename.split(".ipynb")[0]+'.py'
#    except:
#        raise ValueError(".py cannot be generated via current notebook.")
#    return False

def get_notebook_name():
    from time import sleep
    from IPython.display import display, Javascript
    import subprocess
    import os
    import uuid
    magic = str(uuid.uuid1()).replace('-', '')
    print(magic)
    # saves it (ctrl+S)
    display(Javascript('IPython.notebook.save_checkpoint();'))
    nb_name = None
    while nb_name is None:
        try:
            sleep(0.1)
            nb_name = subprocess.check_output(f'grep -l {magic} *.ipynb', shell=True).decode().strip()
        except:
            pass
    return nb_name

 
def save_this_nb_to_py(dest_dir="./"):
    import subprocess
    name= get_notebook_name()
    filepath = os.getcwd()+os.sep+name
    ipynbfilename=name
    try:
        #!jupyter nbconvert --to script {filepath} --output-dir={dest_dir}
        subprocess.check_output("jupyter nbconvert --to script "+filepath+" --output-dir="+dest_dir, shell=True)
        return dest_dir+os.sep+ipynbfilename.split(".ipynb")[0]+'.py'
    except:
        raise ValueError(".py cannot be generated via current notebook.")
    return False



######################################
import os
import shutil
import stat
def copytree(src, dst, symlinks = False, ignore = None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                pass # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)