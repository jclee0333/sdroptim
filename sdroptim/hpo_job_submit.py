"""
  Job Submit API for Hyper Parameter Optimization by Jeongcheol lee
  (for Jupyter User)
  -- jclee@kisti.re.kr
"""
from sdroptim.PythonCodeModulator import get_jobpath_with_attr, get_batch_script, from_userpy_to_mpipy
import json

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
                print(cate_items)
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
def create_hpojob(study_name=None, workspace_name=None, job_id=None):
    return Job(study_name, workspace_name, job_id)

def load_hpojob(workspace_name=None, job_id=None):
    user_home1 = "/EDISON/SCIDATA/sdr/draft/"
    user_home2 = "/science-data/sdr/draft/"
    user_home3_for_test = "C:\\Users\\user\\Documents\\GitHub\\"
    user_homes = [user_home1, user_home2, user_home3_for_test]
    cwd=os.getcwd()
    find_token = False
    each = ""
    for each in user_homes:
        if cwd.startswith(each):
            try:
                uname = cwd.split(each)[1].split('/')[0]
                find_token=True
                break
            except:
                pass
    if not find_token:
        raise ValueError("cannot find user_id, please check the current user directory.")
    ###
    if workspace_name is None:
        if cwd.startswith(user_home3_for_test):
            wsname="test"
        else:
            wsname=cwd.split('/workspace/')[1].split('/')[0]
    else:
        wsname = workspace_name
    ###
    if job_id is None:
        raise ValueError("load_hpojob() requires job_id(directory name). Try again.")
    jobpath = each+uname+'/workspace/'+str(wsname)+'/job/'+str(job_id)
    ###
    with open(jobpath+os.sep+"metadata.json") as data_file:
        gui_params = json.load(data_file)
    return Job(gui_params=gui_params)

class Job(object):
    def __init__(self,
                 study_name=None,
                 workspace_name=None,
                 job_id=None,
                 env_name=None,
                 task_name="unknown_task",
                 algorithm="unknown_algo",
                 gui_params=None):
        if not gui_params:
            gui_params = {'kernel':'Python','task':task_name, 'algorithm':[algorithm],'hpo_system_attr':{}} # set default 
            #self.task_name = task_name
            #self.algorithm = algorithm
            if study_name is not None:
                gui_params['hpo_system_attr'].update({"study_name":study_name})
            if workspace_name is not None:
                gui_params['hpo_system_attr'].update({"workspace_name":workspace_name})
            if job_id is not None:
                gui_params['hpo_system_attr'].update({"job_id":job_id})
            if env_name is not None:
                gui_params['hpo_system_attr'].update({"env_name":env_name})
                self.env_name=env_name
            jobpath, (uname, study_name, jname, workspace_name, job_id) = get_jobpath_with_attr(gui_params)
            gui_params['hpo_system_attr'].update({'user_name':uname})
            self.job_path = jobpath
            gui_params['hpo_system_attr'].update({"job_path":self.job_path})
            self.study_name = study_name
            gui_params['hpo_system_attr'].update({"study_name":self.study_name})
            self.workspace_name = workspace_name
            gui_params['hpo_system_attr'].update({"workspace_name":self.workspace_name})
            self.job_id = job_id
            gui_params['hpo_system_attr'].update({"job_id":self.job_id})
            self.job_name = jname
            gui_params['hpo_system_attr'].update({"job_name":self.job_name})
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
            self.job_name = gui_params['hpo_system_attr']['job_name']
            self.workspace_name = gui_params['hpo_system_attr']['workspace_name']
            self.job_id = gui_params['hpo_system_attr']['job_id']
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
        max_sec=300,
        direction='maximize',
        #greedy=True,
        stepwise=False,
        searching_space="searching_space"
        ):
        #
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
        # update gui_params
        self.gui_params['hpo_system_attr'].update({'n_nodes':int(self.n_nodes)})
        print(str(self.n_nodes)+" nodes are preparing for this job ...")
        self.gui_params['hpo_system_attr'].update({'time_deadline_sec':int(self.max_sec)})
        print("This job will be terminated within "+str(self.max_sec)+" (sec) after beginning the job.")
        self.gui_params['hpo_system_attr'].update({'direction':self.direction})
        #self.gui_params['hpo_system_attr'].update({'greedy':1 if self.greedy == True else 0})
        self.gui_params['hpo_system_attr'].update({'stepwise':1 if self.stepwise == True else 0})
        self.gui_params['hpo_system_attr'].update({'searching_space':searching_space+".json"})
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
        copied = copy_all_files_to_jobpath(cur_dir=os.getcwd(), dest_dir=self.job_path, by='symlink')
        if copied:
            print("Symlinks are generated in "+str(self.job_path))
        gen_py_pathname = save_this_nb_to_py(dest_dir=self.job_path)
        if gen_py_pathname:
            print("This notebook ("+str(gen_py_pathname)+") has been copied as a python file(.py) ,successively.")
        generated_code = generate_mpipy(objective_name=objective.__name__, userpy=gen_py_pathname, postfunc=mod_func_stepwise)
        with open(self.job_path+os.sep+self.job_name+'_generated.py', 'w') as f:
            f.write(generated_code)
        if generated_code:
            print("The Python Script for submit a job has been generated successively.")
        jsonfile = json.dumps(self.gui_params)
        with open(self.job_path+os.sep+'metadata.json', 'w') as f:
            f.write(jsonfile)
        if jsonfile:
            print("metadata.json has been updated successively.")
        #
        jobscripts= get_batch_script(self.gui_params)
        with open(self.job_path+os.sep+'job.sh', 'w') as f:
            f.write(jobscripts)
        if jobscripts:
            print("job.sh has been generated successively.")
        #
        ## 이후과정은 sbatch job.sh 실행하는 내용
        #results=run_job_script(user_name = self.gui_params['hpo_system_attr']['user_name'], dest_dir=self.jobpath)
        #print(results)

#####################################
#####################################

def generate_mpipy(objective_name, userpy, postfunc=""):
    import ast, astunparse
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
    if args.job_id == "": ## generate job_id(directory)
        jobpath, (uname, sname, jname, wsname, job_id) = get_jobpath_with_attr()
        args.update({'jobpath':jobpath})
        args.update({'uname':uname})
        args.update({'sname':sname})
        args.update({'jname':jname})
        args.update({'wsname':wsname})
        args.update({'job_id':job_id})
        copy_all_files_to_jobpath(cur_dir=os.getcwd(), dest_dir=jobpath, by='symlink')
    else:
        with open(args.metadata) as data_file:
            gui_params = json.load(data_file)
        jobpath = gui_params['hpo_system_attr']['job_id']
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
    with open(jobpath+os.sep+args.jname+'_generated.py', 'w') as f:
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
    results+= '"job_name":"'+args.jname+'", '+env_name+'"workspace_name":"'+args.wsname+'", "job_id":"'+args.job_id+'", '
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