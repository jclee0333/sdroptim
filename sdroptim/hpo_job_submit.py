"""
  Job Submit API for Hyper Parameter Optimization by Jeongcheol lee
  (for Jupyter User)
  -- jclee@kisti.re.kr
"""
from sdroptim.PythonCodeModulator import get_jobpath_with_attr, get_batch_script

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
    save_this_nb_to_py(dest_dir=jobpath):
    # 1. generates gui_params and its metadata.json
    if args.metadata_json == "":
        if args.task_name == "":
        	args.task_name = "unknown_task"
        if args.algorithm_name == "":
        	args.algorithm_name = "unknown_algorithm"
        generates_metadata_json(args=args, dest_dir=jobpath)

    #
    generated_code = PyMod.from_gui_to_code(gui_params)        
    with open(jobpath+os.sep+jname+'_generated.py', 'w') as f:
        f.write(prefix_generated_code+generated_code)
    #
    jobscripts= get_batch_script(gui_params, new_job=False)
    with open(jobpath+os.sep+'job.sh', 'w') as f:
        f.write(jobscripts)
    ##

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
    results+= '"job_name":"'+args.jname+'", '+env_name+'"workspace_name":"'+args.wsname+'", "job_id":"'+job_id+'", '
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






############################################################################
#######
# file exists error need to be handled
def copy_all_files_to_jobpath(cur_dir, dest_dir, by='symlink'):
    if by == 'symlink':
        for item in os.listdir(cur_dir):
            try:
                os.symlink(cur_dir+os.sep+item, dest_dir+os.sep+item)
            except:
                pass
    elif by == 'copy':
        copytree(cur_dir, dest_dir)

######################################

def current_notebook_name():
    import ipyparams
    notebook_name = ipyparams.notebook_name
    return notebook_name
 

def save_this_nb_to_py(dest_dir="./"):
    name= current_notebook_name()
    filepath = os.getcwd()+os.sep+name
    try:
        !jupyter nbconvert --to script {filepath} --output-dir={dest_dir}
        return True
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