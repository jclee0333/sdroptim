'''
# GUI based automatic Python code generator for Machine Learning

implemented by Jeongcheol Lee
0% ver 0.05 @ 2018. 8. 22. -
modified by Jeongcheol Lee
ver 0.1 @ 2020. 6. 10 - basic algorithm converting (SVM, RF, MLR, BT)
ver 0.2 @ 2020. 6. 15 - mpirun available
ver 0.3 @ 2020. 6. 16 - add DL(FFN) (pytorch)-gpu, ####lightgbm, xgboost not yet
ver 0.4 @ 2020. 6. 23 - add DL(CNN) (pytorch)-gpu, load_data (indirect type)
ver 0.5 @ 2020. 7. 3  - add LightGBM(gpu), XGBoost(gpu)
ver 0.6 @ 2020. 7. 10 - applying stepwise
ver 0.7 @ 2020. 7. 22 - add sbatch script generator (get_batch_script())
'''
### configurations n_jobs per process

n_jobs = 15 # cpu jobs

##############################################################################

import json, uuid, os, datetime, base64
from sdroptim.searching_strategies import generate_default_searching_space_file

def from_userpy_to_mpipy(objective_name_list, userpy):
    import ast, astunparse
    with open(userpy) as f:
        p = ast.parse(f.read())
    for node in p.body[:]:
        if type(node) not in [ast.FunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef]:
            p.body.remove(node)
    #objective_name_list = []
    #for node in p.body[:]:
    #    if type(node) in [ast.FunctionDef, ast.ClassDef]:
    #        if 'objective' in node.name.lower():
    #            objective_name_list.append(node.name)
    if len(objective_name_list)>2:
        raise ValueError("Objective Functions cannot exceed by two.")
    pre = astunparse.unparse(p)
    pre+="\n\n"
    body ='if __name__ == "__main__":\n'
    body+='    import optuna\n'
    body+='    import sdroptim\n'
    body+='    stepwise, task_and_algorithm = sdroptim.check_stepwise_available("metadata.json")\n'
    body+='    args = sdroptim.get_argparse(automl=True, json_file_name="metadata.json")\n'
    #
    if args.task_type == 'both':
        post ='    if stepwise:\n'
        post+='        sdroptim.stepwise_mpi_time_dobj('+objective_name_list[0]+', '+objective_name_list[1]+', args, task_and_algorithm)\n'
        post+='    else:\n'
        post+='        sdroptim.optuna_mpi_dobj('+objective_name_list[0]+', '+objective_name_list[1]+', args)\n'
    else:
        post ='    if stepwise:\n'
        post+='        sdroptim.stepwise_mpi_time('+objective_name_list[0]+', args, task_and_algorithm)\n'
        post+='    else:\n'
        post+='        sdroptim.optuna_mpi('+objective_name_list[0]+', args)\n'
    return pre+body+post

def get_user_id(debug=False):
    user_home1 = str(base64.b64decode(b'L0VESVNPTi9TQ0lEQVRBL3Nkci9kcmFmdC8='))[2:-1]
    user_home2 = str(base64.b64decode(b'L3NjaWVuY2UtZGF0YS9zZHIvZHJhZnQv'))[2:-1]
    user_homes = [user_home1, user_home2]
    cwd=os.getcwd()
    if debug:
        uname = cwd.split(os.sep)[-1]
        each = cwd
        return uname, each
    #
    each = ""
    uname = ""
    cannot_find=False
    for each in user_homes:
        if cwd.startswith(each):
            try:
                uname = cwd.split(each)[1].split('/')[0]
                return uname, each
            except:
                cannot_find=True
                in_user_home_list=True
        else:
            cannot_find=True
            in_user_home_list=False
    if cannot_find:         
        raise ValueError(("The current user directory cannot be founded in the pre-defined userhome list. " if in_user_home_list else "")+
            "Failed to find user_id, please check the current user directory.")

def get_jobpath_with_attr(gui_params=None, debug=False):
    if not gui_params:
        gui_params = {'hpo_system_attr':{}} # set default 
    cwd=os.getcwd()
    uname, each = get_user_id(debug=debug) # each == user home( under workspace )
    #########################################################################
    if debug:
        if not os.path.exists(cwd+os.sep+"workspace/"):
            os.mkdir(cwd+os.sep+"workspace/")
        if not os.path.exists(cwd+os.sep+"workspace/default_ws/"):
            os.mkdir(cwd+os.sep+"workspace/default_ws/")
        if not os.path.exists(cwd+os.sep+"workspace/default_ws/job/"):
            os.mkdir(cwd+os.sep+"workspace/default_ws/job/")
        
        if 'job_directory' in gui_params['hpo_system_attr']:
            job_directory=gui_params['hpo_system_attr']['job_directory'] # directory name
            jobpath = cwd+os.sep+"workspace/default_ws/job/"+job_directory
        else: # if it is first try -> generate it
            timenow = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            job_directory = "job-"+timenow
            jobpath = cwd+os.sep+"workspace/default_ws/job/job-"+timenow
        if not os.path.exists(jobpath):
            os.mkdir(jobpath)
        sname=gui_params['hpo_system_attr']['study_name'] if 'study_name' in gui_params['hpo_system_attr'] else str(uuid.uuid4())        
        job_title = sname+"_in_"+uname
        wsname = "default_ws"
        return jobpath, (uname, sname, job_title, wsname, job_directory)
    ########################################################################        
    # otherwise, use 'user_name' in the params
    if 'user_name' in gui_params['hpo_system_attr']:
        uname=gui_params['hpo_system_attr']['user_name'] 
    sname=gui_params['hpo_system_attr']['study_name'] if 'study_name' in gui_params['hpo_system_attr'] else str(uuid.uuid4())
    job_title=sname+"_in_"+uname
    ##########################
    if 'workspace_name' in gui_params['hpo_system_attr']:
        wsname=gui_params['hpo_system_attr']['workspace_name'] # directory name (MANDATORY)    
    else:
        wsname=cwd.split('/workspace/')[1].split('/')[0]
    ###########################
    if 'job_directory' in gui_params['hpo_system_attr']:
        job_directory=gui_params['hpo_system_attr']['job_directory'] # directory name
    else:
        timenow = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        job_directory="job-"+timenow
    if not os.path.exists(each+uname+'/workspace/'+str(wsname)+'/job/'):
        os.mkdir(each+uname+'/workspace/'+str(wsname)+'/job/')
    jobpath = each+uname+'/workspace/'+str(wsname)+'/job/'+str(job_directory)
    if not os.path.exists(jobpath):
        os.mkdir(jobpath)
    return jobpath, (uname, sname, job_title, wsname, job_directory)

def get_batch_script(gui_params, debug=False):
    jobpath, (uname, sname, job_title, wsname, job_directory) = get_jobpath_with_attr(gui_params=gui_params, debug=debug)
    ###########################
    #
    time_deadline_sec = gui_params['hpo_system_attr']['time_deadline_sec']
    #
    #if 'algorithm' in gui_params:
    #    cpuhas = getAlgorithmListAccordingToResourceType(gui_params['algorithm'], 'cpu')
    #    gpuhas = getAlgorithmListAccordingToResourceType(gui_params['algorithm'], 'gpu')
    #else:
    #    if 'n_nodes' in gui_params['hpo_system_attr']:
    #        if 'task_type' in gui_params['hpo_system_attr']:
    #            if gui_params['hpo_system_attr']['task_type']=='cpu':
    #                cpuhas=1
    #                gpuhas=0
    #            elif gui_params['hpo_system_attr']['task_type']=='gpu':
    #                cpuhas=0
    #                gpuhas=1
    #            elif gui_params['hpo_system_attr']['task_type']=='both':
    #                cpuhas=1
    #                gpuhas=1
    #            else:
    #                raise ValueError("A custom job should clarify the 'task_type' in the argument params={ 'hpo_system_attr':{'task_type': ~ } } ('cpu', 'gpu', or 'both')")
    #    else:
    #        raise ValueError("A custom job should clarify the 'n_nodes' in the argument params={ 'hpo_system_attr':{'n_nodes': (int) } } ")
    cpuhas=[]
    gpuhas=[]
    if 'algorithm' in gui_params:
        cpuhas = getAlgorithmListAccordingToResourceType(gui_params['algorithm'], 'cpu')
        gpuhas = getAlgorithmListAccordingToResourceType(gui_params['algorithm'], 'gpu')
    if 'n_nodes' not in gui_params['hpo_system_attr']:
        raise ValueError("A custom job should clarify the 'n_nodes' in the argument params={ 'hpo_system_attr':{'n_nodes': (int) } } ")        
    cpu_task = 1
    gpu_task = 1
    if len(cpuhas)>0:
        cpu_task = 2
    if len(gpuhas)>0:
        gpu_task = 2
    n_nodes = gui_params['hpo_system_attr']['n_nodes']
    ntasks = n_nodes*cpu_task*gpu_task # ntasks calculation for GUI-hpo
    if 'n_task' in gui_params['hpo_system_attr']: # n_tasks for jupyter-hpo
        n_tasks = gui_params['hpo_system_attr']['n_tasks']
    #
    prefix ='#!/bin/bash\n'
    prefix+='#SBATCH --job-name='+ job_title +'\n'
    prefix+='#SBATCH --output=/EDISON/SCIDATA/sdr/draft/'+uname+'/workspace/'+str(wsname)+'/job/'+str(job_directory)+'/std.out\n'
    prefix+='#SBATCH --error=/EDISON/SCIDATA/sdr/draft/'+uname+'/workspace/'+str(wsname)+'/job/'+str(job_directory)+'/std.err\n'
    prefix+='#SBATCH --nodes='+str(n_nodes)+'\n'
    prefix+='#SBATCH --ntasks='+str(ntasks)+'\n'
    prefix+='#SBATCH --ntasks-per-node='+str(int(ntasks/n_nodes))+'\n'
    #prefix+='#SBATCH --gres=gpu:'+str(n_gpu)+'\n'
    
    timed=datetime.timedelta(seconds=time_deadline_sec)
    n_days = timed.days
    rest_seconds = timed.seconds# + 300 # marginal seconds (5min)
    timed_without_days=datetime.timedelta(seconds=rest_seconds)
    rval=str(n_days)+"-"+str(timed_without_days)
    #
    prefix+='#SBATCH --time='+ rval +'\n' # e.g., 34:10:33
    prefix+='#SBATCH --exclusive\n'
    paths = 'HOME=/EDISON/SCIDATA/sdr/draft/'+uname+'\n'
    jobdir= 'JOBDIR=/home/'+uname+'/workspace/'+str(wsname)+'/job/'+str(job_directory)+'\n' # path in singularity image (after mounting)
    paths += jobdir
    #
    types = "scripts" if 'env_name' in gui_params['hpo_system_attr'] else "python"
    #
    if types=="scripts":
        if 'env_name' in gui_params['hpo_system_attr']:
            env_name = gui_params['hpo_system_attr']['env_name']
            env_script = "source activate "+env_name + "\n"
        else:
            env_script = ""
        with open(jobpath+os.sep+job_title+"_running_with_custom_env.sh", 'w') as f:
            sh_scripts = jobdir+env_script +"cd ${JOBDIR}\npython ${JOBDIR}/"+job_title+"_generated"+".py\n"
            f.write(sh_scripts)
    ## JOB init @ portal // modified 0812 --> deprecated @ 0.1.1 -> used in register function
    job_init ="\n## JOB init @ portal\n"
    job_init+="deJobId=$(curl https://sdr.edison.re.kr:8443/api/jsonws/SDR_base-portlet.dejob/studio-submit-de-job \\ "
    #job_init+="-d userId="+str(gui_params['hpo_system_attr']['userId'])+" \\ "
    #job_init+="-d groupId="+str(gui_params['hpo_system_attr']['groupId'])+" \\ "
    #job_init+="-d companyId="+str(gui_params['hpo_system_attr']['companyId'])+" \\ "
    job_init+="-d screenName="+uname+" \\ "
    job_init+="-d title="+job_title+" \\ "
    job_init+="-d targetType=82 \\ " # 82 = HPO job
    job_init+="-d workspaceName="+wsname+" \\ "
    #job_init+="-d status=TRAINING \\ "
    job_init+="-d location="+jobpath+")\n\n"
    ##### mpirun command
    mpirun_command = "## mpirun command\n"
    mpirun_command+= "/usr/local/bin/mpirun -np " + str(ntasks)
    mpirun_options = "-x PATH -x HOROVOD_MPI_THREADS_DISABLE=1 -x NCCL_SOCKET_IFNAME=^docker0,lo -mca btl_tcp_if_exclude lo,docker0  -mca pml ob1"
    ##### singularity command
    singularity_command = "singularity exec --nv"
    user_home_mount_for_custom_enviromnent = "-H ${HOME}:"+"/home/"+uname        # final
    #user_home_mount_for_custom_enviromnent = "-H /home/"+uname+":"+"/home/"+uname  # my custom
    user_jobdir_mount = ""#"-B ${JOBDIR}:${JOBDIR}"                               # final
    #user_jobdir_mount = "-B /home/jclee/automl_jclee:/${JOBDIR}"                 # my custom
    singularity_image = "/EDISON/SCIDATA/singularity-images/userenv3"
    running_command = ("python ${JOBDIR}/"+job_title+"_generated"+".py") if types == "python" else ("/bin/bash ${JOBDIR}/"+job_title+"_running_with_custom_env.sh")
    ## JOB done @ portal
    job_done = "## JOB done @ portal\n"
    job_done+= "curl https://sdr.edison.re.kr:8443/api/jsonws/SDR_base-portlet.dejob/studio-update-status \\ "
    #if 'deJobId' in gui_params['hpo_system_attr']:
    #    job_done+="-d deJobId="+str(gui_params['hpo_system_attr']['deJobId'])+" \\ "
    job_done+="-d deJobId=${deJobID} -d Status=SUCCESS\n"
    results = prefix+paths+(job_init if 'n_tasks' not in gui_params['hpo_system_attr'] else "")+mpirun_command+ " " + mpirun_options + " " + singularity_command + " " + user_home_mount_for_custom_enviromnent+ " " + user_jobdir_mount + " " +singularity_image+" " + running_command + "\n\n"+job_done
    # job_init can be added when gui-hpo, while jupyter-hpo exploits its own python-api _request_submit_job()
    return results    
#    

##############################################################################################

def from_gui_to_code(gui_params):
    prev = ""
    body = ""
    post = ""
    if gui_params:  # if it exists
        hpo = True if type(gui_params['algorithm']) is list and (len(gui_params['algorithm']) >= 1) else False
        if 'ml_file_info' in gui_params:
            indirect = True if (gui_params['ml_file_info']['info_type'] == 'Indirect') else False
        else:
            indirect = False
        stepwise = False
        if 'hpo_system_attr' in gui_params:
            if 'greedy' in gui_params['hpo_system_attr']:
                greedy = False if (gui_params['hpo_system_attr']['greedy'] == 0) else True
            else:
                greedy = True
            if 'stepwise' in gui_params['hpo_system_attr']:
                if len(gui_params['algorithm'])==1:
                    stepwise = True if (gui_params['hpo_system_attr']['stepwise']==1) else False
        imp = getImportingList(hpo)
        prev = removeAbnormal(indirect)
        body = pre_feature(gui_params, indirect) # indirect data 는 prefeature 가 필요없음..
        pre, flag_label = pre_target(gui_params)
        pt = post_target(gui_params,flag_label)
        #indirect_body = (getIndirectDataLoader(gui_params)) if indirect else ""
        indirect_body = (getIndirectDataLoader(gui_params)) if indirect else ""
        #main_body = (getData(gui_params, flag_label) if not hpo else "def load_data():\n") + getIndent((getData(gui_params, flag_label)+"return train_data, test_data, features, target" + (", label_names" if flag_label else "")),indent_level=4)
        if hpo:
            main_body = "def load_data():\n"
            main_body += getIndent((getData(gui_params, flag_label)+"return train_data, test_data, features, target" + (", label_names" if flag_label else "")),indent_level=4)
        else:
            main_body = getData(gui_params, flag_label)
        #algo_body = getAlgorithmBody(gui_params, indirect=indirect) if not hpo else getObjectiveFunction(gui_params, indirect=indirect, stepwise=stepwise, greedy=greedy)
        if not hpo:
            algo_body = getAlgorithmBody(gui_params, indirect=indirect)
        else:
            algo_body, ObjectiveFunction_names = getObjectiveFunction(gui_params, indirect=indirect, stepwise=stepwise, greedy=greedy)
        eval_body = getEvaluation(gui_params) if not hpo else getHPOMainFunction(gui_params, ObjectiveFunction_names)
    return imp+"\n\n"+prev+"\n\n"+body+"\n\n"+pre+"\n\n"+pt+"\n\n\n"+indirect_body+"\n"+main_body+"\n\n\n"+algo_body+"\n\n\n"+eval_body

##############################################################################################

def getIndirectDataLoader(gui_params):
    results = ""
    if 'ml_file_info' in gui_params:
        if 'data_type' in gui_params['ml_file_info']:
            if (gui_params['ml_file_info']['data_type'] == 'image_2d'):
                if 'DL_Pytorch' in gui_params['algorithm']:
                    results += "import torch\nfrom PIL import Image\n"
                    results += "class CustomImageDatasetLoaderforPytorch(torch.utils.data.Dataset):\n"
                    results += "    ''' X and y should be array type\n"
                    results += "        e.g., train = CustomImageDatasetLoaderforPytorch(X_train, y_train)\n"
                    results += "    '''\n"
                    results += "    def __init__(self, X, y, transform=None):\n"
                    results += "        self.X = X\n"
                    results += "        self.y = y\n"
                    results += "        self.transform = transform\n"
                    results += "        self.data_len = len(X)\n"
                    results += "    def __len__(self):\n"
                    results += "        return self.data_len\n"
                    results += "    def getHeight(self):\n"
                    results += "        n_sample = 10\n"
                    results += "        r10_idx = torch.zeros(n_sample).long().random_(0,self.data_len)\n"
                    results += "        arr = max([max(Image.open(self.X[i.item()][0]).size) for i in r10_idx])\n"
                    results += "        return arr\n"
                    results += "    def __getitem__(self, idx):\n"
                    results += "        if torch.is_tensor(idx):\n"
                    results += "            idx = idx.tolist()\n"
                    results += "        img = Image.open(self.X[idx][0]).convert('RGB')\n"
                    results += "        if self.transform:\n"
                    results += "            img = self.transform(img)\n"
                    results += "        label = self.y[idx]\n"
                    results += "        return img, label\n\n"
                results += "def fetch_images(ids, img_type='RGB', out_channels=3):\n"
                results += "    # channels x heights x widths (e.g., 3 x 34 x 34)\n"
                results += "    # Array to load images into\n"
                results += "    arr = []\n"
                results += "    mod = 3\n"
                results += "    i = 0\n"
                results += "    for img_id in ids:\n"
                results += "        img = plt.imread(img_id)\n"
                results += "        if i==0:\n"
                results += "            h = img[:,:,0].shape[0]\n"
                results += "            w = img[:,:,0].shape[1]\n"
                results += "        if img_type=='GRAYSCALE':\n"
                results += "            r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]\n"
                results += "            img = 0.2989 * r + 0.5870*g + 0.1140 * b\n"
                results += "            mod = 1\n"
                results += "        if out_channels==3:\n"
                results += "            arr.append(img.reshape(1*mod, h, w))\n"
                results += "        elif out_channels==1:\n"
                results += "            arr.append(img.ravel())\n"
                results += "        i+=1\n"
                results += "    # Turn into numpy array and normalize pixel values\n"
                results += "    arr = np.array(arr).astype('float32')\n"
                results += "    return arr\n\n"
                if 'image_color' in gui_params['ml_file_info']:
                    if gui_params['ml_file_info']['image_color'] == 'GRAYSCALE':
                        # do something for GRAYSCALE
                        results += "##* fetched = fetch_images(train_data['file_loc'], 'GRAYSCALE', 1)\n"
                    elif gui_params['ml_file_info']['image_color'] == 'RGB':
                        # do something for RGB
                        results += "##* fetched = fetch_images(train_data['file_loc'], 'RGB', 3)\n"
                else:
                    results += "## [json type ERROR] 'ml_file_info' -> 'image_color' is required.\n"
            elif (gui_params['ml_file_info']['data_type'] == 'image_3d'):
                pass
            elif (gui_params['ml_file_info']['data_type'] == 'xyz_structure'):
                pass
            elif (gui_params['ml_file_info']['data_type'] == 'smiles'):
                pass
        else:
            results += "## [json type ERROR] 'ml_file_info' -> 'data_type' is required.\n"
    else:
        results += "## [json type ERROR] 'ml_file_info' is required.\n"
    return results


def getHPOMainFunction(gui_params, ObjectiveFunction_names): # 입력 시간 등
    pre = '''
if __name__ == "__main__":
    import optuna
    import sdroptim
    #
    stepwise, task_and_algorithm = sdroptim.check_stepwise_available(json_file_name)
    args = sdroptim.get_argparse(automl=True, json_file_name=json_file_name)
    #
'''
    if len(ObjectiveFunction_names)==1:
        post = "    if stepwise:\n"
        post+= "        sdroptim.stepwise_mpi_time("+ObjectiveFunction_names[0]+", args, task_and_algorithm)\n"
        post+= "    else:\n"
        post+= "        sdroptim.optuna_mpi("+ObjectiveFunction_names[0]+", args)\n"
    elif len(ObjectiveFunction_names)==2:
        post = "    if stepwise:\n"
        post+= "        sdroptim.stepwise_mpi_time_dobj("+ObjectiveFunction_names[0]+", "+ObjectiveFunction_names[1]+", args, task_and_algorithm)\n"
        post+= "    else:\n"
        post+= "        sdroptim.optuna_mpi_dobj("+ObjectiveFunction_names[0]+", "+ObjectiveFunction_names[1]+", args)\n"
    return pre+post

def getImportingList(hpo=False):
    default = '''#-*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
'''
    #if not hpo:
    #    default = default + 'import matplotlib.pyplot as plt\n%matplotlib inline\nfrom matplotlib import style\n'
    return default + "\n"

def removeAbnormal(indirect=False):
    if indirect:
        default = '''##*\ndef _remove_abnormal(all_data):
    ## Pre-processing - 1. Input Missing values
    ## Note that the type of the dataset is 'Indirect'.
    ## Find a missing value to remove the corressponding row in the dataset.
    all_data = all_data.dropna()
    all_data.index = pd.Index(np.arange(len(all_data)))

    return all_data
    '''
    else:
        default = '''##*\ndef _remove_abnormal(all_data):
    ## Pre-processing - 1. Input Missing values
    ## There are many options in order to deal with a missing value such as:
    ## - A constant value that has meaning within the domain, such as 0, distinct from all other values.
    ## - A value from another randomly selected record.
    ## - A mean, median or mode value for the column.
    ## - A value estimated by another predictive model.
    ## We are going to replace a missing value to 0 for numeric columns and 'empty' for string columns, respectively.

    ## Filling for missing values
    categorical_columns_list = []
    for each in all_data.columns:
        if all_data[each].dtypes != 'int64':
            if all_data[each].dtypes != 'float64':
                categorical_columns_list.append(each)
    for item in categorical_columns_list:
        all_data[item].fillna("empty",inplace=True)
    all_data.fillna(0, inplace=True)
    
    return all_data
    '''
    return default + "\n"

def pre_feature(gui_params, indirect=False):

    # getting column names
    input_cols, output_col, input_index_list, output_index = getInputOutputColumnNames(gui_params)
    if type(input_cols) is list:
        input_cols_string = ', '.join('{!r}'.format(item) for item in input_cols)
    else:
        input_cols_string = input_cols    
    
    default = '''##*\ndef _pre_feature(all_data, processingFeature):
    ## Pre-processing - 2. Vectorization for Input\n'''
    data_analysis = "    column_list = ['"+ output_col +"', " + input_cols_string + "]\n    features = [" + input_cols_string + "]\n    target = '" + output_col + "'\n"
    for_indirect  = "    for each in features:\n        results = processingFeature[each].apply(lambda x: x.replace('\\\\','/'))\n"
    if indirect:
        bottom = "\n    return pd.DataFrame(results, columns=features)"
        return default+data_analysis+for_indirect+bottom
    header = "    ## For saving the converted key list\n    converted_key_list = []\n    "
    
    results = ""
    set_of_lb = "" 
    show_summary_heads = ""
    column_name = ""
    replaced_column_name = ""
    drop_original_column = ""
    add_binarized_columns = ""

    for column_index in gui_params['datatype_of_columns']:
        if gui_params['datatype_of_columns'][column_index] == 'Numeric':
            pass
        elif gui_params['datatype_of_columns'][column_index] == 'String' or 'Boolean' or 'Date':
            if gui_params['output_columns_index_and_name'] is not None:
                for key in gui_params['output_columns_index_and_name']:
                    if column_index != key:
                        # replace '.' -> '_' for python
                        header_local = "\n    ###########################################################"
                        column_name = gui_params['whole_columns_index_and_name'][column_index]
                        replaced_column_name = column_name.replace(".","_") + "_lb"
                        lb_fit_name = replaced_column_name + "_fit"
                        set_of_lb = "\n    ## For the categorical column '" + column_name + "'\n    " + replaced_column_name + " = LabelBinarizer()\n    " + lb_fit_name + " = " + replaced_column_name + '.fit(all_data["' + column_name + '"])\n    ' + replaced_column_name + ' = ' + lb_fit_name + '.transform(all_data["' + column_name + '"])\n    '
                        drop_original_column = "all_data = all_data.drop(columns=['" + column_name + "'])\n    processingFeature['" + column_name + "'].fillna('empty', inplace=True)\n\n    "
                        add_binarized_columns = "column_names = []\n    for i in range(0, " + replaced_column_name + '.shape[1]):\n        column_names.append("' + column_name + '"+"_types_"+str(i+1))\n    ' + 'temp = pd.DataFrame(' + replaced_column_name + ", columns = column_names)\n    " + "all_data = pd.concat([all_data, temp], axis=1)\n    ##\n    "
                        processingName = column_name.replace(".","_") + "_processing"
                        processingPart = processingName + " = " + lb_fit_name + ".transform(processingFeature['" + column_name + "'])\n    processingFeature = processingFeature.drop(columns=['" + column_name + "'])\n    temp = pd.DataFrame(" + processingName + ", columns=column_names)\n    processingFeature = pd.concat([processingFeature, temp], axis=1)\n    "
                        keyAppend = "converted_key_list.append(['"+column_name+"', column_names])\n"
                        results = results + header_local + set_of_lb + drop_original_column + add_binarized_columns + processingPart + keyAppend
    if results == "":
        results = "## Input Columns have Numeric data only. There is nothing to be vectorized.\n"
    else:
        header_local = "\n    ## using Label Binarizer (for Input Columns)\n    from sklearn.preprocessing import LabelBinarizer\n    "
        results = header_local + results

    default = default + data_analysis + header + results

    auto_correction_after_vectorization = "\n    ## Auto correction"+'''
    column_list_old = column_list.copy()
    features_old = features.copy()
    mod_column_list = []
    mod_features = []
    if converted_key_list is not None:
        for keys, values in converted_key_list:
            for item in column_list:
                if item == keys:
                    column_list.remove(keys)
                    for iter_item in values:
                        mod_column_list.append(iter_item) # replace
            for item in features:
                if item == keys:
                    features.remove(keys)
                    for iter_item in values:
                        mod_features.append(iter_item) # replace

    # replace previous list
    if len(mod_column_list) != 0:
        column_list = column_list + mod_column_list
    if len(mod_features) != 0:
        features = features + mod_features
    processingFeature.fillna(0, inplace=True)
    '''
    scaler = "\n    ## Listing features and scaling datasets\n    from sklearn.preprocessing import StandardScaler\n    " + "scaler = StandardScaler()\n    scaled = scaler.fit(all_data[features])\n    " + "results = scaled.transform(processingFeature[features])\n"
    default = default + auto_correction_after_vectorization + scaler
    default = default + "\n    return pd.DataFrame(results, columns=features)"
    return default

def pre_target(gui_params):

    # getting column names
    input_cols, output_col, input_index_list, output_index = getInputOutputColumnNames(gui_params)
    if type(input_cols) is list:
        input_cols_string = ', '.join('{!r}'.format(item) for item in input_cols)
    else:
        input_cols_string = input_cols  

    default = '''##*\ndef _pre_target(all_data, processingTarget):
    ## Pre-processing - 2. Vectorization for Output\n'''
    data_analysis = "    target = '" + output_col + "'\n"

    results = ""
    set_of_lb = ""
    column_name = ""
    replaced_column_name = ""
    flag_label = False

    for column_index in gui_params['datatype_of_columns']:
        if gui_params['datatype_of_columns'][column_index] == 'Numeric':
            pass
        elif gui_params['datatype_of_columns'][column_index] == 'String' or 'Boolean' or 'Date':
            if gui_params['output_columns_index_and_name'] is not None:
                for key in gui_params['output_columns_index_and_name']:
                    if column_index == key:
                        # Label Encoding
                        flag_label = True
                        header_local = "\n    ## using Label Encoder (for Output Column)\n    from sklearn.preprocessing import LabelEncoder\n"
                        #header_local = header_local + " = []\n"
                        column_name = gui_params['whole_columns_index_and_name'][column_index]
                        replaced_column_name = column_name.replace(".","_") + "_le"
                        set_of_lb = "\n    ## For the categorical column '" + column_name + "'\n    " + replaced_column_name + " = LabelEncoder()\n    " + replaced_column_name + " = " + replaced_column_name + '.fit(all_data["' + column_name + '"])\n    label_names = '+replaced_column_name+'.classes_\n    ' + 'processedTarget = processingTarget.copy()\n    processedTarget_tmp = ' + replaced_column_name+'.transform(processingTarget["'+column_name+'"])\n    processedTarget["'+column_name+'"] = processedTarget_tmp\n    '
                        results = results + header_local +set_of_lb
    if results == "":
        results = "    ## Output Column has Numeric data. There is nothing to be vectorized.\n    processedTarget = processingTarget\n"
    
    default = default + data_analysis + results

    if flag_label:
        default = default + "\n    return processedTarget, label_names"
    else:
        default = default + "\n    return processedTarget"
    return default, flag_label

def post_target(gui_params, flag_label):
    
    # getting column names
    input_cols, output_col, input_index_list, output_index = getInputOutputColumnNames(gui_params)
    if type(input_cols) is list:
        input_cols_string = ', '.join('{!r}'.format(item) for item in input_cols)
    else:
        input_cols_string = input_cols  

    data_analysis = "    target = '" + output_col + "'\n"

    default = "##*\ndef _post_target(all_data, predicted): \n    ## Post-processing for Output\n"
    default = default + data_analysis

    results = ""
    set_of_lb = ""
    column_name = ""
    replaced_column_name = ""

    if gui_params["task"] == "Classification":
        if flag_label:
            for column_index in gui_params['datatype_of_columns']:
                if gui_params['datatype_of_columns'][column_index] == 'Numeric':
                    pass
                elif gui_params['datatype_of_columns'][column_index] == 'String' or 'Boolean' or 'Date':
                    if gui_params['output_columns_index_and_name'] is not None:
                        for key in gui_params['output_columns_index_and_name']:
                            if column_index == key:
                                # Label Decoding
                                flag_label = True
                                header_local = "\n    ## Inverse Labeling using Label Encoder (for Output Column)\n    from sklearn.preprocessing import LabelEncoder\n"
                                column_name = gui_params['whole_columns_index_and_name'][column_index]
                                replaced_column_name = column_name.replace(".","_") + "_le"
                                set_of_lb = "\n    ## For the categorical column '" + column_name + "'\n    " + replaced_column_name + " = LabelEncoder()\n    " + replaced_column_name + " = " + replaced_column_name + '.fit(all_data["' + column_name + '"])\n    ' + 'predicted = ' + replaced_column_name+'.inverse_transform(predicted)\n    '
                                results = results + header_local +set_of_lb
    else:
        results = "    ## For the Regression task, We do nothing here\n"
    default = default + results + "\n    return predicted"
    return default

def getData(gui_params, flag_label):

    readData = getCSVData(gui_params)
    processedData = getProcessedData(gui_params, flag_label)

    return readData + processedData


def getCSVData(gui_params):
    data_loading = '''##* Data Loading
try:
    all_data = pd.read_csv("'''+ gui_params['ml_file_path'] + gui_params['ml_file_name'] + '''")
except:
    all_data = pd.read_csv("'''+ gui_params['ml_file_path'] + gui_params['ml_file_name'] + '''", encoding = "ISO-8859-1")

## Discover the head of your data
all_data.head()
'''
    return data_loading

def getProcessedData(gui_params, flag_label):
    ## Remove abnormal values in data (ex, Putting zero in null value)
    ## Preprocess the features and target

    # getting column names
    input_cols, output_col, input_index_list, output_index = getInputOutputColumnNames(gui_params)
    if type(input_cols) is list:
        input_cols_string = ', '.join('{!r}'.format(item) for item in input_cols)
    else:
        input_cols_string = input_cols  

    data_analysis = "target = '" + output_col + "'\n"
    filteredData = "##* Preprocess the feature and target values in data\nall_data = _remove_abnormal(all_data)\n"
    filteredData = filteredData + "xy = all_data.copy()\ny = xy.loc[:, [target]]\nx = xy.drop(target, axis=1)\nprocessedX = _pre_feature(all_data, x)\nfeatures = processedX.columns.tolist()\n"
    if flag_label:
        filteredData = filteredData + "processedY, label_names = _pre_target(all_data, y)\n"
    else:
        filteredData = filteredData + "processedY = _pre_target(all_data, y)\n"
    filteredData = filteredData + "scaled_all_data = pd.concat([processedX, processedY], axis=1)\n"

    train_test_split = "##* Train and Test Dataset Split\n"
    flag = ""
    if gui_params['testing_frame_extract_method'] == 'random':
        flag = "True"
    elif gui_params['testing_frame_extract_method'] == 'basic':
        flag = "False"
    train_test_split += "from sklearn.model_selection import train_test_split\n"
    if gui_params['testing_frame_rate'] == 0.0:
        train_test_split += "train_data = scaled_all_data\ntest_data = scaled_all_data\n"
    else:
        train_test_split += "train_data, test_data = train_test_split(scaled_all_data, test_size = " + str(gui_params['testing_frame_rate']) + ", shuffle = " + flag +(", stratify = scaled_all_data[target]" if (gui_params['task']=='Classification') and (flag=="True") else "")+", random_state=123)\n"
    return data_analysis + filteredData + train_test_split


def getAlgorithmBody(gui_params, algorithm=None, r_val=None, for_hpo_tune=False, indirect=False, stepwise=False):
    results = ""
    if for_hpo_tune:
        prune_available = True
    else:
        prune_available = False
    if algorithm == None:
        algorithm = gui_params['algorithm']
    if indirect:
        if algorithm == 'DL_Pytorch': # in the case of DL_Pytorch -> generates codes in getDLPytorch()
            results += "##* fetch Images ...\n"
        else:
            if gui_params['ml_file_info']['data_type'] == 'image_2d':
                if gui_params['ml_file_info']['image_color'] == 'GRAYSCALE':
                    results += "train_fetched = fetch_images(train_data['file_loc'], 'GRAYSCALE', 1)\n"
                    results += "test_fetched = fetch_images(test_data['file_loc'], 'GRAYSCALE', 1)\n"
                elif gui_params['ml_file_info']['image_color'] == 'RGB':
                    results += "train_fetched = fetch_images(train_data['file_loc'], 'RGB', 1)\n"
                    results += "test_fetched = fetch_images(test_data['file_loc'], 'GRAYSCALE', 1)\n"
                results+="features = ['pixel_'+str(x) for x in range(train_fetched.shape[1])]\n"
                results+='train_data = pd.concat([train_data, pd.DataFrame(train_fetched, columns=["pixel_"+str(x) for x in range(train_fetched.shape[1])], index=train_data.index)], axis=1)\n'
                results+='test_data = pd.concat([test_data, pd.DataFrame(test_fetched, columns=["pixel_"+str(x) for x in range(test_fetched.shape[1])], index=test_data.index)], axis=1)\n'
        if for_hpo_tune:
            results = getIndent(results, indent_level=8)
    if algorithm == 'MLR':
        results += getMultipleLinearRegression(gui_params, for_hpo_tune)
    elif algorithm == 'SVM':
        results += getSupportVectorMachine(gui_params, for_hpo_tune)
    elif algorithm == 'RF':
        results += getRandomForest(gui_params, for_hpo_tune)
    elif algorithm == 'BT':
        results += getBoostedTrees(gui_params, for_hpo_tune )
    elif algorithm == 'LR':
        results += getLocalRegression(gui_params, for_hpo_tune)
    elif algorithm == 'DL':
        results += getDeepLearning(gui_params, for_hpo_tune)
    elif algorithm == 'DL_Pytorch':
        results += getDLPytorch(gui_params, r_val, for_hpo_tune, indirect, prune_available, stepwise)
    elif algorithm == "XGBoost": # 20200701
        results += getXGBoost(gui_params, r_val, for_hpo_tune, prune_available)
    elif algorithm == "LightGBM":
        results += getLightGBM(gui_params, r_val, for_hpo_tune, prune_available)
    else:
        print("* not matched algorithm")
        return -1
    results += "\n"
    if not for_hpo_tune:
        # Call SaveAI Method here
        saveMessage = saveModel(gui_params)
        saveMessage = ""
        results += saveMessage
    return results

############################################################################################
### 20200609 add automl code begin
############################################################################################

def getAlgorithmListAccordingToResourceType(origin_list, types):
    # 20200723 version
    cpu_algorithms_list = ['MLR','SVM','RF','BT']
    gpu_algorithms_list = ['DL_Pytorch','XGBoost','LightGBM']
    comparing_list = cpu_algorithms_list if types=='cpu' else gpu_algorithms_list
    return [x for x in origin_list if x in comparing_list]
    

def getObjectiveFunction(gui_params, indirect=False, stepwise=False, greedy=True):
    #### supported algorithms per resources
    results_cpu = ""
    results_gpu = ""
    ObjectiveFunction_names = []
    #
    #cpu_algorithms_list = ['MLR','SVM','RF','BT']
    #gpu_algorithms_list = ['DL_Pytorch','XGBoost','LightGBM']
    temp_gui_params = gui_params.copy()
    #temp_gui_params['algorithm'] = [x for x in temp_gui_params['algorithm'] if x in cpu_algorithms_list]
    temp_gui_params['algorithm'] = getAlgorithmListAccordingToResourceType(temp_gui_params['algorithm'], 'cpu')
    if len(temp_gui_params['algorithm'])>0:
        results_cpu = getObjectiveFunction_(resources='cpu', gui_params=temp_gui_params, indirect=indirect, stepwise=stepwise, greedy=greedy)
        ObjectiveFunction_names.append('objective_cpu')
    #
    temp_gui_params = gui_params.copy()
    #temp_gui_params['algorithm'] = [x for x in temp_gui_params['algorithm'] if x in gpu_algorithms_list]
    temp_gui_params['algorithm'] = getAlgorithmListAccordingToResourceType(temp_gui_params['algorithm'], 'gpu')
    if len(temp_gui_params['algorithm'])>0:
        results_gpu = getObjectiveFunction_(resources='gpu', gui_params=temp_gui_params, indirect=indirect, stepwise=stepwise, greedy=greedy)
        ObjectiveFunction_names.append('objective_gpu')
    return results_cpu+'\n'+results_gpu, ObjectiveFunction_names

def getObjectiveFunction_(resources, gui_params, indirect=False, stepwise=False, greedy=True):
    results = '##* objective functions for multiple algorithms\ndef objective_'+resources+'(trial'+(", params" if stepwise else "")+'):\n    train_data, test_data, features, target'
    if gui_params['task']=='Classification':
        results = results + ", label_names"
    results += ' = load_data()\n'
    if greedy:
        results += '    algorithm_name = trial.suggest_categorical("algorithm_name_'+resources+'", ' + str(gui_params['algorithm']) +')\n'
    else:
        results += '    import secrets\t# using built-in func. over py_3.6.3\n'
        results += '    algorithm_names = '+str(gui_params['algorithm']) + '\n'
        results += '    algorithm_name = secrets.choice(algorithm_names)\n'

    if 'hpo_system_attr' in gui_params:
        jobpath, (uname, sname, job_title, wsname, job_directory) = get_jobpath_with_attr(gui_params)
        if 'ss_json_name' in gui_params['hpo_system_attr']:
            searching_space_json = jobpath+ os.sep+gui_params['hpo_system_attr']['ss_json_name']
        else: # using default
            searching_space_json = jobpath+os.sep+'searching_space_automl.json'
            if not os.path.exists(searching_space_json):
                generate_default_searching_space_file(searching_space_json)
    #
    with open(searching_space_json, 'r') as __:
        searching_space = json.load(__)
        r_val = searching_space.copy()
    ##################################################
    ## dependency modification 20.06.25
    ##################################################
    if indirect:
        if 'data_type' in gui_params['ml_file_info']:
            if (gui_params['ml_file_info']['data_type'] == 'image_2d'):
                if 'DL_Pytorch' in gui_params['algorithm']:
                    #r_val['Classification']['DL_Pytorch']['model']['choices'] = ['CNN'] # CNN Only
                    r_val['Classification']['DL_Pytorch']['epochs']['high'] = 20 # 5 -> 20 in order to avoid low perfo. due to non-convergence
    ##################################################
    # check process
    if 'hparams' not in gui_params:
        gui_params.update({'hparams':{}})
    for each_algorithm in gui_params['algorithm']:
        results = results + "\n    if algorithm_name == '"+each_algorithm+"':\n"
        gui_params['hparams'] = {}
        gui_params['algorithm'] = each_algorithm
        params_per_algorithm = []
        distribution_context = ""
        for each_hparam in searching_space[gui_params['task']][each_algorithm]:
            #################### 20200714 loop params (n_layers, n_units, dropout) should be controlled in the model generation part.
            #################### so, we remove them in this header part.
            loop_params = ['n_layers', 'n_units','dropout']
            if each_hparam in loop_params:
                continue
            #########################################
            value = each_algorithm+'_'+each_hparam
            gui_params['hparams'].update({each_hparam:value})
            params_per_algorithm.append(value)
            if ('low' in searching_space[gui_params['task']][each_algorithm][each_hparam]) and ('high' in searching_space[gui_params['task']][each_algorithm][each_hparam]):
                low_text = str(searching_space[gui_params['task']][each_algorithm][each_hparam]['low']) if not stepwise else "params['"+each_hparam+"']['low']"
                high_text = str(searching_space[gui_params['task']][each_algorithm][each_hparam]['high']) if not stepwise else "params['"+each_hparam+"']['high']"
                if (type(searching_space[gui_params['task']][each_algorithm][each_hparam]['low']) is float) or (type(searching_space[gui_params['task']][each_algorithm][each_hparam]['high']) is float):
                    distribution_context += value + ' = trial.suggest_float("'+value+'", ' + low_text + ', ' + high_text + ')\n'
                elif (type(searching_space[gui_params['task']][each_algorithm][each_hparam]['low']) is int) and (type(searching_space[gui_params['task']][each_algorithm][each_hparam]['high']) is int):
                    distribution_context += value + ' = trial.suggest_int("'  +value+'", ' + low_text + ', ' + high_text + ')\n'
            ### transformation function
            if 'transformation' in searching_space[gui_params['task']][each_algorithm][each_hparam]:
                trafo_str = searching_space[gui_params['task']][each_algorithm][each_hparam]['transformation']
                trafo_str = value + ' = ' + trafo_str.replace('x',value) + '\n'
                distribution_context += trafo_str
            elif 'choices' in searching_space[gui_params['task']][each_algorithm][each_hparam]:
                categorical_list_string = str(searching_space[gui_params['task']][each_algorithm][each_hparam]['choices']).replace("'True'", "True").replace("'False'","False").replace("'None'",'None')
                distribution_context += value + ' = trial.suggest_categorical("'+value+ '", ' + categorical_list_string+')\n'
        #####
        # set_user_attr into distribution_context --> all algorithms should be recorded in set_user_attr_algorithm_name
        #distribution_context += "# for non-greedy algorithm selection\ntrial.set_user_attr('algorithm_name', algorithm_name)\n" if not greedy else ""
        distribution_context += "# integrated algorithm list\ntrial.set_user_attr('algorithm_name', algorithm_name)\n"
        ######
        results += getIndent(distribution_context, indent_level=8) + getAlgorithmBody(gui_params, algorithm=each_algorithm, r_val=r_val, for_hpo_tune=True, indirect=indirect, stepwise=stepwise)
        #####
        ## auto saving high performance models.. (top-10 for all algo., top-1 for each algo.)
        top_n_all = gui_params['hpo_system_attr']['top_n_all'] if 'top_n_all' in gui_params['hpo_system_attr'] else 10
        top_n_each_algo = gui_params['hpo_system_attr']['top_n_each_algo'] if 'top_n_each_algo' in gui_params['hpo_system_attr'] else 3
        #
        if 'cv' in searching_space[gui_params['task']][each_algorithm]:    
            rval_score_str = "scores.mean()"
        else:
            rval_score_str = "confidence"
        #
        model_name = "model" if each_algorithm == 'DL_Pytorch' else "clf"
        rval_each_algorithm = "sdroptim.retrieve_model(algorithm_name, "+model_name+", trial.number, "+rval_score_str + \
                              ", top_n_all = "+ str(top_n_all)+", top_n_each_algo = " + str(top_n_each_algo) + ")\n"
        rval_each_algorithm+= "return "+rval_score_str
        #
        results = results + getIndent(rval_each_algorithm, indent_level=8)
        distribution_context = "" # refresh for another algorithm
    return results

def getMultipleLinearRegression(gui_params, for_hpo_tune=False):
    # getting column names
    input_cols, output_col, input_index_list, output_index = getInputOutputColumnNames(gui_params)

    # algorithm part
    results = "##* Multiple Linear Regression\nfrom sklearn.linear_model import LinearRegression\nclf = LinearRegression(" + getHyperParametersandClosing(gui_params, for_hpo_tune)
    results = results +"\nclf.fit(train_data[features], train_data[target])\n"

    results = results + "\n"
    if not for_hpo_tune:
        results = results + getDefaultParametersDetails(gui_params) # describing default values of mandantory hyperparameters

    results = results + "## results\n"
    if not for_hpo_tune:
        results = results + "print(clf)\t# We used the MinMax Scaler, instead of normalization options in the LinearRegression().\n"

    results = results +'''
##* Predict using the model we made.
predicted = clf.predict(test_data[features])
confidence = clf.score(test_data[features], test_data[target])\t# Returns the coefficient of determination R^2 of the prediction.
'''
    if not for_hpo_tune:
        results = results + 'print("Prediction accuracy: ", confidence)\n'
        results = results + getBasicAnalyticGraph(output_col, gui_params['task'])
        return results
    else:
        return getIndent(results, indent_level=8)

def getSupportVectorMachine(gui_params, for_hpo_tune=False):
    # getting column names
    input_cols, output_col, input_index_list, output_index = getInputOutputColumnNames(gui_params)

    # title
    if gui_params['task'] == 'Regression':
        results = "##* Support Vector Regression\nfrom sklearn import svm\nclf = svm.SVR("
    elif gui_params['task'] == 'Classification':
        results = "##* Support Vector Classification\nfrom sklearn import svm\nclf = svm.SVC("
    results = results + getHyperParametersandClosing(gui_params, for_hpo_tune)
    results = results +"\nclf.fit(train_data[features], train_data[target])\n"
    if not for_hpo_tune:
        results = results + "\n"+ getDefaultParametersDetails(gui_params) # describing default values of mandantory hyperparameters

    results = results + "## results\n"
    if not for_hpo_tune:
        results = results + "print(clf)\n"

    results = results +'''
##* Predict using the model we made.
predicted = clf.predict(test_data[features])
'''
    results += "confidence = clf.score(test_data[features], test_data[target])\t# Returns the coefficient of determination R^2 of the prediction." if gui_params['task'] == 'Regression' else \
        "confidence = metrics.f1_score(predicted, test_data[target], average='macro')\t# Returns mean f1-score."

    if not for_hpo_tune:
        results = results + 'print("Prediction accuracy: ", confidence)\n'
        results = results + getBasicAnalyticGraph(output_col, gui_params['task'])
        return results
    else:
        return getIndent(results, indent_level=8)

def getRandomForest(gui_params, for_hpo_tune=False):
    # getting column names
    input_cols, output_col, input_index_list, output_index = getInputOutputColumnNames(gui_params)

    # title
    if gui_params['task'] == 'Regression':
        results = "##* Random Forests Regression\nfrom sklearn.ensemble import RandomForestRegressor\nclf = RandomForestRegressor("
    elif gui_params['task'] == 'Classification':
        results = "##* Random Forests Classification\nfrom sklearn.ensemble import RandomForestClassifier\nclf = RandomForestClassifier("
    results = results + getHyperParametersandClosing(gui_params, for_hpo_tune)
    results = results +"\nclf.fit(train_data[features], train_data[target])\n"
    if not for_hpo_tune:
        results = results + "\n"+ getDefaultParametersDetails(gui_params) # describing default values of mandantory hyperparameters

    results = results + "## results\n"
    if not for_hpo_tune:
        results = results + "print(clf)\n"

    results = results +'''
##* Predict using the model we made.
predicted = clf.predict(test_data[features])
'''
    results += "confidence = clf.score(test_data[features], test_data[target])\t# Returns the coefficient of determination R^2 of the prediction." if gui_params['task'] == 'Regression' else \
        "confidence = metrics.f1_score(predicted, test_data[target], average='macro')\t# Returns mean f1-score."

    if not for_hpo_tune:
        results = results + 'print("Prediction accuracy: ", confidence)\n'
        results = results + getBasicAnalyticGraph(output_col, gui_params['task'])
        return results
    else:
        return getIndent(results, indent_level=8)

########################################################################################################

def getBoostedTrees(gui_params, for_hpo_tune=False):
    # getting column names
    input_cols, output_col, input_index_list, output_index = getInputOutputColumnNames(gui_params)

    # title
    if gui_params['task'] == 'Regression':
        results = "##* AdaBoost Regression\nfrom sklearn.ensemble import AdaBoostRegressor\nclf = AdaBoostRegressor("
    elif gui_params['task'] == 'Classification':
        results = "##* AdaBoost Classification\nfrom sklearn.ensemble import AdaBoostClassifier\nclf = AdaBoostClassifier("
    results = results + getHyperParametersandClosing(gui_params, for_hpo_tune)
    results = results +"\nclf.fit(train_data[features], train_data[target])\n"
    if not for_hpo_tune:
        results = results + "\n"+ getDefaultParametersDetails(gui_params) # describing default values of mandantory hyperparameters

    results = results + "## results\n"
    if not for_hpo_tune:
        results = results + "print(clf)\n"

    results = results +'''
##* Predict using the model we made.
predicted = clf.predict(test_data[features])
'''
    results += "confidence = clf.score(test_data[features], test_data[target])\t# Returns the coefficient of determination R^2 of the prediction." if gui_params['task'] == 'Regression' else \
        "confidence = metrics.f1_score(predicted, test_data[target], average='macro')\t# Returns mean f1-score."

    if not for_hpo_tune:
        results = results + 'print("Prediction accuracy: ", confidence)\n'
        results = results + getBasicAnalyticGraph(output_col, gui_params['task'])
        return results
    else:
        return getIndent(results, indent_level=8)

def getLightGBM(gui_params, r_val, for_hpo_tune=False, prune_available=False):
    # getting column names
    input_cols, output_col, input_index_list, output_index = getInputOutputColumnNames(gui_params)
    results = "##* LightGBM\nimport lightgbm as lgb\nimport sklearn, os\n"
    results += "DEVICE = "+ ("int(os.environ['CUDA_VISIBLE_DEVICES'])\n" if for_hpo_tune else "0\n")
    results += "\nlgb_params = {\n"
    lgb_params_inner = ("'objective': " + ("'regression'" if gui_params['task']=="Regression" else "'multiclass'") + ",\n" ) if not for_hpo_tune else ""#"LightGBM_objective"
    #lgb_params_inner += ",\n"   
    lgb_params_inner += "'device_type': 'gpu',\n"
    lgb_params_inner += "'gpu_device_id': DEVICE,\n"
    lgb_params_inner += "'verbose':-1, \n"
    lgb_params_inner += "'num_class':len(label_names),\n" if gui_params['task'] == 'Classification' else ""
    algo_prefix = "LightGBM_"
    if gui_params['hparams']:   # if it exists
        for hparam in gui_params['hparams']: # mantantories first
            if hparam.startswith("__"): # exceptional cases
                continue
            if not for_hpo_tune:
                temp = gui_params['hparams'][hparam]
                if (type(temp) is str) and temp != "None":
                    temp = "'"+temp+"'"
                else:
                    temp = str(temp)
            else: # in case of tune
                temp = algo_prefix+hparam
            lgb_params_inner += "'" + hparam + "': " + temp + ",\n"
    results += getIndent(lgb_params_inner[:-2], indent_level=4) + "}\n"
    results += "##* lightgbm parameter set configuration\n"
    results += "def lgb_check_param_configuration(lgb_params):\n"
    results += "    rval = lgb_params.copy()\n"
    results += "    removal_params = ['cv', 'num_boost_round']\n"
    results += "    for each in lgb_params:\n"
    results += "        if each in removal_params:\n"
    results += "            del rval[each]\n"
    results += "    return rval\n"
    results += "lgb_params = lgb_check_param_configuration(lgb_params)\n"
    if prune_available:
        results +='class modLightGBMPruningCallback(object):\n'
        results +='    def __init__(self, trial, metric, valid_name="valid_0"):\n'
        results +='        self._trial = trial\n'
        results +='        self._valid_name = valid_name\n'
        results +='        self._metric = metric\n'
        results +='    def __call__(self, env):\n'
        results +='        target_valid_name = self._valid_name\n'
        results +='        for evaluation_result in env.evaluation_result_list:\n'
        results +='            valid_name, metric, current_score, is_higher_better = evaluation_result[:4]\n'
        results +='            if valid_name != target_valid_name or metric != self._metric:\n'
        results +='                continue\n'
        results +='            if env.iteration % 100 == 0:\n'
        results +='                self._trial.report(-1*current_score, step=env.iteration)\n'
        results +='                if self._trial.should_prune():\n'
        results +='                    message = "Trial was pruned at iteration {}.".format(env.iteration)\n'
        results +='                    raise optuna.exceptions.TrialPruned(message)\n'
        results +='            return None\n'
    #### get cv_num
    cv_num = -1
    if ('cv' in gui_params['hparams']) and (not for_hpo_tune):
        cv_num = int(gui_params['hparams']['cv'])
        results += "LightGBM_cv = " + str(cv_num) + "\n"
    if for_hpo_tune:
        if 'cv' in r_val[gui_params['task']]['LightGBM']:
            if r_val[gui_params['task']]['LightGBM']['cv']['low'] > 1:
                cv_num = int(r_val[gui_params['task']]['LightGBM']['cv']['low'])
    #####
    ##### get num_boost_round
    num_boost_round = 100
    if ('num_boost_round' in gui_params['hparams']) and (not for_hpo_tune):
        num_boost_round = int(gui_params['hparams']['num_boost_round'])
        results += "LightGBM_num_boost_round = " + str(num_boost_round) + "\n"

    if prune_available:
        metric = "rmse" if gui_params['task'] == 'Regression' else "error"
        results += "pruning_callback = modLightGBMPruningCallback(trial, LightGBM_metric)\n"
    ##### train
    results += getLightGBM_TrainFunc(gui_params, cv_num, for_hpo_tune, prune_available)

    if not for_hpo_tune:
        results = results + "\n"+ getDefaultParametersDetails(gui_params) # describing default values of mandantory hyperparameters

    if not for_hpo_tune:
        if cv_num > 0:
            results += 'print("Prediction accuracy: ", scores.mean())\n'
        else:
            results += 'print("Prediction accuracy: ", confidence)\n'
        return results
    else:
        return getIndent(results, indent_level=8)    

def getLightGBM_TrainFunc(gui_params, cv_num, for_hpo_tune=False, prune_available=False):
    results = "from sklearn.model_selection import KFold\nkfold = KFold(n_splits = "+("LightGBM_cv" if for_hpo_tune else str(cv_num))+")\n" if cv_num>0 else ""
    results += "X_train, y_train = train_data[features].values, train_data[target].values\n"
    results += "X_test, y_test = test_data[features].values, test_data[target].values\n"
    if cv_num<1:
        results += "dtrain = lgb.Dataset(X_train, label = y_train)\n"
        results += "dtest = lgb.Dataset(X_test, label = y_test)\n"
    else:
        results += "scores = []\n"
        results += "for fold, (train_index, valid_index) in enumerate(kfold.split(X_train, y_train)):\n"
        results += "    dtrain = lgb.Dataset(X_train[train_index], label = y_train[train_index])\n"    
        results += "    dvalid = lgb.Dataset(X_train[valid_index], label = y_train[valid_index])\n"
    # val ---
    train = "clf = lgb.train(params = lgb_params, train_set = dtrain, num_boost_round = LightGBM_num_boost_round"
    train += ", early_stopping_rounds = max(int(LightGBM_num_boost_round/10),5)"
    train += (", valid_sets = ["+ ("dtest" if cv_num<1 else "dvalid") +"], verbose_eval=False" ) if for_hpo_tune else ""
    train += (", callbacks=[pruning_callback]" if prune_available else "") + ")\n"
    predict = "predicted = clf.predict(" + ("X_test" if cv_num<1 else "X_train[valid_index]") + ")\n"
    predict += "predicted = np.argmax(predicted, axis=1)\n" if gui_params['task'] == 'Classification' else ""
    if gui_params['task'] == 'Regression':
        confidence = "confidence = sklearn.metrics.r2_score(predicted, " + ("y_test" if cv_num<1 else "y_train[valid_index]") + ")\n"
    elif gui_params['task'] == 'Classification':
        confidence = "confidence = sklearn.metrics.f1_score(predicted, " + ("y_test" if cv_num<1 else "y_train[valid_index]") + ", average='macro')\n"
    confidence += "scores.append(confidence)" if cv_num>1 else ""
    val = train + predict + confidence
    if cv_num>1:
        val = getIndent(val, indent_level = 4)
    val += "scores = np.array(scores)\n" if cv_num>1 else ""
    return results+val



def getXGBoost(gui_params, r_val, for_hpo_tune=False, prune_available=False):
    # getting column names
    input_cols, output_col, input_index_list, output_index = getInputOutputColumnNames(gui_params)

    # title
    results = "##* XGBoost\nimport xgboost as xgb\nimport sklearn, os\n"
    results += "DEVICE = " + ("int(os.environ['CUDA_VISIBLE_DEVICES'])" if for_hpo_tune else "0") + "\n"
    results += "\nxgb_params = {\n"
    xgb_params_inner = "'gpu_id': DEVICE,\n"
    xgb_params_inner += "'tree_method': 'gpu_hist',\n"
    xgb_params_inner += "'disable_default_eval_metric': 1,\n"
    xgb_params_inner += "'verbosity':0, \n"
    xgb_params_inner += "'num_class':len(label_names),\n" if gui_params['task'] == 'Classification' else ""
    algo_prefix = "XGBoost_"
    if gui_params['hparams']:   # if it exists
        for hparam in gui_params['hparams']: # mantantories first
            if hparam.startswith("__"): # exceptional cases
                continue
            if not for_hpo_tune:
                temp = gui_params['hparams'][hparam]
                if type(temp) is str:
                    temp = "'"+temp+"'"
                else:
                    temp = str(temp)
            else: # in case of tune
                temp = algo_prefix+hparam
            ###################
            xgb_params_inner += "'" + hparam + "': " + temp + ",\n"
    results += getIndent(xgb_params_inner[:-2], indent_level=4) + "}\n"
    results += "##* xgboost parameter set configuration\n"
    results += "def booster_check_param_configuration(xgb_params):\n"
    results += "    rval = xgb_params.copy()\n"
    results += "    removal_params = ['cv', 'num_boost_round']\n"
    results += "    not_dart_to_remove = ['sample_type', 'normalize_type', 'rate_drop', 'skip_drop']\n"
    results += "    not_gbtree_to_remove = not_dart_to_remove + ['max_depth','eta','gamma','grow_policy']\n"
    results += "    if xgb_params['booster'] == 'dart':\n"
    results += "        pass\n"
    results += "    elif xgb_params['booster'] == 'gbtree':\n"
    results += "        removal_params += not_dart_to_remove\n"
    results += "    else:\n"
    results += "        removal_params += not_gbtree_to_remove\n"
    results += "    for each in xgb_params:\n"
    results += "        if each in removal_params:\n"
    results += "            del rval[each]\n"
    results += "    return rval\n"
    results += "\nxgb_params = booster_check_param_configuration(xgb_params)\n"
    if prune_available:
        results += 'class modXGBoostPruningCallback(object):\n'
        results += '    def __init__(self, trial, observation_key):\n'
        results += '        self._trial = trial\n'
        results += '        self._observation_key = observation_key\n'
        results += '    def __call__(self, env):\n'
        results += '        evaluation_result_list = env.evaluation_result_list\n'
        results += '        current_score = dict(evaluation_result_list)[self._observation_key]\n'
        results += '        if env.iteration % 100 == 0:\n'
        results += '            self._trial.report(-1*current_score, step=env.iteration)\n'
        results += '            if self._trial.should_prune():\n'
        results += '                message = "Trial was pruned at iteration {}.".format(env.iteration)\n'
        results += '                raise optuna.exceptions.TrialPruned(message)\n'
    #### get cv_num
    cv_num = -1
    if ('cv' in gui_params['hparams']) and (not for_hpo_tune):
        cv_num = int(gui_params['hparams']['cv'])
        results += "XGBoost_cv = " + str(cv_num) + "\n"
    if for_hpo_tune:
        if 'cv' in r_val[gui_params['task']]['XGBoost']:
            if r_val[gui_params['task']]['XGBoost']['cv']['low'] > 1:
                cv_num = int(r_val[gui_params['task']]['XGBoost']['cv']['low'])
    #####
    ##### get num_boost_round
    num_boost_round = 500
    if ('num_boost_round' in gui_params['hparams']) and (not for_hpo_tune):
        num_boost_round = int(gui_params['hparams']['num_boost_round'])
        results += "XGBoost_num_boost_round = " + str(num_boost_round) + "\n"

    if prune_available:
        metric = "rmse" if gui_params['task'] == 'Regression' else "error"
        results += "pruning_callback = modXGBoostPruningCallback(trial, 'valid-'+XGBoost_eval_metric)\n"
    ##### train
    results += getXGBoost_TrainFunc(gui_params, cv_num, for_hpo_tune, prune_available)

    if not for_hpo_tune:
        results = results + "\n"+ getDefaultParametersDetails(gui_params) # describing default values of mandantory hyperparameters

    if not for_hpo_tune:
        if cv_num > 0:
            results += 'print("Prediction accuracy: ", scores.mean())\n'
        else:
            results += 'print("Prediction accuracy: ", confidence)\n'
        return results
    else:
        return getIndent(results, indent_level=8)    

def getXGBoost_TrainFunc(gui_params, cv_num, for_hpo_tune=False, prune_available=False):
    results = "from sklearn.model_selection import KFold\nkfold = KFold(n_splits = "+("XGBoost_cv" if for_hpo_tune else str(cv_num))+")\n" if cv_num>0 else ""
    results += "X_train, y_train = train_data[features].values, train_data[target].values\n"
    results += "X_test, y_test = test_data[features].values, test_data[target].values\n"
    if cv_num<1:
        results += "dtrain = xgb.DMatrix(X_train, label = y_train)\n"
        results += "dtest = xgb.DMatrix(X_test, label = y_test)\n"
    else:
        results += "scores = []\n"
        results += "for fold, (train_index, valid_index) in enumerate(kfold.split(X_train, y_train)):\n"
        results += "    dtrain = xgb.DMatrix(X_train[train_index], label = y_train[train_index])\n"    
        results += "    dvalid = xgb.DMatrix(X_train[valid_index], label = y_train[valid_index])\n"
    # val ---
    train = "clf = xgb.train(xgb_params, dtrain, num_boost_round = XGBoost_num_boost_round" \
             + ", early_stopping_rounds = max(int(XGBoost_num_boost_round/10),5), verbose_eval = XGBoost_num_boost_round, " \
             + "evals=["+ ("(dtest, 'valid')" if cv_num<1 else "(dvalid, 'valid')")  + "]" + (", callbacks=[pruning_callback]" if prune_available else "") + ")\n"
    predict = "predicted = clf.predict(" + ("dtest" if cv_num<1 else "dvalid") + ")\n"
    if gui_params['task'] == 'Regression':
        confidence = "confidence = sklearn.metrics.r2_score(predicted, " + ("y_test" if cv_num<1 else "y_train[valid_index]") + ")\n"
    elif gui_params['task'] == 'Classification':
        confidence = "confidence = sklearn.metrics.f1_score(predicted, " + ("y_test" if cv_num<1 else "y_train[valid_index]") + ", average='macro')\n"
    confidence += "scores.append(confidence)" if cv_num>1 else ""
    val = train + predict + confidence
    if cv_num>1:
        val = getIndent(val, indent_level = 4)
    val += "scores = np.array(scores)\n" if cv_num>1 else ""
    return results+val

########################################################################################################

def getDLPytorch(gui_params, r_val=None, for_hpo_tune=False, indirect=False, prune_available=False, stepwise=False):
    # getting column names
    # r_val is searching_space_automl.json
    input_cols, output_col, input_index_list, output_index = getInputOutputColumnNames(gui_params)
    model_name = ""
    results = getDLPytorchImport()
    #    if for_hpo_tune:
    #        results += getDLPytorch_FNNmodel(gui_params, r_val, for_hpo_tune) 
    #        model_name = 'FNN'
    #        if gui_params['task'] == 'Classification':
    #            results += getDLPytorch_CNNmodel(gui_params, r_val, for_hpo_tune) 
    #            model_name = 'CNN'
    #    else:
    #        model_name = gui_params['hparams']['model']
    if for_hpo_tune:
        results += getDLPytorch_FNNmodel(gui_params, r_val, for_hpo_tune, stepwise)
        model_name = 'FNN'
        if gui_params['task'] == 'Classification':
            results += getDLPytorch_CNNmodel(gui_params, r_val, for_hpo_tune, stepwise) 
            model_name = 'CNN'
    else:
        model_name = gui_params['hparams']['model']
        if model_name == 'FNN':
            results += getDLPytorch_FNNmodel(gui_params, r_val, for_hpo_tune)
            results += ""
        elif model_name == 'CNN':
            results += getDLPytorch_CNNmodel(gui_params, r_val, for_hpo_tune) 

    #results += getDLPytorch_FNNmodel(gui_params, for_hpo_tune) if gui_params['hparams']['model'] == 'FNN' else getDLPytorch_CNNmodel(gui_params, for_hpo_tune)
    #results += getDLPytorch_Data2Tensor()
    cv_num = -1
    results += "##* Hyperparameters for deep learning\n"
    if indirect:
        results += "input_dim = CustomImageDatasetLoaderforPytorch(train_data[features].values, train_data[target].values).getHeight()\n"
        if 'model' in gui_params['hparams']:
            if gui_params['hparams']['model'] == 'FNN':
                results += "DL_Pytorch_model = 'FNN'\n"
            elif gui_params['hparams']['model'] == 'CNN':
                results += "DL_Pytorch_model = 'CNN'\n"
    else:
        results += "input_dim = train_data[features].shape[1]\t# number of features\n"
    results += "DEVICE = torch.device('cuda')\n"
    if model_name == 'FNN':
        results += "model = define_"+model_name+"model(" + ("trial, " if for_hpo_tune else "") + ("input_dim*input_dim" if indirect else "input_dim")+(", len(label_names)" if gui_params['task']=='Classification' else "")+").to(DEVICE)\n"
    elif model_name == 'CNN':
        results += "if DL_Pytorch_model == 'FNN':\n"
        results += "    model = define_FNNmodel(" + ("trial, " if for_hpo_tune else "") + ("input_dim*input_dim" if indirect else "input_dim")+(", len(label_names)" if gui_params['task']=='Classification' else "")+").to(DEVICE)\n"
        results += "elif DL_Pytorch_model == 'CNN':\n"
        results += "    model = define_CNNmodel(" + ("trial, " if for_hpo_tune else "") + "input_dim"+(", len(label_names)" if gui_params['task']=='Classification' else "")+").to(DEVICE)\n"
    if gui_params['hparams']:   # if it exists
        for hparam in gui_params['hparams']: # mantantories first
            if hparam == 'optimizer':
                temp = str(gui_params['hparams'][hparam])
                if not for_hpo_tune:
                    temp = "'"+ temp+ "'"
                results += "optimizer_name = " + temp + "\n"
            elif hparam == 'lr':
                results += "lr = " + str(gui_params['hparams'][hparam]) + "\n"
            elif hparam == 'momentum':
                results += "momentum = " + str(gui_params['hparams'][hparam]) + "\n"
            elif hparam == 'epochs':
                results = results + "epochs = " + str(gui_params['hparams'][hparam]) + "\n"
            elif hparam == 'batch_size':
                results = results + "batch_size = " + str(gui_params['hparams'][hparam]) + "\n"
            elif hparam == 'loss':
                temp = str(gui_params['hparams'][hparam])
                if not for_hpo_tune:
                    temp = "'"+ temp+ "'"
                results = results + "loss_name = " + temp + "\n"
            elif hparam == 'cv':
                if type(gui_params['hparams'][hparam]) is int:
                    cv_num = int(gui_params['hparams'][hparam])
                else:
                    cv_num = 0
    #results += "optimizer = getattr(optim, optimizer_name)(model.parameters()"+(",lr = lr" if 'lr' in gui_params['hparams'] else "")+(", momentum = momentum" if 'momentum' in gui_params['hparams'] else "")+")\n"
    if for_hpo_tune:
        results += "if optimizer_name == 'Adam':\n"
        results += "    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)\n"
        results += "else:\n"
        results += "    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, momentum=momentum)\n"
        if 'cv' in r_val[gui_params['task']]['DL_Pytorch']:
            if r_val[gui_params['task']]['DL_Pytorch']['cv']['low'] > 1:
                cv_num = r_val[gui_params['task']]['DL_Pytorch']['cv']['low']
    else:
        results += "optimizer = getattr(optim, optimizer_name)(model.parameters()"+(",lr = lr" if 'lr' in gui_params['hparams'] else "")+(", momentum = momentum" if 'momentum' in gui_params['hparams'] else "")+")\n"

    results += "loss_func = getattr("+("nn" if gui_params['task']=='Regression' else "F")+", loss_name)"+("(reduction='sum')" if gui_params['task']=='Regression' else "") + "\n"
    results += getDLPytorch_TrainFunc(gui_params, cv_num, for_hpo_tune, indirect, prune_available)
    if not for_hpo_tune:
        if cv_num > 0:
            results += 'print("Prediction accuracy: ", scores.mean())\n'
        else:
            results += 'print("Prediction accuracy: ", confidence)\n'
        #results = results + getBasicAnalyticGraph(output_col, gui_params['task'])
        return results
    else:
        return getIndent(results, indent_level=8)

def getDLPytorchImport():
    results = '''import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
'''
    return results

def getDLPytorch_FNNmodel(gui_params, r_val, for_hpo_tune=False, stepwise=False):
    results = "\ndef define_FNNmodel("+ ("trial, " if for_hpo_tune else "" )+ "input_dim"+(", output_dim" if gui_params['task']=='Classification' else "")+"):\n"
    input_layer = "layers = []\n"
    hidden_layer = ""
    output_layer=""
    add_softmax_layer=""
    bottom_layer=""
    if not for_hpo_tune:
        layers_string_list = gui_params['hparams']['hidden'].split(',')
        input_layer += "layers.append(nn.Linear(input_dim, "+layers_string_list[0]+"))\n"
        input_layer += "layers.append(nn.ReLU())\n"
        hidden_layer = ""
        for i in range(len(layers_string_list)-1):
            hidden_layer +="layers.append(nn.Linear("+layers_string_list[i]+", "+layers_string_list[i+1]+"))\n"
            hidden_layer += "layers.append(nn.ReLU())\n"
        output_layer = "layers.append(nn.Linear("+layers_string_list[i+1]+", "+ ("output_dim" if gui_params['task'] == 'Classification' else "1")+"))\n"
        add_softmax_layer = "layers.append(nn.Softmax(dim=1))\n" if gui_params['task']=='Classification' else "" 
        bottom_layer = "return nn.Sequential(*layers)\n"
        results += getIndent(input_layer+hidden_layer+output_layer+add_softmax_layer+bottom_layer, indent_level=4)
        return results
    else:
        input_layer += 'n_layers = trial.suggest_int("DL_Pytorch_FNN_n_layers", '+ ("params['n_layers']['low']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['n_layers']['low']))+', '+("params['n_layers']['high']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['n_layers']['high']))+')\n'
        input_layer += 'in_features = input_dim\n'
        hidden_layer = "for i in range(n_layers):\n"
        hidden_layer += '    out_features = trial.suggest_int("DL_Pytorch_FNN_n_units_l{}".format(i),' + ("params['n_units']['low']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['n_units']['low'])) +', ' + ("params['n_units']['high']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['n_units']['high'])) +')\n'
        hidden_layer += '    layers.append(nn.Linear(in_features, out_features))\n'
        hidden_layer += '    layers.append(nn.ReLU())\n'
        hidden_layer += '    p = trial.suggest_uniform("DL_Pytorch_FNN_dropout_l{}".format(i), '+ ("params['dropout']['low']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['dropout']['low'])) + ',' + ("params['dropout']['high']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['dropout']['high'])) + ')\n'
        hidden_layer += '    layers.append(nn.Dropout(p))\n'
        hidden_layer += '    in_features = out_features\n'
        output_layer = "layers.append(nn.Linear(in_features, "+ ("output_dim" if gui_params['task'] == 'Classification' else "1")+"))\n"
        add_softmax_layer = "layers.append(nn.Softmax(dim=1))\n" if gui_params['task']=='Classification' else "" 
        bottom_layer = "return nn.Sequential(*layers)\n"
        results += getIndent(input_layer+hidden_layer+output_layer+add_softmax_layer+bottom_layer, indent_level=4)
        return results

def getTransformsFunc(gui_params):
    results = "preprocessing = {\n"
    results += "    'train': "+getTransformsCompose(gui_params, train=True) + "    ,\n"
    results += "    'test_or_valid': "+getTransformsCompose(gui_params, train=False)
    results += "}\n"
    return results

def getTransformsCompose(gui_params, train):
    results = "transforms.Compose([\n"
    results += "    transforms.Resize((input_dim, input_dim)),\n"
    if train:
        results += "    transforms.RandomHorizontalFlip(),\n"
        results += "    transforms.RandomRotation(20),\n"
    if gui_params['ml_file_info']['data_type'] == 'image_2d':
        if gui_params['ml_file_info']['image_color'] == 'GRAYSCALE':
            results += "    transforms.Grayscale(num_output_channels=1),\n"
            results += "    transforms.ToTensor(),\n"
            results += "    transforms.Normalize([0.5], [0.5])"
        elif gui_params['ml_file_info']['image_color'] == 'RGB':
            results += "    transforms.ToTensor(),\n"
            results += "    transforms.Normalize((0.485,0.456,0.406), (0.229, 0.224, 0.225))"
    results += "])\n"
    return results

def getDLPytorch_TrainFunc(gui_params, cv_num, for_hpo_tune=False, indirect=False, prune_available=False):
    results = "from sklearn.model_selection import KFold\nkfold = KFold(n_splits = "+("DL_Pytorch_cv" if for_hpo_tune else str(cv_num))+")\n" if cv_num>0 else ""
    #results += "X_train, y_train = torch.Tensor(train_data[features].values), torch.Tensor(train_data[target].values)\nX_test, y_test = torch.Tensor(test_data[features].values), torch.Tensor(test_data[target].values)\n\n"
    labeltypeflag = "int" if gui_params['task'] == 'Classification' else "float"
    if indirect:    
        results += "X_train, y_train = train_data[features].values, train_data[target].values\n"
        results += "X_test, y_test = test_data[features].values, test_data[target].values\n"
    else:
        results += "X_train, y_train = torch.from_numpy(train_data[features].values).float(), torch.from_numpy(train_data[target].values)."+labeltypeflag+"()\n"
        results += "X_test, y_test = torch.from_numpy(test_data[features].values).float(), torch.from_numpy(test_data[target].values)."+labeltypeflag+"()\n"
    state_store = "# state store for cross-validation\nimport copy\ninit_state = copy.deepcopy(model.state_dict())\ninit_state_opt = copy.deepcopy(optimizer.state_dict())\n"
    results += state_store if cv_num > 0 else ""
    #
    preprocessing_for_indirect_images = getTransformsFunc(gui_params) if indirect else ""
    #
    fold_data = "scores = []\n"
    fold_data += "for fold, (train_index, valid_index) in enumerate(kfold.split(X_train, y_train)):\n"
    fold_data += "    model.load_state_dict(init_state)\n"
    fold_data += "    optimizer.load_state_dict(init_state_opt)\n"
    ##
    if indirect:
        body = "train = CustomImageDatasetLoaderforPytorch(X_train"+ ("[train_index]" if cv_num >0 else "") + ", y_train"+ ("[train_index]" if cv_num>0 else "") +", transform = preprocessing['train'])\n"
        body += "valid = CustomImageDatasetLoaderforPytorch(X_train[valid_index], y_train[valid_index], transform = preprocessing['test_or_valid'])\n" if cv_num>0 else ""
        body += "test = CustomImageDatasetLoaderforPytorch(X_test, y_test, transform = preprocessing['test_or_valid'])\n"
    else:
        body = "train = torch.utils.data.TensorDataset(X_train"+ ("[train_index]" if cv_num >0 else "") + ", y_train"+ ("[train_index]" if cv_num>0 else "") +")\n"
        body += "valid = torch.utils.data.TensorDataset(X_train[valid_index], y_train[valid_index])\n" if cv_num>0 else ""
        body += "test = torch.utils.data.TensorDataset(X_test, y_test)\n"
    ##
    body += "train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)\n"
    body += "valid_loader = torch.utils.data.DataLoader(valid, batch_size = batch_size, shuffle = False)\n" if cv_num>0 else ""
    body += "test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)\n"
    ##
    if cv_num < 2:
        body += "\n"
    epoch_title = "check_lr_sensitivity = True\n" if for_hpo_tune else ""
    epoch_title += "model.train()\nfor epoch in range(epochs):\n"
    batch_tr_title = "for batch_index, (data, label) in enumerate(train_loader):\n"
    batch_tr_body = "optimizer.zero_grad()\n"
    #############
    #data = data.view(-1, input_dim).to(DEVICE) # when FNN direct
    #data = data.view(-1, input_dim*input_dim).to(DEVICE) # when FNN indirect
    #data = data.to(DEVICE) # when CNN indirect
    #data = data.to(DEVICE) # when CNN direct(X) -- no case
    #########
    batch_tr_body += "data, label = data"+("" if indirect else ".view(-1, input_dim)")+".to(DEVICE), label.to(DEVICE"+(", dtype=torch.int64" if labeltypeflag=="int" else "")  +")\n"
    batch_tr_body += "if DL_Pytorch_model == 'FNN':\n    data = data.view(data.shape[0], input_dim*input_dim)\n" if indirect else ""
    #########
    batch_tr_body += "output = model(data)"+(".view(-1)" if labeltypeflag == "float" else "")+"\n"
    batch_tr_body += "loss = loss_func(output, label)\nloss.backward()\noptimizer.step()\n"
    batch_tr_loop = batch_tr_title + getIndent(batch_tr_body, indent_level=4)
    batch_tr_loop += "if check_lr_sensitivity:\n    check_lr_sensitivity = False\n    if str(loss.item()) == 'nan':\n        raise optuna.exceptions.TrialPruned()\n" if for_hpo_tune else ""
    if prune_available:
        batch_tr_loop += "l_val = -1*(loss**0.5) # RMSE loss for ASHA pruning\n" if gui_params['task']=='Regression' else "l_val = -1*loss # multi log loss for ASHA pruning\n"
        batch_tr_loop += "trial.report(l_val, epoch)\nif trial.should_prune():\n    raise optuna.exceptions.TrialPruned()\n"
    epoch_loop = epoch_title + getIndent(batch_tr_loop, indent_level=4)
    valid_loop = getDLPytorch_TestFunc(gui_params, 'valid_loader', cv_num, indirect) if cv_num >1 else ""
    test_loop = getDLPytorch_TestFunc(gui_params, 'test_loader', cv_num, indirect) if cv_num <2 else ""
    finalize = "\nscores = np.array(scores)\n" if cv_num >1 else ""
    if cv_num>0:
        body = getIndent(body,indent_level=4)
        epoch_loop = getIndent(epoch_loop, indent_level=4)
        valid_loop = getIndent(valid_loop, indent_level=4, add_return = False)
        test_loop = getIndent(test_loop, indent_level=4, add_return = False)
    results += preprocessing_for_indirect_images + (fold_data if cv_num>0 else "")+ body + epoch_loop + valid_loop +test_loop + finalize
    return results

def getDLPytorch_TestFunc(gui_params, loader_name, cv_num, indirect=False):
    results = "model.eval()\n"
    results += "y_pred_list = []\ny_true_list = []\n"
    results += "with torch.no_grad():\n"
    results += "    for batch_index, (data, label) in enumerate("+loader_name+"):\n"
    results += "        data, label = data"+("" if indirect else ".view(-1, input_dim)")+".to(DEVICE), label.to(DEVICE"+ ("" if gui_params['task']=='Regression' else ", dtype=torch.int64")+")\n"
    results += "        if DL_Pytorch_model == 'FNN':\n            data = data.view(data.shape[0], input_dim*input_dim)\n" if indirect else ""
    results += "        pred = model(data).view(-1)\n" if gui_params['task'] == 'Regression' else "        output = model(data)\n        pred = output.argmax(dim=1, keepdim=True)\n"
    results += "        y_pred_list += pred.cpu().numpy().tolist()\n"
    results += "        y_true_list += label.cpu().numpy().tolist()\n"
    vs_loader_name = "vs"+("" if loader_name == 'test_loader' else "_"+loader_name)
    results += vs_loader_name +" = pd.DataFrame(np.c_[y_pred_list, y_true_list], columns=['Predicted', 'Actual'])\n"
    results += "confidence"+("" if loader_name == 'test_loader' else "_"+loader_name)
    # binary classification 의 f1 score case 적용이 필요함
    results += " = metrics.r2_score("+vs_loader_name+"['Actual'], "+vs_loader_name+"['Predicted'])\n" if gui_params['task']=='Regression' else " = metrics.f1_score("+vs_loader_name+"['Actual'], "+vs_loader_name+"['Predicted'], average='macro')\n"
    results += "scores.append(confidence_"+loader_name+")\n" if loader_name != "test_loader" else ""
    return results

def getDLPytorch_CNNmodel(gui_params, r_val, for_hpo_tune=False, stepwise=False):
    results = "\ndef define_CNNmodel("+ ("trial, " if for_hpo_tune else "" )+ "input_dim"+(", output_dim" if gui_params['task']=='Classification' else "")+"):\n"
    results += "    def calc_cond2d_tensor_size(height_in, kernel_size, padding=0, dilation=1, stride=1):\n        height_out = int( (height_in + 2*padding - dilation*(kernel_size-1)-1)/stride + 1)\n        return height_out\n"
    results += "    def calc_maxpool2d_tensor_size(height_in, kernel_size, padding=0, dilation=1, stride=None):\n        if stride is None:\n            stride = kernel_size\n        height_out = int( (height_in + 2*padding-dilation*(kernel_size-1)-1)/stride + 1 )\n        return height_out\n"
    input_layer = "layers = []\n"
    input_layer += "#default_params\nwindow_size = 2\npooling_size = 2\np = 0.1"
    hidden_layer = ""
    output_layer=""
    add_softmax_layer=""
    bottom_layer=""
    if not for_hpo_tune:
        conv_layers_list = gui_params['hparams']['hidden']['conv'].split(',')
        fcn_layers_list = gui_params['hparams']['hidden']['fcn'].split(',')
        conv_layer = "# 1. conv layers\n"
        conv_layer += 'in_features = 1\n'
        for i in range(len(conv_layers_list)):
            conv_layer += 'out_features = ' + str(conv_layers_list[i])+'\n'
            conv_layer += 'layers.append(nn.Conv2d(in_features, out_features, window_size))\n'
            conv_layer += 'layers.append(nn.ReLU())\n'
            conv_layer += 'layers.append(nn.MaxPool2d(pooling_size))\n'
            conv_layer += 'layers.append(nn.Dropout2d(p))\n'
            conv_layer += 'in_features = out_features\n'
            conv_layer += 'dimK = calc_maxpool2d_tensor_size(calc_cond2d_tensor_size(input_dim, window_size), pooling_size)\n'
            conv_layer += 'input_dim = dimK\n'
        conv_layer += '# to avoid zero division error\n'
        conv_layer += 'if input_dim < window_size:\n'
        conv_layer += '    print("[ERR] the number of convolutional layers should be reduced due to too much pooling. The code will be terminated.")\n'
        conv_layer += '    import sys\n'
        conv_layer += '    sys.exit()\n'
        conv_layer += 'conv_out_dim = out_features * dimK * dimK\n'
        conv_layer += 'layers.append(nn.Flatten())\n'      
        fcn_layer = '# 2. fcn layers\n'
        fcn_layer += "in_features = conv_out_dim\n"
        for i in range(len(fcn_layers_list)):
            fcn_layer += 'out_features = ' + str(fcn_layers_list[i]) + '\n'
            fcn_layer += 'layers.append(nn.Linear(in_features, out_features))\n'
            fcn_layer += 'layers.append(nn.ReLU())\n'
            fcn_layer += 'in_features = out_features\n'
        fcn_layer += 'layers.append(nn.Linear(in_features, output_dim))\n'
        bottom_layer = "return nn.Sequential(*layers)\n"
        results += getIndent(input_layer+conv_layer+fcn_layer+bottom_layer, indent_level=4)     
    else:
        conv_layer = "# 1. conv layers\n"
        conv_layer += 'n_layers_conv = trial.suggest_int("DL_Pytorch_CNN_conv_n_layers",'+ ("params['n_layers']['low']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['n_layers']['low']))+', '+( "params['n_layers']['high']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['n_layers']['high']))+')\n'
        conv_layer += 'in_features = 1\n'
        conv_layer += "for i in range(n_layers_conv):\n"
        conv_layer += '    out_features = trial.suggest_int("DL_Pytorch_CNN_conv_n_units_l{}".format(i),' + ("params['n_units']['low']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['n_units']['low'])) +', ' + ("params['n_units']['high']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['n_units']['high'])) +')\n'
        conv_layer += '    p = trial.suggest_uniform("DL_Pytorch_CNN_conv_dropout_l{}".format(i), '+ ("params['dropout']['low']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['dropout']['low'])) + ',' + ("params['dropout']['high']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['dropout']['high'])) + ')\n'
        conv_layer += '    layers.append(nn.Conv2d(in_features, out_features, window_size))\n'
        conv_layer += '    layers.append(nn.ReLU())\n'
        conv_layer += '    layers.append(nn.MaxPool2d(pooling_size))\n'
        conv_layer += '    layers.append(nn.Dropout2d(p))\n'
        conv_layer += '    in_features = out_features\n'
        conv_layer += '    dimK = calc_maxpool2d_tensor_size(calc_cond2d_tensor_size(input_dim, window_size), pooling_size)\n'
        conv_layer += '    input_dim = dimK\n'
        conv_layer += '    # to avoid zero division error \n'
        conv_layer += '    if input_dim < window_size:\n'
        conv_layer += '        raise optuna.exceptions.TrialPruned()\n'
        conv_layer += 'conv_out_dim = out_features * dimK * dimK\n'
        conv_layer += 'layers.append(nn.Flatten())\n'
        fcn_layer = '# 2. fcn layers\n'
        fcn_layer += 'n_layers_fc = trial.suggest_int("DL_Pytorch_CNN_fc_n_layers",'+ ("params['n_layers']['low']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['n_layers']['low']))+', '+("params['n_layers']['high']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['n_layers']['high']))+')\n'
        fcn_layer += "in_features = conv_out_dim\n"
        fcn_layer += "for i in range(n_layers_fc):\n"
        fcn_layer += '    out_features = trial.suggest_int("DL_Pytorch_CNN_fc_n_units_l{}".format(i),' + ("params['n_units']['low']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['n_units']['low'])) +', ' + ("params['n_units']['high']" if stepwise else str(r_val[gui_params['task']]['DL_Pytorch']['n_units']['high'])) +')\n'
        fcn_layer += '    layers.append(nn.Linear(in_features, out_features))\n'
        fcn_layer += '    layers.append(nn.ReLU())\n'
        fcn_layer += '    in_features = out_features\n'
        fcn_layer += 'layers.append(nn.Linear(in_features, output_dim))\n'
        bottom_layer = "return nn.Sequential(*layers)\n"
        results += getIndent(input_layer+conv_layer+fcn_layer+bottom_layer, indent_level=4)
    return results

def getIndent(sentenses,indent_level=4, indent_type='space', add_return = True):
    if indent_type=='space':
        indent = ' '*indent_level
    elif indent_type=='tab':
        indent = '\t'*indent_level
    return "\n".join([indent+each_line for each_line in sentenses.split('\n')])+("\n" if add_return else "")


############################################################################################
### 20200609 add automl code end
############################################################################################
def saveModel(gui_params):
    results = "\n######################################################################\n## You have to save the AI Model for submitting your model\n"
    results = results + "## You have to prepare a parameter: model\n## (trained model that you want to register)\n"
    results = results + "## saveAIModel parameters description below ##\n"
    results = results + "## total is the number of model you want to register, and modelInfo is the array of dict that contains the information of model"
    results = results + "## modelInfo example: [{'model' : clf, 'modelName' : 'preprocess.pkl', 'framework' : 'sklearn'}, {'model' : clf.model, 'modelName' : 'predict.h5', 'framework' : 'keras'},\n"
    results = results + "## {'model' : clf, 'modelName' : 'model.pth', 'framework' : 'pytorch'}, {'model' : clf, 'modelName' : 'model.pkl', 'framework' : 'chainer'}]\n"
    results = results + "######################################################################\nimport sdr\n"
    if gui_params['algorithm'] == 'DL':
        results = results + "aiModel = sdr.AIModel()\nmodelInfo = [{'model': clf.model, 'modelName': 'model.h5', 'framework': 'keras'}]\naiModel.saveAIModel(total=1, modelInfo=modelInfo)"
    else:
        results = results + "aiModel = sdr.AIModel()\nmodelInfo = [{'model': clf, 'modelName': 'model.pkl', 'framework': 'sklearn'}]\naiModel.saveAIModel(total=1, modelInfo=modelInfo)"
    return results

def getEvaluation(gui_params):
    results = ""
    if gui_params['analysis']  is not None:
        results = results + getAdditionalAnalyticResults(gui_params)
    if gui_params['perf_eval'] is not None:
        results = results + "\n##* Performance Evaluation\n" + getPerformanceEvaluationResults(gui_params)
    return results

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
    #for index in range(len(gui_params['input_columns_index_and_name'])):
    #    input_cols.append(gui_params['input_columns_index_and_name'][str(index)])
    #    input_index_list.append(index)

    return input_cols, output_col, input_index_list, output_index

def getExceptionalHyperParameters(gui_params):
    inner_options_list=[]
    inner_options_string=""
    #annotate_list_list=[]
    #annotate_list_string=""

    for hparam in gui_params['hparams']:
        if gui_params['algorithm'] == 'LR':
            if hparam == 'nn':
                #annotate_list.append("nn")
                inner_options_list.append("nn = "+str(gui_params['hparams'][hparam]))
            elif hparam == 'scale':
                #annotate_list.append("scale")
                inner_options_list.append("scale = "+str(gui_params['hparams'][hparam]))
        # generate strings through option and annotation lists
        if inner_options_list:
            inner_options_string = ", " + ", ".join(inner_options_list)
        #if annotate_list:
        #    annotate_list_string = "\t# " + ", ".join(annotate_list)
    return inner_options_string

def getHyperParametersandClosing(gui_params, for_hpo_tune):
    results = ""
    options_and_close = ")\n"


    python_cv_template = '''
##* Cross validation score

from sklearn.model_selection import KFold

kf = KFold(n_splits=2, random_state=None, shuffle=False)


'''
    if gui_params['hparams']:   # if it exists
        inner_options_list = []
        inner_options_string=""

        annotate_list = []
        annotate_list_string=""

        outer_options_list=[]
        outer_options_string=""
        for hparam in gui_params['hparams']:
            if gui_params['algorithm'] == 'MLR':
                if hparam == 'cv':
                    clf_name = "clf"
                    annotate_list.append("K-fold cross-validation")
                    outer_options_string = "\n## Cross validation score\nfrom sklearn.model_selection import cross_val_score\nscores = cross_val_score("+clf_name+", train_data[features], train_data[target], cv = "+str(gui_params['hparams'][hparam])+", n_jobs = "+str(n_jobs)+")\n" + "" if for_hpo_tune else "print(scores.mean())"
                elif hparam == 'fit_intercept':
                    annotate_list.append("Fit_intercept")
                    inner_options_list.append("fit_intercept = " + str(gui_params['hparams'][hparam]))
                elif hparam == 'normalize':
                    annotate_list.append("Normalize") #'linear', 'polynomial', 'radial', and 'sigmoid'
                    inner_options_list.append('normalize = '+ str(gui_params['hparams'][hparam]))
            elif gui_params['algorithm'] == 'SVM':
                if hparam == 'cv':
                    clf_name = "clf"
                    annotate_list.append("K-fold cross-validation")
                    outer_options_string = "\n## Cross validation score\nfrom sklearn.model_selection import cross_val_score\nscores = cross_val_score("+clf_name+", train_data[features], train_data[target], cv = "+str(gui_params['hparams'][hparam])+(", scoring='f1_macro'" if gui_params['task']=='Classification' else "")+", n_jobs = "+str(n_jobs)+")\n" + "" if for_hpo_tune else "print(scores.mean())"
                elif hparam == 'C':
                    annotate_list.append("SVM error term")
                    inner_options_list.append('C = '+ str(gui_params['hparams'][hparam]))
                elif hparam == 'kernel':
                    annotate_list.append("SVM Kernel select") #'linear', 'polynomial', 'radial', and 'sigmoid'
                    temp = str(gui_params['hparams'][hparam])
                    if not for_hpo_tune:
                        temp = "'"+temp+"'"
                    inner_options_list.append('kernel = '+ temp )
                elif hparam == 'degree':
                    annotate_list.append("Degree of polynomial kernal function")
                    inner_options_list.append('degree = '+ str(gui_params['hparams'][hparam]))
                elif hparam == 'gamma':
                    annotate_list.append("Gamma (kernel coefficient)")
                    temp = gui_params['hparams'][hparam]
                    if not for_hpo_tune:
                        if type(temp) is not str:
                            temp = str(temp)
                        else: # when string
                            temp = "'"+temp+"'"
                    inner_options_list.append('gamma = '+ temp)
                ## 20200609 automl added
                elif hparam == 'tol':
                    annotate_list.append("Tolerance for stopping criterion")
                    inner_options_list.append('tol = '+ str(gui_params['hparams'][hparam]))
                elif hparam == 'epsilon':
                    annotate_list.append("Epsilon in the epsilon-SVR model")
                    inner_options_list.append('epsilon = '+ str(gui_params['hparams'][hparam]))
                #elif hparam == 'class_weight':
                #    annotate_list.append("class weight to C")
                #    if gui_params['hparams'][hparam] not in ['True', 'False', 'None']:
                #        mod_hparam = "'" + gui_params['hparams'][hparam] + "'"
                #    else:
                #        mod_hparam = gui_params['hparams'][hparam]
                #    inner_options_list.append('class_weight = '+ mod_hparam)
                elif hparam == 'class_weight':
                    annotate_list.append("class weight to C")
                    temp = str(gui_params['hparams'][hparam])
                    if not for_hpo_tune:
                        if 'None' != temp:
                            temp = "'" + temp +"'"
                    inner_options_list.append('class_weight = '+ temp)
            elif gui_params['algorithm'] == 'RF':
                if hparam == 'cv':
                    clf_name = "clf"
                    annotate_list.append("K-fold cross-validation")
                    outer_options_string = "\n## Cross validation score\nfrom sklearn.model_selection import cross_val_score\nscores = cross_val_score("+clf_name+", train_data[features], train_data[target], cv = "+str(gui_params['hparams'][hparam])+(", scoring='f1_macro'" if gui_params['task']=='Classification' else "")+", n_jobs = "+str(n_jobs)+")\n" + "" if for_hpo_tune else "print(scores.mean())"
                elif hparam == 'n_estimators':
                    annotate_list.append("Number of trees")
                    inner_options_list.append('n_estimators = '+ str(gui_params['hparams'][hparam]))
                elif hparam == 'criterion':
                    annotate_list.append("Quality of a split")
                    temp = str(gui_params['hparams'][hparam])
                    if not for_hpo_tune:
                        temp = "'"+temp+"'"
                    inner_options_list.append('criterion = '+ temp)
                elif hparam == 'max_depth':
                    annotate_list.append("The max. depth of the tree")
                    inner_options_list.append('max_depth = '+ str(gui_params['hparams'][hparam]))
                elif hparam == 'min_samples_split':
                    annotate_list.append("The min. num required to split an internal node")
                    inner_options_list.append('min_samples_split = '+ str(gui_params['hparams'][hparam]))
                elif hparam == 'min_samples_leaf':
                    annotate_list.append("The max. num required to be at a leaf node")
                    inner_options_list.append('min_samples_leaf = '+ str(gui_params['hparams'][hparam]))
                ## 20200609 automl added
                elif hparam == 'max_features':
                    annotate_list.append("The number of features to consider when looking for the basic split:")
                    inner_options_list.append('max_features = '+ str(gui_params['hparams'][hparam]))
            elif gui_params['algorithm'] == 'BT':
                if hparam == 'cv':
                    clf_name = "clf"
                    annotate_list.append("K-fold cross-validation")
                    outer_options_string = "\n## Cross validation score\nfrom sklearn.model_selection import cross_val_score\nscores = cross_val_score("+clf_name+", train_data[features], train_data[target], cv = "+str(gui_params['hparams'][hparam])+(", scoring='f1_macro'" if gui_params['task']=='Classification' else "")+", n_jobs = "+str(n_jobs)+")\n" + "" if for_hpo_tune else "print(scores.mean())"
                elif hparam == 'n_estimators':
                    annotate_list.append("Number of estimators")
                    inner_options_list.append('n_estimators = '+ str(gui_params['hparams'][hparam]))
                elif hparam == 'learning_rate':
                    annotate_list.append("Learning rate")
                    inner_options_list.append('learning_rate = '+ str(gui_params['hparams'][hparam]))
                elif hparam == 'loss':
                    annotate_list.append("Loss function")
                    temp = str(gui_params['hparams'][hparam])
                    if not for_hpo_tune:
                        temp = "'"+temp+"'"
                    inner_options_list.append('loss = '+ temp)
                elif hparam == 'algorithm':
                    annotate_list.append("algorithm")
                    temp = str(gui_params['hparams'][hparam])
                    if not for_hpo_tune:
                        temp = "'"+temp+"'"
                    inner_options_list.append('algorithm = '+ temp)
##########################################
##########################################
            elif gui_params['algorithm'] == 'LR':
                #nothing to do
                pass
            elif gui_params['algorithm'] == 'DL':
                if hparam == 'cv':
                    clf_name = "clf"
                    annotate_list.append("k-fold cross-validation")
                    outer_options_string = "\n## Cross validation score\nfrom sklearn.model_selection import cross_val_score\nscores = cross_val_score("+clf_name+", train_data[features], train_data[target], cv = "+str(gui_params['hparams'][hparam])+(", scoring='f1_macro'" if gui_params['task']=='Classification' else "")+", n_jobs = "+str(n_jobs)+")\n" + "" if for_hpo_tune else "print(scores.mean())"
                elif hparam == 'epochs':
                    annotate_list.append("dataset iteration number")
                    inner_options_list.append("epochs = "+ str(gui_params['hparams'][hparam]))
                elif hparam == 'hidden':
                    annotate_list.append("hidden layer network design")
                    inner_options_list.append("hidden = c(" + str(gui_params['hparams'][hparam]) + ")")

            #######################################
            # common options - custom options
            if hparam == 'custom':
                annotate_list.append("custom options")
                inner_options_list.append(str(gui_params['hparams'][hparam]))

            # generate strings through option and annotation lists
            if inner_options_list:
                #inner_options_string = ", " + ", ".join(inner_options_list)
                inner_options_string = ", ".join(inner_options_list)

            if annotate_list:
                annotate_list_string = "# " + ", ".join(annotate_list)
            #######################################
        return inner_options_string + options_and_close + annotate_list_string +outer_options_string#+ "\n\n"
    else:
        return options_and_close + "\n"

def getTrainXyTestXyarray(column_name_list, which_data):
    begin = "np.c_["
    intermediate = ""
    converted = list()

    if type(column_name_list) is list:
        for item in column_name_list:
            converted.append(which_data+"['"+item+"']") # ex) item == thickness -> train_data['thickness']
        intermediate = ", ".join(converted)
    else:
        intermediate = which_data + "['" + column_name_list + "']"
    end = "]"
    return begin+intermediate+end + "\n"


##############################################################################################
##  Algorithms
##############################################################################################

def getDeepLearningHiddenNetwork(gui_params, hidden_net_params):
    top = ""
    middle_1 = ""
    middle_2 = ""
    middle_3 = ""
    bottom = ""
    activation = ""
    metrics = ""
    top = "## Learning network design\nfrom keras.models import Sequential\nfrom keras.layers import Dense\n\ndef create_network():\n"
    if gui_params['task'] == 'Regression':
        activation = "relu"
        metrics = "'mse', 'mae'"
    elif gui_params['task'] == 'Classification':
        activation = 'tanh'
        metrics = "'accuracy'"
    middle_1 = "    model = Sequential()\n    model.add(Dense(" + hidden_net_params[0] + ", input_dim = input_dim, activation = '"+activation+"'))\n"
    for i in range(1,len(hidden_net_params)):
        middle_2 = middle_2 + getDeepLearningDenseLayer(hidden_net_params[i], activation)
    if gui_params['task'] == 'Regression':
        middle_3 = getDeepLearningDenseLayer(1, "")
    elif gui_params['task'] == 'Classification':
        if gui_params['hparams']['loss'] == 'binary_crossentropy':
            middle_3 = getDeepLearningDenseLayer("1", "softmax")
        else:
            middle_3 = getDeepLearningDenseLayer("len(label_names)", "softmax")
    bottom = "    model.compile(loss = loss, optimizer = optimizer, metrics = ["+metrics+"])\n    model.summary()\n    return model\n"
    return top + middle_1 + middle_2 + middle_3 + bottom

def getDeepLearningDenseLayer(num, act):
    if act == "":
        return "    model.add(Dense("+str(num)+"))\n"
    else:
        return "    model.add(Dense("+str(num)+", activation = '" + str(act) + "'))\n"

def getDeepLearning(gui_params):
    # getting column names
    input_cols, output_col, input_index_list, output_index = getInputOutputColumnNames(gui_params)
    hidden_net_params=""
    cv_num = -1
    results = "##* Hyperparameters for deep learning\ninput_dim = train_data[features].shape[1]\t# number of features\n" 
    if gui_params['hparams']:   # if it exists
        for hparam in gui_params['hparams']: # mantantories first
            if hparam == 'epochs':
                results = results + "training_epochs = " + str(gui_params['hparams'][hparam]) + "\n"
            elif hparam == 'batch_size':
                results = results + "batch_size = " + str(gui_params['hparams'][hparam]) + "\n"
            elif hparam == 'loss':
                results = results + "loss = '" + str(gui_params['hparams'][hparam]) + "'\n"
            elif hparam == 'optimizer':
                results = results + "optimizer = '" + str(gui_params['hparams'][hparam]) + "'\n"
            elif hparam == 'hidden':
                hidden_net_params = str(gui_params['hparams'][hparam]).split(",")
            elif hparam == 'cv':
                if type(gui_params['hparams'][hparam]) is int:
                    cv_num = int(gui_params['hparams'][hparam])
                else:
                    cv_num = 0
   
    results = results +"\n" +getDefaultParametersDetails(gui_params) # describing default values of mandantory hyperparameters [nfolds, epochs, hidden]
    
    if gui_params['task'] == 'Regression':
        results = results + getDeepLearningHiddenNetwork(gui_params, hidden_net_params)
        results = results + "\n##* Deep Learning Regression\nfrom keras.wrappers.scikit_learn import KerasRegressor\nclf = KerasRegressor(build_fn=create_network, epochs = training_epochs, batch_size = batch_size, verbose=True)\n"
    elif gui_params['task'] == 'Classification':
        results = results + getDeepLearningHiddenNetwork(gui_params, hidden_net_params)
        results = results + "\n##* Deep Learning Classification\nfrom keras.wrappers.scikit_learn import KerasClassifier\nclf = KerasClassifier(build_fn=create_network, epochs = training_epochs, batch_size = batch_size, verbose=True)\n"
    if cv_num > 0:
        results = results + "##* cross validation score\nfrom sklearn.model_selection import KFold\nfrom sklearn.model_selection import cross_val_score\nseed = np.random.seed()\nn_splits = " + str(cv_num) + "\nkfold = KFold(n_splits = n_splits, shuffle = True, random_state = seed)\nresults = cross_val_score(clf, train_data[features], train_data[target], scoring='r2', cv = kfold)\nprint('*** Mean value of the cross-validation scores(r2) : ', results.mean())\n"
    ## results
    #results = results + "\n## results\ntemp = create_network()\ntemp.summary()\n"

    results = results + "\n## Fitting model and make prediction\nhistory = clf.fit(train_data[features], train_data[target])\npredicted = clf.predict(test_data[features])\n"

    results = results + getBasicAnalyticGraph(output_col, gui_params['task'])
    return results

##############################################################################################
##  Several Analysis Tools and Performance Evaluations (optional)
##############################################################################################

def getBasicAnalyticGraph(output_col, task):
    results = '''
## Comparing output data of the testing datasets(never learned) with the predicted output data using input data of the testing datasets.
vs = pd.DataFrame(np.c_[predicted, test_data[target]], columns=['Predicted', 'Actual'])
print(vs.head())
'''

    if task == 'Regression':
        results = results + '''
##* Plotting testing datasets vs. predicted datasets
vsplot, ax = plt.subplots(1, 1, figsize=(12,12))
ax.scatter(x = predicted, y = test_data[target], color='c', edgecolors=(0, 0, 0))
ax.plot([test_data[target].min(), test_data[target].max()], [test_data[target].min(), test_data[target].max()], 'k--', lw=4)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()'''

    return results

def getAdditionalAnalyticResults(gui_params):
    results = ""
    value = ""
    for item in gui_params['analysis']:
        if item == "ScoringHistory":
            if gui_params['task'] == 'Regression':
                value = "mean_squared_error"
            elif gui_params['task'] == 'Classification':
                value = "acc"
            results = results + "\n##* Training Scoring History\nplt.rcParams['figure.figsize']=(12,11)\nplt.plot(history.history['"+str(value)+"'])\n"+"plt.title('Training Scoring History')\nplt.ylabel('training_score')\nplt.xlabel('epoch')\nplt.legend(['train'], loc='upper left')\nplt.show()\n"
        elif item == "CorrelationHeatmap":
            results = results + "\n##* Correlation among all columns in all_data\nall_corr = all_data.corr()\nfig, ax = plt.subplots(figsize=(8,6))\nsns.heatmap(all_corr)\n"
        elif item == "MaxInputCorrelation":
            results = results + "\n##* Correlation among input columns regarding the target column in train_data\ncorr = train_data[all_data.columns].corr()[target].abs()\nprint(corr)\nmax_corr = corr.drop(target).idxmax()\t# get the highest correlated column regarding the target column\nsns.lmplot(x = max_corr, y = target, data = train_data)\n"
        elif item == 'Variable Importance':
            if gui_params['algorithm'] == 'RF' or 'BT':
                results = results + "\n##* Importance of each predictor"
                results = results + "\nvi = pd.DataFrame([clf.feature_importances_], columns=features)\nprint(vi)\n"
        elif item == 'BoxPlot':
            results = results + "\n##* Box Plot\nall_data[features].plot(kind='box', subplots=True, sharex=False, sharey=False)\nplt.show()\n"
        elif item == 'Histogram':
            results = results + "\n##* Histogram\nall_data[features].hist()\nplt.show()\n"
        elif item == 'ScatterMatrix':
            results = results + "\n##* Scatter Plot Matrix\nfrom pandas.plotting import scatter_matrix\nscatter_matrix(all_data[features])\nplt.show()\n"
        elif item == 'ConfusionMatrix':
            results = results + '''
##* Confusion Matrix
## https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


## Compute confusion matrix
cnf_matrix = confusion_matrix(vs['Actual'].values, vs['Predicted'].values)
np.set_printoptions(precision=2)

##* Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=label_names,
                      title='Confusion matrix, without normalization')
plt.show()

##* Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=label_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
'''
    return results

def getAdditionalAnalyticResults2(gui_params):
    results = ""
    for item in gui_params['analysis']:
        if item == 'leveragePlots': # available in MLR, not in SVM
            if gui_params['algorithm'] == 'MLR':
                #results = results + '\n##* Leverage Plots\nlibrary(car)\nleveragePlots(model)\n'
                results = ""

        elif item == 'qqPlot':  # available in MLR, not in SVM
            if gui_params['algorithm'] == 'MLR':
                #results = results + '\n##* Quantile-Quantile Plot\nlibrary(car)\nqqPlot(model, main="QQ Plot")\n'
                results = ""

        elif item == 'Distribution of Std. Residuals': # available in MLR, not in SVM
            if gui_params['algorithm'] == 'MLR':
                #results = results + '\n##* Distribution of Studentized Residuals\nlibrary(MASS)\nsresid <- studres(model)\nhist(sresid, freq=FALSE, main="Distribution of Studentized Residuals")\nxfit<-seq(min(sresid),max(sresid),length=40)\nyfit<-dnorm(xfit)\nlines(xfit, yfit)\n'
                results = ""

        elif item == 'Diagnostic Plots': # available in MLR, not in SVM
            if gui_params['algorithm'] == 'MLR':
                #results = results + "\n##* Diagnostic Plots\nlayout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page\nplot(model)\nlayout(1)\n"
                results = ""

        elif item == 'OutlierTest': # available in MLR, not in SVM
            if gui_params['algorithm'] == 'MLR':
                #results = results + "\n##* Assessing Outliers\nprint('***Outlier Test ***')\nprint(outlierTest(model)) # Bonferonni p-value for most extreme obs\n"
                results = ""

        elif item == 'Variable Importance':
            results = results + "\n##* Importance of each predictor"
            if gui_params['algorithm'] == 'MLR':
                #results = results + "\nlibrary(caret)\nvarImp(model)\n"
                results = ""
            elif gui_params['algorithm'] == 'RF':
                results = results + "\nprint('Variable Importance: ',clf.feature_importances_)\n"
            elif gui_params['algorithm'] == 'BT':
                results = results + "\nsummary(model)\n"
            elif gui_params['algorithm'] == 'DL':
                results = results + "\nh2o.varimp(model)\nh2o.varimp_plot(model)\n"
            else:
                results = results + "\n# Variable importance may not supported in this algorithm.\n"

        elif item == 'AP Confusion Matrix':
            if gui_params['task'] == 'Classification':
                results = results + "\n##* Alluvial Plotting using Confusion Matrix\n"
                results = results + '''alluvial_plot_from_cm <- function(cm){
cmdf <- as.data.frame(cm[["table"]])
cmdf[["color"]] <- ifelse(cmdf[[1]] == cmdf[[2]], "blue", "red")

alluvial::alluvial(cmdf[,1:2]
                   , freq = cmdf$Freq
                   , col = cmdf[["color"]]
                   , border = "white"
                   , alpha = 0.5
                   , hide  = cmdf$Freq == 0)
}
alluvial_plot_from_cm(cmatrix)
'''
    return results

def getPerformanceEvaluationResults(gui_params):
    results = ""
    temp = ""
    if gui_params['task'] == 'Regression':
        for item in gui_params['perf_eval']:
            temp = '''# We support several metrics for regression by using sklearn.metrics functions
# metrics.explained_variance_score(y_true, y_pred)\tExplained variance regression score function
# metrics.mean_absolute_error(y_true, y_pred)\tMean absolute error regression loss
# metrics.mean_squared_error(y_true, y_pred[, ...])\tMean squared error regression loss
# metrics.mean_squared_log_error(y_true, y_pred)\tMean squared logarithmic error regression loss
# metrics.median_absolute_error(y_true, y_pred)\tMedian absolute error regression loss
# metrics.r2_score(y_true, y_pred[, ...])\tR^2 (coefficient of determination) regression score function.\n'''
            if item == 'mae':
                results = results + "\n## Mean Absolute Error\nperf_mae = metrics.mean_absolute_error(vs['Actual'],vs['Predicted'])\nprint('*** Mean Absolute Error = ', perf_mae)\n"
            elif item == 'mape':
                results = results + "\n## Mean Absolute Percentage Error\ndef mean_absolute_percentage_error(y_true, y_pred):\n    y_true, y_pred = np.array(y_true), np.array(y_pred)\n    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100\nperf_mape = mean_absolute_percentage_error(vs['Actual'], vs['Predicted'])\nprint('*** Mean Absolute Percentage Error = ', perf_mape)\n"
            elif item == 'rmse':
                results = results + "\n## Root Mean Squared Error\nperf_rmse = np.sqrt(metrics.mean_squared_error(vs['Actual'], vs['Predicted']))\nprint('*** Root Mean Squared Error = ', perf_rmse)\n"
            elif item == 'rmsle':
                results = results + "\n## Root Mean Squared Log Error\ndef rmsle(y, y0):\n    return np.sqrt(np.mean(np.square(np.log1p(abs(y)) - np.log1p(abs(y0)))))\t# vectorization for supporting rmsle calculation with negative values\n\nperf_rmsle = rmsle(vs['Actual'], vs['Predicted'])\nprint('*** Root Mean Squared Log Error = ', perf_rmsle)\n"
            elif item == 'r2':
                results = results + "\n## Coefficient of Determination (R Squared)\nperf_r2 = metrics.r2_score(vs['Actual'], vs['Predicted'])\nprint('*** Coefficient of Determination (R Squared) = ', perf_r2)\n"

    elif gui_params['task'] == 'Classification':
        for item in gui_params['perf_eval']:
            if item == 'accuracy':
                results = results + "\n## Categorization Accuracy\nfrom sklearn.metrics import accuracy_score\nperf_acc_score = accuracy_score(vs['Actual'], vs['Predicted'])\nprint('*** Categorization Accuracy = ',perf_acc_score)\n"
            elif item == 'meanf1':
                results = results + "\n## Mean F1 Score\nfrom sklearn.metrics import f1_score\nperf_mean_f1_score = f1_score(vs['Actual'], vs['Predicted'], average=None).mean()\nprint('*** Mean F-1 Score = ', perf_mean_f1_score)\n"
    return results

def getDefaultParametersDetails(gui_params):
    defaultParameters = ""
    seeAllParameters = ""
    referenceAddress = ""
    description = "#- [OPTION_NAME = (default value)]"
    indent = "\n"

    if gui_params['algorithm'] == 'MLR':
        defaultParameters ='''
#- fit_intercept = True // boolean, optional. Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered).
#- normalize = False // boolean, optional. This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. If you wish to standardize, please use sklearn.preprocessing.StandardScaler before calling fit on an estimator with normalize=False.
'''
        referenceAddress = "#- http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html\n"

    elif gui_params['algorithm'] == 'SVM':
        defaultParameters ='''
#- C = 1.0 // float, optional. Penalty parameter C of the error term.
#- kernel = 'rbf' // string, optional. Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable. If none is given, 'rbf' will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
#- degree = 3 // int, optional. Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
#- gamma = 'auto' // float, optional. Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Current default is 'auto' which uses 1 / n_features, if gamma='scale' is passed then it uses 1 / (n_features * X.std()) as value of gamma. The current default of gamma, 'auto', will change to 'scale' in version 0.22. 'auto_deprecated', a deprecated version of 'auto' is used as a default indicating that no explicit value of gamma was passed.
'''
        referenceAddress = "#- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR\n#- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC\n"

    elif gui_params['algorithm'] == 'RF':
        defaultParameters ='''
#- n_estimators = 100 // int, optional. The number of trees in the forest.
#- criterion // string, optional. The function to measure the quality of a split.
#- criterion = 'mse' // string, optional. For regression only. 'mse' and 'mae' are supported.
#- criterion = 'gini' // string, optional. For classification only. 'gini' and 'entropy' are supported.
#- max_depth = None // int or None, optional. The maximum depth of the tree. If none, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
#- min_samples_split = 2 // int, optional. The minimum number of samples required to split an internal node: If int, then consider min_samples_split as the minimum number. If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
#- min_samples_leaf = 1 // int, optional. The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If int, then consider min_samples_leaf as the minimum number. If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
'''
        #seeAllParameters = ""
        referenceAddress = "#- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html\n#- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html\n"

    elif gui_params['algorithm'] == 'BT':
        defaultParameters ='''
#- n_estimators = 50 // int, optional. The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
#- learning_rate = 1.0 // float, optional. Learning rate shrinks the contribution of each regressor by learning_rate. There is a trade-off between learning_rate and n_estimators.
#- loss = 'linear' // string, optional. For regression only. The loss function to use when updating the weights after each boosting iteration. 'linear', 'square', 'exponential' are supported.
#- algorithm = 'SAMME.R' // string, optional. For classification only. If 'SAMME.R' then use the SAMME.R real boosting algorithm. base_estimator must support calculation of class probabilities. If 'SAMME' then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.
'''
        #seeAllParameters =""
        referenceAddress = "#- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor\n#- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier\n"

    elif gui_params['algorithm'] == 'DL':
        # mandantories - nfolds, epochs, hidden
        defaultParameters = '''

#- n_splits = None // Number of folds for K-fold cross-validation (0 to disable or >= 2).
#- training_epochs = 10 // How many times the dataset should be iterated (streamed).
#- batch_size = 32 // Batch size is a term used in machine learning and refers to the number of training examples utilized in one iteration.
#- hidden = "20,20" // Hidden layer sizes. It can be modified in the create_network() function.
#- loss = "mean_squared_error"// A loss function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. 'mean_squared_error', 'mean_absolute_error', 'categorical_crossentropy', etc., can be supported. https://keras.io/losses/
#- optimizer = "adam" //  A optimizer is an optimization algorithm. 'adam', 'sgd', 'rmsprop', etc., can be supported. https://keras.io/optimizers/
'''
        #seeAllParameters = '''#- [Parameters Details] h2o.deeplearning(x, y, training_frame, model_id = NULL,validation_frame = NULL, nfolds = 0,keep_cross_validation_predictions = FALSE,keep_cross_validation_fold_assignment = FALSE, fold_assignment = c("AUTO","Random", "Modulo", "Stratified"), fold_column = NULL,ignore_const_cols = TRUE, score_each_iteration = FALSE,weights_column = NULL, offset_column = NULL, balance_classes = FALSE,class_sampling_factors = NULL, max_after_balance_size = 5,max_hit_ratio_k = 0, checkpoint = NULL, pretrained_autoencoder = NULL,overwrite_with_best_model = TRUE, use_all_factor_levels = TRUE,standardize = TRUE, activation = c("Tanh", "TanhWithDropout", "Rectifier","RectifierWithDropout", "Maxout", "MaxoutWithDropout"), hidden = c(200,200), epochs = 10, train_samples_per_iteration = -2,target_ratio_comm_to_comp = 0.05, seed = -1, adaptive_rate = TRUE,h2o.deeplearning 59rho = 0.99, epsilon = 1e-08, rate = 0.005, rate_annealing = 1e-06,rate_decay = 1, momentum_start = 0, momentum_ramp = 1e+06,momentum_stable = 0, nesterov_accelerated_gradient = TRUE,input_dropout_ratio = 0, hidden_dropout_ratios = NULL, l1 = 0, l2 = 0,max_w2 = 3.4028235e+38, initial_weight_distribution = c("UniformAdaptive","Uniform", "Normal"), initial_weight_scale = 1, initial_weights = NULL,initial_biases = NULL, loss = c("Automatic", "CrossEntropy", "Quadratic","Huber", "Absolute", "Quantile"), distribution = c("AUTO", "bernoulli","multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace","quantile", "huber"), quantile_alpha = 0.5, tweedie_power = 1.5,huber_alpha = 0.9, score_interval = 5, score_training_samples = 10000,score_validation_samples = 0, score_duty_cycle = 0.1,classification_stop = 0, regression_stop = 1e-06, stopping_rounds = 5,stopping_metric = c("AUTO", "deviance", "logloss", "MSE", "RMSE", "MAE","RMSLE", "AUC", "lift_top_group", "misclassification","mean_per_class_error"), stopping_tolerance = 0, max_runtime_secs = 0,score_validation_sampling = c("Uniform", "Stratified"),diagnostics = TRUE, fast_mode = TRUE, force_load_balance = TRUE,variable_importances = TRUE, replicate_training_data = TRUE,single_node_mode = FALSE, shuffle_training_data = FALSE,missing_values_handling = c("MeanImputation", "Skip"), quiet_mode = FALSE,autoencoder = FALSE, sparse = FALSE, col_major = FALSE,average_activation = 0, sparsity_beta = 0,max_categorical_features = 2147483647, reproducible = FALSE,export_weights_and_biases = FALSE, mini_batch_size = 1,categorical_encoding = c("AUTO", "Enum", "OneHotInternal", "OneHotExplicit","Binary", "Eigen", "LabelEncoder", "SortByResponse", "EnumLimited"),elastic_averaging = FALSE, elastic_averaging_moving_rate = 0.9,elastic_averaging_regularization = 0.001, verbose = FALSE)\n'''
        referenceAddress = "#- https://keras.io\n"

    return  description + defaultParameters + "#- You can find parameters details at\n" + referenceAddress + indent
