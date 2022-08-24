"""
  Searching Strategies for Hyper Parameter Optimization by Jeongcheol lee
  -- jclee@kisti.re.kr
"""
import os,sys, copy, random

from mpi4py import MPI
import optuna
######################################


class ModObjectiveFunctionWrapper(object):
    def __init__(self, objective_function, params): 
        self.params = params
        self.objective_function = objective_function
    def __call__(self, trial): # 여기가 기존 objective function 이 위치할 곳. 먼저 string wrapping 되어있는 params을 compile하고 기존 func 호출
        return self.objective_function(trial, self.params)
        try:
            return self.objective_function(trial, self.params)
        except:
            optuna.exceptions.OptunaError("[err] object function cannot be generated.")
            raise optuna.exceptions.TrialPruned()

### save module added @ 20200812
#retrieve_model('SVM', clf, trial.number,[predicted, test_data[features]], metric='r2')
def retrieve_model(algorithm_name, model, trial_number, score, metric = None, label_names = None, top_n_all = 10, top_n_each_algo = 3, direction='maximize',\
                   ): # if classification, one can generates confusion matrix png with a specific label_names = ['apple', 'banana', 'mango']
    ''' top_n_all and top_n_each_algo are at least >= 1'''
    ''' 'score' can be the score as well as the dataframe consisting of predicted values and original values ;; [y_pred, y_true]'''
    ''' if the dataframe cases, both a model and a performance figure file will be generated, respectively.'''
    ##############################
    import numpy as np
    make_png = False
    vs=None
    if metric:
        vs = score
        try:
            vs[0] = vs[0].values # y_pred
        except:
            pass
        try:
            vs[1] = vs[1].values # y_true
        except:
            pass
        try:
            vs[0] = np.array(vs[0])
            vs[1] = np.array(vs[1])
        except:
            pass
        y_pred=vs[0]
        y_true=vs[1]
        make_png = True
        if metric == 'r2':
            from sklearn.metrics import r2_score
            score = r2_score(y_true, y_pred)
        elif metric == 'f1':
            from sklearn.metrics import f1_score
            try:
                score = f1_score(y_true, y_pred, average='macro')
            except:
                y_pred = np.vstack(y_pred)
                y_true = np.vstack(y_true)
                vs[0] = y_pred
                vs[1] = y_true
                score = f1_score(y_true, y_pred, average='macro')
    import os, glob
    ##
    output_model_path = "output_models/"
    try:
        if not os.path.exists(output_model_path):
            os.mkdir(output_model_path)
            os.chmod(output_model_path, 0o776) # add permission 201030
        ##
        algorithm_path = output_model_path + algorithm_name+ "/"
        if not os.path.exists(algorithm_path):
            os.mkdir(algorithm_path)
            os.chmod(algorithm_path, 0o776) # add permission 201030
        ##
        top_n_path = output_model_path + "top_"+str(top_n_all)+"/"
        if not os.path.exists(top_n_path):
            os.mkdir(top_n_path)
            os.chmod(top_n_path, 0o776) # add permission 201030
    except:
        pass
    ##
    ##
    file_prefix = str(trial_number)+"__"+algorithm_name+"__"+str(score)
    extension = ".pth" if algorithm_name == 'DL_Pytorch' else ".pkl"
    #
    ############################################################################
    compare_and_search(target_path=algorithm_path, top_n=top_n_each_algo, specific_extension=extension, \
        algorithm_name=algorithm_name, file_prefix=file_prefix, model=model,                            \
        trial_number=trial_number, score=score, direction=direction,                                    \
        make_png=make_png, vs=vs, metric=metric, label_names=label_names)
    #
    compare_and_search(target_path=top_n_path, top_n=top_n_all, specific_extension=None,                \
        algorithm_name=algorithm_name, file_prefix=file_prefix, model=model,                            \
        trial_number=trial_number, score=score, direction=direction,                                    \
        make_png=make_png, vs=vs, metric=metric, label_names=label_names)
    return score
    ######### part 1 remaining top_n_each_algo

def compare_and_search(target_path, top_n, specific_extension, algorithm_name, file_prefix, model, trial_number, score, direction,
                        make_png=False, vs=None, metric=None, label_names=None):
    import glob, os
    if specific_extension is None:
        extension = ""
    else:
        extension = specific_extension
    cur_files = list(set(glob.glob(target_path+"*"+extension)) - set(glob.glob(target_path+"*"+".png")))
    if len(cur_files) >= top_n:
        cur_scores = {}
        for each in cur_files:
            each_score = float(os.path.splitext(each)[0].split("__")[-1])
            cur_scores.update({each:each_score})
        y_hat=sorted(cur_scores.items(), key=(lambda x:x[1]), reverse=(False if direction=='maximize' else True))
        # y_hat order: [min(bad) ....... >> ... max(good)] when maximize
        # y_hat order: [max(bad) ....... >> ... min(good)] when minimize
        should_be_remove_num = len(cur_files) - top_n
        i = -1
        if should_be_remove_num>0:
            for i in range(0,should_be_remove_num):
                target_file_name = y_hat[i][0]
                #os.remove(target_file_name) --> remove_model_and_others(target_file_name) # 20201030
                remove_model_and_others(target_file_name)
        target_file_name = y_hat[i+1][0]
        bad_score = y_hat[i+1][1]
        ## comparing
        if direction == 'maximize':
            if score > bad_score: # for better current score, replace it
                #os.remove(target_file_name) --> remove_model_and_others(target_file_name) # 20201030
                remove_model_and_others(target_file_name)
                savemodel(algorithm_name, model, target_path+file_prefix, make_png, vs, metric, label_names)
        elif direction == 'minimize':
            if score < bad_score: # for better current score, replace it
                #os.remove(target_file_name) --> remove_model_and_others(target_file_name) # 20201030
                remove_model_and_others(target_file_name)
                savemodel(algorithm_name, model, target_path+file_prefix, make_png, vs, metric, label_names)
    else:
        savemodel(algorithm_name, model, target_path+file_prefix, make_png, vs, metric, label_names)

def remove_model_and_others(target_model_name):
    import glob,os
    for_delete = glob.glob(os.path.splitext(target_model_name)[0]+".*")
    for each in for_delete:
        os.remove(each)

def savemodel(algorithm_name, model, file_prefix, make_png, vs, metric, label_names):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.use('Agg')
    # add first
    extension = ".pth" if algorithm_name == 'DL_Pytorch' else ".pkl"
    ## make_png
    if make_png:
        if metric == 'r2':  #regression
            vsplot, ax = plt.subplots(1, 1, figsize=(12,12))
            ax.scatter(x = vs[0], y = vs[1], color='c', edgecolors=(0, 0, 0))
            ax.plot([vs[1].min(), vs[1].max()], [vs[1].min(), vs[1].max()], 'k--', lw=4)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.savefig(file_prefix+'.png', dpi=300)
            plt.close('all')
        elif metric == 'f1':#classification
            ######################## for classification
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
            cnf_matrix = confusion_matrix(vs[1], vs[0])
            np.set_printoptions(precision=2)
            ######################## for classification
            ##* Plot non-normalized confusion matrix
            plt.figure()
            if label_names is None:
                from sklearn.preprocessing import LabelEncoder
                import numpy as np
                class_le = LabelEncoder()
                class_le = class_le.fit(np.append(vs[0],vs[1]))
                label_names = class_le.classes_
            plot_confusion_matrix(cnf_matrix, classes=label_names,title='Confusion matrix, without normalization')
            plt.savefig(file_prefix+'.png', dpi=300)
            plt.close('all')
    ## model save
    #import pandas as pd
    #pd.DataFrame(np.c_[vs[0],vs[1]], columns=['Predicted','Actual']).to_csv(file_prefix+".csv")
    #
    if algorithm_name == 'DL_Pytorch':
        import torch
        torch.save(model, file_prefix+extension)
    else:
        import joblib
        extension = ".pkl"
        joblib.dump(model, file_prefix+extension) 

######################################
## get_argparse and check_stepwise_available are moved @ 20200810 by jclee
##
def get_sample_seaching_space():
    sample_dict = {
        "boosting_type":{"choices":["gbdt", "goss"]},
        "num_leaves":{"choices":[15,31,63,127,255]},
        "max_depth":{"low":-1, "high":12},
        "subsample_for_bin":{"choices":[20000, 50000, 100000, 200000]},
        "class_weight":{"choices":[None,"balanced"]},
        "min_child_weight":{"choices":[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]},
        "min_child_samples":{"low":1, "high":1000},
        "subsample":{"low":0.2, "high":1.0},
        "learning_rate":{"choices":[0.05]},
        "colsample_bytree":{"low":0.2, "high":1.0}
    }
    return sample_dict

def gen_custom_searching_space_file(task_name, algorithm_name, space, out_json_filename):
    ''' example)
    gen_custom_searching_space_file('clf','lgb',get_sample_seaching_space(), 'custom_space')
    res) Successively generated default searching space! -> custom_space.json
    '''
    results = '{\n\t"'+str(task_name)+'":{\n\t\t"'+str(algorithm_name)+'":\n'
    results+= '\t\t\t'+str(space).replace("'",'"').replace("None", "null") + "\n"
    results+= '\n\t}\n}\n'
    try:
        with open(out_json_filename+".json", 'w') as f:
            f.write(results)
        print("Successively generated default searching space! -> "+str(out_json_filename)+".json")
        token=True
    except:
        print("Cannot generate searching space!")
        token=False
    return token

def generate_default_searching_space_file(out_file_pathname=None):
    default_strings='''{
    "Regression":{
        "MLR":
        {
        "fit_intercept":{"choices":["True","False"]},
        "normalize":{"choices":["False","True"]}
        },
        "SVM":
        {
        "C":{"low":-10.0, "high":10.0, "transformation":"2**x"},
        "kernel":{"choices":[ "rbf", "linear", "poly","sigmoid"]},
        "degree":{"low":2, "high":5},
        "gamma":{"low":-10.0, "high":10.0,"transformation":"2**x"},
        "tol":{"low":-5,"high":-1, "transformation":"10**x"},
        "__comment__epsilon":"The epsilon param is only for a regression task.",
        "epsilon":{"low":0.01, "high":0.99}
        },
        "RF":
        {
        "n_estimators":{"low":1, "high":1000},
        "criterion":{"choices":["mse", "mae"]},
        "min_samples_split":{"low":0.0, "high":1.0},
        "min_samples_leaf":{"low":0.0, "high":0.5}
        },
        "BT":
        {
        "n_estimators":{"low":1, "high":2000},
        "learning_rate":{"low":1e-5, "high":1e-1},
        "loss":{"choices":["linear","square","exponential"]}
        },
        "DL_Pytorch":
        {
        "model":{"choices":["FNN"]},
        "batch_size":{"choices":[32,64,128,256]},
        "epochs":{"choices":[5, 10, 20]},
        "optimizer":{"choices":["Adam","RMSprop","SGD"]},
        "lr":{"low":1e-5,"high":1e-1},
        "momentum":{"low":0.0, "high":1.0},
        "n_layers":{"low":1, "high":5},
        "n_units":{"low":4, "high":128},
        "dropout":{"low":0.01, "high":0.2},
        "loss":{"choices":["MSELoss"]}
        },
        "XGBoost":
        {
        "__comment__":"general params: booster(type), objective",
        "eval_metric":{"choices":["rmse"]},
        "num_boost_round":{"choices":[100, 500, 1000, 2000]},
        "booster":{"choices":["gbtree","dart"]},
        "objective":{"choices":["reg:squarederror"]},
        "__comment__regularization":"lambda(def=1) regarding L2 reg. weight, alpha(def=0) regarding L1 reg. weight",
        "lambda":{"low":1e-8, "high":1.0},
        "alpha":{"low":1e-8, "high":1.0},
        "__comment__cv":"if cv, min_child_weight(def=1), max_depth(def=6) should be tuned",
        "min_child_weight":{"low":0, "high":10},
        "__comment__others":"[booster] Both gbtree and dart require max_depth, eta, gamma, and grow_policy. Only dart requires sample_type, normalize_type, rate_drop, and skip_drop.",
        "max_depth":{"low":1, "high":9},
        "eta":{"low":1e-8, "high":1.0},
        "gamma":{"low":1e-8, "high":1.0},
        "grow_policy":{"choices":["depthwise","lossguide"]},
        "sample_type":{"choices":["uniform","weighted"]},
        "normalize_type":{"choices":["tree","forest"]},
        "rate_drop":{"low":1e-8, "high":1.0},
        "skip_drop":{"low":1e-8, "high":1.0}
        },
        "LightGBM":
        {
        "objective":{"choices":["regression"]},
        "num_boost_round":{"choices":[100,500,1000, 2000]},
        "metric":{"choices":["rmse"]},
        "boosting_type":{"choices":["gbdt", "dart", "goss"]},
        "num_leaves":{"choices":[15,31,63,127,255]},
        "max_depth":{"low":-1, "high":12},
        "subsample_for_bin":{"choices":[20000, 50000, 100000, 200000]},
        "min_child_weight":{"low":-4, "high":4, "transformation":"10**x"},
        "min_child_samples":{"low":1, "high":100},
        "subsample":{"low":0.2, "high":1.0}, 
        "learning_rate":{"low":1e-5,"high":1e-1},
        "colsample_bytree":{"low":0.2, "high":1.0}
        }
    },
    "Classification":{
        "SVM":
        {
        "__comment__":"default hyperparameters of SVM are selected from q0.05 to q0.95 @Tunability (P.Probst et al., 2019)",
        "C":{"low":0.025, "high":943.704},
        "kernel":{"choices":[ "rbf", "linear", "poly","sigmoid"]},
        "degree":{"low":2, "high":4},
        "gamma":{"low":0.007, "high":276.02},
        "tol":{"low":-5,"high":-1, "transformation":"10**x"},
        "class_weight":{"choices":["None", "balanced"]}
        },
        "RF":
        {
        "__comment__":"default hyperparameters of RF are selected from q0.05 to q0.95 @Tunability (P.Probst et al., 2019)",
        "n_estimators":{"low":203, "high":1909},
        "criterion":{"choices":["gini", "entropy"]},
        "__comment__min_samples_split":"min_samples_split is sample.fraction in R(ranger)",
        "min_samples_split":{"low":0.257,"high":0.971},
        "__comment__max_feature":"max_features is mtry in R(ranger), but automatically transformed to int(max_features * n_features)",
        "max_features":{"low":0.081, "high":0.867},
        "__comment__min_samples_leaf":"mean_samples_leaf is min.node.size in R(ranger)",
        "min_samples_leaf":{"low":0.009, "high":0.453}
        },
        "BT":
        {
        "n_estimators":{"low":1, "high":2000},
        "learning_rate":{"low":1e-5, "high":1e-1},
        "algorithm":{"choices":["SAMME.R","SAMME"]}
        },
        "DL_Pytorch":
        {
        "model":{"choices":["FNN","CNN"]},
        "batch_size":{"choices":[32,64,128,256]},
        "epochs":{"choices":[5, 10, 20]},
        "optimizer":{"choices":["Adam","RMSprop","SGD"]},
        "lr":{"low":1e-5,"high":1e-1},
        "momentum":{"low":0.0, "high":1.0},
        "n_layers":{"low":1, "high":3},
        "n_units":{"low":4, "high":128},
        "dropout":{"low":0.01, "high":0.2},
        "loss":{"choices":["cross_entropy"]}
        },
        "XGBoost":
        {
        "__comment__":"general params: booster(type), objective",
        "eval_metric":{"choices":["mlogloss"]},
        "num_boost_round":{"choices":[100, 500, 1000, 2000]},
        "booster":{"choices":["gbtree","dart"]},
        "objective":{"choices":["multi:softmax"]},
        "__comment__regularization":"lambda(def=1) regarding L2 reg. weight, alpha(def=0) regarding L1 reg. weight",
        "lambda":{"low":1e-8, "high":1.0},
        "alpha":{"low":1e-8, "high":1.0},
        "__comment__cv":"if cv, min_child_weight(def=1), max_depth(def=6) should be tuned",
        "min_child_weight":{"low":0, "high":10},
        "__comment__others":"[booster] Both gbtree and dart require max_depth, eta, gamma, and grow_policy. Only dart requires sample_type, normalize_type, rate_drop, and skip_drop.",
        "max_depth":{"low":1, "high":9},
        "eta":{"low":1e-8, "high":1.0},
        "gamma":{"low":1e-8, "high":1.0},
        "grow_policy":{"choices":["depthwise","lossguide"]},
        "sample_type":{"choices":["uniform","weighted"]},
        "normalize_type":{"choices":["tree","forest"]},
        "rate_drop":{"low":1e-8, "high":1.0},
        "skip_drop":{"low":1e-8, "high":1.0}
        },
        "LightGBM":
        {
        "objective":{"choices":["multiclass"]},
        "num_boost_round":{"choices":[100,500,1000, 2000]},
        "metric":{"choices":["multi_logloss"]},
        "boosting_type":{"choices":["gbdt", "dart", "goss"]},
        "num_leaves":{"choices":[15,31,63,127,255]},
        "max_depth":{"low":-1, "high":12},
        "subsample_for_bin":{"choices":[20000, 50000, 100000, 200000]},
        "class_weight":{"choices":["None", "balanced"]},
        "min_child_weight":{"low":-4, "high":4, "transformation":"10**x"},
        "min_child_samples":{"low":1, "high":100},
        "subsample":{"low":0.2, "high":1.0}, 
        "learning_rate":{"low":1e-5,"high":1e-1},
        "colsample_bytree":{"low":0.2, "high":1.0}
        }
    }
}
'''
    try:
        with open(out_file_pathname, 'w') as f:
            f.write(default_strings)
        os.chmod(out_file_pathname, 0o776) # add permission 201030            
        print("Successively generated default searching space! -> searching_space_automl.json")
    except:
        print("Cannot generate default searching space!")
    return default_strings
def get_argparse(automl=False, json_file_name=None):
    import argparse, json
    parser = argparse.ArgumentParser()
    # study setups
    parser.add_argument('--user_name', help="username", type=str, default = '')
    parser.add_argument('--study_name', help="name of study", type=str, default = '')
    parser.add_argument('--db_ip', help="db ip address", type=str, default = '150.183.247.244')
    parser.add_argument('--db_port', help="db port", type=str, default = '5432')
    parser.add_argument('--db_id', help="db id", type=str, default = 'postgres')
    parser.add_argument('--db_pass', help="db pass", type=str, default = 'postgres')
    parser.add_argument('--direction', help="study direction", type=str, default = 'maximize')
    parser.add_argument('--max_trials', help="maximum trials", type=int, default = 100000)
    parser.add_argument('--max_sec', help="maximul seconds(time)", type=int, default = 300)
    parser.add_argument('--ss_json', help='searching space file location', default ="searching_space_automl.json")
    parser.add_argument('--job_path', help='job path', type=str, default='./')
    # default params
    parser.add_argument("--seed", type=int, default=2020)
    ## error handling when jupyter call
    args = parser.parse_args()
    if json_file_name:
        automl=True
    if automl: # study_name, time_deadline_sec, ss_json_path, ss_json_name should be controlled by gui_params json.
        with open(json_file_name) as data_file:
            gui_params = json.load(data_file)
        if 'hpo_system_attr' in gui_params:
            #if ['study_name', 'time_deadline_sec', 'n_cpu', 'n_gpu'] == [each for each in gui_params['hpo_system_attr']]:
            if 'job_path' in gui_params['hpo_system_attr']:
                args.job_path = gui_params['hpo_system_attr']['job_path']
            if 'user_name' in gui_params['hpo_system_attr']:
                args.user_name = gui_params['hpo_system_attr']['user_name']
            if 'study_name' in gui_params['hpo_system_attr']:
                args.study_name = gui_params['hpo_system_attr']['study_name']
            if 'time_deadline_sec' in gui_params['hpo_system_attr']:
                args.max_sec = gui_params['hpo_system_attr']['time_deadline_sec']
            #if ('ss_json_path' in gui_params['hpo_system_attr']) and ('ss_json_name' in gui_params['hpo_system_attr']):
            #    searching_space_json = gui_params['hpo_system_attr']['ss_json_path'] + gui_params['hpo_system_attr']['ss_json_name']
            if 'searching_space' in gui_params['hpo_system_attr']:
                args.ss_json = gui_params['hpo_system_attr']['searching_space'] 
            if 'direction' in gui_params['hpo_system_attr']:
                args.direction=gui_params['hpo_system_attr']['direction']
            if 'seed' in gui_params['hpo_system_attr']:
                args.seed=gui_params['hpo_system_attr']['seed']
                # ncpu gpu will not be controlled in the python script
        if not os.path.exists(args.ss_json):
            generate_default_searching_space_file(args.job_path+os.sep+args.ss_json)
    return args

        
#def get_argparse(automl=False, json_file_name=None):
#    import argparse, json
#    parser = argparse.ArgumentParser()
#    # study setups
#    parser.add_argument('--user_name', help="username", type=str, default = '')
#    parser.add_argument('--study_name', help="name of study", type=str, default = '')
#    parser.add_argument('--db_ip', help="db ip address", type=str, default = '150.183.247.244')
#    parser.add_argument('--db_port', help="db port", type=str, default = '5432')
#    parser.add_argument('--db_id', help="db id", type=str, default = 'postgres')
#    parser.add_argument('--db_pass', help="db pass", type=str, default = 'postgres')
#    parser.add_argument('--direction', help="study direction", type=str, default = 'maximize')
#    parser.add_argument('--max_trials', help="maximum trials", type=int, default = 100000)
#    parser.add_argument('--max_sec', help="maximul seconds(time)", type=int, default = 300)
#    parser.add_argument('--ss_json', help='searching space file location', default ="searching_space_automl.json")
#    # default params
#    parser.add_argument("--seed", type=int, default=2020)
#    ## error handling when jupyter call
#    in_jupyter=False
#    try:
#        args = parser.parse_args()
#    except:
#        import easydict
#        args = easydict.EasyDict({
#            #"user_name":"",
#            "ss_json":"searching_space_automl.json",
#            "nb_name":"",
#            "study_name":"",
#            "job_directory":"",
#            "metadata_json":"",
#            "task_name":"",
#            "algorithm_name":"",
#            "env_name":"",
#            "task_type":"gpu",
#            "n_nodes":1,
#            "max_sec":300,
#            "seed":2020,
#            "direction":"maximize",
#            "greedy":False,
#            "stepwise":False,
#            "top_n_all":10,
#            "top_n_each_algo":3
#            #"db_ip":'150.183.247.244',
#            #"db_port":'5432',
#            #"db_id":"postgres",
#            #"db_pass":"postgres",
#            #"max_trials":100000
#        })
#        print("*Note that algorithm_name can be multiple. If multiple, write down the algorithms using comma. e.g., args.algorithm_name = 'lgb, xgboost, pytorch'")
#        in_jupyter=True
#        if json_file_name:
#            automl=True
#    if not os.path.exists(args.ss_json):
#        if not in_jupyter:
#            if json_file_name:
#                with open(json_file_name) as data_file:
#                    gui_params = json.load(data_file)
#                #filepath=gui_params['ml_file_path']
#                jobpath, (uname, sname, job_title, wsname, job_directory) = get_jobpath_with_attr(gui_params)
#                filename=os.path.basename(args.ss_json)
#                location = jobpath+os.sep+filename
#            else:
#                location=args.ss_json
#            generate_default_searching_space_file(location)
#    if automl: # study_name, time_deadline_sec, ss_json_path, ss_json_name should be controlled by gui_params json.
#        with open(json_file_name) as data_file:
#            gui_params = json.load(data_file)
#        if 'hpo_system_attr' in gui_params:
#            #if ['study_name', 'time_deadline_sec', 'n_cpu', 'n_gpu'] == [each for each in gui_params['hpo_system_attr']]:
#            if 'user_name' in gui_params['hpo_system_attr']:
#                args.user_name = gui_params['hpo_system_attr']['user_name']
#            if 'study_name' in gui_params['hpo_system_attr']:
#                args.study_name = gui_params['hpo_system_attr']['study_name']
#            if 'time_deadline_sec' in gui_params['hpo_system_attr']:
#                args.max_sec = gui_params['hpo_system_attr']['time_deadline_sec']
#            #if ('ss_json_path' in gui_params['hpo_system_attr']) and ('ss_json_name' in gui_params['hpo_system_attr']):
#            #    searching_space_json = gui_params['hpo_system_attr']['ss_json_path'] + gui_params['hpo_system_attr']['ss_json_name']
#            if 'searching_space' in gui_params['hpo_system_attr']:
#                args.ss_json = gui_params['hpo_system_attr']['searching_space'] 
#            if 'direction' in gui_params['hpo_system_attr']:
#                args.direction=gui_params['hpo_system_attr']['direction']
#                #print(args)
#                # ncpu gpu will not be controlled in the python script
#    return args
    
def check_stepwise_available(json_file_name):
    import json
    with open(json_file_name, "r") as __:
        gui_params = json.load(__)
    if 'algorithm' in gui_params:
        if len(gui_params['algorithm']) == 1:
            if 'hpo_system_attr' in gui_params:
                if 'stepwise' in gui_params['hpo_system_attr']:
                    if gui_params['hpo_system_attr']['stepwise'] == 1:
                        return True, (gui_params['task'], gui_params['algorithm'][0])
    return False, (None)
######################################
def find_categories(params):
    return ["params_"+key for key, values in params.items() if type(values) is dict if 'choices' in values]

def find_linear(params): #updated 20200707 - remove cv
    return [key for key, values in params.items() if type(values) is dict if not 'choices' in values if key != 'cv']

def add_num(n):
    if n==2:
        return 1
    return 2**(n-2) + add_num(n-1)

def do_score_rf_modeling(study, given_params):
    df=study.trials_dataframe().copy(deep=True)
    df=df[df['state']=='COMPLETE']
    # onehot encoding
    import pandas as pd
    categories_feature_names=find_categories(given_params)
    for each in categories_feature_names:
        df[each] = pd.Categorical(df[each])
        dfDummies = pd.get_dummies(df[each], prefix=each)
        df = pd.concat([df,dfDummies],axis=1)
        df = df.drop([each], axis=1)
    features = [item for item in df.columns if item[0:6]=='params']
    target='value'
    import numpy as np
    df=df.replace(np.nan,0)
    X=df[features].values
    y=df[target].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    from sklearn.ensemble import RandomForestRegressor
    regr=RandomForestRegressor()#)n_estimators=100)
    #regr.fit(X,y)
    regr.fit(X_train,y_train)
    vi = pd.DataFrame([regr.feature_importances_], columns=features)
    vit=vi.transpose().sort_values(by=[0], ascending=False)
    score=regr.score(X_test,y_test)
    #print("score r2: ", score)
    return vit.index.to_list(), score

def do_score_rf_modeling_ver2(study, given_params):
    df=study.trials_dataframe().copy(deep=True)
    df2=study.trials_dataframe().copy(deep=True)
    df=df[df['state']=='COMPLETE']
    # onehot encoding
    import pandas as pd
    categories_feature_names=find_categories(given_params)
    for each in categories_feature_names:
        df[each] = pd.Categorical(df[each])
        dfDummies = pd.get_dummies(df[each], prefix=each)
        df = pd.concat([df,dfDummies],axis=1)
        df = df.drop([each], axis=1)
    features = [item for item in df.columns if item[0:6]=='params']
    target='value'
    import numpy as np
    df=df.replace(np.nan,0)
    X=df[features].values
    y=df[target].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    from sklearn.ensemble import RandomForestRegressor
    regr=RandomForestRegressor()#)n_estimators=100)
    #regr.fit(X,y)
    regr.fit(X_train,y_train)
    vi = pd.DataFrame([regr.feature_importances_], columns=features)
    vit=vi.transpose().sort_values(by=[0], ascending=False)
    score=regr.score(X_test,y_test)
    #print("score r2: ", score)
#    df2=df2[df2['state']=='PRUNED']
    return vit.index.to_list(), score

def do_score_rf_modeling_ver3(df, given_params):
    #df=study.trials_dataframe().copy(deep=True)
    #df2=study.trials_dataframe().copy(deep=True)
    #df=df[df['state']=='COMPLETE']
    # onehot encoding
    import pandas as pd
    categories_feature_names=find_categories(given_params)
    for each in categories_feature_names:
        df[each] = pd.Categorical(df[each])
        dfDummies = pd.get_dummies(df[each], prefix=each)
        df = pd.concat([df,dfDummies],axis=1)
        df = df.drop([each], axis=1)
    features = [item for item in df.columns if item[0:6]=='params']
    target='value'
    import numpy as np
    df=df.replace(np.nan,0)
    X=df[features].values
    y=df[target].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    from sklearn.ensemble import RandomForestRegressor
    regr=RandomForestRegressor()#)n_estimators=100)
    #regr.fit(X,y)
    regr.fit(X_train,y_train)
    vi = pd.DataFrame([regr.feature_importances_], columns=features)
    vit=vi.transpose().sort_values(by=[0], ascending=False)
    score=regr.score(X_test,y_test)
    #print("score r2: ", score)
#    df2=df2[df2['state']=='PRUNED']
    return vit.index.to_list(), score

def reduction_param_space(prev_params,adjust_target_param, sorted_df, already_reduced_params_list):
    out_params=copy.deepcopy(prev_params)
    for key in out_params.keys():
        if key in adjust_target_param:
            #print(key)
            if 'choices' in out_params[key]: # in case of categories
                ########################################################
                #ValueError('CategoricalDistribution does not support dynamic value space.')
                ########################################################
                #reduction_param_space_when_categories()
                #print("-----------------------------------------")
                #print(key, out_params[key])
                #out_params[key]['choices'] = sorted_df["params_"+key].unique().tolist()
                #print('are changed into->\n', key, out_params[key])
                #print("-----------------------------------------")
                return out_params, already_reduced_params_list
            else:
                #reduction_param_space_when_numbers()
                if key in already_reduced_params_list:
                    return out_params, already_reduced_params_list
                else:
                    print("*****************************************")
                    print(key, ': (low) ',out_params[key]['low'], ' (high) ', out_params[key]['high'])
                    out_params[key]['low'] = sorted_df[adjust_target_param].min()
                    out_params[key]['high'] = sorted_df[adjust_target_param].max()
                    already_reduced_params_list.append(key)
                    print('are changed into->\n',key, ': (low) ',out_params[key]['low'], ' (high) ', out_params[key]['high'])
                    print("*****************************************")
                    return out_params, already_reduced_params_list
#20200424 added
def n_iter_calculation_old(n_total_trials, n_search_dim, current_step, strategy="exp", warning=True):
    if current_step > n_search_dim:
        print("(error) CURRENT STEP cannot exceed N_SEARCH_DIM. Try again.") # should be log
        return -1
    if strategy == 'equal':
        seed = max([1,n_total_trials // n_search_dim])
        if current_step == n_search_dim:
            return #n_total_trials - (current_step-1)*seed
        else:
            return seed
    elif strategy == 'exp':
        seed = max([1,n_total_trials//add_num(n_search_dim+1)])
        if add_num(n_search_dim)*seed > n_total_trials:
            if warning:
                print("(warning) N_TOTAL_TRIALS is too small. The 'equal' method will be running instead of the 'exp' method.")
            return n_iter_calculation(n_total_trials, n_search_dim, current_step, "equal")
        else:
            if current_step == n_search_dim:
                t=0
                for j in range(1, n_search_dim+1):
                    t=t+2**(j-1)*seed
                margin = n_total_trials - t
                return 2**(current_step-1)*seed + margin
            else:
                return 2**(current_step-1)*seed

#20200604 modified -> equal 메카니즘 half and the rest 로 변경
def n_iter_calculation(n_total_trials, n_search_dim, current_step, strategy="exp", warning=True):
    '''>>> for i in range(1,6):
    ...     print(n_iter_calculation(3600,5,i,strategy='equal'), n_iter_calculation(3600,5,i,strategy='exp'))
    ...
    360 116
    720 232
    1080 464
    1440 928
    1800 1860
    >>>  '''
    if current_step > n_search_dim:
        print("(error) CURRENT STEP cannot exceed N_SEARCH_DIM. Try again.") # should be log
        return -1
    if strategy == 'equal':
        half = int(n_total_trials/2)
        seed = max([1,half // n_search_dim])
        the_rest = n_total_trials - seed*n_search_dim #n_total_trials - half
        if current_step == n_search_dim:
            return the_rest#n_total_trials - (current_step-1)*seed
        else:
            return seed*current_step
    elif strategy == 'exp':
        seed = max([1,n_total_trials//add_num(n_search_dim+1)])
        if add_num(n_search_dim)*seed > n_total_trials:
            if warning:
                print("(warning) N_TOTAL_TRIALS is too small. The 'equal' method will be running instead of the 'exp' method.")
            return n_iter_calculation(n_total_trials, n_search_dim, current_step, "equal")
        else:
            if current_step == n_search_dim:
                t=0
                for j in range(1, n_search_dim+1):
                    t=t+2**(j-1)*seed
                margin = n_total_trials - t
                return 2**(current_step-1)*seed + margin
            else:
                return 2**(current_step-1)*seed

def n_iter_calculation_with_rung(n_total_trials, n_search_dim, current_step, strategy="exp", rung=2, warning=True):
    if current_step > n_search_dim:
        print("(error) CURRENT STEP cannot exceed N_SEARCH_DIM. Try again.") # should be log
        return -1
    if strategy == 'equal':
        half = int(n_total_trials/rung)
        seed = max([1,half // n_search_dim])
        the_rest = n_total_trials - seed*n_search_dim #n_total_trials - half
        if current_step == n_search_dim:
            return the_rest#n_total_trials - (current_step-1)*seed
        else:
            return seed
    elif strategy == 'exp':
        seed = max([1,n_total_trials//add_num(n_search_dim+1)])
        if add_num(n_search_dim)*seed > n_total_trials:
            if warning:
                print("(warning) N_TOTAL_TRIALS is too small. The 'equal' method will be running instead of the 'exp' method.")
            return n_iter_calculation(n_total_trials, n_search_dim, current_step, "equal")
        else:
            if current_step == n_search_dim:
                t=0
                for j in range(1, n_search_dim+1):
                    t=t+rung**(j-1)*seed
                margin = n_total_trials - t
                return rung**(current_step-1)*seed + margin
            else:
                return rung**(current_step-1)*seed

def uncertainty_reduction_ver7(study, params, n_trials, default_params=None):
    # stepwise expansion - random search - fine tuning stage 로 상세 설정
    out_params=copy.deepcopy(params)
    randomfix_paramlist = find_linear(params)
    ###########################################
    # set all linear params to default_params (in out_params)
    if default_params: # given the default params
        for each_param in randomfix_paramlist:
            out_params[each_param]['low'] = default_params[each_param]
            out_params[each_param]['high'] = default_params[each_param]
    else: # if not use default params
        # try 1 optimize - rank 0 should run this
        ##################
        mod_objective = ModObjectiveFunctionWrapper(objective, params)        
        study.optimize(mod_objective, 1)
        ##################
        n_trials = n_trials - 1
        # find the default values
        for each_param in randomfix_paramlist:
            # search min/max values among the trial, e.g., n_units_l0 50,n_units_l1 115,n_units_l2 70
            values = []
            for k, v in study.best_params.items():
                if each_param in k:
                    values.append(v)
            default_value = sorted(values)[int(len(values)/2)]
            out_params[each_param]['low'] = default_value
            out_params[each_param]['high'] = default_value
    ###########################################
    # iteration calculation (ver7 modified)
    fine_tune_trials = n_trials // 10
    n_trials_90 = n_trials - fine_tune_trials # n_trials = 90%
    #
    ###########################################
    # step-wise search (random sequence)
    random.shuffle(randomfix_paramlist) # if we do not have ordered (a prior) params regarding importance of each param
    current_step = 1
    tuned = []
    for each_param in randomfix_paramlist:
        tuned.append(each_param)
        if len(tuned) == len(randomfix_paramlist):
            print("* [FULL-RANGE SEARCH] with CMA-ES sampler and SHA pruner *")
            study.sampler = sampler
            study.pruner = pruner
        print("* TUNING TARGET PARAM: [ ", tuned, " ]")
        out_params[each_param]['low'] = params[each_param]['low']
        out_params[each_param]['high'] = params[each_param]['high']
        #print("run: (modified)_", each_param, out_params, "*TRIALS", each_n_trials, the_rest_n_trials)
        ##################
        mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
        study.optimize(mod_objective, n_iter_calculation(n_trials_90, len(randomfix_paramlist), current_step))
        ##################
        current_step += 1
        ##
        #################################
        # half reduction on each param
        # 여기서 죄다 pruning 되었으면 min max 가 안나옴.. (동일값이 나와버림)
        #################################
        #df = study.trials_dataframe()
        #df = df[df['state']=='COMPLETE']
        #ascending_token = True if study.direction == StudyDirection.MAXIMIZE else False
        #sorted_half_df=df.sort_values(by=['value'], ascending=ascending_token).head(len(df)//2)
        #low_values = []
        #high_values = []
        #for col_name in sorted_half_df.columns.tolist():
        #    if each_param in col_name: #### using each_param in randomfix_paramlist
        #        low_values.append(sorted_half_df[col_name].min())
        #        high_values.append(sorted_half_df[col_name].max())
        #out_params[each_param]['low'] = min(low_values)
        #out_params[each_param]['high'] = max(high_values)
        #print("** PARAM: [ ", each_param, " ] range has been reduced as (low): ", min(low_values), "and (high): ",max(high_values), "after tuning.")
    #######################################
    # the rest study - finetune 10% and the rest
    ##################
    #df = study.trials_dataframe()
    #df = df[df['state']=='COMPLETE']
    #ascending_token = True if study.direction == StudyDirection.MAXIMIZE else False
    #sorted_half_df=df.sort_values(by=['value'], ascending=ascending_token).head(len(df)//2)
    #low_values = []
    #high_values = []
    #for col_name in sorted_half_df.columns.tolist():
    #    if each_param in col_name: #### using each_param in randomfix_paramlist
    #        low_values.append(sorted_half_df[col_name].min())
    #        high_values.append(sorted_half_df[col_name].max())
    #out_params[each_param]['low'] = min(low_values)
    #out_params[each_param]['high'] = max(high_values)
    #print("** PARAM: [ ", each_param, " ] range has been reduced as (low): ", min(low_values), "and (high): ",max(high_values), "after tuning.")
    ####################################
    mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
    study.optimize(mod_objective, fine_tune_trials)
    ##################
    return study

def uncertainty_reduction_ver8(study, objective, params, n_trials, sampler, default_params=None):
    # stepwise expansion - random search - fine tuning stage 로 상세 설정
    out_params=copy.deepcopy(params)
    randomfix_paramlist = find_linear(params)
    ###########################################
    # set all linear params to default_params (in out_params)
    if default_params: # given the default params
        for each_param in randomfix_paramlist:
            out_params[each_param]['low'] = default_params[each_param]
            out_params[each_param]['high'] = default_params[each_param]
    else: # if not use default params
        # try 1 optimize - rank 0 should run this
        ##################
        mod_objective = ModObjectiveFunctionWrapper(objective, params)        
        study.optimize(mod_objective, 1)
        ##################
        n_trials = n_trials - 1
        # find the default values
        for each_param in randomfix_paramlist: # remove iterative finding in a specific param
            for k, v in study.best_params.items():
                if each_param == k:
                    default_value = v
            out_params[each_param]['low'] = default_value
            out_params[each_param]['high'] = default_value
    ###########################################
    # iteration calculation (ver7 modified)
    fine_tune_trials = n_trials // 5
    n_trials_80 = n_trials - fine_tune_trials # n_trials = 80%
    #
    ###########################################
    # step-wise search (random sequence)
    random.shuffle(randomfix_paramlist) # if we do not have ordered (a prior) params regarding importance of each param
    current_step = 1
    tuned = []
    #others=copy.deepcopy(randomfix_paramlist)
    for each_param in randomfix_paramlist:
        tuned.append(each_param)
        #print("***", tuned, others)
        #others.remove(each_param)
        if len(tuned) == len(randomfix_paramlist):
            print("* [FULL-RANGE SEARCH] *")
            #study.sampler = sampler
            #study.pruner = pruner
        #print("* TUNING TARGET PARAM: [ ", tuned, " ]", "* STUCKED PARAMS: [ ", others, " ]")
        print("* TUNING TARGET PARAM: [ ", tuned, " ]", "* CONTROLLED PARAMS: [ ", randomfix_paramlist, " ]")
        out_params[each_param]['low'] = params[each_param]['low']
        out_params[each_param]['high'] = params[each_param]['high']
        #print("run: (modified)_", each_param, out_params, "*TRIALS", each_n_trials, the_rest_n_trials)
        ##################
        mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
        # 20200428 when modified objective -> sampler object should run again
        study.sampler = sampler
        study.optimize(mod_objective, n_iter_calculation(n_trials_80, len(randomfix_paramlist), current_step))
        ##################
        current_step += 1
    mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
    study.sampler = optuna.integration.CmaEsSampler(seed = 1991)
    study.optimize(mod_objective, fine_tune_trials)
    ##################
    return study

def stepwise_simple(study, objective, params, n_trials, default_params=None):
    out_params=copy.deepcopy(params)
    randomfix_paramlist = find_linear(params)
    ###########################################
    # set all linear params to default_params (in out_params)
    if default_params: # given the default params
        for each_param in randomfix_paramlist:
            out_params[each_param]['low'] = default_params[each_param]
            out_params[each_param]['high'] = default_params[each_param]
    else: # if not use default params
        # try 1 optimize - rank 0 should run this
        ##################
        mod_objective = ModObjectiveFunctionWrapper(objective, params)        
        study.optimize(mod_objective, 1)
        ##################
        n_trials = n_trials - 1
        # find the default values
        for each_param in randomfix_paramlist: # remove iterative finding in a specific param
            for k, v in study.best_params.items():
                if each_param == k:
                    default_value = v
            out_params[each_param]['low'] = default_value
            out_params[each_param]['high'] = default_value
    ###########################################
    # step-wise search (random sequence)
    random.shuffle(randomfix_paramlist) # if we do not have ordered (a prior) params regarding importance of each param
    current_step = 1
    tuned = []
    for each_param in randomfix_paramlist:
        tuned.append(each_param)
        if len(tuned) == len(randomfix_paramlist):
            print("* [FULL-RANGE SEARCH] *")
        print("* TUNING TARGET PARAM: [ ", tuned, " ]", "* CONTROLLED PARAMS: [ ", randomfix_paramlist, " ]")
        out_params[each_param]['low'] = params[each_param]['low']
        out_params[each_param]['high'] = params[each_param]['high']
        mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
        study.optimize(mod_objective, n_iter_calculation(n_trials, len(randomfix_paramlist), current_step))
        ##################
        current_step += 1
    return study

def params_sorting_by_guided_list(base_list, target_list):
    ''' 20200521 guided stepwise by Jeongcheol Lee
    1. base_list에 있는 항목들만 추리기@target_list
    2. base_list 기준으로 정렬
    3. 1에서 추려진 없는 항목들 뒤에 붙이기
    '''
    base_subset = [each for each in base_list if each in target_list]
    target_subset = [each for each in target_list if each in base_list]
    target_others = [each for each in target_list if each not in base_list]
    return sorted(target_subset, key=lambda x: base_subset.index(x)) + target_others

def stepwise_get_current_step(study, n_inner_loop, arg):
    # initial step 인지 확인하고
    # initial step 이 아니라면 현재 몇번째 trial인지 알아야함
    # arg.method 가 time일 경우 추가 구현 필요함 20200526
    if arg.method == 'max_trials':
        cur_n_trial = len(study.trials)
        if cur_n_trial == 0:
            return 1# means "initial"
        #max_step = arg.max_trials
        else:
            trials_index = 1
            for each_step in range(1,n_inner_loop+1):
                from_index = trials_index
                trials_index += n_iter_calculation(arg.max_trials - 1,n_inner_loop,each_step, warning=False)
                to_index = trials_index
                if (from_index <= cur_n_trial) and (cur_n_trial < to_index):
                    return each_step+1 # 2, 3, 4, 5...
    elif arg.method == 'max_sec':
        cur_n_trial = len(study.trials)
        if cur_n_trial == 0:
            return 1# means "initial"
        #max_step = arg.max_trials
        else:
            trials_index = 1
            for each_step in range(1,n_inner_loop+1):
                from_index = trials_index
                trials_index += n_iter_calculation(arg.max_sec - 1,n_inner_loop,each_step, warning=False)
                to_index = trials_index
                if (from_index <= cur_n_trial) and (cur_n_trial < to_index):
                    return each_step+1 # 2, 3, 4, 5...

#def stepwise_get_current_step_by_time(max_sec, elapsed_sec, n_inner_loop):
#    for i in range(1,n_inner_loop+1):
#        bound_time = n_iter_calculation(n_total_trials= max_sec, n_search_dim=n_inner_loop, current_step = i, warning=False)
#        if elapsed_sec < bound_time:
#            return i
#    return n_inner_loop+1            


def stepwise_get_current_step_by_time(max_sec, elapsed_sec, n_inner_loop):
    ''' modified by jclee @ 20201117 // (as-is) 0~25%time: exploration / 25%~100%time: exploitation
                                        (to-be) 0~50%time: exploration / 50%~100%time: exploitation'''
    rval = 1
    for i in range(1,n_inner_loop+1):
        #bound_time = n_iter_calculation(n_total_trials= max_sec, n_search_dim=n_inner_loop, strategy="equal",current_step = i,warning=False)
        bound_time = n_iter_calculation(n_total_trials= max_sec, n_search_dim=n_inner_loop, current_step = i,warning=False)
        #print(bound_time, i)
        if bound_time < elapsed_sec:
            print(rval, i)
            rval = i
    rval = min(rval, n_inner_loop)
    return rval


def stepwise_get_current_step_test(cur_n_trial, n_inner_loop, max_trials):
    # initial step 인지 확인하고
    # initial step 이 아니라면 현재 몇번째 trial인지 알아야함
    if True:
        if cur_n_trial == 0:
            return 1# means "initial"
        #max_step = arg.max_trials
        else:
            trials_index = 1
            for each_step in range(1,n_inner_loop+1):
                from_index = trials_index
                trials_index += n_iter_calculation(max_trials - 1,n_inner_loop,each_step)
                to_index = trials_index
                if (from_index <= cur_n_trial) and (cur_n_trial < to_index):
                    return each_step+1 # 2, 3, 4, 5...
        

def stepwise_guided_mpi_202005(study, objective, original_stepwise_params, controlled_stepwise_params, n_trials, guided_order_list, current_step,default_params=None):
    out_params=copy.deepcopy(controlled_stepwise_params)
    #print("BEFORE,,",out_params)
    randomfix_paramlist = find_linear(controlled_stepwise_params)
    ###########################################
    # set all linear params to default_params (in out_params)
    if current_step == 1:
        if default_params: # given the default params
            for each_param in randomfix_paramlist:
                out_params[each_param]['low'] = default_params[each_param]
                out_params[each_param]['high'] = default_params[each_param]
            mod_objective = ModObjectiveFunctionWrapper(objective, out_params)        
            study.optimize(mod_objective, 1)
            ##################
            n_trials = n_trials - 1
        else: # if not use default params
            # try 1 optimize - rank 0 should run this
            ##################
            mod_objective = ModObjectiveFunctionWrapper(objective, original_stepwise_params)        
            study.optimize(mod_objective, 1)
            ##################
            n_trials = n_trials - 1
            # find the default values
            for each_param in randomfix_paramlist: # remove iterative finding in a specific param
                default_value_list = []
                for k, v in study.best_params.items():
                    if k.startswith(each_param):
                        default_value_list.append(v)
                #print(default_value_list)
                random.shuffle(default_value_list)
                default_value = default_value_list.pop()
                out_params[each_param]['low'] = default_value # 여기 버그를 고쳤음. 다른곳도 적용 필요 20200526
                out_params[each_param]['high'] = default_value
        return study, out_params
    else: # current_step is bigger than 1
        ###########################################
        # step-wise search (random sequence)
        #random.shuffle(randomfix_paramlist) # if we do not have ordered (a prior) params regarding importance of each param
        randomfix_paramlist = params_sorting_by_guided_list(base_list = guided_order_list, target_list = randomfix_paramlist)
        current_target_param_name = randomfix_paramlist[current_step - 2]
        tuning_target_params_list = randomfix_paramlist[ : current_step - 1]
        print("* CURRENT STEP: ", current_step)#, current_target_param_name, tuning_target_params_list)
        if len(tuning_target_params_list) == len(randomfix_paramlist):
            print("* [FULL-RANGE SEARCH] *")
        print("* TUNING TARGET PARAM: [ ", tuning_target_params_list, " ]", "* CONTROLLED PARAMS: [ ", randomfix_paramlist, " ]")
        for each_param in tuning_target_params_list:
            out_params[each_param]['low'] = original_stepwise_params[each_param]['low']
            out_params[each_param]['high'] = original_stepwise_params[each_param]['high']
        #print(original_stepwise_params, out_params)
        mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
        study.optimize(mod_objective, 1)
        return study, out_params

# dataset 배분(early epoch)은 아직 적용안된 버전
def stepwise_guided_mpi_202006(study, objective, original_stepwise_params, controlled_stepwise_params, n_trials, guided_order_list, current_step,default_params=None):
    out_params=copy.deepcopy(controlled_stepwise_params)
    #print("BEFORE,,",out_params)
    randomfix_paramlist = find_linear(controlled_stepwise_params)
    ###########################################
    # set all linear params to default_params (in out_params)
    if current_step == 1:
        comm = MPI.COMM_WORLD   # get MPI communicator object
        size = comm.size        # total number of processes
        rank = comm.rank        # rank of this process
        if default_params: # given the default params
            if rank == 1:
                for each_param in randomfix_paramlist:
                    out_params[each_param]['low'] = default_params[each_param]
                    out_params[each_param]['high'] = default_params[each_param]
                mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
                study.optimize(mod_objective, 1)
                ##################
                n_trials = n_trials - 1
            else: # rank > 1
                mod_objective = ModObjectiveFunctionWrapper(objective, original_stepwise_params)        
                study.optimize(mod_objective, 1)
                ##################
                n_trials = n_trials - 1                 
        else: # if not use default params
            # try 1 optimize - rank 0 should run this
            ##################
            mod_objective = ModObjectiveFunctionWrapper(objective, original_stepwise_params)        
            study.optimize(mod_objective, 1)
            ##################
            n_trials = n_trials - 1
            #print("why it shuould run ************************************")
            # bugfix 0604
            if study.trials_dataframe()['value'].any():
                # find the default values if result exists
                for each_param in randomfix_paramlist: # remove iterative finding in a specific param
                    default_value_list = []
                    for k, v in study.best_params.items():
                        if k.startswith(each_param):
                            default_value_list.append(v)
                    #print(default_value_list)
                    random.shuffle(default_value_list)
                    default_value = default_value_list.pop()
                    out_params[each_param]['low'] = default_value # 여기 버그를 고쳤음. 다른곳도 적용 필요 20200526
                    out_params[each_param]['high'] = default_value
        return study, out_params
    else: # current_step is bigger than 1
        ###########################################
        # step-wise search (random sequence)
        #random.shuffle(randomfix_paramlist) # if we do not have ordered (a prior) params regarding importance of each param
        randomfix_paramlist = params_sorting_by_guided_list(base_list = guided_order_list, target_list = randomfix_paramlist)
        current_target_param_name = randomfix_paramlist[current_step - 2]
        tuning_target_params_list = randomfix_paramlist[ : current_step - 1]
        print("* CURRENT STEP: ", current_step)#, current_target_param_name, tuning_target_params_list)
        if len(tuning_target_params_list) == len(randomfix_paramlist):
            print("* [FULL-RANGE SEARCH] *")
        print("* TUNING TARGET PARAM: [ ", tuning_target_params_list, " ]", "* CONTROLLED PARAMS: [ ", randomfix_paramlist, " ]")
        for each_param in tuning_target_params_list:
            out_params[each_param]['low'] = original_stepwise_params[each_param]['low']
            out_params[each_param]['high'] = original_stepwise_params[each_param]['high']
        #print(original_stepwise_params, out_params)
        mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
        study.optimize(mod_objective, 1)
        return study, out_params

def stepwise_guided_mpi_by_time(study, objective, original_stepwise_params, controlled_stepwise_params, n_trials, guided_order_list, current_step,algorithm_name=None, default_params=None):
    controlled_stepwise_params_has_been_changed=False
    out_params=copy.deepcopy(controlled_stepwise_params)
    randomfix_paramlist = find_linear(controlled_stepwise_params)
    ###########################################
    # set all linear params to default_params (in out_params)
    if study.trials == []:
        comm = MPI.COMM_WORLD   # get MPI communicator object
        size = comm.size        # total number of processes
        rank = comm.rank        # rank of this process
        if default_params: # given the default params
            if rank == 0:
                for each_param in randomfix_paramlist:
                    out_params[each_param]['low'] = default_params[each_param]
                    out_params[each_param]['high'] = default_params[each_param]
                mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
                study.optimize(mod_objective, 1)
                ##################
                n_trials = n_trials - 1
            else: # rank > 1
                mod_objective = ModObjectiveFunctionWrapper(objective, original_stepwise_params)        
                study.optimize(mod_objective, 1)
                ##################
                n_trials = n_trials - 1                 
        else: # if not use default params
            # try 1 optimize - rank 0 should run this
            ##################
            mod_objective = ModObjectiveFunctionWrapper(objective, original_stepwise_params)        
            study.optimize(mod_objective, 1)
            ##################
            n_trials = n_trials - 1
            #print("why it shuould run ************************************")
            # bugfix 0604
            if study.trials_dataframe()['value'].any():
                # find the default values if result exists
                for each_param in randomfix_paramlist: # remove iterative finding in a specific param
                    if algorithm_name:
                        search_param = algorithm_name+"_"+each_param # add algorithm prefix by 20200708
                    else:
                        search_param = each_param
                    default_value_list = []
                    print(study.best_params, "*******************************")
                    for k, v in study.best_params.items():
                        if k.startswith(search_param):
                            default_value_list.append(v)
                    #print(default_value_list)
                    random.shuffle(default_value_list)
                    default_value = default_value_list.pop()
                    out_params[each_param]['low'] = default_value # 여기 버그를 고쳤음. 다른곳도 적용 필요 20200526
                    out_params[each_param]['high'] = default_value
        # if out_params has been changed,
        if out_params!=controlled_stepwise_params:
            controlled_stepwise_params_has_been_changed = True
        return study, out_params, controlled_stepwise_params_has_been_changed
    else:
        ###########################################
        # step-wise search (random sequence)
        #random.shuffle(randomfix_paramlist) # if we do not have ordered (a prior) params regarding importance of each param
        randomfix_paramlist = params_sorting_by_guided_list(base_list = guided_order_list, target_list = randomfix_paramlist)
        #print("randomfix_paramlist========================================================",randomfix_paramlist)
        #current_target_param_name = randomfix_paramlist[current_step - 1]
        tuning_target_params_list = randomfix_paramlist[:current_step]
        print("* CURRENT STEP: ", current_step)#, current_target_param_name, tuning_target_params_list)
        if len(tuning_target_params_list) == len(randomfix_paramlist):
            print("* [FULL-RANGE SEARCH] *")
        print("* TUNING TARGET PARAM: [ ", tuning_target_params_list, " ]", "* CONTROLLED PARAMS: [ ", [x for x in randomfix_paramlist if x not in tuning_target_params_list], " ]")
        for each_param in tuning_target_params_list:
            out_params[each_param]['low'] = original_stepwise_params[each_param]['low']
            out_params[each_param]['high'] = original_stepwise_params[each_param]['high']
        #print(original_stepwise_params, out_params)
        #print(out_params)
        mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
        study.optimize(mod_objective, 1)
        ######
        # if out_params has been changed,
        if out_params!=controlled_stepwise_params:
            controlled_stepwise_params_has_been_changed = True
        return study, out_params, controlled_stepwise_params_has_been_changed


def stepwise_guided_mpi_by_time_backup0723(study, objective, original_stepwise_params, controlled_stepwise_params, n_trials, guided_order_list, current_step,algorithm_name=None, default_params=None):
    out_params=copy.deepcopy(controlled_stepwise_params)
    randomfix_paramlist = find_linear(controlled_stepwise_params)
    ####
    ###########################################
    # set all linear params to default_params (in out_params)
    if study.trials == []:
        comm = MPI.COMM_WORLD   # get MPI communicator object
        size = comm.size        # total number of processes
        rank = comm.rank        # rank of this process
        if default_params: # given the default params
            if rank == 0:
                for each_param in randomfix_paramlist:
                    out_params[each_param]['low'] = default_params[each_param]
                    out_params[each_param]['high'] = default_params[each_param]
                mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
                study.optimize(mod_objective, 1)
                ##################
                n_trials = n_trials - 1
            else: # rank > 1
                mod_objective = ModObjectiveFunctionWrapper(objective, original_stepwise_params)        
                study.optimize(mod_objective, 1)
                ##################
                n_trials = n_trials - 1                 
        else: # if not use default params
            # try 1 optimize - rank 0 should run this
            ##################
            mod_objective = ModObjectiveFunctionWrapper(objective, original_stepwise_params)        
            study.optimize(mod_objective, 1)
            ##################
            n_trials = n_trials - 1
            #print("why it shuould run ************************************")
            # bugfix 0604
            if study.trials_dataframe()['value'].any():
                # find the default values if result exists
                for each_param in randomfix_paramlist: # remove iterative finding in a specific param
                    if algorithm_name:
                        search_param = algorithm_name+"_"+each_param # add algorithm prefix by 20200708
                    else:
                        search_param = each_param
                    default_value_list = []
                    for k, v in study.best_params.items():
                        if k.startswith(search_param):
                            default_value_list.append(v)
                    #print(default_value_list)
                    random.shuffle(default_value_list)
                    default_value = default_value_list.pop()
                    out_params[each_param]['low'] = default_value # 여기 버그를 고쳤음. 다른곳도 적용 필요 20200526
                    out_params[each_param]['high'] = default_value
        #print("stucked*********))()()()()()()(", out_params)
        return study, out_params
    else:
        try:
            we_have_best_value=study.best_value
        except:
            we_have_best_value=-1
        # current_step is bigger than 1
        ###########################################
        #return study, out_params
        # step-wise search (random sequence)
        #random.shuffle(randomfix_paramlist) # if we do not have ordered (a prior) params regarding importance of each param
        randomfix_paramlist = params_sorting_by_guided_list(base_list = guided_order_list, target_list = randomfix_paramlist)
        #print("randomfix_paramlist========================================================",randomfix_paramlist)
        #current_target_param_name = randomfix_paramlist[current_step - 1]
        tuning_target_params_list = randomfix_paramlist[:current_step]
        print("* CURRENT STEP: ", current_step)#, current_target_param_name, tuning_target_params_list)
        if len(tuning_target_params_list) == len(randomfix_paramlist):
            print("* [FULL-RANGE SEARCH] *")
        print("* TUNING TARGET PARAM: [ ", tuning_target_params_list, " ]", "* CONTROLLED PARAMS: [ ", [x for x in randomfix_paramlist if x not in tuning_target_params_list], " ]")
        for each_param in tuning_target_params_list:
            out_params[each_param]['low'] = original_stepwise_params[each_param]['low']
            out_params[each_param]['high'] = original_stepwise_params[each_param]['high']
        #print(original_stepwise_params, out_params)
        #print(out_params)
        mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
        study.optimize(mod_objective, 1)
        ######
        if (we_have_best_value==-1) and (out_params == original_stepwise_params):
            # do not update this out params
            print("do not update in this")
            out_params = None
        return study, out_params


def stepwise_guided_mpi(study, objective, original_stepwise_params, controlled_stepwise_params, n_trials, guided_order_list, current_step,algorithm_name=None, default_params=None):
    out_params=copy.deepcopy(controlled_stepwise_params)
    randomfix_paramlist = find_linear(controlled_stepwise_params)
    ###########################################
    # set all linear params to default_params (in out_params)
    if (current_step == 1):# and (study.trials==None):
        comm = MPI.COMM_WORLD   # get MPI communicator object
        size = comm.size        # total number of processes
        rank = comm.rank        # rank of this process
        if default_params: # given the default params
            if rank == 1:
                for each_param in randomfix_paramlist:
                    out_params[each_param]['low'] = default_params[each_param]
                    out_params[each_param]['high'] = default_params[each_param]
                mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
                study.optimize(mod_objective, 1)
                ##################
                n_trials = n_trials - 1
            else: # rank > 1
                mod_objective = ModObjectiveFunctionWrapper(objective, original_stepwise_params)        
                study.optimize(mod_objective, 1)
                ##################
                n_trials = n_trials - 1                 
        else: # if not use default params
            # try 1 optimize - rank 0 should run this
            ##################
            mod_objective = ModObjectiveFunctionWrapper(objective, original_stepwise_params)        
            study.optimize(mod_objective, 1)
            ##################
            n_trials = n_trials - 1
            #print("why it shuould run ************************************")
            # bugfix 0604
            if study.trials_dataframe()['value'].any():
                # find the default values if result exists
                for each_param in randomfix_paramlist: # remove iterative finding in a specific param
                    if algorithm_name:
                        search_param = algorithm_name+"_"+each_param # add algorithm prefix by 20200708
                    else:
                        search_param = each_param
                    default_value_list = []
                    for k, v in study.best_params.items():
                        if k.startswith(search_param):
                            default_value_list.append(v)
                    #print(default_value_list)
                    random.shuffle(default_value_list)
                    default_value = default_value_list.pop()
                    out_params[each_param]['low'] = default_value # 여기 버그를 고쳤음. 다른곳도 적용 필요 20200526
                    out_params[each_param]['high'] = default_value
        return study, out_params
    else: # current_step is bigger than 1
        ###########################################
        # step-wise search (random sequence)
        #random.shuffle(randomfix_paramlist) # if we do not have ordered (a prior) params regarding importance of each param
        randomfix_paramlist = params_sorting_by_guided_list(base_list = guided_order_list, target_list = randomfix_paramlist)
        current_target_param_name = randomfix_paramlist[current_step - 2]
        tuning_target_params_list = randomfix_paramlist[ : current_step - 1]
        print("* CURRENT STEP: ", current_step)#, current_target_param_name, tuning_target_params_list)
        if len(tuning_target_params_list) == len(randomfix_paramlist):
            print("* [FULL-RANGE SEARCH] *")
        print("* TUNING TARGET PARAM: [ ", tuning_target_params_list, " ]", "* CONTROLLED PARAMS: [ ", randomfix_paramlist, " ]")
        for each_param in tuning_target_params_list:
            out_params[each_param]['low'] = original_stepwise_params[each_param]['low']
            out_params[each_param]['high'] = original_stepwise_params[each_param]['high']
        #print(original_stepwise_params, out_params)
        print(out_params)
        mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
        study.optimize(mod_objective, 1)
        return study, out_params

def stepwise_guided(study, objective, params, n_trials, guided_order_list, default_params=None):
    out_params=copy.deepcopy(params)
    randomfix_paramlist = find_linear(params)
    ###########################################
    # set all linear params to default_params (in out_params)
    if default_params: # given the default params
        for each_param in randomfix_paramlist:
            out_params[each_param]['low'] = default_params[each_param]
            out_params[each_param]['high'] = default_params[each_param]
        mod_objective = ModObjectiveFunctionWrapper(objective, out_params)        
        study.optimize(mod_objective, 1)
        ##################
        n_trials = n_trials - 1
    else: # if not use default params
        # try 1 optimize - rank 0 should run this
        ##################
        mod_objective = ModObjectiveFunctionWrapper(objective, params)        
        study.optimize(mod_objective, 1)
        ##################
        n_trials = n_trials - 1
        # find the default values
        for each_param in randomfix_paramlist: # remove iterative finding in a specific param
            for k, v in study.best_params.items():
                if each_param == k:
                    default_value = v
            out_params[each_param]['low'] = default_value
            out_params[each_param]['high'] = default_value
    ###########################################
    # step-wise search (random sequence)
    #random.shuffle(randomfix_paramlist) # if we do not have ordered (a prior) params regarding importance of each param
    randomfix_paramlist = params_sorting_by_guided_list(base_list = guided_order_list, target_list = randomfix_paramlist)
    current_step = 1
    tuned = []
    for each_param in randomfix_paramlist:
        tuned.append(each_param)
        if len(tuned) == len(randomfix_paramlist):
            print("* [FULL-RANGE SEARCH] *")
        print("* TUNING TARGET PARAM: [ ", tuned, " ]", "* CONTROLLED PARAMS: [ ", randomfix_paramlist, " ]")
        out_params[each_param]['low'] = params[each_param]['low']
        out_params[each_param]['high'] = params[each_param]['high']
        mod_objective = ModObjectiveFunctionWrapper(objective, out_params)
        study.optimize(mod_objective, n_iter_calculation(n_trials, len(randomfix_paramlist), current_step))
        ##################
        current_step += 1
    return study

def _init_seed_fix_torch(manualSeed):
    import numpy as np
    import torch
    import random
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
##########################################################################################
def history_plot(df):
    # type: (Study) -> go.Figure
    df['cum_time_to_sec'] = df['datetime_complete'].apply(lambda x: (x - df['datetime_start'].iloc[0]).seconds)
    df=df[df['state']=='COMPLETE']
    layout = go.Layout(
        title="Optimization History Plot",
        xaxis={"title": "#sec"},
        yaxis={"title": "cv5_f-1 score(macro)"},
    )
    best_values=[]
    cur_max = -float("inf")
    df=df.sort_values(by=['cum_time_to_sec'])
    for i in range(len(df)):
        cur_max = max(cur_max, df['value'].iloc[i])
        best_values.append(cur_max)
    traces = [
        go.Scatter(
            x=df['cum_time_to_sec'],
            y=df['value'],
            mode="markers",
            name="f-1",
        ),
        go.Scatter(x=df['cum_time_to_sec'], y=best_values, name="Best Value", mode='lines+markers'),
    ]
    figure = go.Figure(data=traces, layout=layout)
    return figure

def history_plot_compare(df, df2):
    # type: (Study) -> go.Figure
    layout = go.Layout(
            title="Optimization History Plot_#nproc_6",
            xaxis={"title": "#sec"},
            yaxis={"title": "cv5_f-1 score(macro)"},
        )
    def get_traces(df, df_name):
        df['cum_time_to_sec'] = df['datetime_complete'].apply(lambda x: (x - df['datetime_start'].iloc[0]).seconds)
        df=df[df['state']=='COMPLETE']
        
        best_values=[]
        cur_max = -float("inf")
        df=df.sort_values(by=['cum_time_to_sec'])
        for i in range(len(df)):
            cur_max = max(cur_max, df['value'].iloc[i])
            best_values.append(cur_max)
        traces = [
            go.Scatter(
                x=df['cum_time_to_sec'],
                y=df['value'],
                mode="markers",
                marker=dict(size=3),
                name=df_name+" : f-1",
            ),
            go.Scatter(x=df['cum_time_to_sec'], y=best_values, name=df_name+" : Best Value", mode='lines+markers'),
        ]
        return traces
    t1=get_traces(df, "stepwise")
    t2=get_traces(df2, "optuna-base")
    #return t1,t2
    figure = go.Figure(data=t1+t2, layout=layout)
    return figure    


# time consumed calculation
def study_sec_calculation(study):
    '''
    (whole time, best perf time, best_time_percentage),
    (n_complete, total complete time, percentage, avg complete time),
    (n_pruned, total pruned time, percentage, avg pruned time),
    (n_others, total others time, percentage, avg others time)
    '''
    best_trial_index = study.best_trial.number
    s=study.trials_dataframe().copy(deep=True)
    #
    s['delta'] = s['datetime_complete'] - s['datetime_start']
    s['delta_to_sec'] = s['delta'].apply(lambda x:x.seconds)
    wt = s['delta_to_sec'].sum()
    bpt = s.loc[:best_trial_index]['delta_to_sec'].sum()
    bt_per = bpt/wt*100
    #
    comp_df = s[s['state']=='COMPLETE']
    prun_df = s[s['state']=='PRUNED']
    others_df = s[s['state']!='COMPLETE']
    others_df = others_df[others_df['state']!='PRUNED']
    #
    def get_len_sum_avg(df, wt):
        length_df = len(df)
        sum_df = df['delta_to_sec'].sum()
        avg_df = sum_df / length_df
        per_df = sum_df / wt * 100
        #print(sum_df, wtds, per_df)
        return length_df, sum_df, per_df, avg_df
    #
    return (wt, bpt, bt_per), (get_len_sum_avg(comp_df, wt)), (get_len_sum_avg(prun_df, wt)), (get_len_sum_avg(others_df, wt))

#############################
#params={    'n_layers': {'low':10, 'high':20},
#            'n_units': {'low':4, 'high':128},
#            'dropout': {'low':0.2, 'high':0.5},
#            'optimizer': {'choices':['Adam', 'RMSprop', 'SGD']},
#            'lr': {'low':1e-5, 'high':1e-1},
#            'batchsize':{'choices':[32, 64, 128, 256]},
#            'epochs':20,
#            'device':'cuda',
#            'classes':10
#        }
#default_params ={
#            'n_layers': 1,
#            'n_units': 4,
#            'dropout': 0.2,
#            'optimizer': 'Adam',
#            'lr': 0.001,
#            'epochs':10,
#            'batchsize':64,
#        }
############################
def do_optimize_test(n_trials):
    study_name = 'pytorch_asha_mnist-with-sampler_mod0423-ver1'
    #sampler=optuna.integration.CmaEsSampler(warn_independent_sampling=False)
    #pruner=optuna.pruners.SuccessiveHalvingPruner()
    direction='maximize'
    study = optuna.create_study(study_name=study_name, direction=direction,load_if_exists=True)#, pruner=pruner)
    mod_objective = ModObjectiveFunctionWrapper(objective, params)
    #study.pruner=optuna.pruners.SuccessiveHalvingPruner()
    #study.optimize(mod_objective, n_trials)
    #
    study = uncertainty_reduction_ver7(study, params, n_trials)
    study.trials_dataframe().to_csv('out.csv')
    return study