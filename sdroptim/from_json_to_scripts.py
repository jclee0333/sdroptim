import json, os
import sdroptim
import sdroptim.RCodeGenerator as RGen
import sdroptim.PythonCodeModulator as PyMod

def FullscriptsGenerator(json_file_name):
    # GUI parameters loading part
    json_file_path = "./"
    json_file_number = ""
    with open(json_file_name) as data_file:
        gui_params = json.load(data_file)
    #
    prefix_generated_code = "json_file_name = "+"'"+json_file_path+json_file_number+json_file_name+"'\n"
    #
    temp = gui_params.copy()
    #############################
    ## make python script (.py)
    #############################
    if type(gui_params['algorithm']) is list:
        jobpath, (uname, sname, jname, wsname, job_id) = sdroptim.get_jobpath_with_attr(gui_params)
        generated_code = PyMod.from_gui_to_code(gui_params)        
        with open(jobpath+os.sep+jname+'_generated.py', 'w') as f:
            f.write(prefix_generated_code+generated_code)
    #############################
        #############################
        ## make job script(sbatch)
        #############################
        jobscripts = PyMod.get_batch_script(gui_params)
        with open(jobpath+os.sep+'job.sh', 'w') as f:
        	f.write(jobscripts)
        #############################
    else: # if not hpo
        if gui_params['kernel'] == 'R':
            generated_code = RGen.from_gui_to_code(gui_params)
        elif gui_params['kernel'] == 'Python':
            generated_code = PyMod.from_gui_to_code(gui_params)
            generated_code = prefix_generated_code + generated_code
        else:
            generated_code = '[ERR] Empty kernel!'
        with open('generated.py', 'w') as f:
            f.write(generated_code)
####
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_name', help="name of jsonfile generated by GUI interfaces", default="aml_pytorch-classification-indirect.json")
    args=parser.parse_args()
    FullscriptsGenerator(args.json_file_name)