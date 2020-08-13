"""
  Job Submit API for Hyper Parameter Optimization by Jeongcheol lee
  (for Jupyter User)
  -- jclee@kisti.re.kr
"""

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
    ###
    if args.job_id == "": ## generate job_id(directory)
        jobpath, (uname, sname, jname, wsname, job_id) = sdroptim.get_jobpath_with_attr()
    else:
    	with open(args.metadata) as data_file:
            gui_params = json.load(data_file)
