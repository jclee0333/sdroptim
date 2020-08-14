import ast, astunparse
userpy="sample.py"
with open(userpy) as f:
    p = ast.parse(f.read())
for node in p.body[:]:
    if type(node) not in [ast.FunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef]:
        p.body.remove(node)
objective_name_list = []
for node in p.body[:]:
    if type(node) in [ast.FunctionDef, ast.ClassDef]:
        if 'objective' in node.name.lower():
            objective_name_list.append(node.name)
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



for each_line in pre.split("\n"):
    if 'trial.suggest_' in each_line:
        print(each_line)
#
d = {}
lines=pre.split("\n")
for i in range(0,len(lines)):
    if 'trial.suggest_' in lines[i]:
        for j in range(0,len(lines[i])):
            if lines[i][j]==''
            if lines[i][j]=='(':
                from_index=j
        for j in range(0,len(lines[i])):
            if lines[i][j]==')':
                to_index=j
        print(lines[i][from_index:to_index+1])

        print(lines[i])

###################################################################################
d = {}
lines=pre.split("\n")
for i in range(0,len(lines)):
    if 'trial.suggest_' in lines[i]:
        from_index=lines[i].index('(')
        to_index=lines[i].index(')')
        target = lines[i][from_index+1:to_index]
        target.replace("'","").replace('"',"")
        #
        #print(target)
        if 'trial.suggest_categorical' in lines[i]:
            cate_from_index=target.index('[')
            cate_to_index=target.index(']')
            cate_items=target[cate_from_index:cate_to_index+1]
            #target.replace(cate_items,"---")
        else:
            #targets=target.split(',')
            targets=[x.strip() for x in target.split(',')]
            print(targets)

###########################################################################################



        #from_index=lines[i].index('trial.suggest_')
        to_index=lines[i].index(')')
        print(lines[i][from_index:to_index+1])