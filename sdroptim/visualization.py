import plotly.graph_objects as go
from plotly.offline import plot as offplot
import numpy as np
import optuna, base64, json, argparse

def history_plot(df, study_name, direction):
    # type: (Study) -> go.Figure
    layout = go.Layout(
            title=study_name,
            xaxis={"title": "# seconds"},
            yaxis={"title": "target score"},
        )
    def get_traces(df, study_name, direction):
        df['cum_time_to_sec'] = df['datetime_complete'].apply(lambda x: (x - df['datetime_start'].iloc[0]).seconds)
        df=df[df['state']=='COMPLETE']
        
        best_values=[]
        df=df.sort_values(by=['cum_time_to_sec'])
        if direction == 'maximize':
            cur_max = -float("inf")
            for i in range(len(df)):
                cur_max = max(cur_max, df['value'].iloc[i])
                best_values.append(cur_max)
        elif direction == 'minimize':
            cur_min = float("inf")
            for i in range(len(df)):
                cur_min = min(cur_min, df['value'].iloc[i])
                best_values.append(cur_min)
        #
        traces = [
            go.Scatter(
                x=df['cum_time_to_sec'],
                y=df['value'],
                mode="markers",
                marker=dict(size=3),
                name="Individual Score",
            ),
            go.Scatter(x=df['cum_time_to_sec'], y=best_values, name="Cummulative Score", mode='lines+markers'),
        ]
        return traces
    t1=get_traces(df, study_name, direction)
    #t2=get_traces(df2, "optuna-base")
    #return t1,t2
    figure = go.Figure(data=t1, layout=layout)
    return figure

def get_study_df(json_file_name):
    with open(json_file_name) as data_file:
        gui_params = json.load(data_file)
    key = str(base64.b64decode('cG9zdGdyZXNxbDovL3Bvc3RncmVzOnBvc3RncmVzQDE1MC4xODMuMjQ3LjI0NDo1NDMyLw=='))[2:-1]
    url = key + gui_params['hpo_system_attr']['user_name']
    study_name = gui_params['hpo_system_attr']['study_name']
    try:
        rdbs = optuna.storages.RDBStorage(url)
        study_id = rdbs.get_study_id_from_name(study_name)
    except:
        raise ValueError(
                "Cannot find the study_name {}".format(
                    study_name#, args.storage_url
                ))
    try:
        direction='minimize'
        s = optuna.create_study(load_if_exists=True, study_name=study_name, storage=url, direction=direction)
    except:
        direction='maximize'
        s = optuna.create_study(load_if_exists=True, study_name=study_name, storage=url, direction=direction)
    return s.trials_dataframe(), study_name, direction


def get_chart_html(args, with_df_csv=False):
    df, study_name, direction = get_study_df(args.json_file_name)
    figure = history_plot(df, study_name, direction)
    if not args.outfile_name:
        args.outfile_name = study_name+"_df.csv"
    if with_df_csv:
        df.to_csv(args.outfile_name)
    offplot(figure, filename = args.outhtml_name, auto_open=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_name', help="json_file_name", type=str, default = '') # json_file_name
    parser.add_argument('--outfile_name', help="output filename", type=str)
    parser.add_argument('--outhtml_name', help="output htmlname", type=str, default='cum_chart.html')
    args = parser.parse_args()
    import easydict
    args = easydict.EasyDict({
            #"user_name":"",
            "json_file_name":"metadata.json",
            "outfile_name":"",
            "outhtml_name":"cum_chart.html"
    })
    get_chart_html(args, with_df_csv=True)
