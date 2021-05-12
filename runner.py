#!/usr/bin/python3
"""os-migrate grapher script."""

"""
## Requirements:
## From this folder execute:

python3 -m pip install ansible-runner
git clone git@github.com:os-migrate/os-migrate.git
cd os-migrate
ansible-galaxy collection build os_migrate -v --force --output-path releases/
cd releases
LATEST=$(ls os_migrate-os_migrate*.tar.gz | grep -v latest | sort -V | tail -n1)
ln -sf $LATEST os_migrate-os_migrate-latest.tar.gz
ansible-galaxy collection install --force os_migrate-os_migrate-latest.tar.gz
"""

import os
import tempfile
import subprocess
from pathlib import Path
import sys, yaml, json;
import ansible_runner
import shutil

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.dates
from matplotlib.dates import SECONDLY,WEEKLY,MONTHLY, DateFormatter, rrulewrapper, RRuleLocator
import numpy as np
import matplotlib.dates as mdates

import pandas as pd
import matplotlib as mpl
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

#
# How many times we will execute the experiments
#
# This means that we will execute the os-migrate to
# migrate [a, b, c, d] amount of resources x times
sample_iterations = 10
sample_iterations_list = [2, 4, 8]

sample_data_path = "./sample_data/"

#
# Two lists of playbooks we will execute before and after the main run list executes
# these lists are useful i.e. in the case of seeding an environment or cleaning up after.
#
sample_run_pre = [] # "/home/ccamacho/chart/aux/clean_src.yml", "/home/ccamacho/chart/aux/clean_dst.yml"]
sample_run_pre_experiments = ["/home/ccamacho/chart/aux/seed.yml"]
sample_run_post_experiment = ["/home/ccamacho/chart/aux/clean_dst.yml"]
sample_run_post = ["/home/ccamacho/chart/aux/clean_src.yml", "/home/ccamacho/chart/aux/clean_dst.yml"]

#
# This is a list of playbooks we will execute to generate the graph details
#
sample_run_list = [
    {'resource': 'networks', 'type': 'export', 'playbook': 'export_networks.yml', 'graph': True},
    {'resource': 'networks', 'type': 'import', 'playbook': 'import_networks.yml', 'graph': True},
    {'resource': 'subnets', 'type': 'export', 'playbook': 'export_subnets.yml', 'graph': True},
    {'resource': 'subnets', 'type': 'import', 'playbook': 'import_subnets.yml', 'graph': True},
    {'resource': 'routers', 'type': 'export', 'playbook': 'export_routers.yml', 'graph': True},
    {'resource': 'routers', 'type': 'import', 'playbook': 'import_routers.yml', 'graph': True},
    {'resource': 'security_groups', 'type': 'export', 'playbook': 'export_security_groups.yml', 'graph': True},
    {'resource': 'security_groups', 'type': 'import', 'playbook': 'import_security_groups.yml', 'graph': True},
    {'resource': 'security_group_rules', 'type': 'export', 'playbook': 'export_security_group_rules.yml', 'graph': True},
    {'resource': 'security_group_rules', 'type': 'import', 'playbook': 'import_security_group_rules.yml', 'graph': True},
    {'resource': 'workloads', 'type': 'export', 'playbook': 'export_workloads.yml', 'graph': False},
    {'resource': 'workloads', 'type': 'import', 'playbook': 'import_workloads.yml', 'graph': False}
]

global_times = {}


def main():
    """Execute all the methods."""
    # Initialize global_times
    for resource in sample_run_list:
        global_times[resource['type']+"_"+resource['resource']] = {}
        for amount_resources in sample_iterations_list:
            global_times[resource['type']+"_"+resource['resource']][amount_resources]=[]
    print(global_times)


    clean_local_folders()
    run_extra_playbooks(sample_run_pre)

    for amount_resources in sample_iterations_list:
        # seed
        run_extra_playbooks(sample_run_pre_experiments, amount_resources)


        for resource in sample_run_list:
            if resource['graph']:
                for experiment_index in range(sample_iterations):
                    # run
                    global_times[resource['type']+"_"+resource['resource']][amount_resources].append(render_gantt_data(sample_data_path, resource, experiment_index, amount_resources))
                    render_gantt_chart(sample_data_path, resource, experiment_index, amount_resources)
                    if resource['type'] == "import":
                        # We clean the destination tenant
                        run_extra_playbooks(sample_run_post_experiment, amount_resources)

    render_boxplot_data()
    for key in global_times:
        render_box_plot_chart(sample_data_path, key)
    # clean
    run_extra_playbooks(sample_run_post, max(sample_iterations_list))
"""
{'export_networks': {1: [], 2: []}, 'import_networks': {1: [], 2: []}, 'export_subnets': {1: [], 2: []}, 'import_subnets': {1: [], 2: []}, 'export_routers': {1: [], 2: []}, 'import_routers': {1: [], 2: []}, 'export_security_groups': {1: [], 2: []}, 'import_security_groups': {1: [], 2: []}, 'export_security_group_rules': {1: [], 2: []}, 'import_security_group_rules': {1: [], 2: []}, 'export_workloads': {1: [], 2: []}, 'import_workloads': {1: [], 2: []}}

"""
def render_boxplot_data():
    # We parse each resource to be exported
    for key in global_times:
        with open(os.path.join(sample_data_path, key+'.csv'), 'w') as the_file:
            the_file.write('"usage","bw","execution_time","experiment"'+ "\n")
        for value in global_times[key]:
            for time in global_times[key][value]:
                with open(os.path.join(sample_data_path, key+'.csv'), 'a') as the_file2:
                    the_file2.write("100,33," + str(time) + "," + str(value) + "\n")

def clean_local_folders():
    private_data_dir = '/tmp/osm/'
    Path(private_data_dir).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(private_data_dir, ignore_errors = True)

def run_extra_playbooks(playbook_list, amount_resources=1):
    """Execute Ansible runner for additional playbooks."""

    local_inventory = {
        'hosts': {
            'migrator': {
                'ansible_host': '127.0.0.1'
            }
        },
        'vars': {
            'ansible_python_interpreter': 'python3',
            'callback_whitelist': 'ansible.posix.profile_json, ansible.posix.profile_tasks'
        }
    }

    home = os.environ.get('HOME', '/root/')
    clouds = '/etc/openstack/clouds.yaml'
    private_data_dir = '/tmp/osm/'
    Path(private_data_dir).mkdir(parents=True, exist_ok=True)

    private_data_dir_env = os.path.join(private_data_dir, 'env/')
    private_data_dir_osm = os.path.join(private_data_dir, 'osm-data/')
    private_data_dir_artifacts = os.path.join(private_data_dir, 'artifacts/')

    shutil.rmtree(private_data_dir_artifacts, ignore_errors = True)

    Path(os.path.join(home, 'os-migrate-data')).mkdir(parents=True, exist_ok=True)
    Path(private_data_dir_env).mkdir(parents=True, exist_ok=True)
    Path(private_data_dir_osm).mkdir(parents=True, exist_ok=True)
    Path(sample_data_path).mkdir(parents=True, exist_ok=True)

    src_cloud = 'psisrc16'
    dst_cloud = 'psidst16'
    osm_collection_root = os.path.join(home, '.ansible/collections/ansible_collections/os_migrate/os_migrate')
    osm_playbooks_root = os.path.join(osm_collection_root, 'playbooks')

    val = subprocess.check_call("./os-migrate/scripts/auth-from-clouds.sh --config " + clouds + " --src " + src_cloud + " --dst " + dst_cloud + " > " + os.path.join(private_data_dir_env, 'extravars.yml'), shell=True)

    with open(os.path.join(private_data_dir_env, 'extravars.yml'), 'r') as yaml_in, open(os.path.join(private_data_dir_env, 'extravars'), "w") as json_out:
        yaml_object = yaml.safe_load(yaml_in)
        json.dump(yaml_object, json_out)

    envvars = {
        'ANSIBLE_INVENTORY_PLUGIN_EXTS': '.json',
        'OS_MIGRATE': osm_collection_root,
        'OS_MIGRATE_DATA': private_data_dir_osm,
        'SRC_CLOUD': src_cloud,
        'DST_CLOUD': dst_cloud,
        'CLOUDS': clouds
    }

    extravars = {
        'amount_resources': amount_resources,
        'ansible_connection': 'local',
        'os_migrate_src_release': '16',
        'os_migrate_dst_release': '16',
        'os_migrate_data_dir': private_data_dir_osm,
        'os_migrate_tests_tmp_dir': private_data_dir_osm,
        'os_migrate_src_validate_certs': False,
        'os_migrate_dst_validate_certs': False,
        'os_migrate_conversion_host_ssh_user': 'centos',
        'os_migrate_networks_filter': [{'regex': '^osm_bench_'}],
        'os_migrate_subnets_filter': [{'regex': '^osm_bench_'}],
        'os_migrate_routers_filter': [{'regex': '^osm_bench_'}],
        'os_migrate_security_groups_filter': [{'regex': '^osm_bench_'}],
        'os_migrate_workloads_filter': [{'regex': '^osm_bench_'}]
    }

    #
    # Ansible configuration ends
    #

    #
    # Playbooks execution begins
    #

    for pre_task in playbook_list:
        kwargs = {
            'verbosity': 0,
            'playbook': pre_task,
            'inventory': {'all': local_inventory},
            'envvars': envvars,
            'extravars': extravars,
            'private_data_dir': private_data_dir
        }

        pre_runner_obj = ansible_runner.interface.init_runner(**kwargs)
        pre_runner_obj.run()
        pre_runner_obj = None
        ansible_runner.utils.cleanup_artifact_dir(private_data_dir_artifacts, 1)

def render_gantt_data(sample_data_path, resource, experiment_index, amount_resources):
    """Execute Ansible runner."""
    #
    # Ansible configuration begins
    #

    local_inventory = {
        'hosts': {
            'migrator': {
                'ansible_host': '127.0.0.1'
            }
        },
        'vars': {
            'ansible_python_interpreter': 'python3',
            'callback_whitelist': 'ansible.posix.profile_json, ansible.posix.profile_tasks'
        }
    }

    home = os.environ.get('HOME', '/root/')
    clouds = '/etc/openstack/clouds.yaml'
    private_data_dir = '/tmp/osm/'
    Path(private_data_dir).mkdir(parents=True, exist_ok=True)

    private_data_dir_env = os.path.join(private_data_dir, 'env/')
    private_data_dir_osm = os.path.join(private_data_dir, 'osm-data/')
    private_data_dir_artifacts = os.path.join(private_data_dir, 'artifacts/')

    shutil.rmtree(private_data_dir_artifacts, ignore_errors = True)

    Path(os.path.join(home, 'os-migrate-data')).mkdir(parents=True, exist_ok=True)
    Path(private_data_dir_env).mkdir(parents=True, exist_ok=True)
    Path(private_data_dir_osm).mkdir(parents=True, exist_ok=True)
    Path(sample_data_path).mkdir(parents=True, exist_ok=True)

    src_cloud = 'psisrc16'
    dst_cloud = 'psidst16'
    osm_collection_root = os.path.join(home, '.ansible/collections/ansible_collections/os_migrate/os_migrate')
    osm_playbooks_root = os.path.join(osm_collection_root, 'playbooks')

    val = subprocess.check_call("./os-migrate/scripts/auth-from-clouds.sh --config " + clouds + " --src " + src_cloud + " --dst " + dst_cloud + " > " + os.path.join(private_data_dir_env, 'extravars.yml'), shell=True)

    with open(os.path.join(private_data_dir_env, 'extravars.yml'), 'r') as yaml_in, open(os.path.join(private_data_dir_env, 'extravars'), "w") as json_out:
        yaml_object = yaml.safe_load(yaml_in)
        json.dump(yaml_object, json_out)

    envvars = {
        'ANSIBLE_INVENTORY_PLUGIN_EXTS': '.json',
        'OS_MIGRATE': osm_collection_root,
        'OS_MIGRATE_DATA': private_data_dir_osm,
        'SRC_CLOUD': src_cloud,
        'DST_CLOUD': dst_cloud,
        'CLOUDS': clouds
    }

    extravars = {
        'ansible_connection': 'local',
        'os_migrate_src_release': '16',
        'os_migrate_dst_release': '16',
        'os_migrate_data_dir': private_data_dir_osm,
        'os_migrate_tests_tmp_dir': private_data_dir_osm,
        'os_migrate_src_validate_certs': False,
        'os_migrate_dst_validate_certs': False,
        'os_migrate_conversion_host_ssh_user': 'centos',
        'os_migrate_networks_filter': [{'regex': '^osm_bench_'}],
        'os_migrate_subnets_filter': [{'regex': '^osm_bench_'}],
        'os_migrate_routers_filter': [{'regex': '^osm_bench_'}],
        'os_migrate_security_groups_filter': [{'regex': '^osm_bench_'}],
        'os_migrate_workloads_filter': [{'regex': '^osm_bench_'}]
    }

    #
    # Ansible configuration ends
    #

    #
    # Playbooks execution begins
    #

    # There is a gantt chart per experiment
    with open(os.path.join(sample_data_path, str(amount_resources)+"_"+str(experiment_index) +"_"+ resource['resource'] + "_" +resource['type'] + '.txt'), 'w') as the_file2:
        the_file2.write('')
    kwargs = {
        'verbosity': 0,
        'playbook': os.path.join(osm_playbooks_root, resource['playbook']),
        'inventory': {'all': local_inventory},
        'envvars': envvars,
        'extravars': extravars,
        'private_data_dir': private_data_dir
    }

    runner_obj = ansible_runner.interface.init_runner(**kwargs)
    runner_obj.run()

    stdout = runner_obj.stdout.read()
    events = list(runner_obj.events)
    stats = runner_obj.stats
    runner_obj = None
    ansible_runner.utils.cleanup_artifact_dir(private_data_dir_artifacts, 1)

    first_event = ''
    last_event = ''

    for event in events:
        print('-------------')
        if 'event_data' in event:
            if 'duration' in event['event_data']:
                if first_event == '':
                    first_event = event['event_data']['start']
                with open(os.path.join(sample_data_path, str(amount_resources)+"_"+str(experiment_index) +"_" + resource['resource'] + "_" +resource['type'] + '.txt'), 'a') as the_file2:
                    the_file2.write(event['event_data']['task'] + "," + event['event_data']['start'] +"," + event['event_data']['end'] + "\n")
                print("Task:" + event['event_data']['task'])
                print("Created:" + event['created'])
                print("Start:" + event['event_data']['start'])
                print("End:" + event['event_data']['end'])
                print("Duration:" + str(event['event_data']['duration']))
                print(event)
                last_event = event['event_data']['end']
    print("Playbook started at:" + first_event)
    print("Playbook ended at:" + last_event)
    first_event_obj = dt.datetime.strptime(first_event, '%Y-%m-%dT%H:%M:%S.%f')
    last_event_obj = dt.datetime.strptime(last_event, '%Y-%m-%dT%H:%M:%S.%f')
    print((last_event_obj-first_event_obj).total_seconds())

    resource['playbook'].split('.')[0]
    # Write the results to the gantt file
    f = open("gantt_"+resource['playbook'].split('.')[0]+".txt", "a")
    f.writelines(["See you soon!", "Over and out."])
    f.close()

    #
    # The artifact dir must be cleaned on every execution
    # otherwise it will append the events from every run.
    #
    ansible_runner.utils.cleanup_artifact_dir(private_data_dir_artifacts, 1)

    return (last_event_obj-first_event_obj).total_seconds() or 0

    #
    # Playbooks execution begins
    #

def _create_date(datetxt):
    """Creates the date"""
    day,month,year=datetxt.split('-')
    date = dt.datetime.strptime(datetxt, '%Y-%m-%dT%H:%M:%S.%f')
    mdate = matplotlib.dates.date2num(date)
    return mdate


def render_gantt_chart(sample_data_path, resource, experiment_index, amount_resources):
    """
        Create gantt charts with matplotlib
        Give file name.
    """
    fname=os.path.join(sample_data_path, str(amount_resources)+"_"+str(experiment_index) +"_" + resource['resource'] + "_" +resource['type'] + '.txt')
    ylabels = []
    customDates = []
    try:
        with open(fname, 'r') as gfile:
            textlist=gfile.readlines()
    except:
        return

    all_dates = []

    for tx in textlist:
        if not tx.startswith('#'):
            ylabel,startdate,enddate=tx.split(',')
            ylabels.append((ylabel[:25] + '..').replace('\n','') if len(ylabel) > 25 else ylabel.replace('\n',''))
            customDates.append([_create_date(startdate.replace('\n','')),_create_date(enddate.replace('\n',''))])
            all_dates.append(_create_date(startdate.replace('\n','')))
            all_dates.append(_create_date(enddate.replace('\n','')))
    ilen=len(ylabels)
    pos = np.arange(0.5,ilen*0.5+0.5,0.5)
    task_dates = {}
    for i,task in enumerate(ylabels):
        task_dates[task] = customDates[i]
    fig = plt.figure(figsize=(30,10))
    ax = fig.add_subplot(111)
    for i in range(len(ylabels)):
         start_date,end_date = task_dates[ylabels[i]]
         ax.barh((i*0.5)+0.5, end_date - start_date, left=start_date, height=0.3, align='center', edgecolor='lightgreen', color='orange', alpha = 0.8)
    locsy, labelsy = plt.yticks(pos,ylabels)
    plt.setp(labelsy, fontsize = 14)
#    ax.axis('tight')
    ax.set_ylim(ymin = -0.1, ymax = ilen*0.5+0.5)
    ax.grid(color = 'g', linestyle = ':')
    ax.xaxis_date()

    locator = mdates.AutoDateLocator(minticks=10, maxticks=15)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(min(all_dates),max(all_dates))

    #plt.xticks(all_dates)
    labelsx = ax.get_xticklabels()
    plt.setp(labelsx, rotation=30, fontsize=10)

    font = font_manager.FontProperties(size='small')
    # No legend
    #ax.legend(loc=1,prop=font)

    plt.title((resource['type']+" "+resource['resource'] + " (" + str(amount_resources)+"_"+str(experiment_index) + ")").upper())
    ax.invert_yaxis()
    fig.autofmt_xdate()
    plt.savefig(os.path.join(sample_data_path, str(amount_resources)+"_"+str(experiment_index) +"_" + resource['resource'] + "_" +resource['type'] + '.svg'))
    #plt.show()


def render_box_plot_chart(sample_data_path, key):

    if len(open(os.path.join(sample_data_path, key+'.csv')).readlines()) > 1:

        large = 22; med = 16; small = 12
        params = {'axes.titlesize': large,
                  'legend.fontsize': med,
                  'figure.figsize': (16, 10),
                  'axes.labelsize': med,
                  'axes.titlesize': med,
                  'xtick.labelsize': med,
                  'ytick.labelsize': med,
                  'figure.titlesize': large}
        plt.rcParams.update(params)
        plt.style.use('seaborn-whitegrid')
        sns.set_style("white")
        #%matplotlib inline

        # Version
        print(mpl.__version__)  #> 3.0.0
        print(sns.__version__)  #> 0.9.0

        df = pd.read_csv(os.path.join(sample_data_path, key+'.csv'))

        # Draw Plot
        fig = plt.figure(figsize=(13,10), dpi= 80)

        #
        grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
        # Define the main axis
        #ax_main = fig.add_subplot(grid[:-1, :-1])


        ax = sns.boxplot(x='experiment', y='execution_time', data=df, hue='usage')
        ax.set_yscale('log')
        sns.stripplot(x='experiment', y='execution_time', data=df, color='black', size=3, jitter=1)



        for i in range(len(df['experiment'].unique())-1):
            plt.vlines(i+.5, 10, 6000, linestyles='solid', colors='gray', alpha=0.2)


        # Decoration
        plt.title(key, fontsize=22)
        #plt.legend(title='Volume usage (%)')

        #
        # # # # THe bottom axis
        # ax_bottom = fig.add_subplot(grid[-1, 0:-1], sharex = ax_main)
        # sns.boxplot(x='experiment', y='bw', data=df, hue='usage')
        #sns.stripplot(x='experiment', y='bw', data=df, color='black', size=3, jitter=1)
        #sns.histplot(data=df, x="experiment", y='bw', hue='usage')

        #ax_bottom.hist(df.bw, 40, histtype='stepfilled', orientation='vertical', color='deeppink')
        # #
        # for i in range(len(df['experiment'].unique())-1):
        #     plt.vlines(i+.5, 10, 45, linestyles='solid', colors='gray', alpha=0.2)


        #plt.show()
        plt.savefig(os.path.join(sample_data_path, key + '.svg'))


if __name__ == "__main__":
    main()
