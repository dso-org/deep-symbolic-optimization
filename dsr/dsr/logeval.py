"""Tools to evaluate generated logfiles based on log directory."""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import click
import json
import os
import pandas as pd
import seaborn as sns

from pkg_resources import resource_filename

from matplotlib import pyplot as plt

class LogEval():
    """Class to hold all logged information and provide tools
    to analyze experiments."""

    PLOT_HELPER = {
        'hof': {
            'name': 'Hall of Fame',
            'x_label': [
                'HoF reward distrubtion',
                'HoF error distrubtion'],
            'x': [
                'index',
                'index'],
            'y': [
                'r',
                'nmse_test']
        },
        'pf': {
            'name': 'Pareto Front',
            'x_label': [
                'Reward over Complexity',
                'Error over Complexity'],
            'x': [
                'complexity',
                'complexity'],
            'y': [
                'r',
                'nmse_test']
        }
    }

    def __init__(self,
                 log_path,
                 config_file='config.json',
                 cmd_file='cmd.out',
                 benchmark_file='benchmark_dsr.csv'):
        """Loads all files from log path."""
        # Prepare variable to store warnings when reading the log
        self.warnings = []
        # Prepare variable to store calculated metrics
        self.metrics = {}

        # define pathes
        self.path = {}
        self.path['log'] = log_path
        self.path['config'] = os.path.join(log_path, config_file)
        self.path['benchmark'] = os.path.join(log_path, benchmark_file)
        self.path['cmd'] = os.path.join(log_path, cmd_file)

        # Get information about the command line arguments
        self.cmd_params = self._get_cmd()
        # Load the saved configuration data
        self.exp_config = self._get_config()
        # Load benchmark data (one row per seed)
        #self.benchmark_df = self._get_benchmarks()

        # TODO: Get data for all tasks if more than one
        # Load hof if available
        self.hof_df = self._get_log(log_type='hof')
        # Load pareto front if available
        self.pf_df = self._get_log(log_type='pf')

        if len(self.warnings) > 0:
            print('### Experiment has warnings:')
            [print('    --> {}'.format(warning)) for warning in self.warnings]

    def _get_cmd(self):
        """Get all command line parameter."""
        params = None
        try:
            with open (self.path['cmd'], 'r') as cmd_file:
                cmd_content = cmd_file.readlines()
            tokens = cmd_content[0].split('--')
            params = {}
            for token in tokens[1:]:
                setting = token.split('=')
                params[setting[0]] = self._get_correct_type(setting[1].strip())
            if not 'mc' in params:
                params['mc'] = 1
        except:
            self.warnings.append('Missing command file!')
        return params

    def _get_correct_type(self, token):
        """Make sute the token are recognized in the correct type."""
        if any(c.isdigit() for c in token):
            if any(c.isalpha() for c in token):
                return str(token)
            elif "." in token:
                return float(token)
            else:
                return int(token)
        return str(token)

    def _get_config(self):
        """Read the experiment configuration file."""
        exp_config = None
        try:
            with open(self.path['config'], 'r') as read_file:
                exp_config = json.load(read_file)
        except:
            self.warnings.append('Missing config file!')
        if exp_config['task']['function_set'] is None:
            exp_config['task']['function_set'] = self._get_tokenset(exp_config['task'])
        return exp_config

    def _get_tokenset(self, task_config):
        # load all available benchmarks
        root_dir = resource_filename("dsr.task", "regression") \
            if task_config['dataset']['root'] == None else task_config['dataset']['root']
        benchmark_df = pd.read_csv(
            os.path.join(root_dir, "benchmarks.csv"),
            index_col=None, encoding="ISO-8859-1")
        tokenset_df = pd.read_csv(
            os.path.join(root_dir, "function_sets.csv"),
            index_col=None, encoding="ISO-8859-1")
        tokenset_name = benchmark_df[benchmark_df['name']==task_config['name']]['function_set'].item()
        return tokenset_df[tokenset_df['name']==tokenset_name]['function_set'].item()

    def _get_benchmarks(self):
        """Read summarized benchmark data for each seed."""
        benchmark_df = None
        try:
            benchmark_df = pd.read_csv(self.path['benchmark'])  #.to_dict(orient='records')
            benchmark_df = benchmark_df.reset_index(drop=True)
            benchmark_df.drop('name', 1).sort_values('seed')
            try:
                self.metrics['successrate'] = sum(benchmark_df['success'])/len(benchmark_df['success'])
            except:
                self.metrics['successrate'] = 0.0
        except:
            self.warnings.append('Missing benchmark file!')
        return benchmark_df

    def _get_log(self, log_type='hof'):
        """Read data from log files ('hof' or 'pf')."""
        log_df = None
        log_exists = False
        log_not_found = []
        if 'mc' in self.cmd_params:
            for seed in range(self.cmd_params['mc']):
                log_file = 'dsr_{}_{}_{}.csv'.format(self.exp_config['task']['name'], seed, log_type)
                try:
                    df = pd.read_csv(os.path.join(self.path['log'], log_file))
                    df.insert(0, 'seed', seed)
                    if log_exists:
                        log_df = pd.concat([log_df, df])
                    else:
                        log_df = df.copy()
                        log_exists = True
                except:
                    log_not_found.append(seed)
            try:
                if log_type == 'hof':
                    log_df = log_df.sort_values(by=['r','success','seed'], ascending=False)
                if log_type == 'pf':
                    log_df = self._apply_pareto_filter(log_df)
                    log_df = log_df.sort_values(by=['r','complexity','seed'], ascending=False)
                log_df = log_df.reset_index(drop=True)
                log_df['index'] = log_df.index
            except:
                self.warnings.append('No data for {}!'.format(log_type))
            if len(log_not_found) > 0:
                self.warnings.append('Missing {} files for seeds: {}'.format(log_type, log_not_found))
        else:
            self.warnings.append('Cannot read {} files!'.format(log_type))
        return log_df

    def _apply_pareto_filter(self, df):
        df = df.sort_values(by=['complexity'],ascending=True)
        df = df.reset_index(drop=True)
        filtered_df = pd.DataFrame(columns=list(df))
        for index, row in df.iterrows():
            if not (filtered_df['r'] >= row['r']).any() and \
                    not (filtered_df['complexity'] >= row['complexity']).any() or \
                    index == 0 :
                filtered_df = filtered_df.append(row, ignore_index=True)
        return filtered_df

    def plot_results(self, results, log_type='hof', boxplot_on=False, show_plots=False, safe_plots=False):
        """Plot data from log files ('hof' or 'pf')."""
        col_count = len(self.PLOT_HELPER[log_type]['y'])
        row_count = 2 if boxplot_on else 1
        fig, ax = plt.subplots(row_count, col_count, figsize=(8 * col_count, 4* row_count))
        for i in range(col_count):
            if boxplot_on:
                sns.lineplot(
                    data=results,
                    x=self.PLOT_HELPER[log_type]['x'][i],
                    y=self.PLOT_HELPER[log_type]['y'][i],
                    ax=ax[0, i])
                ax[0, i].set_xlabel(
                    self.PLOT_HELPER[log_type]['x_label'][i])
                ax[0, i].set_ylabel(
                    self.PLOT_HELPER[log_type]['y'][i])
                sns.boxplot(
                    results[self.PLOT_HELPER[log_type]['y'][i]],
                    ax=ax[1, i])
                ax[1, i].set_xlabel(
                    self.PLOT_HELPER[log_type]['y'][i])
            else:
                sns.lineplot(
                    x=results[self.PLOT_HELPER[log_type]['x'][i]],
                    y=results[self.PLOT_HELPER[log_type]['y'][i]],
                    ax=ax[i])
                ax[i].set_xlabel(
                    self.PLOT_HELPER[log_type]['x_label'][i])
                ax[i].set_ylabel(
                    self.PLOT_HELPER[log_type]['y'][i])
        plt.suptitle(
            "{} - {}".format(self.PLOT_HELPER[log_type]['name'], self.exp_config['task']['name']),
            fontsize=14)
        plt.tight_layout()
        if safe_plots:
            safe_path = os.path.join(self.path['log'], "plot_{}.png".format(log_type))
            print('  Saving {} plot to {}'.format(self.PLOT_HELPER[log_type]['name'], safe_path))
            plt.savefig(safe_path)
        if show_plots:
            plt.show()
        plt.close()

    def analyze_log(self, log_count=5, show_hof=True, show_pf=True, show_plots=False, save_plots=False):
        """Generates a summary of important experiment outcomes."""
        print('=== Log analysis =========================================')
        try:
            print('Task_____________{}'.format(self.exp_config['task']['name']))
            print('Log path_________{}'.format(self.path['log']))
            print('Token set________{}'.format(self.exp_config['task']['function_set']))
            print('Runs_____________{}'.format(self.cmd_params['mc']))
            print('Samples/run______{}'.format(self.exp_config['training']['n_samples']))
            if 'successrate' in self.metrics:
                print('Successrate______{}'.format(self.metrics['successrate']))
            if len(self.warnings) > 0:
                print('Found issues_____')
                for i in range(len(self.warnings)):
                    print('  {}'.format(self.warnings[i]))
            if not self.hof_df is None and show_hof:
                print('Hallo of Fame____')
                for i in range(min(log_count,len(self.hof_df.index))):
                    print('  {:3d}: S={:03d} R={:8.6f} <-- {}'.format(
                        i, self.hof_df.iloc[i]['seed'], self.hof_df.iloc[i]['r'],
                        self.hof_df.iloc[i]['expression']))
                if show_plots or save_plots:
                    self.plot_results(
                        self.hof_df, log_type='hof', boxplot_on=True,
                        show_plots=show_plots, safe_plots=save_plots)
            if not self.pf_df is None and show_pf:
                print('Pareto Front_____')
                for i in range(min(log_count,len(self.pf_df.index))):
                    print('  {:3d}: S={:03d} R={:8.6f} C={:03d} <-- {}'.format(
                        i, self.pf_df.iloc[i]['seed'], self.pf_df.iloc[i]['r'],
                        self.pf_df.iloc[i]['complexity'], self.pf_df.iloc[i]['expression']))
                if show_plots or save_plots:
                    self.plot_results(
                        self.pf_df, log_type='pf',
                        show_plots=show_plots, safe_plots=save_plots)
        except:
            print('Error when analyzing!')
            [print('    --> {}'.format(warning)) for warning in self.warnings]
        print('==========================================================\n')


@click.command()
@click.argument('log_path', default=None)
@click.option('--log_count', default=10, type=int, help="Number of results we want to see from each metric.")
@click.option('--show_hof', is_flag=True, help='Show Hall of Fame results.')
@click.option('--show_pf', is_flag=True, help='Show Pareto Front results.')
@click.option('--show_plots', is_flag=True, help='Generate plots and show results as simple plots.')
@click.option('--save_plots', is_flag=True, help='Generate plots and safe to log file as simple plots.')
def main(log_path, log_count, show_hof, show_pf, show_plots, save_plots):
    log = LogEval(log_path)
    log.analyze_log(
        log_count=log_count,
        show_hof=show_hof,
        show_pf=show_pf,
        show_plots=show_plots,
        save_plots=save_plots)

if __name__ == "__main__":
    main()