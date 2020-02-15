
import imp
import os
import socket
import inspect
import sys
import subprocess

# *******************************************************************************************************************
def std_error_print(message,exception):
    print("\n======================================================================")
    print("ERROR: {}".format(message))
    print("----------------------------------------------------------------------\n")
    
    raise exception

# *******************************************************************************************************************
def walltime_to_seconds(walltime):    
    parts   = walltime.split(':')
    
    if len(parts) == 1:
        secs    = int(parts[0]) * 60
    else:
        secs    = int(parts[2]) + int(parts[1])*60 + int(parts[0])*3600
        
    return secs

# *******************************************************************************************************************
# *******************************************************************************************************************
# *******************************************************************************************************************
class host_configure:
    r"""
        Manual database of node properties. 
        
        This is used primarily to formulate job/batch execution on LC
    """
    def __init__(self):
        self.config = {}
        
        # Get interpreter
        self.config['python_bin']           = self._get_interp()
        # Where is your home on LC
        self.config['home_dir']             = os.path.expanduser("~") 
        # User name derived from home dir.
        self.config['user_name']            = self.config['home_dir'].split('/')[-1]
        # Lets keep track of what host we are on
        self.config['host_name']            = socket.gethostname()
        # What is the name of the cluster (remove numbers from host name)
        self.config['cluster_name']         = ''.join([i for i in self.config['host_name'] if not i.isdigit()])
        # Our current working directory
        self.config['working_dir']          = os.getcwd()
        
        # Who called me?
        stack_ret                           = inspect.stack()[1]
        self.config['main_script']          = inspect.getmodule(stack_ret[0]).__file__
        self.config['root_dir']             = os.path.abspath(os.path.dirname(self.config['main_script']))
        
        '''
            These are related to doing recurrent jobs
        '''
        self.config['lsf_run_once']         = True      # Run one job and end (lsf), no recurrent runs
        self.config['msub_do_depend']       = False     # Run recurrent jobs (slurm)
        self.config['lsf_job_verbose']      = True
        self.config['lsf_recursive']        = True
        self.config['lsf_max_dep_jobs']     = 100
        
        self.config['extra_batch_commands'] = None   
        
        self.config['login_nodes']          = []
        self.config['node_num']             = ""
        self.config['job_sched']            = ""
        self.config['walltime']             = ""
        self.config['partition']            = ""
        self.config['account']              = ""
        self.config['lsf_remote_run_cmd']   = ""
        self.config['gpus']                 = []
    
    # *******************************************************************************************************************  
    def _get_interp(self):
    
        # Get ID, processing and memory utilization for all GPUs
        try:
            from subprocess import Popen, PIPE
            p = Popen(["which", "python"], stdout=PIPE)
            stdout, stderror = p.communicate()
        except:
            print("Cannot determine python interpretor")
            return None
        
        output      = stdout.decode('UTF-8')
        lines       = output.split(os.linesep)
        
        return lines[0]

    # *******************************************************************************************************************
    def __call__(self, job_label, account, partition="", projects_dir='projects', figures_dir='figures', log_dir='log'):
        
        assert(isinstance(job_label,str))
        assert(isinstance(account,str))
        
        # Our job_label from function arg
        self.config['job_label']            = job_label
        
        # *******************************************************************************************************************
        # Pascal (TOSS)
        # *******************************************************************************************************************
                   
        if 'pascal' in self.config['host_name']: 
            
            # Where to save your checkpoint models to. Should probably not be home dir since it stores lots of data here.
            # Use workspace or lustre instead (make sure dir exists)
            self.config['proj_base_dir']        = os.path.join("/p/lustre1/",self.config['user_name'],projects_dir,job_label)  
            
            self.config['node_num']             = int(self.config['host_name'][6:])
            self.config['job_sched']            = "slurm"
            self.config['gpus']                 = [0,1]
            self.config['walltime']             = "24:00:00"
            self.config['partition']            = partition
            self.config['account']              = account
            self.config['login_nodes']          = [83]
        
        # *******************************************************************************************************************
        # Surface (TOSS)
        # *******************************************************************************************************************       
        
        elif 'surface' in self.config['host_name']:
            
            # Where to save your checkpoint models to. Should probably not be home dir since it stores lots of data here.
            # Use workspace or lustre instead (make sure dir exists)
            self.config['proj_base_dir']        = os.path.join("/p/lustre1/",self.config['user_name'],projects_dir,job_label)  
            
            self.config['node_num']             = int(self.config['host_name'][7:])
            self.config['job_sched']            = "slurm"
            self.config['login_nodes']          = [86]
            if self.config['node_num'] > 162:
                self.config['gpus']                 = [0,1,2,3]
                self.config['walltime']             = "12:00:00"
                self.config['partition']            = partition
                self.config['account']              = account
            else:
                self.config['gpus']                 = [0,1]        
                self.config['walltime']             = "24:00:00"
                self.config['partition']            = partition
                self.config['account']              = account
            
        # *******************************************************************************************************************
        # Quartz (TOSS)
        # *******************************************************************************************************************    
        
        elif 'quartz' in self.config['host_name']:                                              
            
            # Where to save your checkpoint models to. Should probably not be home dir since it stores lots of data here.
            # Use workspace or lustre instead (make sure dir exists)
            self.config['proj_base_dir']        = os.path.join("/p/lustre1/",self.config['user_name'],projects_dir,job_label)  
            
            self.config['node_num']             = int(self.config['host_name'][6:])
            self.config['job_sched']            = "slurm"
            self.config['gpus']                 = []
            self.config['walltime']             = "24:00:00"
            self.config['partition']            = partition
            self.config['account']              = account
            self.config['login_nodes']          = [188,380,386,764,770,1148,1154,1532,1538,1916,1922,2300,2306,2498]
            
        # *******************************************************************************************************************
        # For RZManta (CORAL)
        # *******************************************************************************************************************
        
        # CORAL (RAY,LASSEN) is a little different and we need to change a few things. 
        # You only need to change these lines if using Ray.
    
        elif 'rzmanta' in config['host_name']:   
            
            # Where to save your checkpoint models to. Should probably not be home dir since it stores lots of data here.
            # Use workspace or lscratch instead (make sure dir exists)
            self.config['proj_base_dir']        = os.path.join("/p/gscratchrzm/",self.config['user_name'],projects_dir,job_label) 
            
            self.config['node_num']             = int(self.config['host_name'][7:])
            self.config['job_sched']            = "lsf"
            self.config['gpus']                 = [0,1,2,3]  
            self.config['walltime']             = "720" # New job.lim
            self.config['partition']            = ""
            self.config['account']              = "guests"
            self.config['login_nodes']          = []

            self.config['lsf_excl_host']        = []
            self.config['lsf_remote_run_cmd']   = "lexec "
            
        # *******************************************************************************************************************
        # For Lassen (CORAL)
        # *******************************************************************************************************************    
               
        elif 'lassen' in config['host_name']:   
            
            # Where to save your checkpoint models to. Should probably not be home dir since it stores lots of data here.
            # Use workspace or lscratch instead (make sure dir exists)
            self.config['proj_base_dir']        = os.path.join("/p/gpfs1/",self.config['user_name'],projects_dir,job_label)    
            
            self.config['node_num']             = int(self.config['host_name'][6:])
            self.config['job_sched']            = "lsf"
            self.config['gpus']                 = [0,1,2,3]  
            self.config['walltime']             = "720"
            self.config['partition']            = ""
            self.config['account']              = account
            self.config['login_nodes']          = [708]

            self.config['lsf_excl_host']        = []
            self.config['lsf_remote_run_cmd']   = "lexec "     
                                                                
        # *******************************************************************************************************************
        # For RZAnsel (CORAL)
        # *******************************************************************************************************************
                    
        elif 'rzansel' in config['host_name']:   
            
            # Where to save your checkpoint models to. Should probably not be home dir since it stores lots of data here.
            # Use workspace or lscratch instead (make sure dir exists)
            self.config['proj_base_dir']        = os.path.join("/p/gpfs1/",self.config['user_name'],projects_dir,job_label)                                                     

            self.config['node_num']             = int(self.config['host_name'][7:])
            self.config['job_sched']            = "lsf"
            self.config['gpus']                 = [0,1,2,3]  
            self.config['walltime']             = "1440" # Using standby, so ask for 24 hour blocks
            self.config['partition']            = "standby"
            self.config['account']              = "guests"
            self.config['login_nodes']          = []

            self.config['lsf_excl_host']        = []
            self.config['lsf_remote_run_cmd']   = "lexec "

        # *******************************************************************************************************************
        # For Ray (CORAL)
        # *******************************************************************************************************************
        
        # CORAL (RAY,LASSEN) is a little different and we need to change a few things. 
        # You only need to change these lines if using Ray.
        
        elif  'ray' in config['host_name']:   
            
            # Where to save your checkpoint models to. Should probably not be home dir since it stores lots of data here.
            # Use workspace or lscratch instead (make sure dir exists)
            self.config['proj_base_dir']        = os.path.join("/p/gscratchr/",self.config['user_name'],projects_dir,job_label)  
            
            self.config['node_num']             = int(self.config['host_name'][3:])
            self.config['job_sched']            = "lsf"
            self.config['gpus']                 = [0,1,2,3]  
            self.config['walltime']             = "720" # New job.lim
            self.config['partition']            = ""
            self.config['account']              = "guests"
            self.config['login_nodes']          = [23]

            self.config['lsf_excl_host']        = []
            self.config['lsf_remote_run_cmd']   = "lexec "
            
        else:
            std_error_print("Unknown host \"{}\"!".format(config['host_name']), NotImplementedError)

            
        # *******************************************************************************************************************   
        # DIRECTORY PARAMETERS
        # *******************************************************************************************************************
        # Where are the job log files?
        self.config['proj_log_dir']             = os.path.join(self.config['proj_base_dir'],log_dir)
            
        # My main startup script. So batch jobs can set the environment you want.  
        self.config['profile']                  = os.path.join(self.config['home_dir'],".profile")
        
        # Where do we save figures and graph outputs
        self.config['figures_dir']              = os.path.join(self.config['home_dir'],figures_dir)
                
        # *******************************************************************************************************************   
        # Create Dirs if needed
        # *******************************************************************************************************************   
        
        if not os.path.isdir(self.config['proj_base_dir']):
            print("**********************************************************************************************************************")
            print("Creating Directory: {}".format(self.config['proj_base_dir']))
            print("**********************************************************************************************************************")
            os.makedirs(self.config['proj_base_dir'])
            
        if not os.path.isdir(self.config['figures_dir']):
            print("**********************************************************************************************************************")
            print("Creating Directory: {}".format(self.config['figures_dir']))
            print("**********************************************************************************************************************")
            os.makedirs(self.config['figures_dir'])
            
        if not os.path.isdir(self.config['proj_log_dir']):
            print("**********************************************************************************************************************")
            print("Creating Directory: {}".format(self.config['proj_log_dir']))
            print("**********************************************************************************************************************")
            os.makedirs(self.config['proj_log_dir'])
       
        # *******************************************************************************************************************   
        # Define common vars and things
        # *******************************************************************************************************************        
        
        if self.config['job_sched'] == "slurm":
        
            self.config['msub_preamble'] = "\
#!/bin/bash\n\
#SBATCH --nodes=1            # use 1 node\n\
#SBATCH --time={}            # ask for n hours\n\
#SBATCH --partition={}       # use the {} partition\n\
#SBATCH --account={}         # use my account\n\
#SBATCH --signal=INT@120     # ask for signal two minutes before out of time\n"\
            .format(self.config['walltime'],self.config['partition'],self.config['partition'],self.config['account'])   
        elif  self.config['job_sched'] == "lsf":  
            self.config['lsf_job_init']                  = "bsub -W {} -G {} -wa SIGHUP -wt 5 ".format(self.config['walltime'],self.config['account'])
        else:
            self.config['msub_preamble'] = "NOT YET DEFINED\n"
            
        return self.config
    
class make_job_file:
    r"""
        Create the job/batch file used to launch a training session.
        
        This is typically done when creating an lc project (see for instance pytorch_tools.lc_project)               
        
    """
    
    def __init__(self, proj_snapshot_base, log_dir, profile, job_label, main_script, config, cmd_line_args=None):
        r"""
    
            Args:
                proj_snapshot_base (string) :
                    This is where your new project will be created
                    
                    example: /p/lscratchh/mundhetn/pytorch_proj//lc_proj.My_Cool_ImageNet_Project_AlexNet
        
                log_dir (string) :
                    This is where job logs will be stored. It can be convenient if this is inside 
                    the proj_snapshot_base directory
                    
                    example:/p/lscratchh/mundhetn/pytorch_proj//lc_proj.My_Cool_ImageNet_Project_AlexNet/log
                     
                profile (string) :
                    This is the full path to a script to run at login such as ~/.profile . This will
                    help you to set your environment on the nodes
                    
                    example: /g/g17/mundhetn/.profile
                    
                job_label (string) :
                    This is the label that gets stampted on everything. Logs will be named with this 
                    string and it will also be used to name jobs on LC.
                    
                    example: My_Cool_ImageNet_Project_AlexNet
                    
                main_script (string) :
                    This is the primary starting (main) python script that gets called from command line
                    Extra leading path variables will be stripped.    
                    
                    example: /g/g17/mundhetn/llama_workspace/PyTorch/project_imagenet_pytorch.py
                                         
                config (dict) :
                    This contains various extra arguments. You can get these from node_props class (see way above)
                    
                    Generic params:
                    
                        config['python_bin']
                        config['walltime']
                    
                    slurm/moab params:
                    
                        config['msub_preamble']
                        config['msub_do_depend']
                        
                    lsf params:
                    
                        config['lsf_max_dep_jobs']
                        config['lsf_job_init']
                        config['lsf_recursive'] 
                        config['lsf_job_verbose']
                        
                cmd_line_args (dict) :
                    This is a list of command line args which need to be appended to the command line. This can be None.
                    
                    example: We might like to append an extra parameter to run remote, so we would give something like:
                    
                        my_batch_script.py --awesome_option=totally_dude
                        
                        So we would then have in this dict:
                        
                            cmd_line_args['awesome_option'] = "totally_dude"
                    
        """
    
        assert(isinstance(config, dict))
        assert(isinstance(job_label, str))
        assert(isinstance(proj_snapshot_base, str))
        assert(isinstance(main_script, str))
        assert('walltime'   in config)
        assert('python_bin' in config)
        assert('job_sched'  in config)
            
        self.config                 = config
        self.job_label              = job_label
        self.proj_snapshot_base     = proj_snapshot_base
        self.main_script            = main_script
        self.walltime_secs          = walltime_to_seconds(config['walltime'])
        self.runtime_secs           = self.walltime_secs - 120  # Give us 2 minutes to quit
        self.batch_run_line         = None
        self.cmd_line_args          = cmd_line_args
        
        assert(self.runtime_secs > 0)
        
        lparts  = self.main_script.split('/') 

        if len(lparts) > 1:
            self.script_suffix          = lparts[-2] + "/" + lparts[-1]
        else:
            pre_dir                     = os.path.split(os.path.abspath(os.path.dirname(self.main_script)))
            self.script_suffix          = pre_dir[-1] + "/" + lparts[0]

        self.log_file               = log_dir               + "/lc_job." + job_label + ".log"
        self.tee_file               = log_dir               + "/lc_job." + job_label + ".$hostname.$day.log"
        self.easy_run_line          ="{} {}/{} {}".format(config['python_bin'], proj_snapshot_base, self.script_suffix, self.cmd_line_args)
        self.run_line               = "unbuffer {} --is_batch_job=1 2>&1 | tee -a $logfile &".format(self.easy_run_line)
        self.profile_line           = "source {}\n".format(profile)
        self.msub_file              = ""
        self.script_file            = proj_snapshot_base    + "/run.sh"
        
        self.tee_log_def            = "\
# Set up a redirect log file to catch program output\n\
# We use two logging methods because both seem to have their own bugs\n\
day=$(date +%A.%b.%d.%y)\n\
hostname=$(hostname -s)\n\
logfile=\"{}\"\n\n".format(self.tee_file)

        r'''
            Work around for Slurm/Moab signal which doesn't work in a useful way
            
            This will send SIGUSR1 to all python scripts on the machine 120 seconds 
            before we run out of time. 
        '''

        self.quit_on_time           = "\n\n\
# Send SIGUSR1 to all python processes after {} seconds\n\
# This lets out program know that it is about out of time on the node.\n\
# We do this because using \"SBATCH --signal\" seems unreliable.\n\
echo \"Script waiting\"\n\
sleep {}s\n\
killall -10 -w python\n\
sleep 110s\n\
echo \"Script done\"\n\n".format(self.runtime_secs, self.runtime_secs)


        if self.config['job_sched'] == 'slurm':
            self._create_sbatch()
        elif self.config['job_sched'] == 'lsf':
            self._create_lsf()
        else:
            std_error_print("Unknown batching system \"{}\"!".format(config['job_sched']), NotImplementedError)
            
        self._create_run_script()

    # *******************************************************************************************************************   
    def _create_run_script(self):
        
        # Write simple run script
        File            = open(self.script_file,'w')
        File.write("{}\n".format(self.easy_run_line))
        File.close()
        
        
    # *******************************************************************************************************************
    def _create_sbatch(self):
        r""" 
            Create a moab/slurm compatible batch/job script
        """
            
        self.msub_file  = self.proj_snapshot_base    + "/slurm.sbatch"
        
        File  = open(self.msub_file,'w')
    
        File.write(self.config['msub_preamble'])
    
        job_line        = "#SBATCH -J {} # user defined job name\n".format(self.job_label)
        File.write(job_line)
        log_line        = "#SBATCH --output {} # user defined job log file\n".format(self.log_file)
        File.write(log_line)        
        log_line        = "#SBATCH --error {} # user defined job log file\n".format(self.log_file)
        File.write(log_line)
        
        #print(self.config)
        
        if self.config['extra_batch_commands'] is not None:
            File.write(self.config['extra_batch_commands'])
            
        
        more_lines      = "\
\n\
# print message that a new run is starting\n\
echo \"Starting new run: $SLURM_JOBID\"\n\
date\n\
\n"
    
        File.write(more_lines)
        
        File.write(self.tee_log_def)
        
        if self.config['msub_do_depend'] == True:
            more_lines      = "\
# to create a chain of dependent jobs (optional)\n\
echo \"Submitting dependent job\"\n"
        
            File.write(more_lines)
        
            depend_line     = "sbatch --dependency=afterany:$SLURM_JOBID {}\n".format(self.msub_file)
            File.write(depend_line)
        
        more_lines = "# User profile to set environment\n"
        
        File.write(more_lines)
        
        File.write(self.profile_line)
        
        more_lines = "# Run our actual scripts, use unbuffer to make sure any display output is collected.\n"
        
        File.write(more_lines)
        
        File.write(self.run_line)
        
        File.write(self.quit_on_time)
        
        self.batch_run_line = "sbatch {}".format(self.msub_file)
    
        print("**********************************************************************************************************************")
        print("NOTE: Run me stand alone as: {}".format(self.easy_run_line))
        print("OR:   Run me stand alone as: sh {}".format(self.script_file))
        print("OR:   Run me as a batch as: {}".format(self.batch_run_line))
        print("**********************************************************************************************************************")
    
        File.close()           
        
    # *******************************************************************************************************************
    def _create_lsf(self):
        r"""
            Create an LSF (IBM/CORAL) compatible batch/job script
        """
        
        self.msub_file          = self.proj_snapshot_base    + "/lsf.cmd"
        
        File                    = open(self.msub_file,'w')
        
        proj_name               = self.job_label[:64] if len(self.job_label) > 64 else self.job_label
        
        ##############################################
        #### Main calling script, sets up job chains
        
        # this job_labels jobs to start when entire job_label jobs have ended
        if 'lsf_prior_start_job_label' in self.config:
            prior_start_job_label = "        -w ended({}) \\\n".format(self.config['lsf_prior_start_job_label'])
        else:
            prior_start_job_label = ""
        
        if self.config['lsf_run_once']:
            self.config['lsf_recursive'] = False
        
        File.write("#!/bin/bash\n")
        File.write("\n")
        File.write("# Nate-o-matic LFS Job Script Generator\n")
        File.write("#\n")
        File.write("# \"It\'s the Shine That Makes it Sparkle!\"\n")
        File.write("#\n")
        File.write("# Copyright 1948, Acme and Sons Scripting Company\n")
        File.write("# Wallawalla, Washington\n")
        File.write("# All rights reserved\n")
        File.write("\n")
        File.write("Max={}\n".format(self.config['lsf_max_dep_jobs']))
        File.write("\n")
        # If we got an argument, this must be a dependant job
        File.write("if [ $# -eq 1 ] ; then\n") 
        File.write("    Last=$1\n")
        File.write("    Curr=$(($Last + 1))\n")
        File.write("    CMD=\"{} \\\n".format(self.config['lsf_job_init']))
        if len(self.config['partition']) > 0:
            File.write("        -q \"{}\"\n".format(self.config['partition']))
        for cf in self.config['lsf_excl_host']:
            File.write("        -R \"select[hname!={}]\" \\\n".format(cf))
        File.write("        -w ended({}_$Last) \\\n".format(self.job_label))
        File.write("        -J {}_$Curr\"\n".format(self.job_label))   

        # First job in the chain 
        File.write("else\n")
        File.write("    Curr=0\n")
        File.write("    CMD=\"{} \\\n".format(self.config['lsf_job_init']))     
        if len(self.config['partition']) > 0:
            File.write("        -q \"{}\"\n".format(self.config['partition']))     
        for cf in self.config['lsf_excl_host']:
            File.write("        -R \"select[hname!={}]\" \\\n".format(cf))
        File.write(prior_start_job_label) # Will not be empty if we have a special job_label to run first
        File.write("        -J {}_0\"\n".format(self.job_label))  

        File.write("fi\n")
        File.write("\n")
        
        # We can either create a job chain of finite size, of an infinity self calling recursion. 
        if self.config['lsf_recursive'] == False:
            File.write("CMD_2=\"$CMD sh {}/lsf.sh\"\n".format(self.proj_snapshot_base))
        else:
            File.write("CMD_2=\"$CMD sh {}/lsf.sh $Curr\"\n".format(self.proj_snapshot_base))
        
        File.write("\n")
        if self.config['lsf_job_verbose'] == True:
            File.write("echo \"***********************************************\"\n")
            File.write("echo \"STARTING: $Curr\"\n")
            File.write("echo $CMD_2\n")
            File.write("echo \"-----------------------------------------------\"\n")
        
        if self.config['lsf_recursive'] == False and not self.config['lsf_run_once']:   
            File.write("if [ \"$Curr\" -lt \"$Max\" ] ; then\n")
            File.write("    $CMD_2 \n")
            File.write("    sh {}/lsf.cmd $Curr &\n".format(self.proj_snapshot_base))
            File.write("fi\n")
        else:
            File.write("$CMD_2 \n")
            
        if self.config['lsf_job_verbose'] == True:
            File.write("echo \"***********************************************\"\n")
        
        File.close()
        
        sh_file                 = self.proj_snapshot_base    + "/lsf.sh"
        
        ##############################################
        #### Secondary script that executes job, but then calls the main script if using infinite recursion. 
        
        File                    = open(sh_file,'w')
        
        File.write("#!/bin/bash\n")
        
        more_lines      = "\
# print message that a new run is starting\n\
echo \"Starting new run: $LSB_JOBID\"\n\
date\n"
        File.write(more_lines)
        File.write(self.profile_line)
        File.write(self.tee_log_def)
        
        if self.config['lsf_recursive'] == True:
            File.write("sh {}/lsf.cmd $1 &\n".format(self.proj_snapshot_base))
        
        more_lines = "# Run our actual scripts, use unbuffer to make sure any display output is collected.\n"
        
        File.write(more_lines)
           
        File.write("echo \"STARTING...\" | tee -a $logfile &\n")
        #File.write("/bin/killall -9 python\n")                      # Added for Lassen since it doesn't kill your jobs
        File.write("{} {}\n".format(self.config['lsf_remote_run_cmd'],self.run_line))
            
        '''
        File.write("CMD=\"{}\"\n".format(self.run_line))
        File.write("echo $CMD\n")
        File.write("$CMD\n")
        '''
        
        self.batch_run_line = "source {}".format(self.msub_file)
        
        print("**********************************************************************************************************************")
        print("NOTE: Run me stand alone as: {}".format(self.easy_run_line))
        print("OR:   Run me stand alone as: sh {}".format(self.script_file))
        print("OR:   Run me as a batch as: {}".format(self.batch_run_line))
        print("**********************************************************************************************************************")
        
        File.close()
        
# *******************************************************************************************************************
# *******************************************************************************************************************
# *******************************************************************************************************************
class create_project:
    r"""
        Create a LLNL LC computing style project for use on HPC. This will take your script and make a copy of
        if into a specified directory. We will also store checkpoints there and logs. 
        
        This allows one to create many different projects and run them in parallel on different nodes.
    """
    def __init__(self, config, cmd_line_args=None, auto_run_batch=False):
        
        assert(isinstance(config,dict))
        
        self.cmd_line_args      = cmd_line_args
        self.auto_run_batch     = auto_run_batch
        self.config             = config
                        
        self.batch_run_line     = None
        
        self.init_file          = os.path.join(self.config['proj_base_dir'], "init.done")
        
    # *******************************************************************************************************************
    def _save_project_state(self, proj_base_dir, root_dir):
                       
        r'''    
            use rsync so we can exclude hidden files such as all the git stuff
        '''
            
        cmd = "rsync -aL --exclude=\".*\" " + root_dir + " " + proj_base_dir
    
        try:
            subprocess.check_call(cmd, shell=True)
        except Exception as e:
            print('Cannot copy project to {} in line {}'.format(proj_base_dir, sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
    
    # *******************************************************************************************************************    
    def __call__(self, re_init=False):
        
        r'''
            force a manual re-init of the project. 
        '''
        if re_init:
            cmd = "rm -Rf {}".format(self.config['proj_base_dir'])
            
            try:
                subprocess.check_call(cmd, shell=True)
            except Exception as e:
                print('Cannot remove project {} in line {}'.format(self.config['proj_base_dir'], sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        
        r'''
            project already exists?
        '''    
        if not os.path.isfile(self.init_file):
            
            r'''
                Create the batch job execution file
            '''
            #np              = create_job.node_props()
            #self.node_props = np.get(extern_config=extern_config)
            
            mjf             = make_job_file(self.config['proj_base_dir'], self.config['proj_log_dir'], self.config['profile'], self.config['job_label'], 
                                            self.config['main_script'], self.config, self.cmd_line_args)
                        
            self.batch_run_line = mjf.batch_run_line
            
            r'''
                Get a copy of our scripts to run
            '''
            self._save_project_state(self.config['proj_base_dir'], self.config['root_dir'])
            
            r'''
                Note we are done with a special file
            '''
            cmd = "touch {}".format(self.init_file)
            
            try:
                subprocess.check_call(cmd, shell=True)
            except Exception as e:
                print('Cannot set up project {} in line {}'.format(self.config['proj_base_dir'], sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
                
            r'''
                This will auto launch the job on a node once the job has been created.
            '''
            
            if self.auto_run_batch == True:
                print("**********************************************************************************************************************")
                try:
                    subprocess.check_call(self.batch_run_line, shell=True)
                    print('NOTE: Auto launched batch job: {}'.format(self.batch_run_line))
                except Exception as e:
                    print('ERROR: Cannot auto launch batch job: {}'.format(self.batch_run_line))
                print("**********************************************************************************************************************")
            return True
        
        else:
            
            print("**********************************************************************************************************************")
            print("NOTE: Running project directory: " + self.config['proj_base_dir'])
            print("**********************************************************************************************************************")
        
            return False 
