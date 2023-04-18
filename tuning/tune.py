#! /usr/bin/env python3
"""
RUN THIS FROM THE NIX-DEVELOP GPTUNE ENVIRONMENT.
Edit path value to GPtune file path.

python tune.py -nrun 501 -machine a25 -obj cpu_time

Example of invocation of this script:
mpirun -n 1 python superlu_MLA.py -nprocmin_pernode 1 -ntask 20 -nrun 800 -obj time -tla 0

where:
    -nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
    -obj is the tuning objective: "time" or "memory"
    -tla is whether to perform TLA after MLA
"""

################################################################################

import sys
import os
import numpy as np
import argparse
import subprocess
import json
import pickle
import math
import mpi4py

from array import array
import math

sys.path.insert(0, os.path.abspath("/home/grace/GPTune"))

from gptune import * # import all

from autotune.problem import *
from autotune.space import *
from autotune.search import *

from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import math

# should always use this name for user-defined objective function
def objectives(point):
    
    # ------------------ constants defined in TuningProblem ------------------ #
    matrix_size = int(point['matrix_size'])
    cores = point['cores']
    mhz_per_cpu = point['mhz_per_cpu']
    target = point['target']
    assert target in {'real_time', 'cpu_time', 'flops'}
    
    # --------------------------- tuning paramaters -------------------------- #
    
    #tune over power-of-2 blocks only.
    convertToBlockSize = lambda x: int(2 ** round(x))

    dgemm_m_blk = convertToBlockSize(point['dgemm_m_blk'])
    dgemm_n_blk = convertToBlockSize(point['dgemm_n_blk'])
    dgemm_k_blk = convertToBlockSize(point['dgemm_k_blk'])
    dgemm_m_reg = convertToBlockSize(point['dgemm_m_reg'])
    dgemm_n_reg = convertToBlockSize(point['dgemm_n_reg'])

    # this is just for a nice printout...
    # don't worry about what's inside - it's just just 
    params = {'target':target,
              'matrix_size':matrix_size,
              'dgemm_m_blk':dgemm_m_blk,
              'dgemm_n_blk':dgemm_n_blk,
              'dgemm_k_blk':dgemm_k_blk,
              'dgemm_m_reg':dgemm_m_reg,
              'dgemm_n_reg':dgemm_n_reg,
            }

    # ----------------------- actually run the program ----------------------- #

    RUNDIR = os.path.abspath(__file__ + "/../")
    BENCHSCRIPT = os.path.abspath(__file__ + "/../compile_and_run_sgemm.sh")

    bench = subprocess.run([BENCHSCRIPT, str(matrix_size), str(dgemm_m_blk), str(dgemm_n_blk), str(dgemm_k_blk), str(dgemm_m_reg), str(dgemm_n_reg) ], capture_output=True)
    # error in the code often means an invalid configuration
    # in this case, return something huge
    # to prevent it from being chosen
    # [TODO] there's probably a 'best practice' for this... ask GPT ppl
    # doing this may cause the bias the tuner
    if bench.returncode != 0:
        print(params,
              '\nExecution error: objective set to 1e30',
              '\nError msg: ',
              bench.stderr.decode()
             )
        return [1e30]

    #the 0 is there because we only run one benchmark per run
    #so we only look at the first and only result
    objective = json.loads(bench.stdout)['benchmarks'][0][target]

    
    #GPTune is a minimizer, and we want to maximize flops
    if(target=='flops'):
        objective = -objective

    if target == 'flops':
        objective_prettyprinted = str(round(-objective/1e9)) + "Gflop/s"
    else:
        objective_prettyprinted = str(round(objective)) + "ns"

    print(params, '\n',
          target, ': ', objective_prettyprinted, '\n'
         )

    return [objective] 


# ---------------------------------------------------------------------------- #
#                       constraints for the tuning points                      #
# ---------------------------------------------------------------------------- #

def constraint_1(dgemm_n_blk, dgemm_n_reg):
    return dgemm_n_reg <= dgemm_n_blk
def constraint_2(dgemm_m_blk, dgemm_m_reg):
    return dgemm_m_reg <= dgemm_m_blk
def constraint_3(dgemm_m_blk, matrix_size):
    return dgemm_m_blk <= int(math.log(matrix_size, 2))
def constraint_4(dgemm_n_blk, matrix_size):
    return dgemm_n_blk <= int(math.log(matrix_size, 2))
def constraint_5(dgemm_k_blk, matrix_size):
    return dgemm_k_blk <= int(math.log(matrix_size, 2))
def constraint_6(dgemm_m_reg, matrix_size):
    return dgemm_m_reg <= int(math.log(matrix_size, 2))
def constraint_7(dgemm_n_reg, matrix_size):
    return dgemm_n_reg <= int(math.log(matrix_size, 2))

# ---------------------------------------------------------------------------- #
#                         Tuner configuration/execution                        #
# ---------------------------------------------------------------------------- #

def main():

    # --------------------- Parse command line arguments --------------------- #
    args   = parse_args()
    tla          = args.tla           # transfer learning?
    ntask        = args.ntask         # number of tasks
    TUNER_NAME   = args.optimization  # tuner to use: opentuner, hpbandster, GPTune?
    nrun         = args.nrun          # number of calls per task 
    obj          = args.obj           # objective to optimize

    target       = obj

    # TODO (low priority): figure out if this does anything
    json_handle = open(os.path.abspath(__file__ + "/../.gptune/meta.json"), "r")
    tuning_metadata = json.load(json_handle)
    json_handle.close()
    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " nodes: " + str(nodes) + " cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    # ----------------------------- TUNING SPACE ----------------------------- #

    # ======== (IS): task parameter space (e.g. problem size) =========
    # matrix_sizes = ["64", "192", "221", "256", "320", "397", "412", "448",
    #                 "512", "576", "704", "732", "832", "911", "960", "1024",
    #                 "1088", "1216", "1344", "1472", "1600", "1728", "1856",
    #                 "1984", "2048"]

    # matrix_size = Categoricalnorm (matrix_sizes, transform="onehot", name="matrix_size")     
    matrix_size = Integer(1,1e6, transform="identity", name="matrix_size")
    IS = Space([matrix_size])

    # ======== (PS): tuning parameter space (e.g. block sizes) =========
    
    dgemm_m_blk = Integer(3, 13, transform = "normalize", name = "dgemm_m_blk")
    dgemm_n_blk = Integer(3, 13, transform = "normalize", name = "dgemm_n_blk")
    dgemm_k_blk = Integer(3, 13, transform = "normalize", name = "dgemm_k_blk")
    dgemm_m_reg = Integer(3, 13, transform = "normalize", name = "dgemm_m_reg")
    dgemm_n_reg = Integer(3, 13, transform = "normalize", name = "dgemm_n_reg")
    
    PS = Space([ dgemm_m_blk, dgemm_n_blk, dgemm_k_blk, dgemm_m_reg, dgemm_n_reg])
    # ======== (OS): Output space, just runtime or flops =========
    result = Real(float("-Inf"), float("Inf"), name=target)
    OS = Space([result])

    # ======== constraints on the parameters =========
    constraints = {"constraint_1":constraint_1,
                   "constraint_2":constraint_2,
                   "constraint_3":constraint_3,
                   "constraint_4":constraint_4,
                   "constraint_5":constraint_5,
                   "constraint_6":constraint_6,
                   "constraint_7":constraint_7}

    # ======== available performance models =========
    models = {}
    
    # ======== global constants =========
    # FIXME (low priority): improve this to take data from benchmark
    constants={"nodes":1, "cores":6, "mhz_per_cpu":2800, "target":target}
    
    # ======== Print all input and parameter samples =========
    print(IS, PS, OS, constraints, models, constants)

    problem = TuningProblem(IS, PS, OS, objectives, constraints, None, constants=constants)
    computer = Computer(nodes = 1, cores = 1, hosts = None)  

    # ----------------------- Set and validate options ----------------------- #

    options = Options()
    options['model_processes'] = 1
    # options['model_threads'] = 1
    options['model_restarts'] = 1
    # options['search_multitask_processes'] = 1
    # options['model_restart_processes'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    options['model_class'] = 'Model_GPy_LCM'# 'Model_GPy_LCM' # 'Model_LCM'
    options['verbose'] = False
    options['oversubscribe'] = True
    options.validate(computer = computer)

    historydb = HistoryDB(meta_dict=tuning_metadata)

    # """ Building MLA with the given list of tasks """
    # giventask = [[np.random.choice(matrices,size=1)[0]] for i in range(ntask)]
    # matrix_sizes = ["64", "192", "221", "256", "320", "397", "412", "448",
    #                 "512", "576", "704", "732", "832", "911", "960", "1024",
    #                 "1088", "1216", "1344", "1472", "1600", "1728", "1856",
    #                 "1984", "2048"]
    giventask = [ [4096] ]# [64], [192], [256], [732], [1024], [1600], [2048] ]#, ["911"] ]
    data = Data(problem)

    # ------------------------------- RUN TUNER ------------------------------ #

    if(TUNER_NAME=='GPTune'):
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))        
        
        NI = len(giventask)
        NS = nrun
        (data, model, stats) = gt.MLA(NS=NS, NI=NI, Tgiven=giventask, NS1=max(NS//2, 1))
        print("stats: ", stats)

        """ Print all input and parameter samples """	
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    matrix:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

        if(tla==1):
            """ Call TLA for a new task using the constructed LCM model"""    
            newtask = [["big.rua"]]
            # newtask = [["H2O.rb"]]
            (aprxopts,objval,stats) = gt.TLA1(newtask, NS=None)
            print("stats: ",stats)

            """ Print the optimal parameters and function evaluations"""	
            for tid in range(len(newtask)):
                print("new task: %s"%(newtask[tid]))
                print('    predicted Popt: ', aprxopts[tid], ' objval: ',objval[tid]) 	

    if(TUNER_NAME=='opentuner'):
        NI = len(giventask)
        NS = nrun
        (data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    matrix:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        NI = len(giventask)
        NS = nrun
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    matrix:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='cgp'):
        from callcgp import cGP
        NI = len(giventask)
        NS = nrun
        options['EXAMPLE_NAME_CGP']='SuperLU_DIST'
        options['N_PILOT_CGP']=int(NS/2)
        options['N_SEQUENTIAL_CGP']=NS-options['N_PILOT_CGP']
        (data,stats)=cGP(T=giventask, tp=problem, computer=computer, options=options, run_id="cGP")
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    matrix:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))



def parse_args():

    parser = argparse.ArgumentParser()

    # Problem related arguments
    parser.add_argument('-obj', type=str, default='cpu_time', help='Tuning objective (time or memory)')	
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
    parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
    # Algorithm related arguments
    parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, help='Number of runs per task')
    parser.add_argument('-tla', type=int, default=0, help='Whether to perform TLA after MLA')

    args   = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
