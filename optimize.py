from asyncio.windows_events import NULL
from pathlib import Path

from cv2 import threshold
import optimizers.PSO as pso
import optimizers.MVO as mvo
import optimizers.GWO as gwo
import optimizers.GWO_copy as gwo_copy
import optimizers.MFO as mfo
import optimizers.CS as cs
import optimizers.BAT as bat
import optimizers.WOA as woa
import optimizers.FFA as ffa
import optimizers.SSA as ssa
import optimizers.GA as ga
import optimizers.HHO as hho
import optimizers.HHOMP as hhomp
import optimizers.SCA as sca
import optimizers.JAYA as jaya
import optimizers.DE as de
import optimizers.HHO_copy as hho_copy
import optimizers.HHO_copy2 as hho_copy2
import optimizers.GROM as grom
import optimizers.MROM as mrom
import optimizers.BBO as bbo
import optimizers.CCO as cco
import optimizers.COVIDHHO as covidhho
import optimizers.MROM_SCA as mrom_sca
import optimizers.TSA as tsa
import optimizers.RLCO as rlco
import optimizers.WOA_SCA_GWO as woa_sca_gwo
import optimizers.SEO as seo 
import optimizers.AHA as aha
import optimizers.AHA_L as aha_l
import benchmarks
import csv
import numpy
import time
import warnings
import os

warnings.simplefilter(action="ignore")


def selector(algo, func_details, popSize, Iter ,dim):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]

    if algo == "SSA":
        x = ssa.SSA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "PSO":
        x = pso.PSO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "GA":
        x = ga.GA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "BAT":
        x = bat.BAT(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "FFA":
        x = ffa.FFA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "GWO":
        x = gwo.GWO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "BBO":
        x = bbo.BBO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "WOA":
        x = woa.WOA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "MVO":
        x = mvo.MVO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "MFO":
        x = mfo.MFO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "CS":
        x = cs.CS(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "HHO":
        x = hho.HHO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "HHOMP":
        x = hhomp.HHOMP(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "SCA":
        x = sca.SCA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "JAYA":
        x = jaya.JAYA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "DE":
        x = de.DE(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "HHO_copy":
        x = hho_copy.HHO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "HHO_copy2":
        x = hho_copy2.HHO_copy2(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "GROM":
        x = grom.GROM(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "MROM":
        x = mrom.MROM(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "CCO":
        x = cco.CCO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "COVIDHHO":
        x = covidhho.COVIDHHO2(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "GWO_copy":
        x = gwo_copy.GWO_copy(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "MROM_SCA":
        x = mrom_sca.MROM_SCA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "TSA":
        x = tsa.TSA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "RLCO":
        x = rlco.RLCO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "WOA_SCA_GWO":
        x = woa_sca_gwo.WOA_SCA_GWO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "SEO":
        x = seo.SEO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "AHA":
        x = aha.AHA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "AHA_L":
        x = aha_l.AHA_L(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    else:
        return NULL
    return x


def run(optimizer, objectivefunc, NumOfRuns, params, export_flags):

    """
    It serves as the main interface of the framework for running the experiments.

    Parameters
    ----------
    optimizer : list
        The list of optimizers names
    objectivefunc : list
        The list of benchmark functions
    NumOfRuns : int
        The number of independent runs
    params  : set
        The set of parameters which are:
        1. Size of population (PopulationSize)
        2. The number of iterations (Iterations)
    export_flags : set
        The set of Boolean flags which are:
        1. Export (Exporting the results in a file)
        2. Export_details (Exporting the detailed results in files)
        3. Export_convergence (Exporting the covergence plots)
        4. Export_boxplot (Exporting the box plots)

    Returns
    -----------
    N/A
    """

    # Select general parameters for all optimizers (population size, number of iterations) ....
    PopulationSize = params["PopulationSize"]
    Iterations = params["Iterations"]

    # Export results ?
    Export = export_flags["Export_avg"]
    Export_details = export_flags["Export_details"]
    Export_convergence = export_flags["Export_convergence"]
    Export_boxplot = export_flags["Export_boxplot"]

    Flag = False
    Flag_details = False

    # CSV Header for for the cinvergence
    CnvgHeader = []

    results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    for l in range(0, Iterations):
        CnvgHeader.append("Iter" + str(l + 1))

    for l in range(0, Iterations):
        CnvgHeader.append("Iterpsnr" + str(l + 1))

    for l in range(0, Iterations):
        CnvgHeader.append("Iterssim" + str(l + 1))

    for l in range(0, Iterations):
        CnvgHeader.append("Iterfsim" + str(l + 1))

    for l in range(0, Iterations):
        CnvgHeader.append("Iterncc" + str(l + 1))

    for i in range(0, len(optimizer)):
        for j in range(0, len(objectivefunc)):
            convergence = [0] * NumOfRuns
            psnr = [0] * NumOfRuns
            ssim = [0] * NumOfRuns
            fsim = [0] * NumOfRuns
            ncc = [0] * NumOfRuns
            executionTime = [0] * NumOfRuns
            threshold= [2,3,5,7,8,10,12,13,15,17,18,20]
            for dim in threshold:
                for k in range(0, NumOfRuns):
                    func_details = benchmarks.getFunctionDetails(objectivefunc[j])
                    x = selector(optimizer[i], func_details, PopulationSize, Iterations ,dim)
                    convergence[k] = x.convergence
                    psnr[k] = x.psnr
                    ssim[k] = x.ssim
                    fsim[k] = x.fsim
                    ncc[k] = x.ncc
                    optimizerName = x.optimizer
                    objfname = x.objfname
                    if Export_details == True:
                        ExportToFile = results_directory + "experiment_details.csv"
                        with open(ExportToFile, "a", newline="\n") as out:
                            writer = csv.writer(out, delimiter=",")
                            if (
                                Flag_details == False
                            ):  # just one time to write the header of the CSV file
                                header = numpy.concatenate(
                                    [["Optimizer", "dim","objfname", "ExecutionTime"], CnvgHeader]
                                )
                                writer.writerow(header)
                                Flag_details = True  # at least one experiment
                            executionTime[k] = x.executionTime
                            a = numpy.concatenate(
                                [[x.optimizer,dim, x.objfname, x.executionTime], x.convergence, x.psnr, x.ssim, x.fsim, x.ncc]
                            )
                            writer.writerow(a)
                        out.close()

                if Export == True:
                    ExportToFile = results_directory + "experiment.csv"

                    with open(ExportToFile, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if (
                            Flag == False
                        ):  # just one time to write the header of the CSV file
                            header = numpy.concatenate(
                                [["Optimizer", "dim", "objfname", "ExecutionTime"], CnvgHeader]
                            )
                            writer.writerow(header)
                            Flag = True

                        avgExecutionTime = float("%0.2f" % (sum(executionTime) / NumOfRuns))
                        avgConvergence = numpy.around(
                            numpy.mean(convergence, axis=0, dtype=numpy.float64), decimals=2
                        ).tolist()
                        avgpsnr = numpy.around(
                            numpy.mean(psnr, axis=0, dtype=numpy.float64), decimals=2
                        ).tolist()
                        avgssim = numpy.around(
                            numpy.mean(ssim, axis=0, dtype=numpy.float64), decimals=2
                        ).tolist()
                        avgfsim = numpy.around(
                            numpy.mean(convergence, axis=0, dtype=numpy.float64), decimals=2
                        ).tolist()
                        avgncc = numpy.around(
                            numpy.mean(convergence, axis=0, dtype=numpy.float64), decimals=2
                        ).tolist()
                        a = numpy.concatenate(
                            [[optimizerName,dim, objfname, avgExecutionTime], avgConvergence,avgpsnr, avgssim, avgpsn]
                        )
                        writer.writerow(a)
                    out.close()

    if Export_convergence == True:
        print(optimizer)
        conv_plot.run(results_directory, optimizer, objectivefunc, Iterations,dim)

    if Export_boxplot == True:
        box_plot.run(results_directory, optimizer, objectivefunc, Iterations,dim)

    if Flag == False:  # Faild to run at least one experiment
        print(
            "No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions"
        )

    print("Execution completed")
