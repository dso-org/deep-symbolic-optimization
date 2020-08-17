#!/usr/bin/env wolframscript

(* 
Description:
  Evaluates FindFormula algorithm with 100 MC seeds over the 1D tests in:
  Uy,  N. Q.,  Hoai,  N. X.,  ONeill,  M.,  McKay,  R. I.,  and Galvan-Lopez, E.  
  Semantically-based crossover in genetic programming: application to real-valued 
  symbolic regression. Genetic Programming and Evolvable Machines, 12(2):91–119, 2011. 
  
  Needs Mathematica 12.0.0. Runs in parallel over the number of kernels available.

Options:
  RUNDEFAULTS : Boolean flag to run FindFormula with the default values. If 
                RUNDEFAULTS=False, FindFormula is evaluated with the options
                PerformanceGoal-> "Quality",
                SpecificityGoal-> Infinity,
                TargetFunctions-> {Plus,Times,Power,Sin,Cos,Exp,Log}
  NOISE : Boolean flag to add gaussian noise to data.
  STD : If NOISE=True, standard deviation of noise.

To run: 
  (In Linux/LC) wolfram-12.0.0 -script wolfram_nguyen_baseline.wls
  (In MAC) wolframscript -script wolfram_nguyen_baseline.wls

Output:
  OUTPUTPATH is by default this ./wolfram_nguyen_baseline
*)

DSRCODEPATH="../../..";
OUTPUTPATH="./wolfram_nguyen_baseline";

TESTS={"Nguyen-1","Nguyen-2","Nguyen-3","Nguyen-4","Nguyen-5","Nguyen-6","Nguyen-7","Nguyen-8"}; (*1D tests*)
MC=100;
VERBOSE=True;
RUNDEFAULTS=False;
NOISE=False;
STD=0.01;


LIBRARY={Plus,Times,Power,Sin,Cos,Exp,Log};(*Function lib: Koza,"add,sub,mul,div,sin,cos,exp,log"*)
DSRREGRESSIONPATH=DSRCODEPATH<>"/dsr/dsr/task/regression/data"
RULESODDEXP={
Abs[x]^0.5-> x^0.5 ,
Sqrt[Abs[x]]-> x^0.5,
(-x)^2.->  x^2.,(-1. x)^2.->  x^2., Abs[x]^2.-> x^2.,
(-x)^4.->  x^4.,(-1. x)^4.->  x^4., Abs[x]^4.-> x^4.,
(-x)^6.->  x^6.,(-1. x)^6.->  x^6., Abs[x]^6.-> x^6.
};
RULESFLOATEXP={x^1.->  x^1,x^2.->  x^2,x^3.->  x^3,x^4.->  x^4,x^4.->  x^4,x^5.->  x^5,x^6.->  x^6};

iso = DateString["ISODateTime"];
TIMESTAMP = StringJoin[StringSplit[StringSplit[iso, "T"][[1]], "-"]]<>"-"<>StringJoin[StringSplit[StringSplit[iso, "T"][[2]], ":"]];
NAMEDIR=OUTPUTPATH<>"/log_"<>"default_values_"<>ToString[RUNDEFAULTS];
If[NOISE,
NAMEDIR=NAMEDIR<>"_"<>"noise_"<>ToString[NOISE]<>"_"<>"std_"<>ToString[STD];
]
NAMEDIR=NAMEDIR<>"_"<>TIMESTAMP;
dir = CreateDirectory[NAMEDIR];
Print["Log directory : "<>dir];

NguyenDict=<|
"Nguyen-1"->(1.0#^3+1.0#^2+1.0# &),
"Nguyen-2"->(1.0#^4+1.0#^3+1.0#^2+1.0#&),
"Nguyen-3"->(1.0#^5+1.0#^4+1.0#^3+1.0#^2+1.0#&),
"Nguyen-4"->(1.0#^6+1.0#^5+1.0#^4+1.0#^3+1.0#^2+1.0#&),
"Nguyen-5"->(1.0Sin[#^2.]Cos[#]-1.0&),
"Nguyen-6"->(1.0Sin[#]+1.0Sin[#+#^2.]&),
"Nguyen-7"->(1.0Log[#+1.0]+1.0Log[#^2+1.0]&),
"Nguyen-8"->(1.0Sqrt[#]&)
|>;

Print["  Parallel kernels : " <> ToString[Length[ParallelEvaluate[$KernelID]]]];

For[testIndex=1,testIndex<=Length[TESTS],testIndex++,
If[VERBOSE,Print["Processing "<>TESTS[[testIndex]]]];

Print["Reading data : "<>DSRREGRESSIONPATH<>"/"<>TESTS[[testIndex]]<>".csv"];
data=Normal[Import[DSRREGRESSIONPATH<>"/"<>TESTS[[testIndex]]<>".csv","Dataset"]];
Print["  Data dimensions : "<>ToString[Dimensions[data]]];

If[NOISE,
{
Print["Add noise to data"];
data=Map[{#[[1]],#[[2]]+RandomVariate[NormalDistribution[0,STD],1][[1]]}&,data];
}
];

SEEDS=Range[0,MC-1];
If[RUNDEFAULTS,
FITS=ParallelMap[FindFormula[data,x, 1, {"Score","Error","Complexity"},RandomSeeding->#,TargetFunctions->LIBRARY]&,SEEDS],
FITS=ParallelMap[FindFormula[data,x, 1, {"Score","Error","Complexity"},PerformanceGoal-> "Quality",SpecificityGoal->Infinity,RandomSeeding->#,TargetFunctions->LIBRARY]&,SEEDS]
];

(* Check for success *)
FITS=Map[Append[Flatten[#,1],0.0]&,FITS];
For[i=1,i<=Length[FITS],i++,
success=0.0;
If[PossibleZeroQ[Chop[((FITS[[i]][[1]]-NguyenDict[TESTS[[testIndex]]][x])/.RULESODDEXP)/.RULESFLOATEXP]],
success=1.0;
];
If[success==0.0,
If[TESTS[[testIndex]]=="Nguyen-8"&&
(PossibleZeroQ[(FITS[[i]][[1]]-1.0 x^0.5)/.RULESODDEXP]),
success=1.0;
];
];
FITS[[i,-1]]=success;
];

Export[dir<>"/"<>"run_stats_"<>TESTS[[testIndex]]<>".csv",FITS];
Export[dir<>"/"<>"mean_run_stats_"<>TESTS[[testIndex]]<>".txt",ToString[Map[FortranForm[#]&,Mean[FITS[[All,2;;]]]]]];
Export[dir<>"/"<>"run_stats_"<>TESTS[[testIndex]]<>".m",Dataset[FITS]];
];
