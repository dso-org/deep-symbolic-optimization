#!/usr/bin/env wolframscript
(* ::Package:: *)

(* 
Description:
  Evaluates FindFormula algorithm with 100 MC seeds over the 1D tests in:
  Uy,  N. Q.,  Hoai,  N. X.,  ONeill,  M.,  McKay,  R. I.,  and Galvan-Lopez, E.  
  Semantically-based crossover in genetic programming: application to real-valued 
  symbolic regression. Genetic Programming and Evolvable Machines, 12(2):91\[Dash]119, 2011. 
  
  Needs Mathematica 12.0.0. Runs in parallel over the number of kernels available.

Run: 
  wolfram-12.0.0 -script wolfram_nguyen_baseline.wls TESTFAMILY MC RUNMODE USENOISE NSE MTP

Command options:
  TESTFAMILY :
      - "Nguyen"
      - "Constant" 
      - "Custom" 
  MC : Monte Carlo seed
  RUNMODE :
      - "Default"
      - "Error"
      - "Score"
  USENOISE : False/True
  NSE : {"0.00", "0.02", "0.04", "0.06", "0.08", "0.10"}
  MTP : {"1", "10"}

Output:
  OUTPUTPATH is by default this ./wolfram_nguyen_baseline
*)

DSRCODEPATH="../../..";
OUTPUTPATH="./wolfram_nguyen_baseline";

TESTS={"Nguyen-1","Nguyen-2","Nguyen-3","Nguyen-4","Nguyen-5","Nguyen-6","Nguyen-7","Nguyen-8"}; (*1D tests*)

TESTFAMILY=$ScriptCommandLine[[2]]
Which[
TESTFAMILY=="Nguyen", TESTS={"Nguyen-1","Nguyen-2","Nguyen-3","Nguyen-4","Nguyen-5","Nguyen-6","Nguyen-7","Nguyen-8"} (*1D tests*),
TESTFAMILY=="Constant", TESTS={"Constant-1","Constant-2"} (*1D tests*),
TESTFAMILY=="Custom", TESTS={"Custom-1","Custom-2","Custom-6","Custom-13","Custom-16","Custom-21"} (*1D tests*)
];

MC=ToExpression[$ScriptCommandLine[[3]]];
VERBOSE=True;
RUNMODE=$ScriptCommandLine[[4]];
USENOISE=ToExpression[$ScriptCommandLine[[5]]];
NSE=$ScriptCommandLine[[6]];
MTP=$ScriptCommandLine[[7]];


If[VERBOSE,Print[" MC: "<>ToString[MC]<>"\n RUNMODE: "<>RUNMODE<>"\n USENOISE: "<>ToString[USENOISE]<>"\n NSE: "<>NSE<>"\n MTP: "<>MTP]];

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
NAMEDIR=OUTPUTPATH<>"/log_"<>ToString[TESTFAMILY]<>"_"<>ToString[MC]<>"_"<>ToString[RUNMODE];
If[USENOISE,
NAMEDIR=NAMEDIR<>"_"<>"nse_"<>ToString[NSE]<>"_"<>"mtp_"<>ToString[MTP];
]
NAMEDIR=NAMEDIR<>"_"<>TIMESTAMP;
dir = CreateDirectory[NAMEDIR];
Print["Log directory : "<>dir];

FUNCDICT=<|
"Nguyen-1"->(1.0#^3+1.0#^2+1.0# &),
"Nguyen-2"->(1.0#^4+1.0#^3+1.0#^2+1.0#&),
"Nguyen-3"->(1.0#^5+1.0#^4+1.0#^3+1.0#^2+1.0#&),
"Nguyen-4"->(1.0#^6+1.0#^5+1.0#^4+1.0#^3+1.0#^2+1.0#&),
"Nguyen-5"->(1.0Sin[#^2.]Cos[#]-1.0&),
"Nguyen-6"->(1.0Sin[#]+1.0Sin[#+#^2.]&),
"Nguyen-7"->(1.0Log[#+1.0]+1.0Log[#^2+1.0]&),
"Nguyen-8"->(1.0Sqrt[#]&),
"Constant-1"->(3.39*(#^3)+2.12*(#^2)+1.78*#&),
"Constant-2"->(Sin[(#^2)]*Cos[#]-0.75&),
"Custom-1"-> (1.0/3.0+#+1.0*Sin[#^2]&),
"Custom-2"-> (1.0 Sin[#^2]Cos[#]-2.0&),
"Custom-6"-> (4.0#^4+3.0#^3+2.0#^2+1.0#&),
"Custom-13"->(#^(1.0/3)&),
"Custom-16"->(#^(2.0/3)&),
"Custom-21"->(Exp[-1.0*#^2]&)
|>;

Print["  Parallel kernels : " <> ToString[Length[ParallelEvaluate[$KernelID]]]];

For[testIndex=1,testIndex<=Length[TESTS],testIndex++,
If[VERBOSE,Print["Processing "<>TESTS[[testIndex]]]];

CSVNAME=DSRREGRESSIONPATH<>"/"<>TESTS[[testIndex]];
If[USENOISE,
{
Print["Use noise"];
CSVNAME = CSVNAME<>"_n"<>NSE<>"_d"<>MTP;
}
];
CSVNAME=CSVNAME<>".csv";

Print["Reading data : "<>CSVNAME];
data=Normal[Import[CSVNAME,"Dataset"]];
Print["  Data dimensions : "<>ToString[Dimensions[data]]];

SEEDS=Range[0,MC-1];
Which[
RUNMODE=="Default", FITS=ParallelMap[FindFormula[data,x, 1, {"Score","Error","Complexity"}, SpecificityGoal->"High", RandomSeeding-> #,TargetFunctions->LIBRARY]&,SEEDS],
RUNMODE=="Error", FITS=ParallelMap[FindFormula[data,x, 1, "Error", PerformanceGoal->"Quality", SpecificityGoal-> "High", RandomSeeding->#,TargetFunctions->LIBRARY]&,SEEDS],
RUNMODE=="Score", FITS=ParallelMap[FindFormula[data,x, 1, "Score", PerformanceGoal->"Quality", SpecificityGoal-> "High", RandomSeeding->#,TargetFunctions->LIBRARY]&,SEEDS]
];

(* Check for success *)
FITS=Map[Append[Flatten[#,1],0.0]&,FITS];
For[i=1,i<=Length[FITS],i++,
success=0.0;
If[PossibleZeroQ[Chop[((FITS[[i]][[1]]-FUNCDICT[TESTS[[testIndex]]][x])/.RULESODDEXP)/.RULESFLOATEXP]],
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
];
