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
  wolframscript -script wolfram_nguyen_baseline.wls DSOPATH TESTFAMILY MC RUNMODE SPCGOAL USENOISE NSE MTP

  - In LC:
	wolframscript in /usr/tce/packages/mathematica/12.0.0/Executables/wolframscript

Command options:
  DSOPATH : Path to DSO (eg., "/usr/workspace/landajue/code/hypothesis_testing")
  TESTFAMILY : {"Nguyen", "Constant", "Custom"}
  MC : Monte Carlo seed (eg, 100)
  RUNMODE : {"Default", "Error", "Score"}
  SPCGOAL : {"High", "Low"}
  USENOISE : {False, True}
  NSE : {"0.00", "0.02", "0.04", "0.06", "0.08", "0.10"}
  MTP : {"1", "10"}
  
Output:
  OUTPUTPATH is by default this ./wolfram_nguyen_baseline
*)

CMDIX = 1
DSOCODEPATH=$ScriptCommandLine[[++CMDIX]];
OUTPUTPATH=".";
TESTFAMILY=$ScriptCommandLine[[++CMDIX]]
MC=ToExpression[$ScriptCommandLine[[++CMDIX]]];
VERBOSE=True;
RUNMODE=$ScriptCommandLine[[++CMDIX]];
SPCGOAL=$ScriptCommandLine[[++CMDIX]];
USENOISE=ToExpression[$ScriptCommandLine[[++CMDIX]]];
NSE=ToString[NumberForm[ToExpression[$ScriptCommandLine[[++CMDIX]]],{1,2}]];
MTP=ToString[NumberForm[ToExpression[$ScriptCommandLine[[++CMDIX]]],{1,2}]];


SUMMARY={};

If[VERBOSE,Print[" MC: "<>ToString[MC]<>"\n RUNMODE: "<>RUNMODE<>"\n USENOISE: "<>ToString[USENOISE]<>"\n NSE: "<>NSE<>"\n MTP: "<>MTP<>"\n SPCGOAL: "<>ToString[SPCGOAL]]];

If[USENOISE,
If[TESTFAMILY=="Constant"||TESTFAMILY=="Custom",
{
Print["Noise not supported for "<>TESTFAMILY];
Abort[];
}]
];

Which[
TESTFAMILY=="Nguyen", TESTS={"Nguyen-1","Nguyen-2","Nguyen-3","Nguyen-4","Nguyen-5","Nguyen-6","Nguyen-7","Nguyen-8"} (*1D tests*),
TESTFAMILY=="Constant", TESTS={"Constant-1","Constant-2","Constant-5","Constant-6","Constant-8"} (*1D tests*),
TESTFAMILY=="Custom", TESTS={"Custom-1","Custom-2","Custom-6","Custom-13","Custom-16","Custom-21"} (*1D tests*)
];

Which[
RUNMODE=="Default",HEADER={"name","expression","score","error","complexity","nmse_test","nmse_test_noiseless","success"},
RUNMODE=="Error",HEADER={"name","expression","error","nmse_test","nmse_test_noiseless","success"},
RUNMODE=="Score",HEADER={"name","expression","score","nmse_test","nmse_test_noiseless","success"},
RUNMODE=="Complexity",HEADER={"name","expression","complexity","nmse_test","nmse_test_noiseless","success"}
];

LIBRARY={Plus,Times,Power,Sin,Cos,Exp,Log};(*Function lib: Koza,"add,sub,mul,div,sin,cos,exp,log"*)
DSOREGRESSIONPATH=DSOCODEPATH<>"/dso/dso/task/regression/data"
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
NAMEDIR=OUTPUTPATH<>"/log_"<>ToString[TESTFAMILY]<>"_"<>ToString[MC]<>"_"<>ToString[RUNMODE]<>"_"<>ToString[SPCGOAL];
If[USENOISE,
NAMEDIR=NAMEDIR<>"_"<>"nse_"<>ToString[NSE]<>"_"<>"mtp_"<>ToString[MTP];
]
NAMEDIR=NAMEDIR<>"_"<>TIMESTAMP;
dir = CreateDirectory[NAMEDIR];
Print["Log directory : "<>dir];

MyVariance[list_]:=1/Length[list] Sum[(list[[i]]-1/Length[list] Total[list])^2,{i,1,Length[list]}];

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
"Constant-5"->(Sqrt[1.23*#]&),
"Constant-6"->(#^(0.426)&),
"Constant-8"->(1.0Log[#+1.4]+1.0Log[#^2+1.3]&),
"Custom-1"-> (1.0/3.0+#+1.0*Sin[#^2]&),
"Custom-2"-> (1.0 Sin[#^2]Cos[#]-2.0&),
"Custom-6"-> (4.0#^4+3.0#^3+2.0#^2+1.0#&),
"Custom-13"->(#^(1.0/3)&),
"Custom-16"->(#^(2.0/3)&),
"Custom-21"->(Exp[-1.0*#^2]&)
|>;

Print["  Parallel kernels : " <> ToString[Length[ParallelEvaluate[$KernelID]]]];

(* Loop over all tests*)
For[testIndex=1,testIndex<=Length[TESTS],testIndex++,
If[VERBOSE,Print["Processing "<>TESTS[[testIndex]]]];

CSVNAME=DSOREGRESSIONPATH<>"/"<>TESTS[[testIndex]];
If[USENOISE,
{
Print["Use noise"];
CSVNAME = CSVNAME<>"_n"<>NSE<>"_d"<>MTP;
}
];

Print["Reading data : "<>CSVNAME<>".csv"];
data=Normal[Import[CSVNAME<>".csv","Dataset"]];
datatest=Normal[Import[CSVNAME<>"_test"<>".csv","Dataset"]];
datatestNoiseLess=Normal[Import[DSOREGRESSIONPATH<>"/"<>TESTS[[testIndex]]<>"_test"<>".csv","Dataset"]];

(*for nmse calcualtions*)
varydatatest=MyVariance[datatest[[All,2]]];
varydatatestNoiseLess=MyVariance[datatestNoiseLess[[All,2]]];

Print["  Data dimensions : "<>ToString[Dimensions[data]]];

SEEDS=Range[0,MC-1];
Which[
RUNMODE=="Default", FITS=ParallelMap[FindFormula[data,x, 1, {"Score","Error","Complexity"}, SpecificityGoal->SPCGOAL, RandomSeeding-> #,TargetFunctions->LIBRARY]&,SEEDS],
RUNMODE=="Error", FITS=ParallelMap[FindFormula[data,x, 1, "Error", PerformanceGoal->"Quality", SpecificityGoal-> SPCGOAL, RandomSeeding->#,TargetFunctions->LIBRARY]&,SEEDS],
RUNMODE=="Score", FITS=ParallelMap[FindFormula[data,x, 1, "Score", PerformanceGoal->"Quality", SpecificityGoal-> SPCGOAL, RandomSeeding->#,TargetFunctions->LIBRARY]&,SEEDS],
RUNMODE=="Complexity", FITS=ParallelMap[FindFormula[data,x, 1, "Complexity", PerformanceGoal->"Quality", SpecificityGoal-> SPCGOAL, RandomSeeding->#,TargetFunctions->LIBRARY]&,SEEDS]
];

(* Add "nmse_test", "nmse_test_noiseless" and  "success"*)
(* Add a zero column at the end of the FITS table*)
FITS=Map[Flatten[Append[Flatten[#,1],{0.0,0.0,0.0}]]&,FITS];
For[i=1,i<=Length[FITS],i++,

nmsedatatest=Mean[(Map[FITS[[i]][[1]]/.{x-> #}&,datatest[[All,1]]]-datatest[[All,2]])^2]/varydatatest;
nmsedatatestNoiseLess=Mean[(Map[FITS[[i]][[1]]/.{x-> #}&,datatestNoiseLess[[All,1]]]-datatestNoiseLess[[All,2]])^2]/varydatatestNoiseLess;

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

FITS[[i,-3]]=nmsedatatest;
FITS[[i,-2]]=nmsedatatestNoiseLess;
FITS[[i,-1]]=success;

];

FITS=Map[Prepend[#,TESTS[[testIndex]]]&,FITS];
Map[AppendTo[SUMMARY,#]&,FITS]

(*dsFITS=Dataset[Table[AssociationThread[HEADER\[Rule]fit],{fit,FITS}]];
Export[dir<>"/"<>"run_stats_"<>TESTS[[testIndex]]<>".csv",dsFITS];*)

];  (* END of Loop over all tests*)

LABEL = "_"<>ToString[TESTFAMILY]<>"_"<>ToString[MC]<>"_"<>ToString[RUNMODE]<>"_"<>ToString[SPCGOAL];
If[USENOISE,
LABEL=LABEL<>"_"<>"nse_"<>ToString[NSE]<>"_"<>"mtp_"<>ToString[MTP];
]

ds=Dataset[Table[AssociationThread[HEADER->summ],{summ,SUMMARY}]];
Export[dir<>"/"<>"summary"<>LABEL<>".csv",ds];
dsII=N[ds[[All,DeleteCases[HEADER,"expression"]]]];
Export[dir<>"/"<>"summary_no_expr"<>LABEL<>".csv",dsII];
Export[dir<>"/"<>"summary_mean"<>LABEL<>".csv",dsII[GroupBy["name"],Mean]];
