clc
clear
close all
%%
addpath('dynalog_function')

dataPath = "C:\Users\user\Downloads\dynalog_leaf_position\dynalog_leaf_position_matlab\raw_data\train_1st"
bankAFileName = 'A20170616160055_RT02530.dlg'

BankAFileName = bankAFileName
bankA = dynRead(fullfile(dataPath, BankAFileName));
BankAFileName(1) = [];
bankBName = strcat('B', BankAFileName);
bankB = dynRead(fullfile(dataPath, bankBName));
%
numLeaves = bankA.numLeaves;
numFraction = bankA.numFractions;
%
bankA_pos = bankA.planPosition;
bankB_pos = bankB.planPosition;

[mapPlanned mapActual] = dynFluence(bankA, bankB, 1.0, 0.1); 