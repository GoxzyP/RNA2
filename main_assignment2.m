% MAIN_ASSIGNMENT2  Artificial Neural Networks - Assignment 2 (2026/27).
%
% Runs the whole study by calling the functions in ann_lib.m.
% Compatible with MATLAB R2007a or later and with GNU Octave.
%
% Current content
% ---------------
% Step 1 - Data acquisition and initial inspection.
% Step 2 - Exploratory data analysis (tables printed, figures saved).
%
% How to run
% ----------
% 1. Put ann_lib.m, main_assignment2.m and the three data files in the same
%    folder (see "data_folder" below).
% 2. Make that folder the current folder of MATLAB / Octave.
% 3. Type: main_assignment2

clear;
clc;
close all;

L = ann_lib();


%% ========================================================================
%  STEP 1 - DATA ACQUISITION AND INITIAL INSPECTION
%  ========================================================================

% Folder that contains the data files. "pwd" is the current folder.
data_folder = pwd;

concrete_file_path   = fullfile(data_folder, 'Concrete_Data.csv');
melbourne_file_path  = fullfile(data_folder, 'daily-minimum-temperatures-in-me.csv');

% ---- Concrete (static regression: 8 inputs -> compressive strength) -------
concrete_dataset = L.load_concrete_dataset(concrete_file_path);
concrete_inspection_report = L.inspect_concrete_dataset(concrete_dataset);

% ---- Melbourne (time series: minimum temperature of day t+1) --------------
melbourne_dataset = L.load_melbourne_dataset(melbourne_file_path);
melbourne_inspection_report = L.inspect_melbourne_dataset(melbourne_dataset);


%% ========================================================================
%  STEP 2 - EXPLORATORY DATA ANALYSIS
%  ========================================================================

% Figures are shown on screen and saved as PNG files in this folder.
figures_folder = fullfile(data_folder, 'figures');

concrete_eda_report  = L.explore_concrete_dataset(concrete_dataset, figures_folder);
melbourne_eda_report = L.explore_melbourne_dataset(melbourne_dataset, figures_folder);
