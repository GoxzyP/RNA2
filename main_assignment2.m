% MAIN_ASSIGNMENT2  Artificial Neural Networks - Assignment 2 (2026/27).
%
% Runs the whole study by calling the functions in ann_lib.m.
% Compatible with MATLAB R2007a or later and with GNU Octave.
%
% Current content
% ---------------
% Step 1 - Data acquisition and initial inspection.
% Step 2 - Exploratory data analysis (tables printed, figures saved).
% Step 3 - Preprocessing and feature engineering.
% Step 4 - Data partitioning (and scaling with training statistics).
% Step 5 - Implementation and verification (gradient check, smoke test).
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


%% ========================================================================
%  STEP 3 - PREPROCESSING AND FEATURE ENGINEERING
%  ========================================================================
%
% Both datasets end up with d = 9 input features, used unchanged by every
% architecture and algorithm. Scaling statistics need the training set, so
% the scalers (L.fit_standardization) are fitted and applied in Step 4,
% right after the split.

% ---- Concrete: remove exact duplicates; log10(Age); binder/water ratio ----
[concrete_features, concrete_preprocessing_report] = L.preprocess_concrete_dataset(concrete_dataset);

% ---- Melbourne: fill 2 missing days; choose p; lags + calendar features ---
melbourne_candidate_window_lengths = [1 2 3 4 5 7 10 14 21 30];
melbourne_split_fractions          = [0.70 0.15 0.15];   % train / validation / test

melbourne_features = L.preprocess_melbourne_dataset(melbourne_dataset, melbourne_candidate_window_lengths, melbourne_split_fractions);


%% ========================================================================
%  STEP 4 - DATA PARTITIONING
%  ========================================================================
%
% The splits are computed once here and reused for the whole assignment, so
% the variability across repetitions comes only from the weight
% initialisation and the order of the mini-batches. Scalers are fitted on
% the training subset only and applied to all three subsets.

% ---- Concrete: random 70/15/15 split + evidence for the discussion --------
concrete_split_fractions  = [0.70 0.15 0.15];
concrete_split_seed       = 42;
concrete_comparison_seeds = 1:20;   % only for the random-versus-grouped comparison

concrete_partition = L.partition_concrete_dataset(concrete_features, concrete_split_fractions, concrete_split_seed, concrete_comparison_seeds);

% ---- Melbourne: chronological blocks fixed in Step 3 ----------------------
melbourne_partition = L.partition_melbourne_dataset(melbourne_features);


%% ========================================================================
%  STEP 5 - IMPLEMENTATION AND VERIFICATION
%  ========================================================================
%
% The library (network, loss_grad, mini-batch GD, BFGS, Adam, metrics) is
% in ann_lib.m. Here it is only verified:
%   1. Bonus: analytical gradient against central finite differences for
%      architectures A, B and C on both datasets.
%   2. Smoke test: a short training of architecture A with the three
%      algorithms, to check that each one runs and lowers the loss. The
%      learning rates here are provisional; they are tuned in Step 6.

% ---- 1. Gradient check (bonus) ---------------------------------------------
gradient_check_samples = 50;     % first 50 training rows
gradient_check_seed    = 42;     % seed of the initial weights (training seed of the assignment)
finite_difference_step = 1e-5;

concrete_gradient_checks  = L.verify_gradients_for_all_architectures('CONCRETE', concrete_partition, gradient_check_samples, gradient_check_seed, finite_difference_step);
melbourne_gradient_checks = L.verify_gradients_for_all_architectures('MELBOURNE', melbourne_partition, gradient_check_samples, gradient_check_seed, finite_difference_step);

% ---- 2. Smoke test of the three algorithms ---------------------------------
smoke_test_options = L.default_training_options();
smoke_test_options.budget_epoch_equivalents = 50;
smoke_test_options.shuffle_seed = 42;               % initial weights and mini-batch order
smoke_test_options.learning_rate_gd   = 0.05;    % provisional (Step 6 tunes it)
smoke_test_options.learning_rate_adam = 0.005;   % provisional (Step 6 tunes it)

concrete_smoke_test  = L.run_training_smoke_test('CONCRETE', concrete_partition, smoke_test_options);
melbourne_smoke_test = L.run_training_smoke_test('MELBOURNE', melbourne_partition, smoke_test_options);

%% ========================================================================
%  STEP 6 - HYPERPARAMETER TUNING
%  ========================================================================
%
% The learning rates of mini-batch GD and Adam are tuned on the VALIDATION
% set, separately for each dataset and architecture: 7 values on a
% logarithmic grid (half-decade steps), 5 repetitions per value. The batch
% size is fixed at 32 for both algorithms. BFGS has no learning rate; its
% line-search settings and stopping criteria are printed below.
%
% Tuning seeds 42..46: 42 is the training seed of the assignment, and they
% do not overlap with the seeds 1..30 of the final runs (Step 7), so the
% final results are not measured on the initialisations used to choose
% the learning rates.

tuning_options = L.default_training_options();

% Reduced budget to keep the search affordable (7 x 5 x 2 x 3 runs per
% dataset). Set it to 1000 (same as Step 7) if there is time.
tuning_options.budget_epoch_equivalents = 200;

learning_rate_grids.gd   = 10 .^ (-3:0.5:0);     % 0.001 ... 1
learning_rate_grids.adam = 10 .^ (-4:0.5:-1);    % 0.0001 ... 0.1

tuning_seeds = 42:46;

L.print_bfgs_settings(L.default_training_options());

% ---- Concrete ---------------------------------------------------------------
concrete_tuning = L.tune_learning_rates('CONCRETE', concrete_partition, learning_rate_grids, tuning_seeds, tuning_options);
save(fullfile(data_folder, 'step6_tuning_concrete.mat'), 'concrete_tuning', '-mat');

figure_handle = L.plot_learning_rate_tuning(concrete_tuning);
L.save_figure(figure_handle, figures_folder, 'step6_concrete_learning_rate_tuning');

% ---- Melbourne --------------------------------------------------------------
melbourne_tuning = L.tune_learning_rates('MELBOURNE', melbourne_partition, learning_rate_grids, tuning_seeds, tuning_options);
save(fullfile(data_folder, 'step6_tuning_melbourne.mat'), 'melbourne_tuning', '-mat');

figure_handle = L.plot_learning_rate_tuning(melbourne_tuning);
L.save_figure(figure_handle, figures_folder, 'step6_melbourne_learning_rate_tuning');

% ---- Learning rates used in Step 7 ------------------------------------------
% Row = architecture (A, B, C), column 1 = mini-batch GD, column 2 = Adam.
concrete_learning_rates  = concrete_tuning.chosen_learning_rate;
melbourne_learning_rates = melbourne_tuning.chosen_learning_rate;
