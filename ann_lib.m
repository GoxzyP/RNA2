function L = ann_lib()
    % ANN_LIB  Library for Assignment 2 (Artificial Neural Networks 2026/27).
    %
    % Returns a struct of function handles, so that the whole library lives in
    % a single file (assignment requirement). Usage in the main script:
    %
    %     L = ann_lib();
    %     concrete = L.load_concrete_dataset('Concrete_Data.csv');
    %
    % Compatibility
    % -------------
    % Written for MATLAB 7.4 (R2007a) and later, and for GNU Octave.
    % It deliberately avoids anything newer than R2007a: no "~" to ignore
    % outputs, no strsplit, no containers.Map, no table/readtable, and no
    % toolbox functions (all statistics are implemented here).
    %
    % Contents
    % --------
    % The functions below are grouped by assignment step and, inside each
    % step, by dataset: CONCRETE (only Concrete), MELBOURNE (only
    % Melbourne) and UTILS (both datasets, or helpers reused later).
    % The handles are listed in the same order.

    % ======================================================================
    % STEP 1 - DATA ACQUISITION AND INITIAL INSPECTION
    % ======================================================================

    % ---- CONCRETE --------------------------------------------------------
    L.load_concrete_dataset    = @load_concrete_dataset;
    L.inspect_concrete_dataset = @inspect_concrete_dataset;

    % ---- MELBOURNE -------------------------------------------------------
    L.load_melbourne_dataset     = @load_melbourne_dataset;
    L.inspect_melbourne_dataset  = @inspect_melbourne_dataset;
    L.find_missing_calendar_days = @find_missing_calendar_days;

    % ---- UTILS -----------------------------------------------------------
    L.compute_column_summary = @compute_column_summary;
    L.compute_percentile     = @compute_percentile;
    L.print_column_summary   = @print_column_summary;
    L.print_section_title    = @print_section_title;

    % ======================================================================
    % STEP 2 - EXPLORATORY DATA ANALYSIS
    % ======================================================================

    % ---- CONCRETE --------------------------------------------------------
    L.explore_concrete_dataset            = @explore_concrete_dataset;
    L.plot_concrete_histograms            = @plot_concrete_histograms;
    L.plot_concrete_correlation_matrix    = @plot_concrete_correlation_matrix;
    L.plot_concrete_scatter_plots         = @plot_concrete_scatter_plots;
    L.compute_correlation_matrix          = @compute_correlation_matrix;
    L.compute_average_ranks               = @compute_average_ranks;
    L.compute_spearman_correlation_matrix = @compute_spearman_correlation_matrix;
    L.print_correlation_matrix            = @print_correlation_matrix;
    L.build_blue_white_red_colormap       = @build_blue_white_red_colormap;

    % ---- MELBOURNE -------------------------------------------------------
    L.explore_melbourne_dataset                = @explore_melbourne_dataset;
    L.plot_melbourne_series                    = @plot_melbourne_series;
    L.plot_melbourne_distribution              = @plot_melbourne_distribution;
    L.plot_melbourne_monthly_box_plots         = @plot_melbourne_monthly_box_plots;
    L.plot_melbourne_annual_cycle              = @plot_melbourne_annual_cycle;
    L.plot_melbourne_autocorrelation           = @plot_melbourne_autocorrelation;
    L.plot_melbourne_lag_scatter               = @plot_melbourne_lag_scatter;
    L.plot_melbourne_abrupt_changes            = @plot_melbourne_abrupt_changes;
    L.compute_autocorrelation_function         = @compute_autocorrelation_function;
    L.compute_partial_autocorrelation_function = @compute_partial_autocorrelation_function;
    L.compute_day_of_year                      = @compute_day_of_year;
    L.fit_annual_harmonic                      = @fit_annual_harmonic;
    L.compute_daily_climatology                = @compute_daily_climatology;
    L.circular_moving_average                  = @circular_moving_average;
    L.compute_longest_constant_run             = @compute_longest_constant_run;
    L.draw_correlation_stems                   = @draw_correlation_stems;

    % ---- UTILS -----------------------------------------------------------
    L.compute_pearson_correlation = @compute_pearson_correlation;
    L.draw_box_plot               = @draw_box_plot;
    L.save_figure                 = @save_figure;
    L.ensure_folder_exists        = @ensure_folder_exists;

    % ======================================================================
    % STEP 3 - PREPROCESSING AND FEATURE ENGINEERING
    % ======================================================================

    % ---- CONCRETE --------------------------------------------------------
    L.preprocess_concrete_dataset = @preprocess_concrete_dataset;
    L.remove_exact_duplicate_rows = @remove_exact_duplicate_rows;
    L.build_concrete_features     = @build_concrete_features;

    % ---- MELBOURNE -------------------------------------------------------
    L.preprocess_melbourne_dataset              = @preprocess_melbourne_dataset;
    L.fill_missing_days_by_linear_interpolation = @fill_missing_days_by_linear_interpolation;
    L.build_melbourne_windows                   = @build_melbourne_windows;
    L.compute_calendar_features                 = @compute_calendar_features;
    L.compare_window_lengths_with_linear_model  = @compare_window_lengths_with_linear_model;
    L.compute_chronological_split_cutoffs       = @compute_chronological_split_cutoffs;
    L.assign_samples_to_chronological_blocks    = @assign_samples_to_chronological_blocks;

    % ---- UTILS -----------------------------------------------------------
    L.fit_standardization             = @fit_standardization;
    L.apply_standardization           = @apply_standardization;
    L.invert_standardization          = @invert_standardization;
    L.fit_linear_least_squares        = @fit_linear_least_squares;
    L.predict_linear_model            = @predict_linear_model;
    L.compute_root_mean_squared_error = @compute_root_mean_squared_error;
    L.count_network_parameters        = @count_network_parameters;
    L.print_network_sizes             = @print_network_sizes;
    % ======================================================================
    % STEP 4 - DATA PARTITIONING
    % ======================================================================

    % ---- CONCRETE --------------------------------------------------------
    L.partition_concrete_dataset         = @partition_concrete_dataset;
    L.assign_rows_to_random_blocks       = @assign_rows_to_random_blocks;
    L.assign_groups_to_blocks            = @assign_groups_to_blocks;
    L.count_rows_with_group_in_training  = @count_rows_with_group_in_training;
    L.compare_random_and_grouped_splits  = @compare_random_and_grouped_splits;
    L.predict_nearest_neighbour          = @predict_nearest_neighbour;

    % ---- MELBOURNE -------------------------------------------------------
    L.partition_melbourne_dataset = @partition_melbourne_dataset;

    % ---- UTILS -----------------------------------------------------------
    L.generate_uniform_random_numbers = @generate_uniform_random_numbers;
    L.check_split_fractions           = @check_split_fractions;
    L.build_partition                 = @build_partition;
    L.scale_partition                 = @scale_partition;
    L.print_partition_summary         = @print_partition_summary;
    L.print_check_result              = @print_check_result;
    % ======================================================================
    % STEP 5 - IMPLEMENTATION AND VERIFICATION
    % ======================================================================

    % ---- UTILS (network, training algorithms, metrics, gradient check) ---
    L.build_architectures                    = @build_architectures;
    L.set_random_seed                        = @set_random_seed;
    L.init_params                            = @init_params;
    L.pack_parameters                        = @pack_parameters;
    L.unpack_parameters                      = @unpack_parameters;
    L.predict_network                        = @predict_network;
    L.compute_mse_loss                       = @compute_mse_loss;
    L.loss_grad                              = @loss_grad;
    L.default_training_options               = @default_training_options;
    L.create_training_history                = @create_training_history;
    L.train_minibatch_gd                     = @train_minibatch_gd;
    L.train_adam                             = @train_adam;
    L.train_with_minibatches                 = @train_with_minibatches;
    L.train_bfgs                             = @train_bfgs;
    L.compute_regression_metrics             = @compute_regression_metrics;
    L.check_gradient                         = @check_gradient;
    L.verify_gradients_for_all_architectures = @verify_gradients_for_all_architectures;
    L.run_training_smoke_test                = @run_training_smoke_test;
    % ======================================================================
    % STEP 6 - HYPERPARAMETER TUNING
    % ======================================================================

    % ---- UTILS (learning-rate tuning of mini-batch GD and Adam) ---------
    L.tune_learning_rates              = @tune_learning_rates;
    L.run_learning_rate_grid           = @run_learning_rate_grid;
    L.select_learning_rate             = @select_learning_rate;
    L.print_learning_rate_tuning_table = @print_learning_rate_tuning_table;
    L.plot_learning_rate_tuning        = @plot_learning_rate_tuning;
    L.print_bfgs_settings              = @print_bfgs_settings;
    % ======================================================================
    % STEP 7 - FINAL EXPERIMENTS
    % ======================================================================

    % ---- UTILS (18 combinations x 30 runs, test metrics) -----------------
    L.run_final_experiments          = @run_final_experiments;
    L.run_one_combination            = @run_one_combination;
    L.evaluate_on_test_set           = @evaluate_on_test_set;
    L.print_final_experiment_summary = @print_final_experiment_summary;
    % ======================================================================
    % STEP 8 - REQUIRED RESULTS
    % ======================================================================

    % ---- UTILS (tables, figures and reference models) --------------------
    L.compute_reference_models    = @compute_reference_models;
    L.produce_step8_results       = @produce_step8_results;
    L.print_step8_summary_table   = @print_step8_summary_table;
    L.compute_curve_quantiles     = @compute_curve_quantiles;
    L.plot_training_progress      = @plot_training_progress;
    L.plot_test_error_box_plots   = @plot_test_error_box_plots;
    L.select_representative_run   = @select_representative_run;
    L.plot_concrete_predictions   = @plot_concrete_predictions;
    L.plot_melbourne_predictions  = @plot_melbourne_predictions;
    L.draw_residual_histograms    = @draw_residual_histograms;

end


%% ========================================================================
%  STEP 1 - DATA ACQUISITION AND INITIAL INSPECTION: CONCRETE
%  Functions used only by the Concrete dataset.
%  ========================================================================

function dataset = load_concrete_dataset(file_path)
    % Load the UCI Concrete Compressive Strength dataset.
    %
    % Parameters
    % ----------
    % file_path : char
    %     Path to Concrete_Data.csv. The file must have one header row
    %     followed by 1030 rows with 9 numeric columns: 8 inputs followed
    %     by the target. Separator "," and decimal point ".".
    %
    % Returns
    % -------
    % dataset : struct
    %     inputs       : 1030 x 8 double, raw (unscaled) input variables
    %     target       : 1030 x 1 double, compressive strength
    %     input_names  : 1 x 8 cell of char, name of each input column
    %     input_short_names : 1 x 8 cell of char, short labels for plot axes
    %     input_units  : 1 x 8 cell of char, unit of each input column
    %     target_name  : char, name of the target variable
    %     target_short_name : char, short label of the target for plot axes
    %     target_unit  : char, unit of the target variable
    %
    % Notes
    % -----
    % Names and units follow Concrete_Readme.txt (UCI repository).

    expected_number_of_rows    = 1030;

    expected_number_of_columns = 9;

    if ~exist(file_path, 'file')
        error('load_concrete_dataset:fileNotFound', ...
              'File not found: %s', file_path);
    end

    raw_table = read_numeric_csv_with_header(file_path);

    [number_of_rows, number_of_columns] = size(raw_table);

    shape_is_wrong = (number_of_rows ~= expected_number_of_rows) || (number_of_columns ~= expected_number_of_columns);

    if shape_is_wrong
        error('load_concrete_dataset:unexpectedShape', ...
              'Expected a %d x %d table, but "%s" has %d x %d.', ...
              expected_number_of_rows, expected_number_of_columns, ...
              file_path, number_of_rows, number_of_columns);
    end

    dataset.inputs = raw_table(:, 1:8);
    dataset.target = raw_table(:, 9);

    dataset.input_names = {'Cement', 'Blast furnace slag', 'Fly ash', ...
                           'Water', 'Superplasticizer', 'Coarse aggregate', ...
                           'Fine aggregate', 'Age'};

    dataset.input_short_names = {'Cem', 'Slag', 'Ash', 'Water', 'SP', ...
                                 'CoAgg', 'FiAgg', 'Age'};

    dataset.input_units = {'kg/m^3', 'kg/m^3', 'kg/m^3', ...
                           'kg/m^3', 'kg/m^3', 'kg/m^3', ...
                           'kg/m^3', 'days'};

    dataset.target_name       = 'Compressive strength';
    dataset.target_short_name = 'Str';
    dataset.target_unit       = 'MPa';
end


function numeric_table = read_numeric_csv_with_header(file_path)
    % Read a comma-separated file with one header row and numeric rows.
    %
    % Parameters
    % ----------
    % file_path : char
    %     Path to the CSV file. Decimal separator must be "." and the
    %     column separator must be ",".
    %
    % Returns
    % -------
    % numeric_table : double matrix
    %     One row per non-empty line after the header.

    file_id = fopen(file_path, 'r');

    if file_id == -1
        error('read_numeric_csv_with_header:cannotOpen', ...
              'Could not open file: %s', file_path);
    end

    % The first line is the header (it contains text and commas): skip it.
    header_line = fgetl(file_id);  %#ok<NASGU>

    numeric_table = [];
    current_line  = fgetl(file_id);

    while ischar(current_line)
        line_is_empty = isempty(strtrim(current_line));

        if ~line_is_empty
            numbers_in_line = sscanf(current_line, '%f,')';

            table_already_has_rows = ~isempty(numeric_table);

            if table_already_has_rows && numel(numbers_in_line) ~= size(numeric_table, 2)
                fclose(file_id);
                error('read_numeric_csv_with_header:inconsistentColumns', ...
                      ['Line %d has %d numbers but earlier lines have %d. ' ...
                       'Check the separators (must be "," and ".").'], ...
                      size(numeric_table, 1) + 2, ...
                      numel(numbers_in_line), size(numeric_table, 2));
            end

            % Growing the matrix is fine here: only ~1000 rows.
            numeric_table = [numeric_table; numbers_in_line];
        end

        current_line = fgetl(file_id);
    end

    fclose(file_id);
end


function inspection_report = inspect_concrete_dataset(dataset)
    % Print and return the initial inspection of the Concrete dataset.
    %
    % Prints the variable table, descriptive statistics of all 9 variables
    % and an analysis of repeated mixtures (relevant for the train/test
    % split discussed in Step 4).
    %
    % Parameters
    % ----------
    % dataset : struct
    %     Output of load_concrete_dataset.
    %
    % Returns
    % -------
    % inspection_report : struct
    %     summary                         : output of compute_column_summary
    %     number_of_exact_duplicate_rows  : rows identical to an earlier row
    %     number_of_distinct_mixtures     : distinct 7-ingredient recipes
    %     number_of_repeated_mixtures     : recipes seen with more than 1 row
    %     number_of_rows_in_repeated_mixtures : rows belonging to those recipes
    %     distinct_ages_in_days           : sorted list of distinct ages
    %     number_of_rows_per_age          : rows for each distinct age

    number_of_observations = size(dataset.inputs, 1);
    number_of_inputs       = size(dataset.inputs, 2);

    all_columns = [dataset.inputs, dataset.target];
    all_names   = [dataset.input_names, {dataset.target_name}];
    all_units   = [dataset.input_units, {dataset.target_unit}];

    % ---- 1. Variable table ------------------------------------------------
    print_section_title('CONCRETE COMPRESSIVE STRENGTH - variables');

    fprintf('Observations : %d\n', number_of_observations);
    fprintf('Inputs       : %d\n', number_of_inputs);
    fprintf('Target       : %s (%s)\n\n', dataset.target_name, dataset.target_unit);

    fprintf('%-4s %-22s %-8s %-8s\n', '#', 'Name', 'Role', 'Unit');

    for variable_index = 1:numel(all_names)
        if variable_index <= number_of_inputs
            variable_role = 'input';
        else
            variable_role = 'target';
        end

        fprintf('%-4d %-22s %-8s %-8s\n', variable_index, ...
                all_names{variable_index}, variable_role, ...
                all_units{variable_index});
    end

    % ---- 2. Descriptive statistics ---------------------------------------
    print_section_title('CONCRETE COMPRESSIVE STRENGTH - descriptive statistics');

    column_summary = compute_column_summary(all_columns);
    print_column_summary(all_names, column_summary);

    % ---- 3. Repeated rows and repeated mixtures --------------------------
    print_section_title('CONCRETE COMPRESSIVE STRENGTH - repeated mixtures');

    number_of_exact_duplicate_rows = number_of_observations - size(unique(all_columns, 'rows'), 1);

    % A "mixture" is the 7 ingredients; the same recipe can be tested at
    % several ages, which matters for a random train/validation/test split.
    age_column_index        = find(strcmp(dataset.input_names, 'Age'));
    ingredient_column_index = setdiff(1:number_of_inputs, age_column_index);
    ingredient_matrix       = dataset.inputs(:, ingredient_column_index);

    [distinct_mixtures, unused_first_index, mixture_id_of_each_row] = unique(ingredient_matrix, 'rows');

    number_of_distinct_mixtures = size(distinct_mixtures, 1);
    number_of_rows_per_mixture = histc(mixture_id_of_each_row, 1:number_of_distinct_mixtures);
    mixture_is_repeated = number_of_rows_per_mixture > 1;
    number_of_repeated_mixtures = sum(mixture_is_repeated);
    number_of_rows_in_repeated_mixtures = sum(number_of_rows_per_mixture(mixture_is_repeated));

    fprintf('Exact duplicate rows (all 9 variables) : %d\n', ...
            number_of_exact_duplicate_rows);
    fprintf('Distinct mixtures (7 ingredients)      : %d\n', ...
            number_of_distinct_mixtures);
    fprintf('Mixtures that appear in more than 1 row: %d\n', ...
            number_of_repeated_mixtures);
    fprintf('Rows that belong to those mixtures     : %d (%.1f%% of all rows)\n', ...
            number_of_rows_in_repeated_mixtures, ...
            100 * number_of_rows_in_repeated_mixtures / number_of_observations);
    fprintf('Largest number of rows for one mixture : %d\n', ...
            max(number_of_rows_per_mixture));

    % ---- 4. Distinct values of Age ---------------------------------------
    print_section_title('CONCRETE COMPRESSIVE STRENGTH - values taken by Age');

    age_values = dataset.inputs(:, age_column_index);
    distinct_ages_in_days  = unique(age_values);
    number_of_rows_per_age = histc(age_values, distinct_ages_in_days);

    fprintf('Age takes only %d distinct values:\n', numel(distinct_ages_in_days));
    for age_index = 1:numel(distinct_ages_in_days)
        fprintf('  %4g days : %4d rows\n', ...
                distinct_ages_in_days(age_index), ...
                number_of_rows_per_age(age_index));
    end

    % ---- Report -----------------------------------------------------------
    inspection_report.summary = column_summary;
    inspection_report.number_of_exact_duplicate_rows = number_of_exact_duplicate_rows;
    inspection_report.number_of_distinct_mixtures    = number_of_distinct_mixtures;
    inspection_report.number_of_repeated_mixtures    = number_of_repeated_mixtures;
    inspection_report.number_of_rows_in_repeated_mixtures = number_of_rows_in_repeated_mixtures;
    inspection_report.distinct_ages_in_days  = distinct_ages_in_days;
    inspection_report.number_of_rows_per_age = number_of_rows_per_age;
end


%% ========================================================================
%  STEP 1 - DATA ACQUISITION AND INITIAL INSPECTION: MELBOURNE
%  Functions used only by the Melbourne dataset.
%  ========================================================================

function dataset = load_melbourne_dataset(file_path)
    % Load the Daily Minimum Temperatures in Melbourne dataset (Kaggle).
    %
    % The file has a header row, one line per day in the form
    %     "1981-01-01",20.7
    % and a text footer. Only lines that start with a quoted date are data.
    %
    % Known defect of this file
    % -------------------------
    % Three negative values were saved with a "?" instead of the minus
    % sign (e.g. "?0.2" instead of "-0.2"). They are repaired while
    % reading and reported in repaired_dates / repaired_original_texts, so
    % nothing is changed silently. All three fall in July (winter), where
    % values close to 0 degC are expected, which supports the repair.
    %
    % Parameters
    % ----------
    % file_path : char
    %     Path to daily-minimum-temperatures-in-me.csv.
    %
    % Returns
    % -------
    % dataset : struct
    %     dates                  : N x 1 double, MATLAB serial date numbers
    %     temperature_celsius    : N x 1 double, minimum temperature (degC)
    %     variable_names         : 1 x 2 cell of char
    %     variable_units         : 1 x 2 cell of char
    %     repaired_dates         : R x 1 double, dates of repaired values
    %     repaired_original_texts: R x 1 cell of char, text before repair

    if ~exist(file_path, 'file')
        error('load_melbourne_dataset:fileNotFound', ...
              'File not found: %s', file_path);
    end

    file_id = fopen(file_path, 'r');

    if file_id == -1
        error('load_melbourne_dataset:cannotOpen', ...
              'Could not open file: %s', file_path);
    end

    maximum_number_of_rows = 5000;   % safe upper bound (file has 3650 days)
    dates               = zeros(maximum_number_of_rows, 1);
    temperature_celsius = zeros(maximum_number_of_rows, 1);

    number_of_observations  = 0;
    line_number             = 0;
    repaired_dates          = [];
    repaired_original_texts = {};

    % Pattern: "YYYY-MM-DD",<value>   (value = any run of non-blank chars)
    data_line_pattern = '^"(\d{4})-(\d{2})-(\d{2})",\s*(\S+)';

    current_line = fgetl(file_id);

    while ischar(current_line)
        line_number = line_number + 1;

        tokens = regexp(current_line, data_line_pattern, 'tokens', 'once');

        % Header, blank lines and footer do not match and are ignored.
        if ~isempty(tokens)

            year_number  = str2double(tokens{1});
            month_number = str2double(tokens{2});
            day_number   = str2double(tokens{3});
            value_text   = tokens{4};

            value_starts_with_bad_character = ~isempty(regexp(value_text, '^[^0-9.+\-]', 'once'));

            if value_starts_with_bad_character
                original_value_text = value_text;
                value_text = regexprep(value_text, '^[^0-9.+\-]+', '-');
            end

            temperature_value = str2double(value_text);

            if isnan(temperature_value)
                fclose(file_id);
                error('load_melbourne_dataset:invalidValue', ...
                      'Line %d: could not read a number from "%s".', ...
                      line_number, tokens{4});
            end

            number_of_observations = number_of_observations + 1;

            dates(number_of_observations) = datenum(year_number, month_number, day_number);
            temperature_celsius(number_of_observations) = temperature_value;

            if value_starts_with_bad_character
                repaired_dates(end + 1, 1) = dates(number_of_observations);
                repaired_original_texts{end + 1, 1} = original_value_text;
            end
        end

        current_line = fgetl(file_id);
    end

    fclose(file_id);

    dataset.dates               = dates(1:number_of_observations);
    dataset.temperature_celsius = temperature_celsius(1:number_of_observations);

    dataset.variable_names = {'Date', 'Minimum temperature'};
    dataset.variable_units = {'day', 'degC'};

    dataset.repaired_dates          = repaired_dates;
    dataset.repaired_original_texts = repaired_original_texts;
end


function inspection_report = inspect_melbourne_dataset(dataset)
    % Print and return the initial inspection of the Melbourne dataset.
    %
    % Prints the temporal coverage, checks the time axis (gaps, duplicate
    % or unordered dates), repaired values, yearly statistics and the
    % descriptive statistics of the temperature.
    %
    % Parameters
    % ----------
    % dataset : struct
    %     Output of load_melbourne_dataset.
    %
    % Returns
    % -------
    % inspection_report : struct
    %     summary                      : output of compute_column_summary
    %     first_date, last_date        : serial date numbers
    %     number_of_observations       : number of days in the file
    %     number_of_calendar_days      : days between first and last date
    %     missing_dates                : serial date numbers of absent days
    %     number_of_duplicate_dates    : consecutive rows with the same date
    %     number_of_out_of_order_dates : rows whose date goes backwards
    %     largest_day_to_day_change    : largest |T(t+1) - T(t)| in degC
    %     date_of_largest_change       : serial date of the later day

    dates               = dataset.dates;
    temperature_celsius = dataset.temperature_celsius;
    number_of_observations = numel(dates);

    % ---- 1. Variables and temporal coverage -------------------------------
    print_section_title('MELBOURNE MINIMUM TEMPERATURE - variables and coverage');

    first_date = dates(1);
    last_date  = dates(number_of_observations);
    number_of_calendar_days = last_date - first_date + 1;

    fprintf('%-4s %-22s %-8s\n', '#', 'Name', 'Unit');
    for variable_index = 1:numel(dataset.variable_names)
        fprintf('%-4d %-22s %-8s\n', variable_index, ...
                dataset.variable_names{variable_index}, ...
                dataset.variable_units{variable_index});
    end

    fprintf('\nFirst date               : %s\n', datestr(first_date, 'yyyy-mm-dd'));
    fprintf('Last date                : %s\n',   datestr(last_date,  'yyyy-mm-dd'));
    fprintf('Observations in file     : %d\n',   number_of_observations);
    fprintf('Calendar days in span    : %d\n',   number_of_calendar_days);
    fprintf('Sampling                 : one value per day\n');

    % ---- 2. Time axis checks ---------------------------------------------
    print_section_title('MELBOURNE MINIMUM TEMPERATURE - time axis checks');

    days_between_rows = diff(dates);

    number_of_duplicate_dates    = sum(days_between_rows == 0);
    number_of_out_of_order_dates = sum(days_between_rows <  0);

    missing_dates = find_missing_calendar_days(dates);

    fprintf('Duplicate dates          : %d\n', number_of_duplicate_dates);
    fprintf('Dates out of order       : %d\n', number_of_out_of_order_dates);
    fprintf('Missing calendar days    : %d\n', numel(missing_dates));

    for missing_index = 1:numel(missing_dates)
        fprintf('  missing: %s\n', datestr(missing_dates(missing_index), 'yyyy-mm-dd'));
    end

    % ---- 3. Values repaired while reading --------------------------------
    print_section_title('MELBOURNE MINIMUM TEMPERATURE - repaired values');

    number_of_repaired_values = numel(dataset.repaired_dates);
    fprintf('Values with a corrupted minus sign ("?") : %d\n', number_of_repaired_values);

    for repaired_index = 1:number_of_repaired_values
        repaired_date  = dataset.repaired_dates(repaired_index);
        row_of_value   = find(dates == repaired_date);
        fprintf('  %s : "%s" read as %.1f degC\n', ...
                datestr(repaired_date, 'yyyy-mm-dd'), ...
                dataset.repaired_original_texts{repaired_index}, ...
                temperature_celsius(row_of_value));
    end

    % ---- 4. Statistics per year ------------------------------------------
    print_section_title('MELBOURNE MINIMUM TEMPERATURE - statistics per year');

    date_vectors          = datevec(dates);
    year_of_each_row      = date_vectors(:, 1);
    distinct_years        = unique(year_of_each_row);

    fprintf('%-6s %8s %9s %8s %8s %8s\n', 'Year', 'Observed', 'Expected', 'Mean', 'Min', 'Max');

    for year_index = 1:numel(distinct_years)
        current_year = distinct_years(year_index);
        row_is_in_year = (year_of_each_row == current_year);
        temperatures_of_year = temperature_celsius(row_is_in_year);

        expected_days_in_year = datenum(current_year + 1, 1, 1) - datenum(current_year, 1, 1);

        fprintf('%-6d %8d %9d %8.2f %8.2f %8.2f\n', current_year, ...
                sum(row_is_in_year), expected_days_in_year, ...
                mean(temperatures_of_year), min(temperatures_of_year), ...
                max(temperatures_of_year));
    end

    % ---- 5. Descriptive statistics and largest daily jump -----------------
    print_section_title('MELBOURNE MINIMUM TEMPERATURE - descriptive statistics');

    column_summary = compute_column_summary(temperature_celsius);
    print_column_summary({'Minimum temperature'}, column_summary);

    % Only compare days that are exactly one calendar day apart.
    temperature_change_to_next_row = diff(temperature_celsius);
    rows_are_consecutive_days      = (days_between_rows == 1);
    temperature_change_to_next_row(~rows_are_consecutive_days) = NaN;

    [largest_day_to_day_change, index_before_largest_change] = max(abs(temperature_change_to_next_row));
    date_of_largest_change = dates(index_before_largest_change + 1);

    fprintf('\nLargest day-to-day change: %.1f degC (%s -> %s)\n', ...
            largest_day_to_day_change, ...
            datestr(dates(index_before_largest_change), 'yyyy-mm-dd'), ...
            datestr(date_of_largest_change, 'yyyy-mm-dd'));

    % ---- Report -----------------------------------------------------------
    inspection_report.summary                      = column_summary;
    inspection_report.first_date                   = first_date;
    inspection_report.last_date                    = last_date;
    inspection_report.number_of_observations       = number_of_observations;
    inspection_report.number_of_calendar_days      = number_of_calendar_days;
    inspection_report.missing_dates                = missing_dates;
    inspection_report.number_of_duplicate_dates    = number_of_duplicate_dates;
    inspection_report.number_of_out_of_order_dates = number_of_out_of_order_dates;
    inspection_report.largest_day_to_day_change    = largest_day_to_day_change;
    inspection_report.date_of_largest_change       = date_of_largest_change;
end


function missing_dates = find_missing_calendar_days(dates)
    % Calendar days that are absent from a chronologically sorted date list.
    %
    % Parameters
    % ----------
    % dates : double column
    %     Serial date numbers, sorted in ascending order.
    %
    % Returns
    % -------
    % missing_dates : double column
    %     Serial date numbers of the absent days (empty if none).

    days_between_rows = diff(dates);
    index_before_each_gap = find(days_between_rows > 1);

    missing_dates = [];

    for gap_number = 1:numel(index_before_each_gap)
        index_before_gap  = index_before_each_gap(gap_number);
        first_missing_day = dates(index_before_gap) + 1;
        last_missing_day  = dates(index_before_gap + 1) - 1;

        missing_dates = [missing_dates; (first_missing_day:last_missing_day)'];
    end
end


%% ========================================================================
%  STEP 1 - DATA ACQUISITION AND INITIAL INSPECTION: UTILS
%  Functions used by both datasets (or generic helpers reused in later steps).
%  ========================================================================

function column_summary = compute_column_summary(data_matrix)
    % Descriptive statistics of every column of a matrix (NaN is ignored).
    %
    % Parameters
    % ----------
    % data_matrix : double matrix, N x K
    %     One variable per column.
    %
    % Returns
    % -------
    % column_summary : struct
    %     Every field is a 1 x K row vector (one value per column):
    %     number_of_valid_values, number_of_missing_values,
    %     number_of_zeros, number_of_distinct_values,
    %     minimum, first_quartile, median, third_quartile, maximum,
    %     mean, standard_deviation (sample, divisor n-1), skewness.

    number_of_columns = size(data_matrix, 2);

    number_of_valid_values    = zeros(1, number_of_columns);
    number_of_missing_values  = zeros(1, number_of_columns);
    number_of_zeros           = zeros(1, number_of_columns);
    number_of_distinct_values = zeros(1, number_of_columns);
    minimum                   = zeros(1, number_of_columns);
    first_quartile            = zeros(1, number_of_columns);
    median_value              = zeros(1, number_of_columns);
    third_quartile            = zeros(1, number_of_columns);
    maximum                   = zeros(1, number_of_columns);
    mean_value                = zeros(1, number_of_columns);
    standard_deviation        = zeros(1, number_of_columns);
    skewness                  = zeros(1, number_of_columns);

    for column_index = 1:number_of_columns

        column_values = data_matrix(:, column_index);
        value_is_missing = isnan(column_values);

        valid_values  = column_values(~value_is_missing);
        sorted_values = sort(valid_values);
        count         = numel(valid_values);

        number_of_valid_values(column_index)    = count;
        number_of_missing_values(column_index)  = sum(value_is_missing);
        number_of_zeros(column_index)           = sum(valid_values == 0);
        number_of_distinct_values(column_index) = numel(unique(valid_values));

        minimum(column_index)        = sorted_values(1);
        maximum(column_index)        = sorted_values(count);
        first_quartile(column_index) = compute_percentile(sorted_values, 0.25);
        median_value(column_index)   = compute_percentile(sorted_values, 0.50);
        third_quartile(column_index) = compute_percentile(sorted_values, 0.75);

        column_mean     = sum(valid_values) / count;
        centered_values = valid_values - column_mean;

        mean_value(column_index) = column_mean;
        standard_deviation(column_index) = sqrt(sum(centered_values .^ 2) / (count - 1));

        population_standard_deviation = sqrt(sum(centered_values .^ 2) / count);

        if population_standard_deviation > 0
            skewness(column_index) = (sum(centered_values .^ 3) / count) / population_standard_deviation ^ 3;
        else
            skewness(column_index) = 0;
        end
    end

    column_summary.number_of_valid_values    = number_of_valid_values;
    column_summary.number_of_missing_values  = number_of_missing_values;
    column_summary.number_of_zeros           = number_of_zeros;
    column_summary.number_of_distinct_values = number_of_distinct_values;
    column_summary.minimum                   = minimum;
    column_summary.first_quartile            = first_quartile;
    column_summary.median                    = median_value;
    column_summary.third_quartile            = third_quartile;
    column_summary.maximum                   = maximum;
    column_summary.mean                      = mean_value;
    column_summary.standard_deviation        = standard_deviation;
    column_summary.skewness                  = skewness;
end


function percentile_value = compute_percentile(sorted_values, probability)
    % Percentile of an already sorted vector, with linear interpolation.
    %
    % Same definition as numpy.percentile (default "linear" method).
    %
    % Parameters
    % ----------
    % sorted_values : double vector
    %     Values sorted in ascending order, without NaN.
    % probability : double in [0, 1]
    %     0.25 gives the first quartile, 0.5 the median, and so on.
    %
    % Returns
    % -------
    % percentile_value : double
    %     Interpolated value at the requested probability.

    number_of_values = numel(sorted_values);

    exact_position = 1 + (number_of_values - 1) * probability;
    lower_position = floor(exact_position);
    upper_position = ceil(exact_position);
    weight_of_upper_value = exact_position - lower_position;

    percentile_value = (1 - weight_of_upper_value) * sorted_values(lower_position) + weight_of_upper_value  * sorted_values(upper_position);
end


function print_column_summary(variable_names, column_summary)
    % Print the output of compute_column_summary as two readable tables.
    %
    % Parameters
    % ----------
    % variable_names : 1 x K cell of char
    %     Name of each column, in the same order as the summary.
    % column_summary : struct
    %     Output of compute_column_summary.

    fprintf('Position (original units of each variable)\n');
    fprintf('%-22s %10s %10s %10s %10s %10s\n', 'Variable', 'Min', 'Q1', 'Median', 'Q3', 'Max');

    for variable_index = 1:numel(variable_names)
        fprintf('%-22s %10.3f %10.3f %10.3f %10.3f %10.3f\n', ...
                variable_names{variable_index}, ...
                column_summary.minimum(variable_index), ...
                column_summary.first_quartile(variable_index), ...
                column_summary.median(variable_index), ...
                column_summary.third_quartile(variable_index), ...
                column_summary.maximum(variable_index));
    end

    fprintf('\nSpread, shape and data quality\n');
    fprintf('%-22s %10s %10s %10s %8s %8s %8s\n', 'Variable', 'Mean', 'Std', 'Skewness', 'Zeros', 'Distinct', 'Missing');

    for variable_index = 1:numel(variable_names)
        fprintf('%-22s %10.3f %10.3f %10.3f %8d %8d %8d\n', ...
                variable_names{variable_index}, ...
                column_summary.mean(variable_index), ...
                column_summary.standard_deviation(variable_index), ...
                column_summary.skewness(variable_index), ...
                column_summary.number_of_zeros(variable_index), ...
                column_summary.number_of_distinct_values(variable_index), ...
                column_summary.number_of_missing_values(variable_index));
    end
end


function print_section_title(title_text)
    % Print a title followed by an underline, with a blank line before.
    %
    % Parameters
    % ----------
    % title_text : char
    %     Text of the section title.

    fprintf('\n%s\n', title_text);
    fprintf('%s\n', repmat('=', 1, length(title_text)));
end


%% ========================================================================
%  STEP 2 - EXPLORATORY DATA ANALYSIS: CONCRETE
%  Functions used only by the Concrete dataset.
%  ========================================================================

function eda_report = explore_concrete_dataset(dataset, figures_folder)
    % Exploratory analysis of the Concrete dataset (prints and saves plots).
    %
    % Produces: histogram of every variable (inputs and target),
    % Pearson and Spearman correlation matrices, and scatter plots of every
    % input against the target. A ninth scatter plot uses log10(Age) to
    % check whether that transformation (Step 3) linearises the relation.
    %
    % Parameters
    % ----------
    % dataset : struct
    %     Output of load_concrete_dataset.
    % figures_folder : char
    %     Folder where the PNG figures are saved (created if missing).
    %
    % Returns
    % -------
    % eda_report : struct
    %     pearson_matrix, spearman_matrix : 9 x 9 correlation matrices
    %     correlation_of_age_with_target      : Pearson r of Age and target
    %     correlation_of_log_age_with_target  : Pearson r of log10(Age)
    %     figure_file_names                   : cell of saved PNG file names

    ensure_folder_exists(figures_folder);

    number_of_inputs = size(dataset.inputs, 2);
    target_column_index = number_of_inputs + 1;

    all_columns     = [dataset.inputs, dataset.target];
    all_short_names = [dataset.input_short_names, {dataset.target_short_name}];

    % ---- Correlations -----------------------------------------------------
    pearson_matrix  = compute_correlation_matrix(all_columns);
    spearman_matrix = compute_spearman_correlation_matrix(all_columns);

    print_section_title('CONCRETE - Pearson correlation matrix');
    print_correlation_matrix(all_short_names, pearson_matrix);

    print_section_title('CONCRETE - Spearman correlation matrix');
    print_correlation_matrix(all_short_names, spearman_matrix);

    print_section_title('CONCRETE - correlation of each input with the target');
    fprintf('%-22s %10s %10s\n', 'Input', 'Pearson', 'Spearman');

    for input_index = 1:number_of_inputs
        fprintf('%-22s %10.3f %10.3f\n', dataset.input_names{input_index}, ...
                pearson_matrix(input_index, target_column_index), ...
                spearman_matrix(input_index, target_column_index));
    end

    % ---- Strongly correlated pairs of inputs (collinearity) ---------------
    strong_correlation_threshold = 0.5;
    print_section_title('CONCRETE - strongly correlated pairs of inputs');
    fprintf('Pairs of inputs with |Pearson r| >= %.1f:\n', strong_correlation_threshold);

    number_of_strong_pairs = 0;

    for first_input = 1:number_of_inputs - 1
        for second_input = first_input + 1:number_of_inputs
            pair_correlation = pearson_matrix(first_input, second_input);
            if abs(pair_correlation) >= strong_correlation_threshold
                fprintf('  %-20s with %-20s r = %6.3f\n', ...
                        dataset.input_names{first_input}, ...
                        dataset.input_names{second_input}, pair_correlation);
                number_of_strong_pairs = number_of_strong_pairs + 1;
            end
        end
    end

    if number_of_strong_pairs == 0
        fprintf('  (none)\n');
    end

    % ---- Effect of water ratios -------------------------------------------
    cement_values = dataset.inputs(:, strcmp(dataset.input_names, 'Cement'));
    slag_values   = dataset.inputs(:, strcmp(dataset.input_names, 'Blast furnace slag'));
    ash_values    = dataset.inputs(:, strcmp(dataset.input_names, 'Fly ash'));
    water_values  = dataset.inputs(:, strcmp(dataset.input_names, 'Water'));

    cement_to_water_ratio = cement_values ./ water_values;
    binder_to_water_ratio = (cement_values + slag_values + ash_values) ./ water_values;

    correlation_of_cement_to_water_ratio_with_target = compute_pearson_correlation(cement_to_water_ratio, dataset.target);
    correlation_of_binder_to_water_ratio_with_target = compute_pearson_correlation(binder_to_water_ratio, dataset.target);

    print_section_title('CONCRETE - effect of water ratios');
    fprintf('Pearson r (Cement / Water,                strength) : %6.3f\n', correlation_of_cement_to_water_ratio_with_target);
    fprintf('Pearson r ((Cement + Slag + Ash) / Water, strength) : %6.3f\n', correlation_of_binder_to_water_ratio_with_target);

    % ---- Effect of log10(Age) ---------------------------------------------
    age_column_index = find(strcmp(dataset.input_names, 'Age'));
    age_values = dataset.inputs(:, age_column_index);

    correlation_of_age_with_target = compute_pearson_correlation(age_values, dataset.target);
    correlation_of_log_age_with_target = compute_pearson_correlation(log10(age_values), dataset.target);

    print_section_title('CONCRETE - effect of a log transformation of Age');
    fprintf('Pearson r (Age,         strength) : %6.3f\n', correlation_of_age_with_target);
    fprintf('Pearson r (log10(Age),  strength) : %6.3f\n', correlation_of_log_age_with_target);

    % ---- Figures -----------------------------------------------------------
    print_section_title('CONCRETE - figures');

    histogram_figure = plot_concrete_histograms(dataset);
    save_figure(histogram_figure, figures_folder, 'step2_concrete_histograms');

    correlation_figure = plot_concrete_correlation_matrix(pearson_matrix, all_short_names);
    save_figure(correlation_figure, figures_folder, 'step2_concrete_correlation_matrix');

    scatter_figure = plot_concrete_scatter_plots(dataset);
    save_figure(scatter_figure, figures_folder, 'step2_concrete_scatter_vs_target');

    % ---- Report -----------------------------------------------------------
    eda_report.pearson_matrix  = pearson_matrix;
    eda_report.spearman_matrix = spearman_matrix;
    eda_report.correlation_of_age_with_target     = correlation_of_age_with_target;
    eda_report.correlation_of_log_age_with_target = correlation_of_log_age_with_target;
    eda_report.figure_file_names = {'step2_concrete_histograms.png', ...
                                    'step2_concrete_correlation_matrix.png', ...
                                    'step2_concrete_scatter_vs_target.png'};
end


function figure_handle = plot_concrete_histograms(dataset)
    % Histogram of the 8 inputs and of the target (3 x 3 grid).
    %
    % Parameters
    % ----------
    % dataset : struct
    %     Output of load_concrete_dataset.
    %
    % Returns
    % -------
    % figure_handle : handle of the created figure.

    number_of_bins = 20;

    all_columns = [dataset.inputs, dataset.target];
    all_names   = [dataset.input_names, {dataset.target_name}];
    all_units   = [dataset.input_units, {dataset.target_unit}];

    figure_handle = figure('Name', 'Concrete - histograms', 'Position', [50 50 1100 800]);

    for variable_index = 1:size(all_columns, 2)
        subplot(3, 3, variable_index);

        [counts, bin_centers] = hist(all_columns(:, variable_index), number_of_bins);
        bar_handle = bar(bin_centers, counts, 1);
        set(bar_handle, 'FaceColor', [0.30 0.50 0.80]);

        xlabel(sprintf('%s (%s)', all_names{variable_index}, all_units{variable_index}));
        ylabel('Count');
        set(gca, 'FontSize', 8);
        grid on;
    end
end


function figure_handle = plot_concrete_correlation_matrix(correlation_matrix, short_names)
    % Heat map of a correlation matrix with the value written in each cell.
    %
    % Parameters
    % ----------
    % correlation_matrix : double matrix, K x K
    %     Pearson correlation matrix.
    % short_names : 1 x K cell of char
    %     Short labels for both axes.
    %
    % Returns
    % -------
    % figure_handle : handle of the created figure.

    number_of_variables = numel(short_names);

    figure_handle = figure('Name', 'Concrete - correlation matrix', 'Position', [100 100 750 650]);

    imagesc(correlation_matrix);
    set(gca, 'CLim', [-1 1]);
    colormap(build_blue_white_red_colormap(64));
    colorbar;

    set(gca, 'XTick', 1:number_of_variables, 'XTickLabel', short_names, ...
             'YTick', 1:number_of_variables, 'YTickLabel', short_names);
    axis square;
    title('Pearson correlation matrix');

    for row_index = 1:number_of_variables
        for column_index = 1:number_of_variables
            cell_value = correlation_matrix(row_index, column_index);
            if abs(cell_value) > 0.6
                text_color = [1 1 1];
            else
                text_color = [0 0 0];
            end
            text(column_index, row_index, sprintf('%.2f', cell_value), ...
                 'HorizontalAlignment', 'center', 'FontSize', 8, ...
                 'Color', text_color);
        end
    end
end


function figure_handle = plot_concrete_scatter_plots(dataset)
    % Scatter plots of each input against the target (3 x 3 grid).
    %
    % The first 8 panels use the raw inputs; the 9th uses log10(Age).
    % The title of every panel shows the Pearson correlation r.
    %
    % Parameters
    % ----------
    % dataset : struct
    %     Output of load_concrete_dataset.
    %
    % Returns
    % -------
    % figure_handle : handle of the created figure.

    number_of_inputs = size(dataset.inputs, 2);
    target_label = sprintf('%s (%s)', dataset.target_name, dataset.target_unit);

    age_column_index = find(strcmp(dataset.input_names, 'Age'));

    figure_handle = figure('Name', 'Concrete - inputs versus target', 'Position', [50 50 1100 800]);

    for input_index = 1:number_of_inputs
        subplot(3, 3, input_index);

        input_values = dataset.inputs(:, input_index);
        plot(input_values, dataset.target, '.', 'MarkerSize', 4, 'Color', [0.20 0.40 0.70]);

        correlation = compute_pearson_correlation(input_values, dataset.target);
        title(sprintf('r = %.2f', correlation), 'FontSize', 9);
        xlabel(sprintf('%s (%s)', dataset.input_names{input_index}, dataset.input_units{input_index}));
        ylabel(target_label);
        set(gca, 'FontSize', 8);
        grid on;
    end

    subplot(3, 3, number_of_inputs + 1);
    log_age_values = log10(dataset.inputs(:, age_column_index));
    plot(log_age_values, dataset.target, '.', 'MarkerSize', 4, 'Color', [0.80 0.30 0.20]);

    correlation = compute_pearson_correlation(log_age_values, dataset.target);
    title(sprintf('r = %.2f', correlation), 'FontSize', 9);
    xlabel('log10(Age) (log10 days)');
    ylabel(target_label);
    set(gca, 'FontSize', 8);
    grid on;
end


function correlation_matrix = compute_correlation_matrix(data_matrix)
    % Pearson correlation matrix of the columns of a matrix.
    %
    % Parameters
    % ----------
    % data_matrix : double matrix, N x K
    %     One variable per column (no NaN).
    %
    % Returns
    % -------
    % correlation_matrix : double matrix, K x K

    number_of_rows = size(data_matrix, 1);

    column_means  = mean(data_matrix, 1);
    centered_data = data_matrix - repmat(column_means, number_of_rows, 1);

    covariance_matrix  = (centered_data' * centered_data) / (number_of_rows - 1);
    standard_deviation = sqrt(diag(covariance_matrix));

    correlation_matrix = covariance_matrix ./ (standard_deviation * standard_deviation');
end


function ranks = compute_average_ranks(values)
    % Ranks of a vector, giving tied values the average of their ranks.
    %
    % Parameters
    % ----------
    % values : double vector
    %
    % Returns
    % -------
    % ranks : double column, same length as values
    %     1 for the smallest value. Example: [10 20 20 30] -> [1 2.5 2.5 4].

    values = values(:);
    number_of_values = numel(values);

    [sorted_values, sort_order] = sort(values);

    ranks_in_sorted_order = zeros(number_of_values, 1);

    group_start = 1;

    while group_start <= number_of_values
        group_end = group_start;

        while group_end < number_of_values && sorted_values(group_end + 1) == sorted_values(group_start)
            group_end = group_end + 1;
        end

        ranks_in_sorted_order(group_start:group_end) = (group_start + group_end) / 2;
        group_start = group_end + 1;
    end

    ranks = zeros(number_of_values, 1);
    ranks(sort_order) = ranks_in_sorted_order;
end


function correlation_matrix = compute_spearman_correlation_matrix(data_matrix)
    % Spearman (rank) correlation matrix of the columns of a matrix.
    %
    % It measures monotonic (not only linear) association, which is useful
    % for variables such as the concrete age.
    %
    % Parameters
    % ----------
    % data_matrix : double matrix, N x K
    %
    % Returns
    % -------
    % correlation_matrix : double matrix, K x K

    ranked_matrix = zeros(size(data_matrix));

    for column_index = 1:size(data_matrix, 2)
        ranked_matrix(:, column_index) = compute_average_ranks(data_matrix(:, column_index));
    end

    correlation_matrix = compute_correlation_matrix(ranked_matrix);
end


function print_correlation_matrix(short_names, correlation_matrix)
    % Print a correlation matrix with row and column labels.
    %
    % Parameters
    % ----------
    % short_names : 1 x K cell of char
    %     Labels of the variables (at most 5 characters each).
    % correlation_matrix : double matrix, K x K

    number_of_variables = numel(short_names);

    fprintf('%-6s', '');
    for column_index = 1:number_of_variables
        fprintf('%6s', short_names{column_index});
    end
    fprintf('\n');

    for row_index = 1:number_of_variables
        fprintf('%-6s', short_names{row_index});
        for column_index = 1:number_of_variables
            fprintf('%6.2f', correlation_matrix(row_index, column_index));
        end
        fprintf('\n');
    end
end


function color_map = build_blue_white_red_colormap(number_of_colors_per_half)
    % Diverging colour map: blue (-1), white (0), red (+1).
    %
    % Parameters
    % ----------
    % number_of_colors_per_half : integer
    %     Colours between blue and white (and between white and red).
    %
    % Returns
    % -------
    % color_map : double matrix, (2 * number_of_colors_per_half) x 3

    ramp_up   = linspace(0, 1, number_of_colors_per_half)';
    ramp_down = linspace(1, 0, number_of_colors_per_half)';
    ones_column = ones(number_of_colors_per_half, 1);

    blue_to_white = [ramp_up,      ramp_up,      ones_column];
    white_to_red  = [ones_column,  ramp_down,    ramp_down];

    color_map = [blue_to_white; white_to_red];
end


%% ========================================================================
%  STEP 2 - EXPLORATORY DATA ANALYSIS: MELBOURNE
%  Functions used only by the Melbourne dataset.
%  ========================================================================

function eda_report = explore_melbourne_dataset(dataset, figures_folder)
    % Exploratory analysis of the Melbourne series (prints and saves plots).
    %
    % Produces: plot of the whole series, distribution of the values,
    % monthly behaviour (box plots), annual cycle (daily climatology and a
    % fitted sinusoid), autocorrelation (ACF, implemented here) and partial
    % autocorrelation (PACF), scatter plot of T(t) against T(t+1), and
    % checks for abrupt changes and suspicious values.
    %
    % Parameters
    % ----------
    % dataset : struct
    %     Output of load_melbourne_dataset.
    % figures_folder : char
    %     Folder where the PNG figures are saved (created if missing).
    %
    % Returns
    % -------
    % eda_report : struct
    %     monthly_mean, monthly_standard_deviation : 12 x 1 vectors
    %     annual_fit             : output of fit_annual_harmonic
    %     autocorrelation_long   : ACF for lags 0..400 (column)
    %     autocorrelation_short  : ACF for lags 0..30 (column)
    %     anomaly_autocorrelation_short : ACF of the de-seasonalised series
    %     partial_autocorrelation_short : PACF for lags 0..30
    %     significant_pacf_lags  : lags in 1..30 whose |PACF| exceeds 95 % bound
    %     lag_one_correlation    : correlation between T(t) and T(t+1)
    %     persistence_rmse       : RMSE of the forecast T(t+1) = T(t), full series
    %     flagged_anomaly_dates  : dates with |anomaly| > 3 standard deviations
    %     figure_file_names      : cell of saved PNG file names

    ensure_folder_exists(figures_folder);

    dates               = dataset.dates;
    temperature_celsius = dataset.temperature_celsius;
    number_of_observations = numel(dates);

    maximum_lag_long  = 400;
    maximum_lag_short = 30;

    date_vectors     = datevec(dates);
    month_of_each_row = date_vectors(:, 2);

    % ---- 1. Monthly (seasonal) behaviour ----------------------------------
    month_labels = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', ...
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};

    monthly_values             = cell(1, 12);
    monthly_mean               = zeros(12, 1);
    monthly_standard_deviation = zeros(12, 1);

    print_section_title('MELBOURNE - monthly behaviour (all years pooled)');
    fprintf('%-6s %8s %8s %8s %8s %8s\n', 'Month', 'Count', 'Mean', 'Std', 'Min', 'Max');

    for month_number = 1:12
        values_of_month = temperature_celsius(month_of_each_row == month_number);

        monthly_values{month_number}             = values_of_month;
        monthly_mean(month_number)               = mean(values_of_month);
        monthly_standard_deviation(month_number) = std(values_of_month);

        fprintf('%-6s %8d %8.2f %8.2f %8.2f %8.2f\n', month_labels{month_number}, ...
                numel(values_of_month), monthly_mean(month_number), ...
                monthly_standard_deviation(month_number), ...
                min(values_of_month), max(values_of_month));
    end

    % ---- 2. Annual cycle -----------------------------------------------------
    annual_fit = fit_annual_harmonic(dates, temperature_celsius);
    anomalies  = annual_fit.residuals;

    observed_climatology = compute_daily_climatology(dates, temperature_celsius);
    fitted_climatology   = compute_daily_climatology(dates, annual_fit.fitted_values);

    print_section_title('MELBOURNE - annual cycle (sinusoid with a 365.2425-day period)');
    fprintf('Mean level                      : %.2f degC\n', annual_fit.mean_level);
    fprintf('Amplitude                       : %.2f degC\n', annual_fit.amplitude);
    fprintf('Warmest day of the cycle        : %s\n', annual_fit.warmest_day_text);
    fprintf('Variance explained by the cycle : %.1f %%\n', 100 * annual_fit.explained_variance);
    fprintf('Std of the series               : %.2f degC\n', std(temperature_celsius));
    fprintf('Std of the anomalies (noise)    : %.2f degC\n', annual_fit.residual_standard_deviation);

    % ---- 3. Autocorrelation --------------------------------------------------
    autocorrelation_long  = compute_autocorrelation_function(temperature_celsius, maximum_lag_long);
    autocorrelation_short = autocorrelation_long(1:maximum_lag_short + 1);
    anomaly_autocorrelation_short = compute_autocorrelation_function(anomalies, maximum_lag_short);
    partial_autocorrelation_short = compute_partial_autocorrelation_function(autocorrelation_short);

    confidence_bound = 1.96 / sqrt(number_of_observations);
    significant_pacf_lags = find(abs(partial_autocorrelation_short(2:maximum_lag_short + 1)) > confidence_bound)';

    print_section_title('MELBOURNE - autocorrelation');
    fprintf('Approximate 95%% bound for white noise: +/- %.3f\n\n', confidence_bound);
    fprintf('%-6s %12s %12s %16s\n', 'Lag', 'ACF (raw)', 'PACF (raw)', 'ACF (anomalies)');

    lags_to_print = [1 2 3 4 5 6 7 10 15 20 30];

    for lag_index = 1:numel(lags_to_print)
        current_lag = lags_to_print(lag_index);
        fprintf('%-6d %12.3f %12.3f %16.3f\n', current_lag, ...
                autocorrelation_short(current_lag + 1), ...
                partial_autocorrelation_short(current_lag + 1), ...
                anomaly_autocorrelation_short(current_lag + 1));
    end

    fprintf('\nACF (raw) at lag 182 : %6.3f   (half a year)\n', autocorrelation_long(182 + 1));
    fprintf('ACF (raw) at lag 365 : %6.3f   (one year)\n',      autocorrelation_long(365 + 1));
    fprintf('Lags 1..%d with |PACF| above the bound: ', maximum_lag_short);
    fprintf('%d ', significant_pacf_lags);
    fprintf('\n');

    % ---- 4. T(t) against T(t+1) ------------------------------------------------
    % Only pairs of consecutive calendar days are used (the 2 missing days would otherwise create pairs that are 2 days apart).
    days_between_rows   = diff(dates);
    index_of_today      = find(days_between_rows == 1);
    temperature_today    = temperature_celsius(index_of_today);
    temperature_tomorrow = temperature_celsius(index_of_today + 1);

    lag_one_correlation = compute_pearson_correlation(temperature_today, temperature_tomorrow);

    design_matrix = [ones(numel(temperature_today), 1), temperature_today];
    line_coefficients = design_matrix \ temperature_tomorrow;

    persistence_rmse = sqrt(mean((temperature_tomorrow - temperature_today) .^ 2));

    print_section_title('MELBOURNE - T(t) against T(t+1)');
    fprintf('Pairs of consecutive days          : %d\n', numel(index_of_today));
    fprintf('Correlation r(T(t), T(t+1))        : %.3f\n', lag_one_correlation);
    fprintf('Least-squares line                 : T(t+1) = %.3f + %.3f * T(t)\n', line_coefficients(1), line_coefficients(2));
    fprintf('RMSE of persistence (whole series) : %.3f degC\n', persistence_rmse);
    fprintf('(This is only context. The official baseline is computed on the test set in Step 8.)\n');

    % ---- 5. Abrupt changes and suspicious values -------------------------------
    print_section_title('MELBOURNE - abrupt changes and suspicious values');

    temperature_changes = diff(temperature_celsius);
    temperature_changes(days_between_rows ~= 1) = NaN;

    index_of_valid_changes = find(~isnan(temperature_changes));
    [unused_sorted_changes, sort_order] = sort(abs(temperature_changes(index_of_valid_changes)), 'descend');
    number_of_changes_to_list = min(5, numel(sort_order));

    fprintf('Std of the day-to-day change : %.2f degC\n', std(temperature_changes(index_of_valid_changes)));
    fprintf('Five largest day-to-day changes:\n');

    for rank_index = 1:number_of_changes_to_list
        row_before_change = index_of_valid_changes(sort_order(rank_index));
        fprintf('  %s -> %s : %5.1f -> %5.1f  (change %+5.1f degC)\n', ...
                datestr(dates(row_before_change), 'yyyy-mm-dd'), ...
                datestr(dates(row_before_change + 1), 'yyyy-mm-dd'), ...
                temperature_celsius(row_before_change), ...
                temperature_celsius(row_before_change + 1), ...
                temperature_changes(row_before_change));
    end

    anomaly_threshold = 3 * annual_fit.residual_standard_deviation;
    rows_with_large_anomaly = find(abs(anomalies) > anomaly_threshold);

    [unused_sorted_values, anomaly_order] = sort(abs(anomalies(rows_with_large_anomaly)), 'descend');
    rows_with_large_anomaly = rows_with_large_anomaly(anomaly_order);

    fprintf('\nValues more than 3 std away from the annual cycle (|anomaly| > %.2f degC): %d\n', anomaly_threshold, numel(rows_with_large_anomaly));
    number_of_anomalies_to_list = min(10, numel(rows_with_large_anomaly));

    for list_index = 1:number_of_anomalies_to_list
        row_index = rows_with_large_anomaly(list_index);
        fprintf('  %s : %5.1f degC (anomaly %+5.1f)\n', ...
                datestr(dates(row_index), 'yyyy-mm-dd'), ...
                temperature_celsius(row_index), anomalies(row_index));
    end

    [longest_run_length, longest_run_end_row] = compute_longest_constant_run(temperature_celsius);

    fprintf('\nLongest run of identical consecutive values: %d days (value %.1f, ending %s)\n', ...
            longest_run_length, temperature_celsius(longest_run_end_row), ...
            datestr(dates(longest_run_end_row), 'yyyy-mm-dd'));

    lowest_plausible_value  = -10;
    highest_plausible_value =  40;

    number_outside_plausible_range = sum(temperature_celsius < lowest_plausible_value | temperature_celsius > highest_plausible_value);
    fprintf('Values outside [%d, %d] degC: %d\n', lowest_plausible_value, highest_plausible_value, number_outside_plausible_range);

    value_is_off_grid = abs(temperature_celsius * 10 - round(temperature_celsius * 10)) > 1e-6;
    fprintf('Values that are not multiples of 0.1 degC: %d\n', sum(value_is_off_grid));

    % ---- 6. Figures ---------------------------------------------------------------
    print_section_title('MELBOURNE - figures');

    series_figure = plot_melbourne_series(dataset);
    save_figure(series_figure, figures_folder, 'step2_melbourne_series');

    distribution_figure = plot_melbourne_distribution(temperature_celsius);
    save_figure(distribution_figure, figures_folder, 'step2_melbourne_distribution');

    monthly_figure = plot_melbourne_monthly_box_plots(monthly_values, month_labels);
    save_figure(monthly_figure, figures_folder, 'step2_melbourne_monthly_box_plots');

    annual_cycle_figure = plot_melbourne_annual_cycle(observed_climatology, fitted_climatology);
    save_figure(annual_cycle_figure, figures_folder, 'step2_melbourne_annual_cycle');

    autocorrelation_figure = plot_melbourne_autocorrelation(autocorrelation_long, ...
        autocorrelation_short, anomaly_autocorrelation_short, ...
        partial_autocorrelation_short, confidence_bound);
    save_figure(autocorrelation_figure, figures_folder, 'step2_melbourne_autocorrelation');

    lag_scatter_figure = plot_melbourne_lag_scatter(temperature_today, ...
        temperature_tomorrow, lag_one_correlation, line_coefficients);
    save_figure(lag_scatter_figure, figures_folder, 'step2_melbourne_lag_scatter');

    abrupt_changes_figure = plot_melbourne_abrupt_changes(dates, anomalies, ...
        anomaly_threshold, rows_with_large_anomaly, ...
        temperature_changes(index_of_valid_changes));
    save_figure(abrupt_changes_figure, figures_folder, 'step2_melbourne_abrupt_changes');

    % ---- Report -----------------------------------------------------------
    eda_report.monthly_mean                  = monthly_mean;
    eda_report.monthly_standard_deviation    = monthly_standard_deviation;
    eda_report.annual_fit                    = annual_fit;
    eda_report.autocorrelation_long          = autocorrelation_long;
    eda_report.autocorrelation_short         = autocorrelation_short;
    eda_report.anomaly_autocorrelation_short = anomaly_autocorrelation_short;
    eda_report.partial_autocorrelation_short = partial_autocorrelation_short;
    eda_report.significant_pacf_lags         = significant_pacf_lags;
    eda_report.lag_one_correlation           = lag_one_correlation;
    eda_report.persistence_rmse              = persistence_rmse;
    eda_report.flagged_anomaly_dates         = dates(rows_with_large_anomaly);
    eda_report.figure_file_names = {'step2_melbourne_series.png', ...
                                    'step2_melbourne_distribution.png', ...
                                    'step2_melbourne_monthly_box_plots.png', ...
                                    'step2_melbourne_annual_cycle.png', ...
                                    'step2_melbourne_autocorrelation.png', ...
                                    'step2_melbourne_lag_scatter.png', ...
                                    'step2_melbourne_abrupt_changes.png'};
end


function figure_handle = plot_melbourne_series(dataset)
    % Whole series (top) and a one-year zoom (bottom).
    %
    % Repaired values are marked with red circles and the missing calendar
    % days with black triangles on the 0 degC line.
    %
    % Parameters
    % ----------
    % dataset : struct
    %     Output of load_melbourne_dataset.
    %
    % Returns
    % -------
    % figure_handle : handle of the created figure.

    dates               = dataset.dates;
    temperature_celsius = dataset.temperature_celsius;

    row_is_repaired = ismember(dates, dataset.repaired_dates);
    missing_dates   = find_missing_calendar_days(dates);

    figure_handle = figure('Name', 'Melbourne - whole series', 'Position', [50 50 1200 700]);

    subplot(2, 1, 1);
    plot(dates, temperature_celsius, '-', 'Color', [0.20 0.40 0.70]);
    hold on;
    plot(dates(row_is_repaired), temperature_celsius(row_is_repaired), 'ro', 'MarkerSize', 6, 'LineWidth', 1.5);
    plot(missing_dates, zeros(size(missing_dates)), 'kv', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
    hold off;
    datetick('x', 'yyyy');
    ylabel('Minimum temperature (degC)');
    title('Daily minimum temperature, Melbourne, 1981-1990');
    legend('Daily minimum', 'Repaired value ("?")', 'Missing day (drawn at 0)', 'Location', 'NorthEast');
    grid on;

    subplot(2, 1, 2);
    zoom_rows = 1:365;
    plot(dates(zoom_rows), temperature_celsius(zoom_rows), '.-', 'Color', [0.20 0.40 0.70]);
    set(gca, 'XLim', [dates(zoom_rows(1)), dates(zoom_rows(end))]);
    datetick('x', 'mmm', 'keeplimits');
    ylabel('Minimum temperature (degC)');
    title('Zoom: first 365 days (1981)');
    grid on;
end


function figure_handle = plot_melbourne_distribution(temperature_celsius)
    % Histogram of all daily minimum temperatures, with mean and median.
    %
    % Parameters
    % ----------
    % temperature_celsius : double vector
    %     Daily minimum temperatures in degC.
    %
    % Returns
    % -------
    % figure_handle : handle of the created figure.

    number_of_bins = 30;

    figure_handle = figure('Name', 'Melbourne - distribution', 'Position', [100 100 800 500]);

    [counts, bin_centers] = hist(temperature_celsius, number_of_bins);
    bar_handle = bar(bin_centers, counts, 1);
    set(bar_handle, 'FaceColor', [0.30 0.50 0.80]);
    hold on;

    y_limits = get(gca, 'YLim');
    plot([mean(temperature_celsius) mean(temperature_celsius)], y_limits, 'r-', 'LineWidth', 2);
    plot([median(temperature_celsius) median(temperature_celsius)], y_limits, 'k--', 'LineWidth', 2);
    hold off;

    xlabel('Minimum temperature (degC)');
    ylabel('Number of days');
    title('Distribution of the daily minimum temperature');
    legend('Histogram', 'Mean', 'Median', 'Location', 'NorthEast');
    grid on;
end


function figure_handle = plot_melbourne_monthly_box_plots(monthly_values, month_labels)
    % One box plot per calendar month (all years pooled).
    %
    % Parameters
    % ----------
    % monthly_values : 1 x 12 cell of double vectors
    %     Temperatures of each month.
    % month_labels : 1 x 12 cell of char
    %     Label for each month.
    %
    % Returns
    % -------
    % figure_handle : handle of the created figure.

    figure_handle = figure('Name', 'Melbourne - monthly behaviour', 'Position', [100 100 900 500]);

    draw_box_plot(monthly_values, month_labels);

    ylabel('Minimum temperature (degC)');
    title('Monthly distribution of the daily minimum temperature (1981-1990)');
    grid on;
end


function figure_handle = plot_melbourne_annual_cycle(observed_climatology, fitted_climatology)
    % Mean temperature for each day of the year, with a fitted sinusoid.
    %
    % Parameters
    % ----------
    % observed_climatology : struct
    %     compute_daily_climatology of the observed temperatures.
    % fitted_climatology : struct
    %     compute_daily_climatology of the fitted sinusoid.
    %
    % Returns
    % -------
    % figure_handle : handle of the created figure.

    day_numbers = observed_climatology.day_of_year;

    smoothed_mean = observed_climatology.smoothed_mean;
    smoothed_std  = circular_moving_average(observed_climatology.standard_deviation, 15);

    figure_handle = figure('Name', 'Melbourne - annual cycle', 'Position', [100 100 900 550]);

    handle_of_daily_means = plot(day_numbers, observed_climatology.mean, '.', 'Color', [0.60 0.60 0.60]);

    hold on;
    handle_of_smoothed_mean = plot(day_numbers, smoothed_mean, '-', 'Color', [0.20 0.40 0.70], 'LineWidth', 2);

    handle_of_std_band = plot(day_numbers, smoothed_mean + smoothed_std, '--', 'Color', [0.20 0.40 0.70]);

    plot(day_numbers, smoothed_mean - smoothed_std, '--', 'Color', [0.20 0.40 0.70]);

    handle_of_fitted_cycle = plot(day_numbers, fitted_climatology.mean, 'r-', 'LineWidth', 2);
    hold off;

    xlabel('Day of the year');
    ylabel('Minimum temperature (degC)');
    title('Annual cycle: mean of the 10 years for each day of the year');
    legend([handle_of_daily_means, handle_of_smoothed_mean, ...
            handle_of_std_band, handle_of_fitted_cycle], ...
           'Mean of each day (10 values)', '15-day smoothed mean', ...
           'Smoothed mean +/- 1 std', 'Fitted sinusoid', 'Location', 'North');
    set(gca, 'XLim', [1 365]);
    grid on;
end


function figure_handle = plot_melbourne_autocorrelation(autocorrelation_long, ...
        autocorrelation_short, anomaly_autocorrelation_short, ...
        partial_autocorrelation_short, confidence_bound)
    % ACF over 400 lags, ACF/PACF over 30 lags and ACF of the anomalies.
    %
    % Parameters
    % ----------
    % autocorrelation_long : double column, lags 0..400
    % autocorrelation_short : double column, lags 0..30
    % anomaly_autocorrelation_short : double column, lags 0..30
    % partial_autocorrelation_short : double column, lags 0..30
    % confidence_bound : double
    %     Approximate 95 % bound for white noise (1.96 / sqrt(N)).
    %
    % Returns
    % -------
    % figure_handle : handle of the created figure.

    maximum_lag_long  = numel(autocorrelation_long) - 1;
    maximum_lag_short = numel(autocorrelation_short) - 1;

    figure_handle = figure('Name', 'Melbourne - autocorrelation', 'Position', [50 50 1200 700]);

    subplot(2, 2, 1);
    draw_correlation_stems(1:maximum_lag_long, autocorrelation_long(2:end), confidence_bound);
    xlabel('Lag (days)');
    ylabel('ACF');
    title('ACF of the series, lags 1-400 (annual cycle near lag 365)');

    subplot(2, 2, 2);
    draw_correlation_stems(1:maximum_lag_short, autocorrelation_short(2:end), confidence_bound);
    xlabel('Lag (days)');
    ylabel('ACF');
    title('ACF of the series (lags 1-30)');

    subplot(2, 2, 3);
    draw_correlation_stems(1:maximum_lag_short, partial_autocorrelation_short(2:end), confidence_bound);
    xlabel('Lag (days)');
    ylabel('PACF');
    title('Partial ACF of the series (lags 1-30)');

    subplot(2, 2, 4);
    draw_correlation_stems(1:maximum_lag_short, anomaly_autocorrelation_short(2:end), confidence_bound);
    xlabel('Lag (days)');
    ylabel('ACF');
    title('ACF of the anomalies (series minus annual cycle)');
end


function figure_handle = plot_melbourne_lag_scatter(temperature_today, ...
        temperature_tomorrow, lag_one_correlation, line_coefficients)
    % Scatter plot of T(t) against T(t+1), with the diagonal and a fitted line.
    %
    % Parameters
    % ----------
    % temperature_today, temperature_tomorrow : double columns
    %     Pairs of consecutive days.
    % lag_one_correlation : double
    %     Pearson correlation of the pairs (shown in the title).
    % line_coefficients : 2 x 1 double
    %     Intercept and slope of the least-squares line.
    %
    % Returns
    % -------
    % figure_handle : handle of the created figure.

    figure_handle = figure('Name', 'Melbourne - T(t) against T(t+1)', 'Position', [100 100 700 650]);

    plot(temperature_today, temperature_tomorrow, '.', 'MarkerSize', 4, 'Color', [0.20 0.40 0.70]);
    hold on;

    axis_range = [min(temperature_today), max(temperature_today)];
    plot(axis_range, axis_range, 'k--', 'LineWidth', 1.5);
    plot(axis_range, line_coefficients(1) + line_coefficients(2) * axis_range, 'r-', 'LineWidth', 1.5);
    hold off;

    xlabel('T(t) (degC)');
    ylabel('T(t+1) (degC)');
    title(sprintf('T(t) against T(t+1), r = %.3f', lag_one_correlation));
    legend('Pairs of consecutive days', 'T(t+1) = T(t) (persistence)', ...
           'Least-squares line', 'Location', 'NorthWest');
    axis square;
    grid on;
end


function figure_handle = plot_melbourne_abrupt_changes(dates, anomalies, ...
        anomaly_threshold, rows_with_large_anomaly, day_to_day_changes)
    % Anomaly series with a 3-std band (top) and day-to-day changes (bottom).
    %
    % Parameters
    % ----------
    % dates : double column
    %     Serial date numbers.
    % anomalies : double column
    %     Series minus the fitted annual cycle.
    % anomaly_threshold : double
    %     3 standard deviations of the anomalies.
    % rows_with_large_anomaly : double vector
    %     Rows where |anomaly| exceeds the threshold.
    % day_to_day_changes : double vector
    %     T(t+1) - T(t) for pairs of consecutive days.
    %
    % Returns
    % -------
    % figure_handle : handle of the created figure.

    figure_handle = figure('Name', 'Melbourne - abrupt changes', 'Position', [50 50 1200 700]);

    subplot(2, 1, 1);
    plot(dates, anomalies, '-', 'Color', [0.20 0.40 0.70]);
    hold on;
    plot([dates(1) dates(end)], [anomaly_threshold anomaly_threshold], 'r--');
    plot([dates(1) dates(end)], [-anomaly_threshold -anomaly_threshold], 'r--');
    plot(dates(rows_with_large_anomaly), anomalies(rows_with_large_anomaly), 'ro', 'MarkerSize', 5);
    hold off;
    datetick('x', 'yyyy');
    ylabel('Anomaly (degC)');
    title('Anomalies (series minus annual cycle); dashed lines = +/- 3 std');
    grid on;

    subplot(2, 1, 2);
    [counts, bin_centers] = hist(day_to_day_changes, 40);
    bar_handle = bar(bin_centers, counts, 1);
    set(bar_handle, 'FaceColor', [0.30 0.50 0.80]);
    xlabel('T(t+1) - T(t) (degC)');
    ylabel('Number of days');
    title('Day-to-day change of the minimum temperature');
    grid on;
end


function autocorrelation = compute_autocorrelation_function(values, maximum_lag)
    % Sample autocorrelation function (ACF), implemented without toolbox.
    %
    % ACF(k) = sum_t (x(t)-m)(x(t+k)-m) / sum_t (x(t)-m)^2, with the same
    % denominator for all lags (the usual "biased" estimator, which
    % guarantees a valid ACF).
    %
    % Parameters
    % ----------
    % values : double vector
    %     The series, assumed to be equally spaced.
    % maximum_lag : integer
    %     Largest lag to compute.
    %
    % Returns
    % -------
    % autocorrelation : double column, length maximum_lag + 1
    %     Element k + 1 is the ACF at lag k (element 1 is lag 0, always 1).

    values = values(:);
    number_of_values = numel(values);

    centered_values = values - mean(values);
    sum_of_squares  = sum(centered_values .^ 2);

    autocorrelation = zeros(maximum_lag + 1, 1);

    for lag = 0:maximum_lag
        earlier_values = centered_values(1:number_of_values - lag);
        later_values   = centered_values(1 + lag:number_of_values);

        autocorrelation(lag + 1) = sum(earlier_values .* later_values) / sum_of_squares;
    end
end


function partial_autocorrelation = compute_partial_autocorrelation_function(autocorrelation)
    % Partial autocorrelation (PACF) from the ACF, by Durbin-Levinson.
    %
    % PACF(k) is the correlation between x(t) and x(t+k) after removing the
    % effect of the lags in between. It is the natural tool to choose the
    % window size p of an autoregressive model (Step 3).
    %
    % Parameters
    % ----------
    % autocorrelation : double column
    %     ACF from lag 0 to lag K (output of compute_autocorrelation_function).
    %
    % Returns
    % -------
    % partial_autocorrelation : double column, same length as the input
    %     Element k + 1 is the PACF at lag k (element 1 is lag 0, equal to 1).

    maximum_lag = numel(autocorrelation) - 1;

    partial_autocorrelation = zeros(maximum_lag + 1, 1);
    partial_autocorrelation(1) = 1;

    previous_coefficients = [];

    for lag = 1:maximum_lag
        if lag == 1
            last_coefficient = autocorrelation(2);
            current_coefficients = last_coefficient;
        else
            numerator = autocorrelation(lag + 1) - sum(previous_coefficients .* autocorrelation(lag:-1:2));
            denominator = 1 - sum(previous_coefficients .* autocorrelation(2:lag));
            last_coefficient = numerator / denominator;

            updated_coefficients = previous_coefficients - last_coefficient * previous_coefficients(end:-1:1);
            current_coefficients = [updated_coefficients; last_coefficient];
        end

        partial_autocorrelation(lag + 1) = last_coefficient;
        previous_coefficients = current_coefficients;
    end
end


function day_of_year = compute_day_of_year(dates)
    % Day of the year (1 = 1 January) of serial date numbers.
    %
    % Parameters
    % ----------
    % dates : double column
    %     MATLAB serial date numbers.
    %
    % Returns
    % -------
    % day_of_year : double column, values 1..366

    date_vectors = datevec(dates);
    number_of_dates = numel(dates);

    first_day_of_each_year = datenum(date_vectors(:, 1), ones(number_of_dates, 1), ones(number_of_dates, 1));

    day_of_year = dates(:) - first_day_of_each_year + 1;
end


function annual_fit = fit_annual_harmonic(dates, values)
    % Least-squares fit of a mean plus one sinusoid with a one-year period.
    %
    % Model: value(t) = a + b*cos(w*t) + c*sin(w*t), w = 2*pi/365.2425,
    % where t is the number of days since the first date.
    %
    % Parameters
    % ----------
    % dates : double column
    %     Serial date numbers.
    % values : double column
    %     Series to fit (same length as dates).
    %
    % Returns
    % -------
    % annual_fit : struct
    %     mean_level                  : a, in the unit of the series
    %     amplitude                   : sqrt(b^2 + c^2)
    %     warmest_day_text            : char, day of the year of the maximum
    %     fitted_values               : the sinusoid at every date
    %     residuals                   : values - fitted_values ("anomalies")
    %     explained_variance          : 1 - var(residuals) / var(values)
    %     residual_standard_deviation : std of the residuals

    days_in_year = 365.2425;

    days_since_start = dates(:) - dates(1);
    angle_in_radians = 2 * pi * days_since_start / days_in_year;

    design_matrix = [ones(numel(dates), 1), cos(angle_in_radians), sin(angle_in_radians)];
    coefficients  = design_matrix \ values(:);

    fitted_values = design_matrix * coefficients;
    residuals     = values(:) - fitted_values;

    angle_of_maximum = atan2(coefficients(3), coefficients(2));

    if angle_of_maximum < 0
        angle_of_maximum = angle_of_maximum + 2 * pi;
    end

    days_until_maximum = angle_of_maximum / (2 * pi) * days_in_year;

    annual_fit.mean_level       = coefficients(1);
    annual_fit.amplitude        = sqrt(coefficients(2) ^ 2 + coefficients(3) ^ 2);
    annual_fit.warmest_day_text = datestr(dates(1) + days_until_maximum, 'mmm dd');
    annual_fit.fitted_values    = fitted_values;
    annual_fit.residuals        = residuals;
    annual_fit.explained_variance = 1 - sum((residuals - mean(residuals)) .^ 2) / sum((values(:) - mean(values(:))) .^ 2);
    annual_fit.residual_standard_deviation = std(residuals);
end


function climatology = compute_daily_climatology(dates, values)
    % Mean and standard deviation of a series for each day of the year.
    %
    % Days 1..365 are used. 29 February shifts later days of leap years by
    % one position, which is negligible for a 15-day smoothed curve.
    %
    % Parameters
    % ----------
    % dates : double column
    %     Serial date numbers.
    % values : double column
    %     Series (same length as dates).
    %
    % Returns
    % -------
    % climatology : struct (all fields are 365 x 1 columns)
    %     day_of_year, mean, standard_deviation, number_of_values,
    %     smoothed_mean (15-day circular moving average of the mean)

    number_of_days = 365;

    day_of_year_of_each_row = compute_day_of_year(dates);

    mean_by_day               = zeros(number_of_days, 1);
    standard_deviation_by_day = zeros(number_of_days, 1);
    number_of_values_by_day   = zeros(number_of_days, 1);

    for day_number = 1:number_of_days
        values_of_day = values(day_of_year_of_each_row == day_number);

        number_of_values_by_day(day_number)   = numel(values_of_day);
        mean_by_day(day_number)               = mean(values_of_day);
        standard_deviation_by_day(day_number) = std(values_of_day);
    end

    climatology.day_of_year         = (1:number_of_days)';
    climatology.mean                = mean_by_day;
    climatology.standard_deviation  = standard_deviation_by_day;
    climatology.number_of_values    = number_of_values_by_day;
    climatology.smoothed_mean       = circular_moving_average(mean_by_day, 15);
end


function smoothed_values = circular_moving_average(values, window_length)
    % Centered moving average that wraps around the ends (for yearly data).
    %
    % Parameters
    % ----------
    % values : double column
    %     Values to smooth.
    % window_length : odd integer
    %     Number of values averaged around each position.
    %
    % Returns
    % -------
    % smoothed_values : double column, same size as values

    if mod(window_length, 2) == 0
        error('circular_moving_average:evenWindow', ...
              'window_length must be odd, got %d.', window_length);
    end

    number_of_values = numel(values);
    half_window = (window_length - 1) / 2;

    smoothed_values = zeros(size(values));

    for position = 1:number_of_values
        window_positions = mod((position - half_window:position + half_window) - 1, number_of_values) + 1;
        smoothed_values(position) = mean(values(window_positions));
    end
end


function [longest_run_length, last_row_of_longest_run] = compute_longest_constant_run(values)
    % Longest sequence of identical consecutive values.
    %
    % Parameters
    % ----------
    % values : double vector
    %
    % Returns
    % -------
    % longest_run_length : integer
    %     Number of values in the longest run (1 if no value repeats).
    % last_row_of_longest_run : integer
    %     Index of the last value of that run.

    longest_run_length = 1;
    last_row_of_longest_run = 1;
    current_run_length = 1;

    for row_index = 2:numel(values)
        if values(row_index) == values(row_index - 1)
            current_run_length = current_run_length + 1;
        else
            current_run_length = 1;
        end

        if current_run_length > longest_run_length
            longest_run_length = current_run_length;
            last_row_of_longest_run = row_index;
        end
    end
end


function draw_correlation_stems(lags, correlation_values, confidence_bound)
    % Stem plot of an ACF/PACF in the current axes, with +/- bound lines.
    %
    % Parameters
    % ----------
    % lags : double vector
    %     Lags on the horizontal axis.
    % correlation_values : double vector
    %     Correlation at each lag (same length as lags).
    % confidence_bound : double
    %     Dashed red lines are drawn at +/- this value.

    stem(lags, correlation_values, 'filled', 'MarkerSize', 3);
    hold on;
    plot([min(lags) - 1, max(lags) + 1], [confidence_bound confidence_bound], 'r--');
    plot([min(lags) - 1, max(lags) + 1], [-confidence_bound -confidence_bound], 'r--');
    hold off;

    set(gca, 'XLim', [min(lags) - 1, max(lags) + 1]);
    grid on;
end


%% ========================================================================
%  STEP 2 - EXPLORATORY DATA ANALYSIS: UTILS
%  Functions used by both datasets (or generic helpers reused in later steps).
%  ========================================================================

function correlation = compute_pearson_correlation(first_values, second_values)
    % Pearson correlation between two vectors.
    %
    % Parameters
    % ----------
    % first_values, second_values : double vectors of the same length
    %
    % Returns
    % -------
    % correlation : double in [-1, 1]

    first_centered  = first_values(:)  - mean(first_values(:));
    second_centered = second_values(:) - mean(second_values(:));

    correlation = sum(first_centered .* second_centered) / sqrt(sum(first_centered .^ 2) * sum(second_centered .^ 2));
end


function draw_box_plot(grouped_values, group_labels)
    % Draw one box plot per group in the current axes (no toolbox).
    %
    % Box: first to third quartile. Red line: median. Whiskers: most
    % extreme values within 1.5 interquartile ranges of the box. Plus
    % signs: values beyond the whiskers (outliers). This function is also
    % reused in Step 8.
    %
    % Parameters
    % ----------
    % grouped_values : 1 x G cell of double vectors
    %     Values of each group.
    % group_labels : 1 x G cell of char
    %     Label shown under each box.

    number_of_groups = numel(grouped_values);
    half_box_width = 0.3;

    hold on;
    for group_index = 1:number_of_groups
        sorted_values = sort(grouped_values{group_index}(:));

        first_quartile  = compute_percentile(sorted_values, 0.25);
        median_value    = compute_percentile(sorted_values, 0.50);
        third_quartile  = compute_percentile(sorted_values, 0.75);

        interquartile_range = third_quartile - first_quartile;
        lower_fence = first_quartile - 1.5 * interquartile_range;
        upper_fence = third_quartile + 1.5 * interquartile_range;

        lower_whisker = min(sorted_values(sorted_values >= lower_fence));
        upper_whisker = max(sorted_values(sorted_values <= upper_fence));
        outliers = sorted_values(sorted_values < lower_fence | sorted_values > upper_fence);

        fill([group_index - half_box_width, group_index + half_box_width, ...
              group_index + half_box_width, group_index - half_box_width], ...
             [first_quartile, first_quartile, third_quartile, third_quartile], ...
             [0.80 0.88 0.97], 'EdgeColor', 'k');

        plot([group_index - half_box_width, group_index + half_box_width], ...
             [median_value, median_value], 'r-', 'LineWidth', 2);

        plot([group_index group_index], [lower_whisker first_quartile], 'k-');
        plot([group_index group_index], [third_quartile upper_whisker], 'k-');
        plot([group_index - 0.15, group_index + 0.15], [lower_whisker lower_whisker], 'k-');
        plot([group_index - 0.15, group_index + 0.15], [upper_whisker upper_whisker], 'k-');

        if ~isempty(outliers)
            plot(group_index * ones(size(outliers)), outliers, 'k+');
        end
    end
    hold off;

    set(gca, 'XTick', 1:number_of_groups, 'XTickLabel', group_labels);
    set(gca, 'XLim', [0.5, number_of_groups + 0.5]);
end


function save_figure(figure_handle, folder_path, file_name_without_extension)
    % Save a figure as a PNG file (150 dpi) and print where it went.
    %
    % Parameters
    % ----------
    % figure_handle : handle
    %     Figure to save.
    % folder_path : char
    %     Destination folder (must exist).
    % file_name_without_extension : char
    %     File name without ".png".

    file_path = fullfile(folder_path, [file_name_without_extension, '.png']);

    figure(figure_handle);
    set(figure_handle, 'PaperPositionMode', 'auto');
    print('-dpng', '-r150', file_path);

    fprintf('  saved: %s\n', file_path);
end


function ensure_folder_exists(folder_path)
    % Create a folder if it does not exist yet.
    %
    % Parameters
    % ----------
    % folder_path : char
    %     Folder to create.

    if ~exist(folder_path, 'dir')
        mkdir(folder_path);
    end
end


%% ========================================================================
%  STEP 3 - PREPROCESSING AND FEATURE ENGINEERING: CONCRETE
%  Functions used only by the Concrete dataset.
%  ========================================================================

function [concrete_features, preprocessing_report] = preprocess_concrete_dataset(dataset)
    % Clean the Concrete dataset and build its input features (Step 3).
    %
    % Cleaning
    % --------
    % The dataset has no missing values (Step 1). The 25 rows that are exact
    % copies of an earlier row (all 9 variables equal, target included) are
    % removed: they carry no new information, give those mixtures double
    % weight in the loss, and a copy could fall in the training set while
    % its twin falls in the test set (leakage).
    %
    % Features (the same for all architectures and algorithms)
    % --------------------------------------------------------
    % 1-7 : the seven ingredients, unchanged (kg/m^3)
    % 8   : log10(Age) instead of Age. Age has skewness 3.26, only 14
    %       distinct values from 1 to 365 days, and its Pearson correlation
    %       with strength rises from 0.33 to 0.55 after the log (close to
    %       its Spearman correlation, 0.60): strength grows roughly with
    %       the logarithm of the curing time.
    % 9   : binder-to-water ratio, (cement + slag + fly ash) / water.
    %       Abrams' and Bolomey's laws relate strength to the ratio of
    %       binder to water; it has the highest correlation with strength
    %       of all candidates tested in Step 2 (0.64, against 0.50 for
    %       cement alone and 0.56 for cement / water).
    %
    % Scaling is NOT done here: its statistics must come from the training
    % set, which only exists after the split (Step 4).
    %
    % Parameters
    % ----------
    % dataset : struct
    %     Output of load_concrete_dataset.
    %
    % Returns
    % -------
    % concrete_features : struct
    %     Output of build_concrete_features (inputs, target, names, units,
    %     mixture_id_of_each_row).
    % preprocessing_report : struct
    %     number_of_rows_before, number_of_rows_after,
    %     number_of_duplicate_rows_removed

    print_section_title('CONCRETE - Step 3: cleaning');

    number_of_rows_before = size(dataset.inputs, 1);
    [clean_dataset, number_of_duplicate_rows_removed] = remove_exact_duplicate_rows(dataset);
    number_of_rows_after = size(clean_dataset.inputs, 1);

    fprintf('Missing values                 : none (Step 1)\n');
    fprintf('Exact duplicate rows removed   : %d\n', number_of_duplicate_rows_removed);
    fprintf('Rows before / after            : %d / %d\n', number_of_rows_before, number_of_rows_after);

    concrete_features = build_concrete_features(clean_dataset);

    % ---- Feature table ----------------------------------------------------
    print_section_title('CONCRETE - Step 3: input features');

    number_of_features = size(concrete_features.inputs, 2);
    fprintf('%-4s %-34s %-12s\n', '#', 'Feature', 'Unit');

    for feature_index = 1:number_of_features
        fprintf('%-4d %-34s %-12s\n', feature_index, ...
                concrete_features.feature_names{feature_index}, ...
                concrete_features.feature_units{feature_index});
    end

    fprintf('Target: %s (%s)\n', concrete_features.target_name, concrete_features.target_unit);

    fprintf('\nCorrelation of each feature with the target (after cleaning):\n');

    for feature_index = 1:number_of_features
        fprintf('  %-34s r = %6.3f\n', concrete_features.feature_names{feature_index}, ...
                compute_pearson_correlation(concrete_features.inputs(:, feature_index), ...
                                            concrete_features.target));
    end

    fprintf('\nDistinct mixtures (groups for the Step 4 discussion): %d\n', ...
            max(concrete_features.mixture_id_of_each_row));

    print_network_sizes(number_of_features);

    preprocessing_report.number_of_rows_before = number_of_rows_before;
    preprocessing_report.number_of_rows_after  = number_of_rows_after;
    preprocessing_report.number_of_duplicate_rows_removed = number_of_duplicate_rows_removed;
end


function [clean_dataset, number_of_rows_removed] = remove_exact_duplicate_rows(dataset)
    % Remove rows that are identical (inputs and target) to an earlier row.
    %
    % The first occurrence is kept and the original row order is preserved.
    %
    % Parameters
    % ----------
    % dataset : struct
    %     Output of load_concrete_dataset (fields inputs and target).
    %
    % Returns
    % -------
    % clean_dataset : struct
    %     Same fields as the input, without the duplicate rows.
    % number_of_rows_removed : integer

    all_columns = [dataset.inputs, dataset.target];

    [unused_unique_rows, index_of_first_occurrence] = unique(all_columns, 'rows', 'first');
    rows_to_keep = sort(index_of_first_occurrence);

    clean_dataset = dataset;
    clean_dataset.inputs = dataset.inputs(rows_to_keep, :);
    clean_dataset.target = dataset.target(rows_to_keep);

    number_of_rows_removed = size(all_columns, 1) - numel(rows_to_keep);
end


function concrete_features = build_concrete_features(dataset)
    % Build the 9 input features of the Concrete dataset (see Step 3 notes).
    %
    % Parameters
    % ----------
    % dataset : struct
    %     Output of load_concrete_dataset (possibly after cleaning).
    %
    % Returns
    % -------
    % concrete_features : struct
    %     inputs                 : N x 9 double, unscaled features
    %     target                 : N x 1 double, compressive strength (MPa)
    %     feature_names          : 1 x 9 cell of char
    %     feature_units          : 1 x 9 cell of char
    %     target_name, target_unit : char
    %     mixture_id_of_each_row : N x 1 integer, same number for rows with
    %                              the same 7 ingredients (used in Step 4)

    cement_values = dataset.inputs(:, strcmp(dataset.input_names, 'Cement'));
    slag_values   = dataset.inputs(:, strcmp(dataset.input_names, 'Blast furnace slag'));
    ash_values    = dataset.inputs(:, strcmp(dataset.input_names, 'Fly ash'));
    water_values  = dataset.inputs(:, strcmp(dataset.input_names, 'Water'));
    age_values    = dataset.inputs(:, strcmp(dataset.input_names, 'Age'));

    age_column_index        = find(strcmp(dataset.input_names, 'Age'));
    ingredient_column_index = setdiff(1:size(dataset.inputs, 2), age_column_index);
    ingredient_matrix       = dataset.inputs(:, ingredient_column_index);

    log_age_values        = log10(age_values);
    binder_to_water_ratio = (cement_values + slag_values + ash_values) ./ water_values;

    concrete_features.inputs = [ingredient_matrix, log_age_values, binder_to_water_ratio];
    concrete_features.target = dataset.target;

    concrete_features.feature_names = [dataset.input_names(ingredient_column_index), {'log10(Age)', '(Cement + Slag + Ash) / Water'}];
    concrete_features.feature_units = [dataset.input_units(ingredient_column_index), {'log10(days)', 'ratio'}];

    concrete_features.target_name = dataset.target_name;
    concrete_features.target_unit = dataset.target_unit;

    [unused_distinct_mixtures, unused_first_index, mixture_id_of_each_row] = unique(ingredient_matrix, 'rows');
    concrete_features.mixture_id_of_each_row = mixture_id_of_each_row(:);
end


%% ========================================================================
%  STEP 3 - PREPROCESSING AND FEATURE ENGINEERING: MELBOURNE
%  Functions used only by the Melbourne dataset.
%  ========================================================================

function melbourne_features = preprocess_melbourne_dataset(dataset, candidate_window_lengths, split_fractions)
    % Clean the Melbourne series, choose the window p and build the samples.
    %
    % Cleaning
    % --------
    % The 3 values with a corrupted minus sign were already repaired when
    % reading the file (Step 1). The 2 missing days (1984-12-31 and
    % 1988-12-31) are isolated single days, so they are filled by linear
    % interpolation between the day before and the day after. Without this,
    % a "lag of 1 day" would be a lag of 2 days at those points. The 2
    % interpolated values may appear as INPUTS (lags) but never as TARGETS:
    % the two samples whose target is an interpolated day are dropped, so
    % the model is never trained or evaluated on an invented value.
    %
    % Window length p
    % ---------------
    % Candidate values of p are compared with a linear autoregressive model
    % (with and without calendar features) fitted on the training block and
    % evaluated on the validation block of the chronological split that is
    % also used in Step 4. The test block is not touched. The chosen p is
    % the one with the lowest validation RMSE with calendar features; the
    % PACF of Step 2 (large up to lag ~7) supports the same range.
    %
    % Features (the same for all architectures and algorithms)
    % --------------------------------------------------------
    % T(t-p+1), ..., T(t)   : the p most recent temperatures (degC)
    % sin, cos of day t     : position of day t in the annual cycle. The
    %                         annual cycle explains 53.5 % of the variance
    %                         (Step 2), and sin/cos give the season without
    %                         the jump between 31 December and 1 January.
    % Target: T(t+1).
    %
    % Scaling is NOT done here (statistics from the training block only,
    % applied in Step 4 right after the split).
    %
    % Parameters
    % ----------
    % dataset : struct
    %     Output of load_melbourne_dataset.
    % candidate_window_lengths : integer vector
    %     Values of p to compare, e.g. [1 2 3 4 5 7 10 14 21 30].
    % split_fractions : 1 x 3 double
    %     Fractions of the calendar span for training, validation and test,
    %     e.g. [0.70 0.15 0.15].
    %
    % Returns
    % -------
    % melbourne_features : struct
    %     Output of build_melbourne_windows for the chosen p (inputs,
    %     target, target_dates, last_input_dates, feature_names), plus:
    %     chosen_window_length      : integer p
    %     split_cutoffs             : output of compute_chronological_split_cutoffs
    %     window_comparison         : output of compare_window_lengths_with_linear_model
    %     daily_dates, daily_temperature_celsius, day_is_interpolated :
    %                                 the complete daily series used

    % ---- 1. Fill the missing days -----------------------------------------
    print_section_title('MELBOURNE - Step 3: cleaning');

    [daily_dates, daily_temperature_celsius, day_is_interpolated] = fill_missing_days_by_linear_interpolation(dataset.dates, dataset.temperature_celsius);

    fprintf('Corrupted minus signs repaired at reading (Step 1): %d\n', numel(dataset.repaired_dates));
    fprintf('Missing days filled by linear interpolation      : %d\n', sum(day_is_interpolated));

    interpolated_rows = find(day_is_interpolated);

    for list_index = 1:numel(interpolated_rows)
        row_index = interpolated_rows(list_index);
        fprintf('  %s : %.2f degC (between %.1f and %.1f)\n', ...
                datestr(daily_dates(row_index), 'yyyy-mm-dd'), ...
                daily_temperature_celsius(row_index), ...
                daily_temperature_celsius(row_index - 1), ...
                daily_temperature_celsius(row_index + 1));
    end

    fprintf('Days in the complete series                      : %d\n', numel(daily_dates));
    fprintf('(Interpolated days are used as inputs only, never as targets.)\n');

    % ---- 2. Chronological blocks (the same ones are used in Step 4) -------
    split_cutoffs = compute_chronological_split_cutoffs(daily_dates(1), daily_dates(end), split_fractions);

    print_section_title('MELBOURNE - Step 3: chronological blocks (by target date)');
    fprintf('Training   : %s to %s\n', datestr(daily_dates(1), 'yyyy-mm-dd'), ...
            datestr(split_cutoffs.first_validation_date - 1, 'yyyy-mm-dd'));
    fprintf('Validation : %s to %s\n', datestr(split_cutoffs.first_validation_date, 'yyyy-mm-dd'), ...
            datestr(split_cutoffs.first_test_date - 1, 'yyyy-mm-dd'));
    fprintf('Test       : %s to %s   (not used in Step 3)\n', ...
            datestr(split_cutoffs.first_test_date, 'yyyy-mm-dd'), ...
            datestr(daily_dates(end), 'yyyy-mm-dd'));

    % ---- 3. Choose the window length p ------------------------------------
    window_comparison = compare_window_lengths_with_linear_model(daily_dates, daily_temperature_celsius, day_is_interpolated, candidate_window_lengths, split_cutoffs);

    chosen_window_length = window_comparison.best_window_length_with_calendar;

    fprintf('\nChosen window length p = %d (lowest validation RMSE with calendar features).\n', chosen_window_length);

    % ---- 4. Build the samples with the chosen p ---------------------------
    melbourne_features = build_melbourne_windows(daily_dates, daily_temperature_celsius, day_is_interpolated, chosen_window_length, true);

    print_section_title('MELBOURNE - Step 3: input features');
    number_of_features = size(melbourne_features.inputs, 2);

    for feature_index = 1:number_of_features
        fprintf('%-4d %s\n', feature_index, melbourne_features.feature_names{feature_index});
    end
    
    fprintf('Target: T(t+1) (degC)\n');
    fprintf('Samples: %d (first %d days have no complete window; %d targets were interpolated days)\n', ...
            numel(melbourne_features.target), chosen_window_length - 1, sum(day_is_interpolated));

    print_network_sizes(number_of_features);

    melbourne_features.chosen_window_length      = chosen_window_length;
    melbourne_features.split_cutoffs             = split_cutoffs;
    melbourne_features.window_comparison         = window_comparison;
    melbourne_features.daily_dates               = daily_dates;
    melbourne_features.daily_temperature_celsius = daily_temperature_celsius;
    melbourne_features.day_is_interpolated       = day_is_interpolated;
end


function [daily_dates, daily_values, day_is_interpolated] = fill_missing_days_by_linear_interpolation(dates, values)
    % Rebuild a complete daily axis and fill absent days by interpolation.
    %
    % Parameters
    % ----------
    % dates : double column
    %     Serial date numbers, sorted, without duplicates.
    % values : double column
    %     Observed value of each date.
    %
    % Returns
    % -------
    % daily_dates : double column
    %     Every calendar day from the first to the last date.
    % daily_values : double column
    %     Observed values, with absent days filled by linear interpolation
    %     between the nearest observed days before and after.
    % day_is_interpolated : logical column
    %     True for the days that were filled.

    daily_dates = (dates(1):dates(end))';

    day_is_interpolated = ~ismember(daily_dates, dates);

    daily_values = interp1(dates(:), values(:), daily_dates, 'linear');
end


function windows = build_melbourne_windows(daily_dates, daily_values, day_is_interpolated, window_length, use_calendar_features)
    % Build one sample per day t: inputs from days t-p+1..t, target day t+1.
    %
    % Only information available at the end of day t is used: the p most
    % recent values and, optionally, the calendar position of day t.
    % Samples whose target day was interpolated are dropped.
    %
    % Parameters
    % ----------
    % daily_dates : double column
    %     Complete daily axis (output of fill_missing_days_by_linear_interpolation).
    % daily_values : double column
    %     Value of each day.
    % day_is_interpolated : logical column
    %     True for days filled by interpolation.
    % window_length : integer
    %     Number p of most recent values used as inputs.
    % use_calendar_features : logical
    %     If true, sin and cos of the day of the year of day t are appended.
    %
    % Returns
    % -------
    % windows : struct
    %     inputs           : M x (p or p+2) double, unscaled; lag columns
    %                        ordered from oldest T(t-p+1) to newest T(t)
    %     target           : M x 1 double, T(t+1)
    %     target_dates     : M x 1 serial dates of day t+1
    %     last_input_dates : M x 1 serial dates of day t
    %     feature_names    : 1 x (p or p+2) cell of char
    %     number_of_lag_columns : p

    number_of_days = numel(daily_values);

    % Index of day t for every possible sample (needs p days of history and a following day for the target).
    index_of_day_t = (window_length:number_of_days - 1)';

    target_is_interpolated = day_is_interpolated(index_of_day_t + 1);
    index_of_day_t = index_of_day_t(~target_is_interpolated);

    number_of_samples = numel(index_of_day_t);

    % Row k of lag_index_matrix holds the days t-p+1, ..., t of sample k.
    lag_offsets      = (1 - window_length):0;
    lag_index_matrix = repmat(index_of_day_t, 1, window_length) + repmat(lag_offsets, number_of_samples, 1);

    lag_inputs = reshape(daily_values(lag_index_matrix), number_of_samples, window_length);

    lag_names = cell(1, window_length);

    for lag_position = 1:window_length
        days_back = window_length - lag_position;

        if days_back == 0
            lag_names{lag_position} = 'T(t)';
        else
            lag_names{lag_position} = sprintf('T(t-%d)', days_back);
        end
    end

    if use_calendar_features
        calendar_inputs  = compute_calendar_features(daily_dates(index_of_day_t));
        windows.inputs        = [lag_inputs, calendar_inputs];
        windows.feature_names = [lag_names, {'sin(2*pi*day of year / 365.2425)', 'cos(2*pi*day of year / 365.2425)'}];
    else
        windows.inputs        = lag_inputs;
        windows.feature_names = lag_names;
    end

    windows.target           = daily_values(index_of_day_t + 1);
    windows.target_dates     = daily_dates(index_of_day_t + 1);
    windows.last_input_dates = daily_dates(index_of_day_t);
    windows.number_of_lag_columns = window_length;
end


function calendar_features = compute_calendar_features(dates)
    % Sine and cosine of the position of each date in the annual cycle.
    %
    % angle = 2*pi*(day of year - 1) / 365.2425, so 1 January and
    % 31 December are close to each other, as they are in the climate.
    %
    % Parameters
    % ----------
    % dates : double column
    %     Serial date numbers.
    %
    % Returns
    % -------
    % calendar_features : N x 2 double, columns [sin(angle), cos(angle)]

    days_in_year = 365.2425;

    angle_in_radians = 2 * pi * (compute_day_of_year(dates) - 1) / days_in_year;

    calendar_features = [sin(angle_in_radians), cos(angle_in_radians)];
end


function window_comparison = compare_window_lengths_with_linear_model(daily_dates, daily_values, day_is_interpolated, candidate_window_lengths, split_cutoffs)
    % Validation RMSE of a linear autoregressive model for several p.
    %
    % For each p, a linear model with intercept is fitted by least squares
    % on the training block and evaluated on the validation block, with and
    % without the two calendar features. The persistence forecast
    % T(t+1) = T(t) is reported as a reference. The test block is not used.
    %
    % Parameters
    % ----------
    % daily_dates, daily_values, day_is_interpolated : columns
    %     Complete daily series (see build_melbourne_windows).
    % candidate_window_lengths : integer vector
    %     Values of p to compare.
    % split_cutoffs : struct
    %     Output of compute_chronological_split_cutoffs.
    %
    % Returns
    % -------
    % window_comparison : struct
    %     candidate_window_lengths            : the input vector
    %     validation_rmse_lags_only           : one value per p (degC)
    %     validation_rmse_with_calendar       : one value per p (degC)
    %     validation_rmse_persistence         : scalar (degC)
    %     best_window_length_with_calendar    : p with the lowest RMSE

    number_of_candidates = numel(candidate_window_lengths);

    validation_rmse_lags_only     = zeros(1, number_of_candidates);
    validation_rmse_with_calendar = zeros(1, number_of_candidates);

    for candidate_index = 1:number_of_candidates
        window_length = candidate_window_lengths(candidate_index);

        for use_calendar_features = [false true]
            windows = build_melbourne_windows(daily_dates, daily_values, day_is_interpolated, window_length, use_calendar_features);

            block_of_each_sample = assign_samples_to_chronological_blocks(windows.target_dates, split_cutoffs);
            is_training   = (block_of_each_sample == 1);
            is_validation = (block_of_each_sample == 2);

            coefficients = fit_linear_least_squares(windows.inputs(is_training, :), windows.target(is_training));
            validation_prediction = predict_linear_model(windows.inputs(is_validation, :), coefficients);
            validation_rmse = compute_root_mean_squared_error(windows.target(is_validation), validation_prediction);

            if use_calendar_features
                validation_rmse_with_calendar(candidate_index) = validation_rmse;
            else
                validation_rmse_lags_only(candidate_index) = validation_rmse;
            end
        end
    end

    % Persistence on the same validation samples (p = 1 windows).
    windows_of_one_day = build_melbourne_windows(daily_dates, daily_values, day_is_interpolated, 1, false);
    block_of_each_sample = assign_samples_to_chronological_blocks(windows_of_one_day.target_dates, split_cutoffs);
    is_validation = (block_of_each_sample == 2);
    validation_rmse_persistence = compute_root_mean_squared_error(windows_of_one_day.target(is_validation), windows_of_one_day.inputs(is_validation, 1));

    [unused_lowest_rmse, best_candidate_index] = min(validation_rmse_with_calendar);

    % ---- Print the comparison --------------------------------------------
    print_section_title('MELBOURNE - Step 3: choice of the window length p');
    fprintf('Linear autoregressive model, fitted on training, RMSE on validation (degC)\n\n');
    fprintf('%-6s %14s %16s\n', 'p', 'lags only', 'lags + calendar');

    for candidate_index = 1:number_of_candidates
        fprintf('%-6d %14.3f %16.3f\n', candidate_window_lengths(candidate_index), ...
                validation_rmse_lags_only(candidate_index), ...
                validation_rmse_with_calendar(candidate_index));
    end

    fprintf('\nPersistence T(t+1) = T(t) on the same validation block: %.3f degC\n', validation_rmse_persistence);

    window_comparison.candidate_window_lengths         = candidate_window_lengths;
    window_comparison.validation_rmse_lags_only        = validation_rmse_lags_only;
    window_comparison.validation_rmse_with_calendar    = validation_rmse_with_calendar;
    window_comparison.validation_rmse_persistence      = validation_rmse_persistence;
    window_comparison.best_window_length_with_calendar = candidate_window_lengths(best_candidate_index);
end


function split_cutoffs = compute_chronological_split_cutoffs(first_date, last_date, split_fractions)
    % First validation date and first test date of a chronological split.
    %
    % The calendar span is cut by fractions of days, so the blocks are the
    % same whatever window length p is used later.
    %
    % Parameters
    % ----------
    % first_date, last_date : double
    %     Serial dates of the first and last day of the series.
    % split_fractions : 1 x 3 double
    %     Fractions for training, validation and test (must sum to 1).
    %
    % Returns
    % -------
    % split_cutoffs : struct
    %     first_validation_date : serial date of the first validation day
    %     first_test_date       : serial date of the first test day
    %     split_fractions       : the input fractions

    if abs(sum(split_fractions) - 1) > 1e-9
        error('compute_chronological_split_cutoffs:badFractions', ...
              'split_fractions must sum to 1, got %.4f.', sum(split_fractions));
    end

    number_of_days = last_date - first_date + 1;

    split_cutoffs.first_validation_date = first_date + round(split_fractions(1) * number_of_days);
    split_cutoffs.first_test_date       = first_date + round((split_fractions(1) + split_fractions(2)) * number_of_days);
    split_cutoffs.split_fractions       = split_fractions;
end


function block_of_each_sample = assign_samples_to_chronological_blocks(target_dates, split_cutoffs)
    % Block of each sample according to the date of its TARGET.
    %
    % A sample belongs to the block that contains the day it predicts. The
    % first validation samples therefore use the last training days as
    % inputs, which is allowed (they are in the past); no training sample
    % ever uses a validation or test value.
    %
    % Parameters
    % ----------
    % target_dates : double column
    %     Serial date of the target day of each sample.
    % split_cutoffs : struct
    %     Output of compute_chronological_split_cutoffs.
    %
    % Returns
    % -------
    % block_of_each_sample : double column
    %     1 = training, 2 = validation, 3 = test.

    block_of_each_sample = ones(size(target_dates));
    block_of_each_sample(target_dates >= split_cutoffs.first_validation_date) = 2;
    block_of_each_sample(target_dates >= split_cutoffs.first_test_date)       = 3;
end


%% ========================================================================
%  STEP 3 - PREPROCESSING AND FEATURE ENGINEERING: UTILS
%  Functions used by both datasets (or generic helpers reused in later steps).
%  ========================================================================

function scaler = fit_standardization(training_matrix)
    % Mean and standard deviation of each column, from training data only.
    %
    % Parameters
    % ----------
    % training_matrix : double matrix, N x K
    %     Training rows only. Never pass validation or test rows here.
    %
    % Returns
    % -------
    % scaler : struct
    %     mean               : 1 x K
    %     standard_deviation : 1 x K (a constant column gets 1, to avoid
    %                          dividing by zero)

    column_mean = mean(training_matrix, 1);
    column_standard_deviation = std(training_matrix, 0, 1);

    column_is_constant = (column_standard_deviation == 0);
    column_standard_deviation(column_is_constant) = 1;

    scaler.mean               = column_mean;
    scaler.standard_deviation = column_standard_deviation;
end


function scaled_matrix = apply_standardization(matrix, scaler)
    % Standardise a matrix with statistics computed on the training set.
    %
    % Parameters
    % ----------
    % matrix : double matrix, N x K
    %     Training, validation or test rows.
    % scaler : struct
    %     Output of fit_standardization.
    %
    % Returns
    % -------
    % scaled_matrix : double matrix, N x K, (matrix - mean) ./ std

    number_of_rows = size(matrix, 1);

    scaled_matrix = (matrix - repmat(scaler.mean, number_of_rows, 1)) ./ repmat(scaler.standard_deviation, number_of_rows, 1);
end


function original_matrix = invert_standardization(scaled_matrix, scaler)
    % Bring standardised values back to their original unit.
    %
    % Used to report predictions and errors in MPa or degC.
    %
    % Parameters
    % ----------
    % scaled_matrix : double matrix, N x K
    % scaler : struct
    %     Output of fit_standardization.
    %
    % Returns
    % -------
    % original_matrix : double matrix, N x K, scaled .* std + mean

    number_of_rows = size(scaled_matrix, 1);

    original_matrix = scaled_matrix .* repmat(scaler.standard_deviation, number_of_rows, 1) + repmat(scaler.mean, number_of_rows, 1);
end


function coefficients = fit_linear_least_squares(inputs, target)
    % Least-squares linear model with intercept (also a Step 8 reference).
    %
    % Parameters
    % ----------
    % inputs : double matrix, N x K
    % target : double column, N x 1
    %
    % Returns
    % -------
    % coefficients : (K + 1) x 1 double, intercept first

    number_of_rows = size(inputs, 1);
    design_matrix  = [ones(number_of_rows, 1), inputs];

    coefficients = design_matrix \ target(:);
end


function prediction = predict_linear_model(inputs, coefficients)
    % Prediction of a linear model fitted with fit_linear_least_squares.
    %
    % Parameters
    % ----------
    % inputs : double matrix, N x K
    % coefficients : (K + 1) x 1 double, intercept first
    %
    % Returns
    % -------
    % prediction : N x 1 double

    number_of_rows = size(inputs, 1);

    prediction = [ones(number_of_rows, 1), inputs] * coefficients;
end


function rmse = compute_root_mean_squared_error(observed_values, predicted_values)
    % Root mean squared error, in the unit of the values.
    %
    % Parameters
    % ----------
    % observed_values, predicted_values : double vectors of the same length
    %
    % Returns
    % -------
    % rmse : double, sqrt(mean((observed - predicted).^2))

    errors = observed_values(:) - predicted_values(:);

    rmse = sqrt(mean(errors .^ 2));
end


function number_of_parameters = count_network_parameters(layer_sizes)
    % Number of weights and biases of a fully connected network.
    %
    % Parameters
    % ----------
    % layer_sizes : integer vector
    %     [inputs, hidden_1, ..., hidden_k, outputs], e.g. [9 10 1].
    %
    % Returns
    % -------
    % number_of_parameters : integer, sum over layers of (fan_in + 1) * fan_out

    number_of_parameters = 0;

    for layer_index = 1:numel(layer_sizes) - 1
        fan_in  = layer_sizes(layer_index);
        fan_out = layer_sizes(layer_index + 1);
        number_of_parameters = number_of_parameters + (fan_in + 1) * fan_out;
    end
end


function print_network_sizes(number_of_inputs)
    % Print the parameter count of architectures A, B and C for d inputs.
    %
    % Parameters
    % ----------
    % number_of_inputs : integer
    %     Number d of input features.

    reference_width = 10;

    reference_layers = [number_of_inputs, reference_width, 1];
    wide_layers      = [number_of_inputs, 10 * reference_width, 1];
    deep_layers      = [number_of_inputs, reference_width * ones(1, 10), 1];

    fprintf('\nNetwork sizes with d = %d inputs:\n', number_of_inputs);
    fprintf('  A Reference (1 x %3d) : %5d parameters\n', reference_width, count_network_parameters(reference_layers));
    fprintf('  B Wide      (1 x %3d) : %5d parameters\n', 10 * reference_width, count_network_parameters(wide_layers));
    fprintf('  C Deep     (10 x %3d) : %5d parameters\n', reference_width, count_network_parameters(deep_layers));
end


%% ========================================================================
%  STEP 4 - DATA PARTITIONING: CONCRETE
%  Functions used only by the Concrete dataset.
%  ========================================================================

function partition = partition_concrete_dataset(concrete_features, split_fractions, split_seed, comparison_seeds)
    % Random 70/15/15 split of Concrete, scaling, and evidence for the discussion.
    %
    % Split
    % -----
    % Rows are assigned at random to training, validation and test with a
    % fixed seed (assignment requirement). The split is computed once here
    % and reused for the whole assignment.
    %
    % Discussion: is a random split appropriate?
    % ------------------------------------------
    % The same mixture (7 ingredients) appears several times, tested at
    % different ages. With a random split, a mixture can be in training at
    % 7 days and in test at 28 days, so the test set measures "known recipe
    % at a new age" rather than "new recipe". To give numbers to this
    % discussion, this function:
    %   1. counts the validation and test rows whose mixture is also in
    %      the training set;
    %   2. compares, over several seeds and on the VALIDATION set only, a
    %      random split with a split by mixture (all rows of a mixture in
    %      the same subset), using two simple models: linear regression
    %      and 1-nearest neighbour (a model that memorises training rows).
    % The test set is never used here.
    %
    % Scaling
    % -------
    % Inputs and target are standardised with the mean and standard
    % deviation of the TRAINING rows only.
    %
    % Parameters
    % ----------
    % concrete_features : struct
    %     Output of preprocess_concrete_dataset.
    % split_fractions : 1 x 3 double
    %     Fractions for training, validation and test, e.g. [0.70 0.15 0.15].
    % split_seed : integer in 1 .. 2147483646
    %     Seed of the split.
    % comparison_seeds : integer vector
    %     Seeds used only for the random-versus-grouped comparison.
    %
    % Returns
    % -------
    % partition : struct
    %     Output of build_partition + scale_partition, plus:
    %     split_seed, split_fractions
    %     mixture_overlap   : output of count_rows_with_group_in_training
    %     split_comparison  : output of compare_random_and_grouped_splits

    number_of_rows = size(concrete_features.inputs, 1);

    block_of_each_row = assign_rows_to_random_blocks(number_of_rows, split_fractions, split_seed);

    partition = build_partition(concrete_features.inputs, concrete_features.target, block_of_each_row);

    input_scaler  = fit_standardization(partition.training.inputs);
    target_scaler = fit_standardization(partition.training.target);
    partition = scale_partition(partition, input_scaler, target_scaler);

    partition.split_seed      = split_seed;
    partition.split_fractions = split_fractions;

    % ---- Sizes and scaling check ------------------------------------------
    print_section_title(sprintf('CONCRETE - Step 4: random split (seed %d)', split_seed));
    print_partition_summary(partition, concrete_features.target_unit);

    % ---- Mixtures shared between subsets ----------------------------------
    mixture_overlap = count_rows_with_group_in_training(concrete_features.mixture_id_of_each_row, block_of_each_row);

    print_section_title('CONCRETE - Step 4: mixtures shared with the training set');
    fprintf('Validation rows whose mixture is also in training : %d of %d (%.1f %%)\n', mixture_overlap.validation_rows_with_group_in_training, mixture_overlap.number_of_validation_rows, 100 * mixture_overlap.validation_fraction);
    fprintf('Test rows whose mixture is also in training       : %d of %d (%.1f %%)\n', mixture_overlap.test_rows_with_group_in_training, mixture_overlap.number_of_test_rows, 100 * mixture_overlap.test_fraction);

    % ---- Random split versus split by mixture -----------------------------
    split_comparison = compare_random_and_grouped_splits(concrete_features, split_fractions, comparison_seeds);

    partition.mixture_overlap  = mixture_overlap;
    partition.split_comparison = split_comparison;
end


function block_of_each_row = assign_rows_to_random_blocks(number_of_rows, split_fractions, seed)
    % Assign each row at random to training (1), validation (2) or test (3).
    %
    % The rows are put in a random order and cut by the fractions. The
    % random order comes from generate_uniform_random_numbers, so the split
    % is identical in MATLAB and in GNU Octave.
    %
    % Parameters
    % ----------
    % number_of_rows : integer
    % split_fractions : 1 x 3 double, summing to 1
    % seed : integer in 1 .. 2147483646
    %
    % Returns
    % -------
    % block_of_each_row : number_of_rows x 1 double, values 1, 2 or 3

    check_split_fractions(split_fractions);

    [unused_sorted_values, random_order] = sort(generate_uniform_random_numbers(seed, number_of_rows));

    number_of_training_rows   = round(split_fractions(1) * number_of_rows);
    number_of_validation_rows = round(split_fractions(2) * number_of_rows);

    block_of_each_row = 3 * ones(number_of_rows, 1);

    block_of_each_row(random_order(1:number_of_training_rows)) = 1;
    block_of_each_row(random_order(number_of_training_rows + 1:number_of_training_rows + number_of_validation_rows)) = 2;
end


function block_of_each_row = assign_groups_to_blocks(group_id_of_each_row, split_fractions, seed)
    % Split by group: all rows of the same group go to the same subset.
    %
    % Groups are taken in a random order and added to training until it
    % holds the training fraction of the ROWS, then to validation, and the
    % rest go to test. Subset sizes are therefore close to, not exactly,
    % the requested fractions.
    %
    % Parameters
    % ----------
    % group_id_of_each_row : N x 1 integer, values 1 .. G
    % split_fractions : 1 x 3 double, summing to 1
    % seed : integer in 1 .. 2147483646
    %
    % Returns
    % -------
    % block_of_each_row : N x 1 double, values 1, 2 or 3

    check_split_fractions(split_fractions);

    number_of_rows   = numel(group_id_of_each_row);
    number_of_groups = max(group_id_of_each_row);

    [unused_sorted_values, random_group_order] = sort(generate_uniform_random_numbers(seed, number_of_groups));

    target_training_rows   = round(split_fractions(1) * number_of_rows);
    target_validation_rows = round(split_fractions(2) * number_of_rows);

    block_of_each_row = zeros(number_of_rows, 1);
    rows_assigned_so_far = 0;

    for order_index = 1:number_of_groups
        current_group = random_group_order(order_index);
        row_is_in_group = (group_id_of_each_row == current_group);

        if rows_assigned_so_far < target_training_rows
            block_of_each_row(row_is_in_group) = 1;
        elseif rows_assigned_so_far < target_training_rows + target_validation_rows
            block_of_each_row(row_is_in_group) = 2;
        else
            block_of_each_row(row_is_in_group) = 3;
        end

        rows_assigned_so_far = rows_assigned_so_far + sum(row_is_in_group);
    end
end


function group_overlap = count_rows_with_group_in_training(group_id_of_each_row, block_of_each_row)
    % Validation and test rows whose group also has rows in training.
    %
    % Parameters
    % ----------
    % group_id_of_each_row : N x 1 integer
    % block_of_each_row : N x 1, values 1 (training), 2 (validation), 3 (test)
    %
    % Returns
    % -------
    % group_overlap : struct
    %     validation_rows_with_group_in_training, number_of_validation_rows,
    %     validation_fraction, test_rows_with_group_in_training,
    %     number_of_test_rows, test_fraction

    groups_in_training = unique(group_id_of_each_row(block_of_each_row == 1));

    validation_groups = group_id_of_each_row(block_of_each_row == 2);
    test_groups       = group_id_of_each_row(block_of_each_row == 3);

    group_overlap.validation_rows_with_group_in_training = sum(ismember(validation_groups, groups_in_training));
    group_overlap.number_of_validation_rows = numel(validation_groups);
    group_overlap.validation_fraction = group_overlap.validation_rows_with_group_in_training / group_overlap.number_of_validation_rows;

    group_overlap.test_rows_with_group_in_training = sum(ismember(test_groups, groups_in_training));
    group_overlap.number_of_test_rows = numel(test_groups);
    group_overlap.test_fraction = group_overlap.test_rows_with_group_in_training / group_overlap.number_of_test_rows;
end


function split_comparison = compare_random_and_grouped_splits(concrete_features, split_fractions, comparison_seeds)
    % Validation RMSE of simple models with a random and a grouped split.
    %
    % For every seed, both splits are built; a linear regression and a
    % 1-nearest-neighbour model are fitted on training and evaluated on
    % validation. If a grouped split gives clearly larger errors, the
    % random split is optimistic because of the shared mixtures.
    % Only training and validation rows are used.
    %
    % Parameters
    % ----------
    % concrete_features : struct
    %     Output of preprocess_concrete_dataset (needs mixture_id_of_each_row).
    % split_fractions : 1 x 3 double
    % comparison_seeds : integer vector
    %
    % Returns
    % -------
    % split_comparison : struct (each field: one value per seed, MPa)
    %     linear_rmse_random, linear_rmse_grouped,
    %     nearest_neighbour_rmse_random, nearest_neighbour_rmse_grouped,
    %     comparison_seeds

    inputs = concrete_features.inputs;
    target = concrete_features.target;
    number_of_rows = size(inputs, 1);
    number_of_seeds = numel(comparison_seeds);

    linear_rmse_random             = zeros(1, number_of_seeds);
    linear_rmse_grouped            = zeros(1, number_of_seeds);
    nearest_neighbour_rmse_random  = zeros(1, number_of_seeds);
    nearest_neighbour_rmse_grouped = zeros(1, number_of_seeds);

    for seed_index = 1:number_of_seeds
        current_seed = comparison_seeds(seed_index);

        random_blocks  = assign_rows_to_random_blocks(number_of_rows, split_fractions, current_seed);
        grouped_blocks = assign_groups_to_blocks(concrete_features.mixture_id_of_each_row, split_fractions, current_seed);

        for split_kind = 1:2
            if split_kind == 1
                block_of_each_row = random_blocks;
            else
                block_of_each_row = grouped_blocks;
            end

            is_training   = (block_of_each_row == 1);
            is_validation = (block_of_each_row == 2);

            linear_coefficients = fit_linear_least_squares(inputs(is_training, :), target(is_training));
            linear_prediction   = predict_linear_model(inputs(is_validation, :), linear_coefficients);
            linear_rmse = compute_root_mean_squared_error(target(is_validation), linear_prediction);

            neighbour_prediction = predict_nearest_neighbour(inputs(is_training, :), target(is_training), inputs(is_validation, :));
            neighbour_rmse = compute_root_mean_squared_error(target(is_validation), neighbour_prediction);

            if split_kind == 1
                linear_rmse_random(seed_index)            = linear_rmse;
                nearest_neighbour_rmse_random(seed_index) = neighbour_rmse;
            else
                linear_rmse_grouped(seed_index)            = linear_rmse;
                nearest_neighbour_rmse_grouped(seed_index) = neighbour_rmse;
            end
        end
    end

    print_section_title(sprintf('CONCRETE - Step 4: random split versus split by mixture (%d seeds, validation only)', number_of_seeds));
    fprintf('Validation RMSE (MPa), mean +/- std over seeds\n\n');
    fprintf('%-26s %18s %18s\n', 'Model', 'Random split', 'Split by mixture');
    fprintf('%-26s %10.2f +/- %4.2f %10.2f +/- %4.2f\n', 'Linear regression', mean(linear_rmse_random), std(linear_rmse_random), mean(linear_rmse_grouped), std(linear_rmse_grouped));
    fprintf('%-26s %10.2f +/- %4.2f %10.2f +/- %4.2f\n', '1-nearest neighbour', mean(nearest_neighbour_rmse_random), std(nearest_neighbour_rmse_random), mean(nearest_neighbour_rmse_grouped), std(nearest_neighbour_rmse_grouped));

    split_comparison.linear_rmse_random             = linear_rmse_random;
    split_comparison.linear_rmse_grouped            = linear_rmse_grouped;
    split_comparison.nearest_neighbour_rmse_random  = nearest_neighbour_rmse_random;
    split_comparison.nearest_neighbour_rmse_grouped = nearest_neighbour_rmse_grouped;
    split_comparison.comparison_seeds               = comparison_seeds;
end


function prediction = predict_nearest_neighbour(training_inputs, training_target, query_inputs)
    % 1-nearest-neighbour regression on standardised inputs.
    %
    % Each query row gets the target of the closest training row
    % (Euclidean distance after standardising with training statistics).
    % It memorises the training set, so it benefits most from a query
    % that is almost a copy of a training row (same mixture).
    %
    % Parameters
    % ----------
    % training_inputs : N x K double
    % training_target : N x 1 double
    % query_inputs : M x K double
    %
    % Returns
    % -------
    % prediction : M x 1 double

    scaler = fit_standardization(training_inputs);
    scaled_training = apply_standardization(training_inputs, scaler);
    scaled_query    = apply_standardization(query_inputs, scaler);

    number_of_training_rows = size(scaled_training, 1);
    number_of_query_rows    = size(scaled_query, 1);

    prediction = zeros(number_of_query_rows, 1);

    for query_index = 1:number_of_query_rows
        differences = scaled_training - repmat(scaled_query(query_index, :), number_of_training_rows, 1);
        squared_distances = sum(differences .^ 2, 2);

        [unused_smallest_distance, nearest_row] = min(squared_distances);
        prediction(query_index) = training_target(nearest_row);
    end
end


%% ========================================================================
%  STEP 4 - DATA PARTITIONING: MELBOURNE
%  Functions used only by the Melbourne dataset.
%  ========================================================================

function partition = partition_melbourne_dataset(melbourne_features)
    % Chronological split of the Melbourne samples, with scaling and checks.
    %
    % Split
    % -----
    % Each sample goes to the block that contains its TARGET day, using
    % the cutoffs already fixed in Step 3 (training 1981-1987, validation
    % 1988-01-01 to 1989-07-01, test 1989-07-02 to 1990-12-31). Inputs
    % always precede the target, so no training sample contains a
    % validation or test value. The first validation samples use the last
    % training days as lags, which is allowed (it is the past).
    %
    % Scaling
    % -------
    % The lags and the target are the same variable, so they share ONE
    % mean and ONE standard deviation, computed from the training targets
    % only (the training period). The sin and cos columns are already in
    % [-1, 1] and are left unchanged (mean 0, standard deviation 1).
    %
    % Parameters
    % ----------
    % melbourne_features : struct
    %     Output of preprocess_melbourne_dataset.
    %
    % Returns
    % -------
    % partition : struct
    %     Output of build_partition + scale_partition, plus:
    %     split_cutoffs, leakage_checks_passed (logical)

    split_cutoffs = melbourne_features.split_cutoffs;
    block_of_each_sample = assign_samples_to_chronological_blocks(melbourne_features.target_dates, split_cutoffs);

    partition = build_partition(melbourne_features.inputs, melbourne_features.target, block_of_each_sample);

    % ---- One scale for all temperature columns ----------------------------
    temperature_scaler = fit_standardization(partition.training.target);

    number_of_lag_columns = melbourne_features.number_of_lag_columns;
    number_of_features    = size(melbourne_features.inputs, 2);
    number_of_calendar_columns = number_of_features - number_of_lag_columns;

    input_scaler.mean               = [repmat(temperature_scaler.mean, 1, number_of_lag_columns), zeros(1, number_of_calendar_columns)];
    input_scaler.standard_deviation = [repmat(temperature_scaler.standard_deviation, 1, number_of_lag_columns), ones(1, number_of_calendar_columns)];

    partition = scale_partition(partition, input_scaler, temperature_scaler);
    partition.split_cutoffs = split_cutoffs;

    % ---- Sizes and scaling check ------------------------------------------
    print_section_title('MELBOURNE - Step 4: chronological split');
    fprintf('Training   : targets %s to %s\n', datestr(min(melbourne_features.target_dates(block_of_each_sample == 1)), 'yyyy-mm-dd'), datestr(max(melbourne_features.target_dates(block_of_each_sample == 1)), 'yyyy-mm-dd'));
    fprintf('Validation : targets %s to %s\n', datestr(min(melbourne_features.target_dates(block_of_each_sample == 2)), 'yyyy-mm-dd'), datestr(max(melbourne_features.target_dates(block_of_each_sample == 2)), 'yyyy-mm-dd'));
    fprintf('Test       : targets %s to %s\n\n', datestr(min(melbourne_features.target_dates(block_of_each_sample == 3)), 'yyyy-mm-dd'), datestr(max(melbourne_features.target_dates(block_of_each_sample == 3)), 'yyyy-mm-dd'));

    print_partition_summary(partition, 'degC');
    fprintf('\nTemperature scale (lags and target) from training only: mean %.3f degC, std %.3f degC\n', temperature_scaler.mean, temperature_scaler.standard_deviation);
    fprintf('Calendar columns (sin, cos): not scaled\n');

    % ---- Leakage checks ----------------------------------------------------
    last_training_input_date  = max(melbourne_features.last_input_dates(block_of_each_sample == 1));
    last_training_target_date = max(melbourne_features.target_dates(block_of_each_sample == 1));
    last_validation_target_date = max(melbourne_features.target_dates(block_of_each_sample == 2));

    training_uses_only_training_days = (last_training_input_date < split_cutoffs.first_validation_date) && (last_training_target_date < split_cutoffs.first_validation_date);
    validation_ends_before_test      = (last_validation_target_date < split_cutoffs.first_test_date);
    blocks_are_in_time_order         = all(diff(block_of_each_sample(:)) >= 0);

    print_section_title('MELBOURNE - Step 4: leakage checks');
    print_check_result('Training inputs and targets end before the validation block', training_uses_only_training_days);
    print_check_result('Validation targets end before the test block', validation_ends_before_test);
    print_check_result('Blocks follow the time order (no shuffling)', blocks_are_in_time_order);
    print_check_result('Scaler computed from training targets only', true);

    partition.leakage_checks_passed = training_uses_only_training_days && validation_ends_before_test && blocks_are_in_time_order;
end


%% ========================================================================
%  STEP 4 - DATA PARTITIONING: UTILS
%  Functions used by both datasets (or generic helpers reused in later steps).
%  ========================================================================

function uniform_numbers = generate_uniform_random_numbers(seed, count)
    % Reproducible uniform random numbers in (0, 1), identical in MATLAB and Octave.
    %
    % Uses the Park-Miller "minimal standard" generator:
    %     state(k+1) = mod(16807 * state(k), 2^31 - 1),  u(k) = state(k) / (2^31 - 1)
    % Every product is below 2^53, so the arithmetic is exact in double
    % precision and both programs produce the same sequence. MATLAB's and
    % Octave's rand give different numbers for the same seed, which would
    % make the splits differ between the two programs.
    %
    % Parameters
    % ----------
    % seed : integer in 1 .. 2147483646
    % count : integer, how many numbers to generate
    %
    % Returns
    % -------
    % uniform_numbers : count x 1 double

    modulus    = 2147483647;
    multiplier = 16807;

    if seed < 1 || seed >= modulus || seed ~= round(seed)
        error('generate_uniform_random_numbers:badSeed', 'seed must be an integer between 1 and %d, got %g.', modulus - 1, seed);
    end

    uniform_numbers = zeros(count, 1);
    state = seed;

    for number_index = 1:count
        state = mod(multiplier * state, modulus);
        uniform_numbers(number_index) = state / modulus;
    end
end


function check_split_fractions(split_fractions)
    % Stop with an error if the split fractions are not valid.
    %
    % Parameters
    % ----------
    % split_fractions : 1 x 3 double
    %     Must be positive and sum to 1.

    if numel(split_fractions) ~= 3 || any(split_fractions <= 0) || abs(sum(split_fractions) - 1) > 1e-9
        error('check_split_fractions:badFractions', 'split_fractions must be 3 positive values summing to 1.');
    end
end


function partition = build_partition(inputs, target, block_of_each_sample)
    % Separate inputs and target into training, validation and test subsets.
    %
    % Parameters
    % ----------
    % inputs : N x K double (unscaled)
    % target : N x 1 double (unscaled)
    % block_of_each_sample : N x 1, values 1 (training), 2 (validation), 3 (test)
    %
    % Returns
    % -------
    % partition : struct with fields training, validation, test; each has
    %     inputs      : unscaled inputs of the subset
    %     target      : unscaled target of the subset
    %     row_indices : rows of the original matrix that belong to the subset
    % and block_of_each_sample (the input vector).

    subset_names = {'training', 'validation', 'test'};

    for block_number = 1:3
        row_indices = find(block_of_each_sample == block_number);

        subset.inputs      = inputs(row_indices, :);
        subset.target      = target(row_indices);
        subset.row_indices = row_indices;

        partition.(subset_names{block_number}) = subset;
    end

    partition.block_of_each_sample = block_of_each_sample(:);
end


function partition = scale_partition(partition, input_scaler, target_scaler)
    % Add scaled inputs and target to every subset of a partition.
    %
    % The scalers must have been fitted on the training subset only; the
    % same scalers are applied to training, validation and test.
    %
    % Parameters
    % ----------
    % partition : struct
    %     Output of build_partition.
    % input_scaler, target_scaler : struct
    %     Outputs of fit_standardization (or built with the same fields).
    %
    % Returns
    % -------
    % partition : struct
    %     Same as the input, plus inputs_scaled and target_scaled in each
    %     subset, and the fields input_scaler and target_scaler.

    subset_names = {'training', 'validation', 'test'};

    for subset_index = 1:3
        name = subset_names{subset_index};

        partition.(name).inputs_scaled = apply_standardization(partition.(name).inputs, input_scaler);
        partition.(name).target_scaled = apply_standardization(partition.(name).target, target_scaler);
    end

    partition.input_scaler  = input_scaler;
    partition.target_scaler = target_scaler;
end


function print_partition_summary(partition, target_unit)
    % Print subset sizes, target statistics and a check of the scaling.
    %
    % After correct scaling, the training inputs and target have mean 0
    % and standard deviation 1; validation and test are close to, but not
    % exactly, 0 and 1 (they were scaled with training statistics).
    %
    % Parameters
    % ----------
    % partition : struct
    %     Output of scale_partition.
    % target_unit : char
    %     Unit of the target, used in the printed table.

    subset_names  = {'training', 'validation', 'test'};
    subset_labels = {'Training', 'Validation', 'Test'};

    number_of_samples = numel(partition.block_of_each_sample);

    fprintf('%-11s %8s %8s %16s %16s %18s\n', 'Subset', 'Samples', 'Share', ['Target mean (' target_unit ')'], ['Target std (' target_unit ')'], 'Scaled target mean');

    for subset_index = 1:3
        subset = partition.(subset_names{subset_index});
        fprintf('%-11s %8d %7.1f%% %16.2f %16.2f %18.3f\n', subset_labels{subset_index}, numel(subset.target), 100 * numel(subset.target) / number_of_samples, mean(subset.target), std(subset.target), mean(subset.target_scaled));
    end

    training_input_means = mean(partition.training.inputs_scaled, 1);
    training_input_stds  = std(partition.training.inputs_scaled, 0, 1);

    fprintf('\nScaled training inputs: largest |mean| = %.2e, std between %.3f and %.3f\n', max(abs(training_input_means)), min(training_input_stds), max(training_input_stds));
end


function print_check_result(description, check_passed)
    % Print one line "OK" or "FAILED" for a verification.
    %
    % Parameters
    % ----------
    % description : char
    % check_passed : logical

    if check_passed
        result_text = 'OK';
    else
        result_text = 'FAILED';
    end

    fprintf('  [%-6s] %s\n', result_text, description);
end


%% ========================================================================
%  STEP 5 - IMPLEMENTATION AND VERIFICATION: UTILS
%  Functions used by both datasets: network, loss and gradient, the three
%  training algorithms, metrics and the gradient check (bonus).
%  ========================================================================
%
%  Conventions used by every function of this section
%  ---------------------------------------------------
%  layers : [d, h1, ..., hk, 1], number of neurons per layer, inputs first.
%  theta  : column vector with all weights and biases, layer by layer:
%           W1(:), b1, W2(:), b2, ... where Wl is fan_out x fan_in.
%  inputs : N x d matrix (one sample per row); target : N x 1 vector.
%  Hidden layers use tanh; the output neuron is linear; the loss is the
%  mean squared error (MSE) = mean((prediction - target).^2).

function architectures = build_architectures(number_of_inputs)
    % Layer sizes of the three configurations of Section 3 (N = 10).
    %
    % Parameters
    % ----------
    % number_of_inputs : integer
    %     Number d of input features.
    %
    % Returns
    % -------
    % architectures : struct
    %     names  : 1 x 3 cell, {'A Reference', 'B Wide', 'C Deep'}
    %     layers : 1 x 3 cell, layer-size vectors
    %              A = [d 10 1], B = [d 100 1], C = [d 10 10 ... 10 1] (10 hidden)

    reference_width = 10;

    architectures.names  = {'A Reference', 'B Wide', 'C Deep'};
    architectures.layers = {[number_of_inputs, reference_width, 1], [number_of_inputs, 10 * reference_width, 1], [number_of_inputs, reference_width * ones(1, 10), 1]};
end


function set_random_seed(seed)
    % Seed the random number generators of MATLAB / Octave.
    %
    % rand('twister', seed) works in MATLAB R2007a and in GNU Octave. The
    % two programs give different numbers for the same seed, but each one
    % is reproducible, which is what the paired comparison needs.
    %
    % Parameters
    % ----------
    % seed : non-negative integer

    rand('twister', seed);
    randn('state', seed);
end


function theta = init_params(layers, seed)
    % Glorot/Xavier uniform initialisation, identical for all architectures.
    %
    % W ~ U(-limit, limit) with limit = sqrt(6 / (fan_in + fan_out)), biases
    % 0. The same seed gives the same initial weights, so run k of the three
    % algorithms starts from the same point (paired comparison).
    %
    % Parameters
    % ----------
    % layers : integer vector, [d, h1, ..., 1]
    % seed : non-negative integer
    %
    % Returns
    % -------
    % theta : P x 1 double, P = count_network_parameters(layers)

    set_random_seed(seed);

    number_of_layers = numel(layers) - 1;
    weights = cell(1, number_of_layers);
    biases  = cell(1, number_of_layers);

    for layer_index = 1:number_of_layers
        fan_in  = layers(layer_index);
        fan_out = layers(layer_index + 1);

        limit = sqrt(6 / (fan_in + fan_out));

        weights{layer_index} = (2 * rand(fan_out, fan_in) - 1) * limit;
        biases{layer_index}  = zeros(fan_out, 1);
    end

    theta = pack_parameters(weights, biases);
end


function theta = pack_parameters(weights, biases)
    % Put the weight matrices and bias vectors in one column vector.
    %
    % Parameters
    % ----------
    % weights : 1 x L cell, weights{l} is fan_out x fan_in
    % biases : 1 x L cell, biases{l} is fan_out x 1
    %
    % Returns
    % -------
    % theta : P x 1 double, [W1(:); b1; W2(:); b2; ...]

    theta = [];

    for layer_index = 1:numel(weights)
        theta = [theta; weights{layer_index}(:); biases{layer_index}(:)];
    end
end


function [weights, biases] = unpack_parameters(theta, layers)
    % Inverse of pack_parameters.
    %
    % Parameters
    % ----------
    % theta : P x 1 double
    % layers : integer vector, [d, h1, ..., 1]
    %
    % Returns
    % -------
    % weights : 1 x L cell, weights{l} is fan_out x fan_in
    % biases : 1 x L cell, biases{l} is fan_out x 1

    number_of_layers = numel(layers) - 1;
    weights = cell(1, number_of_layers);
    biases  = cell(1, number_of_layers);

    offset = 0;

    for layer_index = 1:number_of_layers
        fan_in  = layers(layer_index);
        fan_out = layers(layer_index + 1);

        number_of_weights = fan_out * fan_in;
        weights{layer_index} = reshape(theta(offset + 1:offset + number_of_weights), fan_out, fan_in);
        offset = offset + number_of_weights;

        biases{layer_index} = theta(offset + 1:offset + fan_out);
        offset = offset + fan_out;
    end
end


function prediction = predict_network(theta, layers, inputs)
    % Forward pass: tanh hidden layers, linear output.
    %
    % Parameters
    % ----------
    % theta : P x 1 double
    % layers : integer vector, [d, h1, ..., 1]
    % inputs : N x d double
    %
    % Returns
    % -------
    % prediction : N x 1 double

    [weights, biases] = unpack_parameters(theta, layers);

    number_of_layers  = numel(weights);
    number_of_samples = size(inputs, 1);

    activation = inputs;

    for layer_index = 1:number_of_layers - 1
        activation = tanh(activation * weights{layer_index}' + repmat(biases{layer_index}', number_of_samples, 1));
    end

    prediction = activation * weights{number_of_layers}' + repmat(biases{number_of_layers}', number_of_samples, 1);
end


function loss = compute_mse_loss(theta, layers, inputs, target)
    % Mean squared error of the network on a data set (no gradient).
    %
    % Parameters
    % ----------
    % theta : P x 1 double
    % layers : integer vector
    % inputs : N x d double
    % target : N x 1 double
    %
    % Returns
    % -------
    % loss : double, mean((prediction - target).^2)

    errors = predict_network(theta, layers, inputs) - target(:);

    loss = mean(errors .^ 2);
end


function [loss, gradient] = loss_grad(theta, layers, inputs, target)
    % MSE loss and its gradient with respect to all weights and biases.
    %
    % Backpropagation for tanh hidden layers and a linear output:
    %   delta_L = 2 (prediction - target) / N
    %   dW_l = delta_l' * a_(l-1),   db_l = sum of delta_l over samples
    %   delta_(l-1) = (delta_l * W_l) .* (1 - a_(l-1).^2)
    %
    % Parameters
    % ----------
    % theta : P x 1 double
    % layers : integer vector, [d, h1, ..., 1]
    % inputs : N x d double
    % target : N x 1 double
    %
    % Returns
    % -------
    % loss : double, MSE on (inputs, target)
    % gradient : P x 1 double, same layout as theta

    [weights, biases] = unpack_parameters(theta, layers);

    number_of_layers  = numel(weights);
    number_of_samples = size(inputs, 1);

    % ---- Forward pass, keeping every layer's output --------------------
    % activations{1} = inputs; activations{l+1} = output of hidden layer l.
    activations = cell(1, number_of_layers);
    activations{1} = inputs;

    for layer_index = 1:number_of_layers - 1
        activations{layer_index + 1} = tanh(activations{layer_index} * weights{layer_index}' + repmat(biases{layer_index}', number_of_samples, 1));
    end

    prediction = activations{number_of_layers} * weights{number_of_layers}' + repmat(biases{number_of_layers}', number_of_samples, 1);

    errors = prediction - target(:);
    loss = mean(errors .^ 2);

    % ---- Backward pass --------------------------------------------------
    weight_gradients = cell(1, number_of_layers);
    bias_gradients   = cell(1, number_of_layers);

    delta = 2 * errors / number_of_samples;

    for layer_index = number_of_layers:-1:1
        weight_gradients{layer_index} = delta' * activations{layer_index};
        bias_gradients{layer_index}   = sum(delta, 1)';

        if layer_index > 1
            delta = (delta * weights{layer_index}) .* (1 - activations{layer_index} .^ 2);
        end
    end

    gradient = pack_parameters(weight_gradients, bias_gradients);
end


function options = default_training_options()
    % Default settings of the three training algorithms.
    %
    % Returns
    % -------
    % options : struct
    %     budget_epoch_equivalents : computational budget, identical for
    %                                all algorithms of a dataset (1000)
    %     batch_size               : 32, same for mini-batch GD and Adam
    %     learning_rate            : placeholder, tuned in Step 6
    %     shuffle_seed             : seed of the mini-batch order
    %     adam_beta1, adam_beta2, adam_epsilon : 0.9, 0.999, 1e-8
    %     bfgs_gradient_tolerance  : stop when max |gradient| < 1e-6
    %     bfgs_armijo_constant     : c1 = 1e-4 (sufficient decrease)
    %     bfgs_backtracking_factor : step multiplied by 0.5 after each
    %                                rejected trial
    %     bfgs_maximum_line_search_evaluations : 20 trials per iteration
    %     bfgs_curvature_threshold : skip the update when s'y <= 1e-10
    %     verbose                  : print progress (false)

    options.budget_epoch_equivalents = 1000;
    options.batch_size               = 32;
    options.learning_rate            = 0.01;
    options.shuffle_seed             = 1;

    options.adam_beta1   = 0.9;
    options.adam_beta2   = 0.999;
    options.adam_epsilon = 1e-8;

    options.bfgs_gradient_tolerance              = 1e-6;
    options.bfgs_armijo_constant                 = 1e-4;
    options.bfgs_backtracking_factor             = 0.5;
    options.bfgs_maximum_line_search_evaluations = 20;
    options.bfgs_curvature_threshold             = 1e-10;

    options.verbose = false;
end


function history = create_training_history(budget_epoch_equivalents)
    % Empty history with one slot per epoch-equivalent.
    %
    % Slot k holds the state after k epoch-equivalents (one pass over the
    % training set for GD/Adam, one full-batch gradient evaluation for
    % BFGS, line-search evaluations included), so the curves of the three
    % algorithms share the same horizontal axis.
    %
    % Parameters
    % ----------
    % budget_epoch_equivalents : integer
    %
    % Returns
    % -------
    % history : struct
    %     epoch_equivalents   : (1:budget)'
    %     training_loss       : budget x 1, NaN until filled (scaled units)
    %     validation_loss     : budget x 1, NaN until filled (scaled units)
    %     elapsed_seconds     : budget x 1, cumulative training time
    %     initial_training_loss, initial_validation_loss : at theta0
    %     best_validation_loss, best_epoch_equivalent : model selection
    %     gradient_evaluations : full-dataset gradient evaluations used
    %     total_training_seconds, diverged, stop_reason

    history.epoch_equivalents = (1:budget_epoch_equivalents)';
    history.training_loss     = NaN(budget_epoch_equivalents, 1);
    history.validation_loss   = NaN(budget_epoch_equivalents, 1);
    history.elapsed_seconds   = NaN(budget_epoch_equivalents, 1);

    history.initial_training_loss   = NaN;
    history.initial_validation_loss = NaN;
    history.best_validation_loss    = Inf;
    history.best_epoch_equivalent   = 0;
    history.gradient_evaluations    = 0;
    history.total_training_seconds  = 0;
    history.diverged    = false;
    history.stop_reason = 'budget reached';
end


function [best_theta, history] = train_minibatch_gd(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options)
    % Mini-batch gradient descent: theta <- theta - learning_rate * g.
    %
    % g is the gradient of the MSE on a mini-batch of options.batch_size
    % samples; the samples are reshuffled at every epoch. One epoch = one
    % epoch-equivalent = one full-dataset gradient evaluation. After each
    % epoch the training and validation losses are recorded; the
    % parameters with the lowest validation loss are returned. If the loss
    % becomes Inf or NaN, the run stops and is marked as diverged.
    %
    % Parameters
    % ----------
    % theta0 : P x 1 double, initial parameters (from init_params)
    % layers : integer vector
    % training_inputs, training_target : scaled training data
    % validation_inputs, validation_target : scaled validation data
    % options : struct, see default_training_options
    %     (uses budget_epoch_equivalents, batch_size, learning_rate,
    %     shuffle_seed)
    %
    % Returns
    % -------
    % best_theta : P x 1 double, parameters with the lowest validation loss
    % history : struct, see create_training_history

    [best_theta, history] = train_with_minibatches(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options, 'gd');
end


function [best_theta, history] = train_adam(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options)
    % Adam: first and second moment estimates with bias correction.
    %
    % For every mini-batch gradient g (update counter t):
    %   m = beta1 m + (1 - beta1) g
    %   v = beta2 v + (1 - beta2) g.^2
    %   theta <- theta - learning_rate * (m / (1 - beta1^t)) ./ (sqrt(v / (1 - beta2^t)) + epsilon)
    % Mini-batches, recording and model selection as in train_minibatch_gd.
    %
    % Parameters
    % ----------
    % Same as train_minibatch_gd; also uses adam_beta1, adam_beta2 and
    % adam_epsilon from options.
    %
    % Returns
    % -------
    % best_theta : P x 1 double
    % history : struct, see create_training_history

    [best_theta, history] = train_with_minibatches(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options, 'adam');
end


function [best_theta, history] = train_with_minibatches(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options, update_rule)
    % Shared loop of mini-batch GD and Adam (they differ only in the update).
    %
    % Parameters
    % ----------
    % theta0, layers, training_*, validation_*, options : see train_minibatch_gd
    % update_rule : char, 'gd' or 'adam'
    %
    % Returns
    % -------
    % best_theta : P x 1 double
    % history : struct, see create_training_history

    budget = options.budget_epoch_equivalents;
    batch_size = options.batch_size;
    learning_rate = options.learning_rate;
    number_of_samples = size(training_inputs, 1);

    history = create_training_history(budget);

    theta = theta0;
    best_theta = theta0;

    history.initial_training_loss   = compute_mse_loss(theta0, layers, training_inputs, training_target);
    history.initial_validation_loss = compute_mse_loss(theta0, layers, validation_inputs, validation_target);
    history.best_validation_loss    = history.initial_validation_loss;

    % Adam state
    first_moment  = zeros(size(theta0));
    second_moment = zeros(size(theta0));
    update_counter = 0;

    % The mini-batch order depends only on the run's shuffle seed.
    set_random_seed(options.shuffle_seed);

    training_seconds = 0;

    for epoch = 1:budget
        epoch_start = clock;

        [unused_sorted, sample_order] = sort(rand(number_of_samples, 1));

        for batch_start = 1:batch_size:number_of_samples
            batch_rows = sample_order(batch_start:min(batch_start + batch_size - 1, number_of_samples));

            [unused_batch_loss, gradient] = loss_grad(theta, layers, training_inputs(batch_rows, :), training_target(batch_rows));

            if strcmp(update_rule, 'gd')
                theta = theta - learning_rate * gradient;
            else
                update_counter = update_counter + 1;
                first_moment  = options.adam_beta1 * first_moment  + (1 - options.adam_beta1) * gradient;
                second_moment = options.adam_beta2 * second_moment + (1 - options.adam_beta2) * gradient .^ 2;

                corrected_first_moment  = first_moment  / (1 - options.adam_beta1 ^ update_counter);
                corrected_second_moment = second_moment / (1 - options.adam_beta2 ^ update_counter);

                theta = theta - learning_rate * corrected_first_moment ./ (sqrt(corrected_second_moment) + options.adam_epsilon);
            end
        end

        training_seconds = training_seconds + etime(clock, epoch_start);

        % ---- Record (monitoring time is not counted as training time) ---
        training_loss   = compute_mse_loss(theta, layers, training_inputs, training_target);
        validation_loss = compute_mse_loss(theta, layers, validation_inputs, validation_target);

        history.training_loss(epoch)   = training_loss;
        history.validation_loss(epoch) = validation_loss;
        history.elapsed_seconds(epoch) = training_seconds;
        history.gradient_evaluations   = epoch;

        if ~isfinite(training_loss) || ~isfinite(validation_loss)
            history.diverged = true;
            history.stop_reason = sprintf('diverged at epoch %d', epoch);
            break;
        end

        if validation_loss < history.best_validation_loss
            history.best_validation_loss  = validation_loss;
            history.best_epoch_equivalent = epoch;
            best_theta = theta;
        end

        if options.verbose && mod(epoch, 100) == 0
            fprintf('    %s epoch %4d: training %.4f, validation %.4f\n', update_rule, epoch, training_loss, validation_loss);
        end
    end

    history.total_training_seconds = training_seconds;
end


function [best_theta, history] = train_bfgs(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options)
    % BFGS quasi-Newton method on the full training set.
    %
    % Search direction p = -H g, where H approximates the INVERSE Hessian.
    % Line search: backtracking with the Armijo condition
    %     f(theta + alpha p) <= f(theta) + c1 alpha g'p,
    % starting at alpha = 1 and multiplying alpha by the backtracking
    % factor after each rejected trial (at most a fixed number of trials).
    % Inverse-Hessian update (s = step, y = change of gradient, rho = 1/(y's)):
    %     H <- (I - rho s y') H (I - rho y s') + rho s s'
    % applied only when y's > curvature threshold (keeps H positive
    % definite); before the first update H is scaled by (y's)/(y'y).
    % If p is not a descent direction or the line search fails, H is reset
    % to the identity.
    %
    % Cost accounting: every evaluation of loss and gradient on the full
    % training set (initial point and every line-search trial) uses one
    % epoch-equivalent. The run stops when the budget is used or when
    % max |g| < gradient tolerance; in that case the remaining slots of
    % the history keep the last values (the parameters no longer change).
    %
    % Parameters
    % ----------
    % Same as train_minibatch_gd; uses budget_epoch_equivalents and the
    % bfgs_* fields of options (no learning rate).
    %
    % Returns
    % -------
    % best_theta : P x 1 double, parameters with the lowest validation loss
    % history : struct, see create_training_history, plus
    %     number_of_iterations, skipped_updates, failed_line_searches

    budget = options.budget_epoch_equivalents;
    number_of_parameters = numel(theta0);

    history = create_training_history(budget);
    history.number_of_iterations = 0;
    history.skipped_updates      = 0;
    history.failed_line_searches = 0;

    training_seconds = 0;
    iteration_start = clock;

    theta = theta0;
    [loss, gradient] = loss_grad(theta, layers, training_inputs, training_target);
    evaluations_used = 1;

    inverse_hessian = eye(number_of_parameters);
    hessian_was_scaled = false;

    training_seconds = training_seconds + etime(clock, iteration_start);

    best_theta = theta0;
    history.initial_training_loss   = loss;
    history.initial_validation_loss = compute_mse_loss(theta0, layers, validation_inputs, validation_target);
    history.best_validation_loss    = history.initial_validation_loss;

    history.training_loss(1)   = loss;
    history.validation_loss(1) = history.initial_validation_loss;
    history.elapsed_seconds(1) = training_seconds;

    while evaluations_used < budget

        if max(abs(gradient)) < options.bfgs_gradient_tolerance
            history.stop_reason = sprintf('gradient tolerance reached after %d evaluations', evaluations_used);
            break;
        end

        iteration_start = clock;
        evaluations_before_iteration = evaluations_used;

        % ---- Search direction ---------------------------------------------
        direction = -inverse_hessian * gradient;
        slope = gradient' * direction;

        if slope >= 0
            inverse_hessian = eye(number_of_parameters);
            direction = -gradient;
            slope = gradient' * direction;
        end

        % ---- Backtracking line search (Armijo) -----------------------------
        step_length = 1;
        step_accepted = false;
        line_search_evaluations = 0;

        while evaluations_used < budget && line_search_evaluations < options.bfgs_maximum_line_search_evaluations
            trial_theta = theta + step_length * direction;
            [trial_loss, trial_gradient] = loss_grad(trial_theta, layers, training_inputs, training_target);

            evaluations_used = evaluations_used + 1;
            line_search_evaluations = line_search_evaluations + 1;

            if isfinite(trial_loss) && trial_loss <= loss + options.bfgs_armijo_constant * step_length * slope
                step_accepted = true;
                break;
            end

            step_length = step_length * options.bfgs_backtracking_factor;
        end

        % ---- Update of theta and of the inverse Hessian --------------------
        if step_accepted
            step = trial_theta - theta;
            gradient_change = trial_gradient - gradient;
            curvature = step' * gradient_change;

            theta = trial_theta;
            loss = trial_loss;
            gradient = trial_gradient;

            if curvature > options.bfgs_curvature_threshold
                if ~hessian_was_scaled
                    inverse_hessian = (curvature / (gradient_change' * gradient_change)) * eye(number_of_parameters);
                    hessian_was_scaled = true;
                end

                rho = 1 / curvature;
                hessian_times_change = inverse_hessian * gradient_change;

                inverse_hessian = inverse_hessian - rho * (step * hessian_times_change' + hessian_times_change * step') + (rho ^ 2 * (gradient_change' * hessian_times_change) + rho) * (step * step');
            else
                history.skipped_updates = history.skipped_updates + 1;
            end
        else
            inverse_hessian = eye(number_of_parameters);
            history.failed_line_searches = history.failed_line_searches + 1;
        end

        history.number_of_iterations = history.number_of_iterations + 1;
        training_seconds = training_seconds + etime(clock, iteration_start);

        % ---- Record: the slots used by this iteration get the new state ----
        validation_loss = compute_mse_loss(theta, layers, validation_inputs, validation_target);
        slots = (evaluations_before_iteration + 1):evaluations_used;

        history.training_loss(slots)   = loss;
        history.validation_loss(slots) = validation_loss;
        history.elapsed_seconds(slots) = training_seconds;

        if validation_loss < history.best_validation_loss
            history.best_validation_loss  = validation_loss;
            history.best_epoch_equivalent = evaluations_used;
            best_theta = theta;
        end

        if options.verbose && mod(history.number_of_iterations, 50) == 0
            fprintf('    bfgs iteration %4d (%4d evaluations): training %.4f, validation %.4f\n', history.number_of_iterations, evaluations_used, loss, validation_loss);
        end
    end

    % Early stop: the parameters stay the same until the end of the budget.
    if evaluations_used < budget
        remaining_slots = (evaluations_used + 1):budget;
        history.training_loss(remaining_slots)   = history.training_loss(evaluations_used);
        history.validation_loss(remaining_slots) = history.validation_loss(evaluations_used);
        history.elapsed_seconds(remaining_slots) = history.elapsed_seconds(evaluations_used);
    end

    history.gradient_evaluations   = evaluations_used;
    history.total_training_seconds = training_seconds;
end


function metrics = compute_regression_metrics(observed_values, predicted_values)
    % MSE, RMSE, MAE and R^2 (use values in the ORIGINAL unit of the target).
    %
    % Parameters
    % ----------
    % observed_values, predicted_values : double vectors of the same length
    %
    % Returns
    % -------
    % metrics : struct
    %     mse  : mean((observed - predicted).^2)
    %     rmse : sqrt(mse), in the unit of the target
    %     mae  : mean(|observed - predicted|), in the unit of the target
    %     r2   : 1 - sum of squared errors / sum of squares around the mean

    observed_values  = observed_values(:);
    predicted_values = predicted_values(:);

    errors = observed_values - predicted_values;

    metrics.mse  = mean(errors .^ 2);
    metrics.rmse = sqrt(metrics.mse);
    metrics.mae  = mean(abs(errors));
    metrics.r2   = 1 - sum(errors .^ 2) / sum((observed_values - mean(observed_values)) .^ 2);
end


function gradient_check = check_gradient(theta, layers, inputs, target, finite_difference_step)
    % Compare the backpropagation gradient with central finite differences.
    %
    % For every parameter i:
    %   numerical_i = (loss(theta + h e_i) - loss(theta - h e_i)) / (2 h)
    %   relative error_i = |analytical_i - numerical_i| / max(|analytical_i| + |numerical_i|, 1e-8)
    % The floor 1e-8 avoids dividing by ~0 for parameters whose gradient is
    % practically zero; for those parameters the relative error is still
    % inflated by rounding, so the global relative error (norm of the
    % difference) is also reported. Values around 1e-7 or smaller, with
    % absolute errors near 1e-10, mean the analytical gradient is correct.
    %
    % Parameters
    % ----------
    % theta : P x 1 double
    % layers : integer vector
    % inputs : N x d double (a small subset is enough)
    % target : N x 1 double
    % finite_difference_step : double, h (e.g. 1e-5)
    %
    % Returns
    % -------
    % gradient_check : struct
    %     maximum_relative_error : largest per-parameter relative error
    %     global_relative_error  : norm(analytical - numerical) /
    %                              (norm(analytical) + norm(numerical)),
    %                              not affected by parameters whose
    %                              gradient is almost zero
    %     maximum_absolute_error, number_of_parameters

    [unused_loss, analytical_gradient] = loss_grad(theta, layers, inputs, target);

    number_of_parameters = numel(theta);
    numerical_gradient = zeros(number_of_parameters, 1);

    for parameter_index = 1:number_of_parameters
        theta_plus  = theta;
        theta_minus = theta;
        theta_plus(parameter_index)  = theta_plus(parameter_index)  + finite_difference_step;
        theta_minus(parameter_index) = theta_minus(parameter_index) - finite_difference_step;

        numerical_gradient(parameter_index) = (compute_mse_loss(theta_plus, layers, inputs, target) - compute_mse_loss(theta_minus, layers, inputs, target)) / (2 * finite_difference_step);
    end

    absolute_errors = abs(analytical_gradient - numerical_gradient);
    relative_errors = absolute_errors ./ max(abs(analytical_gradient) + abs(numerical_gradient), 1e-8);

    gradient_check.maximum_relative_error = max(relative_errors);
    gradient_check.global_relative_error  = norm(analytical_gradient - numerical_gradient) / (norm(analytical_gradient) + norm(numerical_gradient));
    gradient_check.maximum_absolute_error = max(absolute_errors);
    gradient_check.number_of_parameters   = number_of_parameters;
end


function gradient_checks = verify_gradients_for_all_architectures(dataset_name, partition, number_of_samples, seed, finite_difference_step)
    % Run check_gradient for architectures A, B and C on one dataset (bonus).
    %
    % Uses the first number_of_samples rows of the scaled training set and
    % the initial weights init_params(layers, seed).
    %
    % Parameters
    % ----------
    % dataset_name : char, used in the printed title
    % partition : struct, output of partition_concrete_dataset or
    %             partition_melbourne_dataset
    % number_of_samples : integer
    % seed : integer, seed of the initial weights
    % finite_difference_step : double
    %
    % Returns
    % -------
    % gradient_checks : 1 x 3 struct array, outputs of check_gradient

    inputs = partition.training.inputs_scaled(1:number_of_samples, :);
    target = partition.training.target_scaled(1:number_of_samples);

    architectures = build_architectures(size(inputs, 2));

    print_section_title(sprintf('%s - Step 5: gradient check (%d samples, h = %g)', dataset_name, number_of_samples, finite_difference_step));
    fprintf('%-14s %11s %20s %22s %20s\n', 'Architecture', 'Parameters', 'Max relative error', 'Global relative error', 'Max absolute error');

    for architecture_index = 1:3
        layers = architectures.layers{architecture_index};
        theta = init_params(layers, seed);

        result = check_gradient(theta, layers, inputs, target, finite_difference_step);
        gradient_checks(architecture_index) = result;

        fprintf('%-14s %11d %20.3e %22.3e %20.3e\n', architectures.names{architecture_index}, result.number_of_parameters, result.maximum_relative_error, result.global_relative_error, result.maximum_absolute_error);
    end
end


function smoke_test_results = run_training_smoke_test(dataset_name, partition, options)
    % Short training of architecture A with the three algorithms.
    %
    % Only checks that every algorithm runs and lowers the loss; it is not
    % an experiment (the learning rates are tuned in Step 6 and the final
    % runs are in Step 7). The test set is not used.
    %
    % Parameters
    % ----------
    % dataset_name : char
    % partition : struct, output of a partition_* function
    % options : struct, see default_training_options (a small budget is
    %           enough); options.learning_rate_gd and
    %           options.learning_rate_adam give the learning rates, and
    %           options.shuffle_seed is used both for the initial weights
    %           and for the mini-batch order
    %
    % Returns
    % -------
    % smoke_test_results : 1 x 3 struct array with fields algorithm,
    %     initial_training_loss, final_training_loss,
    %     best_validation_loss, gradient_evaluations, seconds

    architectures = build_architectures(size(partition.training.inputs_scaled, 2));
    layers = architectures.layers{1};
    theta0 = init_params(layers, options.shuffle_seed);

    training_inputs   = partition.training.inputs_scaled;
    training_target   = partition.training.target_scaled;
    validation_inputs = partition.validation.inputs_scaled;
    validation_target = partition.validation.target_scaled;

    algorithm_names = {'Mini-batch GD', 'BFGS', 'Adam'};

    print_section_title(sprintf('%s - Step 5: smoke test (architecture A, %d epoch-equivalents)', dataset_name, options.budget_epoch_equivalents));
    fprintf('%-14s %14s %14s %16s %12s %10s\n', 'Algorithm', 'Initial train', 'Final train', 'Best validation', 'Grad. evals', 'Seconds');

    for algorithm_index = 1:3
        if algorithm_index == 1
            options.learning_rate = options.learning_rate_gd;
            [unused_theta, history] = train_minibatch_gd(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options);
        elseif algorithm_index == 2
            [unused_theta, history] = train_bfgs(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options);
        else
            options.learning_rate = options.learning_rate_adam;
            [unused_theta, history] = train_adam(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options);
        end

        final_training_loss = history.training_loss(find(~isnan(history.training_loss), 1, 'last'));

        fprintf('%-14s %14.4f %14.4f %16.4f %12d %10.2f\n', algorithm_names{algorithm_index}, history.initial_training_loss, final_training_loss, history.best_validation_loss, history.gradient_evaluations, history.total_training_seconds);

        smoke_test_results(algorithm_index).algorithm             = algorithm_names{algorithm_index};
        smoke_test_results(algorithm_index).initial_training_loss = history.initial_training_loss;
        smoke_test_results(algorithm_index).final_training_loss   = final_training_loss;
        smoke_test_results(algorithm_index).best_validation_loss  = history.best_validation_loss;
        smoke_test_results(algorithm_index).gradient_evaluations  = history.gradient_evaluations;
        smoke_test_results(algorithm_index).seconds               = history.total_training_seconds;
    end
end

%% ========================================================================
%  STEP 6 - HYPERPARAMETER TUNING: UTILS
%  Learning-rate tuning of mini-batch GD and Adam on the validation set.
%  ========================================================================

function tuning_results = tune_learning_rates(dataset_name, partition, learning_rate_grids, tuning_seeds, options)
    % Tune the learning rate of mini-batch GD and Adam for A, B and C.
    %
    % For every architecture and every algorithm, each learning rate of
    % the grid is trained once per tuning seed. The score of one run is
    % its best validation loss (the same model-selection rule used by the
    % training functions). A run that diverges scores Inf: a learning
    % rate that sometimes explodes is not safe for the 30 final runs.
    % The chosen learning rate has the lowest MEDIAN score over the
    % seeds (robust to one unlucky initialisation).
    %
    % Seed k gives the same initial weights to GD and Adam (paired), and
    % is also the seed of the mini-batch order. The test set is not used.
    %
    % Parameters
    % ----------
    % dataset_name : char
    %     Used in the printed titles.
    % partition : struct
    %     Output of partition_concrete_dataset or partition_melbourne_dataset.
    % learning_rate_grids : struct
    %     gd   : vector of learning rates for mini-batch GD (logarithmic grid)
    %     adam : vector of learning rates for Adam (logarithmic grid)
    % tuning_seeds : integer vector
    %     One repetition per seed (at least 5).
    % options : struct
    %     See default_training_options (budget, batch size, Adam constants).
    %
    % Returns
    % -------
    % tuning_results : struct
    %     dataset_name, tuning_seeds, budget_epoch_equivalents
    %     architecture_names : 1 x 3 cell
    %     layers             : 1 x 3 cell
    %     algorithm_names    : {'Mini-batch GD', 'Adam'}
    %     algorithm_codes    : {'gd', 'adam'}
    %     learning_rate_grid : 1 x 2 cell, grid of each algorithm
    %     validation_scores  : 3 x 2 cell, each (grid values x seeds),
    %                          best validation loss (Inf if diverged)
    %     diverged_runs      : 3 x 2 cell, each (grid values x seeds), logical
    %     median_scores      : 3 x 2 cell, each (grid values x 1)
    %     chosen_learning_rate : 3 x 2 double (row = architecture,
    %                            column 1 = GD, column 2 = Adam)
    %     chosen_on_grid_edge  : 3 x 2 logical
    %     tuning_seconds       : 3 x 2 double, total training time

    number_of_inputs = size(partition.training.inputs_scaled, 2);
    architectures = build_architectures(number_of_inputs);

    algorithm_codes = {'gd', 'adam'};
    algorithm_names = {'Mini-batch GD', 'Adam'};
    learning_rate_grid = {learning_rate_grids.gd(:)', learning_rate_grids.adam(:)'};

    number_of_architectures = numel(architectures.layers);
    number_of_algorithms    = numel(algorithm_codes);

    % ---- Check the minimum required by the assignment --------------------
    if numel(tuning_seeds) < 5
        fprintf('WARNING: the assignment asks for at least 5 repetitions (got %d).\n', numel(tuning_seeds));
    end

    for algorithm_index = 1:number_of_algorithms
        if numel(learning_rate_grid{algorithm_index}) < 6
            fprintf('WARNING: the assignment asks for at least 6 learning rates (%s has %d).\n', ...
                    algorithm_names{algorithm_index}, numel(learning_rate_grid{algorithm_index}));
        end
    end

    % ---- Allocate the results --------------------------------------------
    validation_scores    = cell(number_of_architectures, number_of_algorithms);
    diverged_runs        = cell(number_of_architectures, number_of_algorithms);
    median_scores        = cell(number_of_architectures, number_of_algorithms);
    chosen_learning_rate = zeros(number_of_architectures, number_of_algorithms);
    chosen_on_grid_edge  = false(number_of_architectures, number_of_algorithms);
    tuning_seconds       = zeros(number_of_architectures, number_of_algorithms);

    print_section_title(sprintf('%s - Step 6: learning-rate tuning (%d repetitions, %d epoch-equivalents)', ...
                                dataset_name, numel(tuning_seeds), options.budget_epoch_equivalents));

    % ---- Grid search -----------------------------------------------------
    for architecture_index = 1:number_of_architectures
        layers = architectures.layers{architecture_index};

        for algorithm_index = 1:number_of_algorithms
            fprintf('  %-12s | %-14s : %d learning rates x %d seeds ...\n', ...
                    architectures.names{architecture_index}, algorithm_names{algorithm_index}, ...
                    numel(learning_rate_grid{algorithm_index}), numel(tuning_seeds));

            [score_matrix, diverged_matrix, total_seconds] = run_learning_rate_grid(layers, partition, ...
                                                                                    algorithm_codes{algorithm_index}, ...
                                                                                    learning_rate_grid{algorithm_index}, ...
                                                                                    tuning_seeds, options);

            [learning_rate, unused_chosen_index, median_of_each_rate, on_edge] = select_learning_rate(learning_rate_grid{algorithm_index}, score_matrix);

            validation_scores{architecture_index, algorithm_index}    = score_matrix;
            diverged_runs{architecture_index, algorithm_index}        = diverged_matrix;
            median_scores{architecture_index, algorithm_index}        = median_of_each_rate;
            chosen_learning_rate(architecture_index, algorithm_index) = learning_rate;
            chosen_on_grid_edge(architecture_index, algorithm_index)  = on_edge;
            tuning_seconds(architecture_index, algorithm_index)       = total_seconds;

            fprintf('      chosen learning rate = %.4g   (%.1f s of training)\n', learning_rate, total_seconds);

            if on_edge
                fprintf('      WARNING: the chosen value is at the edge of the grid; consider extending the grid.\n');
            end
        end
    end

    % ---- Store everything ------------------------------------------------
    tuning_results.dataset_name             = dataset_name;
    tuning_results.tuning_seeds             = tuning_seeds;
    tuning_results.budget_epoch_equivalents = options.budget_epoch_equivalents;
    tuning_results.architecture_names       = architectures.names;
    tuning_results.layers                   = architectures.layers;
    tuning_results.algorithm_names          = algorithm_names;
    tuning_results.algorithm_codes          = algorithm_codes;
    tuning_results.learning_rate_grid       = learning_rate_grid;
    tuning_results.validation_scores        = validation_scores;
    tuning_results.diverged_runs            = diverged_runs;
    tuning_results.median_scores            = median_scores;
    tuning_results.chosen_learning_rate     = chosen_learning_rate;
    tuning_results.chosen_on_grid_edge      = chosen_on_grid_edge;
    tuning_results.tuning_seconds           = tuning_seconds;

    print_learning_rate_tuning_table(tuning_results);
end


function [score_matrix, diverged_matrix, total_seconds] = run_learning_rate_grid(layers, partition, algorithm_code, learning_rate_grid, tuning_seeds, options)
    % Train one architecture with one algorithm for every (rate, seed) pair.
    %
    % Parameters
    % ----------
    % layers : integer vector, [d, h1, ..., 1]
    % partition : struct, output of a partition_* function
    % algorithm_code : char, 'gd' or 'adam'
    % learning_rate_grid : double vector
    % tuning_seeds : integer vector
    % options : struct, see default_training_options
    %
    % Returns
    % -------
    % score_matrix : (grid values x seeds) double
    %     Best validation loss of each run (scaled units), Inf if diverged.
    % diverged_matrix : (grid values x seeds) logical
    % total_seconds : double, training time of all runs

    training_inputs   = partition.training.inputs_scaled;
    training_target   = partition.training.target_scaled;
    validation_inputs = partition.validation.inputs_scaled;
    validation_target = partition.validation.target_scaled;

    number_of_rates = numel(learning_rate_grid);
    number_of_seeds = numel(tuning_seeds);

    score_matrix    = zeros(number_of_rates, number_of_seeds);
    diverged_matrix = false(number_of_rates, number_of_seeds);
    total_seconds   = 0;

    for rate_index = 1:number_of_rates

        options.learning_rate = learning_rate_grid(rate_index);

        for seed_index = 1:number_of_seeds

            seed = tuning_seeds(seed_index);

            % Same seed -> same initial weights for GD and Adam (paired).
            theta0 = init_params(layers, seed);
            options.shuffle_seed = seed;

            if strcmp(algorithm_code, 'gd')
                [unused_theta, history] = train_minibatch_gd(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options);
            else
                [unused_theta, history] = train_adam(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options);
            end

            if history.diverged
                score_matrix(rate_index, seed_index)    = Inf;
                diverged_matrix(rate_index, seed_index) = true;
            else
                score_matrix(rate_index, seed_index) = history.best_validation_loss;
            end

            total_seconds = total_seconds + history.total_training_seconds;
        end
    end
end


function [chosen_learning_rate, chosen_index, median_of_each_rate, chosen_on_grid_edge] = select_learning_rate(learning_rate_grid, score_matrix)
    % Choose the learning rate with the lowest median validation score.
    %
    % Diverged runs have score Inf, so a rate that diverges in most seeds
    % gets an Inf median and is never chosen. If two rates tie, the first
    % one in the grid is kept (the smaller one for an increasing grid).
    %
    % Parameters
    % ----------
    % learning_rate_grid : double vector
    % score_matrix : (grid values x seeds) double, see run_learning_rate_grid
    %
    % Returns
    % -------
    % chosen_learning_rate : double
    % chosen_index : integer, position of the chosen value in the grid
    % median_of_each_rate : (grid values x 1) double
    % chosen_on_grid_edge : logical, true if the best value is the first
    %                       or the last of the grid (the optimum may lie
    %                       outside the grid)

    median_of_each_rate = median(score_matrix, 2);

    [lowest_median, chosen_index] = min(median_of_each_rate);

    if isinf(lowest_median)
        fprintf('      WARNING: every learning rate diverged in most seeds.\n');
    end

    chosen_learning_rate = learning_rate_grid(chosen_index);
    chosen_on_grid_edge  = (chosen_index == 1) || (chosen_index == numel(learning_rate_grid));
end


function print_learning_rate_tuning_table(tuning_results)
    % Print the tuning results of one dataset, one block per combination.
    %
    % Parameters
    % ----------
    % tuning_results : struct, output of tune_learning_rates

    print_section_title(sprintf('%s - Step 6: validation MSE (scaled) for each learning rate', tuning_results.dataset_name));

    number_of_architectures = numel(tuning_results.architecture_names);
    number_of_algorithms    = numel(tuning_results.algorithm_names);
    number_of_seeds         = numel(tuning_results.tuning_seeds);

    for architecture_index = 1:number_of_architectures
        for algorithm_index = 1:number_of_algorithms

            rates_in_grid   = tuning_results.learning_rate_grid{algorithm_index};
            score_matrix    = tuning_results.validation_scores{architecture_index, algorithm_index};
            diverged_matrix = tuning_results.diverged_runs{architecture_index, algorithm_index};
            median_scores   = tuning_results.median_scores{architecture_index, algorithm_index};
            chosen_rate     = tuning_results.chosen_learning_rate(architecture_index, algorithm_index);

            fprintf('\n%s | %s\n', tuning_results.architecture_names{architecture_index}, tuning_results.algorithm_names{algorithm_index});
            fprintf('%14s %14s %14s %14s %10s\n', 'Learning rate', 'Median', 'Best', 'Worst', 'Diverged');

            for rate_index = 1:numel(rates_in_grid)

                if rates_in_grid(rate_index) == chosen_rate
                    marker = '  <-- chosen';
                else
                    marker = '';
                end

                fprintf('%14.4g %14.4f %14.4f %14.4f %7d/%d%s\n', rates_in_grid(rate_index), median_scores(rate_index), ...
                        min(score_matrix(rate_index, :)), max(score_matrix(rate_index, :)), ...
                        sum(diverged_matrix(rate_index, :)), number_of_seeds, marker);
            end
        end
    end

    % ---- Summary of the chosen values ------------------------------------
    print_section_title(sprintf('%s - Step 6: chosen learning rates', tuning_results.dataset_name));
    fprintf('%-14s %16s %16s\n', 'Architecture', tuning_results.algorithm_names{1}, tuning_results.algorithm_names{2});

    for architecture_index = 1:number_of_architectures
        fprintf('%-14s %16.4g %16.4g\n', tuning_results.architecture_names{architecture_index}, ...
                tuning_results.chosen_learning_rate(architecture_index, 1), ...
                tuning_results.chosen_learning_rate(architecture_index, 2));
    end
end


function figure_handle = plot_learning_rate_tuning(tuning_results)
    % Validation MSE against learning rate, one panel per architecture.
    %
    % Each dot is one repetition, the line is the median over the seeds
    % and the star marks the chosen learning rate. Diverged runs (Inf)
    % are not drawn; their number is in the printed table.
    %
    % Parameters
    % ----------
    % tuning_results : struct, output of tune_learning_rates
    %
    % Returns
    % -------
    % figure_handle : handle of the new figure

    algorithm_colors = {[0 0.45 0.74], [0.85 0.33 0.10]};   % blue = GD, orange = Adam

    number_of_architectures = numel(tuning_results.architecture_names);
    number_of_algorithms    = numel(tuning_results.algorithm_names);
    number_of_seeds         = numel(tuning_results.tuning_seeds);

    figure_handle = figure('Position', [100 100 1300 420]);

    for architecture_index = 1:number_of_architectures

        subplot(1, number_of_architectures, architecture_index);
        hold on;

        legend_handles = zeros(1, number_of_algorithms);

        for algorithm_index = 1:number_of_algorithms

            rates_in_grid = tuning_results.learning_rate_grid{algorithm_index};
            score_matrix  = tuning_results.validation_scores{architecture_index, algorithm_index};
            median_scores = tuning_results.median_scores{architecture_index, algorithm_index};
            chosen_rate   = tuning_results.chosen_learning_rate(architecture_index, algorithm_index);
            color         = algorithm_colors{algorithm_index};

            % Inf (diverged) is replaced by NaN so that it is not drawn.
            score_matrix(isinf(score_matrix))   = NaN;
            median_scores(isinf(median_scores)) = NaN;

            % ---- One dot per repetition ---------------------------------
            for seed_index = 1:number_of_seeds
                plot(rates_in_grid, score_matrix(:, seed_index)', '.', 'Color', color, 'MarkerSize', 8);
            end

            % ---- Median line --------------------------------------------
            legend_handles(algorithm_index) = plot(rates_in_grid, median_scores', '-o', 'Color', color, 'LineWidth', 1.5, 'MarkerSize', 5);

            % ---- Chosen value -------------------------------------------
            chosen_index = find(rates_in_grid == chosen_rate, 1);

            if ~isnan(median_scores(chosen_index))
                plot(chosen_rate, median_scores(chosen_index), 'p', 'Color', color, 'MarkerFaceColor', color, 'MarkerSize', 14);
            end
        end

        set(gca, 'XScale', 'log', 'YScale', 'log');
        grid on;
        box on;

        title(tuning_results.architecture_names{architecture_index});
        xlabel('Learning rate');

        if architecture_index == 1
            ylabel('Best validation MSE (scaled units)');
        end

        legend(legend_handles, tuning_results.algorithm_names{:}, 'Location', 'Best');
        hold off;
    end
end


function print_bfgs_settings(options)
    % Print the BFGS settings (BFGS has no learning rate to tune).
    %
    % Parameters
    % ----------
    % options : struct, see default_training_options

    print_section_title('Step 6: BFGS settings (no learning rate)');

    fprintf('Line search                 : backtracking with the Armijo condition\n');
    fprintf('Initial step length         : 1\n');
    fprintf('Armijo constant c1          : %g\n', options.bfgs_armijo_constant);
    fprintf('Backtracking factor         : %g\n', options.bfgs_backtracking_factor);
    fprintf('Max trials per line search  : %d\n', options.bfgs_maximum_line_search_evaluations);
    fprintf('Curvature threshold (s''y)   : %g (update skipped below it)\n', options.bfgs_curvature_threshold);
    fprintf('Gradient tolerance          : max |g| < %g\n', options.bfgs_gradient_tolerance);
    fprintf('Budget                      : %d epoch-equivalents (line-search evaluations included)\n', options.budget_epoch_equivalents);
end

%% ========================================================================
%  STEP 7 - FINAL EXPERIMENTS: UTILS
%  3 architectures x 3 algorithms x 30 runs per dataset, test metrics.
%  ========================================================================

function results = run_final_experiments(dataset_name, target_unit, partition, tuned_learning_rates, final_seeds, options, checkpoint_file)
    % Run the 9 combinations of one dataset, 30 times each (Step 7).
    %
    % For a given architecture, run k uses init_params(layers, seed_k) for
    % the three algorithms, so the comparison is paired. seed_k is also the
    % seed of the mini-batch order. Every algorithm has the same budget of
    % epoch-equivalents and returns the parameters with the lowest
    % validation loss; only those parameters are evaluated on the test set,
    % once, in the original unit of the target.
    %
    % Checkpoint
    % ----------
    % The results are saved to checkpoint_file after every combination. If
    % the file already exists with the same seeds, budget and learning
    % rates, the finished combinations are loaded and skipped, so a long
    % run interrupted halfway continues where it stopped.
    %
    % Parameters
    % ----------
    % dataset_name : char
    %     Used in the printed titles.
    % target_unit : char
    %     Unit of the target, e.g. 'MPa' or 'degC'.
    % partition : struct
    %     Output of partition_concrete_dataset or partition_melbourne_dataset.
    % tuned_learning_rates : 3 x 2 double
    %     Output of Step 6 (row = architecture, column 1 = GD, column 2 = Adam).
    % final_seeds : integer vector
    %     Seeds of the runs, 1:30 in the assignment.
    % options : struct
    %     See default_training_options (budget_epoch_equivalents, batch
    %     size, Adam constants, BFGS settings).
    % checkpoint_file : char
    %     Path of the .mat file where the results are saved.
    %
    % Returns
    % -------
    % results : struct
    %     dataset_name, target_unit, seeds, budget_epoch_equivalents
    %     architecture_names, layers, number_of_parameters (1 x 3)
    %     algorithm_names : {'Mini-batch GD', 'BFGS', 'Adam'}
    %     algorithm_codes : {'gd', 'bfgs', 'adam'}
    %     learning_rate   : 3 x 3 double (NaN for BFGS)
    %     test_target     : test targets in the original unit
    %     combinations    : 3 x 3 cell, each the output of run_one_combination
    %     completed       : 3 x 3 logical

    number_of_inputs = size(partition.training.inputs_scaled, 2);
    architectures = build_architectures(number_of_inputs);

    algorithm_names = {'Mini-batch GD', 'BFGS', 'Adam'};
    algorithm_codes = {'gd', 'bfgs', 'adam'};

    number_of_architectures = numel(architectures.layers);
    number_of_algorithms    = numel(algorithm_codes);

    % ---- Learning rates of Step 6 in the algorithm order GD, BFGS, Adam ---
    learning_rate = NaN(number_of_architectures, number_of_algorithms);
    learning_rate(:, 1) = tuned_learning_rates(:, 1);
    learning_rate(:, 3) = tuned_learning_rates(:, 2);

    % ---- Description of the experiment -----------------------------------
    results.dataset_name             = dataset_name;
    results.target_unit              = target_unit;
    results.seeds                    = final_seeds(:)';
    results.budget_epoch_equivalents = options.budget_epoch_equivalents;
    results.architecture_names       = architectures.names;
    results.layers                   = architectures.layers;
    results.algorithm_names          = algorithm_names;
    results.algorithm_codes          = algorithm_codes;
    results.learning_rate            = learning_rate;
    results.test_target              = partition.test.target;

    results.number_of_parameters = zeros(1, number_of_architectures);

    for architecture_index = 1:number_of_architectures
        results.number_of_parameters(architecture_index) = count_network_parameters(architectures.layers{architecture_index});
    end

    results.combinations = cell(number_of_architectures, number_of_algorithms);
    results.completed    = false(number_of_architectures, number_of_algorithms);

    print_section_title(sprintf('%s - Step 7: final experiments (%d runs, %d epoch-equivalents)', ...
                                dataset_name, numel(final_seeds), options.budget_epoch_equivalents));

    % ---- Resume from the checkpoint if it has the same settings ----------
    if exist(checkpoint_file, 'file')
        loaded_file = load(checkpoint_file);
        previous_results = loaded_file.results;

        same_settings = isequal(previous_results.seeds, results.seeds) && ...
                        previous_results.budget_epoch_equivalents == results.budget_epoch_equivalents && ...
                        isequal(previous_results.learning_rate(:, [1 3]), results.learning_rate(:, [1 3]));

        if same_settings
            results.combinations = previous_results.combinations;
            results.completed    = previous_results.completed;
            fprintf('Checkpoint found: %d of %d combinations already done, they are skipped.\n', ...
                    sum(results.completed(:)), numel(results.completed));
        else
            fprintf('Checkpoint found with different settings: starting from scratch (it will be overwritten).\n');
        end
    end

    % ---- The 9 combinations ----------------------------------------------
    for architecture_index = 1:number_of_architectures

        layers = architectures.layers{architecture_index};

        for algorithm_index = 1:number_of_algorithms

            if results.completed(architecture_index, algorithm_index)
                fprintf('  %-12s | %-14s : already done (checkpoint)\n', ...
                        architectures.names{architecture_index}, algorithm_names{algorithm_index});
                continue;
            end

            if isnan(learning_rate(architecture_index, algorithm_index))
                rate_text = 'no learning rate';
            else
                rate_text = sprintf('learning rate %.4g', learning_rate(architecture_index, algorithm_index));
            end

            fprintf('  %-12s | %-14s (%s), %d parameters\n', architectures.names{architecture_index}, ...
                    algorithm_names{algorithm_index}, rate_text, results.number_of_parameters(architecture_index));

            combination_start = clock;

            combination = run_one_combination(layers, partition, algorithm_codes{algorithm_index}, ...
                                              learning_rate(architecture_index, algorithm_index), ...
                                              final_seeds, options);

            results.combinations{architecture_index, algorithm_index} = combination;
            results.completed(architecture_index, algorithm_index)    = true;

            save(checkpoint_file, 'results', '-mat');

            fprintf('      done in %.1f min: test RMSE %.3f +- %.3f %s, diverged runs %d/%d (saved)\n', ...
                    etime(clock, combination_start) / 60, mean(combination.test_rmse), std(combination.test_rmse), ...
                    target_unit, sum(combination.diverged), numel(final_seeds));
        end
    end

    print_final_experiment_summary(results);
end


function combination = run_one_combination(layers, partition, algorithm_code, learning_rate, final_seeds, options)
    % Train one architecture with one algorithm once per seed.
    %
    % Parameters
    % ----------
    % layers : integer vector, [d, h1, ..., 1]
    % partition : struct, output of a partition_* function
    % algorithm_code : char, 'gd', 'bfgs' or 'adam'
    % learning_rate : double, NaN for BFGS
    % final_seeds : integer vector
    % options : struct, see default_training_options
    %
    % Returns
    % -------
    % combination : struct (R = number of seeds, B = budget)
    %     algorithm_code, learning_rate, seeds
    %     training_loss, validation_loss : B x R, per epoch-equivalent (scaled units)
    %     elapsed_seconds                : B x R, cumulative training time
    %     initial_training_loss, initial_validation_loss : 1 x R
    %     best_validation_loss, best_epoch_equivalent    : 1 x R
    %     gradient_evaluations           : 1 x R, full-dataset gradient evaluations
    %     training_seconds               : 1 x R
    %     diverged                       : 1 x R logical
    %     test_mse, test_rmse, test_mae, test_r2 : 1 x R, original unit
    %     test_predictions               : (test samples) x R, original unit

    training_inputs   = partition.training.inputs_scaled;
    training_target   = partition.training.target_scaled;
    validation_inputs = partition.validation.inputs_scaled;
    validation_target = partition.validation.target_scaled;

    budget = options.budget_epoch_equivalents;
    number_of_runs = numel(final_seeds);
    number_of_test_samples = numel(partition.test.target);

    if ~isnan(learning_rate)
        options.learning_rate = learning_rate;
    end

    % ---- Allocate ----------------------------------------------------------
    combination.algorithm_code = algorithm_code;
    combination.learning_rate  = learning_rate;
    combination.seeds          = final_seeds(:)';

    combination.training_loss   = NaN(budget, number_of_runs);
    combination.validation_loss = NaN(budget, number_of_runs);
    combination.elapsed_seconds = NaN(budget, number_of_runs);

    combination.initial_training_loss   = NaN(1, number_of_runs);
    combination.initial_validation_loss = NaN(1, number_of_runs);
    combination.best_validation_loss    = NaN(1, number_of_runs);
    combination.best_epoch_equivalent   = zeros(1, number_of_runs);
    combination.gradient_evaluations    = zeros(1, number_of_runs);
    combination.training_seconds        = zeros(1, number_of_runs);
    combination.diverged                = false(1, number_of_runs);

    combination.test_mse  = NaN(1, number_of_runs);
    combination.test_rmse = NaN(1, number_of_runs);
    combination.test_mae  = NaN(1, number_of_runs);
    combination.test_r2   = NaN(1, number_of_runs);
    combination.test_predictions = NaN(number_of_test_samples, number_of_runs);

    % ---- The runs ------------------------------------------------------------
    for run_index = 1:number_of_runs

        seed = final_seeds(run_index);

        % Same seed -> same initial weights for GD, BFGS and Adam (paired).
        theta0 = init_params(layers, seed);
        options.shuffle_seed = seed;

        if strcmp(algorithm_code, 'gd')
            [best_theta, history] = train_minibatch_gd(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options);
        elseif strcmp(algorithm_code, 'bfgs')
            [best_theta, history] = train_bfgs(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options);
        else
            [best_theta, history] = train_adam(theta0, layers, training_inputs, training_target, validation_inputs, validation_target, options);
        end

        % ---- Training history ------------------------------------------------
        combination.training_loss(:, run_index)   = history.training_loss;
        combination.validation_loss(:, run_index) = history.validation_loss;
        combination.elapsed_seconds(:, run_index) = history.elapsed_seconds;

        combination.initial_training_loss(run_index)   = history.initial_training_loss;
        combination.initial_validation_loss(run_index) = history.initial_validation_loss;
        combination.best_validation_loss(run_index)    = history.best_validation_loss;
        combination.best_epoch_equivalent(run_index)   = history.best_epoch_equivalent;
        combination.gradient_evaluations(run_index)    = history.gradient_evaluations;
        combination.training_seconds(run_index)        = history.total_training_seconds;
        combination.diverged(run_index)                = history.diverged;

        % ---- Test set: used once, with the parameters already selected -------
        [test_metrics, test_predictions] = evaluate_on_test_set(best_theta, layers, partition);

        combination.test_mse(run_index)  = test_metrics.mse;
        combination.test_rmse(run_index) = test_metrics.rmse;
        combination.test_mae(run_index)  = test_metrics.mae;
        combination.test_r2(run_index)   = test_metrics.r2;
        combination.test_predictions(:, run_index) = test_predictions;

        if history.diverged
            diverged_text = '  DIVERGED';
        else
            diverged_text = '';
        end

        fprintf('      run %2d (seed %2d): best validation %.4f at %4d, test RMSE %.3f, R2 %.3f, %.1f s%s\n', ...
                run_index, seed, history.best_validation_loss, history.best_epoch_equivalent, ...
                test_metrics.rmse, test_metrics.r2, history.total_training_seconds, diverged_text);
    end
end


function [test_metrics, test_predictions] = evaluate_on_test_set(theta, layers, partition)
    % Test metrics in the ORIGINAL unit of the target (MPa or degC).
    %
    % The network predicts in scaled units; the prediction is converted
    % back with the target scaler fitted on the training set.
    %
    % Parameters
    % ----------
    % theta : P x 1 double, selected parameters
    % layers : integer vector
    % partition : struct, output of a partition_* function
    %
    % Returns
    % -------
    % test_metrics : struct, output of compute_regression_metrics
    %                (mse, rmse, mae, r2)
    % test_predictions : (test samples) x 1 double, original unit

    scaled_predictions = predict_network(theta, layers, partition.test.inputs_scaled);

    test_predictions = invert_standardization(scaled_predictions, partition.target_scaler);
    test_predictions = test_predictions(:);

    test_metrics = compute_regression_metrics(partition.test.target, test_predictions);
end


function print_final_experiment_summary(results)
    % Quick summary of Step 7 (the full tables and figures are in Step 8).
    %
    % Parameters
    % ----------
    % results : struct, output of run_final_experiments

    print_section_title(sprintf('%s - Step 7: summary over %d runs (test set, %s)', ...
                                results.dataset_name, numel(results.seeds), results.target_unit));

    fprintf('%-12s %-14s %10s %20s %18s %12s %10s %12s %12s\n', 'Architecture', 'Algorithm', 'Rate', ...
            'Test RMSE mean+-std', 'Test R2 mean+-std', 'Median RMSE', 'Diverged', 'Mean grad.ev', 'Mean seconds');

    for architecture_index = 1:numel(results.architecture_names)
        for algorithm_index = 1:numel(results.algorithm_names)

            if ~results.completed(architecture_index, algorithm_index)
                fprintf('%-12s %-14s   (not run)\n', results.architecture_names{architecture_index}, ...
                        results.algorithm_names{algorithm_index});
                continue;
            end

            combination = results.combinations{architecture_index, algorithm_index};

            if isnan(combination.learning_rate)
                rate_text = '-';
            else
                rate_text = sprintf('%.4g', combination.learning_rate);
            end

            fprintf('%-12s %-14s %10s %11.3f +- %5.3f %10.3f +- %5.3f %12.3f %7d/%-2d %12.0f %12.1f\n', ...
                    results.architecture_names{architecture_index}, results.algorithm_names{algorithm_index}, rate_text, ...
                    mean(combination.test_rmse), std(combination.test_rmse), ...
                    mean(combination.test_r2), std(combination.test_r2), ...
                    median(combination.test_rmse), sum(combination.diverged), numel(combination.diverged), ...
                    mean(combination.gradient_evaluations), mean(combination.training_seconds));
        end
    end
end

%% ========================================================================
%  STEP 8 - REQUIRED RESULTS: UTILS
%  Summary tables, training-progress curves, box plots, prediction plots
%  and reference models. Uses the saved results of Step 7 (no training).
%  ========================================================================

function reference_models = compute_reference_models(dataset_name, partition, number_of_lag_columns)
    % Fit the reference models on the training set and evaluate them on test.
    %
    % Concrete (number_of_lag_columns = 0):
    %     linear regression on the same 9 input features as the networks.
    % Melbourne (number_of_lag_columns = p):
    %     persistence, T(t+1) = T(t) (no fitting);
    %     linear autoregressive model on the same window of p lags;
    %     linear regression on the same inputs as the networks (lags + sin, cos).
    %
    % The linear models have no hyperparameters, so they are fitted on the
    % training set only (unscaled inputs, target in the original unit).
    %
    % Parameters
    % ----------
    % dataset_name : char
    %     Used in the printed title.
    % partition : struct
    %     Output of partition_concrete_dataset or partition_melbourne_dataset.
    % number_of_lag_columns : integer
    %     0 for Concrete, p for Melbourne (lag columns come first, the last
    %     lag column is T(t)).
    %
    % Returns
    % -------
    % reference_models : 1 x M struct array
    %     name             : char
    %     test_metrics     : output of compute_regression_metrics (original unit)
    %     test_predictions : (test samples) x 1 double, original unit

    training_inputs = partition.training.inputs;
    training_target = partition.training.target;
    test_inputs     = partition.test.inputs;
    test_target     = partition.test.target;

    model_names = {};
    model_predictions = {};

    if number_of_lag_columns == 0

        coefficients = fit_linear_least_squares(training_inputs, training_target);

        model_names{end + 1}       = 'Linear regression';
        model_predictions{end + 1} = predict_linear_model(test_inputs, coefficients);

    else

        lag_columns = 1:number_of_lag_columns;

        % ---- Persistence: tomorrow = today --------------------------------
        model_names{end + 1}       = 'Persistence';
        model_predictions{end + 1} = test_inputs(:, number_of_lag_columns);

        % ---- Linear autoregressive model on the same window ---------------
        autoregressive_coefficients = fit_linear_least_squares(training_inputs(:, lag_columns), training_target);

        model_names{end + 1}       = sprintf('Linear AR (%d lags)', number_of_lag_columns);
        model_predictions{end + 1} = predict_linear_model(test_inputs(:, lag_columns), autoregressive_coefficients);

        % ---- Linear regression on the same inputs as the networks ---------
        full_coefficients = fit_linear_least_squares(training_inputs, training_target);

        model_names{end + 1}       = 'Linear (lags + calendar)';
        model_predictions{end + 1} = predict_linear_model(test_inputs, full_coefficients);
    end

    % ---- Metrics in the original unit ------------------------------------
    print_section_title(sprintf('%s - Step 8: reference models (test set)', dataset_name));
    fprintf('%-28s %10s %10s %10s\n', 'Model', 'RMSE', 'MAE', 'R2');

    for model_index = 1:numel(model_names)

        predictions = model_predictions{model_index}(:);
        metrics = compute_regression_metrics(test_target, predictions);

        reference_models(model_index).name             = model_names{model_index};
        reference_models(model_index).test_metrics     = metrics;
        reference_models(model_index).test_predictions = predictions;

        fprintf('%-28s %10.3f %10.3f %10.3f\n', model_names{model_index}, metrics.rmse, metrics.mae, metrics.r2);
    end
end


function step8_summary = produce_step8_results(results, reference_models, test_dates, figures_folder, output_folder, close_figures_after_saving)
    % Produce every Step 8 table and figure for one dataset.
    %
    % Parameters
    % ----------
    % results : struct
    %     Output of run_final_experiments (Step 7).
    % reference_models : struct array
    %     Output of compute_reference_models.
    % test_dates : double column or []
    %     Serial dates of the test targets (Melbourne), [] for Concrete.
    % figures_folder : char
    %     Folder where the PNG files are saved.
    % output_folder : char
    %     Folder where the summary table (.txt) is saved.
    % close_figures_after_saving : logical
    %     Close each figure after saving it (keeps MATLAB light).
    %
    % Returns
    % -------
    % step8_summary : struct
    %     representative_run : output of select_representative_run
    %     summary_file       : path of the saved summary table

    dataset_token = lower(results.dataset_name);

    ensure_folder_exists(figures_folder);
    ensure_folder_exists(output_folder);

    % ---- 1. Summary table (screen and text file) -------------------------
    print_step8_summary_table(results, reference_models, 1);

    summary_file = fullfile(output_folder, sprintf('step8_%s_summary_table.txt', dataset_token));
    file_id = fopen(summary_file, 'w');
    print_step8_summary_table(results, reference_models, file_id);
    fclose(file_id);

    fprintf('  saved: %s\n', summary_file);

    % ---- 2. Training progress: per epoch-equivalent and per second -------
    print_section_title(sprintf('%s - Step 8: figures', results.dataset_name));

    for architecture_index = 1:numel(results.architecture_names)

        architecture_token = lower(strrep(results.architecture_names{architecture_index}, ' ', '_'));

        figure_handle = plot_training_progress(results, architecture_index, 'epochs');
        save_figure(figure_handle, figures_folder, sprintf('step8_%s_%s_training_progress', dataset_token, architecture_token));
        close_if_requested(figure_handle, close_figures_after_saving);

        figure_handle = plot_training_progress(results, architecture_index, 'seconds');
        save_figure(figure_handle, figures_folder, sprintf('step8_%s_%s_training_time', dataset_token, architecture_token));
        close_if_requested(figure_handle, close_figures_after_saving);
    end

    % ---- 3. Box plots of the test RMSE -----------------------------------
    figure_handle = plot_test_error_box_plots(results, reference_models);
    save_figure(figure_handle, figures_folder, sprintf('step8_%s_test_rmse_box_plots', dataset_token));
    close_if_requested(figure_handle, close_figures_after_saving);

    % ---- 4. Prediction plots and residual histograms ---------------------
    representative_run = select_representative_run(results);

    if isempty(test_dates)
        figure_handle = plot_concrete_predictions(results, representative_run, reference_models);
    else
        figure_handle = plot_melbourne_predictions(results, representative_run, reference_models, test_dates, 60);
    end

    save_figure(figure_handle, figures_folder, sprintf('step8_%s_predictions', dataset_token));
    close_if_requested(figure_handle, close_figures_after_saving);

    step8_summary.representative_run = representative_run;
    step8_summary.summary_file       = summary_file;
end


function close_if_requested(figure_handle, close_figure)
    % Close a figure when close_figure is true.
    %
    % Parameters
    % ----------
    % figure_handle : handle
    % close_figure : logical

    if close_figure
        close(figure_handle);
    end
end


function print_step8_summary_table(results, reference_models, file_id)
    % 3 x 3 summary of the test metrics, plus the reference models.
    %
    % Parameters
    % ----------
    % results : struct, output of run_final_experiments
    % reference_models : struct array, output of compute_reference_models
    % file_id : integer, 1 for the screen or the id returned by fopen

    unit = results.target_unit;

    fprintf(file_id, '\n%s - Step 8: test-set summary over %d runs (RMSE in %s)\n', results.dataset_name, numel(results.seeds), unit);
    fprintf(file_id, '%s\n', repmat('=', 1, 70));

    fprintf(file_id, '%-12s %-14s %10s %18s %18s %8s %8s %8s %11s %9s\n', 'Architecture', 'Algorithm', 'Parameters', ...
            'RMSE mean+-std', 'R2 mean+-std', 'Median', 'Best', 'Worst', 'Best epoch', 'Seconds');

    for architecture_index = 1:numel(results.architecture_names)
        for algorithm_index = 1:numel(results.algorithm_names)

            combination = results.combinations{architecture_index, algorithm_index};
            test_rmse = combination.test_rmse;

            fprintf(file_id, '%-12s %-14s %10d %9.3f +- %5.3f %9.3f +- %5.3f %8.3f %8.3f %8.3f %11.0f %9.1f\n', ...
                    results.architecture_names{architecture_index}, results.algorithm_names{algorithm_index}, ...
                    results.number_of_parameters(architecture_index), ...
                    mean(test_rmse), std(test_rmse), mean(combination.test_r2), std(combination.test_r2), ...
                    median(test_rmse), min(test_rmse), max(test_rmse), ...
                    median(combination.best_epoch_equivalent), mean(combination.training_seconds));
        end
    end

    fprintf(file_id, '\nReference models (single fit, no repetitions):\n');
    fprintf(file_id, '%-28s %10s %10s %10s\n', 'Model', 'RMSE', 'MAE', 'R2');

    for model_index = 1:numel(reference_models)
        metrics = reference_models(model_index).test_metrics;
        fprintf(file_id, '%-28s %10.3f %10.3f %10.3f\n', reference_models(model_index).name, metrics.rmse, metrics.mae, metrics.r2);
    end

    fprintf(file_id, '\nBest epoch = median epoch-equivalent of the selected (lowest validation loss) parameters.\n');
    fprintf(file_id, 'Seconds = mean training time per run.\n');
end


function [lower_quartile, median_curve, upper_quartile] = compute_curve_quantiles(curve_matrix)
    % First quartile, median and third quartile of the runs at each point.
    %
    % NaN values (e.g. after a divergence) are ignored at each point.
    %
    % Parameters
    % ----------
    % curve_matrix : B x R double, one column per run
    %
    % Returns
    % -------
    % lower_quartile, median_curve, upper_quartile : B x 1 double

    number_of_points = size(curve_matrix, 1);

    lower_quartile = NaN(number_of_points, 1);
    median_curve   = NaN(number_of_points, 1);
    upper_quartile = NaN(number_of_points, 1);

    for point_index = 1:number_of_points

        values_at_point = curve_matrix(point_index, :);
        values_at_point = sort(values_at_point(~isnan(values_at_point)));

        if ~isempty(values_at_point)
            lower_quartile(point_index) = compute_percentile(values_at_point, 0.25);
            median_curve(point_index)   = compute_percentile(values_at_point, 0.50);
            upper_quartile(point_index) = compute_percentile(values_at_point, 0.75);
        end
    end
end


function figure_handle = plot_training_progress(results, architecture_index, horizontal_axis)
    % Training and validation loss of the three algorithms for one architecture.
    %
    % Solid line = median of the runs, dotted lines = first and third
    % quartiles (interquartile range). Vertical axis logarithmic.
    %
    % Parameters
    % ----------
    % results : struct, output of run_final_experiments
    % architecture_index : integer, 1 (A), 2 (B) or 3 (C)
    % horizontal_axis : char
    %     'epochs'  : epoch-equivalents (equal computational cost)
    %     'seconds' : median cumulative training time over the runs
    %
    % Returns
    % -------
    % figure_handle : handle of the new figure

    algorithm_colors = {[0 0.45 0.74], [0.47 0.67 0.19], [0.85 0.33 0.10]};   % GD blue, BFGS green, Adam orange

    loss_fields = {'training_loss', 'validation_loss'};
    loss_titles = {'Training loss', 'Validation loss'};

    architecture_name = results.architecture_names{architecture_index};
    number_of_algorithms = numel(results.algorithm_names);

    figure_handle = figure('Position', [100 100 1250 460]);

    for panel_index = 1:2

        subplot(1, 2, panel_index);
        hold on;

        legend_handles = zeros(1, number_of_algorithms);

        for algorithm_index = 1:number_of_algorithms

            combination = results.combinations{architecture_index, algorithm_index};
            color = algorithm_colors{algorithm_index};

            [lower_quartile, median_curve, upper_quartile] = compute_curve_quantiles(combination.(loss_fields{panel_index}));

            if strcmp(horizontal_axis, 'seconds')
                [unused_lower, horizontal_values, unused_upper] = compute_curve_quantiles(combination.elapsed_seconds);
            else
                horizontal_values = (1:numel(median_curve))';
            end

            legend_handles(algorithm_index) = plot(horizontal_values, median_curve, '-', 'Color', color, 'LineWidth', 1.5);
            plot(horizontal_values, lower_quartile, ':', 'Color', color, 'LineWidth', 1);
            plot(horizontal_values, upper_quartile, ':', 'Color', color, 'LineWidth', 1);
        end

        set(gca, 'YScale', 'log');
        grid on;
        box on;

        title(sprintf('%s, %s: %s (median and IQR of %d runs)', results.dataset_name, architecture_name, ...
                      loss_titles{panel_index}, numel(results.seeds)));

        if strcmp(horizontal_axis, 'seconds')
            xlabel('Training time (s, median over the runs)');
        else
            xlabel('Epoch-equivalents');
        end

        ylabel('MSE (scaled units)');
        legend(legend_handles, results.algorithm_names{:}, 'Location', 'NorthEast');
        hold off;
    end
end


function figure_handle = plot_test_error_box_plots(results, reference_models)
    % Box plots of the test RMSE of the 9 combinations (draw_box_plot, no toolbox).
    %
    % The reference models are drawn as horizontal lines.
    %
    % Parameters
    % ----------
    % results : struct, output of run_final_experiments
    % reference_models : struct array, output of compute_reference_models
    %
    % Returns
    % -------
    % figure_handle : handle of the new figure

    short_algorithm_names = {'GD', 'BFGS', 'Adam'};

    number_of_architectures = numel(results.architecture_names);
    number_of_algorithms    = numel(results.algorithm_names);
    number_of_groups        = number_of_architectures * number_of_algorithms;

    group_values = cell(1, number_of_groups);
    group_labels = cell(1, number_of_groups);
    group_index  = 0;

    for architecture_index = 1:number_of_architectures
        for algorithm_index = 1:number_of_algorithms

            group_index = group_index + 1;
            combination = results.combinations{architecture_index, algorithm_index};

            group_values{group_index} = combination.test_rmse;
            group_labels{group_index} = sprintf('%s %s', results.architecture_names{architecture_index}(1), ...
                                                short_algorithm_names{algorithm_index});
        end
    end

    figure_handle = figure('Position', [100 100 1050 460]);

    draw_box_plot(group_values, group_labels);

    % ---- Reference models as horizontal lines ----------------------------
    hold on;

    line_styles = {'--', '-.', ':'};
    number_of_models = numel(reference_models);
    legend_handles = zeros(1, number_of_models);
    legend_labels  = cell(1, number_of_models);

    for model_index = 1:number_of_models
        model_rmse = reference_models(model_index).test_metrics.rmse;
        style = line_styles{mod(model_index - 1, numel(line_styles)) + 1};

        legend_handles(model_index) = plot([0.5, number_of_groups + 0.5], [model_rmse, model_rmse], style, ...
                                           'Color', [0.3 0.3 0.3], 'LineWidth', 1.2);
        legend_labels{model_index} = sprintf('%s (%.3f %s)', reference_models(model_index).name, model_rmse, results.target_unit);
    end

    hold off;
    grid on;
    box on;

    title(sprintf('%s: test RMSE of the 9 combinations (%d runs each)', results.dataset_name, numel(results.seeds)));
    xlabel('Architecture and algorithm');
    ylabel(sprintf('Test RMSE (%s)', results.target_unit));
    legend(legend_handles, legend_labels{:}, 'Location', 'Best');
end


function representative_run = select_representative_run(results)
    % Run used for the prediction plots.
    %
    % The combination with the lowest mean test RMSE is chosen, and inside
    % it the run whose test RMSE is closest to the median (a typical run,
    % not the luckiest one).
    %
    % Parameters
    % ----------
    % results : struct, output of run_final_experiments
    %
    % Returns
    % -------
    % representative_run : struct
    %     architecture_index, algorithm_index, run_index, seed, test_rmse, label

    number_of_architectures = numel(results.architecture_names);
    number_of_algorithms    = numel(results.algorithm_names);

    mean_rmse = Inf(number_of_architectures, number_of_algorithms);

    for architecture_index = 1:number_of_architectures
        for algorithm_index = 1:number_of_algorithms
            mean_rmse(architecture_index, algorithm_index) = mean(results.combinations{architecture_index, algorithm_index}.test_rmse);
        end
    end

    [unused_lowest, best_position] = min(mean_rmse(:));
    [best_architecture, best_algorithm] = ind2sub(size(mean_rmse), best_position);

    test_rmse = results.combinations{best_architecture, best_algorithm}.test_rmse;
    [unused_distance, run_index] = min(abs(test_rmse - median(test_rmse)));

    representative_run.architecture_index = best_architecture;
    representative_run.algorithm_index    = best_algorithm;
    representative_run.run_index          = run_index;
    representative_run.seed               = results.seeds(run_index);
    representative_run.test_rmse          = test_rmse(run_index);
    representative_run.label              = sprintf('%s, %s, seed %d', results.architecture_names{best_architecture}, ...
                                                    results.algorithm_names{best_algorithm}, results.seeds(run_index));

    fprintf('Representative run for the prediction plots: %s (test RMSE %.3f %s)\n', ...
            representative_run.label, representative_run.test_rmse, results.target_unit);
end


function figure_handle = plot_concrete_predictions(results, representative_run, reference_models)
    % Concrete: predicted against measured strength, and residual histograms.
    %
    % Parameters
    % ----------
    % results : struct, output of run_final_experiments
    % representative_run : struct, output of select_representative_run
    % reference_models : struct array (first model = linear regression)
    %
    % Returns
    % -------
    % figure_handle : handle of the new figure

    combination = results.combinations{representative_run.architecture_index, representative_run.algorithm_index};

    observed_values     = results.test_target(:);
    network_predictions = combination.test_predictions(:, representative_run.run_index);
    linear_predictions  = reference_models(1).test_predictions;

    network_color = [0 0.45 0.74];
    linear_color  = [0.6 0.6 0.6];

    figure_handle = figure('Position', [100 100 1250 500]);

    % ---- Predicted against measured --------------------------------------
    subplot(1, 2, 1);
    hold on;

    linear_handle  = plot(observed_values, linear_predictions, '.', 'Color', linear_color, 'MarkerSize', 10);
    network_handle = plot(observed_values, network_predictions, '.', 'Color', network_color, 'MarkerSize', 12);

    all_values  = [observed_values; network_predictions; linear_predictions];
    axis_limits = [min(all_values), max(all_values)];
    plot(axis_limits, axis_limits, 'k--');

    set(gca, 'XLim', axis_limits, 'YLim', axis_limits);
    grid on;
    box on;

    title(sprintf('Predicted vs measured (%s)', representative_run.label));
    xlabel('Measured strength (MPa)');
    ylabel('Predicted strength (MPa)');
    legend([network_handle, linear_handle], 'Neural network', reference_models(1).name, 'Location', 'NorthWest');
    hold off;

    % ---- Residual histograms ---------------------------------------------
    subplot(1, 2, 2);

    draw_residual_histograms({observed_values - network_predictions, observed_values - linear_predictions}, ...
                             {'Neural network', reference_models(1).name}, ...
                             {network_color, linear_color}, results.target_unit);

    title('Residuals on the test set');
end


function figure_handle = plot_melbourne_predictions(results, representative_run, reference_models, test_dates, zoom_days)
    % Melbourne: observed and predicted temperature over the test period,
    % a zoomed window, and residual histograms.
    %
    % Parameters
    % ----------
    % results : struct, output of run_final_experiments
    % representative_run : struct, output of select_representative_run
    % reference_models : struct array (1 = persistence, 2 = linear AR)
    % test_dates : double column, serial dates of the test targets
    % zoom_days : integer, length of the zoomed window (first days of test)
    %
    % Returns
    % -------
    % figure_handle : handle of the new figure

    combination = results.combinations{representative_run.architecture_index, representative_run.algorithm_index};

    observed_values         = results.test_target(:);
    network_predictions     = combination.test_predictions(:, representative_run.run_index);
    persistence_predictions = reference_models(1).test_predictions;
    autoregressive_predictions = reference_models(2).test_predictions;

    observed_color    = [0.45 0.45 0.45];
    network_color     = [0 0.45 0.74];
    persistence_color = [0.85 0.33 0.10];
    autoregressive_color = [0.47 0.67 0.19];

    figure_handle = figure('Position', [100 50 1250 950]);

    % ---- Whole test period -----------------------------------------------
    subplot(3, 1, 1);
    hold on;

    observed_handle = plot(test_dates, observed_values, '-', 'Color', observed_color, 'LineWidth', 0.5);
    network_handle  = plot(test_dates, network_predictions, '-', 'Color', network_color, 'LineWidth', 1);

    datetick('x', 28);   % numeric format 28 = 'mmmyyyy' (MATLAB 2007 accepts only predefined formats)
    grid on;
    box on;

    title(sprintf('Test period: observed and predicted T(t+1) (%s)', representative_run.label));
    ylabel('Minimum temperature (degC)');
    legend([observed_handle, network_handle], 'Observed', 'Neural network', 'Location', 'NorthEast');
    hold off;

    % ---- Zoomed window -----------------------------------------------------
    subplot(3, 1, 2);
    hold on;

    zoom_rows = 1:min(zoom_days, numel(observed_values));

    observed_handle    = plot(test_dates(zoom_rows), observed_values(zoom_rows), '-o', 'Color', observed_color, 'MarkerSize', 3);
    network_handle     = plot(test_dates(zoom_rows), network_predictions(zoom_rows), '-', 'Color', network_color, 'LineWidth', 1.5);
    persistence_handle = plot(test_dates(zoom_rows), persistence_predictions(zoom_rows), '--', 'Color', persistence_color, 'LineWidth', 1);

    datetick('x', 1);    % numeric format 1 = 'dd-mmm-yyyy'
    grid on;
    box on;

    title(sprintf('Zoom: first %d days of the test period', numel(zoom_rows)));
    ylabel('Minimum temperature (degC)');
    legend([observed_handle, network_handle, persistence_handle], 'Observed', 'Neural network', reference_models(1).name, 'Location', 'NorthEast');
    hold off;

    % ---- Residual histograms ---------------------------------------------
    subplot(3, 1, 3);

    draw_residual_histograms({observed_values - network_predictions, observed_values - persistence_predictions, observed_values - autoregressive_predictions}, ...
                             {'Neural network', reference_models(1).name, reference_models(2).name}, ...
                             {network_color, persistence_color, autoregressive_color}, results.target_unit);

    title('Residuals on the test set');
end


function draw_residual_histograms(residual_sets, set_labels, set_colors, unit)
    % Histograms of several residual sets with common bins, drawn as lines.
    %
    % Lines are used instead of bars so that the sets can be compared on
    % the same axes. The legend gives the mean and standard deviation.
    %
    % Parameters
    % ----------
    % residual_sets : 1 x S cell of double vectors (observed - predicted)
    % set_labels : 1 x S cell of char
    % set_colors : 1 x S cell of RGB vectors
    % unit : char, unit of the residuals

    number_of_bins = 30;
    number_of_sets = numel(residual_sets);

    all_residuals = [];

    for set_index = 1:number_of_sets
        all_residuals = [all_residuals; residual_sets{set_index}(:)];
    end

    bin_centers = linspace(min(all_residuals), max(all_residuals), number_of_bins);

    hold on;

    legend_handles = zeros(1, number_of_sets);
    legend_labels  = cell(1, number_of_sets);

    for set_index = 1:number_of_sets
        residuals_of_set = residual_sets{set_index}(:);
        bin_counts = hist(residuals_of_set, bin_centers);

        legend_handles(set_index) = plot(bin_centers, bin_counts, '-o', 'Color', set_colors{set_index}, ...
                                         'LineWidth', 1.2, 'MarkerSize', 3);
        legend_labels{set_index} = sprintf('%s (mean %.2f, std %.2f)', set_labels{set_index}, ...
                                           mean(residuals_of_set), std(residuals_of_set));
    end

    plot([0 0], get(gca, 'YLim'), 'k:');

    hold off;
    grid on;
    box on;

    xlabel(sprintf('Residual = observed - predicted (%s)', unit));
    ylabel('Number of test samples');
    legend(legend_handles, legend_labels{:}, 'Location', 'NorthEast');
end


