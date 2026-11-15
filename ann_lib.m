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
    % Step 1 - data acquisition and initial inspection
    %   Data loading     : load_concrete_dataset, load_melbourne_dataset
    %   Inspection       : inspect_concrete_dataset, inspect_melbourne_dataset
    %   Statistics       : compute_column_summary, compute_percentile
    %   Printing helpers : print_column_summary, print_section_title
    %
    % Step 2 - exploratory data analysis
    %   Entry points     : explore_concrete_dataset, explore_melbourne_dataset
    %   Statistics       : correlations, autocorrelation (ACF), partial
    %                      autocorrelation (PACF), annual harmonic fit,
    %                      daily climatology
    %   Plots            : plot_concrete_*, plot_melbourne_*, draw_box_plot

    L.load_concrete_dataset     = @load_concrete_dataset;
    L.load_melbourne_dataset    = @load_melbourne_dataset;

    L.inspect_concrete_dataset  = @inspect_concrete_dataset;
    L.inspect_melbourne_dataset = @inspect_melbourne_dataset;

    L.compute_column_summary    = @compute_column_summary;
    L.compute_percentile        = @compute_percentile;

    L.print_column_summary      = @print_column_summary;
    L.print_section_title       = @print_section_title;

    % ---- Step 2: exploratory data analysis -------------------------------
    L.explore_concrete_dataset   = @explore_concrete_dataset;
    L.explore_melbourne_dataset  = @explore_melbourne_dataset;

    L.compute_pearson_correlation          = @compute_pearson_correlation;
    L.compute_correlation_matrix           = @compute_correlation_matrix;
    L.compute_average_ranks                = @compute_average_ranks;
    L.compute_spearman_correlation_matrix  = @compute_spearman_correlation_matrix;
    L.compute_autocorrelation_function     = @compute_autocorrelation_function;
    L.compute_partial_autocorrelation_function = @compute_partial_autocorrelation_function;

    L.compute_day_of_year            = @compute_day_of_year;
    L.find_missing_calendar_days     = @find_missing_calendar_days;
    L.fit_annual_harmonic            = @fit_annual_harmonic;
    L.compute_daily_climatology      = @compute_daily_climatology;
    L.circular_moving_average        = @circular_moving_average;
    L.compute_longest_constant_run   = @compute_longest_constant_run;

    L.plot_concrete_histograms          = @plot_concrete_histograms;
    L.plot_concrete_correlation_matrix  = @plot_concrete_correlation_matrix;
    L.plot_concrete_scatter_plots       = @plot_concrete_scatter_plots;
    L.plot_melbourne_series             = @plot_melbourne_series;
    L.plot_melbourne_distribution       = @plot_melbourne_distribution;
    L.plot_melbourne_monthly_box_plots  = @plot_melbourne_monthly_box_plots;
    L.plot_melbourne_annual_cycle       = @plot_melbourne_annual_cycle;
    L.plot_melbourne_autocorrelation    = @plot_melbourne_autocorrelation;
    L.plot_melbourne_lag_scatter        = @plot_melbourne_lag_scatter;
    L.plot_melbourne_abrupt_changes     = @plot_melbourne_abrupt_changes;

    L.draw_box_plot                  = @draw_box_plot;
    L.draw_correlation_stems         = @draw_correlation_stems;
    L.save_figure                    = @save_figure;
    L.ensure_folder_exists           = @ensure_folder_exists;
    L.print_correlation_matrix       = @print_correlation_matrix;
    L.build_blue_white_red_colormap  = @build_blue_white_red_colormap;
end


%% ========================================================================
%  DATA LOADING
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


%% ========================================================================
%  DATASET INSPECTION
%  ========================================================================

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


%% ========================================================================
%  STATISTICS (implemented here: no Statistics Toolbox required)
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


%% ========================================================================
%  PRINTING HELPERS
%  ========================================================================

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


%% ========================================================================
%  STEP 2 - EXPLORATORY DATA ANALYSIS: MELBOURNE
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


%% ========================================================================
%  STEP 2 - STATISTICS
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


%% ========================================================================
%  STEP 2 - DRAWING AND PRINTING HELPERS
%  ========================================================================

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
