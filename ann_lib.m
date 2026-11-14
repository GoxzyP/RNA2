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
    % Contents (Step 1 - data acquisition and initial inspection)
    % -----------------------------------------------------------
    % Data loading     : load_concrete_dataset, load_melbourne_dataset
    % Inspection       : inspect_concrete_dataset, inspect_melbourne_dataset
    % Statistics       : compute_column_summary, compute_percentile
    % Printing helpers : print_column_summary, print_section_title

    L.load_concrete_dataset     = @load_concrete_dataset;
    L.load_melbourne_dataset    = @load_melbourne_dataset;

    L.inspect_concrete_dataset  = @inspect_concrete_dataset;
    L.inspect_melbourne_dataset = @inspect_melbourne_dataset;

    L.compute_column_summary    = @compute_column_summary;
    L.compute_percentile        = @compute_percentile;

    L.print_column_summary      = @print_column_summary;
    L.print_section_title       = @print_section_title;
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
    %     input_units  : 1 x 8 cell of char, unit of each input column
    %     target_name  : char, name of the target variable
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
    shape_is_wrong = (number_of_rows    ~= expected_number_of_rows) || ...
                     (number_of_columns ~= expected_number_of_columns);
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

    dataset.input_units = {'kg/m^3', 'kg/m^3', 'kg/m^3', ...
                           'kg/m^3', 'kg/m^3', 'kg/m^3', ...
                           'kg/m^3', 'days'};

    dataset.target_name = 'Compressive strength';
    dataset.target_unit = 'MPa';
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
            if table_already_has_rows && ...
               numel(numbers_in_line) ~= size(numeric_table, 2)
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

            value_starts_with_bad_character = ...
                ~isempty(regexp(value_text, '^[^0-9.+\-]', 'once'));

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
            dates(number_of_observations) = ...
                datenum(year_number, month_number, day_number);
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

    number_of_exact_duplicate_rows = ...
        number_of_observations - size(unique(all_columns, 'rows'), 1);

    % A "mixture" is the 7 ingredients; the same recipe can be tested at
    % several ages, which matters for a random train/validation/test split.
    age_column_index        = find(strcmp(dataset.input_names, 'Age'));
    ingredient_column_index = setdiff(1:number_of_inputs, age_column_index);
    ingredient_matrix       = dataset.inputs(:, ingredient_column_index);

    [distinct_mixtures, unused_first_index, mixture_id_of_each_row] = ...
        unique(ingredient_matrix, 'rows');

    number_of_distinct_mixtures = size(distinct_mixtures, 1);

    number_of_rows_per_mixture = ...
        histc(mixture_id_of_each_row, 1:number_of_distinct_mixtures);

    mixture_is_repeated = number_of_rows_per_mixture > 1;
    number_of_repeated_mixtures = sum(mixture_is_repeated);
    number_of_rows_in_repeated_mixtures = ...
        sum(number_of_rows_per_mixture(mixture_is_repeated));

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
    inspection_report.number_of_rows_in_repeated_mixtures = ...
        number_of_rows_in_repeated_mixtures;
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

    index_before_each_gap = find(days_between_rows > 1);
    missing_dates = [];
    for gap_number = 1:numel(index_before_each_gap)
        index_before_gap  = index_before_each_gap(gap_number);
        first_missing_day = dates(index_before_gap) + 1;
        last_missing_day  = dates(index_before_gap + 1) - 1;
        missing_dates = [missing_dates; (first_missing_day:last_missing_day)'];
    end

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

    fprintf('%-6s %8s %9s %8s %8s %8s\n', ...
            'Year', 'Observed', 'Expected', 'Mean', 'Min', 'Max');
    for year_index = 1:numel(distinct_years)
        current_year = distinct_years(year_index);
        row_is_in_year = (year_of_each_row == current_year);
        temperatures_of_year = temperature_celsius(row_is_in_year);

        expected_days_in_year = datenum(current_year + 1, 1, 1) - ...
                                datenum(current_year,     1, 1);

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

    [largest_day_to_day_change, index_before_largest_change] = ...
        max(abs(temperature_change_to_next_row));
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
        standard_deviation(column_index) = ...
            sqrt(sum(centered_values .^ 2) / (count - 1));

        population_standard_deviation = sqrt(sum(centered_values .^ 2) / count);
        if population_standard_deviation > 0
            skewness(column_index) = ...
                (sum(centered_values .^ 3) / count) / population_standard_deviation ^ 3;
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

    percentile_value = (1 - weight_of_upper_value) * sorted_values(lower_position) + ...
                            weight_of_upper_value  * sorted_values(upper_position);
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
    fprintf('%-22s %10s %10s %10s %10s %10s\n', ...
            'Variable', 'Min', 'Q1', 'Median', 'Q3', 'Max');
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
    fprintf('%-22s %10s %10s %10s %8s %8s %8s\n', ...
            'Variable', 'Mean', 'Std', 'Skewness', 'Zeros', 'Distinct', 'Missing');
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
