% =========================================================================
%  FUZZY GENETIC ALGORITHM - COMPLETE JOURNAL ANALYSIS
%  Author: Automated Script
%  Description: Analyzes BOTH Phase 1 (Tuning) and Phase 2 (Ablation) data.
%               Generates all plots and statistics in one go.
% =========================================================================

clear; clc; close all;

% --- CONFIGURATION ---
dataDir = 'research_data';
filePhase1 = fullfile(dataDir, 'phase1_tuning_results.csv');
filePhase2 = fullfile(dataDir, 'phase2_ablation_results.csv');

% =========================================================================
%  PART 1: PHASE 1 - HYPERPARAMETER TUNING ANALYSIS
% =========================================================================
fprintf('\n=================================================\n');
fprintf('STARTING PHASE 1 ANALYSIS: Hyperparameter Tuning\n');
fprintf('=================================================\n');

if isfile(filePhase1)
    % 1. Load Data
    opts = detectImportOptions(filePhase1);
    opts = setvartype(opts, {'run_id', 'experiment'}, 'categorical');
    T1 = readtable(filePhase1, opts);
    
    % 2. Process Data
    % We want to compare different tuning configs (run_ids)
    try
        % Pivot: Rows=Gen, Cols=Config(run_id)
        pivoted = unstack(T1, 'best_fitness', 'run_id');
        
        % Extract Generation and Data Matrix
        gens = pivoted.generation;
        % Extract numeric columns that correspond to run_ids
        dataCols = pivoted(:, varfun(@isnumeric, pivoted, 'OutputFormat', 'uniform'));
        % Remove generation column from data matrix
        if ismember('generation', dataCols.Properties.VariableNames)
             tuningMatrix = dataCols{:, ~strcmp(dataCols.Properties.VariableNames, 'generation')};
        else
             tuningMatrix = dataCols{:, :};
        end
        tuningNames = pivoted.Properties.VariableNames(2:end); % Rough approximation of names
        
        % 3. Plot Convergence Comparison
        figure('Name', 'Phase 1: Tuning Convergence', 'Color', 'w', 'Position', [50, 50, 700, 500]);
        hold on; grid on;
        plot(gens, tuningMatrix, 'LineWidth', 2);
        
        xlabel('Generation');
        ylabel('Best Fitness');
        title('Phase 1: Hyperparameter Convergence Speed');
        legend(tuningNames, 'Location', 'southeast', 'Interpreter', 'none');
        set(gca, 'FontSize', 11);
        
        % 4. Bar Chart of Final Scores
        finalScores = tuningMatrix(end, :);
        figure('Name', 'Phase 1: Final Scores', 'Color', 'w', 'Position', [760, 50, 500, 500]);
        b = bar(finalScores);
        b.FaceColor = 'flat';
        xticklabels(tuningNames);
        ylabel('Final Fitness Score');
        title('Comparison of Hyperparameter Sets');
        grid on;
        
        [maxScore, idx] = max(finalScores);
        fprintf('Phase 1 Best Config: Column %d (Score: %.4f)\n', idx, maxScore);
        
    catch ME
        warning('Phase 1 Analysis Failed: %s', ME.message);
    end
else
    fprintf('Skipping Phase 1: File not found (%s)\n', filePhase1);
end

% =========================================================================
%  PART 2: PHASE 2 - ABLATION STUDY (THE CORE PAPER RESULTS)
% =========================================================================
fprintf('\n=================================================\n');
fprintf('STARTING PHASE 2 ANALYSIS: Ablation Study\n');
fprintf('=================================================\n');

if isfile(filePhase2)
    % 1. Load Data
    opts = detectImportOptions(filePhase2);
    opts = setvartype(opts, {'run_id', 'experiment'}, 'categorical');
    T2 = readtable(filePhase2, opts);
    
    experiments = unique(T2.experiment);
    colors = lines(length(experiments)); 
    finalFitnessStruct = struct();
    
    % Prepare Plot 1: Trajectory
    figTraj = figure('Name', 'Phase 2: Evolutionary Trajectory', 'Color', 'w', 'Position', [100, 100, 800, 600]);
    hold on; grid on;
    
    for i = 1:length(experiments)
        expName = string(experiments(i));
        subTable = T2(T2.experiment == expName, :);
        
        try
            % Pivot: Rows=Gen, Cols=Trials
            pivoted = unstack(subTable, 'best_fitness', 'run_id');
            gens = pivoted.generation;
            
            % Extract numeric data safely
            dataCols = pivoted(:, varfun(@isnumeric, pivoted, 'OutputFormat', 'uniform'));
            if ismember('generation', dataCols.Properties.VariableNames)
                 fitMatrix = dataCols{:, ~strcmp(dataCols.Properties.VariableNames, 'generation')};
            else
                 fitMatrix = dataCols{:, :};
            end
            
            % Stats
            mu = mean(fitMatrix, 2, 'omitnan');
            sigma = std(fitMatrix, 0, 2, 'omitnan');
            
            % Store for Box Plot
            safeName = matlab.lang.makeValidName(char(expName));
            finalFitnessStruct.(safeName) = fitMatrix(end, :)';
            
            % Plot Shading
            x_poly = [gens; flipud(gens)];
            y_poly = [mu - sigma; flipud(mu + sigma)];
            fill(x_poly, y_poly, colors(i,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
                'DisplayName', sprintf('%s (Std Dev)', expName));
            
            % Plot Mean
            plot(gens, mu, 'Color', colors(i,:), 'LineWidth', 2.5, ...
                'DisplayName', sprintf('%s (Mean)', expName));
            
            fprintf('Processed Experiment: %s | Final Mean: %.4f\n', expName, mu(end));
            
        catch ME
            warning('Could not process experiment %s: %s', expName, ME.message);
        end
    end
    
    % Format Trajectory Plot
    figure(figTraj);
    xlabel('Generation', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Fitness Score', 'FontSize', 12, 'FontWeight', 'bold');
    title('Evolutionary Trajectory Comparison', 'FontSize', 14);
    legend('Location', 'southeast', 'Interpreter', 'none');
    set(gca, 'FontSize', 11);
    
    % Prepare Plot 2: Box Plot
    figure('Name', 'Phase 2: Distribution', 'Color', 'w', 'Position', [950, 100, 600, 600]);
    
    fieldNames = fieldnames(finalFitnessStruct);
    dataGroups = [];
    groupIndices = [];
    
    if ~isempty(fieldNames)
        for i = 1:length(fieldNames)
            vals = finalFitnessStruct.(fieldNames{i});
            dataGroups = [dataGroups; vals]; %#ok<AGROW>
            for j = 1:length(vals)
                groupIndices = [groupIndices; i]; %#ok<AGROW>
            end
        end
        
        boxplot(dataGroups, groupIndices, 'Labels', fieldNames);
        grid on;
        ylabel('Final Fitness Score');
        title('Statistical Distribution of Final Performance');
        set(gca, 'TickLabelInterpreter', 'none', 'FontSize', 11);
        
        % Jitter overlay
        hold on;
        for i = 1:length(fieldNames)
            vals = finalFitnessStruct.(fieldNames{i});
            x_vals = i + (rand(size(vals))-0.5)*0.15;
            scatter(x_vals, vals, 50, 'k', 'filled', 'MarkerFaceAlpha', 0.6);
        end
        hold off;
        
        % 3. Statistical Test (Automated)
        fprintf('\n--- Pairwise T-Test Analysis ---\n');
        if length(fieldNames) >= 2
            % Compare first two groups (usually Baseline vs Proposed)
            g1 = fieldNames{1};
            g2 = fieldNames{2};
            d1 = finalFitnessStruct.(g1);
            d2 = finalFitnessStruct.(g2);
            
            [h, p, ~, ~] = ttest2(d1, d2);
            fprintf('Comparing %s vs %s:\n', g1, g2);
            fprintf('   p-value: %.5e\n', p);
            if h
                fprintf('   RESULT: Significant Difference (p < 0.05)\n');
                if mean(d1) > mean(d2)
                     fprintf('   WINNER: %s\n', g1);
                else
                     fprintf('   WINNER: %s\n', g2);
                end
            else
                fprintf('   RESULT: No Significant Difference detected.\n');
            end
        end
    end
    
else
    fprintf('Skipping Phase 2: File not found (%s)\n', filePhase2);
end

fprintf('\nAnalysis Complete. All figures generated.\n');