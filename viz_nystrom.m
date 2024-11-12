function viz_nystrom(P, m, subplot_pos, plot_title)
    % P: point coordinates
    % m: number of anchor points
    % subplot_pos: current subplot position
    % n_plots: total number of plots
    
    % Create subplot
    subplot(1, 3, subplot_pos);
    
    % plot anchor points
    plot(P(1, 1:m), P(2, 1:m), 'ro', 'MarkerSize', 8, 'DisplayName', 'Anchor Points');
    hold on;
    plot(P(1, m+1:end), P(2, m+1:end), 'b+', 'MarkerSize', 8, 'DisplayName', 'Target Points');
    
    % Add labels and legend
    xlabel('X-axis');
    ylabel('Y-axis');
    title(plot_title);
    legend;
    
    % Adjust axis limits for better visibility
    axis equal;
    grid on;
end

% Example usage:
% figure('Position', [100 100 1200 400]);  % Make figure wide enough for subplots
% viz_nystrom(P1, m, 1, 3);  % First plot
% viz_nystrom(P2, m, 2, 3);  % Second plot
% viz_nystrom(P3, m, 3, 3);  % Third plot
