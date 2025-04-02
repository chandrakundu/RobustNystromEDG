function write_markdown_table(fid, alpha_values, m_values, rmse_matrix, std_matrix, recovered_matrix)
    % Write Markdown table header
    fprintf(fid, '| m \\ alpha |');
    for alpha = alpha_values
        fprintf(fid, ' %.2f |', alpha);
    end
    fprintf(fid, '\n|---|');
    fprintf(fid, repmat('---|', 1, length(alpha_values)));

    % Write data rows
    for i = 1:length(m_values)
        fprintf(fid, '\n| %d |', m_values(i));
        for j = 1:length(alpha_values)
            fprintf(fid, ' %.3f (%.2f) (%d) |', rmse_matrix(i, j), std_matrix(i, j), recovered_matrix(i, j));
        end
    end

    fprintf(fid, '\n\n');
end
