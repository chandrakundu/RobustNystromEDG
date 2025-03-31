function plot_points(P1, P2, m)
    pt1 = P1 - mean(P1);
    pt2 = P2 - mean(P2);
    [U,~,V] = svd(pt2'*pt1);
    RR = U*V';
    pt2 = pt2*RR;
    figure;
    plot(pt1(1:m,1), pt1(1:m,2), 'r.', 'Markersize', 8, 'DisplayName', 'Original anchor points');
    hold on;
    plot(pt1(m+1:end,1),pt1(m+1:end,2),'b.', 'Markersize', 8, 'DisplayName', 'Original target points')
    plot(pt2(1:m,1), pt2(1:m,2), 'ro', 'Markersize', 8, 'DisplayName', 'Estimated anchor points');
    hold on;
    plot(pt2(m+1:end,1),pt2(m+1:end,2),'bo', 'Markersize', 8, 'DisplayName', 'Estimated target points')
    legend('Location','bestoutside');
    axis equal;
    grid on;    

end
