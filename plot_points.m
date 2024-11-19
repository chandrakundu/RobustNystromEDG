function plot_points(P1, P2, m)
    pt1 = P1 - mean(P1);
    pt2 = P2 - mean(P2);
    [U,~,V] = svd(pt2'*pt1);
    RR = U*V';
    pt2 = pt2*RR;
    figure;
    plot(pt1(1:m,1), pt1(1:m,2), 'r.', 'Markersize', 8, 'DisplayName', 'Point 1: Anchor Points');
    hold on;
    plot(pt1(m+1:end,1),pt1(m+1:end,2),'b.', 'Markersize', 8, 'DisplayName', 'Point 1: Target Points')
    plot(pt2(1:m,1), pt2(1:m,2), 'ro', 'Markersize', 8, 'DisplayName', 'Point 2: Anchor Points');
    hold on;
    plot(pt2(m+1:end,1),pt2(m+1:end,2),'bo', 'Markersize', 8, 'DisplayName', 'Point 2: Target Points')
    legend('Location','bestoutside');
    axis equal;
    grid on;    

end
