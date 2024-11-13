function [RMSE,pt2,RR ] =Compute_RMSE(Point1,Point2)
% Procrustes analysis
Pt1 = Point1 - mean(Point1);
pt2 = Point2 - mean(Point2);
[U,D,V] = svd(pt2'*Pt1);
RR = U*V';
size(RR)
size(pt2)
pt2 = pt2*RR;
figure;
plot(pt2(:,1),pt2(:,2),'r.','Markersize',10);
hold on;
plot(Pt1(:,1),Pt1(:,2),'bo') % return Pt1, Pt1
axis equal;
axis off;
hlen = legend('Recon','GroundTruth');
% set(hlen,'FontSize',20);
RMSE = sqrt(mean(sum((pt2 - Pt1).^2,2)));
% fprintf('RMSE: %f \n', RMSE);
    
end