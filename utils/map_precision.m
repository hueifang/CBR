function [map, precision_at_k] = map_precision (trn_label, trn_binary, tst_label, tst_binary, mode)   
K = size(trn_binary, 1);
QueryTimes = size(tst_binary,1);

AP = zeros(QueryTimes,1);

Ns = 1:1:K;
sum_tp = zeros(1, length(Ns));

for i = 1:QueryTimes
    
    query_label = tst_label(i);
    fprintf('query %d\n',i);
    query_binary = tst_binary(i,:);
    if mode==1
        tic
        similarity = trn_binary * query_binary';
        toc
        [x2,y2] = sort(similarity, 'descend');
        fprintf('Complete Query [Hamming] %.2f seconds\n',toc);
    elseif mode ==2
        tic
        similarity = pdist2(double(trn_binary),double(query_binary),'euclidean');
        toc
        [x2,y2] = sort(similarity);%, 'descend');
        fprintf('Complete Query [Euclidean] %.2f seconds\n',toc);
    end
        
    buffer_yes = trn_label(y2(1:K)) == query_label;
    
    % compute precision
    P = cumsum(buffer_yes) ./ Ns';
    if (sum(buffer_yes) == 0)
        AP(i) = 0;
    else
        AP(i) = sum(P .* buffer_yes) / sum(buffer_yes);
    end
    sum_tp = sum_tp + cumsum(buffer_yes)';
end  
    precision_at_k = sum_tp ./ (Ns * QueryTimes);
    map = mean(AP);
end