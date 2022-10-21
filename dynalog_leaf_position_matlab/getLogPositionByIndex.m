function [planndPositionSet, actualPositionSet, predictedPositionSet] = getLogPositionByIndex(idx_, planning_data_,Delevered_data_ ,prediction_data_, zero_colum_index_, test_data_index_)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
test_data = idx_;
test_index_real_value = 1;
non_zeros_size_all = [];

for iter00 = 1:test_data
    temp_non_zero_size = size(nonzeros(zero_colum_index_(iter00,:))',2);
    non_zeros_size_all = horzcat(non_zeros_size_all, temp_non_zero_size);
end

if test_data >= 2 %  2= 70921 / 3 = 78661 / 4 =
    test_index_real_value = sum((60-(non_zeros_size_all(1:test_data-1))).*test_data_index_(1:test_data-1))+1;
     
end

test_index = test_data_index_(test_data);
predictedPositionSet = zeros(test_index,60);
actualPositionSet = zeros(test_index,60);
planndPositionSet =  zeros(test_index,60);

temp_zero_index = nonzeros(zero_colum_index_(test_data,:))';
zero_index_number = [];
%  iter01 = 60
 
for iter01 = 1:60

    if intersect(temp_zero_index,iter01)
        predictedPositionSet(:,iter01) = 0;
        actualPositionSet(:,iter01) = 0;
        planndPositionSet(:,iter01) = 0;
        zero_index_number = horzcat(zero_index_number,intersect(temp_zero_index,iter01));
    else
        iter02 = iter01 - size(zero_index_number,2);
        nn = test_index_real_value+(test_index*(iter02))-test_index;
        predictedPositionSet(:,iter01) = prediction_data_(nn:nn+test_index-1);
        actualPositionSet(:,iter01) = Delevered_data_(nn:nn+test_index-1);
        planndPositionSet(:,iter01) = planning_data_(nn:nn+test_index-1);
        
    end
    
end



end