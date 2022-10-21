clc
clear
close all

%%

preprocessedDataPath = "C:\Users\user\Downloads\dynalog_leaf_position\dynalog_leaf_position_matlab\deeplearningData-220812-150745"
predictedResultPath = "C:\Users\user\Downloads\dynalog_leaf_position\dynalog_leaf_position_python\result"

prediction_data = load(fullfile(predictedResultPath,'test_result_data.mat'));
prediction_data = prediction_data.test_result_data;
prediction_data = prediction_data';
% prediction_data(1) = [];·
Delevered_data = load(fullfile(preprocessedDataPath, 'test_output_data_1st.mat'));
Delevered_data = Delevered_data.iter00_output_data;

planning_data = load(fullfile(preprocessedDataPath, 'test_planning_data_1st.mat'));
planning_data = planning_data.iter00_planning_data;


test_data_index = load(fullfile(preprocessedDataPath, 'data_label_1st.mat'));
test_data_index = test_data_index.data_label';
test_data_index = test_data_index -10;


zero_colum_index = load(fullfile(preprocessedDataPath, 'zero_colum_index_1st.mat'));
zero_colum_index = zero_colum_index.zero_colum_index;

testDataIndex = load(fullfile(preprocessedDataPath, 'testDataIndex.mat'))
testDataIndex= testDataIndex.dynalogList;
%%

mergedData = [prediction_data, Delevered_data, planning_data];

%% RMSE

for iter001 = 1:size(mergedData,1)
    temp_RMSE_all_pred(iter001,1) = sqrt(mean((mergedData(iter001,2) - mergedData(iter001,1)).^2));
    temp_RMSE_all_plan(iter001,1) = sqrt(mean((mergedData(iter001,2) - mergedData(iter001,3)).^2));

end
mean(temp_RMSE_all_pred)
mean(temp_RMSE_all_plan) % 0.1706

   
%%

test_data = 4;
test_index_real_value = 1;
non_zeros_size_all = [];

for iter00 = 1:test_data
    temp_non_zero_size = size(nonzeros(zero_colum_index(iter00,:))',2);
    non_zeros_size_all = horzcat(non_zeros_size_all, temp_non_zero_size);
end

if test_data >= 2 %  2= 70921 / 3 = 78661 / 4 =
    test_index_real_value = sum((60-(non_zeros_size_all(1:test_data-1))).*test_data_index(1:test_data-1))+1;
     
end

test_index = test_data_index(test_data);
model_pred_image = zeros(test_index,60);
true_image = zeros(test_index,60);
planning_image =  zeros(test_index,60);

temp_zero_index = nonzeros(zero_colum_index(test_data,:))';
zero_index_number = [];
%  iter01 = 60
 
for iter01 = 1:60

    if intersect(temp_zero_index,iter01)
        model_pred_image(:,iter01) = 0;
        true_image(:,iter01) = 0;
        planning_image(:,iter01) = 0;
        zero_index_number = horzcat(zero_index_number,intersect(temp_zero_index,iter01));
    else
        iter02 = iter01 - size(zero_index_number,2);
        nn = test_index_real_value+(test_index*(iter02))-test_index;
        model_pred_image(:,iter01) = prediction_data(nn:nn+test_index-1);
        true_image(:,iter01) = Delevered_data(nn:nn+test_index-1);
        planning_image(:,iter01) = planning_data(nn:nn+test_index-1);
        
    end
    
end

[planndPositionSet, actualPositionSet, predictedPositionSet] = getLogPositionByIndex(4, planning_data, Delevered_data, prediction_data, zero_colum_index);

% figure,
% image(model_pred_image),title('predicted');xlabel('Leaf number');ylabel('Time stamp');
% figure,
% image(true_image),title('Delevered');xlabel('Leaf number');ylabel('Time stamp');
% figure,
% image(planning_image),title('planned');xlabel('Leaf number');ylabel('Time stamp');
figure
subplot(1,3,1), image(model_pred_image),title('predicted');xlabel('Leaf number');ylabel('Time stamp');
subplot(1,3,2), image(true_image),title('Delevered');xlabel('Leaf number');ylabel('Time stamp');
subplot(1,3,3), image(planning_image),title('planned');xlabel('Leaf number');ylabel('Time stamp');

%
selectedDynalogPath_tmp1 = fullfile(testDataIndex(test_data).folder, testDataIndex(test_data).name);
selectedDynalogSt_temp1 = dynRead(selectedDynalogPath_tmp1)

if testDataIndex(test_data).name(1) == 'A'
    pairedDynalogFilename = testDataIndex(test_data).name;
    pairedDynalogFilename = strrep(pairedDynalogFilename, 'A', 'B')
    selectedDynalogPath_tmp2 = fullfile(testDataIndex(test_data).folder, pairedDynalogFilename);
    selectedDynalogSt_temp2 = dynRead(selectedDynalogPath_tmp2)

    dynalogA = selectedDynalogSt_temp1;
    dynalogB = selectedDynalogSt_temp2;
elseif testDataIndex(test_data).name(1) == 'B'
    pairedDynalogFilename = testDataIndex(test_data).name;
    pairedDynalogFilename = strrep(pairedDynalogFilename, 'B', 'A')
    selectedDynalogPath_tmp2 = fullfile(testDataIndex(test_data).folder, pairedDynalogFilename);
    selectedDynalogSt_temp2 = dynRead(selectedDynalogPath_tmp2)

    dynalogA = selectedDynalogSt_temp2;
    dynalogB = selectedDynalogSt_temp1;
end

3d



[plannedFluenceMap, actualFluenceMap] = dynFluence(dynalogA, dynalogB, 1, 1);
dynalogA.actualPosition (11:end)= 
figure, imshow(aa , [])

%% plot
leaf_number = 30 % 1 ~ 60
mergedDataTemp = [true_image(:,leaf_number), model_pred_image(:,leaf_number), planning_image(:,leaf_number)]

for iter05 = 1:size(mergedDataTemp,1)
    temp_RMSE_pred(iter05,1) = sqrt(mean((mergedDataTemp(iter05,1) - mergedDataTemp(iter05,2)).^2));
    temp_RMSE_true(iter05,1) = sqrt(mean((mergedDataTemp(iter05,1) - mergedDataTemp(iter05,3)).^2));

end
    
mean(temp_RMSE_pred)
mean(temp_RMSE_true)

h=figure,
% h.Position = [0.13,0.11,0.775,0.815]
plot(mergedDataTemp(:,2),'o','MarkerSize',3);
hold on
plot(mergedDataTemp(:,1),'*','MarkerSize',3);
hold on
plot(mergedDataTemp(:,3),'^','MarkerSize',3);
legend('predicted', 'Delevered', 'planned');ylabel(sprintf('Posotion of %dth leaf at isocenter [mm]', leaf_number));xlabel('Time stamp');
grid on











