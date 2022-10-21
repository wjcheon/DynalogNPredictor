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

testDataList = load(fullfile(preprocessedDataPath, 'testDataIndex.mat'))
testDataList= testDataList.dynalogList;
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
idx_testDynalog = 4;

% Fluence map 
selectedDynalogFileName = testDataList(idx_testDynalog).name;
isPairedAB = '';
if selectedDynalogFileName(1) == 'A'
    pairedDynalogFilename = strrep(selectedDynalogFileName, 'A', 'B');
    dynalogAFileName = selectedDynalogFileName;
    dynalogBFileName = pairedDynalogFilename;
    isPairedAB= 'B';
elseif selectedDynalogFileName == 'B'
    pairedDynalogFilename = strrep(selectedDynalogFileName, 'B', 'A');
    dynalogAFileName = pairedDynalogFilename;
    dynalogBFileName = selectedDynalogFileName;
    isPairedAB= 'A';
end

%%%%%%%% Selected dynalog 
try 
    dynalogA = dynRead(fullfile(testDataList(idx_testDynalog).folder, dynalogAFileName));
    fprintf("%s: file was successfully loaded !!\n", dynalogAFileName)
catch
    fprintf("%s: file not exist !! ", dynalogAFileName)
end

try 
    dynalogB = dynRead(fullfile(testDataList(idx_testDynalog).folder, dynalogBFileName));
    fprintf("%s: file was successfully loaded !!\n ", dynalogBFileName)
catch
    fprintf("%s: file not exist !! ", dynalogBFileName)
end

idx_pairedDynalogIndex = find(strcmp({testDataList.name}, pairedDynalogFilename));
if isPairedAB=='A'
    [planndPositionSetB, actualPositionSetB, predictedPositionSetB] = getLogPositionByIndex(idx_testDynalog , planning_data, Delevered_data, prediction_data, zero_colum_index, test_data_index);
    [planndPositionSetA, actualPositionSetA, predictedPositionSetA] = getLogPositionByIndex(idx_pairedDynalogIndex, planning_data, Delevered_data, prediction_data, zero_colum_index, test_data_index);

elseif isPairedAB=='B'
    
    [planndPositionSetA, actualPositionSetA, predictedPositionSetA] = getLogPositionByIndex(idx_testDynalog , planning_data, Delevered_data, prediction_data, zero_colum_index, test_data_index);
    [planndPositionSetB, actualPositionSetB, predictedPositionSetB] = getLogPositionByIndex(idx_pairedDynalogIndex, planning_data, Delevered_data, prediction_data, zero_colum_index, test_data_index);
end

figure
subplot(2,3,1), image(planndPositionSetA),title('Planned position: bankA');xlabel('Leaf number');ylabel('Time stamp');
subplot(2,3,2), image(actualPositionSetA),title('Actual position: bankA');xlabel('Leaf number');ylabel('Time stamp');
subplot(2,3,3), image(predictedPositionSetA),title('Predicted position: bankA');xlabel('Leaf number');ylabel('Time stamp');
subplot(2,3,4), image(planndPositionSetB),title('Planned position: bankB');xlabel('Leaf number');ylabel('Time stamp');
subplot(2,3,5), image(actualPositionSetB),title('Actual position: bankB');xlabel('Leaf number');ylabel('Time stamp');
subplot(2,3,6), image(predictedPositionSetB),title('Predicted position: bankB');xlabel('Leaf number');ylabel('Time stamp');



[plannedFluenceMap, actualFluenceMap] = dynFluence(dynalogA, dynalogB, 1, 1);

dynalogA.actualPosition(11:end, :) = predictedPositionSetA;
dynalogB.actualPosition(11:end, :) = predictedPositionSetB;
[~, predictedFluenceMap] = dynFluence(dynalogA, dynalogB, 1, 1);


figure, 
subplot(3,1,1), imshow(plannedFluenceMap, []),title('Planned FluenceMap');
subplot(3,1,2), imshow(actualFluenceMap, []),title('Actual FluenceMap');
subplot(3,1,3), imshow(predictedFluenceMap, []),title('Predicted FluenceMap');

%% plot
leaf_number = 30 % 1 ~ 60
mergedDataTemp = [actualPositionSetA(:,leaf_number), predictedPositionSetA(:,leaf_number), planndPositionSetA(:,leaf_number)]

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











