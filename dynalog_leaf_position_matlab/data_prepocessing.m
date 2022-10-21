clc
clear
close all
addpath("dynalog_function")
%% data load
% dataset : 170 개
% Train : 136
% Test : 34

dynalogTrainDataPath = ".\raw_data\train_1st\";   % training data path
dynalogTestDataPath = ".\raw_data\test_1st\"; % test data path

formatOut = 'yymmdd-HHMMSS';
t_string = datestr(now, formatOut);
deeplearningDatasourcePath = sprintf("deeplearningData-%s", t_string); % result path 
mkdir(deeplearningDatasourcePath)
%% Training data preparation

dynalogList = dir(fullfile(dynalogTrainDataPath,"*.dlg"));   % training data path
%
iter00_input_data = [];
iter00_output_data = [];
iter00_planning_data = [];
%
data_label = [];
zero_colum_index = zeros(34,100);

f=waitbar(0,'Training data preparation...');
for iter00 = 1:size(dynalogList,1)
    % data load
    waitbar(iter00/size(dynalogList,1), f,'Training data preparation...');

    struct_data = dynRead(fullfile(dynalogTrainDataPath, dynalogList(iter00).name));
    data_label=vertcat(data_label,size(struct_data.planPosition,1));

    zero_colum = [] ;

    for iter01 = 1:size(struct_data.leafNumber,2) % ex 1538
        if  sum(struct_data.actualPosition(:,iter01)) == 0
            qqq2 = iter01;
            zero_colum = horzcat(zero_colum,qqq2);
        end

    end

    zero_colum_index(iter00,1:size(zero_colum,2)) = zero_colum;


    struct_data.planPosition(:,zero_colum) = [];
    struct_data.actualPosition(:,zero_colum) = [];

    % lateral data add leaf 맨 앞과 맨 뒤에 data 추가
    struct_data.planPosition = horzcat(struct_data.planPosition,struct_data.planPosition(:,end));
    struct_data.planPosition = horzcat(struct_data.planPosition(:,1),struct_data.planPosition);


    struct_data.actualPosition = horzcat(struct_data.actualPosition,struct_data.actualPosition(:,end));
    struct_data.actualPosition = horzcat(struct_data.actualPosition(:,1),struct_data.actualPosition);

    % make input, output data

    time_step = 10; % deep-learning 에 입력으로 들어가는 한개 input sequence의 길이 
    data_row_size =size(struct_data.actualPosition,1) ;
    data_column_size = size(struct_data.actualPosition,2);


    iter03_input_data = [];
    iter03_output_data = [];
    iter03_planning_data = [];


    for iter03 = 2:data_column_size-1
        iter02_input_data = [];
        iter02_output_data = [];
        iter02_planning_data = [];

        for iter02 = 1:data_row_size-time_step
            temp_input_data_01 = struct_data.planPosition(iter02:iter02+time_step-1,iter03-1:iter03+1);
            temp_input_data_02 = struct_data.gantryRotation(iter02:iter02+time_step-1,1);
            temp_input_data_03 = struct_data.collimatorRotation(iter02:iter02+time_step-1,1);
            temp_input_data_04 = struct_data.beamOn(iter02:iter02+time_step-1,1);

            temp_input_data = horzcat(temp_input_data_02,temp_input_data_03,temp_input_data_04,temp_input_data_01);

            temp_output_data = struct_data.actualPosition(iter02+time_step,iter03);
            temp_planning_data = struct_data.planPosition(iter02+time_step,iter03);

            iter02_input_data = cat(3,iter02_input_data,temp_input_data);
            iter02_output_data = vertcat(iter02_output_data,temp_output_data);
            iter02_planning_data =  vertcat(iter02_planning_data,temp_planning_data);
        end

        iter03_input_data = cat(3,iter03_input_data,iter02_input_data);
        iter03_output_data = vertcat(iter03_output_data,iter02_output_data);
        iter03_planning_data =  vertcat(iter03_planning_data,iter02_planning_data);
    end

    iter00_input_data = cat(3,iter00_input_data,iter03_input_data);
    iter00_output_data = vertcat(iter00_output_data,iter03_output_data);
    iter00_planning_data = vertcat(iter00_planning_data,iter03_planning_data);

end
close(f)

% train data save
savepath_input_tr = fullfile(deeplearningDatasourcePath, "train_input_data_1st.mat");
savepath_output_tr = fullfile(deeplearningDatasourcePath, "train_output_data_1st.mat");
save(savepath_input_tr,'iter00_input_data','-v7.3');
%save("train_input_data_1st.mat",'iter00_input_data');
save(savepath_output_tr,'iter00_output_data');

clear iter00_input_data iter00_output_data dynalogList
%% Test data preparation
dynalogList = dir(fullfile(dynalogTestDataPath,"*.dlg"));   % test data path

iter00_input_data = [];
iter00_output_data = [];
iter00_planning_data = [];

%
data_label = [];
zero_colum_index = zeros(34,100);

f=waitbar(0,'Test data preparation...');
for iter00 = 1:size(dynalogList,1)
    % data load
    waitbar(iter00/size(dynalogList,1), f,'Training data preparation...');

    struct_data = dynRead(fullfile(dynalogTestDataPath, dynalogList(iter00).name));
    data_label=vertcat(data_label,size(struct_data.planPosition,1));

    zero_colum = [] ;

    for iter01 = 1:size(struct_data.leafNumber,2) % ex 1538
        if  sum(struct_data.actualPosition(:,iter01)) == 0
            qqq2 = iter01;
            zero_colum = horzcat(zero_colum,qqq2);
        end

    end

    zero_colum_index(iter00,1:size(zero_colum,2)) = zero_colum;


    struct_data.planPosition(:,zero_colum) = [];
    struct_data.actualPosition(:,zero_colum) = [];

    % lateral data add leaf 맨 앞과 맨 뒤에 data 추가
    struct_data.planPosition = horzcat(struct_data.planPosition,struct_data.planPosition(:,end));
    struct_data.planPosition = horzcat(struct_data.planPosition(:,1),struct_data.planPosition);


    struct_data.actualPosition = horzcat(struct_data.actualPosition,struct_data.actualPosition(:,end));
    struct_data.actualPosition = horzcat(struct_data.actualPosition(:,1),struct_data.actualPosition);

    % make input, output data

    data_row_size =size(struct_data.actualPosition,1) ;
    data_column_size = size(struct_data.actualPosition,2);


    iter03_input_data = [];
    iter03_output_data = [];
    iter03_planning_data = [];


    for iter03 = 2:data_column_size-1
        iter02_input_data = [];
        iter02_output_data = [];
        iter02_planning_data = [];

        for iter02 = 1:data_row_size-time_step
            temp_input_data_01 = struct_data.planPosition(iter02:iter02+time_step-1,iter03-1:iter03+1);
            temp_input_data_02 = struct_data.gantryRotation(iter02:iter02+time_step-1,1);
            temp_input_data_03 = struct_data.collimatorRotation(iter02:iter02+time_step-1,1);
            temp_input_data_04 = struct_data.beamOn(iter02:iter02+time_step-1,1);

            temp_input_data = horzcat(temp_input_data_02,temp_input_data_03,temp_input_data_04,temp_input_data_01);

            temp_output_data = struct_data.actualPosition(iter02+time_step,iter03);
            temp_planning_data = struct_data.planPosition(iter02+time_step,iter03);

            iter02_input_data = cat(3,iter02_input_data,temp_input_data);
            iter02_output_data = vertcat(iter02_output_data,temp_output_data);
            iter02_planning_data =  vertcat(iter02_planning_data,temp_planning_data);
        end

        iter03_input_data = cat(3,iter03_input_data,iter02_input_data);
        iter03_output_data = vertcat(iter03_output_data,iter02_output_data);
        iter03_planning_data =  vertcat(iter03_planning_data,iter02_planning_data);
    end

    iter00_input_data = cat(3,iter00_input_data,iter03_input_data);
    iter00_output_data = vertcat(iter00_output_data,iter03_output_data);
    iter00_planning_data = vertcat(iter00_planning_data,iter03_planning_data);

end
close(f)


% test data save
savepath_zeroColumn_tr = fullfile(deeplearningDatasourcePath, "zero_colum_index_1st.mat");
savepath_input_test = fullfile(deeplearningDatasourcePath, "test_input_data_1st.mat");
savepath_output_test = fullfile(deeplearningDatasourcePath, "test_output_data_1st.mat");
savepath_planningData_test = fullfile(deeplearningDatasourcePath, "test_planning_data_1st.mat");
savepath_data_label = fullfile(deeplearningDatasourcePath, "data_label_1st.mat");
savepath_testDataIndex = fullfile(deeplearningDatasourcePath, "testDataIndex.mat")

save(savepath_zeroColumn_tr,'zero_colum_index'); % actualposition에서 zero영역은 학습에 사용되지 않았음, 사용되지 않은 영역 확인용
save(savepath_input_test,'iter00_input_data');
save(savepath_output_test,'iter00_output_data');
save(savepath_planningData_test,'iter00_planning_data');
save(savepath_data_label, "data_label")
save(savepath_testDataIndex, "dynalogList")
disp('DONE')
%%






