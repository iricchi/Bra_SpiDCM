function [subjects] = load_all_subjects(datapath,N)
%load_all_subjects loads subjects
%   subjects is dict


subjects = cell(1,N);
for i = 1:N
    data = load(fullfile(datapath,['sess1/sub-' num2str(i) '.mat']));
    task = load(fullfile(datapath,['top10/ses1/task/sub-' num2str(i) '_task_vec.mat']));


    % Create struct
    subjects{i}.y = data.TS;              % y is T x 4
    subjects{i}.name = {'L_S1', 'R_S1', 'L_DH', 'R_DH'};  % or load dynamically
    subjects{i}.U = task.TS;              % U is T x 1
end

end
