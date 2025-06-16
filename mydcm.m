clear all; close all; clc;

addpath(genpath('/Users/iricchi/PhD_Local/Model/DCM_new/data'));
datapath = '/Users/iricchi/PhD_Local/Model/DCM_new/data';
%% DCM

n = 4; % L_S1, R_S1, L_DH, R_DH
%boost_prior = 0.0;  % or 1e-2 
L_S1 = 1; R_S1 = 2; L_DH = 3; R_DH = 4;


%% Models 

% model_names = {'C4_only_B0_null','C4_only_B41_ascend','C4_only_B_ascdesc',...
%                 'C4_only_B_spinal', 'C4_only_B43_spinal','C4_only_B_cortical',...
%                 'C4_only_B21_cortical','C4_only_B_full'};


model_names = {'C4_only_B0_null','C4_only_B41_ascend',...
               'C4_only_B_ascdesc','C4_only_B_full'};

subjects = load_all_subjects(datapath,10);


n_models = length(model_names);
n_subjects = length(subjects);
TR = 1.55;  % define TR once
F_values = zeros(n_subjects, n_models);  % to store Free Energy values
DCMs = cell(n_subjects, n_models);       % to store DCM structs


%%
% === STEP 1: DCM ESTIMATION ===

for s = 1:n_subjects
    fprintf('Subject %d...\n', s);
    subj = subjects{s};

    for m = 1:n_models
        model_name = model_names{m};
        [a, b, c] = get_dcm_model(model_name);  % binary masks

        % Construct DCM
        DCM = struct();
        DCM.name = sprintf('DCM_sub%02d_%s', s, model_name);

        % Time series
        DCM.Y.y    = subj.y';                     % [Tx4]
        DCM.Y.dt   = TR;
        DCM.Y.name = subj.name;                 % {'L_S1', 'R_S1', ...}
        %DCM.Y.Q    = ones(size(subj.y, 2), 1);   % include all scans

        DCM.U.u = subj.U';                        % [T x 1]
        DCM.U.name = {'Task'};

        % Model structure
        DCM.a = a;
        DCM.b = b;
        DCM.c = c;
        DCM.d = zeros(size(a,1), size(a,2), 0);  % no bilinear terms

        % Options
        DCM.options.analysis    = 'BOLD';
        DCM.options.nonlinear   = 0;
        DCM.options.two_state   = 0;
        DCM.options.stochastic  = 0;
        DCM.options.centre      = 1;
        DCM.options.h = 1; 
        DCM.options.induced = 0;
        DCM.options.nograph = 1;

        % Priors
        [DCM.M.pE, DCM.M.pC] = spm_dcm_fmri_priors(DCM.a, DCM.b, DCM.c);
        fieldnames(DCM.M.pE);
 
        % Increase prior expectation for modulation (e.g., R_DH → L_S1)
        DCM.M.pE.B(1,4) = 0.1;  % or 0.2, 0.3...
        
        % Boost input to region 4
        DCM.M.pE.C(4) = 1.0;

        DCM.M.pE.transit(1) = 2;
        DCM.M.pE.transit(2) = 2;
        DCM.M.pE.transit(3) = 1.2;
        DCM.M.pE.transit(4) = 1.2;
        DCM.M.pE.decay = 0.64;
        DCM.M.pE.epsilon = 0.54;


        % Set priors variance in params
        A_idx = 1:(n*n);
        DCM.M.pC(A_idx, A_idx) = eye(n*n) * 0.04;
        for r = 1:n
            idx = sub2ind([n, n], r, r);  % linear index for A(r,r)
            DCM.M.pC(idx, idx) = 0;  % fix self-connections
        end
        
        % B 
        B_idx0 = n^2;
        for i = 1:n
            for j = 1:n
                idx_flat = sub2ind([n, n], i, j);    % index in B (flattened)
                idx = B_idx0 + idx_flat;
        
                if i == j
                    DCM.M.pC(idx, idx) = 0;          % fix self-connections in B
                else
                    DCM.M.pC(idx, idx) = 1/64;       % allow estimation
                end
            end
        end    

        % hrf
        % hrf_indices = 37:40;
        % 
        % % Loosen priors for HRF parameters
        % for idx = hrf_indices
        %     DCM.M.pC(idx, idx) = 1/16;  % or 0.5 if you want a bit tighter
        % end

        % transit brain
        DCM.M.pC(37, 37) = 1/16;
        DCM.M.pC(38, 38) = 1/16;
        DCM.M.pC(39, 39) = 1/64;
        DCM.M.pC(40, 40) = 1/64;
        
        %%% 41 and 42 are the epsilon and decay
        DCM.M.pC(41,41) = 1/16;
        DCM.M.pC(42,42) = 1/16;

        DCM = spm_dcm_estimate(DCM);
        F_values(s, m) = DCM.F;
        DCMs{s, m} = DCM;
    end
end

%% Model comparison

% === STEP 2: MODEL COMPARISON PER SUBJECT (FIXED EFFECT) ===
model_probs = zeros(n_subjects, n_models);
best_model_idx = zeros(n_subjects, 1);

for s = 1:n_subjects
    relF = F_values(s, :) - max(F_values(s, :));
    P = exp(relF); P = P / sum(P);
    model_probs(s, :) = P;
    [~, best_model_idx(s)] = max(P);
end


%% === Step 2: Report Best Model per Subject ===
fprintf('\n🧠 Best model per subject:\n');
for s = 1:n_subjects
    fprintf('Subject %02d → Model %d (P=%.2f)\n', s, best_model_idx(s), model_probs(s, best_model_idx(s)));
end

%% Extract params

A_post = cell(n_subjects, 1);
B_post = cell(n_subjects, 1);
C_post = cell(n_subjects, 1);

transit_all = zeros(n_subjects, size(DCMs{1,1}.Ep.transit, 1));
decay_all   = zeros(n_subjects, 1);
epsilon_all = zeros(n_subjects, 1);

for s = 1:n_subjects
    best_idx = best_model_idx(s);
    DCM = DCMs{s, best_idx};

    transit_all(s, :) = DCM.Ep.transit(:)';
    decay_all(s)      = DCM.Ep.decay;
    epsilon_all(s)    = DCM.Ep.epsilon;

    A_post{s} = DCM.Ep.A;  % A: [n x n]
    B_post{s} = DCM.Ep.B;  % B: [n x n x n_mods]
    C_post{s} = DCM.Ep.C;  % C: [n x n_inputs]
end




%%
% === STEP 3: BMA (Subject-wise) ===
BMAs = cell(n_subjects, 1);
for s = 1:n_subjects
    DCM_list = DCMs(s, :);
    BMAs{s} = spm_dcm_bma(DCM_list);
end


% bma = BMAs{1};
% Pp_B = compute_posterior_prob_B(bma);



B_params = cell(n_subjects, 1);
for s = 1:n_subjects
    B_params{s} = spm_vec(BMAs{s}.Ep.B);
end


%% GROUP LEVEL FULL MODEL


% Use model 8 or 4 = full model
DCMs_full = cell(n_subjects, 1);
for s = 1:n_subjects
    DCMs_full{s} = DCMs{s, 4};  % Model 8 or 4 = full B model
end

M.X = ones(n_subjects, 1);
PEB = spm_dcm_peb(DCMs_full, M,  {'A','B','C','H','transit','decay','epsilon'});

[PEB_bmr, BMR_models] = spm_dcm_peb_bmc(PEB);

%%
for i = 1:length(PEB.Pnames)
    mu = full(PEB_bmr.Ep(i));
    sigma = sqrt(full(PEB_bmr.Cp(i,i)));
    Pp = 1 - normcdf(0, abs(mu), sigma);
    fprintf('%s = %.3f (± %.3f), Pp=%.3f\n', PEB_bmr.Pnames{i}, mu, sigma, Pp);
end

%% Only on B


% Use model 8 = full model

M.X = ones(n_subjects, 1);
PEB_B = spm_dcm_peb(DCMs_full, M, {'B'});
%pC = spm_dcm_peb_priors(PEB.Pnames);


[PEB_bmr_B, BMR_models_B] = spm_dcm_peb_bmc(PEB_B);

for i = 1:length(PEB_B.Pnames)
    mu = full(PEB_B.Ep(i));
    sigma = sqrt(PEB_B.Cp(i,i));
    Pp = 1 - normcdf(0, abs(mu), sigma);
    fprintf('%s = %.3f (± %.3f), Pp=%.3f\n', PEB_B.Pnames{i}, mu, sigma, Pp);
end


%% Only on A


% Use model 8 = full model

M.X = ones(n_subjects, 1);
PEB = spm_dcm_peb(DCMs_full, M, {'A'});
%pC = spm_dcm_peb_priors(PEB.Pnames);


[PEB_bmr, BMR_models] = spm_dcm_peb_bmc(PEB);

for i = 1:length(PEB.Pnames)
    mu = full(PEB.Ep(i));
    sigma = sqrt(PEB.Cp(i,i));
    Pp = 1 - normcdf(0, abs(mu), sigma);
    fprintf('%s = %.3f (± %.3f), Pp=%.3f\n', PEB.Pnames{i}, mu, sigma, Pp);
end


%% HRF


M.X = ones(n_subjects, 1);
PEB_hrf = spm_dcm_peb(DCMs_full, M, {'H','transit','decay','epsilon'});
%pC = spm_dcm_peb_priors(PEB.Pnames);


[PEB_bmr_hrf, BMR_models_hrf] = spm_dcm_peb_bmc(PEB_hrf);

for i = 1:length(PEB_hrf.Pnames)
    mu = full(PEB_hrf.Ep(i));
    sigma = sqrt(PEB_hrf.Cp(i,i));
    Pp = 1 - normcdf(0, abs(mu), sigma);
    fprintf('%s = %.3f (± %.3f), Pp=%.3f\n', PEB_hrf.Pnames{i}, mu, sigma, Pp);
end
%%


for s = 1:n_subjects
    trans = DCMs_full{s}.Ep.transit;
    decay = DCMs_full{s}.Ep.decay;
    epsilon = DCMs_full{s}.Ep.epsilon;
    fprintf('Subject %d — transit: %.3f, decay: %.3f, epsilon: %.3f\n', ...
        s, full(trans), full(decay),full(epsilon));  % adjust indices if needed
end


%%%%%%%%%%%%%%
%% RESULTS
%%%%%%%%%%%%%%


%% SIMULATION after BMR

DCM_sub = DCMs_full{1};              % or whichever subject
DCM_sub.Ep = spm_vec(PEB.Ep);
DCM_sub.A = reshape(DCM_sub.Ep(1:n^2), n, n);  % if A is first block
DCM_sub.v  = size(DCM_sub.Y.y, 1);  % number of time points
DCM_sub.Y.dt = 1.55;                % or your TR

% vec_pE = spm_vec(DCM_sub.M.pE);      % full param vector
% vec_pE(PEB.Pind) = spm_vec(PEB.Ep);  % insert group Ep into subject priors
% sim_Ep = spm_unvec(vec_pE, DCM_sub.M.pE);

DCM_sub.M.delays = ones(1, size(DCM_sub.Y.y, 2));  % add this line to avoid error

% Ensure required fields
n = size(DCM_sub.Y.y, 2);
% DCM_sub.delays = ones(1, n);

% Simulate
% or Inf for noiseless
DCM_sim = spm_dcm_generate(DCM_sub);


% === 6. Plot
figure;
plot(DCM_sim.y, 'LineWidth', 1.2);
xlabel('Time (TRs)');
ylabel('Simulated BOLD signal');
legend(DCM_sub.Y.name, 'Location', 'Best');
title(sprintf('Simulated DCM BOLD (Subject %d)', subject_idx));
grid on;

%% HRF

dt = DCM_fit.Y.dt;
T = 32;  % duration in seconds

for r = 1:size(DCM_fit.Ep.transit,1)
    transit = DCM_fit.Ep.transit(r);
    decay   = DCM_fit.Ep.decay;
    epsilon = DCM_fit.Ep.epsilon;

    % Simple balloon model HRF (qualitative)
    hrf = spm_hrf(dt, [decay, transit, epsilon]);

    subplot(2,2,r)
    plot(0:dt:T, hrf(1:length(0:dt:T)), 'LineWidth', 1.5)
    title(sprintf('Region %d HRF', r))
    xlabel('Time (s)'), ylabel('Amplitude')
end