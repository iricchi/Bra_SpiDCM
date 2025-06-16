function [A, B, C] = get_dcm_model(model_name)

% Node indices
L_S1 = 1; R_S1 = 2; L_DH = 3; R_DH = 4;
n = 4;

% Initialize
A = zeros(n, n);
B = zeros(n, n, 1);
C = zeros(n, 1);
C(R_DH) = 1;

% Define B based on model name
if strcmp(model_name, 'C4_only_B0_null')
    % No B modulation

elseif strcmp(model_name, 'C4_only_B41_ascend')
    B(L_S1, R_DH, 1) = 1;

elseif strcmp(model_name, 'C4_only_B_ascdesc')
    B(L_S1, R_DH, 1) = 1;
    B(R_DH, L_S1, 1) = 1;

elseif strcmp(model_name, 'C4_only_B_spinal')
    B(L_DH, R_DH, 1) = 1;
    B(R_DH, L_DH, 1) = 1;

elseif strcmp(model_name, 'C4_only_B43_spinal')
    B(L_DH, R_DH, 1) = 1;

elseif strcmp(model_name, 'C4_only_B_cortical')
    B(L_S1, R_S1, 1) = 1;
    B(R_S1, L_S1, 1) = 1;

elseif strcmp(model_name, 'C4_only_B21_cortical')
    B(R_S1, L_S1, 1) = 1;

elseif strcmp(model_name, 'C4_only_B_full')
    B(L_S1, R_DH, 1) = 1;
    B(R_DH, L_S1, 1) = 1;
    B(L_S1, R_S1, 1) = 1;
    B(R_S1, L_S1, 1) = 1;
    B(L_DH, R_DH, 1) = 1;
    B(R_DH, L_DH, 1) = 1;

else
    error(['Unknown model name: ' model_name])
end

end