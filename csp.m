
% Function 'csp' trains a Common Spatial Pattern (CSP) filter bank.       %
%                                                                         %
%   Input parameters:                                                     %
%       - X1:   Signal for the positive class, dimensions [C x T], where  %
%               C is the no. channels and T the no. samples.              %
%       - X2:   Signal for the negative class, dimensions [C x T], where  %
%               C is the no. channels and T the no. samples.              %
%                                                                         %
%   Output variables:                                                     %
%       - W:        Filter matrix (mixing matrix, forward model). Note that
%                   the columns of W are the spatial filters.             %
%       - lambda:   Eigenvalues of each filter.                           %
%       - A:        Demixing matrix (backward model).                     %


    % Error detection
    if nargin < 2, error('Not enough parameters.'); end
    if length(size(X1))~=2 || length(size(X2))~=2
        error('The size of trial signals must be [C x T]');
    end
    
    % Compute the covariance matrix of each class
    S1 = cov(X1');   % S1~[C x C]
    S2 = cov(X2');   % S2~[C x C]

    % Solve the eigenvalue problem S1·W = l·S2·W
    [W,L] = eig(S1, S1 + S2);   % Mixing matrix W (spatial filters are columns)
    lambda = diag(L);           % Eigenvalues
    A = (inv(W))';              % Demixing matrix
    
    % Further notes:
    %   - CSP filtered signal is computed as: X_csp = W'*X;
end