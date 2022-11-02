function [Destination_fitness,bestPositions,Convergence_curve] = SMA(N,Max_iter,lb,ub,dim,func_num)
% Max_iter: Maximum iterations, N: Populatoin size, Convergence_curve: Convergence curve
rand('seed',sum(100 * clock)); % Random number seed
z = 0.03; % Adjustable parameter
% Initialize the position of slime mould
lb = ones(1,dim).*lb; % Lower boundary 
ub = ones(1,dim).*ub; % Upper boundary
X = initialization(N,dim,ub,lb); % It can be downloaded from https://github.com/Shihong-Yin
bestPositions = zeros(1,dim); % Optimal food source location
Destination_fitness = inf; % Change this to -inf for maximization problems
weight = ones(N,dim); % Fitness weight of each slime mould
Convergence_curve = zeros(1,Max_iter);
% Main loop
for it = 1:Max_iter
    % Check the boundary and calculate the fitness
    FU = X>ub;  FL = X<lb;  X = (X.*(~(FU+FL)))+ub.*FU+lb.*FL;
    PopFitness = cec21_bias_shift_rot_func(X',func_num)'; % https://github.com/Shihong-Yin
    % Sort the fitness thus update the bF and wF
    [SmellOrder,SmellIndex] = sort(PopFitness); % Eq.(2.6)
    bestFitness = SmellOrder(1);
    worstFitness = SmellOrder(N);
    S = bestFitness-worstFitness+eps; % Plus eps to avoid denominator zero
    % Calculate the fitness weight of each slime mould
    for i = 1:N
        if i <= N/2 % Eq.(2.5)
            weight(SmellIndex(i),:) = 1+rand(1,dim)*log10((bestFitness-SmellOrder(i))/S+1);
        else
            weight(SmellIndex(i),:) = 1-rand(1,dim)*log10((bestFitness-SmellOrder(i))/S+1);
        end
    end
    % Update the best position and destination fitness
    if bestFitness < Destination_fitness
        bestPositions = X(SmellIndex(1),:);
        Destination_fitness = bestFitness;
    end
    a = atanh(-(it/Max_iter)+1); % Eq.(2.4)
    vb = unifrnd(-a,a,N,dim); % Eq.(2.3)
    b = 1-it/Max_iter;
    vc = unifrnd(-b,b,N,dim);
    p = tanh(abs(PopFitness-Destination_fitness)); % Eq.(2.2)
    r = rand(N,dim);
    A = randi([1,N],N,dim); % Two positions randomly selected from population
    B = randi([1,N],N,dim);
    % Update the Position of search agents
    for i = 1:N
        if rand < z % Eq.(2.7)
            X(i,:) = (ub-lb).*rand(1,dim)+lb; % The original code is (ub-lb)*rand+lb;
        else
            for j = 1:dim
                if r(i,j) < p(i) % Eq.(2.1)
                    X(i,j) = bestPositions(j)+vb(i,j)*(weight(i,j)*X(A(i,j),j)-X(B(i,j),j));
                else
                    X(i,j) = vc(i,j)*X(i,j);
                end
            end
        end
    end
    Convergence_curve(it) = Destination_fitness;
end
end