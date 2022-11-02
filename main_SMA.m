clear;clc
close all
tic
func_num = 1; % Function number 1:10
Algorithm_name = 'SMA';
lb = -100;  ub = 100;  dim = 20;
Pop_size = 100; % Number of search agents
Max_iter = 10000; % Maximum number of iterations
optimum = [100,1100,700,1900,1700,1600,2100,2200,2400,2500]; % The optimal value of the function
runs = 30; % Number of independent runs
for i = 1:runs
    [Destination_fitness,bestPositions,Convergence_curve] = SMA(Pop_size,Max_iter,lb,ub,dim,func_num);
    fitness(i,:) = Destination_fitness;
    Positions(i,:) = bestPositions;
    curve(i,:) = Convergence_curve;
end
Time = toc; % Record the execution time of the algorithm
fitness = fitness-optimum(func_num);
[~,Index] = sort(fitness);
bestPosition = Positions(Index(1),:);
curve = curve-optimum(func_num);
curve(curve<1e-8) = 0;
Convergence_curve = sum(curve,1)/runs;
Convergence_curve = Convergence_curve(1,1:10:end);
% Display solution results
disp([Algorithm_name,': function ',num2str(func_num)])
disp(['Mean fitness: ',num2str(Convergence_curve(end))])
disp(['Execution time: ',num2str(Time)])