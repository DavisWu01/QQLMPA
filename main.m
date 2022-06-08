%_________________________________________________________________________
%  A quasi-opposition learning and Q-learning based marine predators algorithm for global continuous optimization problems
%  programming: Yulu Wu
%_________________________________________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% Max_iteration = maximum number of iterations
% SearchAgents_no = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n


clear all
clc
format long
SearchAgents_no=25; % Number of search agents

state_num=3;
action_num=3;
t=0;
Function_name='F5'; %including test function 'F5' and the Speed reducer design 'P11'
NUM=30;

   
Max_iteration=500; % Maximum number of iterations

[lb,ub,dim,fobj]=Get_Functions_details(Function_name);





for k=1:NUM
   [QQLMPA_Best_score(1,k),QQLMPA_Best_pos,QQLMPA_Convergence_curve(k,:),D]=QQLMPA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj,state_num,action_num);
   t=t+1;
   fprintf(['the number of iteration: ',num2str(t),'\n']);
end
QQLMPA_Convergence_curve=mean(QQLMPA_Convergence_curve,1);
QQLMPA_Best=min(QQLMPA_Best_score);
QQLMPA_Avg=mean(QQLMPA_Best_score);
QQLMPA_Std=std(QQLMPA_Best_score);

semilogy(QQLMPA_Convergence_curve,'Color',[30,43,113]/255,'LineWidth',2)

title(Function_name)
xlabel('Iteration');
ylabel('Best score obtained so far');
legend('QQLMPA')
display(['Avg score of QQLMPA :', num2str(QQLMPA_Avg,4)]);
disp(sprintf('--------------------------------------'));
