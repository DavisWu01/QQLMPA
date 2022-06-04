%_________________________________________________________________________
%  A quasi-opposition learning and Q-learning based marine predatorsalgorithm for global continuous optimization problems
%  programming:: Yulu Wu
%_________________________________________________________________________

% This function containts full information and implementations of the benchmark 
% functions in Table 1, Table 2, and Table 3 in the paper

% lb is the lower bound: lb=[lb_1,lb_2,...,lb_d]
% up is the uppper bound: ub=[ub_1,ub_2,...,ub_d]
% dim is the number of variables (dimension of the problem)

function [lb,ub,dim,fobj] = Get_Functions_details(F)


switch F
    case 'F5'
        fobj = @F5;
        lb=-1.28;
        ub=1.28;
        dim=50;
end
% F5

function o = F5(x)
dim=size(x,2);
o=sum([1:dim].*(x.^4))+rand;
end

end

