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
        
    case 'P11'
        fobj=@P11;
        lb=[2.6,0.7,17,7.3,0,2.9,5];
        ub=[3.6,0.8,28,10,8.3,3.9,5.5];
        dim=7;
end
% F5

function o = F5(x)
dim=size(x,2);
o=sum([1:dim].*(x.^4))+rand;
end

%Speeder reducer design
function f=P11(x)
g_1=27/(x(1)*(x(2)^2)*x(3))-1;
g_2=397.5/(x(1)*(x(3)^2)*x(2))-1;
g_3=1.93*x(4)^3/(x(1)*x(3)*(x(6)^4))-1;
g_4=1.93*x(4)^3/(x(1)*x(3)*(x(7)^4))-1;
g_5=(1/(100*x(6)^3))*((745*x(4)/(x(2)*x(3)))^2+16.9*(10^6))^0.5-1;
g_6=(1/(85*(x(7)^3)))*(((745*x(4))/(x(2)*x(3)))^2+157.5*(10^6))^0.5-1;
g_7=(x(2)*x(3))/40-1;
g_8=((5*x(2))/x(1))-1;
g_9=(x(1)/(12*x(2)))-1;
g_10=((1.5*x(6)+1.9)/x(4))-1;
g_11=((1.1*x(7)+1.9)/x(5))-1;
if g_1<=0 && g_2<=0 && g_3<=0 && g_4<=0 && g_5<=0 && g_6<=0 && g_7<=0 && g_8<=0 && g_9<=0 && g_10<=0 && g_11<=0
    f=0.7854*x(1)*(x(2)^2)*(3.3333*(x(3)^2)+14.9334*x(3)-43.0934)-1.508*x(1)*(x(6)^2+x(7)^2)+7.4777*(x(6)^3+x(7)^3)+0.7854*(x(1)*x(6)^2+x(5)*x(7)^2);
else
    f=10^5;
end
end
end


