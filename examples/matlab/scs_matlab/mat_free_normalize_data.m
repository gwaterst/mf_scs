function [data, w] = normalize_data(data, K, scale, w)
[m,n] = size(data.A);

MIN_SCALE = 1e-3;
MAX_SCALE = 1e3;
minRowScale = MIN_SCALE * sqrt(n);
maxRowScale = MAX_SCALE * sqrt(n);
minColScale = MIN_SCALE * sqrt(m);
maxColScale = MAX_SCALE * sqrt(m);
SAMPLES = 200;
D = ones(m,1);
E = ones(n,1);

alpha = n/m;
beta = 1;
NN = 1; % NN = 1, other choices bad
for j=1:NN
    %% D scale:
%     Dt = rand_rnsAE(data.A, SAMPLES) + data.b.^2;
     Dt = rand_rnsAE(data.A, SAMPLES);
%     Dt = twonorms(data.A(1:K.f,:)')';
    idx = K.f;
%     Dt = [Dt;twonorms(data.A(idx+1:idx+K.l,:)')'];
    idx = idx + K.l;
    for i=1:length(K.q)
        if (K.q(i) > 0)
%             nmA = mean(twonorms(data.A(idx+1:idx+K.q(i),:)'));
%             Dt = [Dt;nmA*ones(K.q(i),1)];
            Dt(idx+1:idx+K.q(i)) = sum(Dt(idx+1:idx+K.q(i)))/K.q(i);
            idx = idx + K.q(i);
        end
    end
    for i=1:length(K.s)
        if (K.s(i) > 0)
%             nmA = mean(twonorms(data.A(idx+1:idx+getSdConeSize(K.s(i)),:)'));
%             Dt = [Dt;nmA*ones(getSdConeSize(K.s(i)),1)];
            Dt(idx+1:idx+K.s(i)) = sum(Dt(idx+1:idx+K.s(i)))/K.s(i);
            idx = idx + getSdConeSize(K.s(i));
        end
    end
    for i=1:K.ep
%         nmA = mean(twonorms(data.A(idx+1:idx+3,:)'));
%         Dt = [Dt;nmA*ones(3,1)];
        Dt(idx+1:idx+3) = sum(Dt(idx+1:idx+3))/3;
        idx = idx + 3;
    end
    for i=1:K.ed
%         nmA = mean(twonorms(data.A(idx+1:idx+3,:)'));
%         Dt = [Dt;nmA*ones(3,1)];
        Dt(idx+1:idx+3) = sum(Dt(idx+1:idx+3))/3;
        idx = idx + 3;
    end
    for i=1:length(K.p)
%         nmA = mean(twonorms(data.A(idx+1:idx+3,:)'));
%         Dt = [Dt;nmA*ones(3,1)];
        Dt(idx+1:idx+3) = sum(Dt(idx+1:idx+3))/3;
        idx = idx + 3;
    end
    % TODO invert
    Dt = sqrt(Dt)/alpha;
    
    Dt(Dt < minRowScale) = 1; % TODO change this?
    Dt(Dt > maxRowScale) = maxRowScale;
    data.A = sparse(diag(1./Dt))*data.A;
    
    %% E Scale
%     Et = twonorms(data.A)';
%     Et = rand_rnsATD(data.A, SAMPLES) + data.c.^2/max(norm(data.c), MIN_SCALE);
    Et = rand_rnsATD(data.A, SAMPLES);
    Et = sqrt(Et)/beta;
    rns = Et;
    
    Et(Et < minColScale) = 1; % TODO change this?
    Et(Et > maxColScale) = maxColScale;
    data.A = data.A*sparse(diag(1./Et));
    
    %%
    D = D.*Dt;
    E = E.*Et;
end

% nmrowA = mean(twonorms(data.A'));
% nmcolA = mean(twonorms(data.A));
% TODO somehow skip this?
nmrowA = mean(sqrt(rand_rnsAE(data.A, SAMPLES)));
nmcolA = mean(rns./Et);
% nmrowA = 1;
% nmcolA = 1;
data.A = data.A*scale;

data.b = data.b./D;
sc_b = nmcolA/ max(norm(data.b), MIN_SCALE);
data.b = data.b * sc_b * scale;

data.c = data.c./E;
sc_c = nmrowA/max(norm(data.c), MIN_SCALE);
data.c = data.c * sc_c * scale;

w.D = D;
w.E = E;
w.sc_b = sc_b;
w.sc_c = sc_c;

    function twoNorms = twonorms(A)
        twoNorms = sqrt(sum(A.^2,1));
    end

    function rnsAE = rand_rnsAE(A, samples)
        rnsAE = zeros(m,1);
        for q=1:samples
            s = 2*binornd(1,0.5,n,1) - 1;
            rnsAE = rnsAE + (A*s).^2;
        end
        rnsAE = rnsAE/samples;
    end

    function rnsATD = rand_rnsATD(A, samples)
        rnsATD = zeros(n,1);
        for q=1:samples
            s = 2*binornd(1,0.5,m,1) - 1;
            rnsATD = rnsATD + (A'*s).^2;
        end
        rnsATD = rnsATD/samples;
    end
end