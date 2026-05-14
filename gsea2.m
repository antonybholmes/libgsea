function [es,nes,pv,ledge] = gsea2(rk,rs,gs1,gs2,np,w)
%GSEA2 Gene Set Enrichment Analysis (GSEA) for 2 complementary gene sets
%
% ---INPUT---
%  rk    : ranked gene list
%  rs    : ranked score
%  gs1   : positive gene set
%  gs2   : negative gene set
%  np    : number of permutation
%  w     : weight
% ---OUTPUT---
%  es    : enrichment score
%  nes   : normalized enrichment score
%  pv    : p-value from the permutation test
%  ledge : leading edge
%
% Author: Wei Keat Lim
%
% References:
% Lim WK, Lyashenko E, & Califano A. (2009) Master regulators used as  
%    breast cancer metastasis classifier. Pac Symp Biocomput, 14, 504-515.
% Carro MS*, Lim WK*, Alvarez MJ*, et al. (2010) The transcriptional 
%    network for mesenchymal transformation of brain tumours. Nature, 
%    463(7279), 318-325.
%


% check input format
if size(rk,2)>size(rk,1), rk = rk'; end
if size(rs,2)>size(rs,1), rs = rs'; end
nes = NaN;
pv  = NaN;

% combine ranked list and score
% need to be vertical lists
rkc      = [rk;rk];
rsc      = [rs;-rs];
pn       = [ones(size(rs));-ones(size(rs))];
[tmp,ix] = sort(rsc,'descend');
rkc      = rkc(ix);
rsc      = rsc(ix);
pn       = pn(ix);

% check overlap of gene sets and ranked list 
isgs = zeros(size(rkc));
for i=1:length(rkc)
    if ((pn(i)>0) & strmatch(rkc(i),gs1,'exact')) | ...
            ((pn(i)<0) & strmatch(rkc(i),gs2,'exact'))
        isgs(i) = 1;
    end
end

% compute ES
score_hit   = cumsum((abs(rsc.*isgs)).^w);  
score_hit   = score_hit/score_hit(end);
score_miss  = cumsum(1-isgs);
score_miss  = score_miss/score_miss(end);
es_all      = score_hit - score_miss;
es          = max(es_all) + min(es_all);

% identify leading edge
isen = zeros(size(es_all));
if es<0
    ixpk = find(es_all==min(es_all));
    isen(ixpk:end) = 1;
    ledge = rkc((isen==1)&(isgs==1));
    ledge = ledge(end:-1:1);
else
    ixpk = find(es_all==max(es_all));
    isen(1:ixpk) = 1;
    ledge = rkc((isen==1)&(isgs==1));
end

% compute p-value
if np>0
    bg.es = zeros(np,1);
    for i=1:np
        bg.isgs  = isgs(randperm(length(isgs)));  
        bg.hit   = cumsum((abs(rsc.*bg.isgs)).^w);
        bg.hit   = bg.hit/bg.hit(end);
        bg.miss  = cumsum(1-bg.isgs);
        bg.miss  = bg.miss/bg.miss(end);
        bg.all   = bg.hit - bg.miss;
        bg.es(i) = max(bg.all) + min(bg.all);
    end
    if es<0
        pv  = sum(bg.es<=es)/np;
        nes = es/abs(mean(bg.es(bg.es<0)));
    else
        pv  = sum(bg.es>=es)/np;
        nes = es/abs(mean(bg.es(bg.es>0)));
    end
end

