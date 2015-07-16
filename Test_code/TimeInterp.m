

%load MSLP_Aug_6_2012.mat;
%for ii=1:429
%    for jj=1:429
%        tmpx(1:8) = MSLP(ii,jj,1:8); 
%        MSLPx = interp(tmpx,3); 
%        pressure_data(ii,jj,1:24) = MSLPx(1:24);
%    end
%end


%load (evalin('base','windSpeedFilename'),'u','v','time','longitude','latitude'); %%% Don't think is working for dorun1

var1= zeros(429,429,1464);
%161 81 grid 1464-end up with
%429 429 -month
var2 = zeros(161,81,1464);

for ii=1:429

    for jj=1:429

        utmp = u(ii,jj,:);

        utmp = utmp(:);

        vtmp = v(ii,jj,:);

        vtmp = vtmp(:);

        var1(ii,jj,:) = interp1(1:3:1464,utmp,1:1464);

        var2(ii,jj,:) = interp1(1:3:1464,vtmp,1:1464);

    end

end