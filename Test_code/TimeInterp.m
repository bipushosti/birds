

load MSLP_Aug_6_2012.mat;




for ii=1:429
    for jj=1:429
        tmpx(1:8) = MSLP(ii,jj,1:8); 
        MSLPx = interp(tmpx,3); 
        pressure_data(ii,jj,1:24) = MSLPx(1:24);
    end
end


%for i=1:8
%    a = MSLP(:,:,i);
%    c = a.';
%    dlmwrite('MSLP_flipped.txt',c,'-append')
%end
