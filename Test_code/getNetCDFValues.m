function getNetCDFValues(days)
%UNTITLED2 Summary of this function goes here
    
    %clear all;
    
    %k is the total number of var files
    %for k = 1:6
    for k=1:6
        
        %j is number of days
        %for j=1:10 
        temp2 = [];
        tmpx = [];
        temp2x = [];
        output = [];
        out = zeros(429,429,2*days);
        
        
        
        l=1;
        for j=1:days
            data=[];
            %if j<10, add an extra 0 with the number
            %if 9 then 09
            if j<10
                num = strcat('0',num2str(j));
            else
                num = num2str(j);
            end
            disp(num);

            if k == 1
                dir = 'MSLP';
            elseif k==2
                dir = 'V850';
            elseif k == 3
                dir = 'U850';
            elseif k == 4
                dir = 'PRCP';
            elseif k == 5
                dir = 'U10';
            elseif k == 6
                dir = 'V10';
            end

            filename = strcat('~/Documents/Birds_Full/Birds_data/output/',dir,'/CFSR_NA-East_10km_',dir,'_2011-09-',num,'.nc');
            outputFile = strcat('~/Documents/Birds_Full/Birds_data/output/',dir,'_',days,'days_Sept_2011.txt');
            data = importNetCDF(filename,dir);
            %assignin('base','data_check',data);
            %The netCDF has 24 hours for every 3 hours meaning there are 8
            %time steps. Taking only the first 2 since we are looking for 
            %just the first 6 hours in the day(the time starts at 0 UTC or 
            %7 pm eastern)     
            for i= 1:2 
                temp2 = data(:,:,i);
                temp2 = temp2';
                out(:,:,l)=temp2;
                l = l + 1;

               %dlmwrite(outputFile,temp2,'-append');
                %dlmwrite(outputFile,temp2,'-append');
                %dlmwrite(outputFile,temp2,'-append');
            end
            %assignin('base','temp2',temp2);      

         end
        disp(size(temp2));
        for ii=1:429
            for jj=1:429
                tmpx(1:2*days) = out(ii,jj,1:2*days); 
                temp2x = interp(tmpx,3); 
                output(ii,jj,1:6*days) = temp2x(1:6*days);
            end
        end
        
         for ii = 1:6*days
            dlmwrite(outputFile,output(:,:,ii),'-append');
         end
    end
end


