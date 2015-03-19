function untitled2()
%UNTITLED2 Summary of this function goes here
    
    %clear all;
    
    %k is the total number of var files
    %for k = 1:6
    for k=1:6
        
        %j is number of days
        %for j=1:10 
         for j=1:30
             
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
            
            filename = strcat('~/Documents/Birds_Full/Birds_data/output/',dir,'/CFSR_NA-East_10km_',dir,'_2009-08-',num,'.nc');
            data = importNetCDF(filename,dir);
            %assignin('base','data_check',data);
            %The netCDF has 24 hours for every 3 hours meaning there are 8
            %time steps. Taking only the first 2 since we are looking for 
            %just the first 6 hours in the day(the time starts at 0 UTC or 
            %7 pm eastern)     
            for i= 1:2
                
                
                
                
               % if k==1
               %  temp2 = MSLP(:,:,i);
               % elseif k==2
               %     temp2 = V850(:,:,i);
               % elseif k==3
               %     temp2 = U850(:,:,i);
               % elseif k==4
               %     temp2 = PRCP(:,:,i);
               % elseif k==5
               %     temp2 = U10(:,:,i);
               % elseif k==6
               %     temp2 = V10(:,:,i);
               % end
                
               
                
                temp2 = data(:,:,i);
                temp2 = temp2';
                
                outputFile = strcat('full_',dir,'_text_first10_2009.txt');
                dlmwrite(outputFile,temp2,'-append');
                dlmwrite(outputFile,temp2,'-append');
                dlmwrite(outputFile,temp2,'-append');
            end
            %assignin('base','temp2',temp2);
        end
    end
end


