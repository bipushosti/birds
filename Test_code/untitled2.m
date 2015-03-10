function untitled2()
%UNTITLED2 Summary of this function goes here
    
    clear all;
    
    for k = 1:6
        for j=1:10 
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
            
            filename = strcat('~/Documents/Birds/output/',dir,'/CFSR_NA-East_10km_',dir,'_2009-08-',num,'.nc');
            importNetCDF(filename,dir);

            for i= 1:2
                
                if k==1
                    temp2 = MSLP(:,:,i);
                elseif k==2
                    temp2 = V850(:,:,i);
                elseif k==3
                    temp2 = U850(:,:,i);
                elseif k==4
                    temp2 = PRCP(:,:,i);
                elseif k==5
                    temp2 = U10(:,:,i);
                elseif k==6
                    temp2 = V10(:,:,i);
                end
                
                temp2 = temp2';
                outputFile = strcat('full_',dir,'_text_first10_2009.txt');
                dlmwrite(outputFile,temp2,'-append');
                dlmwrite(outputFile,temp2,'-append');
                dlmwrite(outputFile,temp2,'-append');
            end
            %assignin('caller','val',val);
        end
    end
end


