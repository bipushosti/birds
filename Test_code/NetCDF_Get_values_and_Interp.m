function NetCDF_Get_values_and_Interp()
% This function takes in netcdf files from the Birds_data folder and
% interpolates them all. Also outputs files with their respective variables
% and year


    %Total number of days in the 4 months from August to November
    days = 122;
    
    %k is the total number of var files   
    for y = 1:6
        
        
        if y == 1
            year = '2008';
        elseif y == 2
            year = '2009';
        elseif y == 3
            year = '2010';
        elseif y == 4
            year = '2011';   
        elseif y == 5
            year = '2012';
        elseif y == 6
            year = '2013';
        end
            
        %k is the total number of var files
        %for k=1:10
        for k=1:7

            %j is number of days
            %for j=1:10 
            temp2 = [];
            tmpx = [];
            temp2x = [];
            output = [];
            out = zeros(429,429,8*days);

            month_days = 0;
            prev_monthDays = 0;
            l=1;
            
            
            for j=1:days
                %prev_monthDays = month_days + prev_monthDays;
                if j <=31
                    month = '08';
                    month_days = 31; 
                    val = j;
                elseif j>31 && j<=61
                    month = '09';
                    month_days = 30;
                    val = j -31;
                elseif j>61 && j<=92
                    month = '10';
                    month_days = 31;
                    val = j - (31 + 30);
                elseif j>92
                    month = '11';
                    month_days = 30;
                    val = j - (31+30+31);
                end
           
                data=[];
                %if j<10, add an extra 0 with the number
                %if 9 then 09  
                
                %val = j -prev_monthDays;
                disp(val);
                if (val)<10
                    num = strcat('0',num2str(val));
                else
                    num = num2str(val);
                end
                disp(num);

                if k == 1
                    dir = 'U750';
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
                 elseif k == 7
                    dir = 'V750';
                 %elseif k == 8
                  %  dir = 'U925';
                % elseif k == 9
                 %   dir = 'V925';
                 %elseif k == 10
                 %   dir = 'MSLP';
                end

                filename = strcat('~/Documents/Birds_Full/Birds_data/output/',dir,'/CFSR_NA-East_10km_',dir,'_',year,'-',month,'-',num,'.nc');
                outputFile = strcat('~/Documents/Birds_Full/Birds_data/output/',dir,'_',year,'.txt');
                data = importNetCDF(filename,dir);
                %assignin('base','data_check',data);
                
                %The netCDF has 24 hours for every 3 hours meaning there are 8
                %time steps.Taking all 24 hours or 8 timesteps.
                for i= 1:8 
                    temp2 = data(:,:,i);
                    temp2 = temp2';
                    out(:,:,l)=temp2;
                    l = l + 1;

                   %dlmwrite(outputFile,temp2,'-append');
                    %dlmwrite(outputFile,temp2,'-append');
                    %dlmwrite(outputFile,temp2,'-append');
                end
                %assignin('base','temp2',temp2);      
                fclose('all');
            end
            disp(month_days);
            assignin('base','MSLP',out);
            disp(size(temp2));
          
            variable = zeros(429,429,days*24);
            for ii=1:429
                for jj=1:429
                    outTemp = out(ii,jj,:);
                    outTemp = outTemp(:);
                    variable(ii,jj,:) = interp1(1:3:days*24,outTemp,1:days*24);              
                end
            end
        
       
             for ii = 1:24*days
                dlmwrite(outputFile,variable(:,:,ii),'-append');
             end
        end
    end  
end


