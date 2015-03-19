

%function output= LoadNetCDF (nc_filename)

    nc_filename = '~/Documents/Birds_Full/Birds_data/output/MSLP/CFSR_NA-East_10km_MSLP_2008-08-01.nc';
    ncid=netcdf.open(nc_filename, 'nowrite');

    [numdims, numvars, numglobalatts, unlimdimID] = netcdf.inq(ncid);

    disp(' '),disp(' '),disp(' ')
    disp('________________________________________________________')
    disp('^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~')
    disp(['VARIABLES CONTAINED IN THE netCDF FILE: ' nc_filename ])
    disp(' ')
    for i = 0:numvars-1
        [varname, xtype, dimids, numatts] = netcdf.inqVar(ncid,i);
        disp(['--------------------< ' varname ' >---------------------'])
        flag = 0;
        for j = 0:numatts - 1
            attname1 = netcdf.inqAttName(ncid,i,j);
            attname2 = netcdf.getAtt(ncid,i,attname1);
            disp([attname1 ':  ' num2str(attname2)])
            if strmatch('add_offset',attname1);
                offset = attname2;
            end
            if strmatch('scale_factor',attname1)
                scale = attname2;
                flag = 1;
            end        
        end
        disp(' ')

        if flag
            eval([varname '= double(double(netcdf.getVar(ncid,i))*scale + offset);'])
            %assignin(ws,'MSLP',eval([varname '= double(double(netcdf.getVar(ncid,i))*scale + offset);']));
            %disp(eval([varname '= double(double(netcdf.getVar(ncid,i))*scale + offset);'])
        else
            eval([varname '= double(netcdf.getVar(ncid,i));'])   
        end
    end
    disp('^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~')
    disp('________________________________________________________')
    disp(' '),disp(' ')
%end