function data =  importNetCDF( filename,var)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
    ncid = netcdf.open(filename,'NC_NOWRITE');
    varid = netcdf.inqVarID(ncid,var);
    var_name = netcdf.inqVar(ncid,varid);
    disp(var_name);
    
    data = netcdf.getVar(ncid,varid);
   % assignin('base','data2',data);

end

