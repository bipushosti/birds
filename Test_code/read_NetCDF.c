/* This is part of the netCDF package.
Copyright 2006 University Corporation for Atmospheric Research/Unidata.
See COPYRIGHT file for conditions of use.

This is a simple example which reads a small dummy array, which was
written by simple_xy_wr.c. This is intended to illustrate the use
of the netCDF C API.

This program is part of the netCDF tutorial:
http://www.unidata.ucar.edu/software/netcdf/docs/netcdf-tutorial

Full documentation of the netCDF C API can be found at:
http://www.unidata.ucar.edu/software/netcdf/docs/netcdf-c

$Id: simple_xy_rd.c,v 1.9 2006/08/17 23:00:55 russ Exp $
*/
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netcdf.h>

/* This is the name of the data file we will read. */
#define FILE_NAME "../../Birds_data/output/MSLP/CFSR_NA-East_10km_MSLP_2008-08-01.nc"

/* We are reading 2D data, a 6 x 12 grid. */
#define NX 429
#define NY 429
#define NZ 8

#define NDIMS	3
#define NREC	8
/* Handle errors by printing an error message and exiting with a
* non-zero status. */

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}



//------------------------------------------------------------------------------------------------------------------//
					//Function Definitions//

int count_days(char *fileName);
char * convert_to_month(char* month,char* year);
char* yearMonth_lookupTable(char* month,char* year);
//------------------------------------------------------------------------------------------------------------------//
					//Functions//

int count_days(char *fileName)
{
	FILE * days_file;
	days_file = fopen(fileName,"r");
	int total_files,tmp;
	total_files = 0;
	printf("Filename %s\n",fileName);
	do{
		tmp = fgetc(days_file);
		if(tmp == '\n') total_files++;
	}while(tmp != EOF);

	fclose(days_file);
	return total_files;
}

char * convert_to_month(char* month,char* year)
{
	char* ret_month;
	if((strcmp(month,"8")==0)||(strcmp(month,"08")==0)){
		ret_month = "August";
	}else if((strcmp(month,"9")==0)||(strcmp(month,"09")==0)){
		ret_month = "September";
	}else if(strcmp(month,"10")==0){
		ret_month = "September";
	}else if(strcmp(month,"11")==0){
		ret_month = "September";
	}else{
		printf("\n\t\tIncorrect month used\n\t\tUse between August-November inclusive; only use respective numbers eg August = 08 or 8\n");
		return 0;
	}
}

char* yearMonth_lookupTable(char* month,char* year)
{
	char* filename,*month_full;
	
	if(strcmp(year,"2008")){
		month_full = convert_to_month(month,year);
	}else if(strcmp(year,"2009")){
		month_full = convert_to_month(month,year);
	}else if(strcmp(year,"2010")){
		month_full = convert_to_month(month,year);
	}else if(strcmp(year,"2012")){
		month_full = convert_to_month(month,year);
	}else{
		printf("\n\t\tIncorrect year used\n\t\tUse between 2008-2013 inclusive; only use full format eg 2013,2012\n");
		return 0;
	}
		


}
//------------------------------------------------------------------------------------------------------------------//


int main(int argc,char* argv[])
{
	char *command[]= {"/bin/bash","-c","/bin/ls ~/Documents/Birds_Full/Birds_data/output/MSLP/CFSR_NA-East_10km_MSLP_2008-08-*.nc > InputFiles_August2008.txt",NULL};
	execvp(command[0],command);
	if(argc == 1){
		printf("\n\tNot enough arguments\n\tUsage:\tExecutable Year StartMonth EndMonth\n");
		return 0;
	}
	printf("Days: %d\n",count_days("InputFiles_August2008.txt"));
/* This will be the netCDF ID for the file and data variable. */
	int ncid, varid,i,rec,mslp_varid;

	float XLAT[NX][NY],XLONG[NX][NY],MSLP_tmp[NX][NY],*MSLP;
	size_t start[NDIMS],count[NDIMS];
	/* Loop indexes, and error handling. */
	int x, y, retval;

	MSLP = (float*)malloc(NX * NY * NZ * sizeof(float));


	/* Open the file. NC_NOWRITE tells netCDF we want read-only access
	* to the file.*/
	if ((retval = nc_open(FILE_NAME, NC_NOWRITE, &ncid)))
	ERR(retval);

	//---------------------------------Read XLAT----------------------------------//
	/* Get the varid of the data variable, based on its name. */
	if ((retval = nc_inq_varid(ncid, "XLAT", &varid)))
	ERR(retval);

	/* Read the data. */
	if ((retval = nc_get_var_float(ncid, varid, &XLAT[0][0])))
	ERR(retval);

	//---------------------------------Read XLONG----------------------------------//

	/* Get the varid of the data variable, based on its name. */
	if ((retval = nc_inq_varid(ncid, "XLONG", &varid)))
	ERR(retval);

	/* Read the data. */
	if ((retval = nc_get_var_float(ncid, varid, &XLONG[0][0])))
	ERR(retval);


	//---------------------------------Read MSLP----------------------------------//

	/* Get the varid of the data variable, based on its name. */
	if ((retval = nc_inq_varid(ncid, "MSLP", &mslp_varid))){
		ERR(retval);
	}

	/* Read the data. No idea why this is needed*/
	count[0] = 1;
	count[1] = NX;
	count[2] = NY;
	start[1] = 0;
	start[2] = 0;
	//----------------------------------------------------------------------------//
	//Have to read one record at a time//
	int j,k;
	for(rec = 0;rec < NREC;rec++){
		start[0] = rec;
		if((retval = nc_get_vara_float(ncid,mslp_varid,start,count,&MSLP_tmp[0][0]))){
			ERR(retval);
		}
//		printf("Size = %ld\n",sizeof(MSLP)/sizeof(float));
		//printf("%f\n",MSLP_tmp[0][0]);

//		return 0;

//		for(k=0;k<NZ;k++){
			for(i=0;i<NY;i++){
				for(j=0;j<NX;j++){
					//This order gives the correct result
					//printf("%f ",MSLP_tmp[j][i]);
					MSLP[i*NX*NZ+j*NZ+rec] = MSLP_tmp[j][i];
					//printf("%f ",MSLP[i*NX*NZ+j*NZ+rec]);
				}
				//printf("\n");
			}
//		}
		
//		return 0;
	}

	if(retval = nc_close(ncid)){
		ERR(retval);
	}

	
	
	free(MSLP);
	return 0;
}














