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
#include <stdlib.h>
#include <stdio.h>
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

int main()
{
/* This will be the netCDF ID for the file and data variable. */
	int ncid, varid,i,rec,mslp_varid;

	float XLAT[NX][NY],XLONG[NX][NY],MSLP[NX][NY];
	size_t start[NDIMS],count[NDIMS];
	/* Loop indexes, and error handling. */
	int x, y, retval;

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

	/* Read the data. */
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
		if((retval = nc_get_vara_float(ncid,mslp_varid,start,count,&MSLP[0][0]))){
			ERR(retval);
		}

		for(i=0;i<NX;i++){
			for(j=0;j<NY;j++){
				printf("%f ",MSLP[i][j]);
			}
			printf("\n");
		}
	}

	if(retval = nc_close(ncid)){
		ERR(retval);
	}

	
	
	
	return 0;
}














