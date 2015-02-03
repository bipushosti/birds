
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <netcdf.h>
#include <stdlib.h>

#define NDIMS		4	//Number of dimensions
#define NETCDF_FILE	"ERA_Sept_1980_925pl_NA.nc"
#define LAT_SIZE	95
#define LONG_SIZE	55
#define TIMESTEPS	120
#define NUM_RECORDS	120

/* These are used to calculate the values we expect to find. */
#define SAMPLE_U 1.3
#define SAMPLE_V 1.4
#define START_LAT 70.5
#define START_LON 229.5


/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); return 2;}

void main() 
{

	int ncid,varid,u_varid,v_varid;
	int lat_varid,lon_varid,time_varid;	

	
	/* The start and count arrays will tell the netCDF library where to
	read our data. */
	size_t start[NDIMS], count[NDIMS];

	/* Program variables to hold the data we will read. We will only
	need enough space to hold one timestep of data; one record. */

/*	float* udata;
	udata = (float*)malloc(LAT_SIZE  * LONG_SIZE * TIMESTEPS * sizeof(float));
	float* vdata;
	vdata = (float*)malloc(LAT_SIZE  * LONG_SIZE * TIMESTEPS * sizeof(float));
*/
	float udata[LONG_SIZE][LAT_SIZE][TIMESTEPS];
	float vdata[LONG_SIZE][LAT_SIZE][TIMESTEPS];
		
	/* These program variables hold the latitudes and longitudes. */
	float lat[LAT_SIZE],lon[LONG_SIZE];

	char u_units_in[MAX_ATT_LEN], v_units_in[MAX_ATT_LEN];
	char lat_units_in[MAX_ATT_LEN], lon_units_in[MAX_ATT_LEN];

	/* Loop indexes. */
   	int lvl, lat_ind, lon_ind, rec, i = 0;

	int retval;

   	/* We will learn about the data file and store results in these
	program variables. */
	int ndims_in, nvars_in, ngatts_in, unlimdimid_in;

	if((retval = nc_open(NETCDF_FILE,NC_NOWRITE,&ncid))) ERR(retval);

	 if ((retval = nc_inq(ncid, &ndims_in, &nvars_in, &ngatts_in, &unlimdimid_in))) ERR(retval);

	/* In this case we know that there are 2 netCDF dimensions, 4 netCDF variables, 
	no global attributes, and no unlimited dimension. */
	if (ndims_in != 3 || nvars_in != 5 || ngatts_in != 2 || unlimdimid_in != -1) return 2;


	/* Get the varids of the latitude and longitude coordinate variables. */
	if ((retval = nc_inq_varid(ncid,"latitude", &lat_varid))) ERR(retval);
	if ((retval = nc_inq_varid(ncid,"longitude", &lon_varid))) ERR(retval);

	/* Read the coordinate variable data. */
	if ((retval = nc_get_var_float(ncid, lat_varid, &lat[0]))) ERR(retval);
	if ((retval = nc_get_var_float(ncid, lon_varid, &lon[0]))) ERR(retval);

	/* Check the coordinate variable data. */
	for (lat_ind = 0; lat_ind < LAT_SIZE; lat_ind++)
		if (lat[lat_ind] != START_LAT + 5.*lat_ind)
			 return 2;
		for (lon_ind = 0; lon_ind < LONG_SIZE; lon_ind++)
			if (lon[lon_ind] != START_LON + 5.*lon_ind)
				 return 2;

  	 /* Get the varids of u and v wind variables. */
	if ((retval = nc_inq_varid(ncid, "u", &u_varid))) ERR(retval);
	if ((retval = nc_inq_varid(ncid, "v", &v_varid))) ERR(retval);



	/* Each of the netCDF variables has a "units" attribute. Let's read
	them and check them. */
	if ((retval = nc_get_att_text(ncid, lat_varid, UNITS, lat_units_in))) ERR(retval);
	if (strncmp(lat_units_in, LAT_UNITS, strlen(LAT_UNITS))) return 2;

	/* Read and check one record at a time. */
	for (rec = 0; rec < NUM_RECORDS; rec++)
	{
		start[0] = rec;
		if ((retval = nc_get_vara_float(ncid, u_varid, start, count, &udata[0][0][0]))) ERR(retval);
		if ((retval = nc_get_vara_float(ncid, v_varid, start,count, &vdata[0][0][0]))) ERR(retval);

		/* Check the data. */
		i = 0;
		for (lvl = 0; lvl < TIMESTEPS; lvl++)
			for (lat_ind = 0; lat_ind < LAT_SIZE; lat_ind++)
				for (lon_ind = 0; lon_ind < LONG_SIZE; lon_ind++)
				{
					if (udata[lvl][lat][lon] != SAMPLE_U + i) return 2;
					if (vdata[lvl][lat][lon] != SAMPLE_V + i) return 2;
					i++;
				}

	} /* next record */


	/* Close the file. */
	if ((retval = nc_close(ncid))) ERR(retval);

//	free(udata);
//	free(vdata);

	printf("*** SUCCESS reading example file pres_temp_4D.nc!\n");
	return 0;
}
