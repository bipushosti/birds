/**
@file First_run_Full24hrs.c
@mainpage Bird Flight Code Documentation 1.0
@author Bipush Osti

@brief The birds start at 00:00 UTC which is 6pm in central time when there is no day light savings

*/

#include <math.h>
#include <float.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
//#include <gsl/gsl_math.h>

//#include <GL/glut.h>

//--------------------------Starting date--------------------------------------------------------------------------------------//
//#define START_DAY		1
//#define START_MONTH		AUG	//Always abbreviated caps; AUG,SEPT,OCT or NOV
//#define START_YEAR		2008
//-----------------------------------------------------------------------------------------------------------------------------//

#define PI 			3.14159
#define LONG_SIZE		429
#define LAT_SIZE		429

//Total chars in one line of input file
#define LINESIZE		15*LAT_SIZE+LAT_SIZE - 3
//122 because AUG=31,Sept=30,OCT=31,NOV=30; Total days there is 122
#define TOTAL_DAYS		122
#define TIMESTEPS_PER_DAY	24

//Total number of timesteps for the total number of days
#define TIMESTEPS		TOTAL_DAYS*TIMESTEPS_PER_DAY
#define SKIP_TIMESTEPS		0


//This is with respect to 0 UTC or pm. If set as negative then the hour the birds take off will be 7pm minus the number.
//If positive then plus the number. Example, if set at -1, then the birds fly at 6pm so for the first day or AUG 1 they 
//wait till next day to fly.
#define START_TIMESTEP		-1 		


//The maximum lattitude south that the model cares about bird flight. If birds go below
//that lattitude the model stops
//Counted from the North; 
#define MAX_LAT_SOUTH			300

//Stopover days; As of now, if 0 then the bird flies without stopping continiously;
//If 1, then the bird waits for 18 hours after successful 6 hours of flight to fly again
#define STOPOVER_DAYS		0

//#define DESIRED_SPEED	3.6		//Birds want to travel at 10m/s, it is 36km/hr(in the grid it is 3.6 units per hour) 
	
#define DESIRED_SPEED		10.5	//Air speed; Desired speed = flightspeed + windspeed ; Only used in windprofit calculation

#define STD_BIRDANGLE		10	//Standard deviation * 6 = the total difference from max to min angle possible
					//If STD_BIRDANGLE = 10 then the angle can differ +- (10*6)/2 = +- 30 from mean
#define	glCompAcc		1e-8	//If the difference is equal to or less than this then equal

#define MIN_PROFIT		-10
//Defining the x-variable size, it's sum and
//sum of squares as needed for slope calculation

//Since the 24 hour data is not and cannot be included due to
//its size the regression hours currently are from the previous night
//at that point.A new text file has to be created that has the pressure trend
//value for the last 12/24/6 hours at that point for each point in the map 
//for each take off time(6pm or 7pm) instead of including all the pressure data files.
//This helps in reducing the size of the data.

#define REGRESSION_HRS		6

//Precipitation (mm/hr) below which birds can fly
#define MAX_PRECIP		2

//HRS_SUM = sum(1 to 12) before. Now has to be sum(1 to 6) = 21
#define HRS_SUM			21
#define HRS_SQUARE_SUM		91
#define DENOM_SLOPE		(REGRESSION_HRS * HRS_SQUARE_SUM)-(HRS_SUM * HRS_SUM)
// Barometric pressure
// Bird finds the pressure at the time it leaves and compares it with the data from
// the previous day.

//The angle that the bird flies when it is out at sea and needs to get back to land.
//To make the birds head back directly west the angle must be set to 270.
#define BIRD_SEA_ANGLE		270

#define BIRD_HRS_LIMIT		24
//------------------------------Notes---------------------------------------------------------------------------------------
/*
Altitude = 850 millibars
Year = 2009
22 Jan 2015 No upper limit to the bird flight speed currently; Birds can fly well above 10m/s
Precipitation = millimeters
*/


long int starting_l = 0;
//--------------------------------------------------------------------------------------------------------------------------
void get_movementData(FILE* outTxt,float starting_row,float starting_col,long l,float* udata,float* vdata,float* dirData,float* precipData,float* pressureData,float* u10data,float* v10data,int* lwData,char* start_date,char* year);
float getProfitValue(float u,float v,float dirVal,float dir_u,float dir_v);
float bilinear_interpolation(float x,float y,float* data_array,long l,int dataSize);
float WrappedNormal (float MeanAngle,float AngStdDev);
long convert_to_month(char* month,char* day);
char** date_from_days(char** date,long timeStep);
//int flight_over_water(int* lw_grid,float row_pos,float col_pos);
float * get_u_v_dirAngle(float dirAngle,float speed);
float* check_bird_location(FILE* posFile,int* landWaterData,float* udata,float* vdata,float* dirData,float* precipData,float pos_row,float pos_col,int index,long l,char* start_date,float starting_row,float starting_col,char* year,long start_l);
//-------------------------------------------------------------------------------------------------------------------------------------
							//Functions
//------------------------------------------------------------------------------------------------------------------------------------
/** @brief This function is implemented to check the bird location
 *
 * The bird if above water after the 6 hours of flight will move West until it hits land and then
 * it stops. The new position will be that when the bird first hits land. The bird keeps looking for
 * land in the west direction as long as it finds it or if it exceeds a certain number of hours (18 here). 
 * 
 * 
 * @param posFile The file that stores the position of the bird
 * @param landWaterData The matrix that contains either 0 or 1 depending on whether it is land or water
 * @param udata The matrix that contains the u-wind value for each point in all the timesteps
 * @param vdata The matrix that contains the v-wind value for each point in all the timesteps
 * @param dirData The matrix that contains the desired direction of birds in each location in the 429x429 grid
 * @param precipData The matrix that contains the precipitation value for each point in all timesteps
 * @param row_pos The row position of the bird
 * @param col_pos The column position of the bird
 * @param index Integer that differentiates whether the bird is in land, sea or fresh water
 * @param l Integer that represents the timestep
 * @param start_data The starting date for the run
 * @param starting_row Row value that the bird starts from
 * @param starting_col Column value that the bird starts from
 * @return An array containing the new values for position of the bird
*/

float* check_bird_location(FILE* posFile,int* landWaterData,float* udata,float* vdata,float* dirData,float* precipData,float pos_row,float pos_col,int index,long l,char* start_date,float starting_row,float starting_col,char* year,long start_l)
{
	
	static float ret_data[3] = {0}; // ret_data[0] = new l value, 1 = new pos_row value , 2 = new pos_col
	float dir_u,dir_v,u_val,v_val,dirAngle,precip_val,distance;
	float prev_row,prev_col,wind_Speed,bird_AirSpeed;
	char *location;

	//Index shows where the birds is currently at, Sea, Land or Fresh Water
	if(index == 0){
		location = "Sea";
	}else if(index == 1){
		location = "Land";
	}else if(index == 2){
		location = "Fresh Water";
	}

	v_val = 0;
	u_val = 0;
	dir_u = 0;
	dir_v = 0;


	char** date;
	date = malloc(2*sizeof(char*));
	date[0] = malloc(5*sizeof(char));
	date[1] = malloc(5*sizeof(char));


	if(landWaterData[(int)(rintf(pos_row)) * LONG_SIZE + (int)(rintf(pos_col))]==index){

		int count_timesteps = 0;

		for(count_timesteps = 0;count_timesteps<14;count_timesteps++,l++){

			//Bilinear interpolation for u and v data
			u_val = bilinear_interpolation(pos_col,pos_row,udata,l,1);	
			v_val = bilinear_interpolation(pos_col,pos_row,vdata,l,1);
			
			if(index == 0) {
				dirAngle = BIRD_SEA_ANGLE;
			}else {
				dirAngle = dirData[(int)(rintf(pos_row) * LAT_SIZE + rintf(pos_col))];
				dirAngle = WrappedNormal(dirAngle,STD_BIRDANGLE);
			}
		
			prev_row = pos_row;
			prev_col = pos_col;

			dir_u = get_u_v_dirAngle(dirAngle,DESIRED_SPEED)[0];
			dir_v = get_u_v_dirAngle(dirAngle,DESIRED_SPEED)[1];

			wind_Speed = sqrt(u_val * u_val + v_val * v_val);
			bird_AirSpeed = sqrt((u_val+dir_u)*(u_val+dir_u) +(v_val+dir_v)*(v_val+dir_v));



			//Desired speed needs to change in the case of column position or the birds
			//will not fly west
			pos_row = pos_row + (v_val + dir_v) * 0.36 * -1;	
			pos_col = pos_col + (u_val + dir_u) * 0.36;
			
			precip_val = bilinear_interpolation(pos_col,pos_row,precipData,l,1);

			date =  date_from_days(date,l);
			//l++;
			//count_timesteps++;

			//fprintf(posFile,"%s\t %f \t %f \t %f \t %f \t %f \t %f \t At %s!! \n",start_date,STARTING_ROW,STARTING_COL,pos_row,pos_col,u_val,v_val,location);

			distance = sqrt((pos_row - prev_row)*(pos_row - prev_row) + (pos_col - prev_col)*(pos_col - prev_col));

//			fprintf(posFile,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%ld\t%s %s %s\tAt %s\n",start_date,starting_row,starting_col,pos_row,pos_col,u_val,v_val,dirAngle,dir_u,dir_v,u_val+dir_u,v_val+dir_v,bird_AirSpeed,wind_Speed,distance*10,l,date[0],date[1],year,location);
				
			fprintf(posFile,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\tNA\tNA\tNA\tNA\t%d\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%ld\t%s %s %s\t%ld\t%ld\tFlying at sea (After 10 hours)\n",start_date,starting_row,starting_col,pos_row,pos_col,DESIRED_SPEED,u_val,v_val,MIN_PROFIT,l,year,date[1],date[0],(l - start_l +6)/TIMESTEPS_PER_DAY + 1,l-start_l);

			if(landWaterData[(int)(rintf(pos_row)) * LONG_SIZE + (int)(rintf(pos_col))] == 1){
				//loop_check = 1;
				long bckp_l;
				bckp_l = l;

				//This takes it back to the starting time of the previous day
				l = l - (6 + 4 + count_timesteps);
				//And this takes it to the starting time of the next day
				l += (STOPOVER_DAYS+1) * 24;


				int i;
				for(i=0;i<(l-bckp_l);i++){
					u_val = bilinear_interpolation(pos_col,pos_row,udata,bckp_l + i,1);	
					v_val = bilinear_interpolation(pos_col,pos_row,vdata,bckp_l + i,1);


					fprintf(posFile,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\tNA\tNA\tNA\tNA\t%d\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%ld\t%s %s %s\t%ld\t%ld\tBird Landed in Land from Sea(After Flying over 10 hours)\n",start_date,starting_row,starting_col,pos_row,pos_col,DESIRED_SPEED,u_val,v_val,MIN_PROFIT,l,year,date[1],date[0],(l - start_l +6)/TIMESTEPS_PER_DAY + 1,l-start_l);

				}

				break;
				printf("Land Sighted!\n");
			}

			if(pos_row >=  MAX_LAT_SOUTH){
				printf("\t\tBird reached or passed the southern most lattitude\n");
				fprintf(posFile,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\tNA\tNA\tNA\tNA\t%d\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%ld\t%s %s %s\t%ld\t%ld\tBird reached southern most lattitude\n",start_date,starting_row,starting_col,pos_row,pos_col,DESIRED_SPEED,u_val,v_val,MIN_PROFIT,l,year,date[1],date[0],(l - start_l +6)/TIMESTEPS_PER_DAY + 1,l-start_l);
				exit(0);	
			}
			
		}

		//Birds can actually fly upto 3 days to get to land; This needs to change
		//As of now, if the birds can not find their way back by next sunrise or +14 hrs
		//they die. +14 because +4 hours added after 6hour flight if they find themselves at sea.
		if(count_timesteps >= (BIRD_HRS_LIMIT-10)){ 
			//loop_check = 1;
			printf("Dead Bird!; For now the birds can not keep flying for more than a day or they die! \n");
			fprintf(posFile,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\tNA\tNA\tNA\tNA\t%d\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%ld\t%s %s %s\t%ld\t%ld\tDead Bird (Flying for over 18 hours\n",start_date,starting_row,starting_col,pos_row,pos_col,DESIRED_SPEED,u_val,v_val,MIN_PROFIT,l,year,date[1],date[0],(l - start_l +6)/TIMESTEPS_PER_DAY + 1,l-start_l);
			exit(0) ;
		}


//-------------------Testing---Works if here accept for situations in which the bird flies for over 24 hours-----------------//
/*		long bckp_l;
		bckp_l = l;

		//This takes it back to the starting time of the previous day
		l = l - (6 + 4 + count_timesteps);
		//And this takes it to the starting time of the next day
		l += (STOPOVER_DAYS+1) * 24;


		int i;
		for(i=0;i<(l-bckp_l);i++){
			u_val = bilinear_interpolation(pos_col,pos_row,udata,bckp_l + i,1);	
			v_val = bilinear_interpolation(pos_col,pos_row,vdata,bckp_l + i,1);


			fprintf(posFile,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\tNA\tNA\tNA\tNA\t%d\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%ld\t%s %s %s\t%ld\t%ld\tBird Landed in Land from Sea(After Flying over 10 hours)\n",start_date,starting_row,starting_col,pos_row,pos_col,DESIRED_SPEED,u_val,v_val,MIN_PROFIT,l,year,date[1],date[0],(l - start_l +6)/TIMESTEPS_PER_DAY + 1,l-start_l);

		}
*/
//---------------------------------------------------------------------------------------------------------------------------//
		ret_data[0] = (float)l;
		ret_data[1] = pos_row;
		ret_data[2] = pos_col;

		
	}
	else{
		//Going back to the time of the day the birds start their flight
		l+=14;
		ret_data[0] = -1.0;
		ret_data[1] = -1.0;
		ret_data[2] = -1.0;
	}
	
	free(date[0]);
	free(date[1]);
	free(date);

	return ret_data;
}

//-------------------------------------------------------------------------------------------------------------------------------------
/** @brief This function is implemented to use wrapped normal distribution to get a random angle
 * 
 * This function uses wrapped normal distribution to get an angle between 0 to 360
 * depending on the standard deviation and mean angle.
 * 
 * @param MeanAngle The value of mean angle 
 * @param AngStdDev The standard deviation
 * @return Random angle between 0 and 360 
*/
float WrappedNormal (float MeanAngle,float AngStdDev){

	//Fisher 1993 pg. 47-48
	//s = -2.ln(r)

	float u1,u2,x,z,y;
	//float wn;
	u1=0;
	u2=0;

	while(1){
		while(1){
			//Dividing to get values between 0 and 1
			u1 = (float)rand()/(float)RAND_MAX;
			u2 = (float)rand()/(float)RAND_MAX;
			printf("From Wrapped Normal() : u1:%f,u2:%f\n",u1,u2);
			if((u1 > 0) && (u2 > 0)) break;
		}

		//printf("Hello \n");

   		z = 1.715528 * (u1 - 0.5) / u2;
		//printf("z:%f\n",z);

    		x = 0.25 * z *z;
		//printf("x:%f\n",x);

		if ((x - (1 - u2)) < glCompAcc) {
			//check = 0;
			//continue;
			//printf("First\n");
			break;
			
		}
		else if (x - (-log(u2)) < glCompAcc){
			//check = 0;
			//continue;
			break;
			
		}
	}
	

	y = AngStdDev * z + MeanAngle;
	if ((y - 360) > -glCompAcc){ 
	    y = y - 360;
	}
 
	if (y < 0){
	    y = 360 + y;
	}
	
	return y;

  
}
//-------------------------------------------------------------------------------------------------------------------------------//
/** @brief This function calculates the date(Day Month) based on the timestep value
 * 
 * This function converts the timestep value into month and day as each day has 
 * TIMESTEPS_PER_DAY number of timesteps and each month has either 30 or 31 days.
 *
 * @param date The date provided as input argument to the main function in the form MONTH DAY
 * @param timeStep The number of timesteps after the input date
 * @return The date after 'timeStep' number of time steps in the format MONTH DAY
*/
char** date_from_days(char** date,long timeStep)
{

	char day[5];
	int tmpInt;
	int days;

	tmpInt = 0;
	days = 0;
/*
	//Getting the total number of days
	if((timeStep+1) % 24 != 0 ){
		tmpInt =(int)((timeStep+1)/24);		
		printf("TmpInt is: %d,And (int)(timeStep - (long)tmpInt):%d\n",tmpInt,(int)(timeStep - (long)tmpInt));

		days = tmpInt + (int)(timeStep - (long)tmpInt*24);
	}else{
		days = (int)(timeStep+1)/24;	
	}
*/
//	days = (timeStep+1)/24;

	days = (timeStep+1+18)/24;
	printf("Timestep: %ld,Calculated Days: %d\n",timeStep,days);
	//If days fall in August
	//if(days<31){
	if(days<=31){
	//	days +=1;
		//date[0] = "AUG";
		strncpy(date[0],"AUG",3);
		sprintf(day,"%d",days);
		snprintf(day,5 * sizeof(char),"%d",days++);
		//sprintf(day,"%d",days+=1);
		strncpy(date[1],day,2);
	//If Sept
//	}else if((days>=31) && (days <=60)){
	}else if((days>31) && (days <=61)){
		strncpy(date[0],"SEPT",4);
		days = days - 30;
		snprintf(day,5 * sizeof(char),"%d",days-1);
		strncpy(date[1],day,2);
	//If Oct
	}else if((days>=62) && (days <=92)){
		strncpy(date[0],"    ",4);
		strncpy(date[0],"OCT",3);
		days = days - 60;
		snprintf(day,5 * sizeof(char),"%d",days-1);
		strncpy(date[1],day,2);

	//If Nov
	}else if((days>92) && (days <=122)){
//	}else if((days>=92) && (days <=122)){
		strncpy(date[0],"NOV",3);
		days = days - 91;
		snprintf(day,5 * sizeof(char),"%d",days-1);
		strncpy(date[1],day,2);
	}else{
		printf("Incorrect month provided. Exiting! \n");
		exit(-1);
	}
			
	return date;
}
//-------------------------------------------------------------------------------------------------------------------------------//
/** @brief This function converts the date (Day Month) into timesteps to be used in the model
 * 
 * This function is opposite of function date_from_days as it converts given date in format 
 * MONTH DAY into timesteps to be used in the model.
 *
 * @param month The month part of the date
 * @param day The day part of the date
 * @return The timestep that corresponds to the date
*/
long convert_to_month(char* month,char * day)
{
	long index,offset;
	if(strcmp(month,"AUG")==0){
		index = 1; //The data starts in august
	}
	else if(strcmp(month,"SEPT")==0){
		index = 32; //The data for september starts after 31 days of august
	}
	else if(strcmp(month,"OCT")==0){
		index = 62; //The data for october starts after 31+30 days of sept and august respectively.
	}
	else if(strcmp(month,"NOV")==0){
		index = 93; //The data for october starts after 31+30+31 days of sept,aug and oct respectively.
	}
	else{
		printf("\n\t\tIncorrect month used\n\t\tUse between August-November inclusive; Only use abbriviated caps of the months; august = AUG\n");
		return -1;
	}
	
	offset = ((index-1)  + atoi(day))* TIMESTEPS_PER_DAY;
	return offset;

}


//---------------------------------------------------------------------------------------------------------------------

/** @brief Function to implement bilinear interpolation.
 * 
 * This function is necessary to interpolate data points that are not exactly on the grid
 * and hence do not have a value for wind,pressure and precipitation.
 * This was taken from: From: http://www.ajdesigner.com/phpinterpolation/bilinear_interpolation_equation.php
 *
 * @param x The x coordinate of the data point
 * @param y The y coordinate of the data point
 * @param data_array The data that need to be interpolated
 * @param l The timestep
 * @param dataSize It is either 0 (for small data like direction: LAT*LONG) or 1 (for large data like u-v wind: TIMESTEPS*LAT*LONG)
 * @return The interpolated value of the data array at that location
*/

float bilinear_interpolation(float x,float y,float* data_array,long l,int dataSize)
{
	float x1,y1,x2,y2;
	float value,Q11,Q12,Q21,Q22;
	//float val_x1,val_x2,val_y1,val_y2;

	x1 = floorf(x);
	x2 = ceilf(x);
	y1 = floorf(y);
	y2 = ceilf(y);
	value = 0;
	
	//printf("x1:%f,x2:%f,y1:%f,y2:%f\n",x1,x2,y1,y2);
	if(dataSize == 1){
		Q11 = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y1 * LONG_SIZE + x1) ];
		Q12 = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y2 * LONG_SIZE + x1) ];
		Q21 = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y1 * LONG_SIZE + x2) ];
		Q22 = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y2 * LONG_SIZE + x2) ];
	}
	else if(dataSize == 0) {
		Q11 = data_array[(int)(y1 * LONG_SIZE + x1)];
		Q12 = data_array[(int)(y2 * LONG_SIZE + x1)];
		Q21 = data_array[(int)(y1 * LONG_SIZE + x2)];
		Q22 = data_array[(int)(y2 * LONG_SIZE + x2)];
	}

	//If no interpolation needed
	if((x2 == x1) && (y2 == y1)){
		value = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y1 * LONG_SIZE + x1)];
	}
	//If in a straight line with same x coordinates take average
	else if((x2 == x1) && (y2 != y1)){
			if(dataSize == 1) {
				//value+= data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y1 * LONG_SIZE + x)];
				//value+=((y-y1)/(y2-y1))*(data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y2 * LONG_SIZE + x)]-data_array[(int)(l*LAT_SIZE * LONG_SIZE + y1 * LONG_SIZE + x)]);
				value=(data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y1 * LONG_SIZE + x)] + data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y2 * LONG_SIZE + x)])/2;
			}
			else if(dataSize == 0){
				//value+= data_array[(int)(y1 * LONG_SIZE + x)];
				//value+=((y-y1)/(y2-y1))*(data_array[(int)(y2 * LONG_SIZE + x)]-data_array[(int)(y1 * LONG_SIZE + x)]);
				value= (data_array[(int)(y1 * LONG_SIZE + x)]+data_array[(int)(y2 * LONG_SIZE + x)])/2;
			}
	}
	else if((x2 != x1) && (y2 == y1)){
			if(dataSize == 1) {
				//value+= data_array[(int)(l  * LAT_SIZE * LONG_SIZE + x1 * LONG_SIZE + y)];
				//value+=((x-x1)/(x2-x1))*(data_array[(int)(l  * LAT_SIZE * LONG_SIZE + x2 * LONG_SIZE + y)]-data_array[(int)(l*LAT_SIZE*LONG_SIZE + x1 * LONG_SIZE + y)]);
				value=(data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y * LONG_SIZE + x1)] + data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y * LONG_SIZE + x2)])/2;
			}
			else if(dataSize == 0){
				//value+= data_array[(int)(x1 * LAT_SIZE + y)];
				//value+=((x-x1)/(x2-x1))*(data_array[(int)(x2 * LAT_SIZE + y)]-data_array[(int)(x1 * LAT_SIZE + y)]);
				value= (data_array[(int)(y * LONG_SIZE + x1)]+data_array[(int)(y * LONG_SIZE + x2)])/2;
			}
	}

	else{
		value =((x2-x)* ((y2-y)*Q11 + (y-y1)*Q12)+(x-x1)*((y2-y)*Q21 +(y-y1)*Q22))/((x2-x1)*(y2-y1));
	}
	//printf("Q11:%f,Q12:%f,Q21:%f,Q22:%f; And Value=%f\n",Q11,Q12,Q21,Q22,value);
	return value;
}
//-------------------------------------------------------------------------------------------------------------------------------------
/** @brief This function Gets the "profit value" 
 * 
 * This function gets the profit value which is the value that the birds compare against a fixed value 
 * to check if it is profitable to fly at that particular time. The fixed value is define in this file as
 * MIN_PROFIT after the header file declarations.
 * The coordinate system used in the model is flipped;(Actual->In Model) I->I,IV->II,III->III,II->IV 
 *
 * @param u_val The u value at that point at that particular timestep
 * @param v_val The v value at that point at that particular timestep
 * @param dirval The angle at which the birds want to fly at that particular location
 * @param v_val The v value at that point at that particular timestep
 * @param dir_u The u component of the bird's preferred direction and speed at that location
 * @param dir_v The v component of the bird's preferred direction and speed at that location
 * @return The profit value at that location at that particular timestep
*/
float getProfitValue(float u_val,float v_val,float dirVal,float dir_u,float dir_v)
{

	//printf("\n Input DirVal = %f\n",dirVal);
	//All wind data in m/s
	float angle,diffAngle,magnitude,magnitude_squared;

	//vectAngle = angle between the wind vector and the vector orthogonal to the direction vector; or the crosswind vector
	float tailComponent,vectAngle,crossComponent,profit_value;
	tailComponent = 0;

	magnitude_squared = u_val * u_val + v_val * v_val;
	magnitude = (float)sqrtf(magnitude_squared);


	//Getting the tail component of the wind; or the component of the wind in the desired direction of flight
	//From formula of getting the vector projection of wind onto the desired direction
	tailComponent = (dir_v * v_val + dir_u * u_val);
	tailComponent = tailComponent/sqrtf(u_val*u_val + v_val*v_val);
	//Separate profit value methods have to be used if the tail component is less that equal to or greater than the desired speed of the birds


	if(tailComponent <= DESIRED_SPEED) {
		//profit_value = getProfitValue(u_val,v_val,actualAngle);

		//DiffAngle is the angle between the desired direction of the bird 
		//and the direction of the wind
		//DiffAngle has to be calculated such that both the vectors are pointing
		//away from where they meet.

//***** The coordinate system used in this model is not the regular default type. Here the angles are with respect to positive
//	y and instead of measuring them from right to left they go left to right. Therefore Q1,Q2,Q3,Q4 in this model correspond to 
//	these normal quadrants Q1,Q4,Q3,Q2


//For normal quadrants can be written as diffAngle = abs(dirVal - angle)	
		if ((v_val == 0)&&( u_val > 0)){
			angle = 90;
			diffAngle = abs(dirVal - angle);
		}
		else if ((v_val == 0)&&( u_val < 0)){
			angle = 270;
			diffAngle = abs(angle - dirVal);
		}
		else if ((v_val > 0)&&( u_val == 0)){
			angle = 0;
			diffAngle = dirVal + angle;
		}
		else if ((v_val < 0)&&( u_val == 0)){
			angle = 180;
			diffAngle = abs(angle - dirVal);
		}

		//abs value taken so as to take the angles with respect to 
		//quadrant 1. tangent graph is always positive when angle is >0
		//and <90 but negative when angle is >-90 and <0. So no need to check
		//if the value is negative
		else if((v_val>0)&&(u_val>0)){	//Quadrant 1 in both cases
			u_val = abs(u_val);
			v_val = abs(v_val);
			angle = atanf(u_val/v_val) * (180/PI);
			diffAngle = abs(angle - dirVal);
		}
		else if((v_val<0)&&(u_val>0)){	//Quadrant 2 in this model and Quadrant 4 in the regular system
			u_val = abs(u_val);
			v_val = abs(v_val);
			angle = atanf(v_val/u_val) * (180/PI) + 90;
			diffAngle = abs(angle - dirVal);
		}
		else if((v_val<0)&&(u_val<0)){	//Quadrant 3 in  both cases
			u_val = abs(u_val);
			v_val = abs(v_val);
			angle = atanf(u_val/v_val) * (180/PI) + 180;
			diffAngle = abs(angle - dirVal);
		}
		else if((v_val>0)&&(u_val<0)){	//Quadrant 4 in this model and Quadrant 2 in regular coordinate system 
			u_val = abs(u_val);
			v_val = abs(v_val);
			angle = atanf(v_val/u_val) * (180/PI) + 270;
			diffAngle = abs(angle - dirVal);
		}

		printf("dirVal = %f,angle= %f,diffAngle = %f\n",dirVal,angle,diffAngle);
		//Getting the smaller angle
		if(diffAngle > 180) diffAngle = 360 - diffAngle;
		if(diffAngle > 360) diffAngle = diffAngle - 360;
		profit_value = (DESIRED_SPEED * DESIRED_SPEED) + magnitude_squared - 2 * DESIRED_SPEED * magnitude * cos(diffAngle * PI/180);
		profit_value = DESIRED_SPEED - (float)sqrtf(profit_value);
	}
	else {
		vectAngle = (dir_v * v_val + dir_u * u_val);
		vectAngle = acos(vectAngle / sqrtf((u_val*u_val + v_val* v_val)*(dir_v * dir_v + dir_u * dir_u))) * (180/PI);
		vectAngle = (vectAngle <= 90)? 90 - vectAngle: vectAngle - 90;
		crossComponent = sqrtf(u_val*u_val + v_val*v_val)/cos(vectAngle);
		//Getting the absolute value
		crossComponent = (crossComponent<0)? crossComponent * (-1):crossComponent;
		profit_value = tailComponent - crossComponent;
		printf("Over the Desired Speed\n");
	}

	return profit_value;
}
//------------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------------------

/** @brief This function computes the u and v components for a given angle and speed
 * 
 * This function uses trigonometry to calculate the u and v components when a desired
 * speed and desired angle is given.The coordinate system used in the model is flipped;
 * (Actual->In Model) I->I,IV->II,III->III,II->IV 
 *
 * @param dirAngle The angle 
 * @param speed The magnitude in the direction of the dirAngle
 * @return A float array containing speeds in u and v directions
*/
float * get_u_v_dirAngle(float dirAngle,float speed)
{

	//u is index 0; v is index 1
	static float u_v_dirAngles[2];	

	
	if(dirAngle <= 90){//Checked OK
		dirAngle = 90 - dirAngle;
		u_v_dirAngles[0] = speed * cos(dirAngle * (PI/180));
		u_v_dirAngles[1] = speed * sin(dirAngle * (PI/180));	
	}
	else if((dirAngle > 90) && (dirAngle <= 180)){//Checked OK
		dirAngle -= 90;
		u_v_dirAngles[0] = speed * cos(dirAngle * (PI/180));
		u_v_dirAngles[1] = speed * sin(dirAngle * (PI/180)) * -1;
	}
	else if((dirAngle > 180) && (dirAngle <= 270)) {
		dirAngle = 270 - dirAngle;
		u_v_dirAngles[0] = speed * cos(dirAngle * (PI/180)) * -1;
		u_v_dirAngles[1] = speed * sin(dirAngle * (PI/180)) * -1;
	}
	else if((dirAngle > 270) && (dirAngle <= 360)){
		dirAngle -= 270;
		u_v_dirAngles[0] = speed * cos(dirAngle * (PI/180)) * -1;
		u_v_dirAngles[1] = speed * sin(dirAngle * (PI/180));
	}
	return u_v_dirAngles;
}

//------------------------------------------------------------------------------------------------------------------------------------
/** @brief This function computes the movement for the birds
 *
 * This function calculates the bird positions based on wind, precipitation and pressure data. 
 * Wind profit values are calculated and rules are setup such that the birds fly only if 
 * the wind is favourable (wind profit greater than MIN_PROFIT), precipitation is less than a certain value and
 * pressure is less than a certain value and the pressure trend has a certain slope.
 * 
 * @param outTxt The file that will store the position data of the bird 
 * @param starting_row The row that the bird starts in
 * @param starting_col The column that the bird starts in
 * @param l The variable that holds the timestep value
 * @param udata The matrix that holds the u value for the 429x429 grid for each timestep
 * @param vdata The matrix that holds the v value for the 429x429 grid for each timestep
 * @param dirData The matrix that holds the desired direction of the bird in each of the points in the grid
 * @param precipData The matrix that holds the precipitation value for the 429x429 grid for each timestep
 * @param pressureData The matrix that holds the pressure value for the 429x429 grid for each timestep
 * @param u10data The matrix that holds the u value at pressure level 10 in the 429x429 grid for each timestep
 * @param v10data The matrix that holds the v value at pressure level 10 in the 429x429 grid for each timestep
 * @param lwData The matrix that holds the information about whether the point in the grid is either land(1),sea(0) or freshwater(2)
 * @param start_date The start date of the model in the format MONTH DAY
 * @param year The year of the model
 * @return The function does not return anything but it writes the position data to the file specified in outTxt
*/
void get_movementData(FILE* outTxt,float starting_row,float starting_col,long l,float* udata,float* vdata,float* dirData, float* precipData,float* pressureData,float* u10data,float* v10data,int* lwData,char* start_date,char* year)
{
	fprintf(outTxt,"\n\ns_date\ts_row\ts_col\tCur_row\tCur_col\tDSpeed\tUval\tVval\tdirAngle\tPrecip\tPress.\tPress.Slope\tMinWProft\tWProft\tdirU\tdirV\tUair\tVair\tBird_GS\tW_Speed\tDistt\tl\tCurrentDate\tDaysTraveled\tHoursTravelled\tActivity\n");

	float distance,prev_row,prev_col,bird_AirSpeed,wind_Speed;
	long start_l;

	distance = 0;
	bird_AirSpeed = 0;
	wind_Speed = 0;
		

	float pos_row,pos_col;
	//pos_row = LONG_SIZE - STARTING_ROW;

	pos_row = starting_row;
	pos_col = starting_col;
	prev_row = starting_row;
	prev_col = starting_col;

	//fprintf(outTxt,"%f \t %f\n",pos_row,pos_col);
	if(pos_row >= MAX_LAT_SOUTH){
		printf("\t\tProvided starting row is below the southern most lattitude at which the model is set to stop\n");
		printf("\t\tEither change the starting row location and/or MAX_LAT upto which the birds can fly\n");
		return;
	}
	int k;
	long i,l_old;

	float pressure_sum,pressure_MultSum,last_pressure,slope;
	pressure_MultSum = 0;
	pressure_sum = 0;
	slope = 1;
	last_pressure = 1011;

	
	//l = 18;
	//l_old = 18;
	l_old = l;

//--------------------Changing from 7pm bird flight start time to 6pm------//
	char * last_space;
	char month_day[10];
	last_space = strchr(start_date,' '); 
	strncpy(month_day,last_space,last_space - start_date);
	//Specific case for when it is August 1st 6pm because the data is from 7pm onwards
	if(strcmp(month_day,"AUG 1")==0){
		l = 22;
		printf("AUGUST 1st detected; Birds will start from Aug 2 because of lack of data\n");
	//If not in AUG 1st then substract the START_TIMESTEPS from the initial timestep.
	}else{
		l+=START_TIMESTEP; 	
	}

	start_l = l;
//------------------------------------------------------------------------------//
	printf("%s\n",month_day);



	float profit_value,dirAngle,actualAngle;

	
	float dir_v,dir_u;
	//long skip_size = (SKIP_TIMESTEPS * LONG_SIZE * LAT_SIZE) - 1;
	long skip_size = 0;
	

	float u_val,v_val,precip_val,v_ten,u_ten;
	int skip;
	skip=0;

	char** date;
	date = malloc(2*sizeof(char*));
	date[0] = malloc(5*sizeof(char));
	date[1] = malloc(5*sizeof(char));

	float *ret_values;

	//A boolean value; 1 = Yes, at water; 0 = No, at land;
	int in_water;

	printf("i \t l \t k \t precipData \t profit_value \t pos_row \t pos_col \t (v_val,u_val)\n");
	i = skip_size +pos_row * LAT_SIZE + pos_col;

	


	while( i <= (TIMESTEPS-1) * LAT_SIZE * LONG_SIZE ) {

		dir_v = 0;
		dir_u = 0;
		dirAngle = 0;
		actualAngle = 0;
		u_val = 0;
		v_val = 0;
		precip_val = 0;
		v_ten = 0;
		u_ten = 0;
		pressure_sum = 0;
		pressure_MultSum = 0;
		
		//Reinitializing loop index
		k = 0;

		//If current pressure is greater than pressure in the previous day
		//if(pressureData[i] - old_pressure > 0) {
		printf("Main loop[k]\n");

		u_ten = bilinear_interpolation(pos_col,pos_row,u10data,l,1);
		v_ten = bilinear_interpolation(pos_col,pos_row,v10data,l,1);
		
		//The direction angle is chosen only once, before the flight.
		dirAngle = dirData[(int)(rintf(pos_row) * LAT_SIZE + rintf(pos_col))];
		dirAngle = WrappedNormal(dirAngle,STD_BIRDANGLE);
		actualAngle = dirAngle;

		//The grid is upside down; y increases from top to bottom while x increases from left to right 
		//dir_v and dir_u are the x and y components of the wind (v=y,u=x)
		dir_u = get_u_v_dirAngle(dirAngle,DESIRED_SPEED)[0];
		dir_v = get_u_v_dirAngle(dirAngle,DESIRED_SPEED)[1];

		printf("U10 and V10 values are: %f,%f\n",u_ten,v_ten);
		printf("dir_u,dir_v values are %f,%f\n",dir_u,dir_v);
		printf("10 profit value::%f\n",getProfitValue(u_ten,v_ten,actualAngle,dir_u,dir_v));
		printf("pressure value::%f,slope value::%f\n",last_pressure,slope);


		//dirAngle = dirData[(int)(rintf(pos_row) * LAT_SIZE + rintf(pos_col))];
		//dirAngle = WrappedNormal(dirAngle,STD_BIRDANGLE);
		//printf("\n First DirectionAngle = %f,AngleFromFile = %f\n",dirAngle,dirAngleFromFile);
		//actualAngle = dirAngle;

		l_old  = l;

//************************************************************************************************************************************************
//*********************Check profit value at ground; Or for V10 and U10***************************************************************************
		float windProfit;
		windProfit = getProfitValue(u_ten,v_ten,actualAngle,dir_u,dir_v) ;

		//Relation between last_pressure and slope is an OR
		if((windProfit >= MIN_PROFIT) && ((last_pressure>=1009)||(slope >-1))){
			
			
			printf("\n\nl value check: %ld\n\n",l);
//-------------------------------------------Start of the 6 hour bird flight----------------------------------------------------------------------------------	
			

			for(k=0;k<6;k++,l++) {
				i = skip_size + l * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col;

				//dirAngle is with respect to North or the positive y-axis
				//It is the genetic direction of the birds
				
				dirAngle = actualAngle;
				//printf("\n DirectionAngle = %f,AngleFromFile = %f\n",dirAngle,dirAngleFromFile);

				

				//dir_u = get_u_v_dirAngle(dirAngle,DESIRED_SPEED)[0];
				//dir_v = get_u_v_dirAngle(dirAngle,DESIRED_SPEED)[1];

				//Bilinear interpolation for u and v data
				u_val = bilinear_interpolation(pos_col,pos_row,udata,l,1);	
				v_val = bilinear_interpolation(pos_col,pos_row,vdata,l,1);

				//printf("(u_val,v_val)::(%f,%f)\n",u_val,v_val);

				//profit_value = getProfitValue(u_val,v_val,actualAngle,dir_u,dir_v);
				precip_val = bilinear_interpolation(pos_col,pos_row,precipData,l,1);


//Adding precip value

				

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//DO NOT DELETE! - Option to allow birds to change direction if they are at sea on an hourly basis

					//The bird checks every timestep; If the value is changed from 1 to 2 it checks once every 
					//2 timesteps
					/*if(k % 1 == 0){

						if(lwData[(int)(rintf(pos_row)) * LONG_SIZE + (int)(rintf(pos_col))] == 0){
							//Changing the desired direction to 270 if found at sea
							dir_u = DESIRED_SPEED * -1;	
							fprintf(outTxt,"%f \t %f \t %f \t %f \t At Sea!!\n",pos_row,pos_col,u_val,v_val);
						}
					
						
						//ret_values = check_bird_location(outTxt,lwData,udata,vdata,dirData,pos_row,pos_col,0,l);
						//if((*ret_values)!= -1.0){
						//	l = (long)(*ret_values);
						//	pos_row = *(ret_values+1);
						//	pos_col = *(ret_values+2);

						//	printf("6 Hour Loop:: After the new function row = %f,col = %f, l value = %ld\n",pos_row,pos_col,l);
						//}
					}*/
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------	


				prev_col = pos_col;
				prev_row = pos_row;

				//Position is in a 10 km grid. This means for 1m/s, the 
				//position change in 1 hour is 3.6km/10 = 0.36units in the grid

				pos_row = pos_row + (v_val + dir_v)*0.36* -1;
				pos_col = pos_col + (u_val + dir_u)*0.36;

				//float tmp;
				//tmp = sqrtf((v_val+dir_v)*(v_val+dir_v) + (u_val+dir_u)*(u_val+dir_u));
				//printf("\nTailComponent::%f,Speed::%f,dir_v::%f,dir_u::%f\n",tailComponent,tmp,dir_v,dir_u);
			
				printf("%s\t %f \t %f \t %ld \t %ld \t %d \t %f \t %f \t %.6f \t %.6f \t (%.6f,%.6f)\n",start_date,starting_row,starting_col,i,l,k,precip_val,profit_value,pos_row,pos_col,v_val,u_val);
			

				
				//fprintf(outTxt,"%f,%f\n",pos_row,pos_col);
				distance = sqrt((pos_row - prev_row)*(pos_row - prev_row) + (pos_col - prev_col)*(pos_col - prev_col));
				
		
				wind_Speed = sqrt(u_val * u_val + v_val * v_val);
				bird_AirSpeed = sqrt((u_val+dir_u)*(u_val+dir_u) +(v_val+dir_v)*(v_val+dir_v));


				date =  date_from_days(date,l);



				//Distance is in absolute value (kilometers rather than in units of grid points)
				fprintf(outTxt,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%ld\t%s %s %s\t%ld\t%ld\tBirds Flying\n",start_date,starting_row,starting_col,pos_row,pos_col,DESIRED_SPEED,u_val,v_val,dirAngle,precip_val,last_pressure,slope,MIN_PROFIT,windProfit,dir_u,dir_v,u_val+dir_u,v_val+dir_v,bird_AirSpeed,wind_Speed,distance*10,l,year,date[1],date[0],(l - start_l +6)/TIMESTEPS_PER_DAY + 1,l-start_l);
				
				if(pos_row >=  MAX_LAT_SOUTH){
					printf("\t\tBird reached or passed the southern most lattitude\n");
					return;
				}
			}
		
//-------------------------------------------End of the 6 hour bird flight------------------------------------------------------------------------------------

//-------------------------------------------If bird at sea at the end of the 6 hour flight-------------------------------------------------------------------

			//If after 6 hours the birds are at sea then they travel for 4 additional hours in the same preferred direction.
			//If they are still at sea after 4 hours then they head west until they find the sea at an angle defined in BIRD_SEA_ANGLE
			if(lwData[(int)(rintf(pos_row)) * LONG_SIZE + (int)(rintf(pos_col))]!=1){
				//Move in the preferred direction for additional 4 hours
				for(k=6;k<10;k++,l++){

					dirAngle = dirData[(int)(rintf(pos_row) * LAT_SIZE + rintf(pos_col))];
					dirAngle = WrappedNormal(dirAngle,STD_BIRDANGLE);

					dir_u = get_u_v_dirAngle(dirAngle,DESIRED_SPEED)[0];
					dir_v = get_u_v_dirAngle(dirAngle,DESIRED_SPEED)[1];

					//Bilinear interpolation for u and v data
					u_val = bilinear_interpolation(pos_col,pos_row,udata,l,1);	
					v_val = bilinear_interpolation(pos_col,pos_row,vdata,l,1);
					precip_val = bilinear_interpolation(pos_col,pos_row,precipData,l,1);
					pos_row = pos_row + (v_val + dir_v)*0.36* -1;
					pos_col = pos_col + (u_val + dir_u)*0.36;

					wind_Speed = sqrt(u_val * u_val + v_val * v_val);
					bird_AirSpeed = sqrt((u_val+dir_u)*(u_val+dir_u) +(v_val+dir_v)*(v_val+dir_v));

					distance = sqrt((pos_row - prev_row)*(pos_row - prev_row) + (pos_col - prev_col)*(pos_col - prev_col));

					date =  date_from_days(date,l);

					printf("%s\t%f\t%f\t%ld\t%ld\t%d\t%f\t%f\t%.6f\t%.6f\t (%.6f,%.6f)\n",start_date,starting_row,starting_col,i,l,k,precip_val,profit_value,pos_row,pos_col,v_val,u_val);
					

					//fprintf(outTxt,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t\t%s %sDuring the additional 4 hours \n",start_date,starting_row,starting_col,pos_row,pos_col,u_val,v_val,dir_u,dir_v,u_val+dir_u,v_val+dir_v,bird_AirSpeed,wind_Speed,date[0],date[1]);

					fprintf(outTxt,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%ld\t%s %s %s\t%ld\t%ld\tFlying at sea (After 6 hours)\n",start_date,starting_row,starting_col,pos_row,pos_col,DESIRED_SPEED,u_val,v_val,dirAngle,precip_val,last_pressure,slope,MIN_PROFIT,windProfit,dir_u,dir_v,u_val+dir_u,v_val+dir_v,bird_AirSpeed,wind_Speed,distance*10,l,year,date[1],date[0],(l - start_l +6)/TIMESTEPS_PER_DAY + 1,l-start_l);
				}
			}else{
				int ii;
				long tmp_l;
				tmp_l = l;
				for(ii = 0;ii<18;ii++){
					u_val = bilinear_interpolation(pos_col,pos_row,udata,l,1);	
					v_val = bilinear_interpolation(pos_col,pos_row,vdata,l,1);
					precip_val = bilinear_interpolation(pos_col,pos_row,precipData,l,1);
					date =  date_from_days(date,l);

					fprintf(outTxt,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\tNA\tNA\tNA\tNA\t%d\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%ld\t%s %s %s\t%ld\t%ld\tLand After 6 hours\n",start_date,starting_row,starting_col,pos_row,pos_col,DESIRED_SPEED,u_val,v_val,MIN_PROFIT,l,year,date[1],date[0],(l - start_l +6)/TIMESTEPS_PER_DAY + 1,tmp_l - start_l);
					tmp_l ++;
				}

			}

			if(lwData[(int)(rintf(pos_row)) * LONG_SIZE + (int)(rintf(pos_col))]==0){
				ret_values = check_bird_location(outTxt,lwData,udata,vdata,dirData,precipData,pos_row,pos_col,0,l,start_date,starting_row,starting_col,year,start_l);
				in_water = 1;
			}
			else if(lwData[(int)(rintf(pos_row)) * LONG_SIZE + (int)(rintf(pos_col))]==1){
				in_water = 0;
				l = (l-k) + ((STOPOVER_DAYS+1) * 24);
				int uu;
/*
				for(uu = 0;uu<14;uu++){
					u_val = bilinear_interpolation(pos_col,pos_row,udata,l,1);	
					v_val = bilinear_interpolation(pos_col,pos_row,vdata,l,1);
					precip_val = bilinear_interpolation(pos_col,pos_row,precipData,l,1);

					fprintf(outTxt,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%ld\t%s %s %s\tLand After 10 hours\n",start_date,starting_row,starting_col,pos_row,pos_col,u_val,v_val,dirAngle,precip_val,last_pressure,slope,windProfit,dir_u,dir_v,u_val+dir_u,v_val+dir_v,bird_AirSpeed,wind_Speed,distance*10,l,date[0],date[1],year);
				}
*/

				
			}
			else if(lwData[(int)(rintf(pos_row)) * LONG_SIZE + (int)(rintf(pos_col))]==2){	
				ret_values = check_bird_location(outTxt,lwData,udata,vdata,dirData,precipData,pos_row,pos_col,2,l,start_date,starting_row,starting_col,year,start_l);
				in_water = 1;
			}
			


			//ret_values = check_bird_location(outTxt,lwData,udata,vdata,dirData,pos_row,pos_col,2,l);

			if(in_water == 1){
				l = (long)(*ret_values);
				pos_row = *(ret_values+1);
				pos_col = *(ret_values+2);

				printf("After the new function row = %f,col = %f, l value = %ld\n",pos_row,pos_col,l);
			}
		} else{
			//If profit values for V10 and U10 were too low the very first time the bird does not fly
			//It has to still wait for the next day or +24 hours
			//This is because the winds are very unfavourable at pressure level 10
			//v10 and u10 profit values were too low
			printf("u10 v10 profit value too low!\n");

			int jj;

			for(jj = 0;jj<24;jj++,l++){

				//Bilinear interpolation for u and v data
				u_val = bilinear_interpolation(pos_col,pos_row,udata,l,1);	
				v_val = bilinear_interpolation(pos_col,pos_row,vdata,l,1);
				precip_val = bilinear_interpolation(pos_col,pos_row,precipData,l,1);


				fprintf(outTxt,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\tNA\tNA\tNA\tNA\t%d\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%ld\t%s %s %s\t%ld\t%ld\tProfit Value low at 10m\n",start_date,starting_row,starting_col,pos_row,pos_col,DESIRED_SPEED,u_val,v_val,MIN_PROFIT,l,year,date[1],date[0],(l - start_l +6)/TIMESTEPS_PER_DAY + 1,l-start_l);
			}
			l+=24;

		}//End If for profit value of v10 and u10

//*********************End of profit value check at ground; Or for V10 and U10********************************************************************
//************************************************************************************************************************************************		

		//Every day has 24 time steps and this line skips the total desired number of days
		//Adding 17 to skip the hours that do not fall between 7pm and 1am
		//l = (l-k) + ((STOPOVER_DAYS+1) * 24);
		l_old = l - REGRESSION_HRS;

		printf("check l %ld\n",l);

		//Taking the pressure from 6 hours earlier of the location where the bird landed
		for(k=1; (l_old < l) && (k<=REGRESSION_HRS); l_old++,k++){

			pressure_sum += bilinear_interpolation(pos_col,pos_row,pressureData,l_old,1);
			//pressure_sum += pressureData[skip_size + l_old * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col];
			pressure_MultSum += k * bilinear_interpolation(pos_col,pos_row,pressureData,l_old,1);
			//printf("%f\n",pressureData[skip_size + l_old * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col]);

			//last_pressure is the last day or the day of flight
			if(k == REGRESSION_HRS) {
				last_pressure = bilinear_interpolation(pos_col,pos_row,pressureData,l_old,1);
			}
		}

		slope = ((REGRESSION_HRS * pressure_MultSum) - (pressure_sum * HRS_SUM))/(DENOM_SLOPE);


		if(slope <= -1){
			if(last_pressure < 1009) {
				printf("Storm!!\n");
			}
			else if( (last_pressure >= 1009) && (last_pressure < 1020) ){
				printf("Precipitation is very likely````` \n");
			}
			else if(last_pressure >1020) {
				printf("Cloudy & Warm\n");
			}
		}
		//printf("\t Slope is:: %f\n",slope);
		//old_pressure = pressureData[i_old];
		//Since the data now does not contain all 24 hours 
		i = skip_size + l * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col;
		//l_old = l;
		//printf("Running\n");
		printf("-----------------------------------------------------------------------------------------------------------------------\n");

	}
	free(date[0]);
	free(date[1]);
	free(date);//fprintf(outTxt,"----------------------------------------------------------\n");
}

/*
########################################################################################################################################################################################
*/

//--------------------------------------------------------------------------------------------------------------------------
					//Main 
//--------------------------------------------------------------------------------------------------------------------------
int main(int argc,char* argv[])
{
//--------------------------Checking for input arguments------------------------------//

	//char baseFileName[] = "~/Documents/Birds_Full/Birds_data/InterpolatedData/";
	char baseFileName[] = "../../Birds_Full/Birds_data/InterpolatedData/";
	char yearFileName[80];
	char fullFileName[80];
	char start_date[12];

	float starting_row,starting_col;
	starting_row = 0;
	starting_col = 0;

	int starting_time = 0;

	printf("\n\tStart date provided is %s %s %s\n\n",argv[1],argv[2],argv[3]);	
	printf("\n\tStart position is %s %s\n\n",argv[4],argv[5]);


	if(argc < 6){
		printf("\n\tNot enough arguments; Needed 6 provided %d \n\tUsage:\tExecutableFileName StartYear(Full year)  StartMonth(Abbr. all caps) StartDay(Without initial zeroes) StartingRowCoordinate StartingColCoordinate StartingTime(24Hrs/Military::::Ignore for now)\n\n",argc - 1);
		return 0;
	}
	else if (argc>6){
		printf("\n\tToo many arguments; Needed 6 provided %d \n\tUsage:\tExecutableFileName StartYear(Full year)  StartMonth(Abbr. all caps) StartDay(Without initial zeroes) StartingRowCoordinate StartingColCoordinate StartingTime (Without AM or PM; Or 24Hrs/Military::::Ignore for now)\n\n",argc-1);
		return 0;
	}

	starting_row = atof(argv[4]);
	starting_col = atof(argv[5]);
	//#undef STARTING_ROW	
	//#define RADIUS starting_row
//
	//#undef STARTING_COL
	//#define RADIUS starting_col

	//starting_time = atoi(argv[6]);

	//Getting the offset into the data so that user can specify a starting date
	//l in the get_movement function
	long offset_into_data = 0;
	offset_into_data = convert_to_month(argv[2],argv[3]);

	//Checking if correct year specified
	if((strcmp(argv[1],"2008")==0)||(strcmp(argv[1],"2009")==0)||(strcmp(argv[1],"2010")==0)||(strcmp(argv[1],"2011")==0)||(strcmp(argv[1],"2012")==0)||(strcmp(argv[1],"2013")==0)){
		//Add file location here
		strcpy(yearFileName,baseFileName);
		strcat(yearFileName,argv[1]);
		strcat(yearFileName,"/");
	}
	else{
		printf("\n\tInvalid year specified\n\tSpecified %s; Use years from 2008 to 2013 in its full format\n",argv[1]);
		printf("\tUsage:\tExecutableFileName StartYear(Full year)  StartMonth(Abbr. all caps) StartDay(Without initial zeroes)\n\n");
		return 0;		
	}

	
	strcpy(start_date,argv[1]);
	strcat(start_date," ");
	strcat(start_date,argv[2]);
	strcat(start_date," ");
	strcat(start_date,argv[3]);
	
	

	
/*
	//Checking if correct pressure level specified
	if((strcmp(argv[4],"750")==0)||(strcmp(argv[4],"850")==0)||(strcmp(argv[4],"925")==0)){
		//Add file location here
	}
	else{
		printf("\n\t Invalid pressure level specified\n\tSpecified %s; Use 750 850 or 925\n",argv[4]);
		return 0;		
	}
*/
//--------------------------Seeding random function-----------------------------------//

	struct timeval t1;
	gettimeofday(&t1,NULL);
	srand((t1.tv_sec*1000)+(t1.tv_usec/1000));
	//The wind data is in m/s


//--------------------------Memory Allocation-----------------------------------//
	float* udata;
	udata = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	float* vdata;
	vdata = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));

	float* u10data;
	u10data = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	float* v10data;
	v10data = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));

	float* precipData;
	precipData = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	float* pressureData;
	pressureData = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));

	float* dirData;
	dirData = (float*)malloc(LAT_SIZE * LONG_SIZE * sizeof(float));	
	int* lwData;
	lwData = (int*)malloc(LAT_SIZE * LONG_SIZE * sizeof(int));
	
//------------Opening position data file where lat and long data will be stored----------------//
	
	FILE *posdataTxt,*vdataTxt,*udataTxt,*v10dataTxt,*u10dataTxt,*precipTxt,*pressureTxt,*lwTxt,*dirTxt;
	posdataTxt = fopen("posdata.txt","a");
	if(posdataTxt == NULL) {
		perror("Cannot open position data file\n");
		return -1;
	}
//----------------------Opening U850 data file----------------------------//
	memset(fullFileName,0,strlen(fullFileName));
	strcpy(fullFileName,yearFileName);
	strcat(fullFileName,"U850.txt");

	printf("U50 filename is %s \n",fullFileName);
	udataTxt = fopen(fullFileName,"r");

	if(udataTxt == NULL) {
		perror("Cannot open file with U850 data\n");
		return -1;
	}
//------------------------Opening V850 data file--------------------------//
	memset(fullFileName,0,strlen(fullFileName));
	strcpy(fullFileName,yearFileName);
	strcat(fullFileName,"V850.txt");

	vdataTxt = fopen(fullFileName,"r");

	if(vdataTxt == NULL) {
		perror("Cannot open file with V850 data\n");
		return -1;
	}
//-----------------------Opening U10 data file---------------------------//
	//Birds will check the wind at the surface therefore the u and v
	//at 10m is required
	
	memset(fullFileName,0,strlen(fullFileName));
	strcpy(fullFileName,yearFileName);
	strcat(fullFileName,"U10.txt");

	u10dataTxt = fopen(fullFileName,"r");

	if(u10dataTxt == NULL) {
		perror("Cannot open file with U10 data\n");
		return -1;
	}
//-----------------------Opening V10 data file---------------------------//
	memset(fullFileName,0,strlen(fullFileName));
	strcpy(fullFileName,yearFileName);
	strcat(fullFileName,"V10.txt");

	v10dataTxt = fopen(fullFileName,"r");
	
	if(v10dataTxt == NULL) {
		perror("Cannot open file with V10 data\n");
		return -1;
	}
//--------------------Opening PRCP data file------------------------------//
	memset(fullFileName,0,strlen(fullFileName));
	strcpy(fullFileName,yearFileName);
	strcat(fullFileName,"PRCP.txt");

	precipTxt = fopen(fullFileName,"r");
	if(precipTxt == NULL) {
		perror("Cannot open file with PRCP data\n");
		return -1;
	}
//------------------------Opening MSLP data file--------------------------//
	memset(fullFileName,0,strlen(fullFileName));
	strcpy(fullFileName,yearFileName);
	strcat(fullFileName,"MSLP.txt");

	pressureTxt = fopen(fullFileName,"r");
	if(pressureTxt == NULL) {
		perror("Cannot open file with pressure data!\n");
		return -1;
	}
//--------------------------Opening Land vs Water File---------------------//
	lwTxt = fopen("./Lw_and_Dir/land_water_detail.txt","r");
	if(lwTxt == NULL) {
		perror("Cannot open file with direction data\n");
		return -1;
	}
//--------------------------Opening Direction file (Example: ext_crop.txt or extP_crop.txt)-------------//
	dirTxt = fopen("./Lw_and_Dir/ext_Final.txt","r");
	//dirTxt = fopen("ext_crop.txt","r");
	if(dirTxt == NULL) {
		perror("Cannot open file with direction data\n");
		return -1;
	}
//--------------------------Direction file code end-------------------------------------------------------------------//
	

	FILE* inpCheckU;
	inpCheckU = fopen("inpCheckU.txt","w");
	if(inpCheckU == NULL) {
		perror("Cannot open inpCheckU file\n");
		return -1;
	}
	FILE* inpCheckV;
	inpCheckV = fopen("inpCheckV.txt","w");
	if(inpCheckV == NULL) {
		perror("Cannot open inpCheckV file\n");
		return -1;
	}

	char line[LINESIZE];
	memset(line,'\0',sizeof(line));
	char tempVal[15];
	memset(tempVal,'\0',sizeof(tempVal));
	char* startPtr,*endPtr;
	long j;
	int i;
	float Value;
	i=0;
	j=0;

//-------------------------------Reading U850 & V850 values-----------------------------------//
	while(fgets(line,LINESIZE,udataTxt)!=NULL){
		startPtr = line;
		for(i=0;i<LONG_SIZE;i++){
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));
			if(i != (LONG_SIZE - 1)) {
				endPtr = strchr(startPtr,',');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal);
				udata[j * LAT_SIZE + i] = Value;
				
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (LONG_SIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
				udata[j * LAT_SIZE + i] = Value;
				
			}
		}
		j++;
	}
	memset(line,'\0',sizeof(line));
	memset(tempVal,'\0',sizeof(tempVal));
	i=0;
	j=0;

	while(fgets(line,LINESIZE,vdataTxt)!=NULL){
		startPtr = line;
		for(i=0;i<LAT_SIZE;i++){
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));
			if(i != (LAT_SIZE - 1)) {
				endPtr = strchr(startPtr,',');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal);
				vdata[j * LAT_SIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (LAT_SIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
				vdata[j * LAT_SIZE + i] = Value;
			}
		}
		j++;
	}

//-------------------------------Reading U10 & V10 values-----------------------------------//
	memset(line,'\0',sizeof(line));
	memset(tempVal,'\0',sizeof(tempVal));
	i=0;
	j=0;
	
	while(fgets(line,LINESIZE,u10dataTxt)!=NULL){
		startPtr = line;
		for(i=0;i<LAT_SIZE;i++){
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));
			if(i != (LAT_SIZE - 1)) {
				endPtr = strchr(startPtr,',');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal);
				u10data[j * LAT_SIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (LAT_SIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
				u10data[j * LAT_SIZE + i] = Value;
			}
		}
		j++;
	}
	memset(line,'\0',sizeof(line));
	memset(tempVal,'\0',sizeof(tempVal));
	i=0;
	j=0;

	while(fgets(line,LINESIZE,v10dataTxt)!=NULL){
		startPtr = line;
		for(i=0;i<LAT_SIZE;i++){
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));
			if(i != (LAT_SIZE - 1)) {
				endPtr = strchr(startPtr,',');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal);
				v10data[j * LAT_SIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (LAT_SIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
				v10data[j * LAT_SIZE + i] = Value;
			}
		}
		j++;
	}

//----------------------------------Reading precipitation Values-----------------------------//


// Precipitation value read from the text file is multiplied by 3600 to convert from
// kg/(m2*s1) into mm/hour. Formula from: https://www.dkrz.de/daten-en/faq

	memset(line,'\0',sizeof(line));
	memset(tempVal,'\0',sizeof(tempVal));
	i=0;
	j=0;



	while(fgets(line,LINESIZE,precipTxt)!=NULL){
		startPtr = line;
		for(i=0;i<LAT_SIZE;i++){
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));
			if(i != (LAT_SIZE - 1)) {
				endPtr = strchr(startPtr,',');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal)*3600;
				precipData[j * LAT_SIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (LAT_SIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atof(tempVal)*3600;
				precipData[j * LAT_SIZE + i] = Value;
			}
		}
		j++;
	}
//-----------------------------------Reading Pressure Values---------------------------------//

	memset(line,'\0',sizeof(line));
	memset(tempVal,'\0',sizeof(tempVal));
	i=0;
	j=0;

	while(fgets(line,LINESIZE,pressureTxt)!=NULL){
		startPtr = line;
		for(i=0;i<LAT_SIZE;i++){
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));
			if(i != (LAT_SIZE - 1)) {
				endPtr = strchr(startPtr,',');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal);
				pressureData[j * LAT_SIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (LAT_SIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
				pressureData[j * LAT_SIZE + i] = Value;
			}
		}
		j++;
	}
//-----------------------------------Reading Direction Values---------------------------------//
	memset(line,'\0',sizeof(line));
	memset(tempVal,'\0',sizeof(tempVal));
	i=0;
	j=0;

	while(fgets(line,LINESIZE,dirTxt)!=NULL){
		startPtr = line;
		for(i=0;i<LAT_SIZE;i++){
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));
			if(i != (LAT_SIZE - 1)) {
				endPtr = strchr(startPtr,',');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal);
				dirData[j * LAT_SIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (LAT_SIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
				dirData[j * LAT_SIZE + i] = Value;
			}
		}
		j++;
	}


	for(j=0;j<LONG_SIZE;j++) {
		for(i=0;i<LAT_SIZE;i++) {
			if(i == LAT_SIZE -1) {
				fprintf(inpCheckU,"%f\n",precipData[j * LAT_SIZE + i]);
				//fprintf(inpCheckV,"%f\n",=data[j * LAT_SIZE + i]);
				//printf("%f\n",dirData[j * LAT_SIZE + i]);
			}
			else {
			fprintf(inpCheckU,"%f ",precipData[j * LAT_SIZE + i]);
			//fprintf(inpCheckV,"%f ",vdata[j * LAT_SIZE + i]);
			}
		}
	}

//-----------------------------------Reading Land vs Water Values---------------------------------//


	memset(line,'\0',sizeof(line));
	memset(tempVal,'\0',sizeof(tempVal));
	i=0;
	j=0;

	while(fgets(line,LINESIZE,lwTxt)!=NULL){
		startPtr = line;
		for(i=0;i<LAT_SIZE;i++){
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));
			if(i != (LAT_SIZE - 1)) {
				endPtr = strchr(startPtr,',');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				if(strcmp("NaN",tempVal)==0){
					Value = 255;
				}else{
					Value = atoi(tempVal);
				}
				lwData[j * LAT_SIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (LAT_SIZE - 1)){
				strcpy(tempVal,startPtr);
				if(strcmp("NaN",tempVal)==0){
					Value = 255;
				}else{
					Value = atoi(tempVal);
				}
				lwData[j * LAT_SIZE + i] = Value;
			}
		}
		j++;
	}

//-----------------------------------Checking if input is correct-------------------------------//
	for(j=0;j<LONG_SIZE;j++) {
		for(i=0;i<LAT_SIZE;i++) {
			if(i == LAT_SIZE -1) {
				fprintf(inpCheckU,"%f\n",precipData[j * LAT_SIZE + i]);
				//fprintf(inpCheckV,"%f\n",=data[j * LAT_SIZE + i]);
				//printf("%f\n",dirData[j * LAT_SIZE + i]);
			}
			else {
			fprintf(inpCheckU,"%f ",precipData[j * LAT_SIZE + i]);
			//fprintf(inpCheckV,"%f ",vdata[j * LAT_SIZE + i]);
			}
		}
	}

//-----------------------------------Execute bird movement function-------------------------------//

	//for(i = 0;i<sizeof(rows);i++){
		//STARTING_ROW = rows[i];
		//STARTING_COL = cols[i];
		
		//fprintf(posdataTxt,"Starting location(row,col) = (%f,%f) \n Starting Day = %s %s %s\n",STARTING_ROW,STARTING_COL,argv[1],argv[2],argv[3]);
		get_movementData(posdataTxt,starting_row,starting_col,offset_into_data,udata,vdata,dirData,precipData,pressureData,u10data,v10data,lwData,start_date,argv[1]);
	//}



//-----------------------------------------------Freeing allocated memory----------------------------//
	free(udata);
	free(vdata);
	free(u10data);
	free(v10data);
	free(dirData);
	free(precipData);
	free(pressureData);
	free(lwData);
	


	fclose(udataTxt);
	fclose(vdataTxt);
	fclose(u10dataTxt);
	fclose(v10dataTxt);

	fclose(posdataTxt);
	fclose(precipTxt);
	fclose(dirTxt);
	fclose(pressureTxt);
	fclose(inpCheckU);
	fclose(inpCheckV);

	printf("End\n");
	return 0;
}
//------------------------------------------------------------------------------------------------------------------------------------







