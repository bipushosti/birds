 
//This is a copy of First_run_6hrsOnly.c but edits made to make it suitable 
//to change into a CUDA code. 
//This file uses 6 hourly data. Each day is 6 hours long and skipping a day means to add 6
//to the counter that counts the timesteps (l).

#include <math.h>
#include <float.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_math.h>

//#include <GL/glut.h>

#define PI 			3.14159
#define LONG_SIZE		429
#define LAT_SIZE		429
#define LINESIZE		15*LAT_SIZE+LAT_SIZE - 3
#define TIMESTEPS		30*6
#define SKIP_TIMESTEPS		18
//#define DESIRED_ROW
//#define DESIRED_COL
#define STARTING_ROW		110.0
#define STARTING_COL		150.0

#define STOPOVER_DAYS		0

//#define DESIRED_SPEED	3.6		//Birds want to travel at 10m/s, it is 36km/hr(in the grid it is 3.6 units per hour) 
	
#define DESIRED_SPEED		10.5	//Air speed; Desired speed = flightspeed + windspeed ; Only used in windprofit calculation

#define STD_BIRDANGLE		5	//Standard deviation * 6 = the total difference from max to min angle possible
					//If STD_BIRDANGLE = 10 then the angle can differ +- (10*6)/2 = +- 30 from mean
#define	glCompAcc		1e-8	//If the difference is equal to or less than this then equal

#define MIN_PROFIT		-7
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
//------------------------------Notes---------------------------------------------------------------------------------------
/*
Altitude = 850 millibars
Year = 2009
22 Jan 2015 No upper limit to the bird flight speed currently; Birds can fly well above 10m/s
Precipitation = millimeters
*/

float rows[11]={140,141,142,143,144,145,146,147,148,149,150};
float cols[11]={150,151,152,153,154,155,156,157,158,159,160};

//float rows[11]={100,99,98,97,96,95,94,93,92,91,90};
//float cols[11]={250,250,250,250,250,250,250,250,250,250,250};

//l stands for timesteps; i stands for index inside the data
long int starting_l = 0;
long int l = SKIP_TIMESTEPS;
long int l_old = SKIP_TIMESTEPS;
long int i;


//--------------------------------------------------------------------------------------------------------------------------
void get_movementData(FILE* outTxt,float* udata,float* vdata,float* dirData,float* precipData,float* pressureData,float* u10data,float* v10data);
float getProfitValue(float u,float v,float dirVal,float dir_u,float dir_v);
float bilinear_interpolation(float x,float y,float* data_array,long l,int dataSize);
float WrappedNormal (float MeanAngle,float AngStdDev);

//-------------------------------------------------------------------------------------------------------------------------------------
//The Kernel
void get_movementData(FILE* outTxt,float* udata,float* vdata,float* dirData, float* precipData,float* pressureData,float* u10data,float* v10data)
{
	fprintf(outTxt,"pos_row \t pos_col \t u_val \t\t v_val \t\t dir_u \t\t dir_v \t\t u_air \t\t v_air \t\t bird_GroundSpeed \t wind_Speed \t\t distance \t l \t skip\n");
	float distance,prev_row,prev_col,bird_AirSpeed,wind_Speed;
	distance = 0;
	bird_AirSpeed = 0;
	wind_Speed = 0;
	prev_row = STARTING_ROW;
	prev_col = STARTING_COL;

	float pos_row,pos_col;
	//pos_row = LONG_SIZE - STARTING_ROW;
	pos_row = STARTING_ROW;
	pos_col = STARTING_COL;
	fprintf(outTxt,"%f,%f\n",pos_row,pos_col);
	int k;
	//long l_old;

	float pressure_sum,pressure_MultSum,last_pressure,slope;
	pressure_MultSum = 0;
	pressure_sum = 0;
	slope = 1;
	last_pressure = 1011;
	//l = 18;
	//l_old = 18;
	float profit_value,dirAngleFromFile,dirAngle,actualAngle;

	
	float dir_v,dir_u;
	//long skip_size = (SKIP_TIMESTEPS * LONG_SIZE * LAT_SIZE) - 1;
	long skip_size = 0;
	

	float u_val,v_val,precip_val,v_ten,u_ten;
	int skip;
	skip=0;
	//skip_size = 120174
	
	//fprintf(outTxt,"%f,%f\n",pos_row,pos_col);

	printf("i \t l \t k \t precipData \t profit_value \t pos_row \t pos_col \t (v_val,u_val)\n");
	//i = skip_size +pos_row * LAT_SIZE + pos_col;
	//while( i <= (TIMESTEPS-1) * LAT_SIZE * LONG_SIZE ) {
		dir_v = 0;
		dir_u = 0;
		dirAngle = 0;
		dirAngleFromFile = 0;
		actualAngle = 0;
		u_val = 0;
		v_val = 0;
		precip_val = 0;
		v_ten = 0;
		u_ten = 0;
		pressure_sum = 0;
		pressure_MultSum = 0;
		//If current pressure is greater than pressure in the previous day
		//if(pressureData[i] - old_pressure > 0) {
		printf("Main loop[k]\n");

		u_ten = bilinear_interpolation(pos_col,pos_row,u10data,l,1);
		v_ten = bilinear_interpolation(pos_col,pos_row,v10data,l,1);
		
		//The direction angle is chosen only once, before the flight.
		dirAngleFromFile = dirData[(int)(rintf(pos_row) * LAT_SIZE + rintf(pos_col))];
		dirAngle = WrappedNormal(dirAngleFromFile,STD_BIRDANGLE);
		actualAngle = dirAngle;

		if(dirAngle <= 90){
			dirAngle = 90 - dirAngle;
			dir_v = DESIRED_SPEED * sin(dirAngle * (PI/180));
			dir_u = DESIRED_SPEED * cos(dirAngle * (PI/180));
		}
		else if((dirAngle > 90) && (dirAngle <= 180)){
			dirAngle -= 90;
			dir_v = DESIRED_SPEED * sin(dirAngle * (PI/180)) * -1;
			dir_u = DESIRED_SPEED * cos(dirAngle * (PI/180));
		}
		else if((dirAngle > 180) && (dirAngle <= 270)) {
			dirAngle = 270 - dirAngle;
			dir_v = DESIRED_SPEED * sin(dirAngle * (PI/180)) * -1;
			dir_u = DESIRED_SPEED * cos(dirAngle * (PI/180)) * -1;
		}
		else if((dirAngle > 270) && (dirAngle <= 360)){
			dirAngle -= 270;
			dir_v = DESIRED_SPEED * sin(dirAngle * (PI/180));
			dir_u = DESIRED_SPEED * cos(dirAngle * (PI/180)) * -1;
		}

		printf("10 profit value::%f\n",getProfitValue(u_ten,v_ten,actualAngle,dir_u,dir_v));
		printf("pressure value::%f,slope value::%f\n",last_pressure,slope);


		dirAngleFromFile = dirData[(int)(rintf(pos_row) * LAT_SIZE + rintf(pos_col))];
		dirAngle = WrappedNormal(dirAngleFromFile,STD_BIRDANGLE);
		printf("\n First DirectionAngle = %f,AngleFromFile = %f\n",dirAngle,dirAngleFromFile);
		actualAngle = dirAngle;


		//Relation between last_pressure and slope is an OR
		if((getProfitValue(u_ten,v_ten,actualAngle,dir_u,dir_v) >= MIN_PROFIT) && ((last_pressure>=1009)||(slope >-1))){
			

			//dirAngleFromFile = dirData[(int)(rintf(pos_row) * LAT_SIZE + rintf(pos_col))];
			//dirAngle = WrappedNormal(dirAngleFromFile,STD_BIRDANGLE);
			
			printf("\n\nl value check: %ld\n\n",l);
			
			for(k=0;k<6;k++,l++ ) {
				i = skip_size + l * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col;
				skip = 0;

				//dirAngle is with respect to North or the positive y-axis
				//It is the genetic direction of the birds

				//actualAngle = dirAngle;
				//dirAngleFromFile = dirData[(int)(rintf(pos_row) * LAT_SIZE + rintf(pos_col))];
				//dirAngle = WrappedNormal(dirAngleFromFile,STD_BIRDANGLE);
				

				dirAngle = actualAngle;
				printf("\n DirectionAngle = %f,AngleFromFile = %f\n",dirAngle,dirAngleFromFile);
				//The grid is upside down; y increases from top to bottom while x increases from left to right 
				//dir_v and dir_u are the x and y components of the wind (v=y,u=x)
				if(dirAngle <= 90){//Checked OK
					dirAngle = 90 - dirAngle;
					dir_v = DESIRED_SPEED * sin(dirAngle * (PI/180));
					dir_u = DESIRED_SPEED * cos(dirAngle * (PI/180));
				}
				else if((dirAngle > 90) && (dirAngle <= 180)){//Checked OK
					dirAngle -= 90;
					dir_v = DESIRED_SPEED * sin(dirAngle * (PI/180)) * -1;
					dir_u = DESIRED_SPEED * cos(dirAngle * (PI/180));
				}
				else if((dirAngle > 180) && (dirAngle <= 270)) {
					dirAngle = 270 - dirAngle;
					dir_v = DESIRED_SPEED * sin(dirAngle * (PI/180)) * -1;
					dir_u = DESIRED_SPEED * cos(dirAngle * (PI/180)) * -1;
				}
				else if((dirAngle > 270) && (dirAngle <= 360)){
					dirAngle -= 270;
					dir_v = DESIRED_SPEED * sin(dirAngle * (PI/180));
					dir_u = DESIRED_SPEED * cos(dirAngle * (PI/180)) * -1;
				}


				//Bilinear interpolation for u and v data
				u_val = bilinear_interpolation(pos_col,pos_row,udata,l,1);	
				v_val = bilinear_interpolation(pos_col,pos_row,vdata,l,1);

				printf("(u_val,v_val)::(%f,%f)\n",u_val,v_val);

				profit_value = getProfitValue(u_val,v_val,actualAngle,dir_u,dir_v);
				precip_val = bilinear_interpolation(pos_col,pos_row,precipData,l,1);


				//Adding precip value
				//if ((profit_value >= MIN_PROFIT) && (precipData[i] < 2) ) {
				if ((profit_value >= MIN_PROFIT) && (precip_val < MAX_PRECIP) ) {
				

					//Positon is in a 10 km grid. This means for 1m/s, the 
					//position change in 1 hour is 3.6/10 = 0.36units in the grid
	//				pos_row = pos_row + (v_val + dir_v)*0.36;
					pos_row = pos_row + (v_val + dir_v)*0.36*-1;
					pos_col = pos_col + (u_val + dir_u)*0.36;

					//float tmp;
					//tmp = sqrtf((v_val+dir_v)*(v_val+dir_v) + (u_val+dir_u)*(u_val+dir_u));
					//printf("\nTailComponent::%f,Speed::%f,dir_v::%f,dir_u::%f\n",tailComponent,tmp,dir_v,dir_u);
				
					printf("%ld \t %ld \t %d \t %f \t %f \t %f \t %f \t (%f,%f)\n",i,l,k,precip_val,profit_value,pos_row,pos_col,v_val,u_val);
					skip = 0;
				}
				else {
					//l increases but it goes back to the original starting hour for the bird; i.e 7pm
					// 6-k because l++ will not be done in the end because it breaks from the loop
					l += (6-k);
					skip = 1;
					
					printf("Skipped Wind (%f,%f) @ (%f,%f)w_profit = %f,precip=%f,And l = %ld\n",u_val,v_val,pos_row,pos_col,profit_value,precip_val,l);
					break;
				}
				//fprintf(outTxt,"%f,%f\n",pos_row,pos_col);
				distance = sqrt((pos_row - prev_row)*(pos_row - prev_row) + (pos_col - prev_col)*(pos_col - prev_col));
				prev_col = pos_col;
				prev_row = pos_row;
		
				wind_Speed = sqrt(u_val * u_val + v_val * v_val);
				bird_AirSpeed = sqrt((u_val+dir_u)*(u_val+dir_u) +(v_val+dir_v)*(v_val+dir_v));
				//Distance is in absolute value (kilometers rather than in units of grid points)
				fprintf(outTxt,"%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t%ld\t%d\n",pos_row,pos_col,u_val,v_val,dir_u,dir_v,u_val+dir_u,v_val+dir_v,bird_AirSpeed,wind_Speed,distance*10,l,skip);
				
				
			}
		}
		
		//v10 and u10 profit values were too low
		if(l_old == l){
			printf("u10 v10 profit value too low!\n");
			l+=6;

		}
		//Every day has 6 time steps and this line skips the total desired number of days
		l += STOPOVER_DAYS * 6;
		l_old = l - REGRESSION_HRS;

		printf("check l %ld\n",l);

		//Taking the pressure from 6 hours earlier of the location where the bird landed
		for(k=1; (l_old < l) && (k<=REGRESSION_HRS); l_old++,k++){

			//printf("\nPressure Sum Interpolation\n");
			pressure_sum += bilinear_interpolation(pos_col,pos_row,pressureData,l_old,1);
			//pressure_sum += pressureData[skip_size + l_old * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col];
			pressure_MultSum += k * bilinear_interpolation(pos_col,pos_row,pressureData,l_old,1);
			//printf("%f\n",pressureData[skip_size + l_old * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col]);

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
		//Since the data now does not contain all 24 hours but just 7pm to 1am
		//l += 6;
		i = skip_size + l * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col;
		//l_old = l;
		//printf("Running\n");
		printf("-----------------------------------------------------------------------------------------------------------------------\n");

	//}
	//fprintf(outTxt,"----------------------------------------------------------\n");
}

//--------------------------------------------------------------------------------------------------------------------------
int main()
{
	struct timeval t1;
	gettimeofday(&t1,NULL);
	srand((t1.tv_sec*1000)+(t1.tv_usec/1000));
	//The wind data is in m/s
	float* udata;
	udata = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	float* vdata;
	vdata = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));

	//--------------------------U10 & V10-----------------------------------//
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

	FILE *posdataTxt;
	posdataTxt = fopen("posdata.txt","w");
	if(posdataTxt == NULL) {
		perror("Cannot open udataTxt file\n");
		return -1;
	}
	FILE *vdataTxt,*udataTxt;
	udataTxt = fopen("../Birds_data/output/U850_30days_Sept_2011.txt","r");
	vdataTxt = fopen("../Birds_data/output/V850_30days_Sept_2011.txt","r");
	if(udataTxt == NULL) {
		perror("Cannot open file with U850 data\n");
		return -1;
	}
	if(vdataTxt == NULL) {
		perror("Cannot open file with V850 data\n");
		return -1;
	}

	//Birds will check the wind at the surface therefore the u and v
	//at 10m is required
	FILE *v10dataTxt,*u10dataTxt;
	u10dataTxt = fopen("../Birds_data/output/U10_30days_Sept_2011.txt","r");
	v10dataTxt = fopen("../Birds_data/output/V10_30days_Sept_2011.txt","r");
	if(u10dataTxt == NULL) {
		perror("Cannot open file with U10 data\n");
		return -1;
	}
	if(v10dataTxt == NULL) {
		perror("Cannot open file with V10 data\n");
		return -1;
	}


//--------------------------Direction file (Example: ext_crop.txt or extP_crop.txt)-----------------------------------//
	FILE* dirTxt;
	dirTxt = fopen("extP_cropnew.txt","r");
	//dirTxt = fopen("ext_crop.txt","r");
	if(dirTxt == NULL) {
		perror("Cannot open file with direction data\n");
		return -1;
	}
//--------------------------Direction file code end-------------------------------------------------------------------//
	FILE* precipTxt;
	precipTxt = fopen("../Birds_data/output/PRCP_30days_Sept_2011.txt","r");
	if(precipTxt == NULL) {
		perror("Cannot open file with PRCP data\n");
		return -1;
	}
	FILE* pressureTxt;
	pressureTxt = fopen("../Birds_data/output/MSLP_30days_Sept_2011.txt","r");
	if(pressureTxt == NULL) {
		perror("Cannot open file with pressure data!\n");
		return -1;
	}

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
		for(i=0;i<LAT_SIZE;i++){
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));
			if(i != (LAT_SIZE - 1)) {
				endPtr = strchr(startPtr,',');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal);
				udata[j * LAT_SIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (LAT_SIZE - 1)){
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

	//for(i = 0;i<sizeof(rows);i++){
		//STARTING_ROW = rows[i];
		//STARTING_COL = cols[i];
		
		//long skip_size = (SKIP_TIMESTEPS * LONG_SIZE * LAT_SIZE) - 1;
		//Actual index skipped inside the data matrix
		long skip_size = 0;
		i = skip_size +STARTING_ROW* LAT_SIZE + STARTING_COL;
		while(l < TIMESTEPS){
			get_movementData(posdataTxt,udata,vdata,dirData,precipData,pressureData,u10data,v10data);
		}
	//}




	free(udata);
	free(vdata);
	free(u10data);
	free(v10data);
	free(dirData);
	free(precipData);
	free(pressureData);


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

//------------------------------------------------------------------------------------------------------------------------------------

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
		else if((v_val>0)&&(u_val>0)){	//Quadrant 1
			u_val = abs(u_val);
			v_val = abs(v_val);
			angle = atanf(u_val/v_val) * (180/PI);
			diffAngle = abs(angle - dirVal);
		}
		else if((v_val<0)&&(u_val>0)){	//Quadrant 2
			u_val = abs(u_val);
			v_val = abs(v_val);
			angle = atanf(v_val/u_val) * (180/PI) + 90;
			diffAngle = abs(angle - dirVal);
		}
		else if((v_val<0)&&(u_val<0)){	//Quadrant 3
			u_val = abs(u_val);
			v_val = abs(v_val);
			angle = atanf(u_val/v_val) * (180/PI) + 180;
			diffAngle = abs(angle - dirVal);
		}
		else if((v_val>0)&&(u_val<0)){	//Quadrant 4
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
//From: http://www.ajdesigner.com/phpinterpolation/bilinear_interpolation_equation.php

//dataSize: boolean value; either 0 or 1; 
//		0 means the data is small as in LAT*LONG;
//		1 means the data is big as in TIMESTEPS * LAT * LONG

//Incorrect; Getting larger values when surrounded by smaller
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
		Q11 = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y1 * LAT_SIZE + x1) ];
		Q12 = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y2 * LAT_SIZE + x1) ];
		Q21 = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y1 * LAT_SIZE + x2) ];
		Q22 = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y2 * LAT_SIZE + x2) ];
	}
	else if(dataSize == 0) {
		Q11 = data_array[(int)(y1 * LAT_SIZE + x1)];
		Q12 = data_array[(int)(y2 * LAT_SIZE + x1)];
		Q21 = data_array[(int)(y1 * LAT_SIZE + x2)];
		Q22 = data_array[(int)(y2 * LAT_SIZE + x2)];
	}

	if((x2 == x1) && (y2 == y1)){
		value = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y1 * LAT_SIZE + x1)];
	}
	else if((x2 == x1) && (y2 != y1)){
			if(dataSize == 0) {
				value+= data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y1 * LAT_SIZE + x)];
				value+=((y-y1)/(y2-y1))*(data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y2 * LAT_SIZE + x)]-data_array[(int)(l*LAT_SIZE * LONG_SIZE + y1 * LAT_SIZE + x)]);
			}
			else if(dataSize == 1){
				value+= data_array[(int)(y1 * LAT_SIZE + x)];
				value+=((y-y1)/(y2-y1))*(data_array[(int)(y2 * LAT_SIZE + x)]-data_array[(int)(y1 * LAT_SIZE + x)]);
			}
	}
	else if((x2 != x1) && (y2 == y1)){
			if(dataSize == 0) {
				value+= data_array[(int)(l  * LAT_SIZE * LONG_SIZE + x1 * LAT_SIZE + y)];
				value+=((x-x1)/(x2-x1))*(data_array[(int)(l  * LAT_SIZE * LONG_SIZE + x2 * LAT_SIZE + y)]-data_array[(int)(l*LAT_SIZE*LONG_SIZE + x1 * LAT_SIZE + y)]);
			}
			else if(dataSize == 1){
				value+= data_array[(int)(x1 * LAT_SIZE + y)];
				value+=((x-x1)/(x2-x1))*(data_array[(int)(x2 * LAT_SIZE + y)]-data_array[(int)(x1 * LAT_SIZE + y)]);
			}
	}

	else{
		value =((x2-x)* ((y2-y)*Q11 + (y-y1)*Q12)+(x-x1)*((y2-y)*Q21 +(y-y1)*Q22))/((x2-x1)*(y2-y1));
	}
	//printf("Q11:%f,Q12:%f,Q21:%f,Q22:%f; And Value=%f\n",Q11,Q12,Q21,Q22,value);
	return value;
}
//-------------------------------------------------------------------------------------------------------------------------------------
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
			printf("u1:%f,u2:%f\n",u1,u2);
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
			//printf("Second\n");
			break;
			
		}
	}//while(check == 1);
	

	y = AngStdDev * z + MeanAngle;
	if ((y - 360) > -glCompAcc){ 
	    y = y - 360;
	}
 
	if (y < 0){
	    y = 360 + y;
	}
	
	//printf("Last \n");
	return y;

  
}





