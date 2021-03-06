

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdlib.h>
#include <GL/glut.h>

#define PI 3.14159
#define LONG_SIZE	429
#define LAT_SIZE	429
#define LINESIZE	15*LAT_SIZE+LAT_SIZE - 3
#define TIMESTEPS	30*6
#define SKIP_TIMESTEPS	0
//#define DESIRED_ROW
//#define DESIRED_COL
#define STARTING_ROW	150.0
#define STARTING_COL	150.0

#define STOPOVER_DAYS	0

//#define DESIRED_SPEED	3.6		//Birds want to travel at 10m/s, it is 36km/hr(in the grid it is 3.6 units per hour) 
#define DESIRED_SPEED	10

	
#define MIN_PROFIT	-7
//Defining the x-variable size, it's sum and
//sum of squares as needed for slope calculation

//Since the 24 hour data is not and cannot be included due to
//its size the regression hours currently are from the previous night
//at that point.A new text file has to be created that has the pressure trend
//value for the last 12/24/6 hours at that point for each point in the map 
//for each take off time(6pm or 7pm) instead of including all the pressure data files.
//This helps in reducing the size of the data.

#define REGRESSION_HRS	6

//Precipitation (mm/hr) below which birds can fly
#define MAX_PRECIP	2

//HRS_SUM = sum(1 to 12) before. Now has to be sum(1 to 6) = 21
#define HRS_SUM	21
#define HRS_SQUARE_SUM	91
#define DENOM_SLOPE	(REGRESSION_HRS * HRS_SQUARE_SUM)-(HRS_SUM * HRS_SUM)
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
//--------------------------------------------------------------------------------------------------------------------------
void get_movementData(FILE* outTxt,float* udata,float* vdata,float* dirData,float* precipData,float* pressureData,float* u10data,float* v10data);
float getProfitValue(float u,float v,float dirVal,float dir_u,float dir_v);
float bilinear_interpolation(float x,float y,float* data_array,long l,int dataSize);
//--------------------------------------------------------------------------------------------------------------------------
int main()
{
	size_t limit;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 500 * 1024 * 1024);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
	cudaDeviceGetLimit(&limit,cudaLimitPrintfFifoSize);
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
	udataTxt = fopen("../Birds_data/U850_30days_Sept_2011.txt","r");
	vdataTxt = fopen("../Birds_data/V850_30days_Sept_2011.txt","r");
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
	u10dataTxt = fopen("../Birds_data/U10_30days_Sept_2011.txt","r");
	v10dataTxt = fopen("../Birds_data/V10_30days_Sept_2011.txt","r");
	if(u10dataTxt == NULL) {
		perror("Cannot open file with U10 data\n");
		return -1;
	}
	if(v10dataTxt == NULL) {
		perror("Cannot open file with V10 data\n");
		return -1;
	}

	FILE* dirTxt;
	dirTxt = fopen("extP_crop.txt","r");
	if(dirTxt == NULL) {
		perror("Cannot open file with direction data\n");
		return -1;
	}
	FILE* precipTxt;
	precipTxt = fopen("../Birds_data/PRCP_30days_Sept_2011.txt","r");
	if(precipTxt == NULL) {
		perror("Cannot open file with PRCP data\n");
		return -1;
	}
	FILE* pressureTxt;
	pressureTxt = fopen("../Birds_data/MSLP_30days_Sept_2011.txt","r");
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
	get_movementData(posdataTxt,udata,vdata,dirData,precipData,pressureData,u10data,v10data);


	/*
	float resultantMatrix[LONG_SIZE * LAT_SIZE];
	float resultantAngle[LONG_SIZE * LAT_SIZE];
	resultantMatrix[LAT_SIZE * LONG_SIZE -1] = 834.0;
	float *udataPtr,*vdataPtr,*resultantMatrixPtr,*resultantAnglePtr;
	cudaMalloc((void**)&udataPtr,LONG_SIZE * LAT_SIZE * sizeof(float));
	cudaMemcpy(udataPtr,udata,LONG_SIZE * LAT_SIZE * sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&vdataPtr,LONG_SIZE * LAT_SIZE * sizeof(float));
	cudaMemcpy(vdataPtr,vdata,LONG_SIZE * LAT_SIZE * sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&resultantMatrixPtr,LONG_SIZE * LAT_SIZE * sizeof(float));
	cudaMalloc((void**)&resultantAnglePtr,LONG_SIZE * LAT_SIZE * sizeof(float));
	cudaMemcpy(vdataPtr,vdata,LONG_SIZE * LAT_SIZE * sizeof(float),cudaMemcpyHostToDevice);
	dim3 gridSize(1,LONG_SIZE,1);
	dim3 blockSize((LAT_SIZE/32 +1)*32 ,1,1);
	printf("Hello2\n");
	get_resultant<<<gridSize,blockSize>>>(udataPtr,vdataPtr,resultantMatrixPtr,resultantAnglePtr);
	printf("Hello3\n");
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	printf("CUDA Error: %s\n", cudaGetErrorString(error));
	// we can't recover from the error -- exit the program
	return 0;
	}
	cudaMemcpy(resultantMatrix,resultantMatrixPtr,LAT_SIZE * LONG_SIZE * sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(resultantAngle,resultantAnglePtr,LAT_SIZE * LONG_SIZE * sizeof(float),cudaMemcpyDeviceToHost);
	cudaFree(udataPtr);
	cudaFree(vdataPtr);
	cudaFree(resultantMatrixPtr);
	cudaFree(resultantAnglePtr);
	printf("U v Magnitude Angle\n");
	for(j=0;j<LAT_SIZE*LONG_SIZE;j++) {
	printf("%f,%f,%f,%f\n",udata[j],vdata[j],resultantMatrix[j],resultantAngle[j]);
	}
	printf("(5,5)::%f,%f",udata[5 * LAT_SIZE + 5],vdata[5 * LAT_SIZE + 5]);
	get_movementData(posdataTxt,resultantMatrix,resultantAngle,udata,vdata);
	*/
	//--------------------------------------------------------------------------------------------------------------------------
	/*int coords_row[TIMESTEPS];
	int coords_col[TIMESTEPS];
	int *coords_rowPtr,*coords_colPtr;
	cudaMalloc((void**)&coords_rowPtr,TIMESTEPS * sizeof(int));
	cudaMalloc((void**)&coords_colPtr,TIMESTEPS * sizeof(int));
	cudaMalloc((void**)&resultantMatrixPtr,LONG_SIZE * LAT_SIZE * sizeof(float));
	cudaMalloc((void**)&resultantAnglePtr,LONG_SIZE * LAT_SIZE * sizeof(float));
	cudaMemcpy(resultantMatrixPtr,resultantMatrix,LONG_SIZE * LAT_SIZE * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(resultantAnglePtr,resultantAngle,LONG_SIZE * LAT_SIZE * sizeof(float),cudaMemcpyHostToDevice);
	dim3 gridSize2(1,0,0);
	dim3 blockSize2(1,0,0);
	bird_thread<<<gridSize2,blockSize2>>>(resultantMatrix,resultantAngle,coords_row,coords_col);
	cudaMemcpy(coords_row,coords_rowPtr,TIMESTEPS * sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(coords_row,coords_colPtr,TIMESTEPS * sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(coords_rowPtr);
	cudaFree(coords_colPtr);
	cudaFree(resultantMatrixPtr);
	cudaFree(resultantAnglePtr);
	//printf("%d,%d\n",coords_row[0],coords_col[0]);
	*/


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
	//All wind data in m/s
	float angle,diffAngle,magnitude,magnitude_squared;

	//vectAngle = angle between the wind vector and the vector orthogonal to the direction vector; or the crosswind vector
	float tailComponent,vectAngle,crossComponent,profit_value;
	tailComponent = 0;

	magnitude_squared = u_val * u_val + v_val * v_val;
	magnitude = (float)sqrt(magnitude_squared);


	//Getting the tail component of the wind; or the component of the wind in the desired direction of flight
	//From formula of getting the vector projection of wind onto the desired direction
	tailComponent = (dir_v * v_val + dir_u * u_val);
	tailComponent = tailComponent/sqrt(u_val*u_val + v_val*v_val);
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
		profit_value = DESIRED_SPEED - (float)sqrt(profit_value);
	}
	else {
		vectAngle = (dir_v * v_val + dir_u * u_val);
		vectAngle = acos(vectAngle / sqrt((u_val*u_val + v_val* v_val)*(dir_v * dir_v + dir_u * dir_u))) * (180/PI);
		vectAngle = (vectAngle <= 90)? 90 - vectAngle: vectAngle - 90;
		crossComponent = sqrt(u_val*u_val + v_val*v_val)/cos(vectAngle);
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
//Main part of the function
void get_movementData(FILE* outTxt,float* udata,float* vdata,float* dirData, float* precipData,float* pressureData,float* u10data,float* v10data)
{
	float pos_row,pos_col;
	//pos_row = LONG_SIZE - STARTING_ROW;
	pos_row = STARTING_ROW;
	pos_col = STARTING_COL;
	fprintf(outTxt,"%f,%f\n",pos_row,pos_col);
	int k;
	long i,l,l_old;

	float pressure_sum,pressure_MultSum,last_pressure,slope;
	pressure_MultSum = 0;
	pressure_sum = 0;
	slope = 1;
	last_pressure = 1011;
	l = 0;
	l_old = 0;
	float profit_value,dirAngle,actualAngle;

	
	float dir_v,dir_u;
	//long skip_size = (SKIP_TIMESTEPS * LONG_SIZE * LAT_SIZE) - 1;
	long skip_size = 0;
	

	float u_val,v_val,precip_val,v_ten,u_ten;
	//skip_size = 120174
	
	//fprintf(outTxt,"%f,%f\n",pos_row,pos_col);

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
		//If current pressure is greater than pressure in the previous day
		//if(pressureData[i] - old_pressure > 0) {
		printf("Main loop[k]\n");

		u_ten = bilinear_interpolation(pos_col,pos_row,u10data,l,1);
		v_ten = bilinear_interpolation(pos_col,pos_row,v10data,l,1);
		dirAngle = dirData[(int)(rintf(pos_row) * LAT_SIZE + rintf(pos_col))];
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



		//Relation between last_pressure and slope is an OR
		if((getProfitValue(u_ten,v_ten,actualAngle,dir_u,dir_v) >= MIN_PROFIT) && ((last_pressure>=1009)||(slope >-1))){
			

			for(k=0;k<6;k++,l++ ) {
				i = skip_size + l * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col;
			
				//dirAngle is with respect to North or the positive y-axis
				dirAngle = dirData[(int)(rintf(pos_row) * LAT_SIZE + rintf(pos_col))];
				actualAngle = dirAngle;

				//The grid is upside down; y increases from top to bottom while x increases from left to right 
				//dir_v and dir_u are the x and y components of the wind (v=y,u=x)
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
					pos_row = pos_row - (v_val + dir_v)*0.36;
					pos_col = pos_col + (u_val + dir_u)*0.36;

					float tmp;
					tmp = sqrt((v_val+dir_v)*(v_val+dir_v) + (u_val+dir_u)*(u_val+dir_u));
					//printf("\nTailComponent::%f,Speed::%f,dir_v::%f,dir_u::%f\n",tailComponent,tmp,dir_v,dir_u);
				
					printf("%ld \t %ld \t %d \t %f \t %f \t %f \t %f \t (%f,%f)\n",i,l,k,precip_val,profit_value,pos_row,pos_col,v_val,u_val);
				}
				else {
					//l increases but it goes back to the original starting hour for the bird; i.e 7pm
					// 6-k because l++ will not be done in the end because it breaks from the loop
					l += (6-k);
					
					//Every day has 6 time steps and this line skips the total desired number of days
					l += STOPOVER_DAYS * 6;
					printf("Skipped Wind (%f,%f) @ (%f,%f)w_profit = %f,precip=%f,And l = %ld\n",u_val,v_val,pos_row,pos_col,profit_value,precip_val,l);
					break;
				}
				fprintf(outTxt,"%f,%f,%f,%f\n",pos_row,pos_col,u_val,v_val);
			}
		}
		
		//v10 and u10 profit values were too low
		if(l_old == l){
			printf("u10 v10 profit value too low!\n");
			l+=6;

		}
	
		l_old = l - REGRESSION_HRS;

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

	}
}





