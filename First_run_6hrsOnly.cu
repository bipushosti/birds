

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
#define TIMESTEPS	10*6
#define SKIP_TIMESTEPS	0
//#define DESIRED_ROW
//#define DESIRED_COL
#define STARTING_ROW	200
#define STARTING_COL	100
#define DESIREDANGLE	90	//In degrees
#define DESIRED_SPEED	3.6	//In km/hr; Since the grid size is 10x10km,and birds want to
				//travel at 10m/s, it is 36km/hr.And in this grid 3.6 units per hour. 
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
void get_movementData(FILE* outTxt,float* udata,float* vdata,float* dirData,float* precipData,float* pressureData);
float getProfitValue(float* udata,float* vdata,float* dirData,long index,long pos);
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
	udataTxt = fopen("../Birds_data/full_U850_text_first10_2009.txt","r");
	vdataTxt = fopen("../Birds_data/full_V850_text_first10_2009.txt","r");
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
	u10dataTxt = fopen("../Birds_data/full_U10_text_first10_2009.txt","r");
	v10dataTxt = fopen("../Birds_data/full_V10_text_first10_2009.txt","r");
	if(u10dataTxt == NULL) {
		perror("Cannot open file with U10 data\n");
		return -1;
	}
	if(v10dataTxt == NULL) {
		perror("Cannot open file with V10 data\n");
		return -1;
	}

	FILE* dirTxt;
	dirTxt = fopen("South_direction.txt","r");
	if(dirTxt == NULL) {
		perror("Cannot open file with direction data\n");
		return -1;
	}
	FILE* precipTxt;
	precipTxt = fopen("../Birds_data/full_PRCP_text_first10_2009.txt","r");
	if(precipTxt == NULL) {
		perror("Cannot open file with PRCP data\n");
		return -1;
	}
	FILE* pressureTxt;
	pressureTxt = fopen("../Birds_data/full_MSLP_text_first10_2009.txt","r");
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
				Value = atof(tempVal);
				precipData[j * LAT_SIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (LAT_SIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
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
				fprintf(inpCheckU,"%f\n",dirData[j * LAT_SIZE + i]);
				//fprintf(inpCheckV,"%f\n",=data[j * LAT_SIZE + i]);
				//printf("%f\n",dirData[j * LAT_SIZE + i]);
			}
			else {
			fprintf(inpCheckU,"%f ",dirData[j * LAT_SIZE + i]);
			//fprintf(inpCheckV,"%f ",vdata[j * LAT_SIZE + i]);
			}
		}
	}
	get_movementData(posdataTxt,udata,vdata,dirData,precipData,pressureData);


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

float getProfitValue(float* udata,float* vdata,float* dirData,long index,long pos)
{
	float u,v;
	//All wind data in m/s
	v = vdata[index];
	u = udata[index];
	float angle,diffAngle,magnitude,magnitude_squared,wind_profit;
	magnitude_squared = u * u + v * v;
	magnitude = (float)sqrt(magnitude_squared);

	//DiffAngle is the angle between the desired direction of the bird 
	//and the direction of the wind
	//DiffAngle has to be calculated such that both the vectors are pointing
	//away from where they meet.
	
	if ((v == 0)&&( u > 0)){
		angle = 90;
		diffAngle = dirData[pos] - angle;
	}
	else if ((v == 0)&&( u < 0)){
		angle = 270;
		diffAngle = angle - dirData[pos];
	}
	else if ((v > 0)&&( u == 0)){
		angle = 0;
		diffAngle = dirData[pos] + angle;
	}
	else if ((v < 0)&&( u == 0)){
		angle = 180;
		diffAngle = angle - dirData[pos];
	}
	else {
		angle = tanf(u/v) * (180/PI);
		diffAngle = dirData[pos] - angle;
	}
	if(diffAngle > 180) diffAngle = 360 - diffAngle;
	if(diffAngle < 0) diffAngle = diffAngle * -1;
	if(diffAngle > 360) diffAngle = diffAngle - 360;
	wind_profit = (DESIRED_SPEED * DESIRED_SPEED) + magnitude_squared - 2 * DESIRED_SPEED * magnitude * cos(diffAngle * PI/180);
	wind_profit = DESIRED_SPEED - (float)sqrt(wind_profit);
	return wind_profit;
}

//------------------------------------------------------------------------------------------------------------------------------------




//Main part of the function
void get_movementData(FILE* outTxt,float* udata,float* vdata,float* dirData, float* precipData,float* pressureData)
{
	int pos_row,pos_col;
	//pos_row = LONG_SIZE - STARTING_ROW;
	pos_row = STARTING_ROW;
	pos_col = STARTING_COL;
	fprintf(outTxt,"%d,%d\n",pos_row,pos_col);
	int k;
	long i,l,l_old;
	//long i_old;
	float pressure_sum,pressure_MultSum,last_pressure,slope;
	pressure_MultSum = 0;
	pressure_sum = 0;
	slope = 0;
	l = 0;
	l_old = 0;
	//i_old = 0;
	float profit_value,dirAngle,tailComponent,crossComponent;
	//vectAngle = angle between the wind vector and the vector orthogonal to the direction vector; or the crosswind vector
	float dir_v,dir_u,vectAngle;
	long skip_size = (SKIP_TIMESTEPS * LONG_SIZE * LAT_SIZE) - 1;
	//skip_size = 120174

	printf("i \t l \t k \t precipData \t profit_value \t pos_row \t pos_col \t (vdata[i],udata[i])\n");
	i = skip_size +pos_row * LAT_SIZE + pos_col;
	while( i <= (TIMESTEPS-1) * LAT_SIZE * LONG_SIZE ) {
		dir_v = 0;
		dir_u = 0;
		dirAngle = 0;
		//l_old = 1;
		slope = 0;
		pressure_sum = 0;
		pressure_MultSum = 0;
		printf("At timestep Check\n");
		//If current pressure is greater than pressure in the previous day
		//if(pressureData[i] - old_pressure > 0) {
		printf("Main loop\n");

	
		
		for(k=0;k<6;k++,l++ ) {
			i = skip_size + l * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col;
			
			//dirAngle is with respect to North or the positive y-axis
			dirAngle = dirData[pos_row * LAT_SIZE + pos_col];

			//The grid is not upside down anymore; 
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


			//Getting the tail component of the wind; or the component of the wind in the desired direction of flight
			//From formula of getting the vector projection of wind onto the desired direction
			tailComponent = (dir_v * vdata[i] + dir_u * udata[i]);
			tailComponent = tailComponent/sqrt(udata[i]*udata[i] + vdata[i]*vdata[i]);
			//Separate profit value methods have to be used if the tail component is less that equal to or greater than the desired speed of the birds


			if(tailComponent <= DESIRED_SPEED) {
				profit_value = getProfitValue(udata,vdata,dirData,i,pos_row * LAT_SIZE + pos_col);
			}
			else {
				vectAngle = (dir_v * vdata[i] + dir_u * udata[i]);
				vectAngle = acos(vectAngle / sqrt((udata[i]*udata[i] + vdata[i]* vdata[i])*(dir_v * dir_v + dir_u * dir_u))) * (180/PI);
				vectAngle = (vectAngle <= 90)? 90 - vectAngle: vectAngle - 90;
				crossComponent = sqrt(udata[i]*udata[i] + vdata[i]*vdata[i])/cos(vectAngle);
				//Getting the absolute value
				crossComponent = (crossComponent<0)? crossComponent * (-1):crossComponent;
				profit_value = tailComponent - crossComponent;
				printf("Over the Desired Speed\n");
			}
			//Adding precip value
			//if ((profit_value >= MIN_PROFIT) && (precipData[i] < 2) ) {
			if ((profit_value >= MIN_PROFIT) && (precipData[i] < 2) ) {
				

				//Positon is in a 10 km grid. This means for 1m/s, the 
				//position change in 1 hour is 3.6/10 = 0.36units in the grid
				pos_row = (int)(rintf(pos_row + (vdata[i] + dir_v)*0.36));
				pos_col = (int)(rintf(pos_col + (udata[i] + dir_u)*0.36));

				//Bilinear transformation











				//printf("%d,%d\n",pos_row,pos_col);
				float tmp;
				tmp = sqrt((vdata[i]+dir_v)*(vdata[i]+dir_v) + (udata[i]+dir_u)*(udata[i]+dir_u));
				printf("\nTailComponent::%f,Speed::%f,dir_v::%f,dir_u::%f\n",tailComponent,tmp,dir_v,dir_u);
				//printf("%ld\n",i);
				
				printf("%ld \t %ld \t %d \t %f \t %f \t %d \t %d \t (%f,%f)\n",i,l,k,precipData[i],profit_value,pos_row,pos_col,vdata[i],udata[i]);
				//fprintf(outTxt,"%d,%d\n",pos_row,pos_col);
			}
			else {
				//Goes back to the original starting hour for the bird; i.e 7pm
				// 5-k because l++ will be done in the end
				l += (5-k);
				//l = l - k;
				//l_old = l + 12;
				printf("Skipped Wind (%f,%f) @ (%d,%d)w_profit = %f\n",udata[i],vdata[i],pos_row,pos_col,profit_value);
				break;
			}
		}

		for(k=1; (l_old < l) && (k<=REGRESSION_HRS); l_old++,k++){

			pressure_sum += pressureData[skip_size + l_old * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col];
			pressure_MultSum += k * pressureData[skip_size + l_old * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col];
			//printf("%f\n",pressureData[skip_size + l_old * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col]);

			if(k == REGRESSION_HRS) {
				last_pressure = pressureData[skip_size + l_old * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col];
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
		l_old = l;
		printf("Running\n");
	}
}


/*
float bilinear_interpolation(float x,float y,float* data_array)
{
	float value,x1,y1,x2,y2,Q11,Q12,Q21,Q22;

	x1 = floorf(x);
	x2 = ceilf(x);
	y1 = floorf(y);
	y2 = ceilf(y);

	Q11 = data_array[y1 * LAT_SIZE + x1 ];
	Q12 = data_array[y2 * LAT_SIZE + x1 ];
	Q21 = data_array[y1 * LAT_SIZE + x2 ];
	Q22 = data_array[y2 * LAT_SIZE + x2 ];

	return 0.0;
}
*/
