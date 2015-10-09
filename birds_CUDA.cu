


//This file uses 6 hourly data. Each day is 6 hours long and skipping a day means to add 6
//to the counter that counts the timesteps (l).

//The birds start at 00:00 UTC which is 6pm in central time when there is no day light savings

#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

//#include <curand_mtgp32_host.h> 
//#include <curand_mtgp32dc_p_11213.h>

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
#define LINESIZE		15*LONG_SIZE+LONG_SIZE - 3
#define TOTAL_DAYS		122
#define TIMESTEPS_PER_DAY	24
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

#define STD_BIRDANGLE		10.0	//Standard deviation * 6 = the total difference from max to min angle possible
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
//------------------------------Notes---------------------------------------------------------------------------------------
/*
Altitude = 850 millibars
Year = 2009
22 Jan 2015 No upper limit to the bird flight speed currently; Birds can fly well above 10m/s
Precipitation = millimeters
*/

//--------------------------------------------------------------------------------------------------------------------------

__global__ void WrappedNormal (float* MeanAngle,float AngStdDev,float* );
__global__ void setup_kernel(unsigned int seed,curandState *states);
__global__ void generate_kernel(curandState *states,float* numbers,float* angles);

//-------------------------------------------------------------------------------------------------------------------------------------
__global__ void setup_kernel(unsigned int seed,curandState *states)
{

	//Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int id = y * LONG_SIZE + x;

	curand_init(seed,id,0,&states[id]);
}
//-------------------------------------------------------------------------------------------------------------------------------------
__global__ void generate_kernel(curandState *states,float* numbers,float* angles)
{

	//Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int id = y * LONG_SIZE + x;
	
	float value;

	numbers[id] = curand_normal(&states[id]);

	if(id > (LONG_SIZE*LAT_SIZE -1)) return;
	else{
		
	
//		value = STD_BIRDANGLE * z + MeanAngle[id];
		value = STD_BIRDANGLE * numbers[id] + angles[id];

		if ((value - 360) > (-glCompAcc)){ 
		    value = value - 360;
		}
	 
		if (value < 0){
		    value= 360 + value;
		}
		numbers[id] = value;
		//printf("(x,y) = %d,%d,Value = %f \n",x,y,value);
	}
}


//-------------------------------------------------------------------------------------------------------------------------------------
/*
__global__ void WrappedNormal (float* MeanAngle,float AngStdDev){


	//Fisher 1993 pg. 47-48
	//s = -2.ln(r)

	float value;

	//Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int id = y * LONG_SIZE + x;
	
	if(id > (LONG_SIZE*LAT_SIZE -1)) return;
	else{
		
	
		value = AngStdDev * z + MeanAngle[id];

		if ((value - 360) > (-glCompAcc)){ 
		    value = value - 360;
		}
	 
		if (value < 0){
		    value= 360 + value;
		}
		MeanAngle[id] = value;
		//printf("(x,y) = %d,%d,Value = %f \n",x,y,value);
	}
}
*/
//-------------------------------------------------------------------------------------------------------------------------------------
int main()
{

//--------------------------Opening Direction file (Example: ext_crop.txt or extP_crop.txt)-------------//

	FILE* dirTxt;
//--------------------------Memory Allocation-----------------------------------//
	float* dirData;
	dirData = (float*)malloc(LAT_SIZE * LONG_SIZE * sizeof(float));	

	//float* result;
	//result = (float*)malloc(LAT_SIZE * LONG_SIZE * sizeof(float));

//--------------------------Opening Direction file (Example: ext_crop.txt or extP_crop.txt)-------------//
	dirTxt = fopen("./Lw_and_Dir/ext_Final.txt","r");
	//dirTxt = fopen("ext_crop.txt","r");
	if(dirTxt == NULL) {
		perror("Cannot open file with direction data\n");
		return -1;
	}
//--------------------------Direction file code end-------------------------------------------------------------------//
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


//-----------------------------------Reading Direction Values---------------------------------//
	memset(line,'\0',sizeof(line));
	memset(tempVal,'\0',sizeof(tempVal));
	i=0;
	j=0;

	while(fgets(line,LINESIZE,dirTxt)!=NULL){
		startPtr = line;
		for(i=0;i<LONG_SIZE;i++){
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));

			if(i != (LONG_SIZE - 1)) {
				endPtr = strchr(startPtr,',');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				//printf("%s ",tempVal);
				if(strcmp("NaN",tempVal)==0) {
					Value = 0.0;
				}
				else{
					Value = atof(tempVal);
				}

				dirData[j * LAT_SIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
				//printf("%d,%f ",i,Value);
			}
			else if(i == (LONG_SIZE - 1)){
				strcpy(tempVal,startPtr);
				//printf("%s \n",tempVal);

				if(strcmp("NaN\n",tempVal)==0) {
					Value = 0.0;
				}
				else{
					Value = atof(tempVal);
				}
				dirData[j * LAT_SIZE + i] = Value;
				//printf("%d,%f \n",i,Value);
			}
		}
		j++;
	}
//---------------------------------------------------------------------------------------------------------
	curandState_t* states;
	

	cudaMalloc((void**)&states,LAT_SIZE*LONG_SIZE*sizeof(curandState_t));

	dim3 gridSize(1,LAT_SIZE,1);
	dim3 blockSize(512,1,1);

	setup_kernel<<<gridSize,blockSize>>>(time(0),states);


	float cpu_nums[LAT_SIZE * LONG_SIZE];
	float *rand_norm_nums,*d_dirData;

	cudaMalloc((void**)&rand_norm_nums,LAT_SIZE*LONG_SIZE*sizeof(float));
	cudaMalloc((void**)&d_dirData,LAT_SIZE*LONG_SIZE*sizeof(float));

	cudaMemcpy(d_dirData,dirData, LAT_SIZE * LONG_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	generate_kernel<<<gridSize,blockSize>>>(states,rand_norm_nums,d_dirData);
	

	cudaMemcpy(cpu_nums,rand_norm_nums, LAT_SIZE * LONG_SIZE * sizeof(float), cudaMemcpyDeviceToHost);


	/* print them out */
	for (int j = 0; j < LAT_SIZE; j++) {
		for(int i = 0;i<LONG_SIZE;i++){
			printf("%f ", cpu_nums[j*LONG_SIZE + i]);
		}
		printf("\n");
	}

	/* free the memory we allocated for the states and numbers */
	cudaFree(states);
	cudaFree(rand_norm_nums);
	cudaFree(d_dirData);

//-----------------------------------------------Freeing allocated memory----------------------------//

	free(dirData);	
	fclose(dirTxt);

	printf("End\n");
	return 0;
}

//-------------------------------------------------------------------------------------------------------------------------------------
/*
__global__ void setup_kernel(curandState *state)
{
	//Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int id = y * LONG_SIZE + x;

	curand_init(1234,id,0,&state[id]);
}
//-------------------------------------------------------------------------------------------------------------------------------------

__global__ void generate_kernel(curandState *state,int n,unsigned int *result)
{
	//Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int id = y * LONG_SIZE + x;

	curandState localState = state[id];

	for(int i = 0; i < n; i++) {
		x = curand(&localState); // Check if low bit set //
		if(x & 1) {
			count++; 
		}
	} 
	// Copy state back to global memory 
	state[id] = localState; // Store results 
	result[id] += count;	
}
*/


//-------------------------------------------------------------------------------------------------------------------------------//
/*
int main(int argc,char* argv[])
{
	//Setting the output buffer to 500MB
	size_t limit;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 500 * 1024 * 1024);	
	cudaDeviceGetLimit(&limit,cudaLimitPrintfFifoSize);
//--------------------------Opening Direction file (Example: ext_crop.txt or extP_crop.txt)-------------//

	FILE* dirTxt;
//--------------------------Memory Allocation-----------------------------------//
	float* dirData;
	dirData = (float*)malloc(LAT_SIZE * LONG_SIZE * sizeof(float));	

	float* result;
	result = (float*)malloc(LAT_SIZE * LONG_SIZE * sizeof(float));

//--------------------------Opening Direction file (Example: ext_crop.txt or extP_crop.txt)-------------//
	dirTxt = fopen("./Lw_and_Dir/ext_Final.txt","r");
	//dirTxt = fopen("ext_crop.txt","r");
	if(dirTxt == NULL) {
		perror("Cannot open file with direction data\n");
		return -1;
	}
//--------------------------Direction file code end-------------------------------------------------------------------//
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


//-----------------------------------Reading Direction Values---------------------------------//
	memset(line,'\0',sizeof(line));
	memset(tempVal,'\0',sizeof(tempVal));
	i=0;
	j=0;

	while(fgets(line,LINESIZE,dirTxt)!=NULL){
		startPtr = line;
		for(i=0;i<LONG_SIZE;i++){
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));

			if(i != (LONG_SIZE - 1)) {
				endPtr = strchr(startPtr,',');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				//printf("%s ",tempVal);
				if(strcmp("NaN",tempVal)==0) {
					Value = 0.0;
				}
				else{
					Value = atof(tempVal);
				}

				dirData[j * LAT_SIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
				//printf("%d,%f ",i,Value);
			}
			else if(i == (LONG_SIZE - 1)){
				strcpy(tempVal,startPtr);
				//printf("%s \n",tempVal);

				if(strcmp("NaN\n",tempVal)==0) {
					Value = 0.0;
				}
				else{
					Value = atof(tempVal);
				}
				dirData[j * LAT_SIZE + i] = Value;
				//printf("%d,%f \n",i,Value);
			}
		}
		j++;
	}
*/
/*
	for(j=0;j<LAT_SIZE;j++){
		for(i=0;i<LONG_SIZE;i++){
			printf("%f ",dirData[j * 429 + i]);
			if(i == (LONG_SIZE - 1)) printf("\n");
		}
	}
*/
/*
//-----------------------------------Execute bird movement function-------------------------------//
	cudaError_t error;
	float* d_dirAngles,*d_result;
	curandState *devState;
	
	cudaMalloc((void**)&devState,sizeof(curandState));
	cudaMalloc((void**)&d_dirAngles,LONG_SIZE * LAT_SIZE * sizeof(float));
	cudaMalloc((void**)&d_result,LONG_SIZE * LAT_SIZE * sizeof(float));

	cudaMemcpy(d_dirAngles,dirData,LONG_SIZE * LAT_SIZE * sizeof(float),cudaMemcpyHostToDevice);

	dim3 gridSize(1,429,1);
	dim3 blockSize(512,1,1);



	WrappedNormal<<<gridSize,blockSize>>>(dirData,STD_BIRDANGLE,devState,result);
	//error = cudaDeviceSynchronize();
	//if(error != cudaSuccess)
  	//{
	//	printf("CUDA Device Synchronization Error: %s\n", cudaGetErrorString(error));

    	// we can't recover from the error -- exit the program
    	//return 0;
  	//}

	cudaMemcpy(result,d_result,LONG_SIZE * LAT_SIZE * sizeof(int),cudaMemcpyDeviceToHost);

	cudaFree(d_dirAngles);	
	cudaFree(devState);
	cudaFree(d_result);

	for(j=0;j<LAT_SIZE;j++){
		for(i=0;i<LONG_SIZE;i++){
			printf("%f ",result[j * 429 + i]);
			if(i == (LONG_SIZE - 1)) printf("\n");
		}
	}
//-----------------------------------------------Freeing allocated memory----------------------------//

	free(dirData);
	free(result);
	
	
	fclose(dirTxt);

	printf("End\n");
	return 0;
}
*/
//------------------------------------------------------------------------------------------------------------------------------------







