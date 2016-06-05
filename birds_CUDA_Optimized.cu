
//Needs Header Files for the functions; The header file should have both C and CUDA functions



//This file uses 6 hourly data. Each day is 6 hours long and skipping a day means to add 6
//to the counter that counts the timesteps (l).

//The birds start at 00:00 UTC which is 6pm in central time examplewhen there is no day light savings
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <pthread.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <getopt.h>

#include <math.h>


//#include "birds_CUDA.h"
//#define CUDA_API_PER_THREAD_DEFAULT_STREAM


#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


#define PI 			3.14159
#define LONG_SIZE		429
#define LAT_SIZE		429
#define LINESIZE		15*LONG_SIZE+LONG_SIZE - 3
#define TOTAL_DAYS		122
#define TIMESTEPS_PER_DAY	24
#define TIMESTEPS		TOTAL_DAYS*TIMESTEPS_PER_DAY
#define SKIP_TIMESTEPS		0


//This is the number of timesteps that the bird will skip in the beginning to get to the desired 
//takeoff time. Since the data starts at 7 pm, the birds will skip the first 23 hours to get to 
//6pm.
#define INITIAL_SKIP_TIMESTEPS		23		


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
//To make the birds head back directly west the angle must be set to 180.
#define BIRD_SEA_ANGLE		180

//The maximum number of hours that the birds can fly continiously
#define BIRD_HRS_LIMIT		72

#define TOTAL_DATA_FILES	9
//Total number of data files or variables bird flight depends on;Does not include direction files and land water data
#define NUM_DATA_FILES		6

#define THREADS_PER_BLOCK	32
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
//------------------------------Notes---------------------------------------------------------------------------------------
/*
Altitude = 850 millibars
Year = 2009
22 Jan 2015 No upper limit to the bird flight speed currently; Birds can fly well above 10m/s
Precipitation = millimeters
*/

//--------------------------------------------------------------------------------------------------------------------------
__global__ void setup_kernel(unsigned int seed,curandState *states,int NumOfBirds);
__global__ void generate_kernel(curandState *states,float* numbers,int NumOfBirds);
__global__ void bird_movement(float* rowArray,float* colArray,int NumOfBirds,long int start_l,long int cur_l,long int max_timesteps,float* udata,float* vdata,float* u10data,
				float* v10data,float* d_dirData,float* rand_norm_nums,float* precipData,float* pressureData,float* lwData,uint8_t* birdStatus);

__device__ float bilinear_interpolation_SmallData(float x,float y,float* data_array);
__device__ float bilinear_interpolation_LargeData(float x,float y,float* data_array,long l);
__device__ float WrappedNormal(int id,float MeanAngle,float AngStdDev,float* rand_norm_nums,long int cur_timestep);
__device__ float getProfitValue(float u_val,float v_val,float dirVal,float dir_u,float dir_v);
__device__ long int bird_AtSea_Within24Hrs(int id,int arrLength,float* rowArray,float* colArray,long int start_l,
					long int l,float* udata,float* vdata,float* lwData,uint8_t* birdStatus,uint8_t var_product,uint8_t l_product);
__device__ long int bird_AtSea_After24Hrs(int id,int arrLength,float* rowArray,float* colArray,long int start_l,long int l,
float* udata,float* vdata,float* lwData,uint8_t* birdStatus,uint8_t var_product,uint8_t l_product);

static void* write_dataVars(void* arguments);
static void* read_dataFiles(void* arguments);
long int convert_to_month(int month,int day);
static void HandleError( cudaError_t err,const char *file, int line );
long Get_GPU_devices();
//-------------------------------------------------------------------------------------------------------------------------------------
struct file_IO {
	FILE *fp;
	float* inpVals;
	float* streamArray;
	size_t dataSize;
}inpStruct[8]; 
//-------------------------------------------------------------------------------------------------------------------------------------
//Global Variables

float* udata;
float* vdata;
float* u10data;
float* v10data;
float* precipData;
float* pressureData;

float* dir_u;
float* dir_v;
float* lwData;
float* dirData;
//-------------------------------------------------------------------------------------------------------------------------------------
__device__ __constant__ int TotalTimesteps = TIMESTEPS;
__device__ __constant__ int LatSize = LAT_SIZE;
__device__ __constant__ int LongSize = LONG_SIZE;
__device__ __constant__ float pi = PI;
__device__ __constant__ int InitialSkipTimesteps = INITIAL_SKIP_TIMESTEPS;


__device__ __constant__ int StdBirdAngle = STD_BIRDANGLE;
__device__ __constant__ int BirdSeaAngle = BIRD_SEA_ANGLE;
__device__ __constant__ int BirdHrsLimit = BIRD_HRS_LIMIT;
__device__ __constant__ int MinProfit = MIN_PROFIT;
__device__ __constant__ int MaxPrecip = MAX_PRECIP;
__device__ __constant__ int MaxLatSouth = MAX_LAT_SOUTH;
__device__ __constant__ int DesiredSpeed = DESIRED_SPEED;
__device__ __constant__ int StopoverDays = STOPOVER_DAYS;

__device__ __constant__ int DenomSlope = DENOM_SLOPE;
__device__ __constant__ int HrsSum = HRS_SUM;
__device__ __constant__ int RegressionHrs = REGRESSION_HRS;
__device__ __constant__ float GlCompAcc = glCompAcc;

__device__ int CurrentTimestep = 0;
//###########################################################################################################################################//

__device__ long int bird_AtSea_Within24Hrs(int id,int arrLength,float* rowArray,float* colArray,long int start_l,long int l,
float* udata,float* vdata,float* lwData,uint8_t* birdStatus,uint8_t var_product)
{
	float u_val,v_val,u_dir,v_dir,pos_row,pos_col;
	float index = 0;
	long int bckp_l;
	float count_timeSteps = 0;
	uint8_t var_product2;

	var_product2 = var_product;

	pos_row = rowArray[id * arrLength + l - 1];
	pos_col = colArray[id * arrLength + l - 1];
	
	u_dir = DesiredSpeed * cosf(BirdSeaAngle * (pi/180));
	v_dir = DesiredSpeed * sinf(BirdSeaAngle * (pi/180));

	
	for(count_timeSteps = 10;count_timeSteps<24;count_timeSteps++){
		
		var_product2 = var_product2 * birdStatus[id]; 
		
		/** Bilinear interpolation for u and v data **/
		u_val = bilinear_interpolation_LargeData(pos_col,pos_row,udata,l-start_l);	
		v_val = bilinear_interpolation_LargeData(pos_col,pos_row,vdata,l-start_l);

		/** Desired speed needs to change in the case of column position or the birds
		will not fly west **/
		pos_row = pos_row + var_product2 * (v_val + v_dir) * 0.36 * -1;	
		pos_col = pos_col + var_product2 * (u_val + u_dir) * 0.36;

		rowArray[id * arrLength + l] = pos_row;
		colArray[id * arrLength + l] = pos_col;

		printf("At sea within 24 hours; \tRow: %f,Col:%f\n",rowArray[id * arrLength + l],colArray[id * arrLength + l]);
		printf("At sea within 24 hours; Timestep #: %ld\n",l);

		index = lwData[__float2int_rd(pos_row * LongSize + pos_col)];
		printf("Index after 10 hours is %f\n",index);

		if(index == 1.0){
			var_product2 = 0;		
		}
	
		if((pos_row > LatSize)||(pos_row > MaxLatSouth) || (pos_col >LongSize)||(pos_row < 0)||(pos_col < 0 )){
			birdStatus[id] = 0;
		}

		l = l + 1;
	}

	if(index == 0){
		birdStatus[id] = 0;
	}
	return l;
	
}


//###########################################################################################################################################//

__device__ long int bird_AtSea_After24Hrs(int id,int arrLength,float* rowArray,float* colArray,long int start_l,long int l,
float* udata,float* vdata,float* lwData,uint8_t* birdStatus,uint8_t var_product,uint8_t l_product)
{
	float u_val,v_val,u_dir,v_dir,pos_row,pos_col;
	int count_timeSteps, timesteps_limit, index;
	uint8_t var_product2;

		
	index = 0;
	var_product2 = var_product;

	pos_row = rowArray[id * arrLength + l ];
	pos_col = colArray[id * arrLength + l ];
	

	u_dir = DesiredSpeed * cosf(BirdSeaAngle * (pi/180));
	v_dir = DesiredSpeed * sinf(BirdSeaAngle * (pi/180));
	
	if(l_product == 0){
		timesteps_limit = 24;
	}else{
		timesteps_limit = BirdHrsLimit;
	}

	//This loop is skipped if a bird is not at sea after 24 hours
	for(count_timeSteps = 24; count_timeSteps < timesteps_limit; count_timeSteps++){
			
		var_product2 = var_product2 * birdStatus[id]; 
		
		/** Bilinear interpolation for u and v data **/
		u_val = bilinear_interpolation_LargeData(pos_col,pos_row,udata,l-start_l);	
		v_val = bilinear_interpolation_LargeData(pos_col,pos_row,vdata,l-start_l);

		/** Desired speed needs to change in the case of column position or the birds
		will not fly west **/
		pos_row = pos_row + var_product2 * (v_val + v_dir) * 0.36 * -1;	
		pos_col = pos_col + var_product2 * (u_val + u_dir) * 0.36;

		printf("At sea after 24 hours; \tRow: %f,Col:%f\n",rowArray[id * arrLength + l],colArray[id * arrLength + l]);
		printf("At sea after 24 hours; Timestep #: %ld\n",l);

		rowArray[id * arrLength + l + 1] = pos_row;
		colArray[id * arrLength + l + 1] = pos_col;

		index += lwData[__float2int_rd(pos_row * LatSize + pos_col)];

		//Checking if the bird found land
		//Limit calculated only if bird found at land the first time
		if(index == 1){
			var_product2 = 0;
			timesteps_limit = __float2ull_ru(count_timeSteps/24) * 24 + 24 * StopoverDays; 					
		}else if (index == 0){
			var_product2 = 1;
			
		}else if (index > 1){
			var_product2 = 0;
		}

		l = l + var_product2;

		if((pos_row > LatSize)||(pos_row > MaxLatSouth) || (pos_col >LongSize)||(pos_row < 0.0)||(pos_col < 0.0 )){
			birdStatus[id] = 0;
		}
	}

	index = lwData[__float2int_rd(pos_row * LatSize + pos_col)];

	if (index != 1){
		birdStatus[id] = 0;
	}
	
	return l;
	
	
}

//###########################################################################################################################################//

__device__ float getProfitValue(float u_val,float v_val,float dirVal,float dir_u,float dir_v)
{

	/** All wind data in m/s **/
	float diffAngle,magnitude,magnitude_squared,tailComponent,crossComponent,profit_value;

	tailComponent = 0;
	
	magnitude = hypotf(u_val,v_val);
	magnitude_squared = magnitude * magnitude;

	/** Getting the tail component of the wind; or the component of the wind in the desired direction of flight
	From formula of getting the vector projection of wind onto the desired direction **/

	tailComponent = (dir_v * v_val + dir_u * u_val);
	tailComponent = tailComponent/hypotf(dir_u,dir_u);
	

	/** DiffAngle is the angle between the desired direction of the bird and the direction of the wind
	DiffAngle has to be calculated such that both the vectors are pointing away from where they meet.
	Using the formula to get angle between two vectors **/

	diffAngle = acosf( (u_val*dir_u + v_val * dir_v)/ (( hypotf(u_val,v_val) * hypotf(dir_u,dir_v) )) ) * 180/pi;

	/** Separate profit value methods have to be used if the tail component is less that equal to or greater than the desired speed of the birds **/
	if(tailComponent <= DesiredSpeed) {	
		profit_value = (DesiredSpeed * DesiredSpeed) + magnitude_squared - 2 * DesiredSpeed * magnitude * cosf(diffAngle * pi/180);
		profit_value = DesiredSpeed - sqrtf(profit_value);
	}
	else {
		/** Perpendicular to a vector (x,y) is (y,-x) or (-y,x) Cross component is always positive **/

		crossComponent = fabsf((-dir_v*u_val + dir_u*v_val)/hypotf(dir_v,dir_u));
		profit_value = tailComponent - crossComponent;
	}

	return profit_value;
}


//###########################################################################################################################################//

__device__ float bilinear_interpolation_SmallData(float x,float y,float* data_array)
{
	float x1,y1,x2,y2;
	float Q11,Q12,Q21,Q22,R1,R2,R;

	x1 = floorf(x);
	x2 = ceilf(x);
	y1 = floorf(y);
	y2 = ceilf(y);
	R = 0;
	
	Q11 = data_array[(int)(y1 * LongSize + x1)];
	Q12 = data_array[(int)(y2 * LongSize + x1)];
	Q21 = data_array[(int)(y1 * LongSize + x2)];
	Q22 = data_array[(int)(y2 * LongSize + x2)];
	

	R1 = Q11 + (x - x1)*(Q21 - Q11);
	R2 = Q12 + (x - x1)*(Q22 - Q12);
	R = R1 + (y - y1)*(R2 - R1);
	
	//printf("Q11:%f,Q12:%f,Q21:%f,Q22:%f; And Value=%f\n",Q11,Q12,Q21,Q22,value);

	return R;
}


//###########################################################################################################################################//

__device__ float bilinear_interpolation_LargeData(float x,float y,float* data_array,long l)
{
	float x1,y1,x2,y2;
	float Q11,Q12,Q21,Q22,R1,R2,R;

	x1 = floorf(x);
	x2 = ceilf(x);
	y1 = floorf(y);
	y2 = ceilf(y);
	R = 0;
	
	Q11 = data_array[(int)(l  * LatSize * LongSize + y1 * LongSize + x1) ];
	Q12 = data_array[(int)(l  * LatSize * LongSize + y2 * LongSize + x1) ];
	Q21 = data_array[(int)(l  * LatSize * LongSize + y1 * LongSize + x2) ];
	Q22 = data_array[(int)(l  * LatSize * LongSize + y2 * LongSize + x2) ];
	

	R1 = Q11 + (x - x1)*(Q21 - Q11);
	R2 = Q12 + (x - x1)*(Q22 - Q12);
	R = R1 + (y - y1)*(R2 - R1);


	return R;
}
//###########################################################################################################################################//

__global__ void setup_kernel(unsigned int seed,curandState *states,int NumOfBirds)
{

	//Thread indices

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int id = y * TotalTimesteps + x;


	if((x >= TotalTimesteps) || (x < 0)){
		return;
	}else if((y>= NumOfBirds) || (y < 0)){
		return;
	}else if(id >= TotalTimesteps * NumOfBirds){
		return;
	}else{
		curand_init(seed,id,0,&states[id]);
	}
}

//###########################################################################################################################################//

__global__ void generate_kernel(curandState *states,float* numbers,int NumOfBirds)
{

	//Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int id = y * TotalTimesteps + x;

	if((x >= TotalTimesteps) || (x < 0)){
		return;
	}else if((y>= NumOfBirds) || (y < 0)){
		return;
	}else if(id >= TotalTimesteps * NumOfBirds){
		return;
	}else{
		//Making a local copy for efficiency
		curandState localState = states[id];	
		numbers[id] = curand_normal(&localState);
	}
	
	return;
}

//###########################################################################################################################################//
//###########################################################################################################################################//
//###########################################################################################################################################//

__global__ void bird_movement(float* rowArray,float* colArray,int NumOfBirds,long int start_l,long int cur_l,long int max_timesteps,float* udata,
	float* vdata,float* u10data,float* v10data,float* dirData,float* rand_norm_nums,float* precipData,float* pressureData,float* lwData,
	uint8_t* birdStatus)
{

	//Thread indices
	int id = blockIdx.x * blockDim.x + threadIdx.x; 

	//if(id > (NumOfBirds -1)||(birdStatus[id]==0)||(cur_l > max_timesteps)){ 

	//The condition cur_l > max_timesteps is OK now because all birds start at the same time
	//Not OK once birds start flying at different times
	if((id > (NumOfBirds -1))||(cur_l > max_timesteps)){ 
		return;
	}
	
	else{
		//Making a local copy of the timstep variable
		long int l,new_l;

		long l_old;	
		float profit_value,actualAngle,wrappedAngle, index;
		float last_pressure,pressure_sum,pressure_MultSum,slope;
		float u_ten,v_ten,u_val,v_val,uDir_value,vDir_value,precip_val;
		int k,i;
		float pos_row,pos_col;
		int arrLength;


		uint8_t var_sea, var_profit_10m, var_10hrsSea, var_product, l_product;

		l_product = 1;
		l = cur_l;
		new_l = l;
		arrLength = (TotalTimesteps + 1);
	

		slope = 0;
		printf("Value of l is %ld\n",l);

		printf("Array length per bird is %d\n",arrLength);
		printf("id is %d\n",id);

		printf("id * arrayLength is:%d\n",id*arrLength);

		
		while(l < max_timesteps){

			printf("Starting pos_row is %f , pos_col is: %f\n",*(rowArray + id * arrLength + l -1),*(colArray + id * arrLength + l -1));
			printf("Before any computation; Timestep #: %ld\n",l);
			pos_row = rowArray[id * arrLength + l - 1];
			pos_col = colArray[id * arrLength + l - 1];
		

			if((pos_row > LatSize) || (pos_col >LongSize)||(pos_row < 0)||(pos_col < 0 )){
				birdStatus[id] = 0;
			}

			//Getting the wrapped angle
			actualAngle = dirData[__float2int_rd(pos_row * LatSize + pos_col)];
			wrappedAngle = rand_norm_nums[id*TotalTimesteps + l] * STD_BIRDANGLE + actualAngle;
			if(wrappedAngle > 360){
				wrappedAngle = wrappedAngle - 360;
				
			}else if(wrappedAngle < 0 ){
				wrappedAngle = 360 + wrappedAngle;
			}	
			uDir_value = DesiredSpeed * cosf(wrappedAngle * (pi/180));
			vDir_value = DesiredSpeed * sinf(wrappedAngle * (pi/180));

			u_ten = bilinear_interpolation_LargeData(pos_col,pos_row,u10data,l-start_l);
			v_ten = bilinear_interpolation_LargeData(pos_col,pos_row,v10data,l-start_l);

			profit_value = getProfitValue(u_ten,v_ten,wrappedAngle,uDir_value,vDir_value);
		
			if((profit_value >= MinProfit) && ((last_pressure>=1009)||(slope >-1))){
				var_profit_10m = 1;
			}else{
				var_profit_10m = 0;
				printf("Profit value at 10m is low \n");
			}

/*		
			if(new_l > l){  
				l_product = 0;				
				printf("Reached timestep more than others after being at sea \n"); 
			}else{
				l_product = 1;
			}
*/
			for(k=0;k<6 && l<max_timesteps;k++) {
			
				//Getting the wrapped angle
				actualAngle = dirData[__float2int_rd(pos_row * LatSize + pos_col)];
				wrappedAngle = rand_norm_nums[id*TotalTimesteps + l] * STD_BIRDANGLE + actualAngle;

				if(wrappedAngle > 360){
					wrappedAngle = wrappedAngle - 360;
				
				}else if(wrappedAngle < 0 ){
					wrappedAngle = 360 + wrappedAngle;
				}	

				uDir_value = DesiredSpeed * cosf(wrappedAngle * (pi/180));
				vDir_value = DesiredSpeed * sinf(wrappedAngle * (pi/180));

				u_val = bilinear_interpolation_LargeData(pos_col,pos_row,udata,l-start_l); 
				v_val = bilinear_interpolation_LargeData(pos_col,pos_row,vdata,l-start_l);
				precip_val = bilinear_interpolation_LargeData(pos_col,pos_row,precipData,l-start_l);
	
				//Getting the previous position values for row and column
				pos_row = rowArray[id * arrLength + l - 1];
				pos_col = colArray[id * arrLength + l - 1];
				
				if((pos_row > LatSize)||(pos_row > MaxLatSouth) || (pos_col >LongSize)||(pos_row < 0)||(pos_col < 0 )){
					birdStatus[id] = 0;
					printf("Dead bird \n");
				}
				
				var_product = birdStatus[id] * var_profit_10m * l_product;
				//Storing the new values
				rowArray[id * arrLength + l] = pos_row + var_product * (v_val + vDir_value ) * 0.36 * -1;
				colArray[id * arrLength + l] = pos_col + var_product * (u_val + uDir_value) * 0.36;
						
				printf("6 Hour Flight\tRow: %f,Col:%f\n",rowArray[id * arrLength + l],colArray[id * arrLength + l]);
				printf("6 hour flight;Timestep #: %ld\n",l);
				
				l = l + 1;

			}	
			
			//The value of l increases at the last iteration 

			printf("After 6 hour flight over\n");
			pos_row = rowArray[id * arrLength + l - 1];
			pos_col = colArray[id * arrLength + l - 1];
			

			index = lwData[__float2int_rd(pos_row * LatSize + pos_col)];
			// If the bird is at sea after the first 6 hours of flight 
			if( index == 1.0){
				var_sea = 0;
				printf("Not at sea after 6 hours \n");
			}else{
				var_sea = 1;
				printf("At sea after 6 hours \n");
			}

			//Getting the wrapped angle; Same uDir_value and vDir_value used for the 4 hours
			actualAngle = dirData[__float2int_rd(pos_row * LatSize + pos_col)];
			wrappedAngle = rand_norm_nums[id*TotalTimesteps + l] * STD_BIRDANGLE + actualAngle;
			if(wrappedAngle > 360){
				wrappedAngle = wrappedAngle - 360;
				
			}else if(wrappedAngle < 0 ){
				wrappedAngle = 360 + wrappedAngle;
			}	
			uDir_value = DesiredSpeed * cosf(wrappedAngle * (pi/180));
			vDir_value = DesiredSpeed * sinf(wrappedAngle * (pi/180));


//-----------------------At sea after first 6 hours of flight
			for(k=6;k<10 && l<max_timesteps;k++){
								
				u_val = bilinear_interpolation_LargeData(pos_col,pos_row,udata,l-start_l);
				v_val = bilinear_interpolation_LargeData(pos_col,pos_row,vdata,l-start_l);
			
				var_product = birdStatus[id] * var_profit_10m * var_sea * l_product;

				//Getting new position values for row and column and storing it 
				pos_row = pos_row + var_product * (v_val + vDir_value ) * 0.36 * -1;
				pos_col = pos_col + var_product * (u_val + uDir_value) * 0.36;


				printf("+4 Hour Flight\tRow: %f,Col:%f\n",pos_row,pos_col);
				printf("+4 hour flight;Timestep #: %ld\n",l);

				if((pos_row > LatSize)||(pos_row > MaxLatSouth) || (pos_col >LongSize)||(pos_row < 0)||(pos_col < 0 )){
					birdStatus[id] = 0;
				}

				rowArray[id * arrLength + l] = pos_row;
				colArray[id * arrLength + l] = pos_col;
		
				//printf("+4 Hour Flight\tRow: %f,Col:%f\n",rowArray[id * arrLength + l + 1],colArray[id * arrLength + l + 1]);
				
				l = l + 1;
			}

//------------------------

			index = lwData[__float2int_rd(pos_row * LongSize + pos_col)];
	

			if(index == 1){
				var_10hrsSea = 0;
				printf("Not at sea after 10 hours \n");
			}else{
				var_10hrsSea = 1;
				printf("At sea after 10 hours \n");
			}
			
//----------------------- If at sea even after the 10 hours but within 24 hours		
			var_product = birdStatus[id] * var_profit_10m * var_sea * var_10hrsSea;
			l = bird_AtSea_Within24Hrs(id,arrLength,rowArray,colArray,start_l,l,udata,vdata,lwData,birdStatus,var_product);
//------------------------						

			index = lwData[__float2int_rd(pos_row * LongSize + pos_col)];
			if(index == 1.0){
				var_10hrsSea = 0;
				printf("Not at sea after 24 hours \n");
			}else{
				var_10hrsSea = 1;
				printf("At sea after 24 hours \n");
			}
//----------------------- If at sea even after the the 10 hours and beyond 24 hours 	

	
			var_product = birdStatus[id] * var_profit_10m * var_sea * var_10hrsSea;
			if(var_product == 1){ 
				printf("At sea after 24 hours \n");
			}
			new_l = bird_AtSea_After24Hrs(id,arrLength,rowArray,colArray,start_l,l,udata,vdata,lwData,birdStatus,var_product,l_product);

//------------------------	

			l_old = l - RegressionHrs;
	
			pressure_sum = 0;
			pressure_MultSum = 0;
			//Taking the pressure from 6 hours earlier of the location where the bird landed
			for(k=1; (l_old < l) && (k<=RegressionHrs) && (l_old<max_timesteps); l_old++,k++){

				pressure_sum += bilinear_interpolation_LargeData(pos_col,pos_row,pressureData,l_old-start_l);  //<----------------ERROR HERE
				pressure_MultSum += k * bilinear_interpolation_LargeData(pos_col,pos_row,pressureData,l_old-start_l);

				//last_pressure is the last day or the day of flight
				if(k == RegressionHrs) {
					last_pressure = bilinear_interpolation_LargeData(pos_col,pos_row,pressureData,l_old-start_l);
				}
			}
			slope = ((RegressionHrs * pressure_MultSum) - (pressure_sum * HrsSum))/(DenomSlope);
		
		}		
	}
}
//###########################################################################################################################################//
//###########################################################################################################################################//
//###########################################################################################################################################//
long Get_GPU_devices()
{
	cudaDeviceProp prop;
	int whichDevice,DeviceCount;
	long deviceMemory;

	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,whichDevice));
	
	if(!prop.deviceOverlap){
			printf("Device does not handle overlaps so streams are not possible\n");
	return 0;
	}

	DeviceCount = 0;
	
	HANDLE_ERROR(cudaGetDeviceCount(&DeviceCount));
	if(DeviceCount > 0){ 
		printf("%d Devices Found\n",DeviceCount);
	}else{
		printf("No devices found or error in reading the number of devices\n");
		return 0;
	}
	
	int i = 0;

	cudaDeviceProp properties;
	HANDLE_ERROR(cudaGetDeviceProperties(&properties,i));
	printf("Device Number: %d\n", i);
	printf("  Device name: %s\n", properties.name);
	printf("  Device Global Memory size: %zd MB \n",properties.totalGlobalMem/1000000);
	printf("\n");
	
	deviceMemory = properties.totalGlobalMem;


	return deviceMemory;
}

//###########################################################################################################################################//
static void* read_dataFiles(void* arguments)
{

	struct file_IO *inputArgs;
	inputArgs = (struct file_IO *)arguments;

	FILE* textFile;
	float* dataArray;


	textFile = inputArgs->fp;
	dataArray = inputArgs->inpVals;

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

	memset(line,'\0',sizeof(line));
	memset(tempVal,'\0',sizeof(tempVal));
	i=0;
	j=0;

	while(fgets(line,LINESIZE,textFile)!=NULL){
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
				
				dataArray[j * LAT_SIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
				//printf("%d,%f ",i,Value);
			}
			else if(i == (LONG_SIZE - 1)){
				strcpy(tempVal,startPtr);

				if(strcmp("NaN\n",tempVal)==0) {
					Value = 0.0;
				}
				else{
					Value = atof(tempVal);
				}
				dataArray[j * LAT_SIZE + i] = Value;
			}
		}
		j++;
	}
	return NULL;
}

//###########################################################################################################################################//
static void* write_dataVars(void* arguments)
{

	struct file_IO *inputArgs;
	inputArgs = (struct file_IO *)arguments;

	float* dataArray,*destArray;
	size_t totalSize;	
	long int i;

	dataArray = inputArgs->inpVals;
	destArray = inputArgs->streamArray;
	totalSize = inputArgs->dataSize;

	for(i=0;i<totalSize;i++){
		destArray[i] = *(dataArray + i);
	}

	return NULL;
}



//###########################################################################################################################################//
long int convert_to_month(int month,int day)
{
	long int index,offset;
	if(month == 8){
		index = 1; //The data starts in august
	}
	else if(month == 9){
		index = 32; //The data for september starts after 31 days of august
	}
	else if(month == 10){
		index = 62; //The data for october starts after 31+30 days of sept and august respectively.
	}
	else if(month == 11){
		index = 93; //The data for october starts after 31+30+31 days of sept,aug and oct respectively.
	}
	else{
		printf("\n\t\tIncorrect month used\n\t\tUse between August-November inclusive; Only use numbers ; August = 8\n");
		return -1;
	}

	//If 1st or 2nd of August, start at timestep 23 (after 23 hours)
	if(((month == 8) && (day == 1))||((month == 8) && (day == 2))){
		offset = 23;
	//If in August; Gives correct result for starting timestep
	}else if (month == 8){
		offset = 23 + (day - 1) * TIMESTEPS_PER_DAY ;
	//23 added because 1st day only has 23 hours
	}else{
		offset = 23 + (index - 2 + day) * TIMESTEPS_PER_DAY;
	}

	return offset;

}

//###########################################################################################################################################//

static void HandleError( cudaError_t err,const char *file, int line ) {
    if (err != cudaSuccess) {
  		printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        exit( EXIT_FAILURE );
    }
}

//###########################################################################################################################################//
//###########################################################################################################################################//
//###########################################################################################################################################//


int main(int argc,char* argv[])
{

//--------------------------Checking for input arguments------------------------------//

	char baseFileName[] = "../../Birds_Full/Birds_data/InterpolatedData/";
	char yearFileName[80];
	char fullFileName[80];
	char start_date[12];
	char yearStr[4],monthStr[2],dayStr[2];

	float starting_row,starting_col;
	long int offset_into_data = 0;
	int NumOfBirds,year,day,month;

	int option;
	
	while ((option = getopt(argc, argv,"y:m:d:r:c:N:")) != -1) {
        	switch (option) {
             		case 'y' : year = atoi(optarg);
             		    break;
             		case 'm' : month = atoi(optarg);
             		    break;
             		case 'd' : day = atoi(optarg); 
             		    break;
             		case 'r' : starting_row = atof(optarg);
             		    break;
             		case 'c' : starting_col = atof(optarg);
             		    break;
             //		case 't' : breadth = atoi(optarg);
             //		    break;
             		case 'N' : NumOfBirds = atoi(optarg);
             		    break;
             		default: printf("\nUsage: birds -y Year -m Month(Number) -d DayOfTheMonth -r StartingRow -c StartingCol -N NumberOfBirds\n"); 
             		    exit(EXIT_FAILURE);
        	}
   	 }

	
	/** If starting row is greater than or equal the row that we are interested in; Below a particular row we are not interested in the flight of the birds**/
	if(starting_row >= MAX_LAT_SOUTH){
		printf("\t\tProvided starting row is below the southern most lattitude at which the model is set to stop\n");
		printf("\t\tEither change the starting row location and/or MAX_LAT upto which the birds can fly\n");
		return -1;
	}
	
//-----------------------------------------------Day-----------------------------------------//
/** Making sure random date is not provided **/

	if((day>0) && (day<32)){
		sprintf(dayStr,"%d",day);
	}else{
		printf("\t\t Invalid date provided; Date should be greater than 0 and less than 32\n");
		return -1;
	}

//-----------------------------------------------Month-----------------------------------------//
/** Making sure month provided is between August and November inclusive **/

	if((month < 12) && (month > 7)){
		sprintf(monthStr,"%d",month);
	}else{
		printf("\t\t Invalid month provided; Use between 8 and 11 inclusive\n");
		return -1;
	}

	/** Converting month and day information into number of timesteps; Special case of AUG 1st is also taken care of
	Instead of AUG 1 it starts at August 2 (because data starts at 7pm but birds fly at 6pm) **/
	offset_into_data = convert_to_month(month,day);
	
	printf("Offset into data is: %ld\n",offset_into_data);

//-----------------------------------------------Year-----------------------------------------//
/** Checking if correct year specified **/

	if((year>= 2008) && (year<=2013)){
		//Add file location here
		sprintf(yearStr,"%d",year);
		strcpy(yearFileName,baseFileName);
		strcat(yearFileName,yearStr);
		strcat(yearFileName,"/");
	}
	else{
		printf("\n\tInvalid year specified\n\tSpecified %d; Use years from 2008 to 2013 in its full format\n",year);
             	printf("\t\tUsage: birds -y Year -m Month(Number) -d DayOfTheMonth -r StartingRow -c StartingCol -N NumberOfBirds\n"); 
		return -1;		
	}

	strcpy(start_date,yearStr);
	strcat(start_date,"/");	
	strcat(start_date,monthStr);
	strcat(start_date,"/");
	sprintf(dayStr,"%d",day);
	strcat(start_date,dayStr);

//------------Opening row and column data file where lat and long data will be stored----------------//
	FILE *rowdataTxt,*coldataTxt,*birdStatusTxt;
	FILE *vdataTxt,*udataTxt,*v10dataTxt,*u10dataTxt,*precipTxt,*pressureTxt,*lwTxt,*dirTxt;

	rowdataTxt = fopen("row_output.txt","a");
	if(rowdataTxt == NULL) {
		perror("Cannot open output row data file\n");
		return -1;
	}

	coldataTxt = fopen("col_output.txt","a");
	if(coldataTxt == NULL) {
		perror("Cannot open output col data file\n");
		return -1;
	}

	birdStatusTxt = fopen("birdStatus_Final.txt","a");
	if(birdStatusTxt == NULL) {
		perror("Cannot open output birdStatus file\n");
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
//--------------------------Opening Direction file 
//--------------------(Example: ext_crop.txt or extP_crop.txt)-------------//

	dirTxt = fopen("./Lw_and_Dir/ext_Final_NewCoordSystem.txt","r");
	//dirTxt = fopen("ext_crop.txt","r");
	if(dirTxt == NULL) {
		perror("Cannot open file with direction data\n");
		return -1;
	}


//-----------------------------Setting Heap Size,printf buffer size etc--------------------------------------------//
//	size_t limit;
//	HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 500 * 1024 * 1024));
//	cudaDeviceGetLimit(&limit,cudaLimitPrintfFifoSize);


//	HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize,(size_t)(6 * LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float))));
//--------------------------Memory Allocation for global arrays containing weather data----------------------------//
	float *h_row,*h_col;
	float *d_row,*d_col;	
	float *d_udata,*d_vdata,*d_u10data,*d_v10data,*d_lwData;
	float *d_dirData,*d_precipData,*d_pressureData;
	uint8_t *h_birdStatus,*d_birdStatus;

	dirData = (float*) malloc(LAT_SIZE * LONG_SIZE * sizeof(float));
	h_row = (float*) malloc(NumOfBirds * (TIMESTEPS + 1) * sizeof(float));
	h_col = (float*) malloc(NumOfBirds * (TIMESTEPS + 1) * sizeof(float));
	h_birdStatus = (uint8_t*)malloc(NumOfBirds * sizeof(uint8_t));
	lwData = (float*) malloc(LAT_SIZE * LONG_SIZE * sizeof(float));	

	udata = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	vdata = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	u10data = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	v10data = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	precipData = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	pressureData = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	

//------------------------------------------------------------------------------------------------------------------//
/*
	HANDLE_ERROR(cudaMallocHost((void**)&udata,LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&vdata,LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&u10data,LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float)));	
	HANDLE_ERROR(cudaMallocHost((void**)&v10data,LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&precipData,LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&pressureData,LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&lwData,LAT_SIZE * LONG_SIZE * sizeof(float)));	
*/

	
	printf("Size of large arrays is %zd\n",sizeof(udata)/sizeof(udata[0]));
	printf("Size of large arrays is %ld\n",sizeof(udata)/sizeof(float));
	printf("Size of large arrays is %d\n",sizeof(udata)/sizeof(float));

	int ii;
	for(ii=0;ii<(NumOfBirds * (TIMESTEPS + 1));ii++){
		*(h_row + ii) = starting_row;
		*(h_col + ii) = starting_col;
	}

	for(ii=0;ii<NumOfBirds;ii++){
		h_birdStatus[ii] = (uint8_t)1;
	}

//--------------------------Initializing the structures-------------------------------------------------------------------//

	inpStruct[0].fp = vdataTxt;
	inpStruct[0].inpVals = vdata;

	inpStruct[1].fp = udataTxt;
	inpStruct[1].inpVals = udata;

	inpStruct[2].fp = v10dataTxt;
	inpStruct[2].inpVals = v10data;

	inpStruct[3].fp = u10dataTxt;
	inpStruct[3].inpVals = u10data;

	inpStruct[4].fp = precipTxt;
	inpStruct[4].inpVals = precipData;

	inpStruct[5].fp = pressureTxt;
	inpStruct[5].inpVals = pressureData;

	inpStruct[6].fp = lwTxt;
	inpStruct[6].inpVals = lwData;

	inpStruct[7].fp = dirTxt;
	inpStruct[7].inpVals = dirData;


	/** Using pthreads to read from the files in parallel**/
	pthread_t threads[8];

	int i;
	for(i=0;i<8;i++){
		if(pthread_create(&threads[i],NULL,read_dataFiles,(void*)&inpStruct[i]) != 0){
			fprintf(stderr,"ERROR: Thread creation using pthreads failed\n");
			return -1;
		}

	}

	for(i=0;i<8;i++){
		if(pthread_join(threads[i],NULL)!=0){
 			fprintf(stderr,"ERROR: Thread join failed\n");
                        return -1;
		}
	}


	printf("End of parallel data read\n");
	
//-----------------------------------Getting Random Values-------------------------------------------//
	int DeviceCount;
	float *rand_norm_nums;
	curandState_t* states;

	/** Getting the total number of devices available **/
	HANDLE_ERROR(cudaGetDeviceCount(&DeviceCount));
	HANDLE_ERROR(cudaSetDevice(DeviceCount - 1));
	HANDLE_ERROR(cudaDeviceReset());

	HANDLE_ERROR(cudaMalloc((void**)&states,NumOfBirds * (TIMESTEPS+1) * sizeof(curandState_t)));
	HANDLE_ERROR(cudaMalloc((void**)&rand_norm_nums,NumOfBirds * (TIMESTEPS+1) * sizeof(float)));

	//Making each block have total threads of 32
	//GridSize setup such that total y grid is of size NumOfBirds and x grid is of size TIMESTEPS
	dim3 blockSize1(32,1,1); 
	dim3 gridSize1(((TIMESTEPS) + 31)/32,NumOfBirds,1);
	
	setup_kernel<<<gridSize1,blockSize1>>>(time(NULL),states,NumOfBirds);
	HANDLE_ERROR(cudaDeviceSynchronize());
	generate_kernel<<<gridSize1,blockSize1>>>(states,rand_norm_nums,NumOfBirds);
	HANDLE_ERROR(cudaDeviceSynchronize());

	/* print them out */
/*	for ( j = 0; j < LAT_SIZE; j++) {
		for( i = 0;i<LONG_SIZE;i++){
			//printf("%f ", cpu_nums[j*LONG_SIZE + i]);
			if(i == LONG_SIZE -1) {
				printf("%f\n",dir_u[j * LAT_SIZE + i]);
			}
			else {
				printf("%f ",dir_u[j * LAT_SIZE + i]);
			}
		}
//		printf("\n");
	}
*/

	HANDLE_ERROR(cudaDeviceSynchronize());

	// free the memory we allocated for the states 
	HANDLE_ERROR(cudaFree(states));

	printf("Random number generator is working\n");

//-------------------------------------------------------------------------------------------------------------------------//	
	HANDLE_ERROR(cudaMalloc((void**)&d_row,NumOfBirds * (TIMESTEPS + 1 ) * sizeof(float)));	
	HANDLE_ERROR(cudaMalloc((void**)&d_col,NumOfBirds * (TIMESTEPS + 1 ) * sizeof(float)));	
	HANDLE_ERROR(cudaMalloc((void**)&d_lwData,LAT_SIZE * LONG_SIZE * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&d_dirData,LAT_SIZE * LONG_SIZE * sizeof(float)));


	HANDLE_ERROR(cudaMemcpy(d_row,h_row,NumOfBirds * (TIMESTEPS + 1 ) * sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_col,h_col,NumOfBirds * (TIMESTEPS + 1 ) * sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_lwData,lwData,LAT_SIZE * LONG_SIZE * sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_dirData,dirData,LAT_SIZE * LONG_SIZE * sizeof(float),cudaMemcpyHostToDevice));
//-------------------------------------------------------------------------------------------------------------//	
	size_t MemoryEachVar,DataPerTransfer,SizePerTimestep;
	int TimestepsPerTransfer,TimestepsLastTransfer,DaysPerTransfer;		
	size_t MemoryRemaining,TotalMemory;

	HANDLE_ERROR(cudaSetDevice(DeviceCount - 1));

	// Getting the total remaining memory that the device can allocate 
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));

	MemoryRemaining -= 2*NumOfBirds* (TIMESTEPS + 1) * sizeof(float);
	MemoryRemaining -= NumOfBirds * sizeof(uint8_t);	
	//Need to make sure 100MB is free!! For some reason
	MemoryRemaining -= 100 * 1000000;

	
	printf("Total mem: %zd,Free mem: %zd\n",TotalMemory,MemoryRemaining);

 	printf("\n\n\t\tTotal Memory remaining is: %zd \n",MemoryRemaining);

	//Memory that each variable gets every timestep
	MemoryEachVar = MemoryRemaining/NUM_DATA_FILES;

	printf("\t\tMemory for each variable is: %zd \n",MemoryEachVar);

	// Need to send data per timestep so has to be a multiple of LAT_SIZE *LONG_SIZE* sizeof(float) * 24
	//Can also be called as Minimum_Size_Per_Timestep; Sending data so that it is according to days
	SizePerTimestep = LAT_SIZE * LONG_SIZE * TIMESTEPS_PER_DAY * sizeof(float);

	// To get a number divisible by SizePerTimestep
	//DataPerTransfer is the data size to be transferred for each variable
	//Example, if 100MB then 100MB for each of the vars is transferred each time
	DataPerTransfer = (MemoryEachVar/SizePerTimestep) * SizePerTimestep;
	DaysPerTransfer = DataPerTransfer/SizePerTimestep;
	TimestepsPerTransfer = DaysPerTransfer * TIMESTEPS_PER_DAY;
	printf("\t\tChecking Division: %zd\n",MemoryEachVar/SizePerTimestep);		
	printf("\t\tTotal Timesteps per Transfer of data is: %ld \n",TimestepsPerTransfer); 
	printf("\t\tData per transfer is %zd\n",DataPerTransfer);
	printf("\t\tDays per transfer is %zd\n",DaysPerTransfer);		
	
//------------------------------------Getting the size of data needed per transfer---------------------------------------------//
	int divisible,Transfers;
//	long int DataLastTransfer;//Per variable

	Transfers = (TIMESTEPS - INITIAL_SKIP_TIMESTEPS) / TimestepsPerTransfer;

	divisible = (TIMESTEPS - INITIAL_SKIP_TIMESTEPS) % TimestepsPerTransfer;
	
	if(divisible != 0){
		Transfers++;
	}
	
	printf("\t\tTotal Transfers required: %ld\n\n",Transfers);
	/** Tota bytes transfered per data transfer**/

	const int TotalTransfers = Transfers;
	TimestepsLastTransfer = (TIMESTEPS) - (Transfers-1)*TimestepsPerTransfer;
/*
	cudaStream_t stream[TotalTransfers-1];
	for(i=0;i<TotalTransfers-1;i++){
		HANDLE_ERROR(cudaStreamCreate(&stream[i]));
	}
*/
	//DataLastTransfer = (TIMESTEPS * LAT_SIZE * LONG_SIZE * sizeof(float)) - (DataPerTransfer * (TotalTransfers-1)); 

//---------------------------------------Memory allocation per transfer----------------------------------------------------------//

	long int start_timestep,cur_timestep,max_timesteps,ptrOffset;
	ptrOffset = INITIAL_SKIP_TIMESTEPS;

	cur_timestep = offset_into_data;

	for(i=0;i<TotalTransfers-1;i++){


		HANDLE_ERROR(cudaSetDevice(DeviceCount - 1));
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(Before any allocation): %zd\n",TotalMemory,MemoryRemaining);

		
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After SetDevice): %zd\n",TotalMemory,MemoryRemaining);

		//HANDLE_ERROR(cudaStreamCreate(&stream[i]));
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After Stream Create): %zd\n",TotalMemory,MemoryRemaining);

		HANDLE_ERROR(cudaMalloc((void**)&d_udata,DataPerTransfer));	
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After udata allocation): %zd\n",TotalMemory,MemoryRemaining);
	
		HANDLE_ERROR(cudaMalloc((void**)&d_vdata,DataPerTransfer));	
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After vdata allocation): %zd\n",TotalMemory,MemoryRemaining);

		HANDLE_ERROR(cudaMalloc((void**)&d_u10data,DataPerTransfer));	
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After u10data allocation): %zd\n",TotalMemory,MemoryRemaining);

		HANDLE_ERROR(cudaMalloc((void**)&d_v10data,DataPerTransfer));	
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After v10data allocation): %zd\n",TotalMemory,MemoryRemaining);

		HANDLE_ERROR(cudaMalloc((void**)&d_precipData,DataPerTransfer));	
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After precipData allocation): %zd\n",TotalMemory,MemoryRemaining);

		HANDLE_ERROR(cudaMalloc((void**)&d_pressureData,DataPerTransfer));	
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After pressureData allocation): %zd\n",TotalMemory,MemoryRemaining);
	
		HANDLE_ERROR(cudaMalloc((void**)&d_birdStatus,NumOfBirds * sizeof(uint8_t)));
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After birdStatus allocation): %zd\n",TotalMemory,MemoryRemaining);
	
		HANDLE_ERROR(cudaDeviceSynchronize());


		printf("After all the host allocations %d\n",i);



	//-----------------------------------------Initializing gridSize and block Size-------------------------------//		
		//HANDLE_ERROR(cudaSetDevice(DeviceCount - 1));

		dim3 gridSize((NumOfBirds + 32 - 1)/32,1,1);
		dim3 blockSize(32,1,1);

		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After grid and block init): %zd\n",TotalMemory,MemoryRemaining);
	//-----------------------------------------Copying data from CPU to GPU------------------------------------------------//	

		HANDLE_ERROR(cudaSetDevice(DeviceCount - 1));	

		HANDLE_ERROR(cudaMemcpy(d_udata,udata+ptrOffset,DataPerTransfer,cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_vdata,vdata+ptrOffset,DataPerTransfer,cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_u10data,u10data+ptrOffset,DataPerTransfer,cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_v10data,v10data+ptrOffset,DataPerTransfer,cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_precipData,precipData+ptrOffset,DataPerTransfer,cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_pressureData,pressureData+ptrOffset,DataPerTransfer,cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_birdStatus,h_birdStatus,NumOfBirds * sizeof(uint8_t),cudaMemcpyHostToDevice));

/*
		HANDLE_ERROR(cudaMemcpyAsync(d_lwData,lwData,LAT_SIZE * LONG_SIZE * sizeof(float),cudaMemcpyHostToDevice,stream[i]));
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After grid and block init): %zd\n",TotalMemory,MemoryRemaining);
		HANDLE_ERROR(cudaMemcpyAsync(d_udata,udata + ptrOffset,DataPerTransfer,cudaMemcpyHostToDevice,stream[i]));
		HANDLE_ERROR(cudaMemcpyAsync(d_vdata,(vdata+ptrOffset),DataPerTransfer,cudaMemcpyHostToDevice,stream[i]));
		HANDLE_ERROR(cudaMemcpyAsync(d_u10data,(u10data+ptrOffset),DataPerTransfer,cudaMemcpyHostToDevice,stream[i]));
		HANDLE_ERROR(cudaMemcpyAsync(d_v10data,(v10data+ptrOffset),DataPerTransfer,cudaMemcpyHostToDevice,stream[i]));
		HANDLE_ERROR(cudaMemcpyAsync(d_precipData,(precipData+ptrOffset),DataPerTransfer,cudaMemcpyHostToDevice,stream[i]));
		HANDLE_ERROR(cudaMemcpyAsync(d_pressureData,(pressureData+ptrOffset),DataPerTransfer,cudaMemcpyHostToDevice,stream[i]));
*/
	//-----------------------------------------Calling the Kernel-----------------------------------------------------------//
		
		//All of these are inclusive
		//If TimeStepsPerTransfer is 9, then they would be: 0-8, 9-17, 18-26,...
		max_timesteps = ((i+1) * TimestepsPerTransfer) - 1 + INITIAL_SKIP_TIMESTEPS;
		

		printf("Current timestep variable is:%ld\n",cur_timestep);
		printf("Max timestep is: %ld\n",max_timesteps);
		printf("Offset into data is:%ld\n",offset_into_data);


		start_timestep = i * TimestepsPerTransfer + INITIAL_SKIP_TIMESTEPS;

		if((max_timesteps - offset_into_data) > TimestepsPerTransfer){
			cur_timestep = start_timestep;
		}else{
			cur_timestep = offset_into_data;
		}


		printf("Current timestep variable after checking if offset less than max_timesteps is:%ld\n",cur_timestep);

		bird_movement<<<gridSize,blockSize>>>(d_row,d_col,NumOfBirds,start_timestep,cur_timestep,max_timesteps,d_udata,d_vdata,
						d_u10data,d_v10data,d_dirData,rand_norm_nums,d_precipData,d_pressureData,d_lwData,d_birdStatus);


		//HANDLE_ERROR(cudaStreamSynchronize(stream[i]));
		HANDLE_ERROR(cudaDeviceSynchronize());
	//---------------------------------Freeing allocated memory in GPU and pinned memory in CPU-------------------//
		printf("Before freeing;Inside the loop\n");

		HANDLE_ERROR(cudaMemcpy(h_birdStatus,d_birdStatus,NumOfBirds * sizeof(uint8_t),cudaMemcpyDeviceToHost));



		//HANDLE_ERROR(cudaStreamDestroy(stream[i]));	
		HANDLE_ERROR(cudaFree(d_udata));
		HANDLE_ERROR(cudaFree(d_vdata));
		HANDLE_ERROR(cudaFree(d_u10data));
		HANDLE_ERROR(cudaFree(d_v10data));
		HANDLE_ERROR(cudaFree(d_precipData));
		HANDLE_ERROR(cudaFree(d_pressureData));

		
		ptrOffset = (DataPerTransfer/sizeof(float)) * (i + 1) + INITIAL_SKIP_TIMESTEPS;
		printf("After all freeing %d\n",i);
		
	}
/*
	HANDLE_ERROR(cudaMemcpy(h_row,d_row,NumOfBirds * (TIMESTEPS + 1 ) * sizeof(float),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(h_col,d_col,NumOfBirds * (TIMESTEPS + 1 ) * sizeof(float),cudaMemcpyDeviceToHost));
	

	for(i = 0;i < NumOfBirds * (TIMESTEPS + 1); i++ ){
		printf("%f ",h_row[i]);	
		if(i == TIMESTEPS){
			printf("%f \n",h_row[i]);
		}

	}


	printf("\n\n");
	for(i = 0;i < NumOfBirds * (TIMESTEPS + 1); i++ ){
		printf("%f ",h_col[i]);	
		if(i == TIMESTEPS){
			printf("%f \n",h_col[i]);
		}

	}
*/
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------------Last Iteration-----------------------------------------//
//-----------------------------------------------------------------------------------------------------------//

	// Last iteration where the size might not be the same as others 
	long int DataRemaining;
	DataRemaining = (LONG_SIZE * LAT_SIZE * TIMESTEPS * sizeof(float)) - (DataPerTransfer * (TotalTransfers-1));

	start_timestep = (TotalTransfers - 1) * TimestepsPerTransfer + INITIAL_SKIP_TIMESTEPS;
	max_timesteps = TIMESTEPS;
	ptrOffset = (DataPerTransfer/sizeof(float)) * (TotalTransfers - 1) + INITIAL_SKIP_TIMESTEPS;
	
	dim3 gridSize((NumOfBirds + 32 - 1)/32,1,1);
	dim3 blockSize(32,1,1); 
//----------------------------------------------------------------------------------------//

	HANDLE_ERROR(cudaSetDevice(DeviceCount - 1));
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(Before any allocation): %zd\n",TotalMemory,MemoryRemaining);
	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After SetDevice): %zd\n",TotalMemory,MemoryRemaining);

	//HANDLE_ERROR(cudaStreamCreate(&stream[i]));
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After Stream Create): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMalloc((void**)&d_udata,DataRemaining));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After udata allocation): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMalloc((void**)&d_vdata,DataRemaining));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After vdata allocation): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMalloc((void**)&d_u10data,DataRemaining));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After u10data allocation): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMalloc((void**)&d_v10data,DataRemaining));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After v10data allocation): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMalloc((void**)&d_precipData,DataRemaining));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After precipData allocation): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMalloc((void**)&d_pressureData,DataRemaining));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After pressureData allocation): %zd\n",TotalMemory,MemoryRemaining);



	HANDLE_ERROR(cudaDeviceSynchronize());


	printf("After all the host allocations %d\n",i);


//-----------------------------------------Copying data from CPU to GPU----------------------------------------//

	HANDLE_ERROR(cudaSetDevice(DeviceCount - 1));	

	HANDLE_ERROR(cudaMemcpy(d_udata,udata+ptrOffset,DataRemaining,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_vdata,vdata+ptrOffset,DataRemaining,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_u10data,u10data+ptrOffset,DataRemaining,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_v10data,v10data+ptrOffset,DataRemaining,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_precipData,precipData+ptrOffset,DataRemaining,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_pressureData,pressureData+ptrOffset,DataRemaining,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_birdStatus,h_birdStatus,NumOfBirds * sizeof(uint8_t),cudaMemcpyHostToDevice));

//-----------------------------------------Calling the Kernel-------------------------------------------------//


	if((max_timesteps - offset_into_data) > TimestepsLastTransfer){
		cur_timestep = start_timestep;
	}else{
		cur_timestep = offset_into_data;
	}



	printf("Before calling the kernel\n");
	bird_movement<<<gridSize,blockSize>>>(d_row,d_col,NumOfBirds,start_timestep,cur_timestep,max_timesteps,d_udata,d_vdata,
						d_u10data,d_v10data,d_dirData,rand_norm_nums,d_precipData,d_pressureData,d_lwData,d_birdStatus);

	HANDLE_ERROR(cudaDeviceSynchronize());

	HANDLE_ERROR(cudaMemcpy(h_row,d_row,NumOfBirds * (TIMESTEPS + 1 ) * sizeof(float),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(h_col,d_col,NumOfBirds * (TIMESTEPS + 1 ) * sizeof(float),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(h_birdStatus,d_birdStatus,NumOfBirds * sizeof(uint8_t),cudaMemcpyDeviceToHost));

	for(i = 0;i < NumOfBirds * (TIMESTEPS + 1); i++ ){
		fprintf(rowdataTxt,"%f ",h_row[i]);	
		if(((i+1) % (TIMESTEPS + 1)) == 0){
			fprintf(rowdataTxt,"%f \n",h_row[i]);
		}

	}

	for(i = 0;i < NumOfBirds * (TIMESTEPS + 1); i++ ){
		fprintf(coldataTxt,"%f ",h_col[i]);	
		if(((i+1) % (TIMESTEPS + 1)) == 0){
			fprintf(coldataTxt,"%f \n",h_col[i]);
		}

	}

	for(i = 0;i < NumOfBirds; i++ ){
		fprintf(birdStatusTxt,"%d\n",h_birdStatus[i]);
	}
	//Get birdStatus array back as well
//-----------------------------------------------Freeing allocated memory--------------------------------------//
//	HANDLE_ERROR(cudaStreamDestroy(stream[0]));
	HANDLE_ERROR(cudaFree(rand_norm_nums));
	HANDLE_ERROR(cudaFree(d_birdStatus));		
	HANDLE_ERROR(cudaFree(d_udata));
	HANDLE_ERROR(cudaFree(d_vdata));
	HANDLE_ERROR(cudaFree(d_u10data));
	HANDLE_ERROR(cudaFree(d_v10data));
	HANDLE_ERROR(cudaFree(d_precipData));
	HANDLE_ERROR(cudaFree(d_pressureData));
/*	
	HANDLE_ERROR(cudaFreeHost(udata));
	HANDLE_ERROR(cudaFreeHost(vdata));
	HANDLE_ERROR(cudaFreeHost(u10data));
	HANDLE_ERROR(cudaFreeHost(v10data));
	HANDLE_ERROR(cudaFreeHost(precipData));
	HANDLE_ERROR(cudaFreeHost(pressureData));
	HANDLE_ERROR(cudaFreeHost(lwData));
*/

	free(dirData);
	free(udata);
	free(vdata);
	free(u10data);
	free(v10data);
	free(precipData);
	free(pressureData);
	free(lwData);
	free(h_birdStatus);
/*
	HANDLE_ERROR(cudaFree(d_lwData));	
	HANDLE_ERROR(cudaFree(d_u_dirAngle));
	HANDLE_ERROR(cudaFree(d_v_dirAngle));
	printf("After freeing everything\n");
*/
	HANDLE_ERROR(cudaFree(d_row));	
	HANDLE_ERROR(cudaFree(d_col));	
	free(h_row);
	free(h_col);
	//free(lwData);
	//free(dirData);
	
	fclose(birdStatusTxt);
	fclose(dirTxt);
	fclose(udataTxt);
	fclose(vdataTxt);
	fclose(v10dataTxt);
	fclose(u10dataTxt);
	fclose(precipTxt);
	fclose(pressureTxt);
	fclose(lwTxt);
	fclose(rowdataTxt);
	fclose(coldataTxt);
	
	printf("End\n");
	return 0;
}

