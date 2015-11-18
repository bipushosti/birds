
//Needs Header Files for the functions; The header file should have both C and CUDA functions



//This file uses 6 hourly data. Each day is 6 hours long and skipping a day means to add 6
//to the counter that counts the timesteps (l).

//The birds start at 00:00 UTC which is 6pm in central time when there is no day light savings

#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <getopt.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

#include "birds_CUDA.h"
//#define CUDA_API_PER_THREAD_DEFAULT_STREAM


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
//To make the birds head back directly west the angle must be set to 180.
#define BIRD_SEA_ANGLE		180

#define TOTAL_DATA_FILES	9
//Total number of data files or variables bird flight depends on;Does not include direction files and land water data
#define NUM_DATA_FILES		6

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
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
__device__ float bilinear_interpolation_SmallData(float x,float y,float* data_array);
__device__ float bilinear_interpolation_LargeData(float x,float y,float* data_array,long l);

__device__ float getProfitValue(float u_val,float v_val,float dirVal,float dir_u,float dir_v);
__device__ long bird_AtSea(float pos_row,float pos_col,long l,float* udata,float* vdata,float* dir_u,float* dir_v,float* lwData);
__global__ void bird_movement(float pos_row,float pos_col,long l,float* udata,float* vdata,float* u10data,float* v10data,float* dirData,float* dir_u,float* dir_v,float* precipData,float* pressureData,float* lwData);

void read_dataFiles(FILE* textFile,float* dataArray);
long convert_to_month(int month,int day);

static void HandleError( cudaError_t err,const char *file, int line );
long Get_GPU_devices();
//-------------------------------------------------------------------------------------------------------------------------------------
struct file_IO {
	FILE *fp;
	float* inpVals;
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
//###########################################################################################################################################//

static void HandleError( cudaError_t err,const char *file, int line ) {
    if (err != cudaSuccess) {
  		printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
//		cout << cudaGetErrorString(err) << "in" << file << "at line" << line << "\n";
        exit( EXIT_FAILURE );
    }
}

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
	//for(int i = 0;i<DeviceCount;i++){
	cudaDeviceProp properties;
	HANDLE_ERROR(cudaGetDeviceProperties(&properties,i));
	printf("Device Number: %d\n", i);
	printf("  Device name: %s\n", properties.name);
	printf("  Device Global Memory size: %zd MB \n",properties.totalGlobalMem/1000000);
	printf("\n");
	
	deviceMemory = properties.totalGlobalMem;
	//}


	return deviceMemory;
}
//###########################################################################################################################################//

__global__ void setup_kernel(unsigned int seed,curandState *states)
{

	//Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int id = y * LONG_SIZE + x;

	curand_init(seed,id,0,&states[id]);
}

//###########################################################################################################################################//

__global__ void generate_kernel(curandState *states,float* numbers,float* angles,float* u_dirAngles,float* v_dirAngles,float speed)
{

	//Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int id = y * LONG_SIZE + x;
	
	float value;

	numbers[id] = curand_normal(&states[id]);

	if(id > (LONG_SIZE*LAT_SIZE -1)) return;
	else{
		
		u_dirAngles[id] = speed * cosf(angles[id] * (PI/180));
		v_dirAngles[id] = speed * sinf(angles[id] * (PI/180));

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

//###########################################################################################################################################//

__device__ long bird_AtSea(float pos_row,float pos_col,long l,float* udata,float* vdata,float* lwData)
{
	long count_timeSteps = l;
	float u_val,v_val,u_dir,v_dir;
	int index = 0;

	index = lwData[(int)(rintf(pos_row)) * LONG_SIZE + (int)(rintf(pos_col))];

	while(index != 1){

		/** Bilinear interpolation for u and v data **/
		u_val = bilinear_interpolation_LargeData(pos_col,pos_row,udata,l);	
		v_val = bilinear_interpolation_LargeData(pos_col,pos_row,vdata,l);
	
		u_dir = DESIRED_SPEED * cosf(BIRD_SEA_ANGLE * (PI/180));
		v_dir = DESIRED_SPEED * sinf(BIRD_SEA_ANGLE * (PI/180));

		/** Desired speed needs to change in the case of column position or the birds
		will not fly west **/
		pos_row = pos_row + (v_val + v_dir) * 0.36 * -1;	
		pos_col = pos_col + (u_val + u_dir) * 0.36;

		index = lwData[(int)(rintf(pos_row)) * LONG_SIZE + (int)(rintf(pos_col))];

		count_timeSteps = count_timeSteps + 1;

		if(count_timeSteps > 79){
			printf("Dead Bird! Bird has been flying for 80 hours straight!\n");
			return -1;
		}

		if(pos_row >= MAX_LAT_SOUTH){
			printf("Bird reached maximum lattitude; Exiting program\n");
			return -1;
		}
	}
	/** Waiting till next 6pm **/
	while((count_timeSteps+1) % 24 != 0){
		/** Add to position arrays**/
		count_timeSteps ++;
	}
	/** Going back to 6pm after certain stopover days **/
	count_timeSteps = (STOPOVER_DAYS +1)*24 + count_timeSteps - 1 ;

	return count_timeSteps;
	
	
}

//###########################################################################################################################################//

__device__ float bilinear_interpolation_SmallData(float x,float y,float* data_array)
{
	float x1,y1,x2,y2;
	float Q11,Q12,Q21,Q22,R1,R2,R;
	//float val_x1,val_x2,val_y1,val_y2;

	x1 = floorf(x);
	x2 = ceilf(x);
	y1 = floorf(y);
	y2 = ceilf(y);
	R = 0;
	
	Q11 = data_array[(int)(y1 * LONG_SIZE + x1)];
	Q12 = data_array[(int)(y2 * LONG_SIZE + x1)];
	Q21 = data_array[(int)(y1 * LONG_SIZE + x2)];
	Q22 = data_array[(int)(y2 * LONG_SIZE + x2)];
	

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
	//float val_x1,val_x2,val_y1,val_y2;

	x1 = floorf(x);
	x2 = ceilf(x);
	y1 = floorf(y);
	y2 = ceilf(y);
	R = 0;
	
	Q11 = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y1 * LONG_SIZE + x1) ];
	Q12 = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y2 * LONG_SIZE + x1) ];
	Q21 = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y1 * LONG_SIZE + x2) ];
	Q22 = data_array[(int)(l  * LAT_SIZE * LONG_SIZE + y2 * LONG_SIZE + x2) ];
	

	R1 = Q11 + (x - x1)*(Q21 - Q11);
	R2 = Q12 + (x - x1)*(Q22 - Q12);
	R = R1 + (y - y1)*(R2 - R1);

	
	//printf("Q11:%f,Q12:%f,Q21:%f,Q22:%f; And Value=%f\n",Q11,Q12,Q21,Q22,value);
	return R;
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

	diffAngle = acosf( (u_val*dir_u + v_val * dir_v)/ (( hypotf(u_val,v_val) * hypotf(dir_u,dir_v) )) ) * 180/PI;

	/** Separate profit value methods have to be used if the tail component is less that equal to or greater than the desired speed of the birds **/
	if(tailComponent <= DESIRED_SPEED) {	
		profit_value = (DESIRED_SPEED * DESIRED_SPEED) + magnitude_squared - 2 * DESIRED_SPEED * magnitude * cosf(diffAngle * PI/180);
		profit_value = DESIRED_SPEED - sqrtf(profit_value);
	}
	else {
		/** Perpendicular to a vector (x,y) is (y,-x) or (-y,x) Cross component is always positive **/

		crossComponent = fabsf((-dir_v*u_val + dir_u*v_val)/hypotf(dir_v,dir_u));
		profit_value = tailComponent - crossComponent;
	}

	return profit_value;
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
				//printf("%s \n",tempVal);

				if(strcmp("NaN\n",tempVal)==0) {
					Value = 0.0;
				}
				else{
					Value = atof(tempVal);
				}
				dataArray[j * LAT_SIZE + i] = Value;
				//printf("%d,%f \n",i,Value);
			}
		}
		j++;
	}
	return NULL;
}

//###########################################################################################################################################//

long convert_to_month(int month,int day)
{
	long index,offset;
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
	
	offset = (index-1)  + day* TIMESTEPS_PER_DAY;
	return offset;

}

//###########################################################################################################################################//

__global__ void bird_movement(float pos_row,float pos_col,long l,float* udata,float* vdata,float* u10data,float* v10data,float* dirData,float* dir_u,float* dir_v,float* precipData,float* pressureData,float* lwData)
{
	//Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int id = y * LONG_SIZE + x;

	long l_old;	

	float profit_value,actualAngle;
	float last_pressure,pressure_sum,pressure_MultSum,slope;
	float u_ten,v_ten,u_val,v_val,uDir_value,vDir_value; //precip_val;
	int k;

	slope = 0;

	while(l < (TOTAL_DAYS * TIMESTEPS_PER_DAY - 24)){
	
		/** Rounding down to the nearest int **/
		uDir_value = dir_u[__float2int_rd(pos_row * LAT_SIZE + pos_col)];
		vDir_value = dir_v[__float2int_rd(pos_row * LAT_SIZE + pos_col)];
	
		u_ten = bilinear_interpolation_LargeData(pos_col,pos_row,u10data,l);
		v_ten = bilinear_interpolation_LargeData(pos_col,pos_row,v10data,l);

		profit_value = getProfitValue(u_ten,v_ten,actualAngle,uDir_value,vDir_value);

		if((profit_value >= MIN_PROFIT) && ((last_pressure>=1009)||(slope >-1))){

			for(k=0;k<6;k++,l++) {
				u_val = bilinear_interpolation_LargeData(pos_col,pos_row,udata,l);
				v_val = bilinear_interpolation_LargeData(pos_col,pos_row,vdata,l);
				//precip_val = bilinear_interpolation_LargeData(pos_col,pos_row,precipData,l);

				pos_row = pos_row + (v_val + vDir_value ) * 0.36 * -1;
				pos_col = pos_col + (u_val + uDir_value) * 0.36;
			}	


			/** If the bird is at sea after the first 6 hours of flight **/			
			if(lwData[__float2int_rd(pos_row * LAT_SIZE + pos_col)] != 1){

				for(k=6;k<10;k++,l++){
					/** Rounding down to the nearest int **/
					uDir_value = dir_u[__float2int_rd(pos_row * LAT_SIZE + pos_col)];
					vDir_value = dir_v[__float2int_rd(pos_row * LAT_SIZE + pos_col)];

					u_val = bilinear_interpolation_LargeData(pos_col,pos_row,udata,l);
					v_val = bilinear_interpolation_LargeData(pos_col,pos_row,vdata,l);

					pos_row = pos_row + (v_val + vDir_value ) * 0.36 * -1;
					pos_col = pos_col + (u_val + uDir_value) * 0.36;
				}
			}

			/** If at sea even after the 4 hours **/
			if(lwData[__float2int_rd(pos_row * LAT_SIZE + pos_col)] != 1){
				l = bird_AtSea(pos_row,pos_col,l,udata,vdata,lwData);
			}		
		}
		else{
			l += 24;
		}

		l_old = l - REGRESSION_HRS;

		//Taking the pressure from 6 hours earlier of the location where the bird landed
		for(k=1; (l_old < l) && (k<=REGRESSION_HRS); l_old++,k++){

			pressure_sum += bilinear_interpolation_LargeData(pos_col,pos_row,pressureData,l_old);
			pressure_MultSum += k * bilinear_interpolation_LargeData(pos_col,pos_row,pressureData,l_old);

			//last_pressure is the last day or the day of flight
			if(k == REGRESSION_HRS) {
				last_pressure = bilinear_interpolation_LargeData(pos_col,pos_row,pressureData,l_old);
			}
		}
		slope = ((REGRESSION_HRS * pressure_MultSum) - (pressure_sum * HRS_SUM))/(DENOM_SLOPE);
		
	}
	
}


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
	long offset_into_data = 0;
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
             		default: printf("Usage: birds -y Year -m Month(Number) -d DayOfTheMonth -r StartingRow -c StartingCol -N NumberOfBirds\n"); 
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
	if((month == 8) && (day == 1)){
		offset_into_data = 22;
	}
	else {
		offset_into_data = convert_to_month(month,day);
	}

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
	dirTxt = fopen("./Lw_and_Dir/ext_Final_NewCoordSystem.txt","r");
	//dirTxt = fopen("ext_crop.txt","r");
	if(dirTxt == NULL) {
		perror("Cannot open file with direction data\n");
		return -1;
	}

	
//--------------------------Memory Allocation-----------------------------------//


	dirData = (float*)malloc(LAT_SIZE * LONG_SIZE * sizeof(float));	
	dir_u = (float*)malloc(LAT_SIZE * LONG_SIZE * sizeof(float));
	dir_v = (float*)malloc(LAT_SIZE * LONG_SIZE * sizeof(float));
	lwData = (float*)malloc(LAT_SIZE * LONG_SIZE * sizeof(float));

	udata = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	vdata = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	u10data = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	v10data = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));

	precipData = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	pressureData = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));

/*	HANDLE_ERROR(cudaMallocHost((void**)&dirData,LAT_SIZE * LONG_SIZE * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&dir_u,LAT_SIZE * LONG_SIZE * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&dir_v,LAT_SIZE * LONG_SIZE * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&lwData,LAT_SIZE * LONG_SIZE * sizeof(float)));

	HANDLE_ERROR(cudaMallocHost((void**)&udata,LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&vdata,LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&u10data,LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&v10data,LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float)));
	
	HANDLE_ERROR(cudaMallocHost((void**)&pressureData,LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&precipData,LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float)));
*/
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

	int i,j;
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
//-----------------------------------Getting Wrapped Normal Angles--------------------------------------------------------------
	int DeviceCount;
	/** Getting the total number of devices available **/
	HANDLE_ERROR(cudaGetDeviceCount(&DeviceCount));
	HANDLE_ERROR(cudaSetDevice(DeviceCount - 1));
	HANDLE_ERROR(cudaDeviceReset());

	curandState_t* states;
	
	cudaMalloc((void**)&states,LAT_SIZE*LONG_SIZE*sizeof(curandState_t));

	dim3 gridSize(1,LAT_SIZE,1);
	dim3 blockSize(512,1,1);

	setup_kernel<<<gridSize,blockSize>>>(time(0),states);

	float cpu_nums[LAT_SIZE * LONG_SIZE];
	float *rand_norm_nums,*d_dirData,*d_u_dirAngle,*d_v_dirAngle;

	cudaMalloc((void**)&rand_norm_nums,LAT_SIZE*LONG_SIZE*sizeof(float));
	cudaMalloc((void**)&d_dirData,LAT_SIZE*LONG_SIZE*sizeof(float));
	cudaMalloc((void**)&d_u_dirAngle,LAT_SIZE*LONG_SIZE*sizeof(float));
	cudaMalloc((void**)&d_v_dirAngle,LAT_SIZE*LONG_SIZE*sizeof(float));


	cudaMemcpy(d_dirData,dirData, LAT_SIZE * LONG_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	generate_kernel<<<gridSize,blockSize>>>(states,rand_norm_nums,d_dirData,d_u_dirAngle,d_v_dirAngle,(float)DESIRED_SPEED);

//Do not need to get them back at all; Will have to send it back to GPU 
	cudaMemcpy(cpu_nums,rand_norm_nums, LAT_SIZE * LONG_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(dir_u,d_u_dirAngle,LAT_SIZE * LONG_SIZE * sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(dir_v,d_v_dirAngle,LAT_SIZE * LONG_SIZE * sizeof(float),cudaMemcpyDeviceToHost);

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


	/* free the memory we allocated for the states and numbers */
	HANDLE_ERROR(cudaFree(states));
	HANDLE_ERROR(cudaFree(rand_norm_nums));
	HANDLE_ERROR(cudaFree(d_dirData));
	
//--------------------------------------------------------------------------------------------------//	
	
	size_t MemoryEachVar,DataPerTransfer,SizePerTimestep;
	int TimestepsPerTransfer;		
	size_t MemoryRemaining,TotalMemory;




	/** Getting the total remaining memory that the device can allocate **/
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	MemoryRemaining -= 100 * 1000000;

	printf("Total mem: %zd,Free mem: %zd\n",TotalMemory,MemoryRemaining);

 	printf("\n\n\t\t Total Memory remaining is: %zd \n",MemoryRemaining);

	MemoryEachVar = MemoryRemaining/NUM_DATA_FILES;

	printf("\t\t Memory for each variable is: %zd \n",MemoryEachVar);

	/** Need to send data per timestep so has to be a multiple of LAT_SIZE *LONG_SIZE* sizeof(float)**/
	SizePerTimestep = LAT_SIZE * LONG_SIZE * sizeof(float);
	DataPerTransfer = (MemoryEachVar/SizePerTimestep) * SizePerTimestep;
	TimestepsPerTransfer = DataPerTransfer/SizePerTimestep;

	printf("\t\tChecking Division: %zd\n",MemoryEachVar/SizePerTimestep);		
	printf("\t\t Total Timesteps per Transfer of data is: %ld \n",TimestepsPerTransfer); 
	printf("\t\tData per transfer is %zd\n",DataPerTransfer);	
	
//-----------------------------------------------Allocating memory for  Streams_struct structure----------//
	int divisible,lastDataTransfer;
	long int DataLastTransfer;//Per variable

	lastDataTransfer = (TOTAL_DAYS * TIMESTEPS_PER_DAY) / TimestepsPerTransfer;

	divisible = (TOTAL_DAYS*TIMESTEPS_PER_DAY) % TimestepsPerTransfer;
	
	if(divisible != 0){
			lastDataTransfer++;
	}
	
	printf("Total Transfers required: %ld\n",lastDataTransfer);
	/** Tota bytes transfered per data transfer**/

	Stream_struct Stream_values[lastDataTransfer];
	DataLastTransfer = TOTAL_DAYS * TIMESTEPS_PER_DAY * LAT_SIZE * LONG_SIZE * sizeof(float) - DataPerTransfer * (lastDataTransfer-1); 


	for(i=0;i<lastDataTransfer-1;i++){

		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(Before any allocation): %zd\n",TotalMemory,MemoryRemaining);

		//HANDLE_ERROR(cudaSetDevice(DeviceCount - 1));
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After SetDevice): %zd\n",TotalMemory,MemoryRemaining);

		HANDLE_ERROR(cudaStreamCreate(&Stream_values[i].stream));
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After Stream Create): %zd\n",TotalMemory,MemoryRemaining);

		HANDLE_ERROR(cudaMalloc((void**)&Stream_values[i].d_udata,DataPerTransfer));	
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After udata allocation): %zd\n",TotalMemory,MemoryRemaining);
	
		HANDLE_ERROR(cudaMalloc((void**)&Stream_values[i].d_vdata,DataPerTransfer));	
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After vdata allocation): %zd\n",TotalMemory,MemoryRemaining);

		HANDLE_ERROR(cudaMalloc((void**)&Stream_values[i].d_u10data,DataPerTransfer));	
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After u10data allocation): %zd\n",TotalMemory,MemoryRemaining);

		HANDLE_ERROR(cudaMalloc((void**)&Stream_values[i].d_v10data,DataPerTransfer));	
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After v10data allocation): %zd\n",TotalMemory,MemoryRemaining);

		HANDLE_ERROR(cudaMalloc((void**)&Stream_values[i].d_precipData,DataPerTransfer));	
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After precipData allocation): %zd\n",TotalMemory,MemoryRemaining);

		
//----------> Cannot allocate pressureData
		HANDLE_ERROR(cudaMalloc((void**)&Stream_values[i].d_pressureData,DataPerTransfer));	
		HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
		printf("Total mem: %zd,Free mem(After pressureData allocation): %zd\n",TotalMemory,MemoryRemaining);

		HANDLE_ERROR(cudaDeviceSynchronize());

		HANDLE_ERROR(cudaMallocHost((void**)&Stream_values[i].h_udata,DataPerTransfer));	
		HANDLE_ERROR(cudaMallocHost((void**)&Stream_values[i].h_vdata,DataPerTransfer));	
		HANDLE_ERROR(cudaMallocHost((void**)&Stream_values[i].h_u10data,DataPerTransfer));	
		HANDLE_ERROR(cudaMallocHost((void**)&Stream_values[i].h_v10data,DataPerTransfer));	
		HANDLE_ERROR(cudaMallocHost((void**)&Stream_values[i].h_precipData,DataPerTransfer));	
		HANDLE_ERROR(cudaMallocHost((void**)&Stream_values[i].h_pressureData,DataPerTransfer));	

		printf("After all the host allocations %d\n",i);


		HANDLE_ERROR(cudaStreamDestroy(Stream_values[i].stream));
		HANDLE_ERROR(cudaFree(Stream_values[i].d_udata));
		HANDLE_ERROR(cudaFree(Stream_values[i].d_vdata));
		HANDLE_ERROR(cudaFree(Stream_values[i].d_u10data));
		HANDLE_ERROR(cudaFree(Stream_values[i].d_v10data));
		HANDLE_ERROR(cudaFree(Stream_values[i].d_precipData));
		HANDLE_ERROR(cudaFree(Stream_values[i].d_pressureData));
		
		HANDLE_ERROR(cudaFreeHost(Stream_values[i].h_udata));
		HANDLE_ERROR(cudaFreeHost(Stream_values[i].h_vdata));
		HANDLE_ERROR(cudaFreeHost(Stream_values[i].h_u10data));
		HANDLE_ERROR(cudaFreeHost(Stream_values[i].h_v10data));
		HANDLE_ERROR(cudaFreeHost(Stream_values[i].h_precipData));
		HANDLE_ERROR(cudaFreeHost(Stream_values[i].h_pressureData));

		printf("After all freeing %d\n",i);
		
	}

	printf("After allocation of all except last\n");

	HANDLE_ERROR(cudaStreamCreate(&Stream_values[lastDataTransfer - 1].stream));
	printf("After allocation of stream in last struct\n");

	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After SetDevice): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMalloc((void**)&Stream_values[lastDataTransfer - 1].d_udata,DataLastTransfer));
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After SetDevice): %zd\n",TotalMemory,MemoryRemaining);
	
	HANDLE_ERROR(cudaMalloc((void**)&Stream_values[lastDataTransfer - 1].d_vdata,DataLastTransfer));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After SetDevice): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMalloc((void**)&Stream_values[lastDataTransfer - 1].d_u10data,DataLastTransfer));
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After SetDevice): %zd\n",TotalMemory,MemoryRemaining);
	
	HANDLE_ERROR(cudaMalloc((void**)&Stream_values[lastDataTransfer - 1].d_v10data,DataLastTransfer));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After SetDevice): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMalloc((void**)&Stream_values[lastDataTransfer - 1].d_precipData,DataLastTransfer));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After SetDevice): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMalloc((void**)&Stream_values[lastDataTransfer - 1].d_pressureData,DataPerTransfer));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After SetDevice): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMallocHost((void**)&Stream_values[lastDataTransfer - 1].h_udata,DataLastTransfer));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After MallocHost): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMallocHost((void**)&Stream_values[lastDataTransfer - 1].h_vdata,DataLastTransfer));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After MallocHost): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMallocHost((void**)&Stream_values[lastDataTransfer - 1].h_u10data,DataLastTransfer));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After MallocHost): %zd\n",TotalMemory,MemoryRemaining);

	HANDLE_ERROR(cudaMallocHost((void**)&Stream_values[lastDataTransfer - 1].h_v10data,DataLastTransfer));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After MallocHost): %zd\n",TotalMemory,MemoryRemaining);
	
	HANDLE_ERROR(cudaMallocHost((void**)&Stream_values[lastDataTransfer - 1].h_precipData,DataLastTransfer));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After MallocHost): %zd\n",TotalMemory,MemoryRemaining);
	
	HANDLE_ERROR(cudaMallocHost((void**)&Stream_values[lastDataTransfer - 1].h_pressureData,DataLastTransfer));	
	HANDLE_ERROR(cudaMemGetInfo(&MemoryRemaining,&TotalMemory));
	printf("Total mem: %zd,Free mem(After MallocHost): %zd\n",TotalMemory,MemoryRemaining);



//-----------------------------------------------Initializing Stream_values struct-------------------//
	int ptrOffset;
	ptrOffset = 0;	
	for(i=0;i<lastDataTransfer;i++){
		Stream_values[i].h_udata = udata + ptrOffset;	
		Stream_values[i].h_udata = vdata + ptrOffset;
		Stream_values[i].h_udata = u10data + ptrOffset;
		Stream_values[i].h_udata = v10data + ptrOffset;
		Stream_values[i].h_udata = precipData + ptrOffset;
		Stream_values[i].h_udata = pressureData + ptrOffset;
		
		ptrOffset+= DataPerTransfer/sizeof(float); 
	}


//-----------------------------------------------Freeing allocated memory----------------------------//
	


	HANDLE_ERROR(cudaFree(d_u_dirAngle));
	HANDLE_ERROR(cudaFree(d_v_dirAngle));

	free(udata);
	free(vdata);
	free(dir_u);
	free(dir_v);

	free(u10data);
	free(v10data);
	free(dirData);
	free(precipData);
	free(pressureData);
	free(lwData);

	printf("Before freeing everything\n");
/*	for(i=0;i<lastDataTransfer;i++){
		HANDLE_ERROR(cudaSetDevice(DeviceCount - 1));

		HANDLE_ERROR(cudaStreamDestroy(Stream_values[i].stream));
		HANDLE_ERROR(cudaFree(Stream_values[i].d_udata));
		HANDLE_ERROR(cudaFree(Stream_values[i].d_vdata));
		HANDLE_ERROR(cudaFree(Stream_values[i].d_u10data));
		HANDLE_ERROR(cudaFree(Stream_values[i].d_v10data));
		HANDLE_ERROR(cudaFree(Stream_values[i].d_precipData));
		HANDLE_ERROR(cudaFree(Stream_values[i].d_pressureData));
		
		HANDLE_ERROR(cudaFreeHost(Stream_values[i].h_udata));
		HANDLE_ERROR(cudaFreeHost(Stream_values[i].h_vdata));
		HANDLE_ERROR(cudaFreeHost(Stream_values[i].h_u10data));
		HANDLE_ERROR(cudaFreeHost(Stream_values[i].h_v10data));
		HANDLE_ERROR(cudaFreeHost(Stream_values[i].h_precipData));
		HANDLE_ERROR(cudaFreeHost(Stream_values[i].h_pressureData));

	}

*/
	fclose(dirTxt);
	fclose(posdataTxt);	
	fclose(udataTxt);
	fclose(vdataTxt);
	fclose(v10dataTxt);
	fclose(u10dataTxt);
	fclose(precipTxt);
	fclose(pressureTxt);
	fclose(lwTxt);
	
	printf("End\n");
	return 0;
}




