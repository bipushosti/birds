
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
#include <stdio.h>
#include <math.h>
#include <pthread.h>
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
__global__ void bird_movement(float pos_row,float pos_col,long l,float* udata,float* vdata,float* u10data,float* v10data,float* dirData,float* precipData,float* pressureData,float* lwData);

void read_dataFiles(FILE* textFile,float* dataArray);
long convert_to_month(char* month,char * day);
//-------------------------------------------------------------------------------------------------------------------------------------
struct file_IO {
	FILE *fp;
	float* inpVals;
}inpStruct[8]; 
//-------------------------------------------------------------------------------------------------------------------------------------
//Global Variables

float* dirData;
float* udata;
float* vdata;

float* dir_u;
float* dir_v;

float* u10data;
float* v10data;

float* precipData;
float* pressureData;
float* lwData;

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

//------------------------------------------------------------------------------------------------------------------------------------


__global__ void bird_movement(float pos_row,float pos_col,long l,float* udata,float* vdata,float* u10data,float* v10data,float* dirData,float* precipData,float* pressureData,float* lwData)
{

	//Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int id = y * LONG_SIZE + x;

	long int i;

//	while( i <= (TIMESTEPS-1) * LAT_SIZE * LONG_SIZE ) {

	
	
}

//------------------------------------------------------------------------------------------------------------------------------------
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
//------------------------------------------------------------------------------------------------------------------------------------
__device__ float bilinear_interpolation_SmallData(float x,float y,float* data_array)
{
	float x1,y1,x2,y2;
	float value,Q11,Q12,Q21,Q22,R1,R2,R;
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
//------------------------------------------------------------------------------------------------------------------------------------
__device__ float bilinear_interpolation_LargeData(float x,float y,float* data_array,long l)
{
	float x1,y1,x2,y2;
	float value,Q11,Q12,Q21,Q22,R1,R2,R;
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
//------------------------------------------------------------------------------------------------------------------------------------
__device__ float getProfitValue(float u_val,float v_val,float dirVal,float dir_u,float dir_v)
{

	/** 
	All wind data in m/s 
	**/
	float diffAngle,magnitude,magnitude_squared,tailComponent,crossComponent,profit_value;

	tailComponent = 0;
	
	magnitude = hypotf(u_val,v_val);
	magnitude_squared = magnitude * magnitude;

	/** 
	Getting the tail component of the wind; or the component of the wind in the desired direction of flight
	From formula of getting the vector projection of wind onto the desired direction 
	**/

	tailComponent = (dir_v * v_val + dir_u * u_val);
	tailComponent = tailComponent/hypotf(dir_u,dir_u);
	

	/** 
	DiffAngle is the angle between the desired direction of the bird and the direction of the wind
	DiffAngle has to be calculated such that both the vectors are pointing away from where they meet.
	Using the formula to get angle between two vectors
	**/

	diffAngle = acosf( (u_val*dir_u + v_val * dir_v)/ (( hypotf(u_val,v_val) * hypotf(dir_u,dir_v) )) ) * 180/PI;

	/** 
	Separate profit value methods have to be used if the tail component is less that equal to or greater than the desired speed of the birds 
	**/
	if(tailComponent <= DESIRED_SPEED) {	
		profit_value = (DESIRED_SPEED * DESIRED_SPEED) + magnitude_squared - 2 * DESIRED_SPEED * magnitude * cosf(diffAngle * PI/180);
		profit_value = DESIRED_SPEED - sqrtf(profit_value);
	}
	else {
		/** Perpendicular to a vector (x,y) is (y,-x) or (-y,x)
		Cross component is always positive **/

		crossComponent = fabsf((-dir_v*u_val + dir_u*v_val)/hypotf(dir_v,dir_u));
		profit_value = tailComponent - crossComponent;
	}

	return profit_value;
}
//-------------------------------------------------------------------------------------------------------------------------------------
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
//-------------------------------------------------------------------------------------------------------------------------------------
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




//-------------------------------------------------------------------------------------------------------------------------------------
int main(int argc,char* argv[])
{

//--------------------------Checking for input arguments------------------------------//

	char baseFileName[] = "../../Birds_Full/Birds_data/InterpolatedData/";
	char yearFileName[80];
	char fullFileName[80];
	char start_date[12];

	float starting_row,starting_col;
	long offset_into_data = 0;

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

	/** If starting row is greater than or equal the row that we are interested in; Below a particular row we are not interested in the flight of the birds**/
	if(starting_row >= MAX_LAT_SOUTH){
		printf("\t\tProvided starting row is below the southern most lattitude at which the model is set to stop\n");
		printf("\t\tEither change the starting row location and/or MAX_LAT upto which the birds can fly\n");
		return -1;
	}

	/** Converting month and day information into number of timesteps; Special case of AUG 1st is also taken care of**/
	if((strcmp(argv[2],"AUG")==0) && (strcmp(argv[3],"1")==0)){
			offset_into_data = 22;
	}
	else{
			offset_into_data = convert_to_month(argv[2],argv[3]);
	}

	/** Checking if correct year specified **/
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

	udata = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	vdata = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	u10data = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	v10data = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));

	precipData = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	pressureData = (float*)malloc(LAT_SIZE * LONG_SIZE * TIMESTEPS * sizeof(float));
	lwData = (float*)malloc(LAT_SIZE * LONG_SIZE * sizeof(float));




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
//-------------------------If August 1,then it starts at August 2 (because data starts at 7pm but birds fly at 6pm)------------
//	if(strcmp(

//-----------------------------------Getting Wrapped Normal Angles--------------------------------------------------------------
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

	cudaMemcpy(cpu_nums,rand_norm_nums, LAT_SIZE * LONG_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(dir_u,d_u_dirAngle,LAT_SIZE * LONG_SIZE * sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(dir_v,d_v_dirAngle,LAT_SIZE * LONG_SIZE * sizeof(float),cudaMemcpyDeviceToHost);

	/* print them out */
	for ( j = 0; j < LAT_SIZE; j++) {
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



	/* free the memory we allocated for the states and numbers */
	cudaFree(states);
	cudaFree(rand_norm_nums);
	cudaFree(d_dirData);
	cudaFree(d_u_dirAngle);
	cudaFree(d_v_dirAngle);
//-----------------------------------------------Freeing allocated memory----------------------------//
	



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

	fclose(dirTxt);

	printf("End\n");
	return 0;
}




