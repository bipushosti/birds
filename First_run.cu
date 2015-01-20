
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define PI 3.14159

#define LONG_SIZE		95
#define LAT_SIZE		55
#define LINESIZE	8*LAT_SIZE+LAT_SIZE - 3

#define TIMESTEPS	720
#define SKIP_TIMESTEPS	23
//#define DESIRED_ROW	
//#define DESIRED_COL
#define STARTING_ROW	20
#define STARTING_COL	20

#define DESIREDANGLE	90	//In degrees
#define DESIRED_SPEED	10	//In m/s
#define MIN_PROFIT	-7


//Altitude = 850 millibars
//Year = 1980

//--------------------------------------------------------------------------------------------------------------------------

void get_movementData(FILE* outTxt,float* udata,float* vdata,float* dirData);
float getAngleValue(float* udata,float* vdata,float* dirData,long index,long pos);

//--------------------------------------------------------------------------------------------------------------------------

//Kernel to get angle and magnitude from u and v matrices


/*
//Computation Kernel
__global__ void bird_thread(float* resultantMatrix,float* resultantAngle,int* coords_row,int* coords_col)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;	

	int pos_row,pos_col;
	pos_row = 90;
	pos_col = 25;
	
	float tempAngle = 0;
	int i;

	for(i = 0;i<TIMESTEPS;i++) {
		tempAngle = resultantAngle[pos_row * LONG_SIZE + pos_col];
		if((tempAngle >= 0) && (tempAngle < 45)) {
			pos_row -= 1;
		} 
		else if((tempAngle >= 45) && (tempAngle < 90)) {
			pos_row -= 1;
			pos_col += 1;
		}
		else if((tempAngle >= 90) && (tempAngle < 135)) {
			pos_col += 1;
		}
		else if((tempAngle >= 135) && (tempAngle < 180)) {
			pos_row += 1;
			pos_col += 1;
		}
		else if((tempAngle >= 180) && (tempAngle < 225)) {
			pos_row += 1;
		}
		else if((tempAngle >= 225) && (tempAngle < 270)) {
			pos_row += 1;
			pos_col -= 1;
		}
		else if((tempAngle >= 270) && (tempAngle < 315)) {
			pos_col -= 1;
		}
		else if((tempAngle >= 315) && (tempAngle < 360)) {
			pos_row -= 1;
			pos_col -= 1;
		}
		
		coords_row[TIMESTEPS] = pos_row;
		coords_col[TIMESTEPS] = pos_col;
		printf("%d,%d\n",pos_row,pos_col);		
	
	}
}
*/


//--------------------------------------------------------------------------------------------------------------------------
int main()
{

	size_t limit;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 500 * 1024 * 1024);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
	cudaDeviceGetLimit(&limit,cudaLimitPrintfFifoSize);


	//The wind data is in m/s
	float* udata;
	udata = (float*)malloc(LAT_SIZE  * LONG_SIZE * TIMESTEPS * sizeof(float));
	float* vdata;
	vdata = (float*)malloc(LAT_SIZE  * LONG_SIZE * TIMESTEPS * sizeof(float));
	float* dirData;
	dirData = (float*)malloc(LAT_SIZE  * LONG_SIZE * sizeof(float));
	

	FILE *posdataTxt;
	posdataTxt = fopen("posdata.txt","w");
	if(posdataTxt == NULL) {
		perror("Cannot open udataTxt file\n");
		return -1;
	}


	FILE *vdataTxt,*udataTxt;
	udataTxt = fopen("uvalue.txt","r");
	vdataTxt = fopen("vvalue.txt","r");
	if(udataTxt == NULL) {
		perror("Cannot open udataTxt file\n");
		return -1;
	}
	if(vdataTxt == NULL) {
		perror("Cannot open udataTxt file\n");
		return -1;
	}

	FILE* dirTxt;
	dirTxt = fopen("direction.txt","r");
	if(dirTxt == NULL) {
		perror("Cannot open dirTxt file\n");
		return -1;
	}

	FILE* inpCheckU;
	inpCheckU = fopen("inpCheckU.txt","w");
	if(inpCheckU == NULL) {
		perror("Cannot open udataTxt file\n");
		return -1;
	}
	
	FILE* inpCheckV;
	inpCheckV = fopen("inpCheckV.txt","w");
	if(inpCheckV == NULL) {
		perror("Cannot open udataTxt file\n");
		return -1;
	}
	
	char line[LINESIZE];
	memset(line,'\0',sizeof(line));

	char tempVal[8];
	memset(tempVal,'\0',sizeof(tempVal));

	char* startPtr,*endPtr;

	long j;
	int i;
	float Value;
	
	i=0;
	j=0;
	
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
/*
	//for(j=0;j<LONG_SIZE * TIMESTEPS;j++) {
	for(j=0;j<LONG_SIZE;j++) {
		for(i=0;i<LAT_SIZE;i++) {
			if(i == LAT_SIZE -1) {
				//fprintf(inpCheckU,"%f\n",udata[j * LAT_SIZE + i]);
				//fprintf(inpCheckV,"%f\n",vdata[j * LAT_SIZE + i]);
				printf("%f\n",dirData[j * LAT_SIZE + i]);
			}
			else {
				
				//fprintf(inpCheckU,"%f ",udata[j * LAT_SIZE + i]);
				//fprintf(inpCheckV,"%f ",vdata[j * LAT_SIZE + i]);
			}
		}
	}
*/
	get_movementData(posdataTxt,udata,vdata,dirData);
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
	
	printf("U	v	Magnitude	Angle\n");
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
	free(dirData);

	fclose(udataTxt);
	fclose(vdataTxt);
	fclose(inpCheckU);
	fclose(inpCheckV);
	fclose(posdataTxt);
//------------------------------------------------------------------------------------------------------------------------------------
	return 0;

}


float getAngleValue(float* udata,float* vdata,float* dirData,long index,long pos)
{


		float u,v;
		
		//All wind data in m/s
		v = vdata[index];
		u = udata[index];
		
		
		float angle,diffAngle,magnitude,magnitude_squared,wind_profit;

		magnitude_squared = u * u + v * v;
		magnitude = (float)sqrt(magnitude_squared);

		if((v >0)&&( u >0)) {
			angle = tanf(u/v) * (180/PI);
			diffAngle = dirData[pos] - angle;
		}
		else if ((v > 0)&&( u < 0)){
			angle = 360 - (tanf(u/v) * (180/PI));
			diffAngle = dirData[pos] + (tanf(u/v) * (180/PI));
		}
		else if ((v < 0)&&( u > 0)){
			angle = 180 - (tanf(u/v) * (180/PI));
			diffAngle = dirData[pos]- (tanf(u/v) * (180/PI));
		}
		else if ((v < 0)&&( u < 0)){
			angle = 180 + (tanf(u/v) * (180/PI));
			diffAngle = dirData[pos] + (tanf(u/v) * (180/PI)); 
		}
		else if ((v == 0)&&( u > 0)){
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

		if(angle < 0) angle = (360 - angle);
		if(angle > 360) angle = angle - 360;

		if(diffAngle < 0) diffAngle = 360 - angle;
		if(diffAngle > 360) diffAngle = diffAngle - 360;


		wind_profit = (DESIRED_SPEED * DESIRED_SPEED) + magnitude_squared - 2 * DESIRED_SPEED * magnitude * cos(diffAngle * PI/180);
		wind_profit = DESIRED_SPEED - (float)sqrt(wind_profit);
		//printf("%f\n",wind_profit);
		return wind_profit;
}


void get_movementData(FILE* outTxt,float* udata,float* vdata,float* dirData)
{

	
	int pos_row,pos_col;
	//pos_row = LONG_SIZE - STARTING_ROW;
	pos_row = STARTING_ROW;
	pos_col = STARTING_COL;

	fprintf(outTxt,"%d,%d\n",pos_row,pos_col);

	int k;
	long i,l;
	l = 0;

	float profit_value,dirAngle;
	float dir_v,dir_u;
	
	long skip_size = (SKIP_TIMESTEPS * LONG_SIZE  * LAT_SIZE) - 1;
	//skip_size = 120174


	i = skip_size +pos_row * LAT_SIZE + pos_col;


	while( i <= (TIMESTEPS-1) * LAT_SIZE * LONG_SIZE ) {
		dir_v = 0;
		dir_u = 0;
		dirAngle = 0;

		for(k=0;k<6;k++,l++ ) {
			i = skip_size + l * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col;	
			profit_value = getAngleValue(udata,vdata,dirData,i,pos_row * LAT_SIZE + pos_col);

			if (profit_value >= MIN_PROFIT ) { 
				//printf("Found value %f\n",profit_value);
				
				dirAngle = dirData[pos_row * LAT_SIZE + pos_col];
				

				//The grid is upside down; v increases from top to bottom while
				//u increases from left to right
				if(dirAngle <= 90){
					dirAngle = 90 - dirAngle;
					dir_v = DESIRED_SPEED * sin(dirAngle * (PI/180)) * -1;
					dir_u = DESIRED_SPEED * cos(dirAngle * (PI/180));
				}
				else if((dirAngle > 90) && (dirAngle <= 180)){ 
					dirAngle -= 90;
					dir_v = DESIRED_SPEED * sin(dirAngle * (PI/180));
					dir_u = DESIRED_SPEED * cos(dirAngle * (PI/180));
				}
				else if((dirAngle > 180) && (dirAngle <= 270)) {
			 		dirAngle = 270 - dirAngle;
					dir_v = DESIRED_SPEED * sin(dirAngle * (PI/180));
					dir_u = DESIRED_SPEED * cos(dirAngle * (PI/180)) * -1;
				}
				else if((dirAngle > 270) && (dirAngle <= 360)){
					dirAngle -= 270;
					dir_v = DESIRED_SPEED * sin(dirAngle * (PI/180)) * -1;
					dir_u = DESIRED_SPEED * cos(dirAngle * (PI/180)) * -1;
				}
				
				pos_row = (int)(rintf(pos_row + vdata[i] + dir_v));
				pos_col = (int)(rintf(pos_col + udata[i] + dir_u));	
				//printf("%d,%d\n",pos_row,pos_col);
				printf("\nDesired Angle::%f,dir_v::%f,dir_u::%f\n",dirAngle,dir_v,dir_u);

				//printf("%ld\n",i);
				printf("%ld \t %ld \t %d \t %f \t %d \t %d \t (%f,%f)\n",i,l,k,profit_value,pos_row,pos_col,vdata[i],udata[i]);
				//fprintf(outTxt,"%d,%d\n",pos_row,pos_col); 
			}
			else { 
				//i = i - k + 6;				
				l = l - k + 6;
				//printf("Skipped Wind (%f,%f) @ (%d,%d)\n",udata[i],vdata[i],pos_row,pos_col);
				break; 
			}
		}
		l -= 6;

		if (l == 0) l += 23;
		else l += 24;

		i = skip_size + l * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col;

	}
	
}





