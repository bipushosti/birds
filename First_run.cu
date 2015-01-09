
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
#define STARTING_ROW	11
#define STARTING_COL	14
//What is the time limit? How long will the birds keep flying/migrating before they 
//just give up?

//Assuming min tail wind speed = 1km/hr
//Assuming best tail wind speed = 40km/hr
//Assuming Max tail wind speed = 80km/hr
//Assuming Max head wind speed = 30km/hr


//--------------------------------------------------------------------------------------------------------------------------

__global__ void get_resultant(float * u, float* v,float* resultantMatrix,float* resultantAngle);
//void get_movementData(FILE* outTxt,float* resultantMatrix,float* resultantAngle,float* udata,float* vdata);
void get_movementData(FILE* outTxt,float* udata,float* vdata);
float bilinear_interpolation(float x,int x1,int x2,float y,int y1,int y2,float Q11,float Q12,float Q21,float Q22);
//--------------------------------------------------------------------------------------------------------------------------

//Kernel to get angle and magnitude from u and v matrices
__global__ void get_resultant(float * u, float* v,float* resultantMatrix,float* resultantAngle)
{
	float angle;

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + y * LAT_SIZE; 

	if(x < LONG_SIZE) {
		resultantMatrix[index] = hypotf(u[index],v[index]);
		if((v[index] >0)&&( u[index] >0)) {
			angle = atanf(u[index]/v[index]) * (180/PI);
		}
		else if ((v[index] > 0)&&( u[index] < 0)){
			angle = 360 - atanf(u[index]/v[index]) * (180/PI);
		}
		else if ((v[index] < 0)&&( u[index] > 0)){
			angle = 180 - atanf(u[index]/v[index]) * (180/PI);
		}
		else if ((v[index] < 0)&&( u[index] < 0)){
			angle = 180 + atanf(u[index]/v[index]) * (180/PI);
		}
		else if ((v[index] == 0)&&( u[index] > 0)){
			angle = 90;
		}
		else if ((v[index] == 0)&&( u[index] < 0)){
			angle = 270;
		}
		else if ((v[index] > 0)&&( u[index] == 0)){
			angle = 0;
		}
		else if ((v[index] < 0)&&( u[index] == 0)){
			angle = 180;
		}
		if(angle < 0){
			angle = (360 - angle);
		}
		if(angle > 360) {
			angle = angle - 360;
		}
		resultantAngle[index] = angle;
		//printf("%f,%f\n",resultantMatrix[index],angle);
	}	

}

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

	float* udata;
	udata = (float*)malloc(LAT_SIZE  * LONG_SIZE * TIMESTEPS * sizeof(float));
	float* vdata;
	vdata = (float*)malloc(LAT_SIZE  * LONG_SIZE * TIMESTEPS * sizeof(float));
	

	

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

	//udata =  get_inputData(udataTxt);
	//vdata =  get_inputData(vdataTxt);
	
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
		
	for(j=0;j<LONG_SIZE * TIMESTEPS;j++) {
		for(i=0;i<LAT_SIZE;i++) {
			if(i == LAT_SIZE -1) {
				fprintf(inpCheckU,"%f\n",udata[j * LAT_SIZE + i]);
				fprintf(inpCheckV,"%f\n",vdata[j * LAT_SIZE + i]);
			}
			else {
				fprintf(inpCheckU,"%f ",udata[j * LAT_SIZE + i]);
				fprintf(inpCheckV,"%f ",vdata[j * LAT_SIZE + i]);
			}
		}
	}
	get_movementData(posdataTxt,udata,vdata);
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
//------------------------------------------------------------------------------------------------------------------------------------
	return 0;

}


//void get_movementData(FILE* outTxt,float* resultantMatrix,float* resultantAngle,float* udata,float* vdata)
void get_movementData(FILE* outTxt,float* udata,float* vdata)
{

	
	int pos_row,pos_col;
	//pos_row = LONG_SIZE - STARTING_ROW;
	pos_row = STARTING_ROW;
	pos_col = STARTING_COL;

	fprintf(outTxt,"%d,%d\n",pos_row,pos_col);
	//float tempAngle = 0;
	int k;
	long i,j,l;
	j=23;
	l = 0;

	//int coords_row[TIMESTEPS];
	//int coords_col[TIMESTEPS];

	float speedOrMagnitude;
	
	long skip_size = (SKIP_TIMESTEPS) * LONG_SIZE  * LAT_SIZE - 1;
	//for(i = SKIP_TIMESTEPS - 1; i<(TIMESTEPS - 1);i++) {

	i = skip_size +pos_row * LAT_SIZE + pos_col;
	//pos_row = pos_row + (int)udata[i];
	//pos_col = pos_col + (int)vdata[i];	
	//pos_row = floorf(pos_row);
	//pos_col = floorf(pos_col);

	//fprintf(outTxt,"%d,%d\n",pos_row,pos_col);
	//printf("%f,%f,%ld\n",udata[i],vdata[i],j);
	while( i <= (TIMESTEPS-1) * LAT_SIZE * LONG_SIZE ) {
		for(k=0;k<6;k++,i++,j++,l++ ) {

			//speedOrMagnitude = hypotf(udata[SKIP_TIMESTEPS * LONG_SIZE  * LAT_SIZE + LONG_SIZE * j + pos_row * LAT_SIZE + pos_col],
					//	vdata[SKIP_TIMESTEPS * LONG_SIZE  * LAT_SIZE + LONG_SIZE * j + pos_row * LAT_SIZE + pos_col]);
			//pos_row = lroundf(pos_row + udata[i]);
			//pos_col = lroundf(pos_col + vdata[i]);
			
			
			
			pos_row = pos_row + (int)(rintf(vdata[skip_size + l * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col] * 3.6/50));
			pos_col = pos_col + (int)(rintf(udata[skip_size + l * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col] * 3.6/50));	

//			pos_row = floorf(pos_row);
//			pos_col = floorf(pos_col);
			
			
			//printf("%ld\n",i);
			fprintf(outTxt,"%d,%d\n",pos_row,pos_col); 
			printf("%f,%f,%ld\n",udata[skip_size  + l * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col],vdata[skip_size  + l * LAT_SIZE * LONG_SIZE + pos_row * LAT_SIZE + pos_col],j);
			//i = i + 1;
			
		}
		i--;
		i = i-6;
		//i = skip_size + LONG_SIZE * LAT_SIZE * j * 18 +pos_row * LAT_SIZE + pos_col;
		i = i + 24 * LAT_SIZE * LONG_SIZE;
		j--;
		j = j - 6;
		j += 24;
		l--;
		l -= 6;
		l += 24;
	}
	
}

/*
float bilinear_interpolation(float x,int x1,int x2,float y,int y1,int y2,float Q11,float Q12,float Q21,float Q22) 
{
	float value = 0;
	value = [(x2 - x1)[(y2 - y) * Q11 + (y - y1) * Q12] + (x - x1)[(y2 - y) * Q21 + (y - y1) * Q22]]/[(x2 - x1) * (y2 - y1)];
	return value;
}
*/



