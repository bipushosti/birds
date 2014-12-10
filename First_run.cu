
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define PI 3.14159

#define XSIZE		95
#define YSIZE		55
#define LINESIZE	6*95+93

#define TIMESTEPS	50
//#define DESIRED_ROW	
//#define DESIRED_COL
#define STARTING_ROW	30
#define STARTING_COL	75
//What is the time limit? How long will the birds keep flying/migrating before they 
//just give up?

//Assuming min tail wind speed = 1km/hr
//Assuming best tail wind speed = 40km/hr
//Assuming Max tail wind speed = 80km/hr
//Assuming Max head wind speed = 30km/hr


//--------------------------------------------------------------------------------------------------------------------------

__global__ void get_resultant(float * u, float* v,float* resultantMatrix,float* resultantAngle);
float* get_inputData(FILE* dataTxt);
void get_movementData(FILE* outTxt,float* resultantMatrix,float* resultantAngle);

//--------------------------------------------------------------------------------------------------------------------------

//Kernel to get angle and magnitude from u and v matrices
__global__ void get_resultant(float * u, float* v,float* resultantMatrix,float* resultantAngle)
{
	float angle;

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + y * XSIZE; 

	if(x < 95) {
		resultantMatrix[index] = hypotf(u[index],v[index]);
		angle = atanf(u[index]/v[index]) * (180/PI);
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
		tempAngle = resultantAngle[pos_row * XSIZE + pos_col];
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
	//udata = (float*)malloc(YSIZE  * XSIZE * sizeof(float));
	float* vdata;
	//vdata = (float*)malloc(YSIZE  * XSIZE * sizeof(float));
	

	FILE *udataTxt;
	FILE *vdataTxt;
	FILE *posdataTxt;

	udataTxt = fopen("udata.txt","r");
	if(udataTxt == NULL) {
		perror("Cannot open udataTxt file\n");
		return -1;
	}

	vdataTxt =fopen("vdata.txt","r");
	if(vdataTxt == NULL) {
		perror("Cannot open vdataTxt file\n");
		return -1;
	}

	posdataTxt =fopen("posdata.txt","w");
	if(posdataTxt == NULL) {
		perror("Cannot open vdataTxt file\n");
		return -1;
	}


	udata =  get_inputData(udataTxt);
	vdata =  get_inputData(vdataTxt);
	
	int j;
	
	//for(j=0;j<YSIZE*XSIZE;j++) {
	//	printf("%f,%f\n",udata[j],vdata[j]);
	//}


	float resultantMatrix[XSIZE * YSIZE];
	float resultantAngle[XSIZE * YSIZE];
	
	resultantMatrix[YSIZE * XSIZE -1] = 834.0;	

	float *udataPtr,*vdataPtr,*resultantMatrixPtr,*resultantAnglePtr;

	cudaMalloc((void**)&udataPtr,XSIZE * YSIZE * sizeof(float));
	cudaMemcpy(udataPtr,udata,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);

	cudaMalloc((void**)&vdataPtr,XSIZE * YSIZE * sizeof(float));
	cudaMemcpy(vdataPtr,vdata,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);

	cudaMalloc((void**)&resultantMatrixPtr,XSIZE * YSIZE * sizeof(float));
	cudaMalloc((void**)&resultantAnglePtr,XSIZE * YSIZE * sizeof(float));
	
	cudaMemcpy(vdataPtr,vdata,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);

	dim3 gridSize(1,YSIZE,1);
	dim3 blockSize((XSIZE/32 +1)*32 ,1,1);
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

	cudaMemcpy(resultantMatrix,resultantMatrixPtr,YSIZE * XSIZE * sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(resultantAngle,resultantAnglePtr,YSIZE * XSIZE * sizeof(float),cudaMemcpyDeviceToHost);

	cudaFree(udataPtr);
	cudaFree(vdataPtr);
	cudaFree(resultantMatrixPtr);
	cudaFree(resultantAnglePtr);
	

	//for(i=0;i< XSIZE * YSIZE;i++) {
	//	printf("%f\n",resultantAngle[i]);
	//}



	get_movementData(posdataTxt,resultantMatrix,resultantAngle);
//--------------------------------------------------------------------------------------------------------------------------
	/*int coords_row[TIMESTEPS];
	int coords_col[TIMESTEPS];
	int *coords_rowPtr,*coords_colPtr;

	cudaMalloc((void**)&coords_rowPtr,TIMESTEPS * sizeof(int));
	cudaMalloc((void**)&coords_colPtr,TIMESTEPS * sizeof(int));
	cudaMalloc((void**)&resultantMatrixPtr,XSIZE * YSIZE * sizeof(float));
	cudaMalloc((void**)&resultantAnglePtr,XSIZE * YSIZE * sizeof(float));

	cudaMemcpy(resultantMatrixPtr,resultantMatrix,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(resultantAnglePtr,resultantAngle,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);

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

	//free(udata);
	//free(vdata);
	return 0;

}

float* get_inputData(FILE* dataTxt)
{


	static float data[YSIZE * XSIZE];

	char line[LINESIZE];
	memset(line,'\0',sizeof(line));

	char tempVal[8];
	memset(tempVal,'\0',sizeof(tempVal));

	char* startPtr,*endPtr;

	int i,j;
	float Value;
	
	i=0;
	j=0;
	
	while(fgets(line,LINESIZE,dataTxt)!=NULL){
		startPtr = line;
		for(i=0;i<XSIZE;i++){

			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));

			if(i != (XSIZE - 1)) {

				endPtr = strchr(startPtr,' ');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal);
				data[j * XSIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (XSIZE - 1)){

				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
				data[j * XSIZE + i] = Value;
			}			
		}
		j++;
	}	
	return data;
}

void get_movementData(FILE* outTxt,float* resultantMatrix,float* resultantAngle)
{

	int pos_row,pos_col;
	pos_row = STARTING_ROW;
	pos_col = STARTING_COL;
	
	float tempAngle = 0;
	int i;

	//int coords_row[TIMESTEPS];
	//int coords_col[TIMESTEPS];

	for(i = 0;i<TIMESTEPS;i++) {
		
		tempAngle = resultantAngle[pos_row * XSIZE + pos_col];
		printf("%d,%d,%f\n",pos_row,pos_col,tempAngle);
		if((tempAngle >= 0) && (tempAngle < 45)) {
			pos_row += 1;
		} 
		else if((tempAngle >= 45) && (tempAngle < 90)) {
			pos_row += 1;
			pos_col += 1;
		}
		else if((tempAngle >= 90) && (tempAngle < 135)) {
			pos_col += 1;
		}
		else if((tempAngle >= 135) && (tempAngle < 180)) {
			pos_row -= 1;
			pos_col += 1;
		}
		else if((tempAngle >= 180) && (tempAngle < 225)) {
			pos_row -= 1;
		}
		else if((tempAngle >= 225) && (tempAngle < 270)) {
			pos_row -= 1;
			pos_col -= 1;
		}
		else if((tempAngle >= 270) && (tempAngle < 315)) {
			pos_col -= 1;
		}
		else if((tempAngle >= 315) && (tempAngle < 360)) {
			pos_row += 1;
			pos_col -= 1;
		}
		
		if(pos_row >= YSIZE) pos_row = YSIZE - 1;
		else if(pos_row <= 0) pos_row = 1;
		if(pos_col >= XSIZE) pos_col = XSIZE - 1;
		else if(pos_col <= 0) pos_col = 1;
			 
		//coords_row[TIMESTEPS] = pos_row;
		//coords_col[TIMESTEPS] = pos_col;
		fprintf(outTxt,"%d,%d\n",55 - pos_row,pos_col);		
	
	}



}




