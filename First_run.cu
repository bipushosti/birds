
#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>

#define PI 3.14159

#define XSIZE		55
#define YSIZE		95
#define LINESIZE	8*55+53

#define TIMESTEPS	20
//#define DESIRED_ROW	
//#define DESIRED_COL
#define STARTING_ROW	90
#define STARTING_COL	25
//What is the time limit? How long will the birds keep flying/migrating before they 
//just give up?

//Assuming min tail wind speed = 1km/hr
//Assuming best tail wind speed = 40km/hr
//Assuming Max tail wind speed = 80km/hr
//Assuming Max head wind speed = 30km/hr


//----------------------------------------------------------------------------------------------
struct Data
{
	int *temp_row;
	int *temp_col;	
};


__global__ void get_resultant(float * u, float* v,float* resultantMatrix,float* resultantAngle);
struct Data invoke_Computekernel(float* resultantMatrix,float* resultantAngle);

//----------------------------------------------------------------------------------------------

//Kernel to get angle and magnitude from u and v matrices
__global__ void get_resultant(float * u, float* v,float* resultantMatrix,float* resultantAngle)
{
	float magnitude,angle;

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	magnitude = hypotf(u[y * XSIZE + x],v[y * XSIZE + x]);
	angle = asin(u[y * XSIZE + x]/v[y * XSIZE + x]) * 180/PI;
	resultantMatrix[y * XSIZE + x] = magnitude;
	resultantAngle[y * XSIZE  + x] = angle;
}

//Computation Kernel
__global__ void bird_thread(float* resultantMatrix,float* resultantAngle)
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
		
	}

	
	

}



//----------------------------------------------------------------------------------------------
int main()
{
	
	float udata[YSIZE * XSIZE];
	float vdata[YSIZE * XSIZE];

	FILE *udataTxt;
	FILE *vdataTxt;
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

	char line[LINESIZE];
	memset(line,'\0',sizeof(line));

	char tempVal[8];
	memset(tempVal,'\0',sizeof(tempVal));

	char* startPtr,*endPtr;

	int i,j;
	float Value;
	
	i=0;
	j=0;
	
	while(fgets(line,LINESIZE,udataTxt)!=NULL){
		startPtr = line;
		for(i=0;i<XSIZE;i++){

			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));

			if(i != (XSIZE - 1)) {

				endPtr = strchr(startPtr,' ');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal);
				udata[j * XSIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (XSIZE - 1)){

				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
				udata[j * XSIZE + i] = Value;
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
		for(i=0;i<XSIZE;i++){

			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));

			if(i != (XSIZE - 1)) {

				endPtr = strchr(startPtr,' ');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal);
				vdata[j * XSIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (XSIZE - 1)){

				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
				vdata[j * XSIZE + i] = Value;
			}
		}
		j++;
	}	

	
	float resultantMatrix[XSIZE * YSIZE];
	float resultantAngle[XSIZE * YSIZE];
	

	float* udataPtr,*vdataPtr,*resultantMatrixPtr,*resultantAnglePtr;

	cudaMalloc((void**)&udataPtr,XSIZE * YSIZE * sizeof(float));
	cudaMalloc((void**)&vdataPtr,XSIZE * YSIZE * sizeof(float));
	cudaMalloc((void**)&resultantMatrixPtr,XSIZE * YSIZE * sizeof(float));
	cudaMalloc((void**)&resultantAnglePtr,XSIZE * YSIZE * sizeof(float));
	
	

	cudaMemcpy(udataPtr,udata,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(vdataPtr,udata,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);

	dim3 gridSize(1,YSIZE,0);
	dim3 blockSize(XSIZE,1,1);

	get_resultant<<<gridSize,blockSize>>>(udataPtr,vdataPtr,resultantMatrixPtr,resultantAnglePtr);

	cudaMemcpy(resultantMatrix,resultantMatrixPtr,YSIZE * XSIZE * sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(resultantAngle,resultantAnglePtr,YSIZE * XSIZE * sizeof(float),cudaMemcpyDeviceToHost);

	cudaFree(udataPtr);
	cudaFree(vdataPtr);
	cudaFree(resultantMatrixPtr);
	cudaFree(resultantAnglePtr);
	
	//struct Data data_final;

	//data_final = invoke_Computekernel(resultantMatrix,resultantAngle);
	//return 0;
//}

//Invoking computation kernel
//struct Data invoke_Computekernel(float* resultantMatrix,float* resultantAngle)
//{

	int pos_row[TIMESTEPS];
	int pos_col[TIMESTEPS];
//	float *pos_rowPtr,*pos_colPtr,*resultantMatrixPtr,*resultantAnglePtr;
	float *pos_rowPtr,*pos_colPtr;

	cudaMalloc((void**)&pos_rowPtr,TIMESTEPS * sizeof(int));
	cudaMalloc((void**)&pos_colPtr,TIMESTEPS * sizeof(int));
	cudaMalloc((void**)&resultantMatrixPtr,XSIZE * YSIZE * sizeof(float));
	cudaMalloc((void**)&resultantAnglePtr,XSIZE * YSIZE * sizeof(float));

	cudaMemcpy(resultantMatrixPtr,resultantMatrix,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(resultantAnglePtr,resultantAngle,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);

	dim3 gridSize(1,0,0);
	dim3 blockSize(1,0,0);

	bird_thread<<<gridSize,blockSize>>>(resultantMatrix,resultantAngle);
	
	cudaMemcpy(pos_row,pos_rowPtr,TIMESTEPS * sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(pos_row,pos_colPtr,TIMESTEPS * sizeof(int),cudaMemcpyDeviceToHost);
	
	
	//struct Data dataVals;
	//dataVals.temp_row = pos_row;
	//dataVals.temp_col = pos_col;


	cudaFree(pos_rowPtr);
	cudaFree(pos_colPtr);
	cudaFree(resultantMatrixPtr);
	cudaFree(resultantAnglePtr);

	//return dataVals; 
	return 0;

}





