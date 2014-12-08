
#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>

#define XSIZE		55
#define YSIZE		95
#define LINESIZE	8*55+53

#define DESIRED_X
#define DESIRED_Y
#define STARTING_X
#define STARTING_Y
//What is the time limit? How long will the birds keep flying/migrating before they 
//just give up?

//Assuming min tail wind speed = 1km/hr
//Assuming best tail wind speed = 40km/hr
//Assuming Max tail wind speed = 80km/hr
//Assuming Max head wind speed = 30km/hr

_global_ void get_resultant(float * u, float* v,float* resultantMatrix)
{
	float magnitude,angle;

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	magnitude = hypotf(u[y * XSIZE + x],v[y * XSIZE + x]);
	angle = asin(u[y * XSIZE + x]/v[y * XSIZE + x]);


}


int main()
{
	
	float udata[YSIZE * XSIZE];
	float vdata[YSIZE * XSIZE];

	FILE *udataTxt,*vdataTxt;
	udataTxt = fopen("udata.txt","r");
	if(udataTxt == NULL) {
		perror("Cannot open udataTxt file\n");
		return -1;
	}

	vdataTxt =fopen("udata.txt","r");
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

	float pos_x,pos_y;
	pos_x = 0;
	pos_y = 0;
	
	float resultantMatrix[XSIZE * YSIZE];
	float* udataPtr,*vdataPtr,*resultantMatrixPtr;
	cudaMalloc((void**)&udataPtr,XSIZE * YSIZE * sizeof(float));
	cudaMalloc((void**)&vdataPtr,XSIZE * YSIZE * sizeof(float));
	cudaMalloc((void**)&resultantMatrixPtr,XSIZE * YSIZE * sizeof(float));

	cudaMemcpy(udataPtr,udata,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(vdataPtr,udata,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);

	dim3 gridSize(1,YSIZE,0);
	dim3 blockSize(XSIZE,1,1);

	get_resultant<<<gridSize,blockSize>>>(udataPtr,vdataPtr,resultantMatrixPtr);

	cudaMemcpy(resultantMatrix,resultantMatrixPtr,YSIZE * XSIZE * sizeof(float),cudaMemcpyDeviceToHost);

	cudaFree(udataPtr);
	cudaFree(vdataPtr);
	cudaFree(resultantMatrixPtr);



}
