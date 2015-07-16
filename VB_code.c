#define PI 3.14159


#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#define	glCompAcc	1e-8


/*
struct Values{
	double meanAngle;
	double meanConcentration;
}MeanValues;	
*/

//void CircMeanConcentration(double* ValArray);
//Attribute VB_Name = "ModCircStats"
//Option Explicit
float WrappedNormal (float MeanAngle,float AngStdDev);

int main()
{
	struct timeval t1;
	gettimeofday(&t1,NULL);
	srand(t1.tv_usec * t1.tv_sec);

	//Standard deviation  = total angular deviation required(+- 30 is 60) / 6
	float y = 0;
	y = WrappedNormal(110.0,5.0);
	printf("%f\n",y);
	y = 0;
	printf("------------------------------\n");
	y = WrappedNormal(110.0,5.0);
	printf("%f\n",y);
	printf("------------------------------\n");
	printf("%f\n",WrappedNormal(110.0,5.0));	
	printf("------------------------------\n");
	printf("%f\n",WrappedNormal(110.0,5.0));
	printf("------------------------------\n");
	printf("%f\n",WrappedNormal(110.0,5.0));
	printf("------------------------------\n");
	printf("%f\n",WrappedNormal(110.0,5.0));
	printf("------------------------------\n");

	return 0;
}


float WrappedNormal (float MeanAngle,float AngStdDev){

	//Fisher 1993 pg. 47-48
	//s = -2.ln(r)

	time_t t;
	float u1,u2,x,z,y,wn;
	u1=0;
	u2=0;
	int check;
	check = 1;

	while(1){
		do{

			u1 = (float)rand()/(float)RAND_MAX;
			u2 = (float)rand()/(float)RAND_MAX;
			printf("u1:%f,u2:%f\n",u1,u2);
		}while((u1<=0.0) && (u2<=0.0));

		printf("Hello \n");

   		z = 1.715528 * (u1 - 0.5) / u2;
		printf("z:%f\n",z);

    		x = 0.25 * z *z;
		printf("x:%f\n",x);

		if ((x - (1 - u2)) < glCompAcc) {
			//check = 0;
			//continue;
			printf("First\n");
			break;
			
		}
		else if (x - (-log(u2)) < glCompAcc){
			//check = 0;
			//continue;
			printf("Second\n");
			break;
			
		}
	}//while(check == 1);
	

	y = AngStdDev * z + MeanAngle;
	if ((y - 360) > -glCompAcc){ 
	    y = y - 360;
	}
 
	if (y < 0){
	    y = 360 + y;
	}
	
	printf("Last \n");
	return y;

  
}

/*
void CircMeanConcentration(double* ValArray)
{

	//Batschelet 1981, pg.7-18
	//returns directional concentration = r = mean vector length
	//and CircMean = mean angle of sample
	//CircMean (mean angle) is returned as proper flight direction (not in radians)
	//RandNumbs in ValArray() are in radians


	double sumX,sumY,x,y,Alpha,VectorLength,Number,MeanAngle,MeanConc;
	long int i;      

	sumX = 0;
	sumY = 0;
	Number = 0;

	Number = sizeof(ValArray)/sizeof(double);

	for(i=0;i<Number;i++){
		Alpha = *(ValArray+i);
		x = cos(Alpha * PI/180);
		y = sin(Alpha * PI/180);
		sumX = sumX + x;
		sumY = sumY + y;
	}
  

	if(Number > 0 ){
		sumX = sumX/Number;
		sumY = sumY/Number;
	}
	else{
		sumX = 0;
		sumY = 0;
	}
	
  	VectorLength = sqrt(sumY * sumY + sumX * sumX);


	if (sumX > 0){  		//Batschelet 1981, pg.11
		MeanAngle = atan(sumY/sumX);
	}
	else if (sumX == 0){
		MeanAngle = 0;	
	}
	else{
		MeanAngle = PI + atan(sumY/sumX);
	}
      
	MeanValues.meanAngle = MeanAngle * 180/PI;
//  	*MeanAngle = MeanAngle * 180/PI;
	MeanValues.meanConcentration = VectorLength;
// 	*MeanConc = VectorLength;

	
//	return VectorLength;
  
}
*/

//Public Function WrappedNormal(ByVal MeanAngle As Double, ByVal AngStdDev As Double) As Double

/*
double angularDeviation(double conc){

	//converts the concentration parameter (r = mean vector length) from the angular deviation	
	//angular deviation is returned as degrees
	//Batschelet 1981

	double AngDev;
	AngDev = 2 * (1 - conc);
	AngDev = sqrt(AngDev);
  	return AngDev * 180/PI;  
}


double concentration(double AngDev) 
{
	//converts angular deviation to concentration parameter (r = mean vector length)
	//Batschelet 1981
	//returns concentration parameter
	//AngDev in degrees
	double R,S;
	S = AngDev * PI/180;
	R = 1 - ((S * S) / 2);
	return R;
}
*/
