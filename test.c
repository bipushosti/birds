
#include <stdio.h>
#include <math.h>
#include <float.h>



void main() 
{
	long i = 0;
	long j = 20;
	int k = 30;


	for (i=0;i<2000;i++,k++) {
		if(i>j) {
			printf("Limit reached\n");
			break;
			
		} 
		printf("%ld,%d\n",i,k);
		printf("Not\n");
	}
	printf("%ld,%d\n",i,k);
	


}
