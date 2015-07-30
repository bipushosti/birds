
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define LINESIZE	500

int main()
{

	FILE* inpFile;
	inpFile = fopen("InputData.txt","r");

	if(inpFile == NULL) {
		perror("Cannot open file with input data\n");
		return -1;
	}

	int i,j;
	char line[LINESIZE];
	memset(line,'\0',sizeof(line));
	char tempVal[15];
	memset(tempVal,'\0',sizeof(tempVal));
	char* startPtr,*endPtr;
	float Value;
	
	j = 0;

	//Counting number of rows(y)
	do{
		i = fgetc(inpFile);
		if(i == '\n') j++;
	}while(i != EOF);

	i = (j - 5)/2;

	if(i%2 !=0){
		printf("\n\tError: Unequal number of starting locations and starting dates\n\n");
	}

	

	while(fgets(line,LINESIZE,inpFile)!=NULL){
		//startPtr = line;
		//memset(tempVal,'\0',sizeof(tempVal));
		//endPtr = strchr(startPtr,' ');
		//strncpy(tempVal,startPtr,endPtr-startPtr);
		//	
		//endPtr = endPtr + 1;
		//startPtr = endPtr;
		if(i!=0){
		
		}
		i++;

		printf("Lines %d\n",i);
		printf("%c\n",line[0]);
		if(line[0] == '\n') break;
		memset(line,'\0',sizeof(line));
	}
	//printf("Lines %d\n",i);
/*
	//Counting number of rows(y)
	do{
		i = fgetc(datTxt1);
		if(i == '\n') YSIZE++;
	}while(i != EOF);

*/

	fclose(inpFile);
	

}















/*
int convert_to_month(char* month,char* day);
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------
int convert_to_month(char* month,char * day)
{
	int index,offset;
	if(strcmp(month,"AUG")==0){
		index = 0; //The data starts in august
	}
	else if(strcmp(month,"SEPT")==0){
		index = 31; //The data for september starts after 31 days of august
	}
	else if(strcmp(month,"OCT")==0){
		index = 61; //The data for october starts after 31+30 days of sept and august respectively.
	}
	else if(strcmp(month,"SEPT")==0){
		index = 92; //The data for october starts after 31+30+31 days of sept,aug and oct respectively.
	}
	else{
		printf("\n\t\tIncorrect month used\n\t\tUse between August-November inclusive; Only use abbriviated caps of the months; august = AUG\n\n");
		return -1;
	}
	
	offset = index + atoi(day) - 1;
	return offset;

}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------
int main(int argc,char* argv[])
{
	char baseFileName[] = "../Birds_data/InterpolatedData/";
	char yearFileName[80];
	char fullFileName[80];

	if(argc < 4){
		printf("\n\tNot enough arguments; Needed 3 provided %d \n\tUsage:\tExecutableFileName StartYear(Full year)  StartMonth(Abbr. all caps) StartDay(Without initial zeroes)\n\n",argc - 1);
		return 0;
	}
	else if (argc>4){
		printf("\n\tToo many arguments; Needed 3 provided %d \n\tUsage:\tExecutableFileName StartYear(Full year)  StartMonth(Abbr. all caps) StartDay(Without initial zeroes)\n\n",argc-1);
		return 0;
	}

	//Getting the offset into the data so that user can specify a starting date
	int offset_into_data = 0;
	offset_into_data = convert_to_month(argv[2],argv[1]);

	//Checking if correct year specified
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

//----------------------------------------------------------------------------------------------------------------------------------------------------------------
	memset(fullFileName,0,strlen(fullFileName));

	FILE *vdataTxt,*udataTxt;	
	strcpy(fullFileName,yearFileName);
	strcat(fullFileName,"U850");
	strcat(fullFileName,"_");
	strcat(fullFileName,argv[1]);
	strcat(fullFileName,".txt");


	//udataTxt = fopen("../Birds_data/InterpolatedData/U850_30days_Sept_2011.txt","r");
	udataTxt = fopen(fullFileName,"r");
	vdataTxt = fopen("../Birds_data/output/V850_30days_Sept_2011.txt","r");
	if(udataTxt == NULL) {
		perror("Cannot open file with U850 data\n");
		return -1;
	}
	if(vdataTxt == NULL) {
		perror("Cannot open file with V850 data\n");
		return -1;
	}
	
	fclose(udataTxt);
	fclose(vdataTxt);
	
	return 1;
}
*/

