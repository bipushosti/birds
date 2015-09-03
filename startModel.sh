#!/bin/bash

#------------------------------------
#The file that has all the input data

INPUT_FILE='InputData.txt'
#------------------------------------

declare -a Dates
declare -a Coords
line_no=0
line_no2=0
line_no3=0
total_dates=0
filePath="~/Documents/Birds_Full/birds/birds"

make clean
make

get_total_lines()
{
	#echo "Hello!!"
	while read line
	do
		((line_no++))				
	done

	#echo ${Array[@]}
} < $INPUT_FILE



read_all_dates()
{
	while read line2
	do		
		if [[ $line_no2 -ne 0 ]]
		then
			Dates[$line_no2 -1]=$line2

			if [[ $line_no2 -ge $total_dates ]]
			then
				break
			fi
		fi
		((line_no2++))	
	done
	#echo ${Dates[@]}

}< $INPUT_FILE

get_total_lines

if [[ $((($line_no - 5) % 2)) != 0 ]] 
then
	echo "	Dates provided and starting locations are not equal"
	echo "	OR the format is incorrect.Only one empty line below the end of dates (</Dates>)"
else
	total_dates=$(((line_no - 5 )/2))
fi

read_all_dates

index=0
skip_lines=total_dates 
((skip_lines+=4))


replace_inCoords()
{
	index=0	
	#while (($index < $total_dates))
	#do
		while read line3
		do
			if [[ $line_no3 -gt $total_dates+$skip_lines-1 ]]
			then
				break
			fi
			

			if [[ $line_no3 -ge $skip_lines ]]
			then
				line3=${line3//(/ }
				line3=${line3//)/ }
				line3=${line3//,/ }	

				#unset $Coords
				IFS=' ' read -a Coords <<< "$line3"
			
				coordsIdx=0
				coordsIdx2=0
				while (($coordsIdx < ${#Coords[@]}))
				do
#					coordsIdx2 = $(( coordsIdx + 1 ))
	        			./birds ${Dates[index]} ${Coords[coordsIdx]} ${Coords[coordsIdx++]}
					(( coordsIdx+=2 ))

				done

			        (( index++ ))
			       			
			fi

		
			#
			#echo $line3
			((line_no3++))
		
		done
		
	#done



}< $INPUT_FILE

replace_inCoords

run_program()
{
	

	while (($index < $total_dates))
	do
		echo ${Dates[index]}
		./birds ${Dates[index]}
		(( index++ ))
	done
}

#run_program

exit $?
