#!/bin/bash
awk -v var=$1 'BEGIN{count=0}{if($3<var){count=count+1;print $0,NR,count}}END{print count/NR}' $2 

