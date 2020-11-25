#!/bin/bash
# this  gmt5 script can not be used by gmt4
# and it's used for producing survey line data  along the depth profile
# Code by HJ
# 2018-03-30 

# your depth profile must be named like Z0.txt Z20.txt, 20 denote that the depth
# all depth profile must be stored in data directory.


depth=(0 3 10 15 20 30 40 60 80 100 120)
# define survey line startpoint
phaseflag="Vs"  #Vs or vpvs
#aa'
spoint="132.5/46.5"
epoint="93/29"
profilename="ProfileLine_${phaseflag}.txt"
# usually not change
spacing="0.1/0.1"           # xyz2grd -I option 
pointspacing=10               # gmt project -G option, select point every 10 km along the tracked line 
###########################
#       don't change      #
###########################
rm -rf temp
rm -rf *$profilename*
mkdir temp
#  To generate points every 10 km along a small  circle  (-N) 
gmt project  -C${spoint} -E${epoint} -G$pointspacing -Q  > temp/tracka.dat
# loop along the depth
len=${#depth[@]}
for ((i=0;i<$len;i++));do
layer=`echo $i+1|bc`
filename_cnn_usa=data/layers_vs_usa/"lay"$layer"_cnn.txt"
filename_cnn_usa_tibet=data/layers_vs_usa_tibet/"lay"$layer"_cnn.txt"
filename_sws=data/layers_vs_usa/"lay"$layer"_sws.txt"

ncfile_cnn_usa=temp/"layer"$layer"_cnn_usa.nc"
ncfile_cnn_usa_tibet=temp/"layer"$layer"_cnn_usa_tibet.nc"
ncfile_sws=temp/"layer"$layer"_sws.nc"

range_cnn=`gmt gmtinfo -C $filename_cnn_usa | awk '{print "-R"$1"/"$2"/"$3"/"$4}'`
range_cnn=`gmt gmtinfo -C $filename_cnn_usa_tibet | awk '{print "-R"$1"/"$2"/"$3"/"$4}'`
range_sws=`gmt gmtinfo -C $filename_sws | awk '{print "-R"$1"/"$2"/"$3"/"$4}'`

gmt surface  ${filename_cnn_usa}  ${range_cnn} -T0.25  -G$ncfile_cnn_usa   -I${spacing}
gmt surface  ${filename_cnn_usa_tibet}  ${range_cnn} -T0.25  -G$ncfile_cnn_usa_tibet   -I${spacing}
gmt surface  ${filename_sws}  ${range_sws} -T0.25  -G$ncfile_sws   -I${spacing}
# may be you should change
# plot distance
#gmt grdtrack temp/tracka.dat -G$ncfile_cnn | awk -v dep=${depth[$i]} '{print $3, dep, $4}' >> cnn_${profilename}
# plot lat
#gmt grdtrack temp/tracka.dat -G$ncfile | awk -v dep=${depth[$i]} '{print $2, dep, $4}' >> $profilename
# plot lon
gmt grdtrack temp/tracka.dat -G$ncfile_cnn_usa | awk -v dep=${depth[$i]} '{print $1, dep, $4}' >> usa_cnn_${profilename}
gmt grdtrack temp/tracka.dat -G$ncfile_cnn_usa_tibet | awk -v dep=${depth[$i]} '{print $1, dep, $4}' >> usa_tibet_cnn_${profilename}
gmt grdtrack temp/tracka.dat -G$ncfile_sws | awk -v dep=${depth[$i]} '{print $1, dep, $4}' >> sws_${profilename}
echo "processing file: $filename_cnn_usa"
#rm -rf $ncfile
done
#gmt project reloc.txt  -C${spoint} -E${epoint} -Fxz -W-0.5/0.5 >profile.loc.txt
#gmt project reloc.txt  -C${spoint} -E${epoint} -Fyz -W-0.5/0.5 >profile.loc.txt
rm -rf gmt.history
