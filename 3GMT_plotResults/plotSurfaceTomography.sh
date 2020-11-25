#!/bin/bash
################### global set, can be changed##################
gmt gmtset MAP_FRAME_TYPE                 = fancy
gmt gmtset MAP_FRAME_WIDTH                = 1p
gmt gmtset FONT_ANNOT_PRIMARY             = 15p,5,black
#gmt gmtset FONT_ANNOT_PRIMARY             = 12p,Helvetica,black
gmt gmtset FONT_ANNOT_SECONDARY           = 15p,5,black
gmt gmtset FONT_LABEL                     = 15p,5,black
gmt gmtset FONT_LOGO                      = 8p,5,black
gmt gmtset FONT_TITLE                     = 15p,5,black
gmt gmtset MAP_TICK_LENGTH_PRIMARY        = 0.5p/0.5p
gmt gmtset MAP_TICK_LENGTH_SECONDARY      = 0.5p/0.5p
gmt gmtset COLOR_NAN                      = 225   #127
gmt set PS_CHAR_ENCODING ISOLatin1+
gmt set FORMAT_GEO_MAP=ddd:mm:ss 
gmt set FORMAT_GEO_MAP=ddd:mm 
#gmt set FONT_LABEL 8p,35 MAP_LABEL_OFFSET 4p
################################################################
PS=$6 #"background.eps"
proj="-JM3i"   # for 经纬度投影
range="-R73/135/15/55"
data_cnn1=$1 #"../layers_vs_usa/lay3_sws.txt"
data_cnn2=$2 #"../layers_vs_usa/lay3_sws.txt"
data_sws=$3 #"../layers_vs_usa/lay3_sws.txt"
cptrange=$4 #3.2/3.8
depth=$5
gmt nearneighbor $range -I0.05 -S100k -Gship_cnn1.grd -V $data_cnn1
gmt nearneighbor $range -I0.05 -S100k -Gship_cnn2.grd -V $data_cnn2
gmt nearneighbor $range -I0.05 -S100k -Gship_sws.grd -V $data_sws
gmt makecpt  -Cjet -T${cptrange}/0.01 -I  -D  -Z  > tmp.cpt

gmt psbasemap $range $proj -Bxa15f15 -Bya10.0f10+l"Latitude (degree)" -BWeSn  -K > $PS
gmt psclip ~/DataSet/China_data/CN-border-L1.dat $range $proj -O -K >>$PS
#cat $data | awk '{print $1,$2,$3}' | gmt psxy  -J -R -A  -Ctmp.cpt  -Sc0.15  -W0.1+c   -O  -K  -V -P >>$PS
gmt grdimage  ship_cnn1.grd  $range $proj   -Ctmp.cpt   -K -O -P>>$PS
gmt psclip -C -K -O >> $PS
cat ~/DataSet/China_data/China_tectonic.dat_new | gmt psxy -J -R  -W0.5p,- -O -K -V -P >>$PS
cat ~/DataSet/China_data/CN-border-L1.dat | gmt psxy -J -R -W0.5 -O -K -V  -P>>$PS
#echo "73.2 54. 14 0 LM Depth=$depth km"| gmt pstext -R -J -F+f+a+j   -O -K -V -P >>$PS 
echo "75 20 15 0 LM Depth=$depth km"| gmt pstext -R -J -F+f+a+j   -O -K -V -P >>$PS 
#echo "75 53 15 0 LM CNN trained without Tibet"| gmt pstext -R -J -F+f+a+j   -O -K -V -P >>$PS 
#gmt pscoast -R -J -Df -A100k -W1/0.2p     -O -K >>$PS



gmt psbasemap $range $proj -Bxa15f15 -Bya10.0f10+l"Latitude (degree)" -BweSn -X3.3i -O -K >> $PS
gmt psclip ~/DataSet/China_data/CN-border-L1.dat $range $proj -O -K >>$PS
#cat $data | awk '{print $1,$2,$3}' | gmt psxy  -J -R -A  -Ctmp.cpt  -Sc0.15  -W0.1+c   -O  -K  -V -P >>$PS
gmt grdimage  ship_cnn2.grd  $range $proj   -Ctmp.cpt   -K -O -P>>$PS
gmt psclip -C -K -O >> $PS
cat ~/DataSet/China_data/China_tectonic.dat_new | gmt psxy -J -R  -W0.5p,- -O -K -V -P >>$PS
cat ~/DataSet/China_data/CN-border-L1.dat | gmt psxy -J -R -W0.5 -O -K -V  -P>>$PS
#echo "73.2 54. 14 0 LM Depth=$depth km"| gmt pstext -R -J -F+f+a+j   -O -K -V -P >>$PS 
echo "75 20 15 0 LM Depth=$depth km"| gmt pstext -R -J -F+f+a+j   -O -K -V -P >>$PS 
#echo "75 53 15 0 LM CNN trained with Tibet"| gmt pstext -R -J -F+f+a+j   -O -K -V -P >>$PS 


gmt psbasemap $range $proj -Bxa15f15 -Bya10.0f10+l"Latitude (degree)" -BweSn  -X3.3i -O -K >> $PS
gmt psclip ~/DataSet/China_data/CN-border-L1.dat $range $proj -O -K >>$PS
#cat $data | awk '{print $1,$2,$3}' | gmt psxy  -J -R -A  -Ctmp.cpt  -Sc0.15  -W0.1+c   -O  -K  -V -P >>$PS
gmt grdimage  ship_sws.grd  $range $proj   -Ctmp.cpt   -K -O -P>>$PS
gmt psclip -C -K -O >> $PS
cat ~/DataSet/China_data/China_tectonic.dat_new | gmt psxy -J -R  -W0.5p,- -O -K -V -P >>$PS
cat ~/DataSet/China_data/CN-border-L1.dat | gmt psxy -J -R -W0.5 -O -K -V  -P>>$PS
#echo "73.2 54. 14 0 LM Depth=$depth km"| gmt pstext -R -J -F+f+a+j   -O -K -V -P >>$PS 
echo "75 20 15 0 LM Depth=$depth km"| gmt pstext -R -J -F+f+a+j   -O -K -V -P >>$PS 
#echo "75 53 15 0 LM Shen et al. (2016)"| gmt pstext -R -J -F+f+a+j   -O -K -V -P >>$PS 
#gmt pscoast -R -J -Df -A100k -W1/0.2p     -O -K >>$PS










gmt gmtset FONT_LABEL =                     15p
gmt gmtset FONT_ANNOT_PRIMARY             = 15p,5,black
gmt gmtset FONT_ANNOT_SECONDARY           = 15p,5,black
#gmt psbasemap  -R -J -Lg-114.45/34.6+c33+w50k+lkm+f+al  -K -O >>$PS
gmt gmtset MAP_FRAME_WIDTH                = 0.5p
gmt gmtset MAP_FRAME_PEN     = thinner,black
#gmt gmtset MAP_LABEL_OFFSET -0.37i
#gmt psscale -R -J -DJCM+w2.5i/0.12i+o1.7i/0i+v+mc+e   -Ctmp.cpt -Bxa0.2f0.1+l"Vs(km/s)" -G$cptrange  -O   >>$PS
gmt gmtset MAP_LABEL_OFFSET = -0.5i
gmt psscale -R -J -DJCM+w2.35i/0.12i+o1.8i/0i+v   -Ctmp.cpt -Bxa0.2f0.1 -By+l"Vs(km/s)" -G$cptrange  -O   >>$PS
#gmt psscale -R -J -DJRB+w1i/0.2i+o-0.4i/-1.1i+v+mc+e   -Ctmp.cpt -Bxa0.2f0.1  -By+l"Vs(km/s)" -G$cptrange  -O   >>$PS
#gmt psscale -R -J -DJCM+w2i/0.15i+h  -Ctmp.cpt -Bxa5f5+l"Depth (km)" -G0/20 -O  >>$PS
gmt psconvert $PS -A -E300 -P -Tg
rm -rf  gmt.* *.grd 
