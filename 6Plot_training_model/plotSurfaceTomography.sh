#!/bin/bash
################### global set, can be changed##################
gmt gmtset MAP_FRAME_TYPE                 = fancy
gmt gmtset MAP_FRAME_WIDTH                = 1p
gmt gmtset FONT_ANNOT_PRIMARY             = 12p,Helvetica,black
gmt gmtset FONT_ANNOT_SECONDARY           = 12p,Helvetica,black
gmt gmtset FONT_LABEL                     = 12p,Helvetica,black
gmt gmtset FONT_LOGO                      = 8p,Helvetica,black
gmt gmtset FONT_TITLE                     = 12p,Helvetica,black
gmt gmtset MAP_TICK_LENGTH_PRIMARY        = 0.5p/0.5p
gmt gmtset MAP_TICK_LENGTH_SECONDARY      = 0.5p/0.5p
gmt gmtset COLOR_NAN                      = 127.5  #127
gmt set PS_CHAR_ENCODING ISOLatin1+
gmt set FORMAT_GEO_MAP=ddd:mm:ss 
gmt set FORMAT_GEO_MAP=ddd:mm 
#gmt set FONT_LABEL 8p,35 MAP_LABEL_OFFSET 4p
################################################################
PS=$4 #"background.eps"
proj="-JM3i"   # for 经纬度投影
range="-R232/261/28/49.2"
data=$1 #"../layers_vs_usa/lay3_sws.txt"
cptrange=$2 #3.2/3.8
depth=$3
gmt psbasemap $range $proj -Bxa10f10 -Bya5.0f5+l"Latitude (degree)" -BWeSn  -Yc -Xc -K > $PS
gmt nearneighbor $range -I0.05 -S100k -Gship.grd -V $data
#gmt grdimage  ship.grd  $range $proj   -Ctmp.cpt   -K -O -P>>$PS
gmt makecpt  -Cjet -T${cptrange}/0.01 -I  -D  -Z  > tmp.cpt

#gmt psclip ~/DataSet/China_data/CN-border-L1.dat $range $proj -O -K >>$PS
#cat $data | awk '{print $1,$2,$3}' | gmt psxy  -J -R -A  -Ctmp.cpt  -Sc0.15  -W0.1+c   -O  -K  -V -P >>$PS
gmt grdimage  ship.grd  $range $proj   -Ctmp.cpt   -K -O -P>>$PS
#gmt pscoast -R -J -Dh  -N1/0.1p   -N2/0.5p   -O -K >>$PS
#gmt pscoast -R -J -Dh -W1/0.2p -I1/0.2p   -N1/0.5p  -G127.5  -O -K >>$PS
#gmt psclip -C -K -O >> $PS

echo "232 48.5 10 0 LM Depth=$depth km"| gmt pstext -R -J -F+f+a+j -Ggray -W0.1p,white -Gwhite -O -K -V -P >>$PS 

gmt gmtset FONT_LABEL =                     8p
gmt gmtset FONT_ANNOT_PRIMARY             = 10p,0,black
gmt gmtset FONT_ANNOT_SECONDARY           = 10p,0,black
#gmt psbasemap  -R -J -Lg-114.45/34.6+c33+w50k+lkm+f+al  -K -O >>$PS
gmt gmtset MAP_FRAME_WIDTH                = 0.5p
gmt gmtset MAP_FRAME_PEN     = thinner,black
gmt gmtset MAP_LABEL_OFFSET -0.37i
gmt psscale -R -J -DJRM+w2i/0.12i+o0.1i/-0.1i+v+mc+e   -Ctmp.cpt -Bxa0.2f0.1+l"Vs(km/s)"  -G$cptrange  -O  -K  >>$PS
#gmt psscale -R -J -DJRB+w1i/0.2i+o-0.4i/-1.1i+v+mc+e   -Ctmp.cpt -Bxa0.2f0.1  -By+l"Vs(km/s)" -G$cptrange  -O   >>$PS
#gmt psscale -R -J -DJCM+w2i/0.15i+h  -Ctmp.cpt -Bxa5f5+l"Depth (km)" -G0/20 -O  >>$PS

gmt psbasemap  -R230/300/25/51 -JM1i -Bxa10f10 -Bya10.0f10+l"Latitude (degree)" -B+n  -O -K >> $PS
#gmt pscoast   -R -J -Dh   -W1/0.1p  -N1/0.1p -Gwhite -Slightblue     -O >>$PS
gmt pscoast   -R -J  -A10000k -Dh   -W1/0.1p  -N1/0.1p -Gwhite -Slightblue     -O  -K >>$PS
gmt psxy -R -J -A -W0.2p,red -O >> $PS <<EOF
232 28 
232 49
261 49
261 28
232 28
EOF
gmt psconvert $PS -A -E300 -P -Tg
rm -rf  gmt.*
