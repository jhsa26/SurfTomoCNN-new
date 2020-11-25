#!/bin/bash
################### global set, can be changed##################
gmt gmtset MAP_FRAME_TYPE                 = fancy
gmt gmtset MAP_FRAME_WIDTH                = 1p
gmt gmtset FONT_ANNOT_PRIMARY             = 15p,0,black
gmt gmtset MAP_TITLE_OFFSET               = -8p
#gmt gmtset FONT_ANNOT_PRIMARY             = 12p,Helvetica,black
gmt gmtset FONT_ANNOT_SECONDARY           = 15p,0,black
gmt gmtset FONT_LABEL                     = 15p,0,black
gmt gmtset FONT_LOGO                      = 8p,0,black
gmt gmtset FONT_TITLE                     = 15p,0,black
gmt gmtset MAP_TICK_LENGTH_PRIMARY        = 0.5p/0.5p
gmt gmtset MAP_TICK_LENGTH_SECONDARY      = 0.5p/0.5p
gmt gmtset COLOR_NAN                      = 225   #127
gmt set PS_CHAR_ENCODING ISOLatin1+
gmt set FORMAT_GEO_MAP=ddd:mm:ss 
gmt set FORMAT_GEO_MAP=ddd:mm 
#gmt set FONT_LABEL 8p,35 MAP_LABEL_OFFSET 4p
################################################################
period=$1
proj="-JM3i"   # for 经纬度投影
range="-R73/135/15/55"
data="./Output/${period}s.txt"
title="Uncertainties of Group Velocity"
PS=./Figs/groupUn_${period}s.eps
cptrange=0.04/0.1
grdname=temp.grd
gmt psbasemap $range $proj -Bxa15f15 -Bya10.0f10 -BWeSn+t"$title"  -K > $PS
#cat $data | awk '{print $1,$2,$3}' | gmt nearneighbor $range -I0.05 -S100k -GPhVs.grd  
cat $data | awk '{print $1,$2,$6}' | gmt nearneighbor $range -I0.05 -S100k -G$grdname 
gmt makecpt  -Cjet -T${cptrange}/0.01 -I  -D  -Z  > tmp.cpt

gmt psclip ~/DataSet/China_data/CN-border-L1.dat $range $proj -O -K >>$PS
#cat $data_cnn | awk '{print $1,$2,$3}' | gmt psxy  -J -R -A  -Ctmp.cpt  -Sc0.1  -W0.1+c   -O  -K  -V -P >>$PS
gmt grdimage $grdname   $range $proj   -Ctmp.cpt   -K -O -P>>$PS
gmt psclip -C -K -O >> $PS
cat ~/DataSet/China_data/China_tectonic.dat_new | gmt psxy -J -R  -W0.5p,- -O -K -V -P >>$PS
cat ~/DataSet/China_data/CN-border-L1.dat | gmt psxy -J -R -W0.5 -O -K -V  -P>>$PS
gmt pstext -J -R -F+f+a+j -Ggray     -O -K <<EOF >>$PS
75 53       15p,5,black 0  LM   ${period}s

EOF

gmt gmtset FONT_LABEL =                     15p
gmt gmtset FONT_ANNOT_PRIMARY             = 15p,0,black
gmt gmtset FONT_ANNOT_SECONDARY           = 15p,0,black
#gmt psbasemap  -R -J -Lg-114.45/34.6+c33+w50k+lkm+f+al  -K -O >>$PS
gmt gmtset MAP_FRAME_WIDTH                = 0.5p
gmt gmtset MAP_FRAME_PEN     = thinner,black
#gmt gmtset MAP_LABEL_OFFSET -0.37i
#gmt psscale -R -J -DJCM+w2.5i/0.12i+o1.7i/0i+v+mc+e   -Ctmp.cpt -Bxa0.2f0.1+l"Vs(km/s)" -G$cptrange  -O   >>$PS
gmt gmtset MAP_LABEL_OFFSET = -0.8i
#gmt gmtset FONT_LABEL =                     8p,12,black
#gmt gmtset PS_CHAR_ENCODING = Standard+
#gmt psscale -R -J -DJCM+w2.35i/0.12i+o1.8i/0i+v   -Ctmp.cpt -Bxa0.01f0.01 -Bx+l"Uncertainties" -By+l"km/s" -G$cptrange  -O   >>$PS
gmt psscale -R -J -DJCM+w2.35i/0.12i+o1.8i/0i+v   -Ctmp.cpt -Bxa0.01f0.01 -By+l"km/s" -G$cptrange  -O   >>$PS
#gmt psscale -R -J -DJCM+w2.3i/0.12i+o1.8i/0i+v   -Ctmp.cpt -Bxa1f1 -By+l"\143" -G$cptrange  -O   >>$PS
#gmt psscale -R -J -DJRB+w1i/0.2i+o-0.4i/-1.1i+v+mc+e   -Ctmp.cpt -Bxa0.2f0.1  -By+l"Vs(km/s)" -G$cptrange  -O   >>$PS
#gmt psscale -R -J -DJCM+w2i/0.15i+h  -Ctmp.cpt -Bxa5f5+l"Depth (km)" -G0/20 -O  >>$PS
gmt psconvert $PS -A -E300 -P -Tg
rm -rf  gmt.* 
rm -rf $PS
rm -rf *.grd
