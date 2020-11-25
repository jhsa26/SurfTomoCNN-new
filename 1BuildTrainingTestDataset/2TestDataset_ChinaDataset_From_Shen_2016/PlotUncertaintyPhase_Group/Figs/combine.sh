montage ./phaseVs_10s.png ./phaseUn_10s.png  \
        ./phaseVs_30s.png ./phaseUn_30s.png  \
        ./phaseVs_40s.png ./phaseUn_40s.png  \
          -tile x3 -geometry +30+30   comph.jpg
montage ./GroupVs_10s.png ./groupUn_10s.png \
        ./GroupVs_30s.png ./groupUn_30s.png \
        ./GroupVs_40s.png ./groupUn_40s.png \
        -tile x3   -geometry +30+30 comgr.jpg
