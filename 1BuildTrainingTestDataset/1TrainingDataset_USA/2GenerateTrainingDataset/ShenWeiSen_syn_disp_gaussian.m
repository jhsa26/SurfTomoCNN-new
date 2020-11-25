clc
clear
close all
addpath('./src')
vs_syn_dir='./Vs_GroundTruth_0.5km/';
disp_syn_dir='./Shenetal2013_disp_pg_real/';
ipers =1;% to be one, do not use it to increase the number of Vs models 
allfiles=strsplit(ls(vs_syn_dir))';
gaus_flag=1;sigma=0.5;
! rm -rf disp_out vs_out
! mkdir vs_out disp_out
% figure('pos',[675         541        1001         420])
periods=[8 10 12 14 16 18 20 22 24 26 28 30 32 35 40 45 50];
fid=fopen('surfdispgr.in','w');
fprintf(fid,'1Dmodel.txt\ndisp_gr.txt\n');
fprintf(fid,'%d %d %d %d\n',2,1,1,length(periods));
fprintf(fid,'%2.1f ',periods);
fclose(fid);

fid=fopen('surfdispph.in','w');
fprintf(fid,'1Dmodel.txt\ndisp_ph.txt\n');
fprintf(fid,'%d %d %d %d\n',2,1,0,length(periods));
fprintf(fid,'%2.1f ',periods);
fclose(fid);

nfiles=length(allfiles)-1;

for i=1:nfiles
    filename_prefix = allfiles{i};
    for iper=1:ipers
        disp([num2str(i) ' file  ' 'iper   ' num2str(iper)])
        index_num = iper+3*(i-1);
        
        
        fid=fopen('Vsmodel','w');
        fprintf(fid,'MODEL.01\nIsotropic model \nISOTROPIC \nKGS \nFLAT EARTH \n1-D \nCONSTANT VELOCITY \n');
        fprintf(fid,'LINE08\nLINE09\nLINE10\nLINE11\n');
        fprintf(fid,'H(KM) VP(KM/S)  VS(KM/S) RHO(GM/CC)  QP    QS   ETAP   ETAS   FREFP  FREFS\n');
        
        temp=load([vs_syn_dir   allfiles{i,1}]);
        
        temp = temp(:,1:2);
        num_lay=length(temp);
        thk=zeros(1,num_lay);
        th1=temp(:,1)';
        thk(1:end-1)=th1(2:end)-th1(1:end-1);
        if iper>1
            if gaus_flag==1
                vs_temp = temp(:,2)';
                vs = normrnd(vs_temp,sigma);
                for ik=1:num_lay
                    if (vs(ik)>vs_temp(ik)+3*sigma || vs(ik)>5)
                        vs(ik) = vs_temp(ik)+sigma;
                    elseif(vs(ik)<vs_temp(ik)-3*sigma || vs(ik)<1)
                        vs(ik) = vs_temp(ik)-sigma;
                    end
                end
            else
                perturb =sigma/1000*(randi(2000,[1000,1])-1000);
                perturb_vs = perturb(randperm(1000,num_lay))';
                vs=temp(:,2)'+perturb_vs;
            end


        else
            vs=temp(:,2)';
        end 

        
        if vs(2)<=vs(1)
            temp_temp = vs(2);
            vs(2) = vs(1)+0.1;
            vs(1) = temp_temp;
        end
        
        vp=zeros(1,num_lay);rho=vp;
        
        for ith=1:num_lay
            % get_vp
            if temp(ith,1)<120
                vp(ith) = get_vp(vs(ith),1,0);
            else
                vp(ith) = get_vp(vs(ith),2,0);
            end
            % get_rho
            if temp(ith,1)<120
                rho(ith) = get_rho(vp(ith),vs(ith),1,0);
            else
                rho(ith) = get_rho(vp(ith),vs(ith),2,0);
            end
        end
        
        
        
        qp=ones(1,num_lay)*1600; qp(1,1:4)=160;% 160 80  1368 1008
        qs=ones(1,num_lay)*600;  qs(1,1:4)=80;% 80  600
        
        
        fid_vs = fopen('synmodel.txt','w');
        for j=1:length(thk)
            fprintf(fid,'%6.2f %6.2f %6.2f %6.2f %5.0f %5.0f %2.0f %2.0f %2.1f %2.1f\n',thk(j),...
                vp(j),vs(j),rho(j),qp(j),qs(j),0.0,0.0,1.0,1.0);
            fprintf(fid_vs,'%6.2f %6.4f \n',temp(j,1),vs(j));
        end
        fclose(fid_vs);
        fclose(fid);
        ph_name = [num2str(iper) '-ph_'  filename_prefix ];
        gr_name = [num2str(iper) '-gr_'  filename_prefix ];
        pg_name = [num2str(iper) '_'  filename_prefix ];
        vs_name = [num2str(iper) '_'  filename_prefix ];
        setenv('prename_ph',ph_name)
        setenv('prename_gr', gr_name)
        setenv('prename_pg', pg_name)
        setenv('prename_vs',vs_name)
        ! cp Vsmodel 1Dmodel.txt
        ! ./surfdisp < surfdispgr.in>null
        ! ./surfdisp < surfdispph.in>null
        %! cp disp_ph.txt ./disp_out/$prename_ph
        %! cp disp_gr.txt ./disp_out/$prename_gr
        ! awk '{print $2}' disp_gr.txt >gr.tmp
        ! paste disp_ph.txt gr.tmp > ./disp_out/$prename_pg
        ! cp synmodel.txt ./vs_out/$prename_vs
%         if exist([disp_syn_dir  allfiles{i,1}])
%             disp_gr=load('disp_gr.txt');
%             disp_pg=load([disp_syn_dir  allfiles{i,1}]);
%             disp_ph=load('disp_ph.txt');
%             disp_1 = [disp_pg(:,1),disp_pg(:,3)];  % gr
%             disp_2 = [disp_pg(:,1),disp_pg(:,2)];  % ph
%             plotVsVpmdl_Hermman(disp_gr,disp_1,disp_ph,disp_2,'1Dmodel.txt', 'k-', 1, 'r-');
%             pause(0.0001)
% %             r1=rms(disp_gr(:,2)-disp_1(1:17,2));
% %             r2=corrcoef(disp_gr(:,2),disp_1(1:17,2));
% %             r2 = r2(2,1);
% %             disp(['rms= ',num2str(r1,'%10.4f'), '  corr= ',num2str(r2,'%10.3f')])
%         end
    end
end

! rm -rf *.in null synmodel.txt Vsmodel 1Dmodel.txt disp_gr.txt disp_ph.txt gr.tmp
