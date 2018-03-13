% stoi evaluation

pos = [[10, 1]; [10, 6]; [1, 8]; [2, 9]; [3, 9]; [4, 7]; [6, 5]; [8, 3]; [9, 7]];

mode = 'BESTMICsad';


f = fopen(sprintf('eval_stoi_%s.csv', mode), 'w');
fwrite(f, "pos1,pos2,sample,stoi1,gtstoi1,beststoi1,stoi2,gtstoi2,beststoi2");

%% EVAL STOI

for i = 1:9
    for sample = ["A","C"]
     % load so
     [eval_y1, fs] = audioread(sprintf('output_wavs/%s_[%d, %d]_%s_1.wav', mode, pos(i, 1), pos(i, 2), sample));
     [eval_y2, fs] = audioread(sprintf('output_wavs/%s_[%d, %d]_%s_2.wav', mode, pos(i, 1), pos(i, 2), sample));
     [eval_y1_gt, fs] = audioread(sprintf('output_wavs/%s_[%d, %d]_%s_1_gt.wav', mode, pos(i, 1), pos(i, 2), sample));
     [eval_y2_gt, fs] = audioread(sprintf('output_wavs/%s_[%d, %d]_%s_2_gt.wav', mode, pos(i, 1), pos(i, 2), sample));
     
     stoi1 = stoi(eval_y1_gt, eval_y1, fs);
     stoi2 = stoi(eval_y2_gt, eval_y2, fs);
     
     % load emp
     [emp_y1, fs] = audioread(sprintf('output_wavs/emp_lim_[%d, %d]_%s_1.wav', pos(i, 1), pos(i, 2), sample));
     [emp_y2, fs] = audioread(sprintf('output_wavs/emp_lim_[%d, %d]_%s_2.wav', pos(i, 1), pos(i, 2), sample));
     [emp_y1_gt, fs] = audioread(sprintf('output_wavs/emp_lim_[%d, %d]_%s_1_gt.wav', pos(i, 1), pos(i, 2), sample));
     [emp_y2_gt, fs] = audioread(sprintf('output_wavs/emp_lim_[%d, %d]_%s_2_gt.wav', pos(i, 1), pos(i, 2), sample));
     
     gtstoi1 = stoi(emp_y1_gt, emp_y1, fs);
     gtstoi2 = stoi(emp_y2_gt, emp_y2, fs);
     
     % load best
     [best_y1, fs] = audioread(sprintf('output_wavs/best_[%d, %d]_%s_1.wav', pos(i, 1), pos(i, 2), sample));
     [best_y2, fs] = audioread(sprintf('output_wavs/best_[%d, %d]_%s_2.wav', pos(i, 1), pos(i, 2), sample));
     [best_y1_gt, fs] = audioread(sprintf('output_wavs/best_[%d, %d]_%s_1_gt.wav', pos(i, 1), pos(i, 2), sample));
     [best_y2_gt, fs] = audioread(sprintf('output_wavs/best_[%d, %d]_%s_2_gt.wav', pos(i, 1), pos(i, 2), sample));
     
     beststoi1 = stoi(best_y1_gt, best_y1, fs);
     beststoi2 = stoi(best_y2_gt, best_y2, fs);
     
     fwrite(f, sprintf("\n%d,%d,%s,%f,%f,%f,%f,%f,%f", pos(i,1), pos(i,2), sample, stoi1,gtstoi1,beststoi1, stoi2,gtstoi2,beststoi2));
     
    end
end

%% EVAL PEASS
pos = [[10, 1]; [10, 6]; [1, 8]; [2, 9]; [3, 9]; [4, 7]; [6, 5]; [8, 3]; [9, 7]];

mode = 'BESTMICsad';

f = fopen(sprintf('eval_PEASS_%s.csv', mode), 'w');
fwrite(f, "pos1,pos2,sample,OPS1,gtOPS1,OPS2,gtOPS2,TPS1,gtTPS1,TPS2,gtTPS2,APS1,gtAPS1,APS2,gtAPS2,IPS1,gtIPS1,IPS2,gtIPS2");
options.destDir = 'trash/';

for i = 1:9
    for sample = ["A","C"]
     % load so
     originalFiles1 = {...
        sprintf('output_wavs/%s_[%d, %d]_%s_1_gt.wav', mode, pos(i, 1), pos(i, 2), sample);...
        sprintf('output_wavs/%s_[%d, %d]_%s_2_gt.wav', mode, pos(i, 1), pos(i, 2), sample)};
     originalFiles2 = {...
        sprintf('output_wavs/%s_[%d, %d]_%s_2_gt.wav', mode, pos(i, 1), pos(i, 2), sample);...
        sprintf('output_wavs/%s_[%d, %d]_%s_1_gt.wav', mode, pos(i, 1), pos(i, 2), sample)};

     estimateFile1 = sprintf('output_wavs/%s_[%d, %d]_%s_1.wav', mode, pos(i, 1), pos(i, 2), sample);
     estimateFile2 = sprintf('output_wavs/%s_[%d, %d]_%s_2.wav', mode, pos(i, 1), pos(i, 2), sample);
      
     peass_eval1 = PEASS_ObjectiveMeasure(originalFiles1,estimateFile1,options);
     peass_eval2 = PEASS_ObjectiveMeasure(originalFiles2,estimateFile2,options);

     % load emp1
          originalFiles1 = {...
        sprintf('output_wavs/emp_lim_[%d, %d]_%s_1_gt.wav', pos(i, 1), pos(i, 2), sample);...
        sprintf('output_wavs/emp_lim_[%d, %d]_%s_2_gt.wav', pos(i, 1), pos(i, 2), sample)};
     originalFiles2 = {...
        sprintf('output_wavs/emp_lim_[%d, %d]_%s_2_gt.wav', pos(i, 1), pos(i, 2), sample);...
        sprintf('output_wavs/emp_lim_[%d, %d]_%s_1_gt.wav', pos(i, 1), pos(i, 2), sample)};

     estimateFile1 = sprintf('output_wavs/emp_lim_[%d, %d]_%s_1.wav', pos(i, 1), pos(i, 2), sample);
     estimateFile2 = sprintf('output_wavs/emp_lim_[%d, %d]_%s_2.wav',  pos(i, 1), pos(i, 2), sample);
      
     peass_gt1 = PEASS_ObjectiveMeasure(originalFiles1,estimateFile1,options);
     peass_gt2 = PEASS_ObjectiveMeasure(originalFiles2,estimateFile2,options);
     
     % load best
%           originalFiles1 = {...
%         sprintf('output_wavs/best_[%d, %d]_%s_1_gt.wav', pos(i, 1), pos(i, 2), sample);...
%         sprintf('output_wavs/best_[%d, %d]_%s_2_gt.wav', pos(i, 1), pos(i, 2), sample)};
%      originalFiles2 = {...
%         sprintf('output_wavs/best_[%d, %d]_%s_2_gt.wav', pos(i, 1), pos(i, 2), sample);...
%         sprintf('output_wavs/best_[%d, %d]_%s_1_gt.wav', pos(i, 1), pos(i, 2), sample)};
% 
%      estimateFile1 = sprintf('output_wavs/best_[%d, %d]_%s_1.wav', pos(i, 1), pos(i, 2), sample);
%      estimateFile2 = sprintf('output_wavs/best_[%d, %d]_%s_2.wav',  pos(i, 1), pos(i, 2), sample);
%       
%      peass_best1 = PEASS_ObjectiveMeasure(originalFiles1,estimateFile1,options);
%      peass_best2 = PEASS_ObjectiveMeasure(originalFiles2,estimateFile2,options);
     

%      fwrite(f, sprintf("\n%d,%d,%s,%f,%f,%f,%f,%f,%f", pos(i,1), pos(i,2), sample, ...
%          peass_eval1.OPS,peass_gt1.OPS,peass_best1.OPS, peass_eval2.OPS,peass_gt2.OPS,peass_best2.OPS,...
%          peass_eval1.TPS,peass_gt1.TPS,peass_best1.TPS, peass_eval2.TPS,peass_gt2.TPS,peass_best2.TPS,...
%          peass_eval1.APS,peass_gt1.APS,peass_best1.APS, peass_eval2.APS,peass_gt2.APS,peass_best2.APS,...
%          peass_eval1.IPS,peass_gt1.IPS,peass_best1.IPS, peass_eval2.IPS,peass_gt2.IPS,peass_best2.IPS...
%          ));
     fwrite(f, sprintf("\n%d,%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", pos(i,1), pos(i,2), sample, ...
         peass_eval1.OPS,peass_gt1.OPS, peass_eval2.OPS,peass_gt2.OPS,...
         peass_eval1.TPS,peass_gt1.TPS, peass_eval2.TPS,peass_gt2.TPS,...
         peass_eval1.APS,peass_gt1.APS, peass_eval2.APS,peass_gt2.APS,...
         peass_eval1.IPS,peass_gt1.IPS, peass_eval2.IPS,peass_gt2.IPS...
         ));

    end
end




