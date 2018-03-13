pos = [[10, 1]; [10, 6]; [1, 8]; [2, 9]; [3, 9]; [4, 7]; [6, 5]; [8, 3]; [9, 7]];

mode = 'BESTMICsad';

f = fopen(sprintf('best_PEASS_%s.csv', mode), 'w');
fwrite(f, "pos1,pos2,sample,bestOPS1,bestOPS2,bestTPS1,bestTPS2,bestAPS1,bestAPS2,bestIPS1,bestIPS2");
options.destDir = 'trash/';

for i = 1:9
    for sample = ["A","C"]
        [gt1, fs] = audioread(sprintf('output_wavs/best_[%d, %d]_%s_1_gt.wav', pos(i, 1), pos(i, 2), sample));
        [gt2, fs] = audioread(sprintf('output_wavs/best_[%d, %d]_%s_2_gt.wav', pos(i, 1), pos(i, 2), sample));
        [estimateFile1, fs] = audioread(sprintf('output_wavs/best_[%d, %d]_%s_1.wav', pos(i, 1), pos(i, 2), sample));
        [estimateFile2, fs] = audioread(sprintf('output_wavs/best_[%d, %d]_%s_2.wav',  pos(i, 1), pos(i, 2), sample));
        
        min_len = min([length(gt1), length(gt2), length(estimateFile1), length(estimateFile2)]);
        
        audiowrite(sprintf('output_wavs/best_[%d, %d]_%s_1_gt.wav', pos(i, 1), pos(i, 2), sample), gt1(1:min_len), fs)
        audiowrite(sprintf('output_wavs/best_[%d, %d]_%s_2_gt.wav', pos(i, 1), pos(i, 2), sample), gt2(1:min_len), fs)
        audiowrite(sprintf('output_wavs/best_[%d, %d]_%s_1.wav', pos(i, 1), pos(i, 2), sample), estimateFile1(1:min_len), fs)
        audiowrite(sprintf('output_wavs/best_[%d, %d]_%s_2.wav', pos(i, 1), pos(i, 2), sample), estimateFile2(1:min_len), fs)
        
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


          originalFiles1 = {...
        sprintf('output_wavs/best_[%d, %d]_%s_1_gt.wav', pos(i, 1), pos(i, 2), sample);...
        sprintf('output_wavs/best_[%d, %d]_%s_2_gt.wav', pos(i, 1), pos(i, 2), sample)};
     originalFiles2 = {...
        sprintf('output_wavs/best_[%d, %d]_%s_2_gt.wav', pos(i, 1), pos(i, 2), sample);...
        sprintf('output_wavs/best_[%d, %d]_%s_1_gt.wav', pos(i, 1), pos(i, 2), sample)};

     estimateFile1 = sprintf('output_wavs/best_[%d, %d]_%s_1.wav', pos(i, 1), pos(i, 2), sample);
     estimateFile2 = sprintf('output_wavs/best_[%d, %d]_%s_2.wav',  pos(i, 1), pos(i, 2), sample);
      
     peass_gt1 = PEASS_ObjectiveMeasure(originalFiles1,estimateFile1,options);
     peass_gt2 = PEASS_ObjectiveMeasure(originalFiles2,estimateFile2,options);
     
     fwrite(f, sprintf("\n%d,%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,", pos(i,1), pos(i,2), sample, ...
         peass_gt1.OPS,peass_gt2.OPS, ...
         peass_gt1.TPS,peass_gt2.TPS, ...
         peass_gt1.APS,peass_gt2.APS, ...
         peass_gt1.IPS,peass_gt2.IPS ...
         ));
    end
end
