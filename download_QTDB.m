record_list = physionetdb('qtdb');
for i =length(record_list):-1:1
    % 下载数据
    [sig,Fs,tm] = rdsamp(['qtdb/',record_list{i}]);
    % 下载标注
    [ann_pu0,anntype,~,~,num]=rdann(['qtdb/',record_list{i}],'pu0');
    rwave=ann_pu0(strfind(anntype','N'));
    % 保存数据
    save([record_list{i},'data'],'sig','Fs','tm');
    qwave=ann_pu0(strfind(anntype','p'));
    save([record_list{i},'ann'],'rwave','qwave','ann_pu0','anntype');
    % 数据可视化
    % s=sig(:,1);
    % plot(tm,s);hold on;plot(tm(rwave),s(rwave),'*');
    % plot(tm(qwave),s(qwave),'o');
    % legend('ECG','R','P');
end