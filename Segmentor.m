% 信号分割为每个心跳周期，分割起始点为(p)前括号的前0.1秒(Fs=250),即25个点
% 2019/11/26    YANG Jie    Init
mypath='./data/';
mydata=dir(mypath);
anns=[];
segs=[];
for i =1: length(mydata)
    disp(['Processing ',num2str(i),' / ',num2str(length(mydata)),' file......']);
    if mydata(i).isdir==0 && ~isempty(strfind(mydata(i).name,'data'))
        data_file=[mypath,mydata(i).name];
        ann_file=[mypath,strrep(mydata(i).name,'data','ann')];
        load(data_file);
        load(ann_file);
        p_start=strfind(anntype','(p)(N)');
        % 去除第一个点
        p_start_pos=ann_pu0(p_start(2:end))-0.1*Fs;
        p_pos=ann_pu0(p_start(2:end)+1);
        r_pos=ann_pu0(p_start(2:end)+4);
        s=sig(:,1);
        for j =1:length(p_start_pos)-1
            tmp_data=s(p_start_pos(j):p_start_pos(j+1));
            [paddinged,offset]=data_format(tmp_data);
            anns=[anns;p_pos(j)-p_start_pos(j)+offset+1,r_pos(j)-p_start_pos(j)+offset+1];
            segs=[segs;paddinged];
        end
    end
end
save('segmentors','segs','anns');







