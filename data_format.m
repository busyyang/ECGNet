function  [sig_out,offset]=data_format(sig_in)
% format  sig_in to fixed length of 300 points.
% return fixed data and added points before sig_in.
% 2019/11/26    YANG Jie    Init
    valid_len=length(sig_in);
    total_len=300;
    if valid_len<total_len
        %random padding
        padding_len=total_len-valid_len;
        offset=unidrnd(padding_len);
        sig_out=[ones(1,offset)*sig_in(1),sig_in',ones(1,padding_len-offset)*sig_in(end)];
    else
        % resample
        offset=0;
        sig_out=resample(sig_in,total_len,valid_len)';
    end
end