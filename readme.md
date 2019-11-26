ECG R-wave and P-wave localization in paper:
~~~
@InProceedings{Abrishami2018,
  author    = {H. {Abrishami} and M. {Campbell} and C. {Han} and R. {Czosek} and X. {Zhou}},
  title     = {P-{QRS}-T localization in {ECG} using deep learning},
  booktitle = {Proc. IEEE EMBS Int. Conf. Biomedical Health Informatics (BHI)},
  year      = {2018},
  pages     = {210--213},
  month     = mar,
  doi       = {10.1109/BHI.2018.8333406}
}
~~~
Since the code of this paper is not open, I implemented the code according this paper with `keras` framework.
# Data preprocess
Data preprocessed in MATLAB. Download data files from `https://www.physionet.org/content/qtdb/1.0.0/` with `download_QTDB.m`. PC will get `xxxann.mat` for Y and `xxxdata.mat` for X.\
For input data to keras conveniently, `Segmentor.m` will segment all recording into complexes and position of P-wave and R-wave is also saved in `segmentors.mat`.\
if you load `segmentor.mat` into matlab. You will get `segs` with 96863 by 300 and `anns` with dimention of 96863 by 2 in workspace. That mean there are 96863 complexes with length of 300 sampling points.\
`ann[:,1]` presents position of P-wave. `ann[:,2]` presents position of R-wave. More detail can be found in paper. 

# models
for fully-connected net usage:
~~~python
    python ./paper_models_codes/denseNet_P_R_localization.py
~~~
for 1D CNN usage:
~~~python
    python ./paper_models_codes/ECGNet.py
~~~
for 1D CNN with dropout usage:
~~~python
    python ./paper_models_codes/ECGNet_Dropout.py
~~~
