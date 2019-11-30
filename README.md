# Data-Quality-Analyzer

## How to Run
---



## Data
---
- 데이터는 다음과 같은 형식으로 저장되어 있어야한다.
  - img_folder : 이미지들을 저장하고 있는 폴더
  - data_label_info.csv : ImageFileName, Label 두 column으로 구성되어있으며 ImageFileNmae은 이미지명(ex. img001.jpg, etc), Label은 해당 이미지의 label 값으로 되어있다.

```
Project
  |--- data
  |    |--- img_folder
  |         |--- img001.jpg
  |         |--- img002.jpg
  |         |--- ...
  |    |--- data_label_info.csv
  |--- indicator.py
  |--- ...
  
```
- img_folder 안에 있는 이미지들이 Stratified Sampling(층화추출법)을 통해 랜덤으로 선택되어 지표계산에 사용된다.


## Arguments
- img
  - 이미지 데이터 폴더 경로
- meta
  - 이미지명과 label값을 가지고 있는 csv 파일 경로
- dataset
  - 사용하는 데이터 셋 이름, 결과 로그파일 출력을 위해 사용된다.
  - 예를들어 dataset이 caltech256이면 출력파일은 --> **caltech256**\_resize8_ratio0.100000_count300_gvn10_lda_log.txt
- process
  - 지표계산에 사용할 프로세스 수, 많을수록 계산속도가 빨라진다.
- count
  - 각 프로세스 당 랜덤 샘플링할 횟수
  - 예를들어 process는 5 count는 10일 경우 각 프로세스당 10번씩 샘플링하여 총 50번의 샘플링이 이루어진다.
- nworkers
  - 데이터 로딩에 사용할 프로세스 수, 많을수록 데이터 로딩속도가 빨라진다.
- vector
  - 지표계산에 사용할 가우시안 랜덤 벡터 수
- resize
  - 데이터가 리사이즈 되었는지 명시, 결과 로그파일 출력을 위해 사용된다. 
  - 예를들어 resize가 4일경우 출력파일은 --> caltech256_**resize4**_ratio0.100000_count300_gvn10_lda_log.txt
- ratio
  - 샘플링할 비율을 정한다.
  - 예를들어 데이터가 60000개, ratio가 0.1일 경우 6000개의 데이터를 랜덤 샘플링 한다.
- msample
  - 최소 샘플링 수를 정한다.
  - 예를들어 지정된 ratio로 샘플링 할 때 클래스 내 데이터 수가 msample보다 적다면 msample개 만큼 샘플링한다.
  
