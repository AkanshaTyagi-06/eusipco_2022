# Location-invariant-representations-for-acoustic-scene-classification
This repository consists of the python implementation for our work accepted in EUSIPCO 2022 i.e Location-invariant representations for acoustic scene classification.


# Introduction
This work is based on publicly available codes of OpenL3, Soundnet, MCCA, KMCCA and dMCCA. Their respective links are provided as a part of this repository.

### Overview
  DCASE datasets consists of data for different acoustic scenes collected across different cities. For instance DCASE 2019 contains data for 10 cities namely :Barcelona (b), Helsinki (h), London (lo), Paris (pa), Stockholm (s), Vienna (v), Lisbon (li), Lyon (ly), Milan (m) and Prague (pr). DCASE 2018 data is a subset of that of DCASE 2019 and contains the data from the first 6 mentioned cities.
  
  The motivation of this work is to find out the recording location of an acoustic scene as a plausible intra-scene variation causing source and to provide a method to handle the same i.e reduce this variation. In this work, we have used varoius multi-view learning methods to reduce the effect of this variation by considering the multiple recording locations i.e recording cities of an acoustic scene as its multiple views.
  

### Feature Extraction

We use two different pre-trained networks for the purpose of feature extraction namely L3-net and Soundnet. The details about them mentioned as follows:

1. OpenL3 (L3-net)

      Code link : (https://github.com/marl/openl3)  
      
      Manuscript :
          Look, Listen and Learn More: Design Choices for Deep Audio Embeddings
          Jason Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello.
          IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, Brighton, UK, May 2019.
  
2. Soundnet

      Code link : (https://github.com/eborboihuc/SoundNet-tensorflow)
      
      Manuscript :
          SoundNet: Learning Sound Representations from Unlabeled Video 
          Yusuf Aytar, Carl Vondrick, Antonio Torralba
          In Proceedings of the 30th International Conference on Neural Information Processing Systems (NIPS'16)

### View generation

As mentioned earlier, we assume that the location of an acoustic scene is a source causing intra-class variation and to handle the same we use multi-view learning framework. However, multi-view learning frameworks have constraint of equal number of examples across different views but the DCASE data doesn't implicitly satisfy this constraint.

For instance, the total number of examples as a part of training datset across various cities for DCASE 2019 dataset is
1.  Barcelona  - 1051
2.  Helsinki   - 1015
3.  London     - 964
4.  Paris      - 1014
5.  Stockholm  - 1013
6.  Vienna     - 1065
7.  Lisbon     - 1061
8.  Lyon       - 976
9.  Milan      - 1030
10. Prague     - 1026

To satisy, the multi-view constraint we use class-wise mixup per city to make the number of examples same across all the views i.e cities. All the scripts related to the same can be found in **View-generation** folder and the final view wise generated data can be found in **View-data** folder for both DCASE 2018 and DCASE 2019 datasets.

After performing class-wise mixup per city the number of examples all cities become equal i.e 1320 (132 examples per class; total 10 classes). 

Following steps should be followed for generating multi-view learning compatible views

1. Create city-wise text files containing the names of the .wav files for a particular city {citywise_train_txt.py, citywise_test_txt.py}
2. For each such text file, generate class-wise text file (for performing class-wise mixup) for each city {classwise-txt_percity.py}
3. Perform class-wise mixup {class-wise_mixup.py}
4. Finally, create a view matrix for each city consisting of train and test data corresponding to each city {create-view.py}


### Multiview framework

1. MCCA (Multiset CCA)

      Code link : (https://github.com/mvlearn/mvlearn/blob/main/mvlearn/embed/mcca.py)
      
      Manuscript :
            Multi-view canonical correlation analysis
            Rupnik, Jan and Shawe-Taylor, John
            SiKDD 2010
  
2. KMCCA (Kernel MCCA)

      Code link : (https://github.com/mvlearn/mvlearn/blob/main/mvlearn/embed/kmcca.py)
      
      Manuscript :
            Multi-view canonical correlation analysis
            Rupnik, Jan and Shawe-Taylor, John
            SiKDD 2010
  
3. dMCCA (Deep MCCA)

      Code link : (https://github.com/usc-sail/mica-deep-mcca?utm_source=catalyzex.com)
      
      Manuscript :
            Multimodal Representation Learning using Deep Multiset Canonical Correlation
            Krishna Somandepalli and Naveen Kumar and Ruchir Travadi and Shrikanth Narayanan
            arXiv preprint arXiv:1904.01775, 2019
  
  


