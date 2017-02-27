# query_adaptive_fusion_inPython
## About 
Implements query adaptive fusion with python about 'Query-Adaptive Late Fusion for Image Search and Person Re-identification'
## Run environment
Use python2.7 and anaconda
## How to use
### File
In fusion.py file: it's use author's data and you need download the reference data from author's homepage.

In dist.py file: implement eucdliean distance

In normalize.py: implement normalize function

In load.py: load mat and txt file in python
### Data
score_caffe.txt: query score in paper

score_gist.txt: query score in paper
### Use
Run python fusion.py a demo for fusion.

You can load more score data in different feautres, and modifiy fusion.py as other feaure score load in this file.
# Reference
1. [Query-Adaptive Late Fusion for Image Search and Person Re-identification](http://www.liangzheng.com.cn/Project/project_fusion.html)

Paper's intorduction,and you can download reference data in this page.

2.[Read notes about query adaptive fusion](http://www.noneface.com/2017/02/24/Query_Adaptive_Fusion_read_note.html)
