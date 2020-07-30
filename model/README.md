- limit our script input length to 300 words 
- limit our visual metadata samples to 100 samples 
- for speed and memory 
- which covers over 97% of textual data
- 82% of visual samples
- the batch size 16
- cross-entropy loss 
- Adam, learning rate 10**-4, weight decay 10**-5

- difficulty 1 is based on context matching module without script S,M,B
- difficulty 2 is based on context matching module without metadata S,M,B
- difficulty 3 is based on context matching module without bounding box => use only script and metadata
- difficulty 4 is based on context matching module and S,M,B

- conclusion : note that context matching module more dominant than character matching module to get correct answer 


- most QA in shot level inquire visual information 
- most QA in scene level need joint understanding of visual and textual information

- if there is more 1 than use bounding box more 
- if there is more 3 use script and metadata more 


