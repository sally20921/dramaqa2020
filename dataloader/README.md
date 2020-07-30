# A.1 Dataset Analysis
## Hierarchical QAs
- DramaQA has four hierarchical difficulty levels for each questions.
- Because a single scene of a video has multiple shots, the number of questions for 
- Difficulty 1 and 2 is naturally larger than that for Difficulty 3 and 4. 
- In Difficulty 1 and 2, Who and What questions are the majority.
- In case of Difficulty 3, How and What types are the top-2 questions.
- In Difficulty 4, most of questions start with Why.

- 20 main characters 
- for visual metadata annotation, the visual bounding boxes were created using an automated 
- tagging tool, and workers manually annotated the main characters' names, behaviors, and emotions.

- a video clip consists of a sequence of images with visual annotations centering the main characters. 

# 3.3 Character-Centered Video Annotations 
- As visual metadata, all image frames in the video clips are annotated with main characters' information.
- all coreference of the main characters is resolved in the scripts of the video clips

## Visual Metadata
- Bounding Box : In each image frame, bounding boxes of both a face rectangle and a full-body 
