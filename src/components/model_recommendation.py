from src.logger import logging
from src.exception import TrainException
import os,sys
import pandas as pd

'''
class Recommendation:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise TrainException(e,sys)

    def get_recommendations_similar_courses(self, course_name: str):
        try:
            coursesdf = self.load_courses_ingested_features()
            cosine_sim = self.model_training_similar_courses()

            #Construct a reverse mapping of indices and course names, and drop duplicate titles, if any
            indices = pd.Series(coursesdf.index,index=coursesdf['course_name']).drop_duplicates()

            idx = indices[course_name]
            found = False
            # Get the pairwsie similarity scores of all courses with that course
            # And convert it into a list of tuples as described above
            sim_scores = list(enumerate(cosine_sim[idx]))
            # Sort the courses based on the cosine similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Get the scores of the 5-6 most similar courses. Ignore the first courses.
            sim_scores = sim_scores[1:6]
            # Get the courses indices
            course_indices = [i[0] for i in sim_scores]
            for cidx in course_indices:
                if indices[course_name] == cidx:
                    found == True
                    course_indices.remove(cidx)
                    break

            if found == False:
                course_indices.pop()

            # Return the top 5-6 most similar courses
            recommended_courses =  coursesdf['course_name'].iloc[course_indices].tolist()
            return recommended_courses

        except Exception as e:
            raise TrainException(e,sys)

    def get_recommendations_similar_users(self, course_name: str):
        try:
            pass

        except Exception as e:
            raise TrainException(e,sys)
'''