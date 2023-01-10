from src.logger import logging
from src.exception import TrainException
import sys
import pandas as pd

class Recommendation:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise TrainException(e,sys)

    def get_recommendations_similar_courses(self, coursesdf, model, course_name: str):
        try:
            cosine_sim = model

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

    def get_recommendations_similar_users_one(self,interactions_csr, model, user_id: int):
        try:
            ids, scores = model.recommend(user_id, interactions_csr[user_id], N=5, filter_already_liked_items=False)
            return ids, scores

        except Exception as e:
            raise TrainException(e,sys)

    def get_recommendations_similar_users_all(self,interactions_csr, model):
        try:
            recs_imp = model.recommend_all(user_items=interactions_csr, N=10, filter_already_liked_items=False)
            return recs_imp

        except Exception as e:
            raise TrainException(e,sys)
