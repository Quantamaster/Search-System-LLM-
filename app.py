import pandas as pd
from models.search_model import SmartSearch

# Load courses
courses = pd.read_csv("data/courses.csv").to_dict(orient="records")
search_model = SmartSearch()


def search_courses(query):
    results = search_model.find_relevant_courses(query, courses)
    for course in results:
        print(f"Title: {course['title']}\nDescription: {course['description']}\n")


if __name__ == "__main__":
    user_query = input("Enter your course search query: ")
    search_courses(user_query)
