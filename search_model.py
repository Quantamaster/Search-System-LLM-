from transformers import pipeline


class SmartSearch:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Load a model optimized for sentence embeddings
        self.model = pipeline("feature-extraction", model=model_name)

    def encode(self, text):
        # Convert text to embeddings
        return self.model(text)

    def find_relevant_courses(self, query, courses):
        # Encode query
        query_embedding = self.encode(query)[0]

        # Calculate similarity with each course
        results = []
        for course in courses:
            course_embedding = self.encode(course['description'])[0]
            similarity = self.cosine_similarity(query_embedding, course_embedding)
            results.append((similarity, course))

        # Sort by similarity score
        results.sort(key=lambda x: x[0], reverse=True)
        return [course for _, course in results[:5]]  # Top 5 results

    @staticmethod
    def cosine_similarity(a, b):
        return sum(x * y for x, y in zip(a, b)) / ((sum(x ** 2 for x in a) ** 0.5) * (sum(y ** 2 for y in b) ** 0.5))
