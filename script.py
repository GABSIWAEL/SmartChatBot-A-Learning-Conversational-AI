import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class LearningChatbot:
    def __init__(self, responses_file="responses.txt"):
        self.vectorizer = TfidfVectorizer()
        self.responses_file = responses_file
        self.responses = self.load_responses()

    def load_responses(self):
        try:
            with open(self.responses_file, "r", encoding="utf-8") as file:
                return [line.strip() for line in file.readlines()]
        except FileNotFoundError:
            return []

    def save_responses(self):
        with open(self.responses_file, "w", encoding="utf-8") as file:
            file.write("\n".join(self.responses))

    def learn_from_user_input(self, user_input):
        self.vectorizer.fit_transform(self.responses)
        X = self.vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(
            X, self.vectorizer.transform(self.responses))

        if np.max(similarity_scores) > 0.3:  # Adjust the similarity threshold as needed
            best_response_index = np.argmax(similarity_scores)
            # Display the corresponding response
            print(self.responses[best_response_index + 1])
        else:
            print(
                f"Please provide a better response for this situation: {user_input}")
            try:
                better_response = input()
            except KeyboardInterrupt:
                print("Goodbye!")
                self.save_responses()  # Save responses before exiting
                exit()
            # Save user input instead of better response
            self.responses.append(user_input)
            self.responses.append(better_response)
            self.save_responses()
            print("Response saved. Let's continue the conversation.")

    def chat(self):
        user_input = ""
        while user_input.lower() != "exit":
            try:
                user_input = input("You: ")
                self.learn_from_user_input(user_input)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                self.save_responses()  # Save responses before exiting
                exit()


if __name__ == "__main__":
    chatbot = LearningChatbot()
    print("Learning Chatbot: Type 'exit' to end the conversation.")
    chatbot.chat()
