
import streamlit as st 
import streamlit.components.v1 as stc 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Our Dataset
def load_data(data):
	df = pd.read_csv(data)
	return df 

def vectorize_text_to_cosine_mat(data):
	count_vect = CountVectorizer()
	cv_mat = count_vect.fit_transform(data)
	# Get the cosine
	cosine_sim_mat = cosine_similarity(cv_mat)
	return cosine_sim_mat

@st.cache_data
def get_recommendation(title,cosine_sim_mat,df):
	# Convert course titles to lowercase
	title = title.lower()
	df['course_title'] = df['course_title'].str.lower()
  
	course_indices = pd.Series(df.index,index=df['course_title']).drop_duplicates()
	# Index of course
	idx = course_indices[title]

	# Look into the cosine matr for that index
	sim_scores =list(enumerate(cosine_sim_mat[idx]))
	sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
	selected_course_indices = [i[0] for i in sim_scores[1:]]
	selected_course_scores = [i[0] for i in sim_scores[1:]]

	# Get the dataframe & title
	result_df = df.iloc[selected_course_indices]
	result_df['similarity_score'] = selected_course_scores
	final_recommended_courses = result_df[['course_title','similarity_score','url','price','num_subscribers']]
	return final_recommended_courses.head()

# Search For Course 
@st.cache_data
def search_term_if_not_found(term,df):
	result_df = df[df['course_title'].str.contains(term)]
	return result_df


def main():
	st.title("Skillify Course Recommendation")

	menu = ["Home","Recommend","About"]
	choice = st.selectbox("Menu",menu)

	df = load_data("Course-Recommender-System-main/Course-Recommender-System-main/Skillify_courses.csv")

	if choice == "Home":
		st.subheader("Home")
		st.dataframe(df.head(10))


	elif choice == "Recommend":
		st.subheader("Recommend Courses")
		cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
		search_term = st.text_input("Search")
		search_term = search_term.lower()
		if st.button("Recommend"):
			if search_term is not None:
				try:
					results = get_recommendation(search_term,cosine_sim_mat,df)
					with st.beta_expander("Results as JSON"):
						results_json = results.to_dict('index')
						st.write(results_json)

				except:
					st.info("Here are your suggestions")
					result_df = search_term_if_not_found(search_term,df)
					st.dataframe(result_df)

	else:
		st.subheader("About")
		st.text("Course Recommendation system for Skillify")


if __name__ == '__main__':
	main()