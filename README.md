# job-recommendation
ML
"""Dependencies"""
1.matplotlib
2.seaborn
3.pandas
4.numpy
5.ast
6.scipy
7.scikit-learn

*OBJECTIVE
Which country, state and city are popular among job creator?
Which country, state and city are popular among job seekers?
Recommend similar jobs based on the jobs title, description
Recommend jobs based on similar user profiles


*Exploratory Data Analysis (EDA)
-Split training and testing data based on column split
-EDA for job openings based on their location information
-EDA for User profiles based on their location information

*pre-processing
-perform following pre-processing steps:
-consider only US region for building this recommendation engine
-removing data records where state is blank or state data attribute is having numerical value.(If needed)


*revised approach
find out Similar jobs


*Best approach
Find out similar users -- Find out for which jobs they have applied -- suggest those job to the other users who shared similar user profile.
We are finding put similar user profile based on their degree type, majors and total years of experience.

We will get to 10 similar users.
We will find our which are the jobs for which these users have applied
We take an union of these jobs and recommend the jobs all these user base
