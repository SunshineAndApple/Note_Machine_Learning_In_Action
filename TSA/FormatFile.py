import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import OneHotEncoder,LabelEncoder
# from scipy import sparse
# import os

def dealUserFeatureData():
    userFeature_data = []
    count = 0
    with open('userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')


            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            # if i % 1000000 == 0:
            #     print(i)
                # break
            if i == 50:
                break

        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv('userFeature.csv', index=False)
        del userFeature_data

if __name__ == '__main__':
    dealUserFeatureData()
    print('Finish!')
