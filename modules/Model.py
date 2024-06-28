from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score , confusion_matrix , ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns


class Knn:
    test_error_rates = []

    def preprocessing(self, X_train, X_test):
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)
        return scaled_X_train, scaled_X_test

    def train(self, X_train, X_test, y_train,classifier):
        scaled_X_train, scaled_X_test = self.preprocessing(X_train, X_test)
        model = classifier
        model.fit(scaled_X_train, y_train)

        return model

    def evaluate(self, model, X_test, y_test):
        s1 , s2 = st.columns(2)
        scaled_X_test = StandardScaler().fit_transform(X_test)
        y_pred = model.predict(scaled_X_test)
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred , output_dict=True) ).transpose())
        with s1:
            st.write(f"<h4 style ='font-style : italic ;'>Accuracy Score:</h4>", unsafe_allow_html=True)
        with s2:            
            st.write(f"<h4 style ='font-style : italic ; color : red ; '> {accuracy_score(y_test, y_pred ):.3f}</h4>", unsafe_allow_html=True)
        return y_pred
    def plot_conf_matrix(self , y_test , y_pred):
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
 

    def elbow_method(self, X_train, X_test, y_train, y_test):
        """
        OPTİMAL K DEGERİ BULMAK İCİN 2 YONTEM . ELBOW VE GRİD. BİZ GRİDİ KULLANACAGIZ.
        BELKİ İKİSİNİN EN İYİ OLDUGU DEGERİ BULUP ORTALAMASININ ALINDIGI TEKNİKLER VARDIR
        VAKTİMİZ YETERSE BAKACAGIZ.
        """   
        for k in range(1, 30):
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(X_train, y_train)
            y_pred_test = knn_model.predict(X_test)
            test_error = 1 - accuracy_score(y_test, y_pred_test)
            self.test_error_rates.append(test_error)

    
    def plot_elbow_errrate(self):
        plt.plot(range(1, 30), self.test_error_rates)
        plt.ylabel("ERROR RATE")
        plt.xlabel("K VALUE")
        #plt.show()
    
    def grid_search(self , X_train , X_test , y_train , y_test):
        scaler = StandardScaler()
        knn = KNeighborsClassifier()
        operations = [("scaler" , scaler) , ("knn" , knn)]
        pipe=Pipeline(operations)
        k_values = list(range(1,20))
        param_grid = {"knn__n_neighbors" : k_values}
        full_cv_classifier = GridSearchCV(pipe , param_grid=param_grid ,
                                          cv= 5 , scoring="accuracy",)
        #ARTIK SCALE YAPMAMA GEREK YOK CUNKU PİPE 'DA SCALE İŞLEMİ DE YAPILACAK                                          
        full_cv_classifier.fit(X_train ,y_train)
        return full_cv_classifier.best_estimator_.get_params()

class Svm :
    def preprocessing(self ,X_train , X_test):
        scaler =StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)
        return scaled_X_train , scaled_X_test
   
    def grid_serch_predict_plot(self ,scaled_X_train , scaled_X_test ,y_train , y_test):
        col1 , col2 = st.columns(2)
        svc = SVC(class_weight="balanced")
        param_grid={"C" :[0.001 , 0.01 ,0.1 ,0.5, 1], 'kernel': ['rbf', 'poly', 'sigmoid'] ,'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
        grid =GridSearchCV(svc , param_grid)
        grid.fit(scaled_X_train ,y_train)
        grid_preds =grid.predict(scaled_X_test)
        disp =ConfusionMatrixDisplay(confusion_matrix(y_test  ,grid_preds) , display_labels=["Class 0", "Class 1"])
        disp.plot()
        #PLOT WARNİNGİ İPTAL ETME
        st.set_option('deprecation.showPyplotGlobalUse', False)
        with col1:
            st.write(f"<h4 style ='font-style : italic ; border-bottom: 2px solid orange'>CONFUSİON MATRİX::</h4>", unsafe_allow_html=True)
            st.pyplot()
        with col2:
            st.write(f"<h4 style ='font-style : italic ; border-bottom: 2px solid white'>EVALUATİON:</h4>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(classification_report(y_test, grid_preds , output_dict=True) ).transpose())
            st.write("Accuracy Score:", accuracy_score(y_test , grid_preds))
        st.write(f"<h4 style ='font-style : italic ;text-align:center '>OPTİMAL PARAMETERS AFTER GRIDSEARCH:</h4>", unsafe_allow_html=True)
        st.write(f"<h4 style ='font-style : italic ; border-bottom: 2px solid white ; text-align:center ; color:turquoise'>{grid.best_params_}</h4>", unsafe_allow_html=True)

class Naive_Bayes:
    def prep__plot_train_evaluate(self ,X_train , X_test , y_train , y_test):
        c1 , c2 = st.columns(2)
        scale=StandardScaler()
        X_train=scale.fit_transform(X_train)
        X_test = scale.transform(X_test)
        model = GaussianNB()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        with c1:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("y_pred")
            plt.ylabel("y_true")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            
        with c2 :
            s1 , s2 = st.columns(2)
            st.write("<h5 style ='font-style : italic ;margin-bottom:5;padding-bottom:5;'>Classification Report:</h5>" , unsafe_allow_html=True)
            with s1:
                st.write(f"<h4 style ='font-style : italic ;padding-top:0;padding-bottom:5;'>Accuracy Score:</h4>", unsafe_allow_html=True)
            with s2:            
                st.write(f"<h4 style ='font-style : italic ; color : red; padding-top:0; '> {accuracy_score(y_test, y_pred ):.3f}</h4>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(classification_report(y_test, y_pred , output_dict=True)).transpose() , height=225)
           