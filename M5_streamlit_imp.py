from modules.Model import Knn , Svm , Naive_Bayes
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import streamlit as st 
import seaborn as sns
import pandas as pd 
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://source.unsplash.com/person-with-assorted-color-paint-on-face-ndja2LJ4IcM");
background-size: 180%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-position: center; 
background-repeat: no-repeat;

background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
css_tab = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.3rem;
    }
</style>
'''
css_toast =  """
        <style>
            div[data-testid=stToast] {
                padding:  20px 10px 40px 10px;
                margin: 10px 400px 200px 10px;
                background: #ee0979; 
                background: -webkit-linear-gradient(to right, #ff6a00, #ee0979);  
                background: linear-gradient(to right, #ff6a00, #ee0979); 
                width: 30%;
            }
             
            [data-testid=toastContainer] [data-testid=stMarkdownContainer] > p {
                font-size: 20px; font-style: normal; font-weight: 400;
                
            }
        </style>
        """

st.markdown(css_toast, unsafe_allow_html=True)
st.markdown(css_tab, unsafe_allow_html=True)
st.markdown(page_bg_img, unsafe_allow_html=True)


option=None

tab1, tab2 = st.tabs(["👁️ VISUALIZATION    /", "🔎 MODEL TRAIN / EVALUATE       /"] ,)
def train_test(prep_data):
  X = prep_data.drop("diagnosis", axis=1)
  y = prep_data["diagnosis"]
  try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  except Exception as e :
    st.write("❌ MODEL TRAİN TEST SPLİT BASARISIZ" , e)
  return X_train, X_test, y_train, y_test



def preprocessing(df):

    """
    Buradaki preprocessing fonksiyonu ozellesmemis ve bizden istendigi gibi data yüklendigi 
    an calisan ve cesitli iyilestirmeler yapan preprocess.

    Model.py icindeki model classlarindaki preprocess fonksiyonlari ise o model icin ozellişmiş
    preprocess işlemleri.
    """
    dftemp=df.drop("id" , axis=1)
    dftemp.drop("Unnamed: 32" , inplace=True , axis=1)
    dftemp['diagnosis'] = dftemp['diagnosis'].replace({'M': 1, 'B': 0})
    numeric_columns = dftemp.drop(columns=['diagnosis'])
    # Sifir degerlerini ortalama ile dolduralim
    for column in numeric_columns.columns:
        if (dftemp[column] == 0).any():  
            mean_value = dftemp[column].replace(0, pd.NA).mean()
            dftemp[column] = dftemp[column].replace(0, mean_value)

    dftemp.to_csv("./prepdata.csv" ,index="False")
    if "Unnamed: 0" in pd.read_csv("prepdata.csv"):
        return pd.read_csv("prepdata.csv").drop("Unnamed: 0", axis=1)
    else: 
        return pd.read_csv("prepdata.csv")

def plot_corr(dftemp):
    data = dftemp[['radius_mean', 'texture_mean', 'diagnosis']]
    malignant_data = data[data['diagnosis'] == "M"]
    benign_data = data[data['diagnosis'] == "B"]

    fig = plt.figure(figsize=(10, 6))
    sns.set_style(style="dark")
    sns.scatterplot(data=malignant_data, x='radius_mean', y='texture_mean', color='red', label='KOTU',alpha=0.3 )
    sns.scatterplot(data=benign_data, x='radius_mean', y='texture_mean', color='blue', label='IYI', alpha=0.3 )
    plt.title('Scatter Plot of Radius Mean vs Texture Mean')
    plt.xlabel('Radius Mean')
    plt.ylabel('Texture Mean')
    plt.legend()
    st.pyplot(fig)

#with st.sidebar: 
#    st.link_button("POWERED BY İLKER CODER " , url="https://github.com/ilkerCoder")


with tab1 :
    row_10, corr_mat= st.columns(2)
    prep_col , empty = st.columns([999,0.1])
    with st.sidebar:
        title_container = st.container()
        col1, col2 = st.columns(2)
        with title_container:
            with col1:
                st.markdown('<img  src="https://cdn.cezerirobot.com/media/main/cezeri_logo_2019_06_18-01.png" height="120" width="100">', unsafe_allow_html=True)
            with col2:
                st.image("https://cdn.baykartech.com/media/images/contents/baykar.png" ,width=100)
        st.write(f"<h4 style ='font-style : italic ;text-align:center; margin-bottom :0 ; padding:6px; border:2px solid white;box-shadow: rgba(240, 46, 170, 0.4) -5px 5px, rgba(240, 46, 170, 0.3) -10px 10px, rgba(240, 46, 170, 0.2) -15px 15px, rgba(240, 46, 170, 0.1) -20px 20px, rgba(240, 46, 170, 0.05) -25px 25px;'>⚛ YAPAY ZEKA UZMANLIK PROGRAMI MODUL PROJESİ </h4>", unsafe_allow_html=True)
        st.caption(f"<h4 style ='font-style : italic ;text-align: start; padding-top:10px'>⚛ MUSTAFA ILKER KAMIS</h4>", unsafe_allow_html=True)
      

        upload_status = True 
        uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file:
        
           
           
        df = pd.read_csv(filepath_or_buffer=uploaded_file)
        df.to_csv("uploaded_data.csv", index=False)
        with row_10:
            st.write("<h4 class='btn btn-success' style='text-align: center; color: white; border-bottom: 2px solid white;'>📶 FIRST 10 ROW AND COLUMNS:</h4>" , unsafe_allow_html=True)
            st.dataframe(
            df.head(10).reset_index(drop=True),
            hide_index=False, height=222
            )
        with corr_mat:
            st.write("<h4 class='btn btn-success' style='text-align: center; color: white; border-bottom: 2px solid black;'>📈 CORRELASİON MATRİX: </h4>" , unsafe_allow_html=True)
            plot_corr(df)
        
        with prep_col:
            with st.status("🌀 Preprocessing data..." , expanded=True) as status:
                try :
                    st.write(" ✔️ PREPROCESS İSLEMİ YAPILIYOR...")
                    st.write(" ✔️ GEREKSIZ COLUMNLAR SILINIYOR...")
                    st.write(" ✔️MISSING DATA ISLENIYOR...")
                    st.write(" ✔️TRAIN TEST SPLIT OLARAK BOLUNUYOR...")
                    prep_data = preprocessing(df)
                    st.success("PREPROCESS BASARILI!")
                    status.update(label="PREPROCESSİNG COMPLETE!", state="complete", expanded=True)
                    st.write("📉 LAST 10 ROW IN PREPROCESSED DATASET :")
                    st.dataframe(
                    prep_data.tail(10).reset_index(drop=True),
                    hide_index=False,
                    )   
                    X_test,X_train,y_test,y_train= train_test(prep_data)
                    msg=st.toast('DATA IS BEING SPLIT...')
                    msg.toast('...')
                    msg.toast('TRAIN / TEST SPLIT SUCCESFUL !', icon='🎉')
                    
                    st.write("<h4 class='btn btn-success' style='text-align: center; color: white; border-bottom: 2px solid white;'>✅ VERİ YUKLENDİ . 0.2 ORANINDA TRAİN TEST SPLİT BASARILI</h4>" , unsafe_allow_html=True)
        
                except Exception as e:
                 st.warning("❗ PREPROCESSİNG ERROR" , e)
                 st.toast('❗PREPROCESSING ERROR')

        with st.sidebar: 
            option = st.selectbox(
        "DATA IS READY TO TRAIN. PLEASE CHOOSE THE MODEL",
        ("KNN", "SVM", "NAIVE BAYES"),
        index=None,
        placeholder="Select The Model...",
        )
            st.write('You selected:', option)
    else:
        st.error("❗YOU HAVEN'T UPLOADED DATA YET")

with tab2 :
    col1 , col2 = st.columns(2)
    cnf_matrix , empty = st.columns([999 , 1])
    b1,b2 = st.columns(2)
    with st.container():
        if option == None :
            st.warning("FIRST CHOOSE THE MODEL")
        elif option == "KNN" :
            from sklearn.neighbors import KNeighborsClassifier
            st.success(body="KNN CHOOSEN CORRECTLY")
            msg.toast('KNN CHOOSEN !', icon='🎉')

            with b1:
                op = st.button("🤖 TRAIN AND EVALUATE DATA FOR KNN WITH GRID SEARCH" , type="primary" , disabled=False , key="train")
            with b2:
                op2 = st.button("🤖 TRAIN AND EVALUATE DATA FOR KNN WITH YOUR OWN K VALUE" , type="primary" , disabled=False)
                values = st.slider(
            'Select a range of values',
        1, 50)
                st.write('CHOOSEN VALUE:', values)
            try:
                
                if op:
                  with col1 :  
                    with st.status("🌀 TRAIN AND EVALUATE DATA ..." , expanded=True) as status:
                        try:
                        
                            st.write(" ✔️ KNN SECILIYOR...")
                            st.write(" ✔️ TRAIN , TEST YAPILIYOR...")
                            st.write(" ✔️ PIPELINE OLUSTURULUYOR...")
                            st.write(" ✔️ SCALE YAPILIYOR...")
                            st.write(" ✔️ MODEL DEGERLENDIRILIYOR...")
                            st.write(" ✔️ GRID SEARCH OPTİMAL PARAMETRELER BULUNUYOR...")



                            knn = Knn()
                            msg.toast('KNN TRAIN/EVALUATE SUCCESFUL !', icon='🎉')
                            knn.elbow_method(X_train, X_test, y_train, y_test)
                            knn.plot_elbow_errrate()
                            optimal_k_value = knn.grid_search(X_train , X_test , y_train , y_test)['knn__n_neighbors']
                            st.caption(f"🟢   OPTİMAL DEGERLER BULUNDU K DEGERI ==> {optimal_k_value}")

                            #En optimal k degerine gore modeli egit.
                            knn_model = KNeighborsClassifier(n_neighbors=optimal_k_value)
                            knn.train(X_train, X_test, y_train, knn_model)
                        except Exception as e :
                            print(f"ERROR {e}")
                    with col2:
                            # Model degerlendirmesi
                            y_pred = knn.evaluate(knn_model, X_test, y_test)

                            status.update(label="PREPROCESSİNG COMPLETE!", state="complete", expanded=True)
                    with cnf_matrix:
                            knn.plot_conf_matrix(y_test , y_pred)
                            
                if op2:
                  with col1 :  
                    with st.status("🌀 TRAIN AND EVALUATE DATA ..." , expanded=True) as status:
                        try:
                        
                            st.write(" ✔️ KNN SECILIYOR...")
                            st.write(" ✔️ SECILEN PARAMETRELERE GORE ISLEM YAPILIYOR...")
                            st.write(" ✔️ TRAIN , TEST AYRILIYOR...")
                            st.write(" ✔️ PIPELINE OLUSTURULUYOR...")
                            st.write(" ✔️ SCALE YAPILIYOR...")
                            st.write(" ✔️ MODEL DEGERLENDIRILIYOR...")
                            knn = Knn()
                            msg.toast('KNN TRAIN/EVALUATE SUCCESFUL !', icon='🎉')
                            #Secilen k degerine gore modeli egit.
                            knn_model = KNeighborsClassifier(n_neighbors=values)
                            knn.train(X_train, X_test, y_train, knn_model)
                            
                        except Exception as e :
                            print(f"ERROR {e}")
                    with col2:
                            # Modeli değerlendir
                            y_pred = knn.evaluate(knn_model, X_test, y_test)
                        
                            status.update(label="PREPROCESSİNG COMPLETE!", state="complete", expanded=True)
                    with cnf_matrix:
                            knn.plot_conf_matrix(y_test , y_pred)

            except Exception as e:
                                print("❗ MODEL TRAIN AND EVALUATE ERROR" , e)

            
    
    
        elif option =="SVM":
            try:
             st.success("SVM CHOOSEN CORRECTLY")
             svm = Svm()
             msg.toast('SVM MODEL TRAIN / EVALUATE SUCCESFUL !', icon='🎉')

             scaled_X_train , scaled_X_test = svm.preprocessing(X_train ,X_test)
             svm.grid_serch_predict_plot(scaled_X_train=scaled_X_train,scaled_X_test=scaled_X_test ,y_train=y_train,y_test=y_test) 
            except Exception as e:
             st.warning(f"❗ERROR {e}")
        
        
        
        elif option =="NAIVE BAYES":
            try:
                st.success("NAIVE BAYES CHOOSEN CORRECTLY")
                Nb = Naive_Bayes()
                msg.toast('SVM MODEL TRAIN / EVALUATE SUCCESFUL !', icon='🎉')
                Nb.prep__plot_train_evaluate(X_train , X_test ,y_train , y_test)
            except Exception as e:
                st.warning(f"❗ERROR {e}")
    

# with st.sidebar:
#         st.markdown("""
#      <style>
#      .Special_Thanks {
#        background: #ee0979; 
#        background: -webkit-linear-gradient(to right, #ff6a00, #ee0979);  
#        background: linear-gradient(to right, #ff6a00, #ee0979); 
#        color: white;
#        padding: 5px 10px;
#        padding-bottom : 0;
#        border-radius: 4px;
#        border-color: #46b8da;
#      }
#      #btn {
#        position: fixed;
#        bottom: -4px;
#        right: 10px;
                  
#      }
#    </style>

#    <div id="btn">
#    <button id="specialButton" class="Special_Thanks">
#                     <p style='margin-bottom:0'>🎉 SEVGİLİ HOCALARIMA</p>
#                     <p style ='font-size: 12px; text-align:start; margin-bottom:0 ; margin-top:2px ; font-weight :bold'> Program boyunca bizlere sağladığınız desteğiniz,rehberliğiniz için </p>
#                     <p style ='font-size: 12px; text-align:start; margin-bottom:0; font-weight :bold'>teşekkür ederim.Umarım sizler için her şey yolundadır.Umarım</p>
#                     <p style ='font-size: 12px; text-align:start; margin-bottom:0; font-weight :bold'>birgün yüzyüze tanışma fırsatını da yakalarız. Emeklerinize sağlık</p>
#                     <p style ='font-size: 12px; text-align:start; margin-bottom:0; font-weight :bold ; text-align:center;'>(MESAJI KAPATMAK İCİN SİDEBARI KUCULTEBİLİRSİNİZ)</p>
#                     </p>
#                     </button>
#      </div>



   
#     """, unsafe_allow_html=True)

