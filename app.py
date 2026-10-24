#نرم افزار پیش بینی وزن انسان تحت وب با الگوریتم های یادگیری ماشین
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    make_scorer
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from datetime import datetime
weidth_human = {
    "age":[
        7,8,9,10,11,12,13,14,15,16,
        17,18,19,20,21,22,23,24,25,26
    ],
    "gender":[
        "woman","woman","man","man","woman",
        "man","woman","man","woman","man",
        "woman","woman","woman","woman","woman",
        "man","man","man","man","man"
    ],
    "height":[
        145,146,148,150,151,154,156,158,160,162,
        164,166,167,169,170,172,174,173,175,180
    ],
    "weidth":[
        46,47,48,49,50,52,54,56,57,58,
        60,64,65,68,70,72,73,74,75,80
    ]
}
st.set_page_config(
    page_title="پیش بینی وزن",
    page_icon="⚖️",
    layout="centered"
)
st.markdown("""
<style>
html,body,[class*="st-"]{
    direction:rtl;
    text-align:right;
}
</style>
""",unsafe_allow_html=True)

st.title("⚖️نرم افزار پیش بینی وزن")
st.divider()
st.divider()
#توضیح نرم افزار به بازدیدکنندگان
st.markdown("""
             این نرم افزار یک پروژه آموزشی یادگیری ماشین برای پیش بینی وزن
             انسان می باشد.
             در این پروژه از دو الگوریتم KNN و Random Forest استفاده کرده ایم
""")
with st.sidebar:#جهت ایجاد نوار کناری
    st.header("about program")
    st.write("this program is for training machina learnin")
    st.write("selected algoritm:")
    st.write("-KNN")
    st.write("-Random Forest")
    st.caption("version test 1.0")
    st.divider()
    st.subheader("about programmer")
    st.write("create and executer with python and streamlit")
    st.caption("project learning machina learning")
st.divider()#خط جدا کننده ما بین نوار عنوان و برنامه می باشد.
st.write("با وارد کردن سن،قد و وجنسیت وزن تخمینی را مشاهده کنید")
st.caption("این برنامه آموزشی است و نتیجه آن تخمین تقریبی است،نه ارزیابی پزشکی")
st.write("الگوی ماشین وزن را پیش بینی می کند")
st.subheader("اطلاعات فردی")
age = st.number_input("سن",
                      min_value=1,
                      max_value=100,
                      value=25)
height = st.number_input("قد به سانتی متر",
                         min_value=50,
                         max_value=250,
                         value=170)
gender = st.selectbox("جنسیت",
                      ["woman","man"])
st.write("سن وارد شده:",age)
st.write("قد وراد شده:",height)
st.write("جنسیت انتخاب شده",gender)
#انتخاب الگوریتم
st.subheader("select algoritm")
model_name = st.selectbox("select your algoritm:",
                          ["KNN","Random Forest"],
                          key="model_choice")
st.write("selected algoritm is :",model_name)
if st.button("prediction weidth",
             type="primary",
             use_container_width=True):
    df = pd.DataFrame(weidth_human)
    x = df[["age","gender","height"]]
    y = df["weidth"]
    numeric_feature = ["age","height"]
    category_feature = ["gender"]
    preproc = ColumnTransformer(
            transformers=[
                ("num",StandardScaler(),numeric_feature),
                ("cat",OneHotEncoder(handle_unknown="ignore"),
                 category_feature)
            ]
        )
    if age < 1 or height < 50:
        st.error("please enter valid age and height!!!")
    else:
        if model_name == "KNN":
            model = Pipeline([
                ("preprocessor",preproc),
                ("regression",KNeighborsRegressor(
                  n_neighbors=5
                ))
        ])
        else:
            model = Pipeline([
                ("preprocessor",preproc),
                ("regression",RandomForestRegressor(
                  n_estimators=100,
                  random_state=40,
                  max_depth=5
                ))
            ])
        n_split = min(5,len(x))#اعتبار سنجی انجام می دهیم که تعداد بخش ها از تعداد ردیف های داده بیشتر نشود
        kfold = KFold(
                n_splits=n_split,
                shuffle=True,
                random_state=40
            )
        score = cross_val_score(
                model,
                x,
                y,
                cv=kfold,
                scoring="neg_mean_absolute_error"
            )
        mae_each_fold = -score
        avrage_mae = mae_each_fold.mean()
        model.fit(x,y)
        st.write(f"training model with data:{len(x)}")
        new_human = pd.DataFrame([{
            "age":age,
            "gender":gender,
            "height":height
        }])
        prediction = model.predict(new_human)
        st.subheader("prediction result:")
        st.info(f"select algoritm:{model_name}")
        st.success(f"prediction weidth:{prediction[0]:.1f} kg")
        st.metric(
            label="mae for error prediction",
            value=f"{avrage_mae:.2f} kg"
        )
        st.caption(
            "this is value for mean absolute error"
            "number is low for error lowers"
        )
        mae = -score.mean()
        knn_model = Pipeline([
            ("preprocessor",preproc),
            ("regression",KNeighborsRegressor(n_neighbors=5))
        ])
        knn_mae = cross_val_score(
            knn_model,
            x,
            y,
            cv=kfold,
            scoring="neg_mean_absolute_error"
        )
        forest_model = Pipeline([
            ("preprocessor",preproc),
            ("regression",RandomForestRegressor(
                n_estimators=100,
                max_depth=4,
                random_state=40
            ))
        ])
        forest_mae = cross_val_score(
            forest_model,
            x,
            y,
            cv=kfold,
            scoring="neg_mean_absolute_error"
        )
        knn_mae = -knn_mae.mean()
        forest_mae = -forest_mae.mean()
        col1,col2,col3,col4 = st.columns(4)#مشاهده نتایج و مقایسه دو الگوریتم
        with col1:
            st.metric(
                label="prediction weidth",
                value=f"{prediction[0]}kg"
            )
        with col2:
            st.metric(
                label="mae for model",
                value=f"{mae:.2f}kg"
            )
        with col3:
            st.metric(
                label="knn_mae for model",
                value=f"{knn_mae:.2f} kg"
            )
        with col4:
            st.metric(
                label="forest_mae for model",
                value=f"{forest_mae:.2f} kg"
            )
        #مقایسه خطای بین دو الگوریتم که کدام خطای کمتری دارد.
        if knn_mae < forest_mae:
            st.success("knn_model is low in prediction!")
        elif forest_mae < knn_mae:
            st.success("forest_model is low in prediction!")
        else:
            st.info("both of prediction is equal!")
        st.caption(f"usage algoritm:{model_name}")
        #خلاصه ارزیابی مدل ها
        st.subheader("abstract predict models")
        col5,col6,col7 = st.columns(3)
        col5.metric("some samples",len(x))
        col6.metric("knn_mae",f"{knn_mae:.2f} kg")
        col7.metric("forest_mae",f"{forest_mae:.2f} kg")
        #نمودار مقایسه خطای بین دو الگوریتم
        visual = pd.DataFrame({
            "algoritm":["KNN","Random Forest"],
            "MAE":[knn_mae , forest_mae]
        })
        st.subheader("visualation different between two mae")
        st.bar_chart(
            visual,
            x="algoritm",
            y="MAE"
        )
        #مشاهده اطلاعات وارد شده با یک کلیک
        with st.expander("view input information:"):
            st.write(f"age is: {age}")
            st.write(f"height is: {height}")
            st.write(f"gender is: {gender}")
        #منوی تهیه گزارش
        report = f"""
                report prediction weidth
                Data and Hour:{datetime.now().strftime("%Y-%m-%d %H:%M")}
                age:{age}
                height:{height}kg
                gender:{gender}
                algoritm:{model_name}
                prediction_weidth:{prediction[0]:.2f} kg
                MAE:{mae:.2f} kg
                MAE_knn_model:{knn_mae:.2f} kg
                MAE_forest_model:{forest_mae:.2f} kg
        """
        #دانلود تهیه گزارش
        st.download_button(
            label="report download",
            data=report,
            file_name="weidth_prediction.txt",
            mime="text/plain",
            type="secondary",
            use_container_width=True
        )
