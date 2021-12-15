import streamlit as st
import pandas as pd
import datetime as dt
import seaborn as sns
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from plotly import graph_objs as go

import fbprophet
from fbprophet import Prophet
#from prophet import Prophet 
from fbprophet.plot import plot_plotly

import base64

sum=0
sum100=0
sum200=0
sum300=0
sum600=0
sum900=0
i=0
cost=0
cost100=0
cost200=0
cost300=0
cost600=0
cost900=0



image=Image.open('Untitled.png')
st.image(image,width=680,)

#st.title("Electrcity Consumption Prediction")
st.write("""
# Electrcity Usage Prediction App ⚡️

This app predicts the **energy consumption for the next month**!📈  
Download a sample dataset here if you don't have one:""")


backup=pd.read_csv("PJME.csv")

coded_second=base64.b64encode(backup.to_csv(index=False).encode()).decode()
st.markdown(f'<a href="data:file/csv;base64,{coded_second}" download="example.csv">Download Example Dataset</a>',unsafe_allow_html=True)

with st.sidebar:
    
    
    image=Image.open('MicrosoftTeams-image.png')
    st.image(image,use_column_width=True)
    st.header('About Us')
    st.write("""**ESDS** brings intelligence and analytics capablities as well as valuable insights, to where they're needed most.

Connect with us and discover products and solutions to solve your business needs.
""")
    st.subheader('We Unlock these Impossibilities:')
    st.markdown(
"Real-time electrcity monitoring platform  \n"
"Far-distance controlling system  \n"
"On-spot alerting service  \n"
"Electricity forecasting application"
)
    




def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['Datetime'],y=df['Energy_kWh'],name='Energy_kWh'))
    
    fig.layout.update(title_text="Energy Consumption History",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

#new_title = '<p style="font-family:Lobster; color:Green; font-size: 42px;">Step 1: Feed in dataset</p>'
#st.markdown(new_title, unsafe_allow_html=True)

def example(content):
     st.markdown(f'<p style="text-align:center;background-image: linear-gradient(to right,#ee8004, #fbe54b);color:#080807;font-family:Peace Sans; font-size:40px;border-radius:2%;">{content}</p>', unsafe_allow_html=True)
example("Step 1: Tune Parameter")

option=0
Tariff=("Tariff A – Domestic Tariff","Tariff B - Low Voltage Commercial Tariff","Tariff E1 - Medium Voltage General Industrial Tariff")
Selected_period=st.selectbox("Select your tariff:",Tariff)
st.write("Don't know what tariff you're applying? [Refer here](https://www.tnb.com.my/assets/files/Tariff_Rate_Final_01.Jan.2014.pdf)")  
if Selected_period=="Tariff A – Domestic Tariff":
    option=0

if Selected_period=="Tariff B - Low Voltage Commercial Tariff":
    option=1
    
if Selected_period=="Tariff E1 - Medium Voltage General Industrial Tariff":
    option=2

period_forecast=31
Pt=("1 week","2 weeks","1 month")
Selected_period=st.selectbox("Determine the period to forecast:",Pt)
if Selected_period=="1 week":
    period_forecast=7*24+2
    type="H"
if Selected_period=="2 weeks":
    period_forecast=14*24+2
    type="H"

if Selected_period=="1 month":
    period_forecast=31
    type="d"
    
    

    



example("Step 2: Feed in Dataset")    
st.subheader('**Input Dataset**')
data_file=st.file_uploader("Upload at least past 3 months Energy Usage ",type=["csv"])



if data_file is not None:
    #st.write(type(data_file))
    file_details={"filename":data_file.name,"filetype":data_file.type,"filesize":data_file.size}
    #st.write(file_details)
    df=pd.read_csv(data_file)
    with st.spinner('Visualising data.....'):
        my_expander = st.expander("	𝗥𝗮𝘄 𝗗𝗮𝘁𝗮 𝗣𝗿𝗲𝘃𝗶𝗲𝘄", expanded=True)
        with my_expander:
            st.subheader('Raw Data Preview')
            st.write('This dataset contains energy and power consumed per hour for **7** months from **Jan 2021** to **July 2021**')
            st.dataframe(df)
            plot_raw_data()
   
    
    

    with st.spinner('Training model....'):
        #df.columns=['ds','y']
        df=df.reset_index().rename(columns={'Datetime':'ds','Energy_kWh':'y'})
        model = Prophet()
        model.fit(df)
        future_dates = model.make_future_dataframe(periods=period_forecast,freq=type)
        prediction = model.predict(future_dates)
        fig1=plot_plotly(model,prediction, xlabel='Datetime', ylabel='Electricity Consumption')
        fig3=model.plot(prediction, xlabel='Date', ylabel='Electricity Consumption')
        fig2=model.plot_components(prediction)
        future=prediction[['ds','yhat']]
        periodd=future.tail(period_forecast)
        #fig0=plot_plotly(periodd,xlabel='Datetime', ylabel='Electricity Consumption')
        fig0=periodd.plot(x='ds',xlabel='Datetime', ylabel='Electricity Consumption')
        
        m,n=periodd.shape
        values = periodd.values  
        matrix = np.concatenate([values])

        import operator 
        Output = max(matrix, key = operator.itemgetter(1))
        highesttime= Output[0]
        highestdemand= Output[1]

        sum=0
        tsum=0
        sum100=0
        sum200=0
        sum300=0
        sum600=0
        sum900=0
        i=0
        cost=0
        tcost=0
        total=0
        bringtonext1=0
        bringtonext2=0
        bringtonext3=0
        bringtonext4=0
        minimumc=0
        if option==0:
    
            while sum<=200 and i<=(m-1):
                sum+=matrix[i][1]
                sum100+=matrix[i][1]
                i+=1
            if sum>200:
                bringtonext1=sum-200
                sum=200
                

            sum100=sum
            cost100=sum*0.2180
            cost+=cost100


            while sum>200 and sum<=300 and i<=(m-1):
                sum+=matrix[i][1]
                sum200+=matrix[i][1]
                i+=1
            total=sum200+bringtonext1
            if total>100:
                bringtonext2=total-100
                sum200=100
                total=100
                sum=300
            cost200=total*0.3340
            cost+=cost200
            #sum200+=bringtonext1
            sum+=bringtonext1
            

            while sum>300 and sum<=600 and i<=(m-1):
                sum+=matrix[i][1]
                sum300+=matrix[i][1]
                i+=1
            total=sum300+bringtonext2
            if total>300:
                bringtonext3=total-300
                sum300=300
                total=300
            cost300=total*0.5160
            cost+=cost300
            #sum300+=bringtonext2



            while sum>600 and sum<=900 and i<=(m-1):
                sum+=matrix[i][1]
                sum600+=matrix[i][1]
                i+=1
            total=sum600+bringtonext3
            if total>300:
                bringtonext4=total-300
                sum600=300
                total=300
            cost600=total*0.5460
            cost+=cost600
            #sum600+=bringtonext3



            while sum>900 and i<=(m-1):
                sum+=matrix[i][1]
                sum900+=matrix[i][1]
                i+=1
            cost900=sum900*0.5710+bringtonext4*0.5710
            cost+=cost900
            sum900+=bringtonext4
            sum+=bringtonext4
            if cost<3:
                cost100=0
                cost200=0
                cost300=0
                cost600=0
                cost900=0
                minimumc=3
                cost=minimumc


        if option==1:
            sum=0
            sum200=0
            i=0
            cost=0
            while sum<=200 and i<=(m-1):
                sum+=matrix[i][1]
                i+=1
            
            if sum>200:
                bringtonext1=sum-200
                sum=200
            sum100=sum
            cost100=sum*0.4350
            cost+=cost100

            while sum>200 and i<=(m-1):
                sum+=matrix[i][1]
                sum200+=matrix[i][1]
                i+=1

            cost200=round(sum200)*0.5090+round(bringtonext1)*0.5090
            sum200+=bringtonext1
            cost+=cost200
            sum+=sum200
            if cost<7.2:
                cost100=0
                cost200=0
                minimumc=7.2
                cost=minimumc

        
        if option==2:
            sum=0
            sum200=0
   
            i=0
            cost=0
            while i<=(m-1):
                sum+=matrix[i][1]
                i+=1
            
            cost+=sum*0.3360
            
            
            tcost+=sum*0.3360
            tcost+=round(highestdemand)*23.70
            tsum+=round(highestdemand)
            tsum+=sum
            if cost<600:
                cost=0
                minimumc=600
                tcost=minimumc
                


        
         
    st.success('Done Training!')
    with st.spinner('Plotting data.....'):
        example("Step 3: See the Result")
        st.subheader('Forecast Data Overview')
        my_expander = st.expander("Future Electricity Bill", expanded=True)
        with my_expander:
            if option==0 or option==1:
                st.write("The electricity cost for next",Selected_period," is RM",format(cost,".2f"))
            if option==2:
                st.write("The electricity cost for next",Selected_period," is RM",format(tcost,".2f"))
            st.write("**Electrcity Bill Table**")
            if option==0:
            
                table=[['Tariff Blocks(kWh)','Consumption(kWh)','Price(RM)','Amount(RM)'],['200',str(format(round(sum100))),'0.218',str(format(cost100,".2f"))],['100',str(round(sum200)),'0.334',str(format(cost200,".2f"))],['300',str(round(sum300)),'0.516',str(format(cost300,".2f"))],['300',str(round(sum600)),'0.546',str(format(cost600,".2f"))],['Over 900',str(round(sum900)),'0.571',str(format(cost900,".2f"))],['Minimum Monthly Charge','','7.20',str(format(round(minimumc)))],['','','',''],['Total',str(round(sum)),'',str(format(cost,".2f"))]]
                st.table(table)

            if option==1:
            
                table=[['Tariff Blocks(kWh)','Consumption(kWh)','Price(RM)','Amount(RM)'],['200',str(format(round(sum100))),'0.435',str(format(cost100,".2f"))],['Over 200',str(round(sum200)),'0.509',str(format(cost200,".2f"))],['Minimum Monthly Charge','','7.20',str(format(round(minimumc)))],['','','',''],['Total',str(round(sum)),'',str(format(cost,".2f"))]]
                st.table(table)
        
            if option==2:
                table=[['Tariff Blocks(kWh)','Consumption(kWh)','Price(RM)','Amount(RM)'],['For all kWh',str(format(round(sum))),'0.336',str(format(cost,".2f"))],['Maximum Demand Charge',str(format(round(highestdemand))),'23.70',str(round(highestdemand)*23.70)],['Minimum Monthly Charge','','600',str(format(round(minimumc)))],['','','',''],['Total',str(round(tsum)),'',str(format(tcost,".2f"))]]
                st.table(table)
                
        

        my_expander = st.expander("Electricity Consumption", expanded=True)
        with my_expander:
            st.subheader("Overview of predicted eletrcity consumption")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            periodd.plot(x='ds',xlabel='Datetime', ylabel='Electricity Consumption (kWh)')
            st.pyplot()
            st.subheader("Overview of overall eletrcity consumption")
            st.plotly_chart(fig1,use_container_width=True,xlabel='Datetime',ylabel='Electrcity Consumption')
            st.write('⚫Black dots indicate the **actual consumption**.')
            st.write('🔵Blue line indicates the **predicted consumption**.')
            st.write('Note: Black dots only appear from **1 Jan 2021** to **31 July 2021** because the actual consumptions only available in this period.')
            st.subheader('Prediction table')
            st.write('Note: ds represents datatime, yhat represents predicted electricity consumption')
            st.write(prediction[['ds','yhat','yhat_lower','yhat_upper']])
        
            #st.write(prediction)
        my_expander = st.expander("Analysis Report", expanded=True)
        with my_expander:
            st.write('The following graphs show the pattern of the electrcity consumption throughout the period.')
            st.write(fig2) 
            if option==2:
                st.subheader("Analysis Report")
                st.write("We detected that you will be charged for a maximum demand of RM ",format(highestdemand*23.70,".2f"), "[ ", format(highestdemand,".2f")," kW x RM 23.70/kW]")
                st.write("This is due to your highest electrcity consumption, ",format(highestdemand,".2f")," kW on ",highesttime)
                st.write("[Learn more about Maximum Demand charges](https://www.tnb.com.my/commercial-industrial/maximum-demand)")  
                st.write("To avoid being charged for a high amount of payment, please do follow the following practise: ") 
                st.write("1. Opting for any promotional scheme offered by TNB relating to MD such as Sunday Tariff Rider Scheme (STR) so that no maximum charges will be applied on Sundays.") 
                st.write("2. Starts your motor/equipment in stages or during off-peak period.") 
                st.write("3. Practicing demand side management such as peak shift i.e. shifting their peak operation/consumption to off peak period as MD charges is not applicable during off-peak period for customer with peak/off-peak tariff.") 
                image=Image.open('time.png')
                st.image(image,width=680,)
                st.write("You are eligible to enjoy **20% discount** if you consumed electricity between 10pm to 8am [Off Peak Hour]")
                st.write("[Learn more about Off Peak Tariff](https://www.mytnb.com.my/business/special-schemes/off-peak-tariff-rider)") 
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Original Electrcity Bill (RM)", value=format(tcost,".2f") )
                col2.metric(label="Deducted Electrcity Bill (RM)", value=format(tcost*80/100,".2f") , delta="-20%")

                

            if option==1:
                st.write("It has been detected that you will consume electrcity on weekdays and peak hour")
                st.write("Therefore, there's chance for you to reduce the electrcity bills")
                image=Image.open('time.png')
                st.image(image,width=680,)
                st.write("You are eligible to enjoy **20% discount** if you consumed electricity between 10pm to 8am [Off Peak Hour]")
                st.write("[Learn more about Off Peak Tariff](https://www.mytnb.com.my/business/special-schemes/off-peak-tariff-rider)") 
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Original Electrcity Bill (RM)", value=format(cost,".2f") )
                col2.metric(label="Deducted Electrcity Bill (RM)", value=format(cost*80/100,".2f") , delta="-20%")

        data_prediction=pd.DataFrame(periodd)
        coded_data=base64.b64encode(data_prediction.to_csv(index=False).encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{coded_data}" download="prediction_data.csv">Download Predicted Data</a>',unsafe_allow_html=True)
    st.success('Finished!')

    
    
  
    
    
    
#st.write('This line chart predicts the energy consumption in next month(**August 2021**)')
    
##st.write(data.tail())
##st.line_chart(data['AEP_MW'])



